# Architecture

## Frame overview

```
CPU                                   GPU (one command buffer per frame, 2 frames in flight)
---                                   --------------------------------------------------------
poll finished batches  ─────────────► copies scratch -> quad pool, chunk record updates
pick dirty chunks (budgeted)          worldgen batch:  terrain -> props -> edits -> light fill
                                                       -> mesh count -> prefix -> mesh write
                                                       -> readback
light volume follows the camera       light: sky columns -> 4 propagation steps (while changing)
probes (camera biome, minimap)        probe dispatch
                                      cull (frustum + per-direction) -> draw groups
                                      shadow cascades x4 (cull + task/mesh depth)
                                      main pass: terrain (task/mesh) + models + sky
                                      translucent pass: water + volumetric clouds
                                      TAA resolve (compute) -> history
                                      tonemap (+ TAA sharpen) + ImGui -> swapchain
```

All buffers are reached through buffer device addresses in push constants. There is one
pipeline layout: a bindless image set plus 256 bytes of push constants.

## Chunks and LOD

- A chunk is 32³ cells. At LOD L, each cell is 2^L blocks. Generated voxels include a one-cell
  apron, so meshing never needs neighbouring chunks.
- Level L covers a (2R × 2Rv × 2R) chunk box around the camera, minus the box covered by level
  L−1. Defaults: R=8, Rv=4, 5 levels, which gives about a 4 km horizon.
- Retired chunks stay visible until the chunks that replace them are ready, so LOD switches
  don't open holes.
- Voxels are transient. They only exist between generation and meshing inside a batch, so
  only meshes are resident: 8-byte quads in a 512 MiB pool with a CPU range allocator.

## GPU meshing

1. The count pass runs one invocation per (chunk, face direction, 32×32 slice). It builds a
   visibility bitmask per row, then does a greedy merge on (block, AO pattern, biome).
2. The prefix pass computes per-job exclusive offsets and compacts the batch into scratch.
3. The write pass repeats the same greedy walk and writes the quads.

Quads with an identical non-uniform AO pattern merge too. The fragment shader evaluates AO
per cell using `fract()`, so a merged quad looks the same as the unmerged quads would.

Streams per chunk: 6 opaque directions, 6 transparent directions, and model instances.

## Rendering

- **Cull pass**: frustum culling, then back-face culling per face direction (a chunk's +X
  faces are only drawn if the camera is on that side). It emits task-shader groups of at most
  2048 quads.
- **Task and mesh shaders**: 32 threads × 2 quads gives 64 quads per meshlet (256 vertices,
  128 triangles).
- **Models** (plants, slabs, stairs, ...): 6 instances × 10 quad slots per meshlet. Boundary
  faces are culled against opaque neighbours.
- **Shadows**: 4 cascades, fitted to the view frustum slices and texel-snapped in world space
  (f64). Only faces turned away from the sun are drawn, plus a slope bias.
- **Precision**: rendering is camera-relative, with an integer camera block plus a fractional
  offset, so it stays precise anywhere in the i32 world.

## Temporal anti-aliasing

The projection is jittered by a sub-pixel Halton (2,3) offset every frame (8-frame cycle). Culling,
shadows and the editing overlay stay unjittered. `shaders/render/taa.comp` resolves each frame into
a ping-ponged RGBA16F history:

- **Reprojection**: the closest depth in the 3×3 neighbourhood is unprojected with this frame's
  jittered inverse view-projection, then projected with the previous frame's unjittered one. That
  one is re-based on the current camera position (rendering is camera-relative). The current jitter
  is added back, so a static scene samples the history in place. Sky pixels reproject as
  directions.
- **History**: a 5-tap Catmull-Rom sample, clipped against the current neighbourhood's mean ±1.25σ
  in YCoCg. Blending runs in a luminance-compressed space, with 8–20% new frame (more under fast
  motion).
- The tonemap pass reads the resolved image and applies a light 4-neighbour sharpen, clamped to
  the neighbourhood so it doesn't ring.

## Block light and sky light

Light lives in a toroidal 128³ volume at block resolution around the camera, wrapped with
`world & 127` (`crates/engine/src/light.rs`, `world/light.rs`, `shaders/light/`). Voxels are not
resident, so the volume keeps its own copy of the blocks:

1. **Fill**: LOD0 generation jobs inside the volume copy their final blocks (after props and
   edits) into an R16 block image. Chunks that scroll into the volume while already meshed get a
   light-only job, which generates them but skips meshing.
2. **Sky sources**: one thread per column walks down from the top of the volume. Cells keep full
   sky light until the first light-blocking block. Water, ice and leaves on the way dim it.
3. **Propagation**: Jacobi steps between two RGBA8 images (RGB block light, A sky light). Each
   cell takes the brightest of its own emission or sky source and its six neighbours minus 1/15
   per block, plus the block's attenuation. Light moves one block per step, so 15 steps settle
   any change, and light from a removed source fades just as fast. Four steps run per frame,
   only for a few frames after a fill or after the volume moves.
4. **Shading**: faces sample the light image trilinearly half a block in front of the surface,
   which gives smooth lighting. Sky light scales sun and ambient (caves go dark), block light
   adds its colour. The outer 12 blocks of the volume fade to "open sky, no block light".

Each block has an emission colour, an attenuation and a light-blocking bit in the GPU block
table (`BlockInfo.light`).

## World generator (crates/worldgen)

Graph kinds:

- **World**: a Zone Voronoi node feeds the World Output, which also holds the sea level.
- **Zone**: biome layout (Biome Voronoi or a single Biome) plus the cave carver density.
- **Biome**: Terrain (density > 0 means solid), Material, Ground Cover, up to 6 prop layers,
  and environment colours.
- **Function**: a reusable density subgraph with 4 arguments.

The compiler emits GLSL:

- **Live parameters**: every Float/Int parameter becomes `P(i)`, a slot in a GPU buffer.
  Dragging a slider re-uploads parameters and regenerates chunks without recompiling.
- **2D hoisting**: any value that depends only on X/Z is computed once per column in
  `bN_col` and cached in `WgColumn.h[k][...]`. The per-voxel `bN_den` reads the cache.
- **Transform contexts**: domain warp, stretch and offset create evaluation contexts.
  Upstream nodes are re-emitted per context with memoisation.
- **Blending**: up to 3 candidate biomes per column (2 zones × 2 biomes, merged), with
  weighted density blending. Material and decoration choices are dithered by weight.
- **Props**: one workgroup per chunk. Candidate anchors on a jittered grid go through biome,
  chance, mask, surface-scan and surface-block tests, then all threads rasterise the shapes.
  Props are deterministic per anchor, so they are seamless across chunks.

### Shader build tiers

| Tier | Built how | Driver compile (cold) | Generation speed |
|---|---|---|---|
| Quick | big functions marked SPIR-V `DontInline` | ~3 s | ~5× slower |
| Optimised | everything inlined | ~8 s | ~4 ms per 256 chunks |

With the NVIDIA shader cache warm, either tier is ready in about 50 ms. The quick tier
applies first, and the optimised one replaces it in the background.

## Editing

- The world is always `generate(graph, seed) + overlay`. The overlay is a sparse map of
  edited blocks, stored per LOD0 chunk. Regenerating a chunk re-applies it, so edits survive
  graph changes, and undo/redo just restores overlay entries.
- Region reads (copy, replace, paint, smooth, saving a prefab) queue LOD0 generation jobs
  flagged for readback. Their final voxels are copied to the host and assembled into a box.

## Scripted runs and tests

`--script FILE` runs the engine headless and plays a script one frame at a time
(`crates/engine/src/engine/script.rs` documents the commands). Synthetic mouse events go through
the same path as window events, including the ImGui capture check, and ImGui receives them
directly. Scripts read results back from the GPU: the pick buffer, region reads of the real
blocks, and screenshots compared against half-resolution golden images. For determinism,
scripted runs use a fixed 1/60 s time step for everything time-based (animation clock, brush
repeat, fog blending), fixed generation batch sizes and the default UI layout (`imgui.ini` is
ignored). `settle` waits until streaming, meshing, region reads and shader builds are all idle.
The GPU tests in `crates/engine/tests/gpu_scripts.rs` and the README media
(`docs/media/build.sh`) are both scripts.

## Performance (RTX 4080, 1080p, `--benchmark`)

- 279 fps average, 102 fps 1% low, while flying at 90 blocks/s and streaming about 2,000
  chunks/s.
- Full regeneration of the whole 4 km world (about 9.2k chunks) costs about 0.5 s of GPU time
  with the optimised generator.
