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
                                      cull phase 1 (frustum + per-direction, visible last frame)
                                      shadow cascades x4 (cull + task/mesh depth)
                                      main pass 1: terrain (task/mesh) + models
                                      depth pyramid -> cull phase 2 (occlusion) -> sort transparent
                                      main pass 2: newly visible terrain + models + sky
                                      translucent pass: water + volumetric clouds
                                      TAA resolve (compute) -> history
                                      bloom: 13-tap downsample chain -> tent upsample (compute)
                                      tonemap (+ TAA sharpen, + bloom) + ImGui -> swapchain
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

## Block values

- A voxel is a u16: the block id in the low 12 bits and a state in the high 4 (`BLOCK_ID` /
  `BLOCK_STATE` in `shaders/voxel.glsl`, `blocks::block_id` on the CPU). Table lookups mask
  the id. Greedy merging compares whole values, so blocks with different states never merge.
- **Column blocks** (logs, basalt, cactus: one texture on the ends, another on the sides) store
  their axis: 0 = Y, 1 = X, 2 = Z. `face_layer` puts the end texture on that axis, and the
  fragment shader turns the bark grain to follow it.
- **Stairs** store a facing. 0 keeps the automatic orientation (back step against an opaque
  neighbour), and 1..4 fix the model rotation.
- Placing a block orients it: a log takes the axis of the face it is placed against, and stairs
  face away from the camera. Rotating the clipboard turns the states along with the blocks
  (`BlockRegistry::rotated_y`), and so do prefab props placed with a random rotation
  (`rotate_value` in the generator prelude).
- Prefab files store `name` or `name:state` palette entries. Test scripts use the same form.
- **Water** surfaces are drawn 1/8 block below the top of their cell. The mesh shader lowers the
  top face and the top edge of exposed sides, so a lake sits a little below its banks.

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
- **Occlusion culling** (two-phase): phase 1 draws the chunks that were visible last frame. A
  depth pyramid is built from that depth: a power-of-two level 0, with each texel holding the
  farthest depth below it, made with a MIN-reduction sampler. Phase 2 tests every chunk's box
  against the pyramid, at the level where the box covers at most 2×2 texels, so one fetch gives
  the farthest occluder. Visible chunks not drawn in phase 1 are drawn then, transparent quads
  come only from visible chunks, and the per-record visibility bits update for the next frame.
  No popping, since phase 2 catches anything that just became visible. Measured main pass:
  forest at ground level 2.1 → 1.2 ms, inside a cave 2.4 → 0.4 ms.
- **Task and mesh shaders**: 32 threads × 2 quads gives 64 quads per meshlet (256 vertices,
  128 triangles).
- **Models** (plants, slabs, stairs, ...): 6 instances × 10 quad slots per meshlet. Boundary
  faces are culled against opaque neighbours.
- **Shadows**: 4 cascades, texel-snapped in world space (f64). Only faces turned away from the
  sun are drawn, plus a slope bias. The two near cascades are fitted to their view frustum
  slices and redrawn every frame. The two far ones (most of the cost) are cached. Each is a
  window centred on the camera, wide enough for its slice in any direction plus a margin
  (16 / 32 blocks), so turning around doesn't invalidate it. Each keeps the matrix it was drawn
  with, shifted by how far the camera has moved since. A far cascade is redrawn when the camera
  leaves its margin, the sun turns by more than 0.1°, or a LOD 0–1 chunk inside its footprint
  changes (at most every 8 frames). At most one optional redraw happens per frame.
  Occlusion-culling shadow casters was tried and doesn't pay: with back faces only, solid
  ground leaves nothing in the map to hide what is under it, and those casters (cave ceilings)
  are what keep sunlight out of caves.
- **Precision**: rendering is camera-relative, with an integer camera block plus a fractional
  offset, so it stays precise anywhere in the i32 world.

## Transparency order

Transparent quads (water, ice, glass) are blended without depth writes. The cull pass appends
draw groups with atomics, so their order between chunks is arbitrary, and layered surfaces (water
under ice or glass) used to blend in the wrong order wherever parallax made a pixel see layers
from two chunks. `sort_trans.comp` sorts up to 4096 transparent groups back to front in one
workgroup: a bitonic sort in shared memory on a 64-bit key of distance, chunk, direction and
group. Within a chunk, quads are already ordered by slice.

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

## Particles and weather

A pool of 24,576 GPU particles lives in a 64³ box that wraps around the camera: a particle that
leaves one side comes back on the other, so the camera is always surrounded
(`crates/engine/src/particles.rs`, `shaders/particles/`). Each frame `sim.comp` respawns free slots
with a kind picked from per-kind shares, steers velocities towards per-kind targets (falling snow,
fast rain, drifting ash, rising embers, wandering fireflies, pollen), and kills particles inside
solid blocks. Rain and snow also die under a roof, using the light volume's sky sources. The
translucent pass draws camera-facing quads (rain is stretched along its motion). Embers and
fireflies are emissive, so bloom makes them glow.

The shares come from the camera's biome: each Biome Output has ambient particles (ash, embers,
fireflies at night, pollen by day), precipitation (rain or snow) and a density. They're multiplied
by a weather intensity that cycles slowly by itself or is set by hand. Weather also thickens the
clouds and the fog. Everything blends over time like fog. Scripts restart the particles and snap
the blend when `settle` finishes, so golden images stay deterministic.

## Colour grading

Each biome output has a grading tint, saturation and contrast. Like fog, they blend toward the
biome the camera stands in (from the GPU probes). The tonemap pass applies them in linear HDR
before ACES: tint, then saturation around luminance, then contrast around mid-grey (0.18).

## Bloom

A half-resolution RGBA16F mip chain with up to six levels (`crates/engine/src/bloom.rs`). The
first downsample reads the TAA output with the 13-tap filter from Call of Duty: Advanced Warfare,
a soft brightness threshold (quadratic knee, measured after exposure) and a Karis average per 2×2
group against fireflies. Further downsamples use the plain 13-tap filter. Upsampling adds a 3×3
tent of each smaller mip onto the next larger one, and the tonemap pass adds mip 0 times the
bloom strength.

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

**Far field**: a second, coarse volume (block light only, 4 blocks per cell, 128×64×128 cells)
covers the LOD0 streaming box (±256 blocks horizontally at default settings). LOD0 jobs also
summarise their blocks into it, one cell per thread: the brightest emission and the number of
light-blocking blocks. Since this box matches the LOD0 box, chunks entering it are being generated
anyway and need no extra jobs. Propagation uses a step of 4/15 per cell, which gives the same
15-block reach. Surfaces outside the fine volume sample the coarse one, half a cell in front of
the face, so distant lava still lights its banks.

## World generator (crates/worldgen)

Graph kinds:

- **World**: a Zone Voronoi node feeds the World Output, which also holds the sea level and the
  river settings. A River node (or any 2D mask) feeds its Rivers input.
- **Zone**: biome layout (Biome Voronoi, Biome Threshold Select over any 2D selector such as
  humidity noise or a height map, or a single Biome), the cave carver density, and the cave
  biomes (a Biome, or Cave Biome Select picking up to four biomes in warped 3D cells).
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
- **Caves**: the density function reports cells removed by the zone's carver, so the column
  scan knows cave air exactly. The solid cell below cave air is a cave floor and the solid cell
  above it a ceiling. Both take the cave biome's material (compiled a second time in flat form,
  without the 2D column cache, since cave biomes aren't in the column's blend), and floors get its
  ground cover. Natural overhangs aren't carved, so they don't count as caves.
- **Rivers**: a 2D mask in [0, 1] per column. Near a river, terrain density is blended toward a
  bed at `sea_level - river_depth * mask`, up to `river_carve` above sea level, so floating islands
  survive. Rivers fill with the column biome's fluid at sea level (water, ice or lava) and get a
  bed block. The River node follows the zero line of warped noise, using `|n| / |grad n|` as the
  distance, so channel width stays even.
- **Props**: one workgroup per chunk. Candidate anchors on a jittered grid go through biome,
  chance, mask, surface-scan and surface-block tests, then all threads rasterise the shapes.
  The surface scan takes every open surface the prop could reach the chunk from: a solid cell
  under 8 cells of air that no carver removed, which rules out cave floors and spots under low
  overhangs. It only covers heights the layer accepts (height limits, dry ground). Anchors are
  sorted by layer and position and drawn one at a time, since overlapping props resolve by order.
  So every chunk a prop touches draws the same blocks.

### Solo preview

The editor's Solo button shows one zone or biome everywhere. The zone and biome indices sit in
two fixed slots at the end of the parameter buffer (not after the graph's parameters, whose
count changes while a rebuild is pending), so switching just regenerates the world. `wg_column`
forces the zone (a soloed surface biome brings a zone that uses it, for its carver and caves)
and the biome, and `wg_cave_biome` returns a soloed cave biome wherever a zone has caves. The
minimap, probes and biome search see the same world.

### Generation consistency

A chunk also generates its neighbours' apron cells, so a cell must come out the same whichever
chunk generates it. The terrain pass scans each column top-down and measures what a cell sees
above it over fixed windows, never "as far as this chunk happens to look":

- **depth** (material layering): the run of solid cells above, capped at the deepest material
  threshold of the column's biomes (`wg_depth_max`, from the live parameters).
- **open sky** (ocean floors, flooding): no solid cell up to 8 cells above, or, for a cell below
  sea level, up to 8 cells above sea level. A cave below sea level floods only if it opens
  upwards past sea level.
- The 8-cell look-ahead above the chunk covers these windows above sea level. Otherwise the
  scan probes upwards from the top of the look-ahead: below a solid cell for deep layering
  (only desert columns by default), below an air cell for a roof when sea level is far above.
- It also evaluates the cell below the slot, for what that cell does to the bottom apron
  (cave ceiling material, ground cover).

The `check_apron` script command compares a chunk's apron with its neighbours' blocks;
`tests/scripts/apron.vxs` covers places that once disagreed.

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
- **World files** (`saves/NAME.vxw`, RON): the seed, camera, time of day and the overlay. The
  world itself is regenerated, so a save only holds what the generator can't reproduce. Blocks
  are stored by name (`name` or `name:state`) through a palette, so saves survive changes to
  the block list. Each chunk's edits are runs of consecutive cells with one block, so filled
  boxes stay small. Writes go to a temporary file first, then replace the old one. Files from
  before this format (raw block ids) still load.

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
