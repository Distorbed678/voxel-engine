# Project plan

This file is the source of truth for **what was decided**, **what is done** and **what is next**.
Update it whenever a milestone lands or a decision changes.

## 1. Goal

Build an ultra-fast Minecraft-style voxel engine in Rust and Vulkan, with a Dear ImGui debug UI
(dear-imgui-rs). It has a Hytale "World Gen V2"-style generator that you edit in a visual node
editor, with changes appearing live in the world. The world is organized into zones, and each
zone contains biomes.

It is **not a game**: no player, no survival mechanics, just a free-flying camera with block
editing tools.

## 2. Decisions (made by the project owner, 2026-11-11)

### Engine core

| Topic | Choice |
|---|---|
| GPU target | Modern GPUs only: Vulkan 1.4 + mesh shaders (RTX 20+ / RX 6000+ / Arc). No fallback paths. Dev box: RTX 4080, Wayland, CachyOS. |
| Where worldgen runs | GPU compute. The node graph compiles to a compute shader. |
| Rendering technique | Mesh shaders, GPU-driven. Greedy-meshed packed quads, task-shader culling, one indirect dispatch. |
| View distance | About 4 km, with LOD. Far LODs are generated directly at low resolution. |
| Chunks / world size | Cubic chunks (32³), infinite in all 3 axes. |

### Look

| Topic | Choice |
|---|---|
| Lighting style | Stylized, Hytale-like: smooth lighting + AO, colored block light, biome tints, sky and fog, sun shadows (shadow maps). |
| Extra block shapes | Transparent blocks (water, glass, ice), slabs and stairs, custom small models. Cross-quad plants were **not** selected; plants are done as small models. |
| Textures | Procedural 32×32 with normal maps (generated in Rust; PNG overrides allowed). |
| Atmosphere | Day/night + sky, volumetric clouds, wind and water motion, per-biome fog and color grading. |

### World

| Topic | Choice |
|---|---|
| Zone layout | Voronoi zones, each with its own biome map. Borders blend. |
| Built-in zones | Temperate, Arid, Frozen, Volcanic (2–4 biomes each). |
| Terrain features | Cave systems + cave biomes, oceans/rivers/lakes, overhangs + sky islands, ore veins + geodes. Trees, grass, flowers and rocks are always included. |

### Node editor and editing

| Topic | Choice |
|---|---|
| Node editor library | dear-node-editor (imgui-node-editor bindings). |
| Graph organisation | Hytale-style typed graphs: World → Zone → Biome (Density / Materials / Props / Environment), plus reusable Function graphs. |
| Edits vs regeneration | Edits are kept as a sparse overlay on top of generation, so they survive every graph change. |
| Editing tools | Brushes, selection + clipboard, prefabs usable by worldgen props, undo/redo history. |

### Extras

| Topic | Choice |
|---|---|
| Debug tools | GPU profiler panel, render debug views, worldgen overlays (zone/biome tint + minimap with teleport), benchmark mode. |
| Node editor extras | Inline curve editor; editor quality-of-life (search palette, copy/paste, comments, group-to-function, graph undo/redo). **Not** selected: live preview thumbnails, isolated 3D biome preview. |
| Engine extras | Save/load (RON graphs, world edits, prefabs, seed), screenshots (incl. headless), shader hot reload. **Not** selected: camera bookmarks / cinematic paths. |

### Workflow

| Topic | Choice |
|---|---|
| Location | `~/Projects/voxel-engine`, git, T3 Code project "Voxel Engine". |
| Validation | `vulkan-validation-layers` installed by the owner; use `--validate` while developing. |
| Public repo | https://github.com/Distorbed678/voxel-engine (public), branch `main`. Commits keep the existing author identity. |
| License | MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`). |
| Questions | Claude asks questions and proposes features through the question tool (see `CLAUDE.md`). Claude commits and pushes when a milestone is done. |

## 3. Implementation decisions taken during the build

These follow from the choices above.

- **ash 0.38 + dear-imgui-ash 0.22** (dynamic rendering). Buffers are reached through buffer
  device addresses in push constants. There is one global pipeline layout: a bindless image
  set + 256 bytes of push constants.
- **Shader compilation**: GLSL is compiled at runtime with the system shaderc library. This
  makes hot reload and generated worldgen shaders possible.
- **Chunk voxels are transient.** Each chunk is generated with a 1-cell apron, so meshing
  needs no neighbours. Only meshes stay resident: 8-byte quads in a 512 MiB pool.
- **Two-pass GPU greedy mesher** (count → prefix → write). The CPU only reads back per-chunk
  totals to allocate pool ranges.
- **Live parameters**: every float/int node parameter is `P(i)` in a GPU buffer. Value edits
  never recompile; structural edits do.
- **Two-tier generator builds**: a quick build (SPIR-V `DontInline`, about 3 s cold) applies
  first, then a fully inlined build (about 8 s cold, roughly 5× faster generation) replaces it
  in the background. The NVIDIA shader cache makes repeat builds about 50 ms.
- **Edits at coarse LODs** are point-sampled. They are exact at LOD0.
- **Stairs** auto-orient against an opaque neighbour, because blocks have no rotation state yet.

## 4. Status

### Done

- [x] **M0 Setup**: workspace (`vkcore`, `worldgen`, `engine`), git, T3 project, validation
  layers.
- [x] **M1 Vulkan core**
  - device (mesh shaders, BDA, bindless, timeline semaphores, sync2, dynamic rendering),
    swapchain (Wayland), frames in flight
  - ImGui integration, fly camera, GPU profiler, headless offscreen mode + screenshots
- [x] **M2 GPU world pipeline**
  - worldgen compute and props pass
  - edit overlay, GPU greedy meshing, quad pool, chunk records
  - LOD clipmaps (5 levels, about 4 km) with hole-free retirement
  - task/mesh terrain rendering, GPU frustum + per-direction culling
  - procedural textures + normal maps, per-cell AO, GPU picking
- [x] **M3 Node-graph world generator**
  - graph model (RON) and about 60 node types
  - GLSL compiler: 2D hoisting, domain-warp contexts, live parameter buffer
  - zone/biome Voronoi with blending, per-zone cave carvers
  - default world: 4 zones and 12 biomes, including floating isles, badlands, glacier,
    snowy peaks and lava fields
  - node editor: browser, inspector, inline curves, search palette, copy/paste/duplicate,
    undo/redo, group-to-function
  - tiered background builds, biome tints, zone/biome debug overlays
- [x] **M4 Models**: plants, flowers, ferns, mushrooms, crystal clusters, slabs, stairs and
  lanterns via task/mesh shaders. Plants sway in the wind and pick up biome tints.
- [x] **M5 Lighting/atmosphere (partial)**
  - 4-cascade sun shadows with PCF and alpha-tested leaves
  - day/night with sun, moon and stars
  - volumetric clouds
  - water: waves, Fresnel sky reflection, depth absorption, glints
  - leaf wind, height + distance fog with per-biome blending (via GPU probes)
- [x] **M6 Editing**
  - single block, brushes (add, remove, paint, smooth)
  - box selection: fill, clear, replace, copy, paste, rotate
  - async GPU region reads, undo/redo + history panel
  - prefabs (save a selection, load to the clipboard, Prefab prop node, built-in ruin)
- [x] **M7 Tooling**
  - minimap with zone/biome view and click-to-teleport
  - benchmark mode (`--benchmark` → `benchmarks/`), adaptive per-frame generation budget
  - world edit save/load (`saves/world.ron`), F2 screenshots, shader hot reload (F5 + file
    watcher)
  - texture export (`--export-textures`), `--worldgen DIR`
- [x] **Docs**: README.md, docs/ARCHITECTURE.md, this plan.
- [x] **Input handling fix**: UI panels regain mouse capture after an RMB look-drag (before,
  clicks on panels fell through and edited the blocks behind them). Button and key releases now
  always reach ImGui; held mouse buttons are cleared on focus loss. Covered by a unit test that
  replays the event sequence through a real ImGui context. Still needs a hand check.
- [x] **Colored block light + sky light**: a 128³ toroidal light volume around the camera
  (filled by LOD0 generation jobs and light-only jobs), sky columns + Jacobi propagation on the
  GPU, smooth trilinear sampling. Lava, magma, lamps, lanterns, crystals and glow mushrooms light
  their surroundings in colour; caves without sky access go dark. About 0.25 ms on frames where
  it runs (only after changes). Golden images cover a lit room and a removed light source.
- [x] **TAA**: Halton-jittered projection, compute resolve with depth reprojection (jitter
  removed), Catmull-Rom history, YCoCg variance clipping, clamped sharpen in the tonemap pass.
  About 0.04 ms at 720p. Toggle in the Render panel (`taa on|off` in scripts).
- [x] **Bloom**: half-resolution mip chain (13-tap downsample with soft threshold + Karis
  average, tent upsample), added in the tonemap pass. About 0.04 ms. Strength and threshold
  sliders in the Render panel.
- [x] **Cave biomes**: Zone Output "Cave Biomes" input; Cave Biome Select (3D warped cells);
  carved-cell tracking gives exact cave floors/ceilings that take the cave biome's material,
  floors get its ground cover. Defaults: lush (moss, glow mushrooms), crystal (calcite, crystal
  clusters), magma (basalt, magma), frost (snow, packed ice) caves, all lit by block light.
  `find_cave` script command.
- [x] **Rivers**: River node (warped noise zero line, even width via |n|/|grad n|) into the World
  Output's Rivers input; valleys carved to a bed below sea level, filled with the biome's fluid
  (water, ice or lava), river-bed block.
- [x] **Transparency order fix**: transparent draw groups sorted back to front on the GPU (the
  "dark rectangle / grid lines on glacier lakes" issue was blend order between chunks).
- [x] **Per-biome colour grading**: Biome Output tint / saturation / contrast, blended towards
  the camera's biome like fog, applied in the tonemap pass ("Grading" toggle).
- [x] **Overdraw debug view**: fragment shaders count shaded fragments per pixel into an R32 image
  (only in that view), shown as a heatmap. It shows 8–16+ shaded fragments in forests, so early-Z
  is mostly not happening (discard + the pick atomic): input for the Hi-Z work.
- [x] **Node editor**: Auto-layout button (columns by distance to the output, barycentre
  ordering, estimated node heights; the default graphs are laid out with it), Comment boxes
  (resizable group boxes with a note, text edited in the inspector).
- [x] **Far-field block light**: a coarse 4-blocks-per-cell volume (block light only) over the
  LOD0 box, filled by LOD0 jobs as a side effect, sampled beyond the fine volume. Distant lava
  lights its banks.

**Measured** (RTX 4080, 1080p, `--benchmark`):
- 279 fps average, 102 fps 1% low, while streaming about 2,000 chunks/s.
- Full 4 km world regeneration takes about 0.5 s of GPU time.
- Validation-layer runs report 0 errors.

### Verification

Covered by the GPU script tests (`cargo test --release -p voxel-engine -- --ignored`):
- [x] Picking: break/place/pick block in fly mode and in editor mode, undo/redo shortcuts
- [x] Brushes: add/remove/paint/smooth (async region reads) and hold-to-repeat
- [x] Selection: replace/clear/copy/rotate/paste, saving a prefab
- [x] UI capture: panels own the mouse after an RMB look-drag (the input fix)
- [x] Rendering: six golden images (daylight, badlands, night, lava, biome view, normals)
- [x] ImGui window drag/resize and node-canvas zoom via synthetic input (used by the media
  script)

Still needs a hand test (real input devices or complex UI flows):
- [ ] Fly mode: mouse look, WASD, speed wheel, Tab/Esc cursor grab on Wayland
- [ ] The UI capture fix in a real window
- [ ] Node editor: link dragging, drag-to-empty-space create, right-click/Space palette,
  delete, Ctrl+C/V/D, graph undo, group-to-function, inline curve dragging
- [ ] Prefab node picking up a newly saved prefab
- [ ] Save/load of world edits, VSync toggle, window resize

### Known issues (found while capturing media)

- [x] ~~The default biome graphs have overlapping nodes~~: laid out with auto-layout.
- [x] ~~Dark rectangle / grid lines on the glacier lake~~: fixed by sorting transparent draws.
- [x] ~~Desert oasis water looks lavender~~: not water; dry sinkholes where ravines and caves reach
  the surface (they now get cave biome floors).
- [ ] Apron cells on the Y faces of a chunk miss decorations and props owned by the chunk below
  (found with `check_apron`); harmless for meshing today.

### Not done yet (owner's choices that are still open)


### Approved additions (owner, 2026-11-11)

- [x] **Public GitHub repo** with a showcase README (screenshots, flythrough, node editor),
  license files. Media is regenerated by `docs/media/build.sh` (scripted, deterministic).
- [x] **Automated test harness.** `--script FILE`: scripted camera, mouse/keyboard, tools,
  biome search, screenshots/recording and checks, deterministic (fixed time step). Five GPU
  regression scripts + golden images in `crates/engine/tests/`.
- [x] **GitHub Actions CI.** Clippy (`-D warnings`) and the CPU tests on every push, with a badge
  in the README. GPU rendering can't run in CI.
- [ ] **Anti-aliasing (TAA).** There is no AA today, so the 4 km view shimmers. Temporal AA
  with jittered projection, motion vectors from the previous view-projection, neighbourhood
  clamping.
- [ ] **Ambient particles + weather.** GPU-simulated ash (volcanic), snowfall (frozen),
  fireflies (night), rain, driven by the biome probes.

## 5. Work order

Approved by the owner on 2026-11-11 ("work through it"). Items 1–3 come from Claude's
suggestions; the rest is the earlier roadmap.

1. ~~Input handling fix~~ (done, needs a hand check).
2. ~~Public repo, license, showcase README, GitHub Actions CI.~~
3. ~~Automated test harness (scripted input + screenshot regression)~~, covering most of the
   hand-test checklist.
4. ~~Colored block light + sky light~~. Follow-ups worth considering: a larger volume or a
   cheap far-field glow for distant lava (the volume reaches ±64 blocks), bloom.
5. ~~TAA~~, then ~~bloom~~ (approved 2026-11-11).
6. ~~**Far-field lava glow**~~ (approved 2026-11-11).
7. ~~Cave biomes + river node~~.
8. ~~Small open items: colour grading, overdraw view, node comments, node auto-layout~~.
9. Ambient particles + weather.
10. GPU graph interpreter, so structural edits are instant while the compiled shader builds
   (today: about 3 s cold, about 50 ms cached).
11. Hi-Z occlusion culling (main pass is 2–2.5 ms in dense views).
12. Hytale-style noise/height biome selectors (alongside Voronoi).
13. Block rotation states (stairs, logs) and partial-height water.

Still not selected (ask before doing): node preview thumbnails, isolated biome preview,
camera bookmarks / cinematic paths.
