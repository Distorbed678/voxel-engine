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

**Measured** (RTX 4080, 1080p, `--benchmark`):
- 279 fps average, 102 fps 1% low, while streaming about 2,000 chunks/s.
- Full 4 km world regeneration takes about 0.5 s of GPU time.
- Validation-layer runs report 0 errors.

### In progress

- [ ] **Input handling fix** (being worked on in the follow-up thread). UI panels must regain
  mouse capture after an RMB look-drag; before the fix, clicks on panels fell through and
  edited blocks behind them. Touches `crates/engine/src/input.rs` and `engine.rs`.

### Not yet verified by hand

Rendering was only checked headlessly or in a short windowed run. Each of these needs an
interactive test:
- [ ] Fly mode: mouse look, WASD, speed wheel, Tab/Esc cursor grab on Wayland
- [ ] Picking: break/place/pick block in fly mode and in editor mode
- [ ] Brushes: paint/smooth (async region reads) and hold-to-repeat
- [ ] Selection: copy/paste/replace/rotate, saving a prefab, and the Prefab node picking it up
- [ ] Node editor: link dragging, drag-to-empty-space create, right-click/Space palette,
  delete, Ctrl+C/V/D, graph undo, group-to-function, inline curve dragging
- [ ] Save/load of world edits, VSync toggle, window resize

### Not done yet (owner's choices that are still open)

- [ ] **Colored block light.** Emissive blocks glow but don't light their surroundings. Plan:
  GPU light propagation into a clipmap 3D texture around the camera, sampled trilinearly for
  smooth lighting (sky light + RGB block light).
- [ ] **Cave biomes.** Caves exist, but there are no distinct underground biomes yet (lush,
  crystal, magma). Plan: a 3D biome selector in the Zone graph, plus cave floor/ceiling
  materials and cave props (glow mushrooms, crystals, stalactites).
- [ ] **Rivers.** Oceans and lakes come from sea level. Plan: a River node that carves channels
  with a consistent water level.
- [ ] **Colour grading per biome.** Fog is per-biome; grading (tint, contrast, saturation in the
  tonemapper) is not yet.
- [ ] **Overdraw debug view.** It's listed in the debug-view combo but not implemented.
- [ ] **Node comments / group boxes** (editor QoL).

## 5. Suggested next work (not yet approved)

Rough priority order:

1. Finish and verify the interactive input fix, then hand-test the checklist above.
2. Colored block light propagation (the biggest missing piece of the Hytale look).
3. Cave biomes + river node (completes the requested terrain features).
4. GPU graph interpreter, so structural edits are instant while the compiled shader builds
   (today: about 3 s cold, about 50 ms cached).
5. Hi-Z occlusion culling (main pass is 2–2.5 ms in dense views).
6. Hytale-style noise/height biome selectors (alongside Voronoi).
7. Block rotation states (stairs, logs) and partial-height water.
8. Optional extras you previously skipped: node preview thumbnails, isolated biome preview,
   camera bookmarks.
