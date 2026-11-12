# World plan: a layered, fully editable world

Approved by the project owner on 2026-11-12, through the question tool. This is the detailed
plan for roadmap items 19–28 in [PLAN.md](PLAN.md). PLAN.md stays the source of truth for
status. Update both as phases land, and record any decision that changes here.

## 1. What the owner asked for

> Everything about the world customised through the node editor, and custom blocks with a
> texture editor in the game. Trees have their own node graph. Then 8 surface zones with 8
> biomes each, 8 underground zones with 8 biomes (huge sprawling caves and small cramped caves)
> spanning thousands of blocks down, and 8 sky zones with 8 biomes thousands of blocks up. The
> sky must not be visible from the ground. The world should be beautiful and feel alive:
> fallen trees, remnants of forest fires, rubble and anything else, with new blocks as needed.

## 2. Decisions (owner, 2026-11-12)

### World layout
- **Three layers**:
  - Sky: about **y 4,000 to 6,000**.
  - Surface: about **y -100 to ~1,000**, with terrain mostly 0–400.
  - Underground: from below the surface down to about **y -6,000**.
- **A cloud sea hides the sky layer.** A thick permanent cloud deck sits between the surface and
  the sky layer, at roughly y 2,500–3,500 (the exact height is the implementer's choice). From
  the ground you see a normal sky with normal clouds. Flying up, you break through the cloud
  sea, and the sky islands sit in clear air above it.
- **Connections**:
  - Sinkholes and chasms from the surface deep into the underground.
  - Sky pillars: colossal natural structures (ancient vines, stone spires) rising from the
    surface through the cloud sea.
  - Both are **very rare**: about one per 15 km or more, true discoveries.
- **A layer jump in the UI**: Surface / Underground (at a chosen depth) / Sky buttons, and a
  minimap per layer.
- **Underground arrangement: 2 depth bands × 4 zones.**
  - Upper caves: about -100 to -1,200.
  - The Deep: -1,200 to -6,000.
  - Each band has 4 zones side by side (Voronoi). Deeper means hotter and stranger.
- **Scale**: surface zones about **6 km** across, biomes about **800 m**. Pick sizes for the
  other layers that feel right and say so at the check-in.
- **Oceans**: one of the 8 surface zones is an ocean zone. There are lakes and rivers
  everywhere.
- **Cave light**: dark, lit by life. Black where nothing glows, but most cave biomes have
  glowing fungi, crystals, lava, glowworms or luminous water. A faint ambient keeps shapes
  readable.

### Generator architecture
- **One generator per layer, plus a GPU graph interpreter.**
  - Each layer compiles separately, so an edit rebuilds only its layer.
  - A bytecode interpreter applies structural edits instantly while the compiled shader builds
    in the background.
  - This **un-defers roadmap item 10**. Earlier measurement, with 16 biomes: GLSL generation
    ~70 ms, NVIDIA driver compile ~2.3 s cold, ~50 ms warm. Expect much worse with ~200
    biomes, which is why this is needed.
- **Authoring: shared building blocks.** Reusable function graphs (terrain shapes, surface
  layers, forests, cave styles, decorations) that each biome graph combines and tunes. Every
  biome is distinct and editable without 192 copies of the same nodes.
- **Today's 16 biomes: keep the names and concepts, completely rework their generation.**
  Floating isles move to the sky layer. The 4 cave biomes go underground.
- **Today's world is replaced** by the new one (no Classic preset). Old saves still open: their
  edits stay, and the terrain under them changes.

### Customisation ("everything in the node editor")
- **All blocks are data.**
  - The hard-coded list in `crates/engine/src/blocks.rs` moves to editable files: `assets/blocks`
    for definitions, `assets/textures` for PNGs. Built-in blocks are just defaults, editable
    like custom ones.
  - A **Block editor** sets the kind, per-face textures, light colour and strength, tint, shape
    (model), rotation behaviour, and how fluids treat the block, etc.
- **Texture editor (in game)**:
  - 32×32 **pixel painting**: pencil, eraser, fill, line, rectangle, colour picker, mirror,
    palette, undo/redo, zoom and grid, and a live preview on a block and in the world.
  - **Animated textures**: frames and speed for water, lava, glowing and magic blocks.
  - Relief painting was **not** selected. Derive the height map from brightness, as the
    procedural generator does now.
- **Prop graphs for all props.**
  - Trees, fallen logs, stumps, boulders, rubble, ruins, mushrooms, coral, crystals and
    sky-island details are all a new **Prop** graph kind.
  - Each has a **3D preview** of one instance and a re-roll button. Biomes scatter them as
    today.
- **Tree graphs: tree nodes built on shape nodes.**
  - Tree-building nodes: trunk with bend and taper, recursive branches, leaf clusters, roots,
    vines, fruit.
  - They sit on general 3D shape nodes: capsules, blobs, noise, combine and transform. So any
    tree, or anything else, can be modelled.
- **More systems become graph nodes**:
  - **Sky & atmosphere**: sky colours, sun and moon, clouds (the cloud sea too), fog and colour
    grading, per layer, zone and biome.
  - **Particles & weather**: user-defined particle types (look, motion, where they spawn,
    day/night), and weather patterns.
  - **Fluids**: custom fluids (e.g. honey, toxic water, liquid light) with flow rules from the
    flowing-water system. Lava flows too.
  - **Light & time**: day length, sun and moon light colours, the ambient light of each layer.

### Content
- **Surface zones (8 × 8 biomes)**:
  1. *Temperate Heartland*: meadow, oak forest, birch hills, flower fields, old-growth forest,
     rolling hills, river marsh, autumn woods.
  2. *Boreal North*: taiga, snowy taiga, tundra, frozen lakes, peat bog, glacier, ice-spike
     fields, aurora plains.
  3. *Arid Expanse*: dunes, badlands mesas, savanna, salt flats, canyonlands, rocky scrub,
     oasis valleys, red rock arches.
  4. *Tropical Wilds*: rainforest, bamboo forest, mangrove swamp, jungle hills, flower jungle,
     river delta, cenote sinkholes, karst towers.
  5. *Ocean & Coast*: deep ocean, warm reef, kelp forest, archipelago, sandy beaches, sea
     stacks & cliffs, icebergs, volcanic islets.
  6. *Highlands*: snowy peaks, alpine meadows, stony crags, highland lakes, pine slopes,
     glacial valleys, plateaus, waterfall gorges.
  7. *Volcanic Wastes*: ashlands, lava fields, obsidian plains, basalt columns, geyser basins,
     burnt forest, hot springs, caldera.
  8. *Enchanted Wilds*: blossom forest, giant mushroom forest, glowing fen, maple woods,
     crystal grove, twisted forest, fairy meadow, moonwood.
- **Underground**, upper caves (about -100 to -1,200):
  1. *Lush Hollows* (open grottos): mossy grottos, root tangles, spring pools, fern caverns,
     glowworm galleries, cave waterfalls, vine halls, ruin-choked caves.
  2. *Crystal Depths* (geode halls): geode caverns, prism halls, quartz tunnels, glitter
     vaults, salt caves, crystal lakes, mirror caves, gem veins.
  3. *Fungal Realm* (huge caverns): giant mushroom caverns, spore fields, mycelium tunnels,
     glowcap groves, puffball halls, shelf-fungus cliffs, mold bogs, truffle burrows.
  4. *Stone Labyrinth* (cramped): twisting tunnels, dripstone caves, limestone galleries,
     fossil beds, collapsed mines, catacombs, flooded tunnels, crawlways.
- **Underground**, the Deep (-1,200 to -6,000):
  5. *Magma Core* (vast): lava seas, magma caves, obsidian halls, basalt pillars, sulfur vents,
     ember chasms, scorched tunnels, molten falls.
  6. *Frozen Deep* (mixed): ice caverns, frost tunnels, frozen sea, blue-ice halls, rime
     grottos, glacier roots, permafrost fossils, frost chasms.
  7. *Sunken Realm* (vast): underground ocean, flooded caverns, glowing coral grottos, sea
     caves, drowned ruins, pearl lagoons, kelp chasms, abyssal trench.
  8. *Ancient Abyss* (colossal): lost city, void chasms, shadow-stone halls, giant bone fields,
     obsidian temples, echo pits, rune caves, the deep floor.
- **Sky layer (8 × 8, above the cloud sea)**:
  1. *Floating Archipelago*: grassy isles, waterfall isles, forest isles, flower isles, rocky
     isles, lake isles, islet swarms, ancient-tree isle.
  2. *Cloud Forest*: cloud-top meadows, mist forest, puff trees, cloud lakes, rainbow mist,
     cloud canyons, dew gardens, cotton fields.
  3. *Crystal Spires*: crystal pillars, prism fields, floating shards, mirror lakes, glass
     forest, light wells, refraction caves, crystal arches.
  4. *Storm Peaks*: thunderheads, charged-stone crags, dark cloud mountains, wind-carved
     spires, rain isles, tempest rings, the calm eye, storm-blasted forest.
  5. *Sky Ruins*: temple isles, broken bridges, overgrown colonnades, toppled statues, sky
     gardens, observatory, chained isles, ancient arenas.
  6. *Aurora Reaches*: floating glaciers, aurora fields, ice spires, snow isles, starlit
     plains, ice-cave isles, polar lakes, crystal snow.
  7. *Inverted Isles*: upside-down islands, hanging gardens, root curtains, vine forests,
     hanging lakes, stalactite isles, inverted peaks, vine bridges.
  8. *World-Tree Canopy*: branch plains, leaf platforms, hollow-trunk caves, blossom canopy,
     giant nests, sap pools, moss bridges, fruit groves.
- **Lived-in details**:
  - *Forest floor*: fallen trees and logs with moss and mushrooms, stumps, leaf litter, twigs,
    berry bushes, flower clusters.
  - *Fire scars*: charred standing trees, burnt logs, ash and scorched ground, young regrowth.
  - *Rubble & ruins*: collapsed walls, overgrown ruins, boulder fields, scree under cliffs,
    abandoned camps and wells.
  - *Cave dressing*: stalactites and stalagmites, hanging roots and vines, glowworm ceilings,
    fossils, geodes, underground ponds.
- **Moving life** (particle-like; no AI, no gameplay):
  - *Birds & butterflies*: bird flocks circling over forests and cliffs; butterflies and bees
    around flowers.
  - *Water life*: fish schools in lakes and the ocean, lily pads, reeds, swaying kelp.
  - *Wind & falling things*: falling leaves under trees, drifting seeds, gusts rippling across
    grass, waterfalls from flowing water.
  - *Night glow*: pulsing glowing plants and fungi, aurora in cold zones, light motes in the
    sky layer. Fireflies exist already.
- **Extras**:
  - *Waterfalls & springs*: generated water and lava falls from cliffs, cave walls and sky
    islands (pouring into the cloud sea), using the flowing-water levels.
  - *Weather events*: thunderstorms with lightning flashes, sandstorms, blizzards and ash
    storms, chosen per zone by the weather graph.
  - *Rare wonders*: giant fossil skeletons, monoliths, meteor craters, colossal ancient trees,
    natural arches.
  - *Colour variety*: per-patch variation of grass and leaf colour (autumn patches,
    sun-bleached grass), so large biomes don't look uniform.

### Editor tooling
- **Graph browser**: a tree of Layers > Zones > Biomes, plus Props, Functions, Particles,
  Fluids and Blocks, with search. It replaces the flat list.
- **Templates & duplicate**: new biome, zone or prop from a template; duplicate a zone with all
  its biomes.
- **Node previews**: small live images on density and noise nodes. This was "not selected"
  before and is **now selected**.
- **Layer map**: the map window shows any layer (sky, surface, or underground at a chosen
  depth) with zones and biomes. Click to teleport.
- Still **not selected**: camera bookmarks / cinematic paths.

### Process
- **Work on `main`.** Commit and push after each verified milestone (CLAUDE.md). Keep CI
  green.
- **Foundations first**, then content, then life and polish (phase order below).
- **Check in with the owner at the end of each phase**, through the question tool: a summary,
  screenshots and any decisions. In between, ask only when blocked on a decision that is
  genuinely the owner's.
- **Performance**: at least **60 fps at 1080p on the owner's RTX 4080**, and ideally as fast as
  possible without giving up beauty. Measure every phase with `--benchmark` and per-layer
  benchmark flights.

## 3. Today's engine: facts that shape the work

Read `README.md`, `docs/ARCHITECTURE.md` and `docs/PLAN.md` first. Key facts:

- **World generation**:
  - Typed graphs (World, Zone, Biome, Function) are compiled to GLSL by
    `crates/worldgen/src/compile.rs`. The node registry is `nodes.rs`, the default world
    `defaults.rs` (exported to `assets/worldgen` with `--export-worldgen-defaults DIR`), and
    auto-layout is in `layout.rs`.
  - Every float parameter is a live `P(i)` in a GPU buffer (64K floats). The last two slots are
    the solo preview (`SOLO_PARAM`).
  - The engine side is `crates/engine/src/wg.rs`, which builds the generator pipelines on a
    background thread, quick tier then optimised tier.
- **Layout nodes**: Zone Voronoi and Biome Voronoi each take exactly **8 inputs**, so 8×8 per
  layer fits. The World Output has a **single** Zones input and the sea level. Layers need new
  structure.
- **Generator passes**:
  - `shaders/worldgen/terrain_main.glsl`: per-column top-down scan. It uses fixed windows for
    depth, open sky and flooding so every chunk agrees (see "Generation consistency" in
    ARCHITECTURE.md). Keep that property: `check_apron` and `tests/scripts/apron.vxs` must stay
    green and must cover layer boundaries too.
  - `props_main.glsl` / `props.glsl`: prop anchors on a jittered grid. Every open surface is a
    candidate. Anchors are sorted and drawn one at a time. Shapes are **hard-coded** (tree
    species `Round, Birch, Spruce, Palm, Acacia, Dead, Bush, Willow`, plus mushroom, boulder,
    column, spike, geode, prefab). Prop graphs replace these.
- **Blocks**:
  - `crates/engine/src/blocks.rs` is a hard-coded `BlockRegistry::new()` list, about 80 blocks.
  - A block value is u16: **id in the low 12 bits (4,096 ids), state in the high 4** (log axis,
    stairs facing, water level).
  - `textures.rs` generates the 32×32 albedo and height of each texture procedurally by name.
    `assets/textures/<name>.png` overrides one.
  - The GPU block table is `GpuBlockInfo`, and `blocks.glsl` is generated with `B_NAME` defines
    for the generator.
- **Meshing**:
  - Packed 8-byte quads. The greedy key is block 16 bits | AO 8 | **biome 7 bits** |
    transparent 1, so **at most 128 biomes** today. The column buffer masks `& 127u`.
  - ~200 biomes need a wider biome field. The biome is used for tints and grading.
- **Streaming**:
  - LOD clipmaps of 32³ chunks, 5 levels, radius 8 chunks horizontal and **4 vertical** per
    level. The coarsest level reaches about ±2,048 blocks vertically and ±4,096 horizontally.
  - Rendering is camera-relative with i32 block coordinates, so y ±6,000 is fine.
  - Empty space between layers should cost almost nothing: skip generating chunks known to be
    empty from the layer ranges.
- **Lighting**:
  - A fine 128³ light volume around the camera (block light + sky light), plus a coarse
    far-field volume for block light.
  - Underground ambient and sky light need per-layer settings.
- **Rendering**:
  - 4 shadow cascades; the far two are cached (`ShadowCache` in `engine.rs`).
  - Volumetric clouds (`CloudSettings`: base 230, thickness 90). The cloud sea is new work.
  - Also TAA, bloom, per-biome grading.
- **Particles**: `crates/engine/src/particles.rs` + `shaders/particles/*`. 24k particles in a
  camera-wrapping box, 6 built-in kinds, shares per biome × weather.
- **Fluids**: `crates/engine/src/fluid.rs` is a CPU simulation over read-back chunks. Water
  levels are in the block state (0 still, 1..7 flowing, 8 falling). Water only.
- **Editing and saves**:
  - Edits live in an overlay (`world/edits.rs`), so they survive regeneration.
  - World files are `saves/NAME.vxw` (`world/save.rs`) and store blocks by name, so data-driven
    blocks keep saves valid **as long as names stay stable**.
- **Testing**:
  - Scripted headless runs (`--script FILE`, language at the top of
    `crates/engine/src/engine/script.rs`).
  - GPU tests in `crates/engine/tests/scripts/*.vxs` (run with
    `cargo test --release -p voxel-engine -- --ignored`), with golden images.
  - Useful commands: `check_apron` (camera-relative `~dx ~dy ~dz`), `solo`, `find_biome`,
    `profile`, `benchmark`.
  - Validation must report 0 errors.
- **CI**: GitHub Actions runs clippy `-D warnings` (CI's Rust can be newer than local, e.g.
  1.100 vs 1.99) and the CPU tests. GPU tests run locally only.

## 4. Phases

Each phase ends verified and committed: build, `cargo test --workspace`, clippy, the GPU tests
and new ones, 0 validation errors, and a benchmark note. Then comes a check-in with the owner
(summary + screenshots). Phase numbers are roadmap items 19–28 in PLAN.md.

### Phase A (item 19): blocks and textures as data
- **Data**:
  - `assets/blocks/*.ron` block definitions: name, display name, category, kind, per-face
    textures, tint, emission, light colour/strength/attenuation, model, rotation behaviour
    (column, facing), fluid interaction (washes away, fluid), animation.
  - Export the current list as defaults (an `--export-block-defaults DIR` like the worldgen
    one). `BlockRegistry` loads from data. Names stay stable.
- **Runtime rebuild**: create, edit and delete blocks at runtime:
  - rebuild the GPU block table and texture array;
  - regenerate `blocks.glsl` and rebuild the generators;
  - remesh;
  - keep ids stable within a session, and map by name across sessions (graphs, prefabs and
    saves already use names).
- **Block editor window**: list with search and categories, create/duplicate/delete, all
  properties, per-face texture picker, place-test in the world.
- **Texture editor window**:
  - 32×32 pixel painting with pencil, eraser, fill, line, rectangle, picker, mirror X/Y,
    palette + recent colours, undo/redo, zoom, grid.
  - Live preview on a rotating block and in the world (apply instantly).
  - **Animated textures**: frame list, add/duplicate/delete frames, fps, onion skin. Stored as
    PNG (frames as a vertical strip or numbered files).
  - The height map comes from brightness for painted textures.
  - Start a texture from any existing one (the procedural ones included).
- **Renderer**: animated textures (frames as texture array layers, frame count and fps in
  `GpuBlockInfo`, the frame picked from time in the shaders).
- **Tests**: unit tests for block data round trips and name stability; a GPU script that
  creates a block, places it and checks it (`expect_block`, `expect_pick`); a golden of a
  painted texture on a block.

### Phase B (item 20): Prop graphs, trees included
- **New graph kind `Prop`**:
  - Evaluated per anchor in prop-local space (relative position, per-instance seed), returning
    a block or nothing.
  - Declares its bounds (radius, height, depth below) for the anchor search.
  - Compiled to GLSL like densities.
- **Shape nodes**: sphere, ellipsoid, capsule/segment, box, cylinder, cone, blob noise,
  displace, union / subtract / intersect / smooth union, transform (offset, rotate, scale,
  mirror, repeat), per-instance random, block assignment with priorities (which block wins
  where shapes overlap).
- **Tree nodes on top of the shape nodes**:
  - Trunk: height range, bend, twist, taper, base flare.
  - Branches: recursion levels, count, angle, length falloff, droop.
  - Leaf clusters: shape, radius, density, holes.
  - Roots, vines / hanging leaves, fruit and flowers.
  - Performance matters here. Build each anchor's segment list once (for example in shared
    memory in the props pass), then test voxels against it. Don't re-derive the tree per voxel.
- **Scatter / prop layers** reference Prop graphs. Migrate every built-in shape (8 tree
  species, mushroom, boulder, column, spike, geode) to default Prop graphs. The Prefab node
  stays.
- **Prop editor**: a 3D preview of one instance (offscreen render shown in an ImGui window,
  orbit camera), a re-roll seed button, bounds shown.
- **Tests**: golden images of the default trees in a preview row; `check_apron` across props;
  benchmark the props pass against today's (it is ~1.7 ms per loading frame now).

### Phase C (item 21): layered world and per-layer generators + interpreter
- **Layers in the World graph**:
  - Layer nodes (Sky / Surface / Underground) with vertical ranges and blend bands, each with
    its zone layout and layer settings (ambient light, sky and fog, cloud sea).
  - A **Depth Bands** layout node gives the underground its 2 bands × 4 zones.
  - The surface ↔ underground boundary must be seamless. Sky pillars and sinkholes cross
    layers.
- **Per-layer generator pipelines**:
  - Chunks dispatch to their layer's pipeline. Chunks in a boundary band need a defined
    owner, or both pipelines with a blend.
  - Skip empty space between layers without generating it.
- **GPU graph interpreter (item 10)**:
  - Bytecode from the same graph IR as the GLSL compiler.
  - One interpreter shader per pass (terrain, props, probe) used right after structural edits
    until the compiled layer pipeline is ready.
  - The results must be identical: a test compares interpreter and compiled output (region
    reads or goldens).
- **Scale limits**:
  - Widen the biome field (quad key, column buffer, biome tables) to at least 256.
  - Revisit `MAX_ANCHORS` (384), prop layers per biome (6), and node input counts if needed.
- **Streaming across layers**:
  - Vertical LOD that works at y -6,000 and +6,000.
  - The underground stays cheap while the camera is on the surface (occlusion culling helps,
    but don't generate what can't be seen if you can avoid it).
- **Cloud sea**:
  - A dense cloud deck, part of the volumetric clouds or a dedicated pass, with an
    altitude-dependent look. From below it reads as a normal sky; you fly through it; above
    it, a sea of clouds.
  - The sky layer must not be visible from the ground.
- **Layer jump UI and per-layer minimap**: find_biome / probes per layer.
- **Tests**: `apron.vxs` extended across layer boundaries; goldens at each layer, at the cloud
  sea from below and from above, and of a sinkhole; benchmark flights per layer.

### Phase D (item 22): environment as graphs
- **Sky & atmosphere** outputs per layer, zone and biome: sky colours, sun and moon size and
  colour, clouds (and the cloud sea), fog, grading.
- **Particles & weather**:
  - A **Particle** graph kind (or nodes): look (shape, colour, size, glow), motion (fall,
    drift, rise, swirl, flock, orbit), spawn rules (biome, near blocks such as flowers, water
    or leaves, day/night, weather, layer).
  - Weather pattern nodes: storms with lightning flashes, sandstorms, blizzards, ash storms,
    intensity cycles.
  - Generalise the GPU particle system beyond the 6 kinds.
- **Fluids as data and graphs**: fluid definitions (flow distance, speed, light, colour,
  texture, whether sources are infinite). Generalise `fluid.rs`; lava flows.
- **Light & time** in the World graph: day length, sun and moon light colours, per-layer
  ambient (dark-lit-by-life underground), sky light behaviour in the sky layer.
- **Tests**: goldens for a custom particle type and a weather event; a fluid script test for a
  second fluid.

### Phase E (item 23): editor tooling
- **Graph browser**: a tree of Layers > Zones > Biomes, plus Props, Functions, Particles,
  Fluids, Blocks; search; open in the node editor.
- **Templates & duplicate** for biomes, zones and props.
- **Node previews**: thumbnails on density and noise nodes, 2D slices evaluated on the GPU (the
  interpreter makes this cheap), throttled.
- **Layer map**: any layer, underground at a chosen depth, zones and biomes coloured, click to
  teleport.

### Phase F (item 24): surface content (8 × 8)
- Shared function graphs first: terrain archetypes (plains, hills, mountains, mesas, dunes,
  karst, coasts, ocean floors), surface material sets, vegetation sets.
- Then the 64 biomes. Today's biome names and concepts stay, and their generation is redone.
- **New blocks and textures** as needed. See the suggested list in section 5.
- Rivers and lakes everywhere, the ocean zone with reefs and kelp, waterfalls and springs.
- **Check-in**: screenshots of every zone (a contact sheet is fine).

### Phase G (item 25): underground content (2 bands × 4 × 8)
- Cave systems per zone: huge caverns (Fungal Realm, Magma Core, Sunken Realm, Ancient Abyss),
  cramped tunnels (Stone Labyrinth), mixed (Frozen Deep, Crystal Depths, Lush Hollows).
- Underground seas and lava seas.
- Cave dressing: stalactites, roots, glowworms, fossils, geodes, ponds.
- **Very rare** sinkholes and chasms connecting to the surface.
- Lighting reads "dark, lit by life".

### Phase H (item 26): sky content (8 × 8)
- Islands and structures above the cloud sea, waterfalls pouring into it, sky ruins, the
  world-tree canopy, inverted isles.
- **Very rare** sky pillars from the surface.

### Phase I (item 27): life and details
- **Static details**: forest floor, fire scars, rubble & ruins, cave dressing (if not done in
  G), colour variety, rare wonders.
- **Moving life**: birds & butterflies, water life (fish schools, lily pads, reeds, kelp
  sway), wind & falling things, night glow, weather events.

### Phase J (item 28): performance and polish
- Benchmarks per layer and across layer travel. At least 60 fps at 1080p on the RTX 4080,
  ideally much higher.
- Regenerate the README media (`docs/media/build.sh`) with the new world, plus clips of the
  layers, flowing water and the editors.
- Update ARCHITECTURE.md and README.md.

## 5. Suggested new blocks (starting list; add or drop freely)

- **Forest**: leaf litter, twigs, moss carpet, mossy log, mossy cobblestone, berry bush, autumn
  leaves (red/orange/yellow), clover, brown and red mushrooms, shelf fungus.
- **Fire**: charred log, charcoal, ash, scorched grass, burnt soil, glowing embers, sprouts.
- **Ruins**: cracked stone bricks, mossy stone bricks, rubble (model), broken pillar, rotten
  planks, rusty iron, old tiles, statue stone.
- **Arid**: rippled sand, salt crust, cracked mud, smooth and carved sandstone, cactus flower,
  tumbleweed, dry shrub.
- **Tropical**: jungle log and leaves, bamboo, mangrove log and roots, lily pad, reeds,
  tropical flowers, hanging vines, mud.
- **Ocean**: coral blocks and coral plants in several colours, kelp, seagrass, sea stone, wet
  sand, shells, glowing pearl.
- **Cold**: permafrost, blue ice, rime ice, frosted leaves, tundra grass, peat, cotton grass.
- **Highlands**: alpine grass, mountain flowers, granite, slate, scree.
- **Volcanic**: sulfur, geyser rock, pumice, scoria, glowing magma rock, hot spring water (a
  fluid).
- **Enchanted**: blossom leaves, maple leaves, glowing moss, moonstone, glowing flowers,
  twisted wood, silver bark.
- **Underground**: dripstone, stalactite and stalagmite models, limestone, glowworm silk,
  hanging roots, fossil bone, crystals in several colours, salt crystal, mycelium, glowcap,
  shadow stone, rune stone (glowing), deep-stone variants, glowing coral.
- **Sky**: cloud block (soft, translucent), sky grass, skystone, floating crystal, charged
  stone (glowing), aurora ice, marble and gold-veined marble, giant canopy wood, sap (a fluid),
  nest twigs.

## 6. Risks to watch

- **Generator size and compile time** with ~200 biomes. This is the reason for per-layer
  pipelines and the interpreter. Measure early in Phase C.
- **Seams**: between layers (the surface ↔ underground boundary), at chunk borders
  (`check_apron`), and where an interpreter result differs from compiled output.
- **Underground cost while on the surface**: dense, cave-filled chunks within the vertical
  radius.
- **The quad format change** for wider biome ids: the greedy key, the quad layout, the shaders
  that unpack it.
- **Texture array size** with animated frames and many custom blocks (memory, upload time).
- **Determinism**: goldens must stay deterministic (fixed time step in scripts). Particles and
  weather restart at `settle`. Keep it that way for new systems.
