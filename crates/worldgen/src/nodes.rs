//! Node type registry: pins, parameters, categories and help text.
//! Code generation for each node lives in `compile.rs`.

use crate::graph::{GraphKind, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PinType {
    /// Scalar field f(x, y, z) (or f(x, z) - dimensionality is inferred).
    Density,
    /// Block chooser for solid cells.
    Material,
    /// Single-block surface decoration chooser.
    Decor,
    /// A prop shape (tree, boulder, ...).
    Prop,
    /// A placed prop layer (shape + distribution).
    PropLayer,
    Biome,
    Zone,
}

impl PinType {
    pub fn compatible(out: PinType, input: PinType) -> bool {
        out == input
    }
    pub fn color(self) -> [f32; 4] {
        match self {
            PinType::Density => [0.35, 0.75, 1.0, 1.0],
            PinType::Material => [0.95, 0.65, 0.3, 1.0],
            PinType::Decor => [0.5, 0.9, 0.4, 1.0],
            PinType::Prop => [0.3, 0.8, 0.5, 1.0],
            PinType::PropLayer => [0.2, 0.65, 0.35, 1.0],
            PinType::Biome => [0.85, 0.45, 0.9, 1.0],
            PinType::Zone => [1.0, 0.85, 0.3, 1.0],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ParamKind {
    /// Live parameter: lives in the GPU parameter buffer, editing never recompiles.
    Float { default: f32, min: f32, max: f32 },
    /// Live integer (stored as float on the GPU).
    Int { default: i32, min: i32, max: i32 },
    Bool(bool),
    Enum(&'static [&'static str], usize),
    Block(&'static str),
    Ref(GraphKind),
    Color([f32; 3]),
    Curve,
    /// name of a saved prefab (stored as Value::Ref)
    Prefab,
}

impl ParamKind {
    pub fn default_value(&self) -> Value {
        match *self {
            ParamKind::Float { default, .. } => Value::F(default),
            ParamKind::Int { default, .. } => Value::I(default),
            ParamKind::Bool(b) => Value::B(b),
            ParamKind::Enum(opts, i) => Value::E(opts[i].to_string()),
            ParamKind::Block(b) => Value::Block(b.to_string()),
            ParamKind::Ref(_) => Value::Ref(String::new()),
            ParamKind::Color(c) => Value::Color(c),
            ParamKind::Curve => Value::Curve(vec![[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]]),
            ParamKind::Prefab => Value::Ref("ruin".into()),
        }
    }
    /// Live params can change without recompiling the generator.
    pub fn is_live(&self) -> bool {
        matches!(self, ParamKind::Float { .. } | ParamKind::Int { .. } | ParamKind::Color(_))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ParamDef {
    pub name: &'static str,
    pub kind: ParamKind,
}

pub struct NodeDef {
    pub kind: &'static str,
    pub title: &'static str,
    pub category: &'static str,
    pub help: &'static str,
    pub inputs: &'static [(&'static str, PinType)],
    pub outputs: &'static [(&'static str, PinType)],
    pub params: &'static [ParamDef],
    /// Graph kinds this node may be placed in (empty = all).
    pub allowed: &'static [GraphKind],
}

const fn f(name: &'static str, default: f32, min: f32, max: f32) -> ParamDef {
    ParamDef { name, kind: ParamKind::Float { default, min, max } }
}
const fn i(name: &'static str, default: i32, min: i32, max: i32) -> ParamDef {
    ParamDef { name, kind: ParamKind::Int { default, min, max } }
}
const fn b(name: &'static str, default: bool) -> ParamDef {
    ParamDef { name, kind: ParamKind::Bool(default) }
}
const fn blk(name: &'static str, default: &'static str) -> ParamDef {
    ParamDef { name, kind: ParamKind::Block(default) }
}
const fn en(name: &'static str, opts: &'static [&'static str], default: usize) -> ParamDef {
    ParamDef { name, kind: ParamKind::Enum(opts, default) }
}
const fn rf(name: &'static str, kind: GraphKind) -> ParamDef {
    ParamDef { name, kind: ParamKind::Ref(kind) }
}
const fn col(name: &'static str, c: [f32; 3]) -> ParamDef {
    ParamDef { name, kind: ParamKind::Color(c) }
}

use PinType::*;

const D: &[(&str, PinType)] = &[("Out", Density)];
const AB: &[(&str, PinType)] = &[("A", Density), ("B", Density)];
const X: &[(&str, PinType)] = &[("In", Density)];
const NOISE: &[ParamDef] = &[
    f("size", 256.0, 1.0, 8192.0),
    i("octaves", 4, 1, 10),
    f("lacunarity", 2.0, 1.1, 4.0),
    f("gain", 0.5, 0.05, 1.0),
    f("amplitude", 1.0, 0.0, 512.0),
    i("seed", 0, 0, 1000),
];
const ALL: &[GraphKind] = &[];
const BIOME_ZONE_FN: &[GraphKind] = &[GraphKind::Biome, GraphKind::Zone, GraphKind::Function];
const BIOME: &[GraphKind] = &[GraphKind::Biome];
const ZONE: &[GraphKind] = &[GraphKind::Zone];
const WORLD: &[GraphKind] = &[GraphKind::World];
const FUNC: &[GraphKind] = &[GraphKind::Function];

const SPECIES: &[&str] = &["Round", "Birch", "Spruce", "Palm", "Acacia", "Dead", "Bush", "Willow"];

pub static NODES: &[NodeDef] = &[
    // ---------------------------------------------------------------- editor
    NodeDef { kind: "comment", title: "Comment", category: "Editor",
        help: "A resizable box with a note. Drag it to move the nodes inside; the generator ignores it.",
        inputs: &[], outputs: &[],
        params: &[f("width", 420.0, 60.0, 8000.0), f("height", 260.0, 40.0, 8000.0)],
        allowed: ALL },
    // ---------------------------------------------------------------- outputs
    NodeDef { kind: "world_output", title: "World Output", category: "Output", help: "Root of the generator: lays out zones across the infinite world.",
        inputs: &[("Zones", Zone), ("Rivers", Density)], outputs: &[],
        params: &[
            f("sea_level", 62.0, -256.0, 512.0),
            f("river_depth", 5.0, 0.0, 64.0),
            f("river_carve", 48.0, 0.0, 512.0),
            blk("river_bed", "gravel"),
        ],
        allowed: WORLD },
    NodeDef { kind: "zone_output", title: "Zone Output", category: "Output", help: "A zone: which biomes appear inside it and how caves are carved.",
        inputs: &[("Biomes", Biome), ("Caves", Density), ("Cave Biomes", Biome)], outputs: &[], params: &[col("map_color", [0.4, 0.7, 0.3])], allowed: ZONE },
    NodeDef { kind: "biome_output", title: "Biome Output", category: "Output",
        help: "A biome: terrain density (>0 = solid), materials, ground cover, props and environment.",
        inputs: &[("Terrain", Density), ("Material", Material), ("Ground Cover", Decor), ("Props 1", PropLayer), ("Props 2", PropLayer), ("Props 3", PropLayer), ("Props 4", PropLayer), ("Props 5", PropLayer), ("Props 6", PropLayer)],
        outputs: &[],
        params: &[
            en("fluid", &["Water", "Lava", "Ice", "None"], 0),
            col("grass_tint", [0.45, 0.75, 0.3]),
            col("foliage_tint", [0.35, 0.65, 0.25]),
            col("water_tint", [0.15, 0.35, 0.65]),
            col("fog_color", [0.7, 0.8, 0.95]),
            f("fog_density", 1.0, 0.0, 10.0),
            col("grade_tint", [1.0, 1.0, 1.0]),
            en("ambient", &["None", "Ash", "Embers", "Fireflies", "Pollen"], 0),
            en("precipitation", &["None", "Rain", "Snow"], 0),
            f("particle_density", 1.0, 0.0, 4.0),
            f("saturation", 1.0, 0.0, 2.0),
            f("contrast", 1.0, 0.5, 2.0),
            col("map_color", [0.3, 0.6, 0.25]),
        ],
        allowed: BIOME },
    NodeDef { kind: "function_output", title: "Function Output", category: "Output", help: "Result of this reusable density function.",
        inputs: &[("Value", Density)], outputs: &[], params: &[], allowed: FUNC },
    NodeDef { kind: "function_input", title: "Function Input", category: "Function", help: "Argument passed in by a Function node (0-3).",
        inputs: &[], outputs: D, params: &[i("index", 0, 0, 3)], allowed: FUNC },
    NodeDef { kind: "function_call", title: "Function", category: "Function", help: "Evaluate a reusable density function graph.",
        inputs: &[("Arg 0", Density), ("Arg 1", Density), ("Arg 2", Density), ("Arg 3", Density)], outputs: D,
        params: &[rf("function", GraphKind::Function)], allowed: BIOME_ZONE_FN },

    // ---------------------------------------------------------------- sources
    NodeDef { kind: "constant", title: "Constant", category: "Density/Source", help: "A constant value.", inputs: &[], outputs: D, params: &[f("value", 0.0, -1000.0, 1000.0)], allowed: ALL },
    NodeDef { kind: "coord", title: "Coordinate", category: "Density/Source", help: "World coordinate X, Y or Z (in blocks).", inputs: &[], outputs: D, params: &[en("axis", &["X", "Y", "Z"], 1)], allowed: ALL },
    NodeDef { kind: "simplex2d", title: "Simplex 2D", category: "Density/Noise", help: "Fractal simplex noise over X/Z (cheap: computed once per column).", inputs: &[], outputs: D, params: NOISE, allowed: ALL },
    NodeDef { kind: "simplex3d", title: "Simplex 3D", category: "Density/Noise", help: "Fractal 3D simplex noise.", inputs: &[], outputs: D, params: NOISE, allowed: ALL },
    NodeDef { kind: "ridged2d", title: "Ridged 2D", category: "Density/Noise", help: "Ridged multifractal (mountain ridges).", inputs: &[], outputs: D, params: NOISE, allowed: ALL },
    NodeDef { kind: "ridged3d", title: "Ridged 3D", category: "Density/Noise", help: "3D ridged multifractal.", inputs: &[], outputs: D, params: NOISE, allowed: ALL },
    NodeDef { kind: "billow2d", title: "Billow 2D", category: "Density/Noise", help: "Billowy, puffy noise.", inputs: &[], outputs: D, params: NOISE, allowed: ALL },
    NodeDef { kind: "billow3d", title: "Billow 3D", category: "Density/Noise", help: "3D billow noise.", inputs: &[], outputs: D, params: NOISE, allowed: ALL },
    NodeDef { kind: "worley2d", title: "Cellular 2D", category: "Density/Noise", help: "Worley / Voronoi noise over X/Z.", inputs: &[], outputs: D,
        params: &[f("size", 64.0, 1.0, 4096.0), f("jitter", 1.0, 0.0, 1.0), en("mode", &["F1", "F2", "F2-F1", "Cell value"], 0), f("amplitude", 1.0, 0.0, 512.0), i("seed", 0, 0, 1000)], allowed: ALL },
    NodeDef { kind: "worley3d", title: "Cellular 3D", category: "Density/Noise", help: "3D Worley noise.", inputs: &[], outputs: D,
        params: &[f("size", 32.0, 1.0, 4096.0), f("jitter", 1.0, 0.0, 1.0), en("mode", &["F1", "F2", "F2-F1", "Cell value"], 0), f("amplitude", 1.0, 0.0, 512.0), i("seed", 0, 0, 1000)], allowed: ALL },
    NodeDef { kind: "distance", title: "Distance", category: "Density/Source", help: "Distance from a point (2D ignores Y). Great for islands or spawn rings.", inputs: &[], outputs: D,
        params: &[f("x", 0.0, -100000.0, 100000.0), f("y", 64.0, -10000.0, 10000.0), f("z", 0.0, -100000.0, 100000.0), en("mode", &["2D", "3D"], 0)], allowed: ALL },
    NodeDef { kind: "river", title: "River", category: "Density/Terrain",
        help: "Meandering river network over X/Z: 1 in the channel, falling to 0 across the banks. Feed the World Output's Rivers input to carve channels filled to sea level.",
        inputs: &[], outputs: D,
        params: &[
            f("spacing", 1100.0, 64.0, 16384.0),
            f("width", 9.0, 1.0, 256.0),
            f("bank", 36.0, 0.0, 512.0),
            f("warp", 180.0, 0.0, 2048.0),
            i("seed", 0, 0, 1000),
        ],
        allowed: ALL },
    NodeDef { kind: "y_gradient", title: "Y Gradient", category: "Density/Terrain", help: "Linear ramp in Y: value_low at y_low to value_high at y_high.", inputs: &[], outputs: D,
        params: &[f("y_low", 0.0, -2048.0, 2048.0), f("y_high", 128.0, -2048.0, 2048.0), f("value_low", 1.0, -100.0, 100.0), f("value_high", -1.0, -100.0, 100.0)], allowed: ALL },
    NodeDef { kind: "height_terrain", title: "Height to Density", category: "Density/Terrain",
        help: "Turns a 2D height (blocks) into terrain density: height - y. The workhorse terrain node.", inputs: &[("Height", Density)], outputs: D,
        params: &[f("base", 64.0, -1024.0, 1024.0), f("scale", 1.0, 0.0, 512.0), f("falloff", 1.0, 0.01, 10.0)], allowed: ALL },

    // ---------------------------------------------------------------- math
    NodeDef { kind: "add", title: "Add", category: "Density/Math", help: "A + B", inputs: AB, outputs: D, params: &[], allowed: ALL },
    NodeDef { kind: "subtract", title: "Subtract", category: "Density/Math", help: "A - B", inputs: AB, outputs: D, params: &[], allowed: ALL },
    NodeDef { kind: "multiply", title: "Multiply", category: "Density/Math", help: "A * B", inputs: AB, outputs: D, params: &[], allowed: ALL },
    NodeDef { kind: "min", title: "Min", category: "Density/Math", help: "min(A, B) - union of empty space.", inputs: AB, outputs: D, params: &[], allowed: ALL },
    NodeDef { kind: "max", title: "Max", category: "Density/Math", help: "max(A, B) - union of solids.", inputs: AB, outputs: D, params: &[], allowed: ALL },
    NodeDef { kind: "smooth_min", title: "Smooth Min", category: "Density/Math", help: "Smooth minimum (rounded blend).", inputs: AB, outputs: D, params: &[f("smoothness", 8.0, 0.0, 128.0)], allowed: ALL },
    NodeDef { kind: "smooth_max", title: "Smooth Max", category: "Density/Math", help: "Smooth maximum (rounded blend).", inputs: AB, outputs: D, params: &[f("smoothness", 8.0, 0.0, 128.0)], allowed: ALL },
    NodeDef { kind: "abs", title: "Abs", category: "Density/Math", help: "|x|", inputs: X, outputs: D, params: &[], allowed: ALL },
    NodeDef { kind: "negate", title: "Negate", category: "Density/Math", help: "-x", inputs: X, outputs: D, params: &[], allowed: ALL },
    NodeDef { kind: "square", title: "Square", category: "Density/Math", help: "x*x (keeps sign if 'signed')", inputs: X, outputs: D, params: &[b("signed", false)], allowed: ALL },
    NodeDef { kind: "pow", title: "Power", category: "Density/Math", help: "sign(x)*|x|^exponent", inputs: X, outputs: D, params: &[f("exponent", 2.0, 0.1, 8.0)], allowed: ALL },
    NodeDef { kind: "clamp", title: "Clamp", category: "Density/Math", help: "Clamp to [min, max]", inputs: X, outputs: D, params: &[f("min", -1.0, -1000.0, 1000.0), f("max", 1.0, -1000.0, 1000.0)], allowed: ALL },
    NodeDef { kind: "remap", title: "Remap", category: "Density/Math", help: "Linear remap from [in_min, in_max] to [out_min, out_max].", inputs: X, outputs: D,
        params: &[f("in_min", -1.0, -1000.0, 1000.0), f("in_max", 1.0, -1000.0, 1000.0), f("out_min", 0.0, -1000.0, 1000.0), f("out_max", 1.0, -1000.0, 1000.0), b("clamp", false)], allowed: ALL },
    NodeDef { kind: "scale_bias", title: "Scale & Bias", category: "Density/Math", help: "x * scale + bias", inputs: X, outputs: D, params: &[f("scale", 1.0, -1000.0, 1000.0), f("bias", 0.0, -1000.0, 1000.0)], allowed: ALL },
    NodeDef { kind: "lerp", title: "Lerp", category: "Density/Math", help: "mix(A, B, T) with T clamped to [0,1]", inputs: &[("A", Density), ("B", Density), ("T", Density)], outputs: D, params: &[], allowed: ALL },
    NodeDef { kind: "select", title: "Select", category: "Density/Math", help: "Selector > threshold ? B : A, with a smooth blend band.", inputs: &[("A", Density), ("B", Density), ("Selector", Density)], outputs: D,
        params: &[f("threshold", 0.0, -100.0, 100.0), f("blend", 0.1, 0.0, 50.0)], allowed: ALL },
    NodeDef { kind: "curve", title: "Curve", category: "Density/Math", help: "Remap the input through an editable curve (drag the points).", inputs: X, outputs: D,
        params: &[ParamDef { name: "curve", kind: ParamKind::Curve }], allowed: ALL },
    NodeDef { kind: "terrace", title: "Terrace", category: "Density/Math", help: "Quantise into steps (mesas, rice terraces).", inputs: X, outputs: D, params: &[f("step", 8.0, 0.1, 256.0), f("smoothness", 0.2, 0.0, 1.0)], allowed: ALL },
    NodeDef { kind: "lod_switch", title: "LOD Switch", category: "Density/Terrain", help: "Use the cheap input on coarse LODs (e.g. skip caves far away).", inputs: &[("Detail", Density), ("Coarse", Density)], outputs: D,
        params: &[i("min_lod", 2, 1, 8)], allowed: ALL },

    // ---------------------------------------------------------------- transforms
    NodeDef { kind: "warp", title: "Domain Warp", category: "Density/Transform", help: "Evaluate Source at a position offset by the warp inputs (organic shapes).",
        inputs: &[("Source", Density), ("Warp X", Density), ("Warp Z", Density), ("Warp Y", Density)], outputs: D, params: &[f("strength", 32.0, 0.0, 1024.0)], allowed: ALL },
    NodeDef { kind: "scale_coords", title: "Stretch", category: "Density/Transform", help: "Evaluate Source at scaled coordinates (stretch / squash).", inputs: &[("Source", Density)], outputs: D,
        params: &[f("x", 1.0, 0.01, 100.0), f("y", 1.0, 0.01, 100.0), f("z", 1.0, 0.01, 100.0)], allowed: ALL },
    NodeDef { kind: "translate", title: "Offset", category: "Density/Transform", help: "Evaluate Source at offset coordinates.", inputs: &[("Source", Density)], outputs: D,
        params: &[f("x", 0.0, -100000.0, 100000.0), f("y", 0.0, -10000.0, 10000.0), f("z", 0.0, -100000.0, 100000.0)], allowed: ALL },

    // ---------------------------------------------------------------- carvers
    NodeDef { kind: "spaghetti_caves", title: "Spaghetti Caves", category: "Caves", help: "Long winding tunnels (positive = carve).", inputs: &[], outputs: D,
        params: &[f("size", 80.0, 8.0, 1024.0), f("thickness", 0.1, 0.01, 0.5), f("y_min", -256.0, -4096.0, 4096.0), f("y_max", 70.0, -4096.0, 4096.0), i("max_lod", 1, 0, 8), i("seed", 0, 0, 1000)], allowed: BIOME_ZONE_FN },
    NodeDef { kind: "cheese_caves", title: "Cheese Caves", category: "Caves", help: "Large open caverns with pillars.", inputs: &[], outputs: D,
        params: &[f("size", 110.0, 8.0, 1024.0), f("threshold", 0.6, 0.0, 1.0), f("y_min", -256.0, -4096.0, 4096.0), f("y_max", 40.0, -4096.0, 4096.0), i("max_lod", 1, 0, 8), i("seed", 0, 0, 1000)], allowed: BIOME_ZONE_FN },
    NodeDef { kind: "noodle_caves", title: "Noodle Caves", category: "Caves", help: "Thin squiggly tunnels.", inputs: &[], outputs: D,
        params: &[f("size", 32.0, 4.0, 512.0), f("thickness", 0.06, 0.01, 0.3), f("y_min", -256.0, -4096.0, 4096.0), f("y_max", 30.0, -4096.0, 4096.0), i("max_lod", 0, 0, 8), i("seed", 0, 0, 1000)], allowed: BIOME_ZONE_FN },
    NodeDef { kind: "ravines", title: "Ravines", category: "Caves", help: "Deep narrow canyons cut from the surface.", inputs: &[], outputs: D,
        params: &[f("size", 700.0, 64.0, 8192.0), f("width", 0.035, 0.005, 0.2), f("y_top", 140.0, -1024.0, 1024.0), f("depth", 70.0, 4.0, 512.0), i("max_lod", 2, 0, 8), i("seed", 0, 0, 1000)], allowed: BIOME_ZONE_FN },

    // ---------------------------------------------------------------- materials
    NodeDef { kind: "block", title: "Block", category: "Material", help: "Always this block.", inputs: &[], outputs: &[("Out", Material)], params: &[blk("block", "stone")], allowed: BIOME },
    NodeDef { kind: "surface_layers", title: "Surface Layers", category: "Material", help: "Classic layering: top / under / base, with an underwater variant and cave-floor handling.",
        inputs: &[("Base", Material)], outputs: &[("Out", Material)],
        params: &[blk("top", "grass"), f("top_depth", 1.0, 0.0, 16.0), blk("under", "dirt"), f("under_depth", 4.0, 0.0, 64.0), blk("underwater", "sand"), f("beach_height", 2.0, 0.0, 16.0), blk("base", "stone"), blk("cave_floor", "stone")],
        allowed: BIOME },
    NodeDef { kind: "height_select", title: "Height Select", category: "Material", help: "Below / Above a (noisy) height. Snow caps, tree lines...",
        inputs: &[("Below", Material), ("Above", Material), ("Noise", Density)], outputs: &[("Out", Material)], params: &[f("height", 140.0, -2048.0, 2048.0), f("noise_amount", 6.0, 0.0, 256.0)], allowed: BIOME },
    NodeDef { kind: "depth_select", title: "Depth Select", category: "Material", help: "Shallow (near surface) vs Deep material.",
        inputs: &[("Shallow", Material), ("Deep", Material)], outputs: &[("Out", Material)], params: &[f("depth", 3.0, 0.0, 256.0)], allowed: BIOME },
    NodeDef { kind: "noise_select", title: "Noise Select", category: "Material", help: "Patches: Selector > threshold ? B : A.",
        inputs: &[("A", Material), ("B", Material), ("Selector", Density)], outputs: &[("Out", Material)], params: &[f("threshold", 0.3, -10.0, 10.0)], allowed: BIOME },
    NodeDef { kind: "ore", title: "Ore Veins", category: "Material", help: "Sprinkle ore blobs into the Base material.",
        inputs: &[("Base", Material)], outputs: &[("Out", Material)],
        params: &[blk("ore", "coal_ore"), f("size", 6.0, 1.0, 64.0), f("rarity", 0.8, 0.0, 1.0), f("y_min", -256.0, -4096.0, 4096.0), f("y_max", 128.0, -4096.0, 4096.0), blk("replace", "stone"), i("seed", 0, 0, 1000)], allowed: BIOME },
    NodeDef { kind: "strata", title: "Strata", category: "Material", help: "Horizontal coloured bands (badlands / mesas).",
        inputs: &[("Base", Material)], outputs: &[("Out", Material)],
        params: &[blk("band1", "terracotta"), blk("band2", "terracotta_orange"), blk("band3", "terracotta_white"), blk("band4", "terracotta_red"), blk("band5", "terracotta_yellow"), blk("band6", "terracotta_brown"),
                  f("band_height", 3.0, 1.0, 32.0), f("wobble", 4.0, 0.0, 64.0), f("min_depth", 1.0, 0.0, 32.0), blk("replace", "stone")], allowed: BIOME },
    NodeDef { kind: "cave_select", title: "Sky / Cave Select", category: "Material", help: "Open-to-sky material vs cave material.",
        inputs: &[("Open", Material), ("Cave", Material)], outputs: &[("Out", Material)], params: &[], allowed: BIOME },

    // ---------------------------------------------------------------- decorations
    NodeDef { kind: "ground_cover", title: "Ground Cover", category: "Decoration", help: "Single-block plants on top of a surface block (grass, flowers, shrubs).",
        inputs: &[("Mask", Density)], outputs: &[("Out", Decor)],
        params: &[blk("on", "grass"), f("chance", 0.3, 0.0, 1.0), blk("plant1", "tall_grass"), f("weight1", 8.0, 0.0, 100.0), blk("plant2", "flower_red"), f("weight2", 1.0, 0.0, 100.0),
                  blk("plant3", "flower_yellow"), f("weight3", 1.0, 0.0, 100.0), blk("plant4", "fern"), f("weight4", 2.0, 0.0, 100.0), i("seed", 0, 0, 1000)], allowed: BIOME },
    NodeDef { kind: "decor_mix", title: "Decor Stack", category: "Decoration", help: "Try A, then B.", inputs: &[("A", Decor), ("B", Decor)], outputs: &[("Out", Decor)], params: &[], allowed: BIOME },

    // ---------------------------------------------------------------- props
    NodeDef { kind: "tree", title: "Tree", category: "Props", help: "Procedural tree.", inputs: &[], outputs: &[("Out", Prop)],
        params: &[en("species", SPECIES, 0), blk("trunk", "oak_log"), blk("leaves", "oak_leaves"), f("height_min", 5.0, 2.0, 64.0), f("height_max", 9.0, 2.0, 64.0), f("radius", 3.0, 1.0, 12.0)], allowed: BIOME },
    NodeDef { kind: "giant_mushroom", title: "Giant Mushroom", category: "Props", help: "Huge mushroom.", inputs: &[], outputs: &[("Out", Prop)],
        params: &[blk("stem", "mushroom_stem"), blk("cap", "mushroom_cap_red"), f("height_min", 6.0, 3.0, 48.0), f("height_max", 11.0, 3.0, 48.0), f("radius", 4.0, 2.0, 12.0)], allowed: BIOME },
    NodeDef { kind: "boulder", title: "Boulder", category: "Props", help: "Lumpy rock.", inputs: &[], outputs: &[("Out", Prop)],
        params: &[blk("block", "cobblestone"), blk("block2", "mossy_cobblestone"), f("radius_min", 1.5, 0.5, 12.0), f("radius_max", 3.0, 0.5, 12.0)], allowed: BIOME },
    NodeDef { kind: "column", title: "Column", category: "Props", help: "Vertical column (cactus, basalt pillar). Optional arms.", inputs: &[], outputs: &[("Out", Prop)],
        params: &[blk("block", "cactus"), f("height_min", 2.0, 1.0, 64.0), f("height_max", 5.0, 1.0, 64.0), f("radius", 0.5, 0.5, 6.0), b("arms", true)], allowed: BIOME },
    NodeDef { kind: "spike", title: "Spike", category: "Props", help: "Tall cone (ice spikes, rock spires).", inputs: &[], outputs: &[("Out", Prop)],
        params: &[blk("block", "packed_ice"), f("height_min", 8.0, 2.0, 96.0), f("height_max", 24.0, 2.0, 96.0), f("radius", 2.5, 0.5, 12.0)], allowed: BIOME },
    NodeDef { kind: "prefab", title: "Prefab", category: "Props", help: "Place a saved prefab (Tools > Select > Save selection as prefab).", inputs: &[], outputs: &[("Out", Prop)],
        params: &[ParamDef { name: "prefab", kind: ParamKind::Prefab }, b("random_rotation", true), f("sink", 1.0, 0.0, 16.0)], allowed: BIOME },
    NodeDef { kind: "scatter", title: "Scatter", category: "Props",
        help: "Place a prop on the surface on a jittered grid. Mask (0..1) modulates density, e.g. noise -> forests and clearings.",
        inputs: &[("Prop", Prop), ("Mask", Density)], outputs: &[("Out", PropLayer)],
        params: &[f("spacing", 9.0, 2.0, 256.0), f("chance", 0.6, 0.0, 1.0), f("jitter", 0.9, 0.0, 1.0), blk("surface", "grass"), b("any_surface", false),
                  f("y_min", -512.0, -4096.0, 4096.0), f("y_max", 512.0, -4096.0, 4096.0), i("seed", 0, 0, 1000)], allowed: BIOME },
    NodeDef { kind: "geode", title: "Geodes", category: "Props", help: "Hollow crystal geodes buried underground.", inputs: &[], outputs: &[("Out", PropLayer)],
        params: &[blk("outer", "basalt"), blk("middle", "calcite"), blk("inner", "crystal"), f("radius", 6.0, 3.0, 16.0), f("spacing", 96.0, 24.0, 1024.0), f("chance", 0.5, 0.0, 1.0),
                  f("y_min", -200.0, -4096.0, 4096.0), f("y_max", 20.0, -4096.0, 4096.0), i("seed", 0, 0, 1000)], allowed: BIOME },

    // ---------------------------------------------------------------- biome & zone layout
    NodeDef { kind: "biome", title: "Biome", category: "Layout", help: "Reference a biome graph.", inputs: &[], outputs: &[("Out", Biome)], params: &[rf("biome", GraphKind::Biome)], allowed: ZONE },
    NodeDef { kind: "cave_biomes", title: "Cave Biome Select", category: "Layout",
        help: "Pick cave biomes in warped 3D cells underground (feed the Zone Output's Cave Biomes). A cave biome's Material covers cave floors and ceilings; its Ground Cover grows on cave floors.",
        inputs: &[("Biome 1", Biome), ("Biome 2", Biome), ("Biome 3", Biome), ("Biome 4", Biome)],
        outputs: &[("Out", Biome)],
        params: &[
            f("cell_size", 64.0, 8.0, 1024.0),
            f("warp", 24.0, 0.0, 256.0),
            f("weight1", 1.0, 0.0, 10.0),
            f("weight2", 1.0, 0.0, 10.0),
            f("weight3", 1.0, 0.0, 10.0),
            f("weight4", 1.0, 0.0, 10.0),
            i("seed", 0, 0, 1000),
        ],
        allowed: ZONE },
    NodeDef { kind: "biome_threshold", title: "Biome Threshold Select", category: "Layout",
        help: "Pick biomes by the value of a 2D selector (temperature or humidity noise, continentalness, a height map...): the thresholds split it into bands, one per connected biome in order, blended across each threshold.",
        inputs: &[("Selector", Density), ("Biome 1", Biome), ("Biome 2", Biome), ("Biome 3", Biome), ("Biome 4", Biome), ("Biome 5", Biome), ("Biome 6", Biome)],
        outputs: &[("Out", Biome)],
        params: &[
            f("threshold1", -0.4, -1000.0, 1000.0),
            f("threshold2", -0.1, -1000.0, 1000.0),
            f("threshold3", 0.1, -1000.0, 1000.0),
            f("threshold4", 0.4, -1000.0, 1000.0),
            f("threshold5", 0.7, -1000.0, 1000.0),
            f("blend", 0.08, 0.0, 100.0),
        ],
        allowed: ZONE },
    NodeDef { kind: "biome_voronoi", title: "Biome Voronoi", category: "Layout",
        help: "Scatter the connected biomes over irregular cells with blended borders. Weights bias how often each appears.",
        inputs: &[("Biome 1", Biome), ("Biome 2", Biome), ("Biome 3", Biome), ("Biome 4", Biome), ("Biome 5", Biome), ("Biome 6", Biome), ("Biome 7", Biome), ("Biome 8", Biome)],
        outputs: &[("Out", Biome)],
        params: &[f("cell_size", 420.0, 32.0, 8192.0), f("jitter", 0.9, 0.0, 1.0), f("blend", 48.0, 0.0, 512.0), f("warp", 60.0, 0.0, 1024.0),
                  f("weight1", 1.0, 0.0, 100.0), f("weight2", 1.0, 0.0, 100.0), f("weight3", 1.0, 0.0, 100.0), f("weight4", 1.0, 0.0, 100.0),
                  f("weight5", 1.0, 0.0, 100.0), f("weight6", 1.0, 0.0, 100.0), f("weight7", 1.0, 0.0, 100.0), f("weight8", 1.0, 0.0, 100.0), i("seed", 0, 0, 1000)],
        allowed: ZONE },
    NodeDef { kind: "zone", title: "Zone", category: "Layout", help: "Reference a zone graph.", inputs: &[], outputs: &[("Out", Zone)], params: &[rf("zone", GraphKind::Zone)], allowed: WORLD },
    NodeDef { kind: "zone_voronoi", title: "Zone Voronoi", category: "Layout",
        help: "Lay out zones as huge warped Voronoi regions (several km). Borders blend biomes from both sides.",
        inputs: &[("Zone 1", Zone), ("Zone 2", Zone), ("Zone 3", Zone), ("Zone 4", Zone), ("Zone 5", Zone), ("Zone 6", Zone), ("Zone 7", Zone), ("Zone 8", Zone)],
        outputs: &[("Out", Zone)],
        params: &[f("cell_size", 2400.0, 256.0, 32768.0), f("jitter", 0.85, 0.0, 1.0), f("blend", 96.0, 0.0, 1024.0), f("warp", 400.0, 0.0, 4096.0),
                  f("weight1", 1.0, 0.0, 100.0), f("weight2", 1.0, 0.0, 100.0), f("weight3", 1.0, 0.0, 100.0), f("weight4", 1.0, 0.0, 100.0),
                  f("weight5", 1.0, 0.0, 100.0), f("weight6", 1.0, 0.0, 100.0), f("weight7", 1.0, 0.0, 100.0), f("weight8", 1.0, 0.0, 100.0), i("seed", 0, 0, 1000)],
        allowed: WORLD },
];

pub fn def(kind: &str) -> Option<&'static NodeDef> {
    NODES.iter().find(|d| d.kind == kind)
}

pub fn allowed_in(d: &NodeDef, g: GraphKind) -> bool {
    if d.kind.ends_with("_output") {
        return false; // exactly one output node per graph, created automatically
    }
    d.allowed.is_empty() || d.allowed.contains(&g)
}

pub fn categories() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::new();
    for d in NODES {
        if !v.contains(&d.category) {
            v.push(d.category);
        }
    }
    v
}
