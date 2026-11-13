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
    /// A prop for a Scatter (a Prop graph or a prefab).
    Prop,
    /// Shapes of a Prop graph: primitives built once per placed instance.
    Shape,
    /// Tree skeleton in a Prop graph: trunk and branch segments that branches, leaves, roots,
    /// vines and fruit grow from.
    Skeleton,
    /// A placed prop layer (shape + distribution).
    PropLayer,
    Biome,
    Zone,
    /// A layer of the world (surface, underground band, sky): a vertical range with its zones.
    Layer,
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
            PinType::Shape => [0.95, 0.85, 0.45, 1.0],
            PinType::Skeleton => [0.75, 0.55, 0.35, 1.0],
            PinType::PropLayer => [0.2, 0.65, 0.35, 1.0],
            PinType::Biome => [0.85, 0.45, 0.9, 1.0],
            PinType::Zone => [1.0, 0.85, 0.3, 1.0],
            PinType::Layer => [0.45, 0.85, 0.95, 1.0],
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
const BIOME_ZONE_FN: &[GraphKind] = &[GraphKind::Biome, GraphKind::Zone, GraphKind::Function, GraphKind::Prop];
const PROP: &[GraphKind] = &[GraphKind::Prop];
const BIOME: &[GraphKind] = &[GraphKind::Biome];
const ZONE: &[GraphKind] = &[GraphKind::Zone];
const WORLD: &[GraphKind] = &[GraphKind::World];
const FUNC: &[GraphKind] = &[GraphKind::Function];

/// Shape nodes: the block they place and its priority where shapes overlap (higher wins).
const SH: &[(&str, PinType)] = &[("Shape", Shape)];
const ONE_SHAPE: &[(&str, PinType)] = &[("Shape", Shape)];
const LEAF_SHAPES: &[&str] = &["Sphere", "Ellipsoid", "Cone", "Disc", "Fronds"];

pub static NODES: &[NodeDef] = &[
    // ---------------------------------------------------------------- editor
    NodeDef { kind: "comment", title: "Comment", category: "Editor",
        help: "A resizable box with a note. Drag it to move the nodes inside; the generator ignores it.",
        inputs: &[], outputs: &[],
        params: &[f("width", 420.0, 60.0, 8000.0), f("height", 260.0, 40.0, 8000.0)],
        allowed: ALL },
    // ---------------------------------------------------------------- outputs
    NodeDef { kind: "world_output", title: "World Output", category: "Output",
        help: "Root of the generator: the world's layers (surface, underground, sky), each with its own zones and biomes and its own generator, and the cloud sea that hides the sky layer from the ground.",
        inputs: &[("Surface", Layer), ("Underground", Layer), ("Sky", Layer)], outputs: &[],
        params: &[f("cloud_sea_base", 2600.0, 0.0, 8000.0), f("cloud_sea_top", 3400.0, 0.0, 8000.0), f("cloud_sea_density", 1.0, 0.0, 4.0)],
        allowed: WORLD },
    NodeDef { kind: "layer", title: "Layer", category: "Layout",
        help: "A layer of the world between y_min and y_max: its zone layout, sea level and rivers. Surface: air above, solid below; Underground: solid on both sides; Sky: air on both sides. A layer only generates its own range; within `blend` blocks of its edges the terrain fades into what lies beyond (rock or air), so neighbouring layers meet without a seam.",
        inputs: &[("Zones", Zone), ("Rivers", Density)], outputs: &[("Out", Layer)],
        params: &[
            en("kind", &["Surface", "Underground", "Sky"], 0),
            f("y_min", -128.0, -16384.0, 16384.0),
            f("y_max", 1024.0, -16384.0, 16384.0),
            f("blend", 24.0, 0.0, 512.0),
            f("sea_level", 62.0, -16384.0, 16384.0),
            f("river_depth", 5.0, 0.0, 64.0),
            f("river_carve", 48.0, 0.0, 512.0),
            blk("river_bed", "gravel"),
        ],
        allowed: WORLD },
    NodeDef { kind: "depth_bands", title: "Depth Bands", category: "Layout",
        help: "The underground in two bands, each with its own zones: Upper from top down to split, Deep from split down to bottom. Each band has its own sea level (seas of water or lava fill open caves below it; far below the band means none). Caves close within `blend` blocks of a band's edges.",
        inputs: &[("Upper", Zone), ("Deep", Zone)], outputs: &[("Out", Layer)],
        params: &[f("top", -128.0, -16384.0, 16384.0), f("split", -1200.0, -16384.0, 16384.0), f("bottom", -6000.0, -16384.0, 16384.0),
                  f("blend", 24.0, 0.0, 512.0), f("sea_level_upper", -100000.0, -100000.0, 16384.0), f("sea_level_deep", -100000.0, -100000.0, 16384.0)],
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
    NodeDef { kind: "prop_output", title: "Prop Output", category: "Output",
        help: "A prop: its shapes, built once for every placed instance (local space: the anchor is at x = z = 0, y = 0 is the top of the ground). The box (radius around the anchor, height above it, depth below) is what the generator searches and clips to. 'Far' keeps the prop on coarse LODs; 'Everything' lets it overwrite terrain (geodes, ruins) instead of only filling air.",
        inputs: &[("Shape", Shape)], outputs: &[],
        params: &[f("radius", 6.0, 0.5, 64.0), f("height", 12.0, 0.5, 128.0), f("below", 2.0, 0.0, 64.0), b("far", true), en("replace", &["Air only", "Everything"], 0)],
        allowed: PROP },
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
        params: &[f("size", 80.0, 8.0, 1024.0), f("thickness", 0.1, 0.01, 0.5), f("y_min", -256.0, -16384.0, 16384.0), f("y_max", 70.0, -16384.0, 16384.0), i("max_lod", 1, 0, 8), i("seed", 0, 0, 1000)], allowed: BIOME_ZONE_FN },
    NodeDef { kind: "cheese_caves", title: "Cheese Caves", category: "Caves", help: "Large open caverns with pillars.", inputs: &[], outputs: D,
        params: &[f("size", 110.0, 8.0, 1024.0), f("threshold", 0.6, 0.0, 1.0), f("y_min", -256.0, -16384.0, 16384.0), f("y_max", 40.0, -16384.0, 16384.0), i("max_lod", 1, 0, 8), i("seed", 0, 0, 1000)], allowed: BIOME_ZONE_FN },
    NodeDef { kind: "noodle_caves", title: "Noodle Caves", category: "Caves", help: "Thin squiggly tunnels.", inputs: &[], outputs: D,
        params: &[f("size", 32.0, 4.0, 512.0), f("thickness", 0.06, 0.01, 0.3), f("y_min", -256.0, -16384.0, 16384.0), f("y_max", 30.0, -16384.0, 16384.0), i("max_lod", 0, 0, 8), i("seed", 0, 0, 1000)], allowed: BIOME_ZONE_FN },
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
        params: &[blk("ore", "coal_ore"), f("size", 6.0, 1.0, 64.0), f("rarity", 0.8, 0.0, 1.0), f("y_min", -256.0, -16384.0, 16384.0), f("y_max", 128.0, -16384.0, 16384.0), blk("replace", "stone"), i("seed", 0, 0, 1000)], allowed: BIOME },
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

    // ---------------------------------------------------------------- props (biome graphs)
    NodeDef { kind: "prop", title: "Prop", category: "Props", help: "A Prop graph (tree, rock, ruin...), for a Scatter.", inputs: &[], outputs: &[("Out", Prop)],
        params: &[rf("prop", GraphKind::Prop)], allowed: BIOME },
    NodeDef { kind: "prefab", title: "Prefab", category: "Props", help: "Place a saved prefab (Tools > Select > Save selection as prefab).", inputs: &[], outputs: &[("Out", Prop)],
        params: &[ParamDef { name: "prefab", kind: ParamKind::Prefab }, b("random_rotation", true), f("sink", 1.0, 0.0, 16.0)], allowed: BIOME },
    NodeDef { kind: "scatter", title: "Scatter", category: "Props",
        help: "Place a prop on a jittered grid. Surface: on open ground (of the surface block, or any). Buried: inside the ground between y_min and y_max (geodes, fossils). Mask (0..1) modulates density, e.g. noise -> forests and clearings.",
        inputs: &[("Prop", Prop), ("Mask", Density)], outputs: &[("Out", PropLayer)],
        params: &[f("spacing", 9.0, 2.0, 256.0), f("chance", 0.6, 0.0, 1.0), f("jitter", 0.9, 0.0, 1.0), blk("surface", "grass"), b("any_surface", false),
                  f("y_min", -512.0, -16384.0, 16384.0), f("y_max", 512.0, -16384.0, 16384.0), i("seed", 0, 0, 1000), en("placement", &["Surface", "Buried"], 0)], allowed: BIOME },

    // ---------------------------------------------------------------- prop graphs: shapes
    NodeDef { kind: "sphere", title: "Sphere", category: "Shape", help: "A ball. Var: random size variation per instance (0.2 = +-20%).", inputs: &[], outputs: SH,
        params: &[f("x", 0.0, -64.0, 64.0), f("y", 2.0, -64.0, 128.0), f("z", 0.0, -64.0, 64.0), f("radius", 2.0, 0.3, 64.0), f("var", 0.0, 0.0, 0.9), blk("block", "stone"), i("priority", 1, 0, 15)], allowed: PROP },
    NodeDef { kind: "ellipsoid", title: "Ellipsoid", category: "Shape", help: "A stretched ball, turned by yaw (degrees).", inputs: &[], outputs: SH,
        params: &[f("x", 0.0, -64.0, 64.0), f("y", 2.0, -64.0, 128.0), f("z", 0.0, -64.0, 64.0), f("rx", 3.0, 0.3, 64.0), f("ry", 2.0, 0.3, 64.0), f("rz", 3.0, 0.3, 64.0), f("yaw", 0.0, -180.0, 180.0), f("var", 0.0, 0.0, 0.9), blk("block", "stone"), i("priority", 1, 0, 15)], allowed: PROP },
    NodeDef { kind: "capsule", title: "Capsule", category: "Shape", help: "A segment with rounded ends, its radius tapering from r0 to r1 (a cone when r1 is 0).", inputs: &[], outputs: SH,
        params: &[f("x0", 0.0, -64.0, 64.0), f("y0", 0.0, -64.0, 128.0), f("z0", 0.0, -64.0, 64.0), f("x1", 0.0, -64.0, 64.0), f("y1", 6.0, -64.0, 128.0), f("z1", 0.0, -64.0, 64.0),
                  f("r0", 0.6, 0.0, 32.0), f("r1", 0.6, 0.0, 32.0), blk("block", "oak_log"), i("priority", 2, 0, 15)], allowed: PROP },
    NodeDef { kind: "box", title: "Box", category: "Shape", help: "A box of the given size (blocks), centred on x/y/z, turned by yaw.", inputs: &[], outputs: SH,
        params: &[f("x", 0.0, -64.0, 64.0), f("y", 1.5, -64.0, 128.0), f("z", 0.0, -64.0, 64.0), f("sx", 3.0, 0.5, 128.0), f("sy", 3.0, 0.5, 128.0), f("sz", 3.0, 0.5, 128.0), f("yaw", 0.0, -180.0, 180.0), blk("block", "stone_bricks"), i("priority", 1, 0, 15)], allowed: PROP },
    NodeDef { kind: "cylinder", title: "Cylinder", category: "Shape", help: "An upright cylinder (flat ends) from its base, tapering from r0 to r1 (a cone when r1 is 0).", inputs: &[], outputs: SH,
        params: &[f("x", 0.0, -64.0, 64.0), f("y", 0.0, -64.0, 128.0), f("z", 0.0, -64.0, 64.0), f("height", 4.0, 0.5, 128.0), f("r0", 1.5, 0.0, 64.0), f("r1", 1.5, 0.0, 64.0), f("var", 0.0, 0.0, 0.9), blk("block", "stone"), i("priority", 1, 0, 15)], allowed: PROP },
    NodeDef { kind: "blob", title: "Blob", category: "Shape", help: "A lumpy ball: a sphere with noise on its surface (rocks, bushes).", inputs: &[], outputs: SH,
        params: &[f("x", 0.0, -64.0, 64.0), f("y", 1.0, -64.0, 128.0), f("z", 0.0, -64.0, 64.0), f("radius", 2.0, 0.3, 64.0), f("var", 0.25, 0.0, 0.9), f("squash", 1.2, 0.3, 4.0),
                  f("bumps", 0.8, 0.0, 16.0), f("bump_size", 3.0, 0.5, 64.0), blk("block", "cobblestone"), i("priority", 1, 0, 15)], allowed: PROP },

    // ---------------------------------------------------------------- prop graphs: combining
    NodeDef { kind: "union", title: "Union", category: "Shape/Combine", help: "All the shapes together; where they overlap the higher priority wins (the later one on ties).",
        inputs: &[("A", Shape), ("B", Shape), ("C", Shape), ("D", Shape)], outputs: SH, params: &[], allowed: PROP },
    NodeDef { kind: "subtract", title: "Subtract", category: "Shape/Combine", help: "Remove Cut from everything built before it (hollow logs, arches).",
        inputs: &[("Shape", Shape), ("Cut", Shape)], outputs: SH, params: &[], allowed: PROP },
    NodeDef { kind: "intersect", title: "Intersect", category: "Shape/Combine", help: "Keep only what lies inside Mask, of everything built before it.",
        inputs: &[("Shape", Shape), ("Mask", Shape)], outputs: SH, params: &[], allowed: PROP },
    NodeDef { kind: "smooth_union", title: "Smooth Union", category: "Shape/Combine", help: "Melt the shapes into each other with a rounded seam.",
        inputs: &[("A", Shape), ("B", Shape)], outputs: SH, params: &[f("smoothness", 1.5, 0.0, 16.0)], allowed: PROP },
    NodeDef { kind: "displace", title: "Displace", category: "Shape/Combine", help: "Noise on the surface of the shapes (organic, weathered).",
        inputs: ONE_SHAPE, outputs: SH, params: &[f("amount", 0.8, 0.0, 16.0), f("size", 3.0, 0.5, 64.0), i("seed", 0, 0, 1000)], allowed: PROP },
    NodeDef { kind: "holes", title: "Holes", category: "Shape/Combine", help: "Leave out a random share of the cells (sparse leaves, crumbling walls).",
        inputs: ONE_SHAPE, outputs: SH, params: &[f("amount", 0.2, 0.0, 1.0), i("seed", 0, 0, 1000)], allowed: PROP },
    NodeDef { kind: "set_block", title: "Set Block", category: "Shape/Combine", help: "Give the shapes another block (and priority). Block 2 speckles a share of the cells (moss on rocks).",
        inputs: ONE_SHAPE, outputs: SH, params: &[blk("block", "mossy_cobblestone"), i("priority", 1, 0, 15), blk("block2", "air"), f("mix", 0.0, 0.0, 1.0)], allowed: PROP },
    NodeDef { kind: "carve", title: "Carve", category: "Shape/Combine", help: "The shapes cut air into whatever is there (hollow geodes, caves in ruins). Needs 'Everything' in the Prop Output to cut terrain.",
        inputs: ONE_SHAPE, outputs: SH, params: &[], allowed: PROP },
    NodeDef { kind: "transform", title: "Transform", category: "Shape/Combine", help: "Move, turn (degrees) and scale the shapes. Random yaw turns every instance differently; the Scale and Yaw inputs (e.g. a Random) multiply and add.",
        inputs: &[("Shape", Shape), ("Scale", Density), ("Yaw", Density)], outputs: SH,
        params: &[f("x", 0.0, -64.0, 64.0), f("y", 0.0, -64.0, 128.0), f("z", 0.0, -64.0, 64.0), f("yaw", 0.0, -180.0, 180.0), f("pitch", 0.0, -180.0, 180.0), f("roll", 0.0, -180.0, 180.0), f("scale", 1.0, 0.05, 16.0), b("random_yaw", false)], allowed: PROP },
    NodeDef { kind: "mirror", title: "Mirror", category: "Shape/Combine", help: "The shapes and their mirror image.", inputs: ONE_SHAPE, outputs: SH,
        params: &[en("axis", &["X", "Z", "X and Z"], 0)], allowed: PROP },
    NodeDef { kind: "repeat", title: "Repeat", category: "Shape/Combine", help: "Copies of the shapes in a ring (turned to face out) or a line. Each copy gets its own random values; jitter moves copies randomly; each copy is scaled by scale_step more than the last.",
        inputs: ONE_SHAPE, outputs: SH,
        params: &[i("count", 4, 1, 32), en("mode", &["Ring", "Line"], 0), f("radius", 3.0, 0.0, 64.0), f("step_x", 0.0, -32.0, 32.0), f("step_y", 2.0, -32.0, 32.0), f("step_z", 0.0, -32.0, 32.0),
                  f("jitter", 0.0, 0.0, 16.0), f("scale_step", 1.0, 0.3, 2.0), b("random_yaw", false)], allowed: PROP },
    NodeDef { kind: "random", title: "Random", category: "Shape/Value", help: "A random number between min and max, different for every placed instance (and every Repeat copy).",
        inputs: &[], outputs: D, params: &[f("min", 0.8, -1000.0, 1000.0), f("max", 1.2, -1000.0, 1000.0), i("seed", 0, 0, 1000)], allowed: PROP },

    // ---------------------------------------------------------------- prop graphs: trees
    NodeDef { kind: "trunk", title: "Trunk", category: "Tree",
        help: "A trunk from the ground up: random height, radius tapering to the top (taper = top radius / base radius, curve shapes it), a flared base, a bend to one side and a wobble (twist). Sink starts it below the ground (pillars, spikes). Skeleton: for branches, leaves, roots...",
        inputs: &[], outputs: &[("Wood", Shape), ("Skeleton", Skeleton)],
        params: &[f("height_min", 5.0, 0.5, 128.0), f("height_max", 8.0, 0.5, 128.0), f("radius", 0.55, 0.3, 16.0), f("taper", 1.0, 0.0, 2.0), f("taper_curve", 1.0, 0.2, 4.0),
                  f("flare", 0.0, 0.0, 8.0), f("bend", 0.0, 0.0, 32.0), f("twist", 0.0, 0.0, 8.0), f("sink", 0.0, 0.0, 16.0), i("segments", 6, 1, 16),
                  blk("block", "oak_log"), i("priority", 2, 0, 15)], allowed: PROP },
    NodeDef { kind: "branches", title: "Branches", category: "Tree",
        help: "Branches growing from a skeleton between start and end (fractions of each parent's length), count per parent, at an angle from it (degrees), length relative to the parent, droop (blocks down at the tip; negative rises). Levels > 1 branch the branches again (count, length and radius shrink by falloff). Elbow: out level, then straight up or down (cactus arms).",
        inputs: &[("Parent", Skeleton)], outputs: &[("Wood", Shape), ("Skeleton", Skeleton)],
        params: &[i("count", 4, 1, 16), i("levels", 1, 1, 3), f("start", 0.5, 0.0, 1.0), f("end", 0.95, 0.0, 1.0), f("angle", 50.0, 0.0, 180.0), f("length", 0.45, 0.05, 2.0),
                  f("falloff", 0.6, 0.1, 1.0), f("radius", 0.5, 0.3, 8.0), f("droop", 0.0, -32.0, 32.0), b("elbow", false), blk("block", "oak_log"), i("priority", 2, 0, 15)], allowed: PROP },
    NodeDef { kind: "leaves", title: "Leaves", category: "Tree",
        help: "Leaf clusters on a skeleton: at the tips, or along it from start (fraction) every spacing blocks, shrinking towards the top by top_scale. Sphere / Ellipsoid (y_scale) / Cone (conifer layers) / Disc (flat crowns, mushroom caps) / Fronds (palm leaves). Density fills that share of the cells; bumps roughen the edge.",
        inputs: &[("Skeleton", Skeleton)], outputs: SH,
        params: &[en("shape", LEAF_SHAPES, 0), en("at", &["Tips", "Along"], 0), f("radius", 2.5, 0.3, 32.0), f("var", 0.15, 0.0, 0.9), f("y_scale", 0.8, 0.1, 4.0),
                  f("y_offset", 0.0, -16.0, 16.0), f("start", 0.5, 0.0, 1.0), f("spacing", 2.0, 0.5, 16.0), f("top_scale", 0.3, 0.0, 2.0), f("density", 0.85, 0.05, 1.0),
                  f("bumps", 0.6, 0.0, 8.0), blk("block", "oak_leaves"), i("priority", 1, 0, 15)], allowed: PROP },
    NodeDef { kind: "roots", title: "Roots", category: "Tree", help: "Roots spreading from the base of a skeleton, sloping down into the ground.",
        inputs: &[("Skeleton", Skeleton)], outputs: SH,
        params: &[i("count", 4, 1, 12), f("length", 3.0, 0.5, 32.0), f("depth", 1.5, 0.0, 16.0), f("radius", 0.5, 0.3, 4.0), blk("block", "oak_log"), i("priority", 2, 0, 15)], allowed: PROP },
    NodeDef { kind: "vines", title: "Vines", category: "Tree", help: "Hanging strands around a skeleton's tips, at distance spread (willows, jungle trees).",
        inputs: &[("Skeleton", Skeleton)], outputs: SH,
        params: &[i("count", 8, 1, 32), f("spread", 3.0, 0.0, 32.0), f("length_min", 2.0, 0.5, 32.0), f("length_max", 5.0, 0.5, 64.0), f("y_offset", 0.0, -16.0, 16.0), blk("block", "oak_leaves"), i("priority", 1, 0, 15)], allowed: PROP },
    NodeDef { kind: "fruit", title: "Fruit", category: "Tree", help: "Single blocks (fruit, flowers, lanterns) near a skeleton's tips: chance per tip, at distance spread below them.",
        inputs: &[("Skeleton", Skeleton)], outputs: SH,
        params: &[f("chance", 0.5, 0.0, 1.0), i("per_tip", 2, 1, 8), f("spread", 2.0, 0.0, 16.0), blk("block", "flower_red"), i("priority", 3, 0, 15)], allowed: PROP },

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
