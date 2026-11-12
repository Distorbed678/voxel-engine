//! Node graph -> GLSL compiler.
//!
//! * Every Float/Int parameter becomes `P(i)` (a slot in a GPU buffer). Changing a value never
//!   changes the generated source, so slider drags only re-upload parameters.
//! * Values that only depend on X/Z ("2D") are hoisted into a per-column function and cached in
//!   `WgColumn.h[k][...]`, so e.g. a 6-octave height map is evaluated once per column instead of
//!   once per voxel.
//! * Domain transforms (warp/stretch/offset) create evaluation contexts; upstream nodes are
//!   re-emitted per context with memoisation.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write as _;

use anyhow::Result;

use crate::graph::{Graph, GraphKind, Node, Project, Value};
use crate::nodes::{self, ParamKind};

#[derive(Clone, Debug)]
pub struct BiomeInfo {
    pub name: String,
    pub grass_tint: [f32; 3],
    pub foliage_tint: [f32; 3],
    pub water_tint: [f32; 3],
    pub fog_color: [f32; 3],
    pub fog_density: f32,
    pub map_color: [f32; 3],
    /// Colour grading while the camera is in this biome.
    pub grade_tint: [f32; 3],
    pub saturation: f32,
    pub contrast: f32,
    /// Ambient particles (None, Ash, Embers, Fireflies, Pollen), precipitation (None, Rain, Snow).
    pub ambient: u32,
    pub precipitation: u32,
    pub particle_density: f32,
}

#[derive(Clone, Debug)]
pub struct ZoneInfo {
    pub name: String,
    pub map_color: [f32; 3],
}

/// Location of a live parameter in the GPU parameter buffer.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamKey {
    pub graph: (GraphKind, String),
    pub node: u32,
    pub param: String,
}

#[derive(Clone, Debug, Default)]
pub struct Compiled {
    pub source: String,
    pub params: Vec<f32>,
    pub param_index: BTreeMap<ParamKey, u32>,
    pub biomes: Vec<BiomeInfo>,
    pub zones: Vec<ZoneInfo>,
    pub prop_layers: usize,
    pub warnings: Vec<String>,
}

#[derive(Default)]
struct Params {
    values: Vec<f32>,
    index: BTreeMap<ParamKey, u32>,
}

impl Params {
    fn push(&mut self, key: ParamKey, v: f32) -> u32 {
        if let Some(&i) = self.index.get(&key) {
            self.values[i as usize] = v;
            return i;
        }
        let i = self.values.len() as u32;
        self.values.push(v);
        self.index.insert(key, i);
        i
    }
}

#[derive(Clone, Debug)]
struct Val {
    e: String,
    two_d: bool,
    /// width of the value for exports: 1 = float, 3 = vec3
    width: u8,
}

#[derive(Clone, Debug)]
struct Ctx {
    /// scale expression (None = 1)
    s: Option<String>,
    /// offset value (None = 0)
    o: Option<Val>,
}

struct Shared<'a> {
    proj: &'a Project,
    blocks: &'a HashMap<String, u32>,
    prefabs: &'a HashMap<String, (u32, [i32; 3])>,
    params: Params,
    warnings: Vec<String>,
    biome_ids: HashMap<String, u32>,
    zone_ids: HashMap<String, u32>,
    /// depth thresholds used by material nodes (GLSL expressions), per biome (None: cave
    /// materials, used anywhere): the terrain pass needs to know how deep below the surface it
    /// must look to decide every material consistently
    depth_terms: Vec<(Option<u32>, String)>,
    /// the biome whose surface materials are being compiled
    depth_biome: Option<u32>,
}

struct Em<'s, 'a> {
    sh: &'s mut Shared<'a>,
    g: &'a Graph,
    pfx: String,
    pre: Vec<String>,
    body: Vec<String>,
    memo2: HashMap<(u32, u8, usize), Val>,
    memo3: HashMap<(u32, u8, usize), Val>,
    ctxs: Vec<Ctx>,
    exports: Vec<Val>,
    export_set: HashSet<String>,
    n: usize,
    /// No hoisting (function graphs, zone carvers): everything goes into `body`.
    flat: bool,
    depth: usize,
}

impl<'s, 'a> Em<'s, 'a> {
    fn new(sh: &'s mut Shared<'a>, g: &'a Graph, pfx: String, flat: bool) -> Self {
        Self {
            sh,
            g,
            pfx,
            pre: Vec::new(),
            body: Vec::new(),
            memo2: HashMap::new(),
            memo3: HashMap::new(),
            ctxs: vec![Ctx { s: None, o: None }],
            exports: Vec::new(),
            export_set: HashSet::new(),
            n: 0,
            flat,
            depth: 0,
        }
    }

    fn key(&self, node: u32, param: &str) -> ParamKey {
        ParamKey { graph: (self.g.kind, self.g.name.clone()), node, param: param.to_string() }
    }

    fn warn(&mut self, msg: String) {
        let m = format!("{} '{}': {}", self.g.kind.label(), self.g.name, msg);
        if !self.sh.warnings.contains(&m) {
            self.sh.warnings.push(m);
        }
    }

    fn value(&self, node: &Node, name: &str) -> Value {
        if let Some(v) = node.params.get(name) {
            return v.clone();
        }
        nodes::def(&node.kind)
            .and_then(|d| d.params.iter().find(|p| p.name == name))
            .map(|p| p.kind.default_value())
            .unwrap_or(Value::F(0.0))
    }

    /// Live float parameter -> `P(i)`
    fn p(&mut self, node: &Node, name: &str) -> String {
        let v = self.value(node, name).as_f32();
        let k = self.key(node.id, name);
        let i = self.sh.params.push(k, v);
        format!("P({i}u)")
    }

    fn pu(&mut self, node: &Node, name: &str) -> String {
        format!("uint({})", self.p(node, name))
    }

    fn block(&mut self, node: &Node, name: &str) -> String {
        let b = match self.value(node, name) {
            Value::Block(b) => b,
            _ => String::new(),
        };
        match self.sh.blocks.get(&b) {
            Some(id) => format!("{id}u"),
            None => {
                if !b.is_empty() {
                    self.warn(format!("unknown block '{b}' on node {}", node.id));
                }
                "0u".into()
            }
        }
    }

    fn enum_idx(&self, node: &Node, name: &str) -> usize {
        let Value::E(s) = self.value(node, name) else { return 0 };
        nodes::def(&node.kind)
            .and_then(|d| d.params.iter().find(|p| p.name == name))
            .and_then(|p| match p.kind {
                ParamKind::Enum(opts, _) => opts.iter().position(|o| *o == s),
                _ => None,
            })
            .unwrap_or(0)
    }

    fn boolean(&self, node: &Node, name: &str) -> bool {
        matches!(self.value(node, name), Value::B(true))
    }

    fn var(&mut self, ty: &str, expr: String, two_d: bool) -> Val {
        self.n += 1;
        let name = format!("{}v{}", self.pfx, self.n);
        let stmt = format!("    {ty} {name} = {expr};");
        if two_d && !self.flat {
            self.pre.push(stmt);
        } else {
            self.body.push(stmt);
        }
        Val { e: name, two_d: two_d && !self.flat, width: if ty == "vec3" { 3 } else { 1 } }
    }

    /// A 3D statement consumes `v`: if it lives in the 2D prologue it must be exported.
    fn need(&mut self, v: &Val) {
        if v.two_d && !self.flat && v.e.starts_with(&self.pfx) && self.export_set.insert(v.e.clone()) {
            self.exports.push(v.clone());
        }
    }

    fn pos(&self, ctx: usize) -> String {
        let c = &self.ctxs[ctx];
        match (&c.s, &c.o) {
            (None, None) => "p".into(),
            (Some(s), None) => format!("(p * {s})"),
            (None, Some(o)) => format!("(p + {})", o.e),
            (Some(s), Some(o)) => format!("(p * {s} + {})", o.e),
        }
    }

    fn ctx_2d(&self, ctx: usize) -> bool {
        self.ctxs[ctx].o.as_ref().map(|o| o.two_d).unwrap_or(true)
    }

    fn input(&mut self, node: &Node, pin: u8, ctx: usize) -> Option<Val> {
        let l = *self.g.input_link(node.id, pin)?;
        Some(self.density(l.from_node, l.from_pin, ctx))
    }

    fn input_or(&mut self, node: &Node, pin: u8, ctx: usize, default: &str) -> Val {
        self.input(node, pin, ctx).unwrap_or(Val { e: default.into(), two_d: true, width: 1 })
    }

    /// Emit a scalar node result, choosing the 2D prologue when every input is 2D.
    fn emit(&mut self, expr: String, inputs: &[&Val], intrinsic_3d: bool, ctx: usize) -> Val {
        let two_d = !intrinsic_3d && self.ctx_2d(ctx) && inputs.iter().all(|v| v.two_d);
        if !two_d {
            for v in inputs {
                self.need(v);
            }
            if let Some(o) = self.ctxs[ctx].o.clone() {
                self.need(&o);
            }
        }
        self.var("float", expr, two_d)
    }

    fn noise_expr(&mut self, n: &Node, func: &str, dims3: bool, ctx: usize) -> String {
        let pos = self.pos(ctx);
        let size = self.p(n, "size");
        let oct = self.p(n, "octaves");
        let lac = self.p(n, "lacunarity");
        let gain = self.p(n, "gain");
        let amp = self.p(n, "amplitude");
        let seed = self.pu(n, "seed");
        let arg = if dims3 { format!("{pos} / {size}") } else { format!("{pos}.xz / {size}") };
        format!("{func}({arg}, gSeed + {seed} * 7919u + {}u, int({oct}), {lac}, {gain}) * {amp}", n.id)
    }

    fn density(&mut self, id: u32, pin: u8, ctx: usize) -> Val {
        if let Some(v) = self.memo2.get(&(id, pin, ctx)).or_else(|| self.memo3.get(&(id, pin, ctx))) {
            return v.clone();
        }
        self.depth += 1;
        if self.depth > 256 {
            self.warn("graph too deep".into());
            return Val { e: "0.0".into(), two_d: true, width: 1 };
        }
        let Some(node) = self.g.node(id) else {
            return Val { e: "0.0".into(), two_d: true, width: 1 };
        };
        let n = node.clone();
        let pos = self.pos(ctx);
        let v = match n.kind.as_str() {
            "constant" => {
                let p = self.p(&n, "value");
                Val { e: p, two_d: true, width: 1 }
            }
            "coord" => {
                let axis = self.enum_idx(&n, "axis");
                let c = ["x", "y", "z"][axis];
                self.emit(format!("{pos}.{c}"), &[], axis == 1, ctx)
            }
            "simplex2d" => {
                let e = self.noise_expr(&n, "fbm2", false, ctx);
                self.emit(e, &[], false, ctx)
            }
            "simplex3d" => {
                let e = self.noise_expr(&n, "fbm3", true, ctx);
                self.emit(e, &[], true, ctx)
            }
            "ridged2d" => {
                let e = self.noise_expr(&n, "ridged2", false, ctx);
                self.emit(e, &[], false, ctx)
            }
            "ridged3d" => {
                let e = self.noise_expr(&n, "ridged3", true, ctx);
                self.emit(e, &[], true, ctx)
            }
            "billow2d" => {
                let e = self.noise_expr(&n, "billow2", false, ctx);
                self.emit(e, &[], false, ctx)
            }
            "billow3d" => {
                let e = self.noise_expr(&n, "billow3", true, ctx);
                self.emit(e, &[], true, ctx)
            }
            "worley2d" | "worley3d" => {
                let three = n.kind == "worley3d";
                let size = self.p(&n, "size");
                let jit = self.p(&n, "jitter");
                let amp = self.p(&n, "amplitude");
                let seed = self.pu(&n, "seed");
                let mode = ["w.x", "w.y", "(w.y - w.x)", "w.z"][self.enum_idx(&n, "mode")];
                let call = if three {
                    format!("worley3({pos} / {size}, gSeed + {seed} * 7919u + {}u, {jit})", n.id)
                } else {
                    format!("worley2({pos}.xz / {size}, gSeed + {seed} * 7919u + {}u, {jit})", n.id)
                };
                let w = self.var("vec3", call, !three && self.ctx_2d(ctx));
                if three
                    && let Some(o) = self.ctxs[ctx].o.clone() {
                        self.need(&o);
                    }
                let we = w.e.clone();
                let w2 = Val { e: we.clone(), two_d: w.two_d, width: 3 };
                self.emit(format!("{} * {amp}", mode.replace('w', &we)), &[&w2], three, ctx)
            }
            "distance" => {
                let x = self.p(&n, "x");
                let y = self.p(&n, "y");
                let z = self.p(&n, "z");
                if self.enum_idx(&n, "mode") == 0 {
                    self.emit(format!("distance({pos}.xz, vec2({x}, {z}))"), &[], false, ctx)
                } else {
                    self.emit(format!("distance({pos}, vec3({x}, {y}, {z}))"), &[], true, ctx)
                }
            }
            "river" => {
                let (sp, w, bank, warp) = (self.p(&n, "spacing"), self.p(&n, "width"), self.p(&n, "bank"), self.p(&n, "warp"));
                let seed = self.pu(&n, "seed");
                let e = format!("river_mask({pos}.xz, {sp}, {w}, {bank}, {warp}, gSeed + {seed} * 7919u + {}u)", n.id);
                self.emit(e, &[], false, ctx)
            }
            "y_gradient" => {
                let (yl, yh, vl, vh) = (self.p(&n, "y_low"), self.p(&n, "y_high"), self.p(&n, "value_low"), self.p(&n, "value_high"));
                self.emit(format!("mix({vl}, {vh}, clamp(({pos}.y - {yl}) / max({yh} - {yl}, 1e-3), 0.0, 1.0))"), &[], true, ctx)
            }
            "height_terrain" => {
                let h = self.input_or(&n, 0, ctx, "0.0");
                let (base, scale, fall) = (self.p(&n, "base"), self.p(&n, "scale"), self.p(&n, "falloff"));
                self.emit(format!("(({base} + {} * {scale}) - {pos}.y) * {fall}", h.e), &[&h], true, ctx)
            }
            "add" | "subtract" | "multiply" | "min" | "max" | "smooth_min" | "smooth_max" => {
                let a = self.input_or(&n, 0, ctx, "0.0");
                let b = self.input_or(&n, 1, ctx, if n.kind == "multiply" { "1.0" } else { "0.0" });
                let e = match n.kind.as_str() {
                    "add" => format!("{} + {}", a.e, b.e),
                    "subtract" => format!("{} - {}", a.e, b.e),
                    "multiply" => format!("{} * {}", a.e, b.e),
                    "min" => format!("min({}, {})", a.e, b.e),
                    "max" => format!("max({}, {})", a.e, b.e),
                    "smooth_min" => {
                        let k = self.p(&n, "smoothness");
                        format!("smoothmin({}, {}, {k})", a.e, b.e)
                    }
                    _ => {
                        let k = self.p(&n, "smoothness");
                        format!("smoothmax({}, {}, {k})", a.e, b.e)
                    }
                };
                self.emit(e, &[&a, &b], false, ctx)
            }
            "abs" | "negate" | "square" | "pow" | "clamp" | "remap" | "scale_bias" | "curve" | "terrace" => {
                let x = self.input_or(&n, 0, ctx, "0.0");
                let e = match n.kind.as_str() {
                    "abs" => format!("abs({})", x.e),
                    "negate" => format!("-({})", x.e),
                    "square" => {
                        if self.boolean(&n, "signed") { format!("{0} * abs({0})", x.e) } else { format!("{0} * {0}", x.e) }
                    }
                    "pow" => {
                        let ex = self.p(&n, "exponent");
                        format!("sign({0}) * pow(abs({0}), {ex})", x.e)
                    }
                    "clamp" => {
                        let (a, b) = (self.p(&n, "min"), self.p(&n, "max"));
                        format!("clamp({}, {a}, {b})", x.e)
                    }
                    "remap" => {
                        let (a, b, c, d) = (self.p(&n, "in_min"), self.p(&n, "in_max"), self.p(&n, "out_min"), self.p(&n, "out_max"));
                        let t = format!("(({} - {a}) / ({b} - {a}))", x.e);
                        let t = if self.boolean(&n, "clamp") { format!("clamp({t}, 0.0, 1.0)") } else { t };
                        format!("mix({c}, {d}, {t})")
                    }
                    "scale_bias" => {
                        let (s, b) = (self.p(&n, "scale"), self.p(&n, "bias"));
                        format!("{} * {s} + {b}", x.e)
                    }
                    "curve" => {
                        let pts = match self.value(&n, "curve") {
                            Value::Curve(p) if !p.is_empty() => p,
                            _ => vec![[-1.0, -1.0], [1.0, 1.0]],
                        };
                        let mut base = 0;
                        for (i, pt) in pts.iter().enumerate() {
                            let kx = self.key(n.id, &format!("curve.{i}.x"));
                            let ky = self.key(n.id, &format!("curve.{i}.y"));
                            let ix = self.sh.params.push(kx, pt[0]);
                            self.sh.params.push(ky, pt[1]);
                            if i == 0 {
                                base = ix;
                            }
                        }
                        format!("curve_eval({base}u, {}u, {})", pts.len(), x.e)
                    }
                    _ => {
                        let (st, sm) = (self.p(&n, "step"), self.p(&n, "smoothness"));
                        format!("terrace({}, {st}, {sm})", x.e)
                    }
                };
                self.emit(e, &[&x], false, ctx)
            }
            "lerp" => {
                let a = self.input_or(&n, 0, ctx, "0.0");
                let b = self.input_or(&n, 1, ctx, "0.0");
                let t = self.input_or(&n, 2, ctx, "0.5");
                self.emit(format!("mix({}, {}, clamp({}, 0.0, 1.0))", a.e, b.e, t.e), &[&a, &b, &t], false, ctx)
            }
            "select" => {
                let a = self.input_or(&n, 0, ctx, "0.0");
                let b = self.input_or(&n, 1, ctx, "0.0");
                let s = self.input_or(&n, 2, ctx, "0.0");
                let (th, bl) = (self.p(&n, "threshold"), self.p(&n, "blend"));
                self.emit(
                    format!("mix({}, {}, smoothstep({th} - {bl}, {th} + {bl} + 1e-5, {}))", a.e, b.e, s.e),
                    &[&a, &b, &s],
                    false,
                    ctx,
                )
            }
            "lod_switch" => {
                let a = self.input_or(&n, 0, ctx, "0.0");
                let b = self.input_or(&n, 1, ctx, "0.0");
                let m = self.pu(&n, "min_lod");
                self.emit(format!("(gLod >= {m} ? {} : {})", b.e, a.e), &[&a, &b], false, ctx)
            }
            "warp" | "scale_coords" | "translate" => {
                let parent = self.ctxs[ctx].clone();
                let new_ctx = match n.kind.as_str() {
                    "warp" => {
                        let wx = self.input_or(&n, 1, ctx, "0.0");
                        let wz = self.input_or(&n, 2, ctx, "0.0");
                        let wy = self.input_or(&n, 3, ctx, "0.0");
                        let st = self.p(&n, "strength");
                        let two = wx.two_d && wy.two_d && wz.two_d && parent.o.as_ref().map(|o| o.two_d).unwrap_or(true);
                        let base = parent.o.as_ref().map(|o| format!("{} + ", o.e)).unwrap_or_default();
                        if !two {
                            for v in [&wx, &wy, &wz] {
                                self.need(v);
                            }
                            if let Some(o) = &parent.o {
                                self.need(&o.clone());
                            }
                        }
                        let o = self.var("vec3", format!("{base}vec3({}, {}, {}) * {st}", wx.e, wy.e, wz.e), two);
                        Ctx { s: parent.s.clone(), o: Some(o) }
                    }
                    "scale_coords" => {
                        let (x, y, z) = (self.p(&n, "x"), self.p(&n, "y"), self.p(&n, "z"));
                        let sv = format!("vec3({x}, {y}, {z})");
                        let s = Some(match &parent.s {
                            Some(ps) => format!("({ps} * {sv})"),
                            None => sv.clone(),
                        });
                        let o = parent.o.as_ref().map(|o| {
                            let two = o.two_d;
                            if !two {
                                self.need(&o.clone());
                            }
                            self.var("vec3", format!("{} * {sv}", o.e), two)
                        });
                        Ctx { s, o }
                    }
                    _ => {
                        let (x, y, z) = (self.p(&n, "x"), self.p(&n, "y"), self.p(&n, "z"));
                        let base = parent.o.as_ref().map(|o| format!("{} + ", o.e)).unwrap_or_default();
                        let two = parent.o.as_ref().map(|o| o.two_d).unwrap_or(true);
                        let o = self.var("vec3", format!("{base}vec3({x}, {y}, {z})"), two);
                        Ctx { s: parent.s.clone(), o: Some(o) }
                    }
                };
                self.ctxs.push(new_ctx);
                let c = self.ctxs.len() - 1;
                self.input_or(&n, 0, c, "0.0")
            }
            "spaghetti_caves" | "cheese_caves" | "noodle_caves" | "ravines" => {
                let ml = self.pu(&n, "max_lod");
                let seed = self.pu(&n, "seed");
                let s = format!("gSeed + {seed} * 7919u + {}u", n.id);
                let call = match n.kind.as_str() {
                    "spaghetti_caves" => {
                        let (a, b, c, d) = (self.p(&n, "size"), self.p(&n, "thickness"), self.p(&n, "y_min"), self.p(&n, "y_max"));
                        format!("cave_spaghetti({pos}, {a}, {b}, {c}, {d}, {s})")
                    }
                    "cheese_caves" => {
                        let (a, b, c, d) = (self.p(&n, "size"), self.p(&n, "threshold"), self.p(&n, "y_min"), self.p(&n, "y_max"));
                        format!("cave_cheese({pos}, {a}, {b}, {c}, {d}, {s})")
                    }
                    "noodle_caves" => {
                        let (a, b, c, d) = (self.p(&n, "size"), self.p(&n, "thickness"), self.p(&n, "y_min"), self.p(&n, "y_max"));
                        format!("cave_spaghetti({pos}, {a}, {b}, {c}, {d}, {s})")
                    }
                    _ => {
                        let (a, b, c, d) = (self.p(&n, "size"), self.p(&n, "width"), self.p(&n, "y_top"), self.p(&n, "depth"));
                        format!("cave_ravine({pos}, {a}, {b}, {c}, {d}, {s})")
                    }
                };
                self.emit(format!("(gLod > {ml} ? 0.0 : {call})"), &[], true, ctx)
            }
            "function_input" => {
                let idx = self.value(&n, "index").as_f32().clamp(0.0, 3.0) as usize;
                if self.g.kind != GraphKind::Function {
                    self.warn("Function Input outside a function graph".into());
                    Val { e: "0.0".into(), two_d: true, width: 1 }
                } else {
                    Val { e: format!("a{idx}"), two_d: false, width: 1 }
                }
            }
            "function_call" => {
                let name = match self.value(&n, "function") {
                    Value::Ref(r) => r,
                    _ => String::new(),
                };
                if self.sh.proj.get(GraphKind::Function, &name).is_none() {
                    self.warn(format!("function '{name}' not found"));
                    Val { e: "0.0".into(), two_d: true, width: 1 }
                } else {
                    let args: Vec<Val> = (0..4).map(|i| self.input_or(&n, i, ctx, "0.0")).collect();
                    let refs: Vec<&Val> = args.iter().collect();
                    let call = format!("fn_{}({pos}, {}, {}, {}, {})", sanitize(&name), args[0].e, args[1].e, args[2].e, args[3].e);
                    self.emit(call, &refs, true, ctx)
                }
            }
            other => {
                self.warn(format!("node '{other}' cannot produce a density"));
                Val { e: "0.0".into(), two_d: true, width: 1 }
            }
        };
        self.depth -= 1;
        if v.two_d && !self.flat {
            self.memo2.insert((id, pin, ctx), v.clone());
        } else {
            self.memo3.insert((id, pin, ctx), v.clone());
        }
        v
    }

    // ---- materials: available vars p, depth, airAbove, skyOpen ----
    fn material(&mut self, id: u32) -> String {
        let Some(node) = self.g.node(id) else { return "B_STONE".into() };
        let n = node.clone();
        let mat_in = |em: &mut Self, pin: u8, def: &str| -> String {
            match em.g.input_link(n.id, pin) {
                Some(l) => {
                    let l = *l;
                    em.material(l.from_node)
                }
                None => def.to_string(),
            }
        };
        let e = match n.kind.as_str() {
            "block" => self.block(&n, "block"),
            "surface_layers" => {
                let base_blk = self.block(&n, "base");
                let base = mat_in(self, 0, &base_blk);
                let top = self.block(&n, "top");
                let under = self.block(&n, "under");
                let uw = self.block(&n, "underwater");
                let cf = self.block(&n, "cave_floor");
                let (td, ud, beach) = (self.p(&n, "top_depth"), self.p(&n, "under_depth"), self.p(&n, "beach_height"));
                self.sh.depth_terms.push((self.sh.depth_biome, format!("{td} + {ud}")));
                format!(
                    "(!skyOpen ? ((airAbove > 0.0 && depth < 1.0) ? {cf} : {base}) : (depth < {td} ? (p.y < gSeaLevel + {beach} ? {uw} : {top}) : (depth < {td} + {ud} ? (p.y < gSeaLevel + {beach} ? {uw} : {under}) : {base})))"
                )
            }
            "height_select" => {
                let below = mat_in(self, 0, "B_STONE");
                let above = mat_in(self, 1, "B_SNOW");
                let noise = self.input(&n, 2, 0).map(|v| {
                    self.need(&v);
                    v.e
                }).unwrap_or_else(|| "0.0".into());
                let (h, amt) = (self.p(&n, "height"), self.p(&n, "noise_amount"));
                format!("(p.y < {h} + {noise} * {amt} ? {below} : {above})")
            }
            "depth_select" => {
                let a = mat_in(self, 0, "B_DIRT");
                let b = mat_in(self, 1, "B_STONE");
                let d = self.p(&n, "depth");
                self.sh.depth_terms.push((self.sh.depth_biome, d.clone()));
                format!("(depth < {d} ? {a} : {b})")
            }
            "noise_select" => {
                let a = mat_in(self, 0, "B_STONE");
                let b = mat_in(self, 1, "B_DIRT");
                let s = self.input(&n, 2, 0).map(|v| {
                    self.need(&v);
                    v.e
                }).unwrap_or_else(|| "0.0".into());
                let th = self.p(&n, "threshold");
                format!("({s} > {th} ? {b} : {a})")
            }
            "ore" => {
                let base = mat_in(self, 0, "B_STONE");
                let ore = self.block(&n, "ore");
                let rep = self.block(&n, "replace");
                let (sz, r, y0, y1) = (self.p(&n, "size"), self.p(&n, "rarity"), self.p(&n, "y_min"), self.p(&n, "y_max"));
                let seed = self.pu(&n, "seed");
                let b = self.var("uint", base, false);
                format!(
                    "(({0} == {rep} && p.y >= {y0} && p.y <= {y1} && simplex3(p / {sz}, gSeed + {seed} * 7919u + {1}u) > {r}) ? {ore} : {0})",
                    b.e, n.id
                )
            }
            "strata" => {
                let base = mat_in(self, 0, "B_STONE");
                let bands: Vec<String> = (1..=6).map(|i| self.block(&n, &format!("band{i}"))).collect();
                let rep = self.block(&n, "replace");
                let (bh, wob, md) = (self.p(&n, "band_height"), self.p(&n, "wobble"), self.p(&n, "min_depth"));
                self.sh.depth_terms.push((self.sh.depth_biome, md.clone()));
                let b = self.var("uint", base, false);
                format!(
                    "(({0} == {rep} && depth >= {md}) ? strata_pick(p, {bh}, {wob}, gSeed + {1}u, {2}, {3}, {4}, {5}, {6}, {7}) : {0})",
                    b.e, n.id, bands[0], bands[1], bands[2], bands[3], bands[4], bands[5]
                )
            }
            "cave_select" => {
                let a = mat_in(self, 0, "B_GRASS");
                let b = mat_in(self, 1, "B_STONE");
                format!("(skyOpen ? {a} : {b})")
            }
            other => {
                self.warn(format!("node '{other}' cannot produce a material"));
                "B_STONE".into()
            }
        };
        self.var("uint", e, false).e
    }

    // ---- decorations: vars p (air cell above ground), ground ----
    fn decor(&mut self, id: u32) -> String {
        let Some(node) = self.g.node(id) else { return "0u".into() };
        let n = node.clone();
        let e = match n.kind.as_str() {
            "ground_cover" => {
                let on = self.block(&n, "on");
                let ch = self.p(&n, "chance");
                let mask = self.input(&n, 0, 0).map(|v| {
                    self.need(&v);
                    format!("clamp({}, 0.0, 1.0)", v.e)
                }).unwrap_or_else(|| "1.0".into());
                let mut ws = Vec::new();
                let mut bs = Vec::new();
                for i in 1..=4 {
                    ws.push(self.p(&n, &format!("weight{i}")));
                    bs.push(self.block(&n, &format!("plant{i}")));
                }
                let seed = self.pu(&n, "seed");
                format!(
                    "ground_cover(p, ground, {on}, {ch} * {mask}, vec4({}, {}, {}, {}), uvec4({}, {}, {}, {}), gSeed + {seed} * 7919u + {}u)",
                    ws[0], ws[1], ws[2], ws[3], bs[0], bs[1], bs[2], bs[3], n.id
                )
            }
            "decor_mix" => {
                let a = self.g.input_link(n.id, 0).copied().map(|l| self.decor(l.from_node)).unwrap_or_else(|| "0u".into());
                let b = self.g.input_link(n.id, 1).copied().map(|l| self.decor(l.from_node)).unwrap_or_else(|| "0u".into());
                format!("({a} != 0u ? {a} : {b})")
            }
            other => {
                self.warn(format!("node '{other}' cannot produce a decoration"));
                "0u".into()
            }
        };
        self.var("uint", e, false).e
    }

    fn take_body(&mut self) -> Vec<String> {
        self.memo3.clear();
        std::mem::take(&mut self.body)
    }
}

pub fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect()
}

struct PropLayerGen {
    biome: u32,
    desc: String,
    mask_fn: Option<String>,
}

fn color_param(g: &Graph, n: &Node, name: &str) -> [f32; 3] {
    let _ = g;
    match n.params.get(name) {
        Some(Value::Color(c)) => *c,
        _ => match nodes::def(&n.kind).and_then(|d| d.params.iter().find(|p| p.name == name)).map(|p| p.kind) {
            Some(ParamKind::Color(c)) => c,
            _ => [1.0, 0.0, 1.0],
        },
    }
}

/// Compile the whole project. `blocks` maps block names to ids, `prefabs` prefab names to
/// (GPU index, size).
pub fn compile(project: &Project, blocks: &HashMap<String, u32>, prefabs: &HashMap<String, (u32, [i32; 3])>) -> Result<Compiled> {
    let mut sh = Shared {
        proj: project,
        blocks,
        prefabs,
        params: Params::default(),
        warnings: Vec::new(),
        biome_ids: HashMap::new(),
        zone_ids: HashMap::new(),
        depth_terms: Vec::new(),
        depth_biome: None,
    };
    let biome_names = project.names(GraphKind::Biome);
    let zone_names = project.names(GraphKind::Zone);
    for (i, n) in biome_names.iter().enumerate() {
        sh.biome_ids.insert(n.clone(), i as u32);
    }
    for (i, n) in zone_names.iter().enumerate() {
        sh.zone_ids.insert(n.clone(), i as u32);
    }
    if biome_names.is_empty() || zone_names.is_empty() {
        anyhow::bail!("the project needs at least one zone and one biome");
    }
    let mut out = String::new();
    let mut fns = String::new();

    // ---- function graphs ----
    let fn_names = project.names(GraphKind::Function);
    for name in &fn_names {
        let _ = writeln!(out, "float fn_{}(vec3 p, float a0, float a1, float a2, float a3);", sanitize(name));
    }
    for name in &fn_names {
        let g = project.get(GraphKind::Function, name).unwrap();
        let mut em = Em::new(&mut sh, g, format!("f{}_", sanitize(name)), true);
        let res = g
            .output_node()
            .and_then(|o| g.input_link(o.id, 0).copied())
            .map(|l| em.density(l.from_node, l.from_pin, 0).e)
            .unwrap_or_else(|| "0.0".into());
        let body = em.take_body().join("\n");
        let _ = writeln!(fns, "float fn_{}(vec3 p, float a0, float a1, float a2, float a3) {{\n{body}\n    return {res};\n}}\n", sanitize(name));
    }

    // ---- biomes used as cave biomes (Zone Output "Cave Biomes"): they also get flat
    // material / decoration functions, evaluated per voxel on cave floors and ceilings ----
    let mut cave_biome_ids: std::collections::BTreeSet<u32> = Default::default();
    for name in &zone_names {
        let g = project.get(GraphKind::Zone, name).unwrap();
        let Some(o) = g.output_node() else { continue };
        let Some(l) = g.input_link(o.id, 2) else { continue };
        let Some(n) = g.node(l.from_node) else { continue };
        let refs: Vec<&Node> = match n.kind.as_str() {
            "biome" => vec![n],
            "cave_biomes" => (0..4u8).filter_map(|i| g.input_link(n.id, i)).filter_map(|l| g.node(l.from_node)).collect(),
            _ => Vec::new(),
        };
        for bn in refs {
            if let Some(Value::Ref(r)) = bn.params.get("biome")
                && let Some(&id) = sh.biome_ids.get(r)
            {
                cave_biome_ids.insert(id);
            }
        }
    }

    // ---- biomes ----
    let mut biome_infos = Vec::new();
    let mut maxh = 1usize;
    let mut biome_code = String::new();
    let mut prop_layers: Vec<PropLayerGen> = Vec::new();
    let mut fluids = Vec::new();
    for (bi, name) in biome_names.iter().enumerate() {
        let g = project.get(GraphKind::Biome, name).unwrap();
        let pfx = format!("b{bi}_");
        let Some(outn) = g.output_node().cloned() else {
            sh.warnings.push(format!("Biome '{name}' has no output node"));
            continue;
        };
        biome_infos.push(BiomeInfo {
            name: name.clone(),
            grass_tint: color_param(g, &outn, "grass_tint"),
            foliage_tint: color_param(g, &outn, "foliage_tint"),
            water_tint: color_param(g, &outn, "water_tint"),
            fog_color: color_param(g, &outn, "fog_color"),
            fog_density: outn.params.get("fog_density").map(|v| v.as_f32()).unwrap_or(1.0),
            map_color: color_param(g, &outn, "map_color"),
            grade_tint: color_param(g, &outn, "grade_tint"),
            saturation: outn.params.get("saturation").map(|v| v.as_f32()).unwrap_or(1.0),
            contrast: outn.params.get("contrast").map(|v| v.as_f32()).unwrap_or(1.0),
            ambient: 0,
            precipitation: 0,
            particle_density: outn.params.get("particle_density").map(|v| v.as_f32()).unwrap_or(1.0),
        });
        sh.depth_biome = Some(bi as u32);
        let mut em = Em::new(&mut sh, g, pfx.clone(), false);
        fluids.push(em.enum_idx(&outn, "fluid"));
        if let Some(info) = biome_infos.last_mut() {
            info.ambient = em.enum_idx(&outn, "ambient") as u32;
            info.precipitation = em.enum_idx(&outn, "precipitation") as u32;
        }
        // density
        let dens = g
            .input_link(outn.id, 0)
            .copied()
            .map(|l| {
                let v = em.density(l.from_node, l.from_pin, 0);
                em.need(&v);
                v.e
            })
            .unwrap_or_else(|| "(64.0 - p.y)".into());
        let dens_body = em.take_body();
        // material
        let mat = g.input_link(outn.id, 1).copied().map(|l| em.material(l.from_node)).unwrap_or_else(|| "B_STONE".into());
        let mat_body = em.take_body();
        // decor
        let dec = g.input_link(outn.id, 2).copied().map(|l| em.decor(l.from_node)).unwrap_or_else(|| "0u".into());
        let dec_body = em.take_body();
        // props
        let mut masks = Vec::new();
        for slot in 3..9u8 {
            let Some(l) = g.input_link(outn.id, slot).copied() else { continue };
            let Some(layer_node) = g.node(l.from_node).cloned() else { continue };
            let li = prop_layers.len();
            let desc = match layer_node.kind.as_str() {
                "scatter" => {
                    let Some(pl) = g.input_link(layer_node.id, 0).copied() else {
                        em.warn("Scatter without a prop".into());
                        continue;
                    };
                    let Some(prop) = g.node(pl.from_node).cloned() else { continue };
                    let mut d = String::new();
                    let n = &layer_node;
                    let _ = write!(
                        d,
                        "L.spacing = {}; L.chance = {}; L.jitter = {}; L.surface = {}; L.anySurface = {}; L.yMin = {}; L.yMax = {}; L.seed = gSeed + {} * 7919u + {}u; ",
                        em.p(n, "spacing"),
                        em.p(n, "chance"),
                        em.p(n, "jitter"),
                        em.block(n, "surface"),
                        em.boolean(n, "any_surface"),
                        em.p(n, "y_min"),
                        em.p(n, "y_max"),
                        em.pu(n, "seed"),
                        n.id
                    );
                    let p = &prop;
                    match p.kind.as_str() {
                        "tree" => {
                            let _ = write!(
                                d,
                                "L.kind = PROP_TREE; L.variant = {}u; L.b0 = {}; L.b1 = {}; L.a = {}; L.b = {}; L.c = {};",
                                em.enum_idx(p, "species"),
                                em.block(p, "trunk"),
                                em.block(p, "leaves"),
                                em.p(p, "height_min"),
                                em.p(p, "height_max"),
                                em.p(p, "radius")
                            );
                        }
                        "giant_mushroom" => {
                            let _ = write!(
                                d,
                                "L.kind = PROP_MUSHROOM; L.b0 = {}; L.b1 = {}; L.a = {}; L.b = {}; L.c = {};",
                                em.block(p, "stem"),
                                em.block(p, "cap"),
                                em.p(p, "height_min"),
                                em.p(p, "height_max"),
                                em.p(p, "radius")
                            );
                        }
                        "boulder" => {
                            let _ = write!(
                                d,
                                "L.kind = PROP_BOULDER; L.b0 = {}; L.b1 = {}; L.a = {}; L.b = {}; L.c = {};",
                                em.block(p, "block"),
                                em.block(p, "block2"),
                                em.p(p, "radius_min"),
                                em.p(p, "radius_max"),
                                em.p(p, "radius_max")
                            );
                        }
                        "column" => {
                            let _ = write!(
                                d,
                                "L.kind = PROP_COLUMN; L.variant = {}u; L.b0 = {}; L.a = {}; L.b = {}; L.c = {};",
                                em.boolean(p, "arms") as u32,
                                em.block(p, "block"),
                                em.p(p, "height_min"),
                                em.p(p, "height_max"),
                                em.p(p, "radius")
                            );
                        }
                        "spike" => {
                            let _ = write!(
                                d,
                                "L.kind = PROP_SPIKE; L.b0 = {}; L.a = {}; L.b = {}; L.c = {};",
                                em.block(p, "block"),
                                em.p(p, "height_min"),
                                em.p(p, "height_max"),
                                em.p(p, "radius")
                            );
                        }
                        "prefab" => {
                            let name = match em.value(p, "prefab") {
                                Value::Ref(r) => r,
                                _ => String::new(),
                            };
                            let Some(&(idx, size)) = em.sh.prefabs.get(&name) else {
                                em.warn(format!("prefab '{name}' not found"));
                                continue;
                            };
                            let _ = write!(
                                d,
                                "L.kind = PROP_PREFAB; L.variant = {idx}u; L.a = {}.0; L.b = {}.0; L.c = {}.0; L.b0 = {}u; L.b1 = uint({});",
                                size[0],
                                size[1],
                                size[2],
                                em.boolean(p, "random_rotation") as u32,
                                em.p(p, "sink")
                            );
                        }
                        other => {
                            em.warn(format!("'{other}' is not a prop"));
                            continue;
                        }
                    }
                    // optional mask (3D function using the biome's 2D cache)
                    if let Some(ml) = g.input_link(layer_node.id, 1).copied() {
                        let v = em.density(ml.from_node, ml.from_pin, 0);
                        em.need(&v);
                        let body = em.take_body().join("\n");
                        masks.push(format!("float {pfx}mask{li}(vec3 p, in float h[MAXH]) {{\n{{EXPORTS}}\n{body}\n    return {};\n}}", v.e));
                        d.push_str(" L.hasMask = true;");
                    }
                    d
                }
                "geode" => {
                    let n = &layer_node;
                    format!(
                        "L.kind = PROP_GEODE; L.b0 = {}; L.b1 = {}; L.b2 = {}; L.c = {}; L.a = {}; L.b = {}; L.spacing = {}; L.chance = {}; L.yMin = {}; L.yMax = {}; L.seed = gSeed + {} * 7919u + {}u; L.jitter = 0.8; L.anySurface = true;",
                        em.block(n, "outer"),
                        em.block(n, "middle"),
                        em.block(n, "inner"),
                        em.p(n, "radius"),
                        em.p(n, "radius"),
                        em.p(n, "radius"),
                        em.p(n, "spacing"),
                        em.p(n, "chance"),
                        em.p(n, "y_min"),
                        em.p(n, "y_max"),
                        em.pu(n, "seed"),
                        n.id
                    )
                }
                other => {
                    em.warn(format!("'{other}' is not a prop layer"));
                    continue;
                }
            };
            let has_mask = desc.contains("hasMask");
            prop_layers.push(PropLayerGen { biome: bi as u32, desc, mask_fn: if has_mask { Some(format!("{pfx}mask{li}")) } else { None } });
        }

        // exports -> h[] slots
        let mut slots = Vec::new();
        let mut store = Vec::new();
        let mut slot = 0usize;
        for e in &em.exports {
            if e.width == 3 {
                store.push(format!("    h[{}] = {}.x; h[{}] = {}.y; h[{}] = {}.z;", slot, e.e, slot + 1, e.e, slot + 2, e.e));
                slots.push(format!("    vec3 {} = vec3(h[{}], h[{}], h[{}]);", e.e, slot, slot + 1, slot + 2));
                slot += 3;
            } else {
                store.push(format!("    h[{}] = {};", slot, e.e));
                slots.push(format!("    float {} = h[{}];", e.e, slot));
                slot += 1;
            }
        }
        maxh = maxh.max(slot);
        let exports = slots.join("\n");
        let pre = em.pre.join("\n");
        let _ = writeln!(biome_code, "// ---- biome {bi}: {name}");
        let _ = writeln!(biome_code, "void {pfx}col(vec2 xz, inout float h[MAXH]) {{\n    vec3 p = vec3(xz.x, 0.0, xz.y);\n{pre}\n{}\n}}", store.join("\n"));
        let _ = writeln!(biome_code, "float {pfx}den(vec3 p, in float h[MAXH]) {{\n{exports}\n{}\n    return {dens};\n}}", dens_body.join("\n"));
        let _ = writeln!(
            biome_code,
            "uint {pfx}mat(vec3 p, float depth, float airAbove, bool skyOpen, in float h[MAXH]) {{\n{exports}\n{}\n    return {mat};\n}}",
            mat_body.join("\n")
        );
        let _ = writeln!(biome_code, "uint {pfx}dec(vec3 p, uint ground, in float h[MAXH]) {{\n{exports}\n{}\n    return {dec};\n}}", dec_body.join("\n"));
        for m in masks {
            let _ = writeln!(biome_code, "{}", m.replace("{EXPORTS}", &exports));
        }
        sh.depth_biome = None;
        if cave_biome_ids.contains(&(bi as u32)) {
            let mut cem = Em::new(&mut sh, g, format!("b{bi}c_"), true);
            let cmat = g.input_link(outn.id, 1).copied().map(|l| cem.material(l.from_node)).unwrap_or_else(|| "B_STONE".into());
            let cmat_body = cem.take_body().join("\n");
            let cdec = g.input_link(outn.id, 2).copied().map(|l| cem.decor(l.from_node)).unwrap_or_else(|| "0u".into());
            let cdec_body = cem.take_body().join("\n");
            let _ = writeln!(biome_code, "uint b{bi}_cmat(vec3 p, float depth, float airAbove) {{\n    bool skyOpen = false;\n{cmat_body}\n    return {cmat};\n}}");
            let _ = writeln!(biome_code, "uint b{bi}_cdec(vec3 p, uint ground) {{\n{cdec_body}\n    return {cdec};\n}}");
        }
    }

    // ---- zones: biome layout + carvers ----
    let mut zone_infos = Vec::new();
    let mut zone_code = String::new();
    for (zi, name) in zone_names.iter().enumerate() {
        let g = project.get(GraphKind::Zone, name).unwrap();
        let Some(outn) = g.output_node().cloned() else {
            sh.warnings.push(format!("Zone '{name}' has no output node"));
            continue;
        };
        zone_infos.push(ZoneInfo { name: name.clone(), map_color: color_param(g, &outn, "map_color") });
        let mut em = Em::new(&mut sh, g, format!("z{zi}_"), true);
        // carver
        let carve = g.input_link(outn.id, 1).copied().map(|l| em.density(l.from_node, l.from_pin, 0).e).unwrap_or_else(|| "0.0".into());
        let body = em.take_body().join("\n");
        let _ = writeln!(zone_code, "float z{zi}_carve(vec3 p) {{\n{body}\n    return {carve};\n}}");
        // biome selection
        let sel = g.input_link(outn.id, 0).copied().and_then(|l| g.node(l.from_node).cloned());
        let mut f = format!("void z{zi}_biomes(vec2 xz, out uint b0, out uint b1, out float w1) {{\n");
        match sel {
            Some(n) if n.kind == "biome" => {
                let id = match em.value(&n, "biome") {
                    Value::Ref(r) => em.sh.biome_ids.get(&r).copied(),
                    _ => None,
                };
                let id = id.unwrap_or_else(|| {
                    em.warn("Biome node without a valid biome".into());
                    0
                });
                let _ = writeln!(f, "    b0 = {id}u; b1 = {id}u; w1 = 0.0;");
            }
            Some(n) if n.kind == "biome_threshold" => {
                let mut ids = Vec::new();
                for i in 1..=6u8 {
                    let Some(l) = g.input_link(n.id, i).copied() else { continue };
                    let Some(bn) = g.node(l.from_node).cloned() else { continue };
                    match em.value(&bn, "biome") {
                        Value::Ref(r) if em.sh.biome_ids.contains_key(&r) => ids.push(em.sh.biome_ids[&r]),
                        _ => em.warn("Biome node without a valid biome".into()),
                    }
                }
                if ids.is_empty() {
                    em.warn("Biome Threshold Select has no biomes connected".into());
                    ids.push(0);
                }
                // selector: a 2D density evaluated at the column
                let sel = g.input_link(n.id, 0).copied().map(|l| em.density(l.from_node, l.from_pin, 0).e);
                let body = em.take_body().join("\n");
                let sel = sel.unwrap_or_else(|| {
                    em.warn("Biome Threshold Select has no selector connected".into());
                    "0.0".into()
                });
                let th: Vec<String> = (1..ids.len()).map(|i| em.p(&n, &format!("threshold{i}"))).collect();
                let bl = em.p(&n, "blend");
                let _ = writeln!(f, "    vec3 p = vec3(xz.x, 0.0, xz.y);\n{body}\n    float v = {sel};");
                let nb = ids.len();
                let list = ids.iter().map(|i| format!("{i}u")).collect::<Vec<_>>().join(", ");
                let _ = writeln!(f, "    uint ids[{nb}] = uint[{nb}]({list});");
                if nb == 1 {
                    let _ = writeln!(f, "    b0 = ids[0]; b1 = ids[0]; w1 = 0.0;");
                } else {
                    let _ = writeln!(f, "    float th[{}] = float[{}]({});", nb - 1, nb - 1, th.join(", "));
                    let _ = writeln!(
                        f,
                        r#"    int k = 0;
    while (k < {last} && v >= th[k]) k++;
    b0 = ids[k]; b1 = b0; w1 = 0.0;
    float bl = max({bl}, 1e-4);
    float dLo = k > 0 ? v - th[k - 1] : 1e9;
    float dHi = k < {last} ? th[min(k, {last} - 1)] - v : 1e9;
    if (dLo < bl && dLo <= dHi) {{ b1 = ids[k - 1]; w1 = 0.5 * (1.0 - dLo / bl); }}
    else if (dHi < bl) {{ b1 = ids[k + 1]; w1 = 0.5 * (1.0 - dHi / bl); }}"#,
                        last = nb - 1
                    );
                }
            }
            Some(n) if n.kind == "biome_voronoi" => {
                let mut entries = Vec::new();
                for i in 0..8u8 {
                    let Some(l) = g.input_link(n.id, i).copied() else { continue };
                    let Some(bn) = g.node(l.from_node).cloned() else { continue };
                    let id = match em.value(&bn, "biome") {
                        Value::Ref(r) => em.sh.biome_ids.get(&r).copied(),
                        _ => None,
                    };
                    match id {
                        Some(id) => {
                            let w = em.p(&n, &format!("weight{}", i + 1));
                            entries.push((id, w));
                        }
                        None => em.warn("Biome node without a valid biome".into()),
                    }
                }
                if entries.is_empty() {
                    em.warn("Biome Voronoi has no biomes connected".into());
                    entries.push((0, "1.0".into()));
                }
                let (cs, ji, bl, wa) = (em.p(&n, "cell_size"), em.p(&n, "jitter"), em.p(&n, "blend"), em.p(&n, "warp"));
                let seed = em.pu(&n, "seed");
                let _ = writeln!(f, "    uint seed = gSeed + {seed} * 7919u + {}u;", n.id * 31 + zi as u32);
                let _ = writeln!(f, "    vec2 q = xz + vec2(fbm2(xz / ({cs} * 0.6), seed + 11u, 3, 2.0, 0.5), fbm2(xz / ({cs} * 0.6), seed + 12u, 3, 2.0, 0.5)) * {wa};");
                let _ = writeln!(f, "    ivec2 c1, c2; float d1, d2;\n    voronoi2(q / {cs}, {ji}, seed, c1, c2, d1, d2);");
                let total = entries.iter().map(|(_, w)| w.clone()).collect::<Vec<_>>().join(" + ");
                let mut pick = format!("    float tot = max({total}, 1e-4);\n");
                for (k, cell) in ["c1", "c2"].iter().enumerate() {
                    let _ = writeln!(pick, "    float r{k} = rand2({cell}, seed) * tot; uint s{k} = {}u;", entries.last().unwrap().0);
                    let mut acc = String::from("0.0");
                    let mut chain = String::new();
                    for (id, w) in &entries {
                        acc = format!("{acc} + {w}");
                        let _ = writeln!(chain, "    if (r{k} < {acc}) {{ s{k} = {id}u; }} else");
                    }
                    let _ = writeln!(pick, "{chain}    {{}}");
                }
                f.push_str(&pick);
                let _ = writeln!(f, "    b0 = s0; b1 = s1;\n    float edge = (d2 - d1) * {cs};\n    w1 = (s0 == s1) ? 0.0 : 0.5 * (1.0 - smoothstep(0.0, max({bl}, 1e-3), edge));");
            }
            _ => {
                em.warn("zone has no biomes connected".into());
                let _ = writeln!(f, "    b0 = 0u; b1 = 0u; w1 = 0.0;");
            }
        }
        f.push_str("}\n");
        zone_code.push_str(&f);

        // cave biome selection (per voxel)
        let cave_sel = g.input_link(outn.id, 2).copied().and_then(|l| g.node(l.from_node).cloned());
        let biome_ref = |em: &Em, n: &Node| match em.value(n, "biome") {
            Value::Ref(r) => em.sh.biome_ids.get(&r).copied(),
            _ => None,
        };
        let mut cf = format!("uint z{zi}_cave(vec3 p) {{\n");
        match cave_sel {
            Some(n) if n.kind == "biome" => match biome_ref(&em, &n) {
                Some(id) => {
                    let _ = writeln!(cf, "    return {id}u;");
                }
                None => {
                    em.warn("Biome node without a valid biome".into());
                    cf.push_str("    return NO_CAVE_BIOME;\n");
                }
            },
            Some(n) if n.kind == "cave_biomes" => {
                let mut entries = Vec::new();
                for i in 0..4u8 {
                    let Some(l) = g.input_link(n.id, i).copied() else { continue };
                    let Some(bn) = g.node(l.from_node).cloned() else { continue };
                    match biome_ref(&em, &bn) {
                        Some(id) => {
                            let w = em.p(&n, &format!("weight{}", i + 1));
                            entries.push((id, w));
                        }
                        None => em.warn("Biome node without a valid biome".into()),
                    }
                }
                if entries.is_empty() {
                    em.warn("Cave Biome Select has no biomes connected".into());
                    cf.push_str("    return NO_CAVE_BIOME;\n");
                } else {
                    let (cs, wa) = (em.p(&n, "cell_size"), em.p(&n, "warp"));
                    let seed = em.pu(&n, "seed");
                    let _ = writeln!(cf, "    uint seed = gSeed + {seed} * 7919u + {}u;", n.id * 37 + zi as u32);
                    let _ = writeln!(
                        cf,
                        "    vec3 q = p + vec3(fbm3(p / ({cs} * 0.5), seed + 1u, 2, 2.0, 0.5), fbm3(p / ({cs} * 0.5), seed + 2u, 2, 2.0, 0.5), fbm3(p / ({cs} * 0.5), seed + 3u, 2, 2.0, 0.5)) * {wa};"
                    );
                    let total = entries.iter().map(|(_, w)| w.clone()).collect::<Vec<_>>().join(" + ");
                    let _ = writeln!(cf, "    float r = rand3(ivec3(floor(q / {cs})), seed) * max({total}, 1e-4);");
                    let mut acc = String::from("0.0");
                    for (id, w) in &entries {
                        acc = format!("{acc} + {w}");
                        let _ = writeln!(cf, "    if (r < {acc}) return {id}u;");
                    }
                    let _ = writeln!(cf, "    return {}u;", entries.last().unwrap().0);
                }
            }
            Some(n) => {
                em.warn(format!("'{}' cannot choose cave biomes (use Biome or Cave Biome Select)", n.kind));
                cf.push_str("    return NO_CAVE_BIOME;\n");
            }
            None => cf.push_str("    return NO_CAVE_BIOME;\n"),
        }
        cf.push_str("}\n");
        zone_code.push_str(&cf);
    }

    // ---- world: zone layout ----
    let mut world_code = String::new();
    let world = project.graphs.iter().find(|g| g.kind == GraphKind::World);
    let mut sea = "62.0".to_string();
    let mut river = (String::from("float wg_river(vec2 xz) { return 0.0; }"), "5.0".to_string(), "48.0".to_string(), "B_GRAVEL".to_string());
    {
        let _ = writeln!(world_code, "void zone_select(vec2 xz, out uint z0, out uint z1, out float w1) {{");
        match world {
            Some(g) => {
                let mut em = Em::new(&mut sh, g, "w_".into(), true);
                let outn = g.output_node().cloned();
                if let Some(o) = &outn {
                    sea = em.p(o, "sea_level");
                    // rivers: a 2D mask (World Output pin 1) carved down to the river bed
                    if let Some(l) = g.input_link(o.id, 1).copied() {
                        let m = em.density(l.from_node, l.from_pin, 0).e;
                        let body = em.take_body().join("\n");
                        river.0 = format!("float wg_river(vec2 xz) {{\n    vec3 p = vec3(xz.x, 0.0, xz.y);\n{body}\n    return clamp({m}, 0.0, 1.0);\n}}");
                    }
                    river.1 = em.p(o, "river_depth");
                    river.2 = em.p(o, "river_carve");
                    river.3 = em.block(o, "river_bed");
                }
                let sel = outn.as_ref().and_then(|o| g.input_link(o.id, 0).copied()).and_then(|l| g.node(l.from_node).cloned());
                match sel {
                    Some(n) if n.kind == "zone" => {
                        let id = match em.value(&n, "zone") {
                            Value::Ref(r) => em.sh.zone_ids.get(&r).copied(),
                            _ => None,
                        }
                        .unwrap_or(0);
                        let _ = writeln!(world_code, "    z0 = {id}u; z1 = {id}u; w1 = 0.0;");
                    }
                    Some(n) if n.kind == "zone_voronoi" => {
                        let mut entries = Vec::new();
                        for i in 0..8u8 {
                            let Some(l) = g.input_link(n.id, i).copied() else { continue };
                            let Some(zn) = g.node(l.from_node).cloned() else { continue };
                            let id = match em.value(&zn, "zone") {
                                Value::Ref(r) => em.sh.zone_ids.get(&r).copied(),
                                _ => None,
                            };
                            match id {
                                Some(id) => {
                                    let w = em.p(&n, &format!("weight{}", i + 1));
                                    entries.push((id, w));
                                }
                                None => em.warn("Zone node without a valid zone".into()),
                            }
                        }
                        if entries.is_empty() {
                            entries.push((0, "1.0".into()));
                        }
                        let (cs, ji, bl, wa) = (em.p(&n, "cell_size"), em.p(&n, "jitter"), em.p(&n, "blend"), em.p(&n, "warp"));
                        let seed = em.pu(&n, "seed");
                        let _ = writeln!(world_code, "    uint seed = gSeed + {seed} * 7919u + 977u;");
                        let _ = writeln!(world_code, "    vec2 q = xz + vec2(fbm2(xz / ({cs} * 0.5), seed + 1u, 4, 2.0, 0.5), fbm2(xz / ({cs} * 0.5), seed + 2u, 4, 2.0, 0.5)) * {wa};");
                        let _ = writeln!(world_code, "    ivec2 c1, c2; float d1, d2;\n    voronoi2(q / {cs}, {ji}, seed, c1, c2, d1, d2);");
                        let total = entries.iter().map(|(_, w)| w.clone()).collect::<Vec<_>>().join(" + ");
                        let _ = writeln!(world_code, "    float tot = max({total}, 1e-4);");
                        for (k, cell) in ["c1", "c2"].iter().enumerate() {
                            let _ = writeln!(world_code, "    float r{k} = rand2({cell}, seed) * tot; uint s{k} = {}u;", entries.last().unwrap().0);
                            let mut acc = String::from("0.0");
                            for (id, w) in &entries {
                                acc = format!("{acc} + {w}");
                                let _ = writeln!(world_code, "    if (r{k} < {acc}) {{ s{k} = {id}u; }} else");
                            }
                            let _ = writeln!(world_code, "    {{}}");
                        }
                        let _ = writeln!(world_code, "    z0 = s0; z1 = s1;\n    float edge = (d2 - d1) * {cs};\n    w1 = (s0 == s1) ? 0.0 : 0.5 * (1.0 - smoothstep(0.0, max({bl}, 1e-3), edge));");
                    }
                    _ => {
                        let _ = writeln!(world_code, "    z0 = 0u; z1 = 0u; w1 = 0.0;");
                    }
                }
            }
            None => {
                let _ = writeln!(world_code, "    z0 = 0u; z1 = 0u; w1 = 0.0;");
            }
        }
        let _ = writeln!(world_code, "}}");
    }

    // ---- assemble ----
    let nb = biome_infos.len();
    let nz = zone_infos.len();
    let mut s = String::new();
    let _ = writeln!(s, "#define MAXH {maxh}\n#define NUM_BIOMES {nb}u\n#define NUM_ZONES {nz}u\n#define NUM_PROP_LAYERS {}u", prop_layers.len());
    let _ = writeln!(s, "struct WgColumn {{\n    uint zone0; uint zone1; float zw1;\n    uint nb;\n    uint biome[3];\n    float bw[3];\n    float h[3][MAXH];\n    float river;\n}};");
    let _ = writeln!(
        s,
        "float gSeaLevel; float gRiverDepth; float gRiverCarve; uint gRiverBed;\nvoid wg_init() {{ gSeaLevel = {sea}; gRiverDepth = {}; gRiverCarve = {}; gRiverBed = {}; }}",
        river.1, river.2, river.3
    );
    s.push_str(&out);
    s.push_str(&fns);
    s.push_str(&biome_code);
    s.push_str(&zone_code);
    s.push_str(&world_code);
    let _ = writeln!(s, "{}", river.0);

    // dispatchers
    let sw = |name: &str, call: &dyn Fn(usize) -> String, n: usize, default: &str| -> String {
        let mut x = format!("    switch ({name}) {{\n");
        for i in 0..n {
            let _ = writeln!(x, "        case {i}u: {}", call(i));
        }
        let _ = writeln!(x, "        default: {default}\n    }}");
        x
    };
    let _ = writeln!(s, "void zone_biomes(uint z, vec2 xz, out uint b0, out uint b1, out float w1) {{\n    b0 = 0u; b1 = 0u; w1 = 0.0;");
    s.push_str(&sw("z", &|i| format!("z{i}_biomes(xz, b0, b1, w1); break;"), nz, "break;"));
    let _ = writeln!(s, "}}");
    let _ = writeln!(s, "float wg_carve(vec3 p, in WgColumn c) {{\n    uint z = c.zw1 > 0.5 ? c.zone1 : c.zone0;");
    s.push_str(&sw("z", &|i| format!("return z{i}_carve(p);"), nz, "return 0.0;"));
    let _ = writeln!(s, "}}");

    let _ = writeln!(
        s,
        r#"void wg_column(vec2 xz, out WgColumn c) {{
    uint z0, z1; float zw1;
    zone_select(xz, z0, z1, zw1);
    c.zone0 = z0; c.zone1 = z1; c.zw1 = zw1;
    c.river = wg_river(xz);
    uint cb[4]; float cw[4];
    uint a0, a1; float aw;
    zone_biomes(z0, xz, a0, a1, aw);
    cb[0] = a0; cw[0] = (1.0 - aw) * (1.0 - zw1);
    cb[1] = a1; cw[1] = aw * (1.0 - zw1);
    cb[2] = 0u; cw[2] = 0.0; cb[3] = 0u; cw[3] = 0.0;
    if (zw1 > 0.0) {{
        zone_biomes(z1, xz, a0, a1, aw);
        cb[2] = a0; cw[2] = (1.0 - aw) * zw1;
        cb[3] = a1; cw[3] = aw * zw1;
    }}
    // merge duplicates
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            if (cw[j] > 0.0 && cb[j] == cb[i]) {{ cw[i] += cw[j]; cw[j] = 0.0; }}
    // pick top 3 by weight
    c.nb = 0u;
    float total = 0.0;
    for (int k = 0; k < 3; ++k) {{
        int best = -1; float bw = 0.002;
        for (int i = 0; i < 4; ++i) if (cw[i] > bw) {{ bw = cw[i]; best = i; }}
        if (best < 0) break;
        c.biome[c.nb] = cb[best]; c.bw[c.nb] = bw; total += bw; cw[best] = 0.0; c.nb++;
    }}
    if (c.nb == 0u) {{ c.nb = 1u; c.biome[0] = cb[0]; c.bw[0] = 1.0; total = 1.0; }}
    for (uint k = 0u; k < c.nb; ++k) {{
        c.bw[k] /= total;
        switch (c.biome[k]) {{
"#
    );
    for i in 0..nb {
        let _ = writeln!(s, "            case {i}u: b{i}_col(xz, c.h[k]); break;");
    }
    let _ = writeln!(s, "            default: break;\n        }}\n    }}\n}}");

    let _ = writeln!(s, "float biome_density(uint b, vec3 p, in float h[MAXH]) {{");
    s.push_str(&sw("b", &|i| format!("return b{i}_den(p, h);"), nb, "return 0.0;"));
    let _ = writeln!(s, "}}");
    let _ = writeln!(
        s,
        r#"// `carved`: the cell is solid terrain removed by the zone's cave carver (cave air)
float wg_density_c(vec3 p, in WgColumn c, out bool carved) {{
    carved = false;
    float d = 0.0;
    if (c.nb == 1u) d = biome_density(c.biome[0], p, c.h[0]);
    else for (uint k = 0u; k < c.nb; ++k) d += c.bw[k] * biome_density(c.biome[k], p, c.h[k]);
    // rivers: pull the terrain down towards the river bed (sea level - depth at the centre),
    // up to `river_carve` above sea level so floating islands and high overhangs stay intact
    if (c.river > 0.0) {{
        float top = gSeaLevel + gRiverCarve;
        float k = smoothstep(0.0, 1.0, c.river) * (1.0 - smoothstep(top, top + 16.0, p.y));
        if (k > 0.0) d = mix(d, min(d, gSeaLevel - gRiverDepth * c.river - p.y), k);
    }}
    if (d > 0.0) {{
        // carving only removes solid: skip the cave noise for cells that are already air
        float cv = wg_carve(p, c);
        if (cv > 0.0) {{ d = min(d, -cv * 4.0); carved = d <= 0.0; }}
    }}
    return d;
}}
float wg_density(vec3 p, in WgColumn c) {{
    bool carved;
    return wg_density_c(p, c, carved);
}}
uint wg_pick(vec3 p, in WgColumn c) {{
    if (c.nb == 1u) return 0u;
    float r = rand3(ivec3(floor(p)), gSeed ^ 0x5bd1e995u);
    float acc = 0.0;
    for (uint k = 0u; k < c.nb; ++k) {{ acc += c.bw[k]; if (r < acc) return k; }}
    return 0u;
}}
uint wg_main_biome(in WgColumn c) {{ return c.biome[0]; }}"#
    );
    // the deepest material threshold in a column (2.5: the river bed; cave materials count
    // everywhere), read live so slider edits stay exact
    sh.depth_terms.sort();
    sh.depth_terms.dedup();
    let max_of = |b: Option<u32>, init: &str| {
        sh.depth_terms.iter().filter(|(k, _)| *k == b).fold(init.to_string(), |m, (_, t)| format!("max({m}, {t})"))
    };
    let _ = writeln!(s, "float wg_depth_max(in WgColumn c) {{\n    float m = {};\n    for (uint k = 0u; k < c.nb; ++k) {{", max_of(None, "2.5"));
    s.push_str(&format!("    {}", sw("c.biome[k]", &|i| format!("m = {}; break;", max_of(Some(i as u32), "m")), nb, "break;").replace("\n    ", "\n        ").trim_start()));
    let _ = writeln!(s, "    }}\n    return m;\n}}");
    let _ = writeln!(
        s,
        "uint wg_material(vec3 p, float depth, float airAbove, bool skyOpen, in WgColumn c) {{\n    if (c.river > 0.4 && skyOpen && depth < 2.5 && p.y < gSeaLevel + 1.0) return gRiverBed;\n    uint k = wg_pick(p, c);"
    );
    s.push_str(&sw("c.biome[k]", &|i| format!("return b{i}_mat(p, depth, airAbove, skyOpen, c.h[k]);"), nb, "return B_STONE;"));
    let _ = writeln!(s, "}}");
    let _ = writeln!(s, "uint wg_decorate(vec3 p, uint ground, in WgColumn c) {{\n    uint k = wg_pick(p, c);");
    s.push_str(&sw("c.biome[k]", &|i| format!("return b{i}_dec(p, ground, c.h[k]);"), nb, "return 0u;"));
    let _ = writeln!(s, "}}");
    let _ = writeln!(s, "uint wg_fluid(vec3 p, bool skyOpen, in WgColumn c) {{\n    if (!skyOpen || p.y >= gSeaLevel) return 0u;");
    s.push_str(&sw(
        "c.biome[0]",
        &|i| match fluids.get(i).copied().unwrap_or(0) {
            0 => "return B_WATER;".to_string(),
            1 => "return B_LAVA;".to_string(),
            2 => "return p.y >= gSeaLevel - 1.0 ? B_ICE : B_WATER;".to_string(),
            _ => "return 0u;".to_string(),
        },
        nb,
        "return B_WATER;",
    ));
    let _ = writeln!(s, "}}");

    // ---- cave biomes ----
    let _ = writeln!(s, "uint wg_cave_biome(vec3 p, in WgColumn c) {{\n    uint z = c.zw1 > 0.5 ? c.zone1 : c.zone0;");
    s.push_str(&sw("z", &|i| format!("return z{i}_cave(p);"), nz, "return NO_CAVE_BIOME;"));
    let _ = writeln!(s, "}}");
    let mut cm = String::from("uint wg_cave_material(uint b, vec3 p, float depth, float airAbove) {\n    switch (b) {\n");
    let mut cd = String::from("uint wg_cave_decorate(uint b, vec3 p, uint ground) {\n    switch (b) {\n");
    for id in &cave_biome_ids {
        let _ = writeln!(cm, "        case {id}u: return b{id}_cmat(p, depth, airAbove);");
        let _ = writeln!(cd, "        case {id}u: return b{id}_cdec(p, ground);");
    }
    cm.push_str("        default: return B_STONE;\n    }\n}\n");
    cd.push_str("        default: return 0u;\n    }\n}\n");
    s.push_str(&cm);
    s.push_str(&cd);

    // ---- props ----
    let _ = writeln!(s, "void prop_layer(uint i, inout PropLayer L) {{\n    switch (i) {{");
    for (i, l) in prop_layers.iter().enumerate() {
        let _ = writeln!(s, "        case {i}u: L.biome = {}u; {} break;", l.biome, l.desc);
    }
    let _ = writeln!(s, "        default: break;\n    }}\n}}");
    let _ = writeln!(s, "float prop_mask(uint i, vec3 p, in float h[MAXH]) {{\n    switch (i) {{");
    for (i, l) in prop_layers.iter().enumerate() {
        if let Some(m) = &l.mask_fn {
            let _ = writeln!(s, "        case {i}u: return {m}(p, h);");
        }
    }
    let _ = writeln!(s, "        default: return 1.0;\n    }}\n}}");

    Ok(Compiled {
        source: s,
        params: sh.params.values,
        param_index: sh.params.index,
        biomes: biome_infos,
        zones: zone_infos,
        prop_layers: prop_layers.len(),
        warnings: sh.warnings,
    })
}
