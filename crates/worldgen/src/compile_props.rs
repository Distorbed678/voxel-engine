//! Prop graphs -> GLSL build functions (see `shaders/worldgen/props_lib.glsl`).
//!
//! A Prop graph compiles to `void propN_build()`, which one thread runs per placed instance: shape
//! nodes append primitives, tree nodes grow skeletons first and emit primitives from them, and
//! combining nodes change the primitives their inputs emitted (op, block, noise, group) or wrap
//! them in a transform (a saved and restored `gXf`). Values (Density pins, e.g. a Random) are
//! ordinary density expressions evaluated at the anchor (`p` is its world position).

use std::collections::HashMap;

use super::{Em, Shared};
use crate::graph::{Graph, Node, Value};
use crate::nodes::{self, ParamKind};

/// A live parameter of any graph's node (the prop graph's output, read from a biome's scatter).
pub(super) fn live_param(sh: &mut Shared, g: &Graph, n: &Node, name: &str) -> String {
    let v = n
        .params
        .get(name)
        .cloned()
        .or_else(|| nodes::def(&n.kind).and_then(|d| d.params.iter().find(|p| p.name == name)).map(|p| p.kind.default_value()))
        .map(|v| v.as_f32())
        .unwrap_or(0.0);
    let k = super::ParamKey { graph: (g.kind, g.name.clone()), node: n.id, param: name.to_string() };
    format!("P({}u)", sh.params.push(k, v))
}

pub(super) fn bool_param(n: &Node, name: &str) -> bool {
    match n.params.get(name) {
        Some(Value::B(b)) => *b,
        Some(_) => false,
        None => matches!(nodes::def(&n.kind).and_then(|d| d.params.iter().find(|p| p.name == name)).map(|p| p.kind), Some(ParamKind::Bool(true))),
    }
}

impl Em<'_, '_> {
    fn stmt(&mut self, s: String) {
        self.body.push(format!("    {s}"));
    }

    fn fresh(&mut self, what: &str) -> String {
        self.n += 1;
        format!("{}{what}{}", self.pfx, self.n)
    }

    /// A per-instance random number in [0, 1) for this node (`k` tells several apart).
    fn rnd(&self, n: &Node, k: u32) -> String {
        format!("pr({}u)", n.id.wrapping_mul(64).wrapping_add(k))
    }

    fn nkey(&self, n: &Node) -> String {
        format!("{}u", n.id.wrapping_mul(1_000_003))
    }

    fn block_prio(&mut self, n: &Node) -> (String, String) {
        let b = self.block(n, "block");
        let p = self.pu(n, "priority");
        (b, p)
    }

    /// Emit the primitives of shape output `pin` of node `id`.
    pub(super) fn shape(&mut self, id: u32, pin: u8) {
        let Some(node) = self.g.node(id) else { return };
        let n = node.clone();
        self.depth += 1;
        if self.depth > 256 {
            self.warn("prop graph too deep".into());
            self.depth -= 1;
            return;
        }
        match n.kind.as_str() {
            "sphere" => {
                let (x, y, z, r, var) = (self.p(&n, "x"), self.p(&n, "y"), self.p(&n, "z"), self.p(&n, "radius"), self.p(&n, "var"));
                let (b, pr) = self.block_prio(&n);
                let rv = self.rnd(&n, 0);
                self.stmt(format!("pb_sphere(vec3({x}, {y}, {z}), {r} * mix(1.0 - {var}, 1.0 + {var}, {rv}), {b}, {pr});"));
            }
            "ellipsoid" => {
                let (x, y, z) = (self.p(&n, "x"), self.p(&n, "y"), self.p(&n, "z"));
                let (rx, ry, rz, yaw, var) = (self.p(&n, "rx"), self.p(&n, "ry"), self.p(&n, "rz"), self.p(&n, "yaw"), self.p(&n, "var"));
                let (b, pr) = self.block_prio(&n);
                let rv = self.rnd(&n, 0);
                self.stmt(format!("pb_ellipsoid(vec3({x}, {y}, {z}), vec3({rx}, {ry}, {rz}) * mix(1.0 - {var}, 1.0 + {var}, {rv}), {yaw}, {b}, {pr});"));
            }
            "capsule" => {
                let a = (self.p(&n, "x0"), self.p(&n, "y0"), self.p(&n, "z0"));
                let c = (self.p(&n, "x1"), self.p(&n, "y1"), self.p(&n, "z1"));
                let (r0, r1) = (self.p(&n, "r0"), self.p(&n, "r1"));
                let (b, pr) = self.block_prio(&n);
                self.stmt(format!("pb_capsule(vec3({}, {}, {}), vec3({}, {}, {}), {r0}, {r1}, {b}, {pr});", a.0, a.1, a.2, c.0, c.1, c.2));
            }
            "box" => {
                let (x, y, z) = (self.p(&n, "x"), self.p(&n, "y"), self.p(&n, "z"));
                let (sx, sy, sz, yaw) = (self.p(&n, "sx"), self.p(&n, "sy"), self.p(&n, "sz"), self.p(&n, "yaw"));
                let (b, pr) = self.block_prio(&n);
                self.stmt(format!("pb_box(vec3({x}, {y}, {z}), vec3({sx}, {sy}, {sz}), {yaw}, {b}, {pr});"));
            }
            "cylinder" => {
                let (x, y, z, h) = (self.p(&n, "x"), self.p(&n, "y"), self.p(&n, "z"), self.p(&n, "height"));
                let (r0, r1, var) = (self.p(&n, "r0"), self.p(&n, "r1"), self.p(&n, "var"));
                let (b, pr) = self.block_prio(&n);
                let rv = self.rnd(&n, 0);
                let v = self.fresh("v");
                self.stmt(format!("float {v} = mix(1.0 - {var}, 1.0 + {var}, {rv});"));
                self.stmt(format!("pb_cylinder(vec3({x}, {y}, {z}), vec3({x}, {y} + {h} * {v}, {z}), {r0} * {v}, {r1} * {v}, {b}, {pr});"));
            }
            "blob" => {
                let (x, y, z, r, var) = (self.p(&n, "x"), self.p(&n, "y"), self.p(&n, "z"), self.p(&n, "radius"), self.p(&n, "var"));
                let (sq, bumps, bs) = (self.p(&n, "squash"), self.p(&n, "bumps"), self.p(&n, "bump_size"));
                let (b, pr) = self.block_prio(&n);
                let (rv, yv) = (self.rnd(&n, 0), self.rnd(&n, 1));
                let s = self.fresh("s");
                self.stmt(format!("uint {s} = gPrimN;"));
                self.stmt(format!(
                    "pb_ellipsoid(vec3({x}, {y}, {z}), vec3(1.0, 1.0 / max({sq}, 0.1), 1.0) * {r} * mix(1.0 - {var}, 1.0 + {var}, {rv}), {yv} * 360.0, {b}, {pr});"
                ));
                self.stmt(format!("pb_displace({s}, {bumps}, {bs}, {}u);", n.id));
            }
            "union" => {
                for i in 0..4u8 {
                    if let Some(l) = self.g.input_link(n.id, i).copied() {
                        self.shape(l.from_node, l.from_pin);
                    }
                }
            }
            "subtract" | "intersect" => {
                self.shape_in(&n, 0);
                let s = self.fresh("s");
                self.stmt(format!("uint {s} = gPrimN;"));
                self.shape_in(&n, 1);
                self.stmt(format!("pb_set_op({s}, {});", if n.kind == "subtract" { "OP_REMOVE" } else { "OP_KEEP" }));
            }
            "smooth_union" => {
                let k = self.p(&n, "smoothness");
                let s = self.fresh("s");
                self.stmt(format!("uint {s} = gPrimN;"));
                self.shape_in(&n, 0);
                self.shape_in(&n, 1);
                self.stmt(format!("pb_smooth({s}, {k});"));
            }
            "displace" | "holes" | "set_block" | "carve" => {
                let s = self.fresh("s");
                self.stmt(format!("uint {s} = gPrimN;"));
                self.shape_in(&n, 0);
                match n.kind.as_str() {
                    "displace" => {
                        let (a, sz, seed) = (self.p(&n, "amount"), self.p(&n, "size"), self.pu(&n, "seed"));
                        self.stmt(format!("pb_displace({s}, {a}, {sz}, {seed} * 7919u + {}u);", n.id));
                    }
                    "holes" => {
                        let (a, seed) = (self.p(&n, "amount"), self.pu(&n, "seed"));
                        self.stmt(format!("pb_holes({s}, {a}, {seed} * 7919u + {}u);", n.id));
                    }
                    "set_block" => {
                        let (b, pr) = self.block_prio(&n);
                        let b2 = self.block(&n, "block2");
                        let mix = self.p(&n, "mix");
                        self.stmt(format!("pb_set_block({s}, {b}, {pr}, {b2}, {mix});"));
                    }
                    _ => self.stmt(format!("pb_set_op({s}, OP_CARVE);")),
                }
            }
            "transform" => {
                let (x, y, z) = (self.p(&n, "x"), self.p(&n, "y"), self.p(&n, "z"));
                let (yaw, pitch, roll, sc) = (self.p(&n, "yaw"), self.p(&n, "pitch"), self.p(&n, "roll"), self.p(&n, "scale"));
                let scale_in = self.input(&n, 1, 0).map(|v| v.e).unwrap_or_else(|| "1.0".into());
                let yaw_in = self.input(&n, 2, 0).map(|v| v.e).unwrap_or_else(|| "0.0".into());
                let ry = if bool_param(&n, "random_yaw") { format!(" + {} * 360.0", self.rnd(&n, 0)) } else { String::new() };
                let v = self.fresh("xf");
                self.stmt(format!("Xf {v} = gXf;"));
                self.stmt(format!("gXf = xf_mul(gXf, xf_make(vec3({x}, {y}, {z}), vec3({yaw} + {yaw_in}{ry}, {pitch}, {roll}), {sc} * {scale_in}));"));
                self.shape_in(&n, 0);
                self.stmt(format!("gXf = {v};"));
            }
            "mirror" => {
                let axes: &[[f32; 3]] = match self.enum_idx(&n, "axis") {
                    0 => &[[-1.0, 1.0, 1.0]],
                    1 => &[[1.0, 1.0, -1.0]],
                    _ => &[[-1.0, 1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0]],
                };
                let v = self.fresh("xf");
                self.stmt(format!("Xf {v} = gXf;"));
                self.shape_in(&n, 0);
                for a in axes {
                    self.stmt(format!("gXf = xf_mul({v}, xf_mirror(vec3({:.1}, {:.1}, {:.1})));", a[0], a[1], a[2]));
                    self.shape_in(&n, 0);
                }
                self.stmt(format!("gXf = {v};"));
            }
            "repeat" => {
                let (count, radius, jit, ss) = (self.p(&n, "count"), self.p(&n, "radius"), self.p(&n, "jitter"), self.p(&n, "scale_step"));
                let (sx, sy, sz) = (self.p(&n, "step_x"), self.p(&n, "step_y"), self.p(&n, "step_z"));
                let ring = self.enum_idx(&n, "mode") == 0;
                let ry = bool_param(&n, "random_yaw");
                let (v, vi, it, cnt) = (self.fresh("xf"), self.fresh("inst"), self.fresh("i"), self.fresh("n"));
                self.stmt(format!("Xf {v} = gXf;"));
                self.stmt(format!("uint {vi} = gInst;"));
                self.stmt(format!("int {cnt} = clamp(int({count} + 0.5), 1, 32);"));
                self.stmt(format!("for (int {it} = 0; {it} < {cnt}; ++{it}) {{"));
                self.stmt(format!("gInst = hash_u32({vi} ^ (uint({it}) * 0x9E3779B9u + {}u));", n.id));
                let (j1, j2, j3, j4) = (self.rnd(&n, 1), self.rnd(&n, 2), self.rnd(&n, 3), self.rnd(&n, 4));
                let ang = format!("(float({it}) * 6.2831853 / float({cnt}))");
                let off = if ring { format!("(rot_y({ang}) * vec3({radius}, 0.0, 0.0))") } else { format!("(vec3({sx}, {sy}, {sz}) * float({it}))") };
                let yaw = match (ry, ring) {
                    (true, _) => format!("{j4} * 360.0"),
                    (false, true) => format!("degrees({ang})"),
                    _ => "0.0".into(),
                };
                self.stmt(format!(
                    "gXf = xf_mul({v}, xf_make({off} + (vec3({j1}, {j2}, {j3}) - 0.5) * 2.0 * {jit}, vec3({yaw}, 0.0, 0.0), pow({ss}, float({it}))));"
                ));
                // values and skeletons made inside the loop body are local to it
                let (m3, sk) = (self.memo3.clone(), self.skel.clone());
                self.shape_in(&n, 0);
                self.memo3 = m3;
                self.skel = sk;
                self.stmt("}".into());
                self.stmt(format!("gXf = {v};"));
                self.stmt(format!("gInst = {vi};"));
            }
            "trunk" | "branches" => {
                let sk = self.skeleton(n.id, 1);
                let (b, pr) = self.block_prio(&n);
                // the wood: the new segments only (branches start with a copy of their parent)
                self.stmt(format!("tree_wood({sk}.z, {sk}.y, {b}, {pr});"));
                let _ = pin;
            }
            "leaves" => {
                let sk = self.skel_in(&n, 0);
                let (shape, at) = (self.enum_idx(&n, "shape"), self.enum_idx(&n, "at"));
                let ps: Vec<String> = ["radius", "var", "y_scale", "y_offset", "start", "spacing", "top_scale", "density", "bumps"].iter().map(|p| self.p(&n, p)).collect();
                let (b, pr) = self.block_prio(&n);
                let key = self.nkey(&n);
                self.stmt(format!("tree_leaves({sk}.x, {sk}.y, {shape}u, {at}u, {}, {b}, {pr}, {key});", ps.join(", ")));
            }
            "roots" => {
                let sk = self.skel_in(&n, 0);
                let (c, l, d, r) = (self.pu(&n, "count"), self.p(&n, "length"), self.p(&n, "depth"), self.p(&n, "radius"));
                let (b, pr) = self.block_prio(&n);
                let key = self.nkey(&n);
                self.stmt(format!("tree_roots({sk}.x, {sk}.y, {c}, {l}, {d}, {r}, {b}, {pr}, {key});"));
            }
            "vines" => {
                let sk = self.skel_in(&n, 0);
                let (c, s, l0, l1, yo) = (self.pu(&n, "count"), self.p(&n, "spread"), self.p(&n, "length_min"), self.p(&n, "length_max"), self.p(&n, "y_offset"));
                let (b, pr) = self.block_prio(&n);
                let key = self.nkey(&n);
                self.stmt(format!("tree_vines({sk}.x, {sk}.y, {c}, {s}, {l0}, {l1}, {yo}, {b}, {pr}, {key});"));
            }
            "fruit" => {
                let sk = self.skel_in(&n, 0);
                let (ch, per, sp) = (self.p(&n, "chance"), self.pu(&n, "per_tip"), self.p(&n, "spread"));
                let (b, pr) = self.block_prio(&n);
                let key = self.nkey(&n);
                self.stmt(format!("tree_fruit({sk}.x, {sk}.y, {ch}, {per}, {sp}, {b}, {pr}, {key});"));
            }
            other => self.warn(format!("node '{other}' cannot produce a shape")),
        }
        self.depth -= 1;
    }

    fn shape_in(&mut self, n: &Node, pin: u8) {
        if let Some(l) = self.g.input_link(n.id, pin).copied() {
            self.shape(l.from_node, l.from_pin);
        }
    }

    fn skel_in(&mut self, n: &Node, pin: u8) -> String {
        match self.g.input_link(n.id, pin).copied() {
            Some(l) => self.skeleton(l.from_node, l.from_pin),
            None => "uvec3(0u)".into(),
        }
    }

    /// Grow the skeleton of node `id` (once per scope). Returns a `uvec3` variable: its segments
    /// [x, y) and where the node's own segments start (z).
    pub(super) fn skeleton(&mut self, id: u32, pin: u8) -> String {
        if let Some(v) = self.skel.get(&id) {
            return v.clone();
        }
        let Some(node) = self.g.node(id) else { return "uvec3(0u)".into() };
        let n = node.clone();
        let v = self.fresh("sk");
        match n.kind.as_str() {
            "trunk" => {
                let names = ["height_min", "height_max", "radius", "taper", "taper_curve", "flare", "bend", "twist", "sink"];
                let ps: Vec<String> = names.iter().map(|p| self.p(&n, p)).collect();
                let segs = self.pu(&n, "segments");
                let key = self.nkey(&n);
                let s = self.fresh("s");
                self.stmt(format!("uint {s} = tree_trunk({}, {segs}, {key});", ps.join(", ")));
                self.stmt(format!("uvec3 {v} = uvec3({s}, gSegN, {s});"));
            }
            "branches" => {
                let parent = self.skel_in(&n, 0);
                let (count, levels) = (self.pu(&n, "count"), self.pu(&n, "levels"));
                let names = ["start", "end", "angle", "length", "falloff", "radius", "droop"];
                let ps: Vec<String> = names.iter().map(|p| self.p(&n, p)).collect();
                let elbow = bool_param(&n, "elbow");
                let key = self.nkey(&n);
                let s = self.fresh("s");
                self.stmt(format!("uint {s} = tree_branches({parent}.x, {parent}.y, {count}, {levels}, {}, {elbow}, {key});", ps.join(", ")));
                self.stmt(format!("uvec3 {v} = uvec3({s}, gSegN, {s} + ({parent}.y - {parent}.x));"));
            }
            other => {
                self.warn(format!("node '{other}' has no skeleton"));
                return "uvec3(0u)".into();
            }
        }
        let _ = pin;
        self.skel.insert(id, v.clone());
        v
    }
}

/// The build function of every prop graph and the dispatcher, for the props pass only.
pub(super) fn compile_props(sh: &mut Shared, names: &[String], out: &mut String) {
    use std::fmt::Write as _;
    let _ = writeln!(out, "#define NUM_PROP_GRAPHS {}u\n#ifdef PROPS_PASS", names.len());
    for (k, name) in names.iter().enumerate() {
        let g = sh.proj.get(crate::graph::GraphKind::Prop, name).unwrap();
        let mut em = Em::new(sh, g, format!("p{k}_"), true);
        if let Some(o) = g.output_node()
            && let Some(l) = g.input_link(o.id, 0).copied()
        {
            em.shape(l.from_node, l.from_pin);
        } else {
            em.warn("the Prop Output has no shape".into());
        }
        let body = em.take_body().join("\n");
        let _ = writeln!(out, "void prop{k}_build() {{\n    vec3 p = gAnchorP;\n{body}\n}}\n");
    }
    let _ = writeln!(out, "void prop_build(uint g) {{\n    switch (g) {{");
    for k in 0..names.len() {
        let _ = writeln!(out, "        case {k}u: prop{k}_build(); break;");
    }
    let _ = writeln!(out, "        default: break;\n    }}\n}}\n#endif");
}

/// Prop graph name -> index (the order of `Project::names`).
pub(super) fn prop_ids(names: &[String]) -> HashMap<String, u32> {
    names.iter().enumerate().map(|(i, n)| (n.clone(), i as u32)).collect()
}
