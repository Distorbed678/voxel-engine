//! The built-in example world in three layers: the surface (4 zones: temperate, arid, frozen,
//! volcanic, with 11 biomes and 4 cave biomes), the underground in two depth bands (placeholder
//! caverns, magma depths with a lava sea) and the sky (floating isles above the cloud sea), and
//! the Prop graphs the biomes scatter (trees, rocks, spikes, geodes).
//! Everything here is ordinary node graphs - open them in the node editor and change anything.

use crate::graph::{Graph, GraphKind, Project, Value};

struct B {
    g: Graph,
}

fn fv(v: f32) -> Value {
    Value::F(v)
}
fn iv(v: i32) -> Value {
    Value::I(v)
}
fn bk(s: &str) -> Value {
    Value::Block(s.into())
}
fn ev(s: &str) -> Value {
    Value::E(s.into())
}
fn rv(s: &str) -> Value {
    Value::Ref(s.into())
}

impl B {
    fn new(name: &str, kind: GraphKind) -> Self {
        Self { g: Graph::new(name, kind) }
    }
    fn out(&self) -> u32 {
        self.g.output_node().unwrap().id
    }
    fn n(&mut self, kind: &str, x: f32, y: f32, params: &[(&str, Value)]) -> u32 {
        let id = self.g.add(kind, [x, y]);
        for (k, v) in params {
            self.g.set(id, k, v.clone());
        }
        id
    }
    fn l(&mut self, from: u32, to: u32, pin: u8) {
        let ok = self.g.connect(from, 0, to, pin);
        debug_assert!(ok, "bad default link {from}->{to}:{pin} in {}", self.g.name);
    }
    /// link from output pin `out` (e.g. a trunk's Skeleton)
    fn l2(&mut self, from: u32, out: u8, to: u32, pin: u8) {
        let ok = self.g.connect(from, out, to, pin);
        debug_assert!(ok, "bad default link {from}:{out}->{to}:{pin} in {}", self.g.name);
    }
    /// a Prop node for a biome's Scatter
    fn prop(&mut self, name: &str, x: f32, y: f32) -> u32 {
        self.n("prop", x, y, &[("prop", rv(name))])
    }
    fn out_params(&mut self, params: &[(&str, Value)]) {
        let o = self.out();
        for (k, v) in params {
            self.g.set(o, k, v.clone());
        }
    }
    fn noise(&mut self, kind: &str, x: f32, y: f32, size: f32, oct: i32, amp: f32, seed: i32) -> u32 {
        self.n(kind, x, y, &[("size", fv(size)), ("octaves", iv(oct)), ("amplitude", fv(amp)), ("seed", iv(seed))])
    }
    /// height (2D) -> density, returns the density node
    fn terrain(&mut self, height: u32, base: f32, x: f32, y: f32) -> u32 {
        let t = self.n("height_terrain", x, y, &[("base", fv(base))]);
        self.l(height, t, 0);
        t
    }
    /// stone with ores: coal, iron, gold, copper. Returns the material node.
    fn ores(&mut self, base: u32, x: f32, y: f32) -> u32 {
        let coal = self.n("ore", x, y, &[("ore", bk("coal_ore")), ("size", fv(6.0)), ("rarity", fv(0.78)), ("y_max", fv(160.0)), ("seed", iv(1))]);
        self.l(base, coal, 0);
        let iron = self.n("ore", x + 220.0, y, &[("ore", bk("iron_ore")), ("size", fv(4.5)), ("rarity", fv(0.82)), ("y_max", fv(64.0)), ("seed", iv(2))]);
        self.l(coal, iron, 0);
        let copper = self.n("ore", x + 440.0, y, &[("ore", bk("copper_ore")), ("size", fv(5.0)), ("rarity", fv(0.84)), ("y_min", fv(0.0)), ("y_max", fv(96.0)), ("seed", iv(4))]);
        self.l(iron, copper, 0);
        let gold = self.n("ore", x + 660.0, y, &[("ore", bk("gold_ore")), ("size", fv(3.5)), ("rarity", fv(0.86)), ("y_max", fv(10.0)), ("seed", iv(3))]);
        self.l(copper, gold, 0);
        gold
    }
    fn deep(&mut self, x: f32, y: f32) -> u32 {
        // stone, deepslate below y = -32
        let stone = self.n("block", x, y, &[("block", bk("stone"))]);
        let deep = self.n("block", x, y + 80.0, &[("block", bk("deepslate"))]);
        let hs = self.n("height_select", x + 200.0, y, &[("height", fv(-32.0)), ("noise_amount", fv(6.0))]);
        let nz = self.noise("simplex2d", x, y + 160.0, 40.0, 2, 1.0, 9);
        self.l(deep, hs, 0);
        self.l(stone, hs, 1);
        self.l(nz, hs, 2);
        hs
    }
    fn scatter(&mut self, prop: u32, x: f32, y: f32, params: &[(&str, Value)], slot: u8, mask: Option<u32>) {
        let s = self.n("scatter", x, y, params);
        self.l(prop, s, 0);
        if let Some(m) = mask {
            self.l(m, s, 1);
        }
        let o = self.out();
        self.l(s, o, slot);
    }
}

fn surface(b: &mut B, top: &str, under: &str, underwater: &str, under_depth: f32, x: f32, y: f32, base: u32) -> u32 {
    let s = b.n(
        "surface_layers",
        x,
        y,
        &[("top", bk(top)), ("under", bk(under)), ("underwater", bk(underwater)), ("under_depth", fv(under_depth)), ("cave_floor", bk("gravel"))],
    );
    b.l(base, s, 0);
    s
}

fn meadow() -> Graph {
    let mut b = B::new("meadow", GraphKind::Biome);
    let h = b.noise("simplex2d", 0.0, 0.0, 320.0, 4, 9.0, 1);
    let t = b.terrain(h, 69.0, 260.0, 0.0);
    let o = b.out();
    b.l(t, o, 0);
    let st = b.deep(0.0, 200.0);
    let ores = b.ores(st, 260.0, 220.0);
    let s = surface(&mut b, "grass", "dirt", "sand", 3.0, 300.0, 360.0, ores);
    b.l(s, o, 1);
    let flowers = b.noise("simplex2d", 0.0, 480.0, 60.0, 2, 1.0, 7);
    let gc = b.n("ground_cover", 260.0, 500.0, &[("chance", fv(0.45)), ("weight1", fv(10.0)), ("weight2", fv(1.5)), ("weight3", fv(1.5)), ("plant4", bk("flower_white")), ("weight4", fv(1.0))]);
    b.l(flowers, gc, 0);
    b.l(gc, o, 2);
    let oak = b.prop("oak", 0.0, 650.0);
    b.scatter(oak, 260.0, 650.0, &[("spacing", fv(28.0)), ("chance", fv(0.35))], 3, None);
    let rock = b.prop("boulder", 0.0, 800.0);
    b.scatter(rock, 260.0, 800.0, &[("spacing", fv(40.0)), ("chance", fv(0.25)), ("seed", iv(3))], 4, None);
    let ruin = b.n("prefab", 0.0, 950.0, &[("prefab", rv("ruin"))]);
    b.scatter(ruin, 260.0, 950.0, &[("spacing", fv(220.0)), ("chance", fv(0.35)), ("seed", iv(8))], 5, None);
    b.out_params(&[("grass_tint", Value::Color([0.5, 0.82, 0.32])), ("map_color", Value::Color([0.55, 0.8, 0.35]))]);
    b.g
}

fn oak_forest() -> Graph {
    let mut b = B::new("oak_forest", GraphKind::Biome);
    let h = b.noise("simplex2d", 0.0, 0.0, 260.0, 5, 13.0, 2);
    let t = b.terrain(h, 71.0, 260.0, 0.0);
    let detail = b.noise("simplex3d", 0.0, 140.0, 40.0, 2, 2.5, 3);
    let add = b.n("add", 480.0, 60.0, &[]);
    b.l(t, add, 0);
    b.l(detail, add, 1);
    let o = b.out();
    b.l(add, o, 0);
    let st = b.deep(0.0, 260.0);
    let ores = b.ores(st, 260.0, 280.0);
    let s = surface(&mut b, "grass", "dirt", "sand", 3.0, 300.0, 420.0, ores);
    // podzol patches
    let pod = b.n("block", 300.0, 520.0, &[("block", bk("podzol"))]);
    let sel = b.noise("simplex2d", 300.0, 600.0, 50.0, 2, 1.0, 11);
    let ns = b.n("noise_select", 560.0, 460.0, &[("threshold", fv(0.45))]);
    let ds = b.n("depth_select", 760.0, 460.0, &[("depth", fv(1.0))]);
    b.l(s, ns, 0);
    b.l(pod, ns, 1);
    b.l(sel, ns, 2);
    b.l(ns, ds, 0);
    b.l(s, ds, 1);
    b.l(ds, o, 1);
    let gc = b.n("ground_cover", 260.0, 720.0, &[("chance", fv(0.35)), ("plant2", bk("fern")), ("weight2", fv(4.0)), ("plant3", bk("mushroom_red")), ("weight3", fv(0.3)), ("weight4", fv(0.0))]);
    b.l(gc, o, 2);
    let forest_mask = b.n("simplex2d", 0.0, 860.0, &[("size", fv(180.0)), ("octaves", iv(3)), ("seed", iv(21))]);
    let mask = b.n("remap", 220.0, 860.0, &[("in_min", fv(-0.6)), ("in_max", fv(0.3)), ("out_min", fv(0.0)), ("out_max", fv(1.0)), ("clamp", Value::B(true))]);
    b.l(forest_mask, mask, 0);
    let oak = b.prop("oak", 0.0, 1000.0);
    b.scatter(oak, 260.0, 1000.0, &[("spacing", fv(8.0)), ("chance", fv(0.75))], 3, Some(mask));
    let birch = b.prop("birch", 0.0, 1150.0);
    b.scatter(birch, 260.0, 1150.0, &[("spacing", fv(13.0)), ("chance", fv(0.4)), ("seed", iv(5))], 4, Some(mask));
    let bush = b.prop("bush", 0.0, 1300.0);
    b.scatter(bush, 260.0, 1300.0, &[("spacing", fv(11.0)), ("chance", fv(0.3)), ("seed", iv(6))], 5, None);
    b.out_params(&[("grass_tint", Value::Color([0.38, 0.7, 0.26])), ("foliage_tint", Value::Color([0.3, 0.6, 0.2])), ("map_color", Value::Color([0.2, 0.55, 0.2]))]);
    b.g
}

fn birch_hills() -> Graph {
    let mut b = B::new("birch_hills", GraphKind::Biome);
    let r = b.noise("ridged2d", 0.0, 0.0, 420.0, 4, 34.0, 4);
    let s2 = b.noise("simplex2d", 0.0, 150.0, 120.0, 3, 6.0, 5);
    let add = b.n("add", 240.0, 60.0, &[]);
    b.l(r, add, 0);
    b.l(s2, add, 1);
    let t = b.terrain(add, 82.0, 460.0, 60.0);
    let o = b.out();
    b.l(t, o, 0);
    let st = b.deep(0.0, 300.0);
    let ores = b.ores(st, 260.0, 320.0);
    let s = surface(&mut b, "grass", "dirt", "gravel", 3.0, 300.0, 460.0, ores);
    let stone_top = b.n("block", 300.0, 560.0, &[("block", bk("stone"))]);
    let hs = b.n("height_select", 560.0, 460.0, &[("height", fv(128.0)), ("noise_amount", fv(10.0))]);
    let hn = b.noise("simplex2d", 300.0, 640.0, 30.0, 2, 1.0, 8);
    b.l(s, hs, 0);
    b.l(stone_top, hs, 1);
    b.l(hn, hs, 2);
    b.l(hs, o, 1);
    let gc = b.n("ground_cover", 260.0, 760.0, &[("chance", fv(0.3)), ("plant2", bk("flower_blue")), ("weight2", fv(1.0)), ("weight3", fv(0.0))]);
    b.l(gc, o, 2);
    let birch = b.prop("birch", 0.0, 900.0);
    b.scatter(birch, 260.0, 900.0, &[("spacing", fv(10.0)), ("chance", fv(0.55))], 3, None);
    let rock = b.prop("boulder_andesite", 0.0, 1050.0);
    b.scatter(rock, 260.0, 1050.0, &[("spacing", fv(30.0)), ("chance", fv(0.3)), ("seed", iv(9))], 4, None);
    b.out_params(&[("grass_tint", Value::Color([0.55, 0.78, 0.35])), ("foliage_tint", Value::Color([0.5, 0.72, 0.3])), ("map_color", Value::Color([0.6, 0.75, 0.4]))]);
    b.g
}

/// Sky layer: islands of 3D noise in a band around y = 4800, open air above and below.
fn floating_isles() -> Graph {
    let mut b = B::new("floating_isles", GraphKind::Biome);
    // big, thick islands: thin ones turn to dust at the coarse LODs the sky is seen at
    let blob = b.noise("simplex3d", 0.0, 160.0, 140.0, 3, 50.0, 12);
    let center = b.n("coord", 0.0, 400.0, &[("axis", ev("Y"))]);
    let dist = b.n("scale_bias", 200.0, 400.0, &[("scale", fv(1.0)), ("bias", fv(-4800.0))]);
    b.l(center, dist, 0);
    let ad = b.n("abs", 380.0, 400.0, &[]);
    b.l(dist, ad, 0);
    let falloff = b.n("scale_bias", 540.0, 400.0, &[("scale", fv(-0.6)), ("bias", fv(-25.0))]);
    b.l(ad, falloff, 0);
    let isl = b.n("add", 720.0, 250.0, &[]);
    b.l(blob, isl, 0);
    b.l(falloff, isl, 1);
    let o = b.out();
    b.l(isl, o, 0);
    let stone = b.n("block", 0.0, 560.0, &[("block", bk("stone"))]);
    let s = surface(&mut b, "grass", "dirt", "dirt", 3.0, 300.0, 560.0, stone);
    // islands are "open to sky" at their tops; their undersides get the base material
    b.l(s, o, 1);
    let gc = b.n("ground_cover", 260.0, 700.0, &[("chance", fv(0.5)), ("plant2", bk("flower_white")), ("weight2", fv(2.0))]);
    b.l(gc, o, 2);
    let tree = b.prop("willow", 0.0, 840.0);
    b.scatter(tree, 260.0, 840.0, &[("spacing", fv(16.0)), ("chance", fv(0.5)), ("y_min", fv(4000.0)), ("y_max", fv(6200.0))], 3, None);
    b.out_params(&[("fluid", ev("None")), ("grass_tint", Value::Color([0.42, 0.85, 0.45])), ("fog_color", Value::Color([0.8, 0.85, 1.0])), ("map_color", Value::Color([0.4, 0.85, 0.6]))]);
    b.g
}

/// An underground placeholder biome: solid rock with big caverns and tunnels between y_min and
/// y_max, floors of `floor`, walls of `base`, ground cover on the floors. Phase G replaces these.
fn underground_biome(name: &str, base: &str, floor: &str, plants: &[(&str, f32)], fluid: &str, y: (f32, f32), map: [f32; 3]) -> Graph {
    let mut b = B::new(name, GraphKind::Biome);
    let solid = b.n("constant", 0.0, 0.0, &[("value", fv(1.0))]);
    // caverns at every LOD: they are what the underground is
    let ch = b.n("cheese_caves", 0.0, 120.0, &[("size", fv(110.0)), ("y_min", fv(y.0)), ("y_max", fv(y.1)), ("max_lod", iv(8))]);
    let sp = b.n("spaghetti_caves", 0.0, 260.0, &[("y_min", fv(y.0)), ("y_max", fv(y.1)), ("max_lod", iv(8))]);
    let caves = b.n("max", 220.0, 180.0, &[]);
    b.l(ch, caves, 0);
    b.l(sp, caves, 1);
    let k = b.n("scale_bias", 400.0, 180.0, &[("scale", fv(4.0)), ("bias", fv(0.0))]);
    b.l(caves, k, 0);
    let d = b.n("subtract", 580.0, 60.0, &[]);
    b.l(solid, d, 0);
    b.l(k, d, 1);
    let o = b.out();
    b.l(d, o, 0);
    let base_n = b.n("block", 0.0, 420.0, &[("block", bk(base))]);
    let s = b.n("surface_layers", 220.0, 420.0, &[("top", bk(floor)), ("under", bk(base)), ("underwater", bk(base)), ("under_depth", fv(1.0)), ("cave_floor", bk(floor))]);
    b.l(base_n, s, 0);
    b.l(s, o, 1);
    let mut gc_params = vec![("on", bk(floor)), ("chance", fv(0.35))];
    let names = ["plant1", "plant2", "plant3", "plant4"];
    let weights = ["weight1", "weight2", "weight3", "weight4"];
    for i in 0..4 {
        match plants.get(i) {
            Some((pl, w)) => {
                gc_params.push((names[i], bk(pl)));
                gc_params.push((weights[i], fv(*w)));
            }
            None => gc_params.push((weights[i], fv(0.0))),
        }
    }
    let gc = b.n("ground_cover", 220.0, 560.0, &gc_params);
    b.l(gc, o, 2);
    b.out_params(&[("fluid", ev(fluid)), ("fog_color", Value::Color([0.08, 0.08, 0.1])), ("map_color", Value::Color(map))]);
    b.g
}

/// A zone with a single biome and no cave carver (sky and underground layers).
fn single_zone(name: &str, biome: &str, color: [f32; 3]) -> Graph {
    let mut b = B::new(name, GraphKind::Zone);
    let n = b.n("biome", 0.0, 0.0, &[("biome", rv(biome))]);
    let o = b.out();
    b.l(n, o, 0);
    b.out_params(&[("map_color", Value::Color(color))]);
    b.g
}

fn desert() -> Graph {
    let mut b = B::new("desert", GraphKind::Biome);
    let dunes = b.noise("ridged2d", 0.0, 0.0, 150.0, 2, 7.0, 10);
    let swell = b.noise("simplex2d", 0.0, 150.0, 500.0, 3, 8.0, 11);
    let add = b.n("add", 240.0, 60.0, &[]);
    b.l(dunes, add, 0);
    b.l(swell, add, 1);
    let t = b.terrain(add, 70.0, 460.0, 60.0);
    let o = b.out();
    b.l(t, o, 0);
    let ss = b.n("block", 0.0, 300.0, &[("block", bk("sandstone"))]);
    let st = b.deep(0.0, 400.0);
    let ores = b.ores(st, 260.0, 420.0);
    let ds = b.n("depth_select", 300.0, 300.0, &[("depth", fv(12.0))]);
    b.l(ss, ds, 0);
    b.l(ores, ds, 1);
    let s = surface(&mut b, "sand", "sand", "sand", 4.0, 520.0, 300.0, ds);
    b.l(s, o, 1);
    let gc = b.n("ground_cover", 260.0, 600.0, &[("on", bk("sand")), ("chance", fv(0.02)), ("plant1", bk("dead_bush")), ("weight1", fv(4.0)), ("plant2", bk("pebbles")), ("weight2", fv(1.0)), ("weight3", fv(0.0)), ("weight4", fv(0.0))]);
    b.l(gc, o, 2);
    let cactus = b.prop("cactus", 0.0, 760.0);
    b.scatter(cactus, 260.0, 760.0, &[("spacing", fv(14.0)), ("chance", fv(0.35)), ("surface", bk("sand"))], 3, None);
    let palm = b.prop("palm", 0.0, 900.0);
    let oasis = b.noise("simplex2d", 0.0, 1040.0, 200.0, 2, 1.0, 13);
    let om = b.n("remap", 220.0, 1040.0, &[("in_min", fv(0.45)), ("in_max", fv(0.6)), ("out_min", fv(0.0)), ("out_max", fv(1.0)), ("clamp", Value::B(true))]);
    b.l(oasis, om, 0);
    b.scatter(palm, 260.0, 900.0, &[("spacing", fv(12.0)), ("chance", fv(0.8)), ("surface", bk("sand")), ("seed", iv(4))], 4, Some(om));
    b.out_params(&[("grass_tint", Value::Color([0.75, 0.75, 0.4])), ("fog_color", Value::Color([0.95, 0.85, 0.65])), ("fog_density", fv(1.6)), ("map_color", Value::Color([0.93, 0.85, 0.55]))]);
    b.g
}

fn badlands() -> Graph {
    let mut b = B::new("badlands", GraphKind::Biome);
    let h = b.noise("simplex2d", 0.0, 0.0, 380.0, 4, 1.0, 14);
    let c = b.n("curve", 220.0, 0.0, &[("curve", Value::Curve(vec![[-1.0, -6.0], [-0.1, 0.0], [0.15, 26.0], [0.5, 34.0], [1.0, 58.0]]))]);
    b.l(h, c, 0);
    let ter = b.n("terrace", 420.0, 0.0, &[("step", fv(9.0)), ("smoothness", fv(0.25))]);
    b.l(c, ter, 0);
    let t = b.terrain(ter, 70.0, 620.0, 0.0);
    let o = b.out();
    b.l(t, o, 0);
    let st = b.deep(0.0, 260.0);
    let strata = b.n("strata", 260.0, 260.0, &[]);
    b.l(st, strata, 0);
    let s = surface(&mut b, "red_sand", "terracotta_orange", "red_sand", 1.0, 520.0, 260.0, strata);
    b.l(s, o, 1);
    let gc = b.n("ground_cover", 260.0, 500.0, &[("on", bk("red_sand")), ("chance", fv(0.03)), ("plant1", bk("dead_bush")), ("weight2", fv(0.0)), ("weight3", fv(0.0)), ("weight4", fv(0.0))]);
    b.l(gc, o, 2);
    let rock = b.prop("boulder_terracotta", 0.0, 640.0);
    b.scatter(rock, 260.0, 640.0, &[("spacing", fv(24.0)), ("chance", fv(0.3)), ("any_surface", Value::B(true))], 3, None);
    b.out_params(&[("grass_tint", Value::Color([0.7, 0.62, 0.35])), ("fog_color", Value::Color([0.95, 0.75, 0.6])), ("map_color", Value::Color([0.8, 0.45, 0.25]))]);
    b.g
}

fn savanna() -> Graph {
    let mut b = B::new("savanna", GraphKind::Biome);
    let h = b.noise("simplex2d", 0.0, 0.0, 420.0, 4, 8.0, 15);
    let plateau = b.noise("simplex2d", 0.0, 150.0, 700.0, 2, 1.0, 16);
    let pc = b.n("curve", 220.0, 150.0, &[("curve", Value::Curve(vec![[-1.0, 0.0], [0.35, 0.0], [0.45, 22.0], [1.0, 26.0]]))]);
    b.l(plateau, pc, 0);
    let add = b.n("add", 440.0, 60.0, &[]);
    b.l(h, add, 0);
    b.l(pc, add, 1);
    let t = b.terrain(add, 69.0, 640.0, 60.0);
    let o = b.out();
    b.l(t, o, 0);
    let st = b.deep(0.0, 300.0);
    let ores = b.ores(st, 260.0, 320.0);
    let s = surface(&mut b, "dry_grass", "coarse_dirt", "sand", 2.0, 300.0, 460.0, ores);
    b.l(s, o, 1);
    let gc = b.n("ground_cover", 260.0, 600.0, &[("on", bk("dry_grass")), ("chance", fv(0.35)), ("plant1", bk("tall_grass")), ("weight1", fv(6.0)), ("plant2", bk("dry_shrub")), ("weight2", fv(3.0)), ("weight3", fv(0.3)), ("weight4", fv(0.0))]);
    b.l(gc, o, 2);
    let acacia = b.prop("acacia", 0.0, 760.0);
    b.scatter(acacia, 260.0, 760.0, &[("spacing", fv(24.0)), ("chance", fv(0.45)), ("surface", bk("dry_grass"))], 3, None);
    b.out_params(&[("grass_tint", Value::Color([0.75, 0.72, 0.38])), ("foliage_tint", Value::Color([0.6, 0.65, 0.25])), ("fog_color", Value::Color([0.95, 0.88, 0.7])), ("map_color", Value::Color([0.75, 0.7, 0.35]))]);
    b.g
}

fn taiga() -> Graph {
    let mut b = B::new("taiga", GraphKind::Biome);
    let h = b.noise("simplex2d", 0.0, 0.0, 280.0, 5, 15.0, 17);
    let t = b.terrain(h, 73.0, 260.0, 0.0);
    let o = b.out();
    b.l(t, o, 0);
    let st = b.deep(0.0, 200.0);
    let ores = b.ores(st, 260.0, 220.0);
    let s = surface(&mut b, "snowy_grass", "dirt", "gravel", 3.0, 300.0, 360.0, ores);
    let pod = b.n("block", 300.0, 460.0, &[("block", bk("podzol"))]);
    let sel = b.noise("simplex2d", 300.0, 540.0, 40.0, 2, 1.0, 18);
    let ns = b.n("noise_select", 560.0, 400.0, &[("threshold", fv(0.35))]);
    let ds = b.n("depth_select", 760.0, 400.0, &[("depth", fv(1.0))]);
    b.l(s, ns, 0);
    b.l(pod, ns, 1);
    b.l(sel, ns, 2);
    b.l(ns, ds, 0);
    b.l(s, ds, 1);
    b.l(ds, o, 1);
    let gc = b.n("ground_cover", 260.0, 680.0, &[("on", bk("snowy_grass")), ("chance", fv(0.18)), ("plant1", bk("snow_shrub")), ("weight1", fv(3.0)), ("plant2", bk("fern")), ("weight2", fv(2.0)), ("weight3", fv(0.0)), ("weight4", fv(0.0))]);
    b.l(gc, o, 2);
    let spruce = b.prop("spruce", 0.0, 820.0);
    b.scatter(spruce, 260.0, 820.0, &[("spacing", fv(7.0)), ("chance", fv(0.6)), ("any_surface", Value::B(true))], 3, None);
    let geode = b.prop("geode", 0.0, 980.0);
    b.scatter(geode, 260.0, 980.0, &[("spacing", fv(96.0)), ("chance", fv(0.5)), ("jitter", fv(0.8)), ("y_min", fv(-100.0)), ("y_max", fv(20.0)), ("placement", ev("Buried"))], 4, None);
    b.out_params(&[("grass_tint", Value::Color([0.45, 0.62, 0.45])), ("foliage_tint", Value::Color([0.3, 0.5, 0.35])), ("fog_color", Value::Color([0.75, 0.82, 0.9])), ("map_color", Value::Color([0.3, 0.5, 0.4]))]);
    b.g
}

fn glacier() -> Graph {
    let mut b = B::new("glacier", GraphKind::Biome);
    let h = b.noise("simplex2d", 0.0, 0.0, 500.0, 3, 5.0, 19);
    let t = b.terrain(h, 66.0, 260.0, 0.0);
    let o = b.out();
    b.l(t, o, 0);
    let st = b.deep(0.0, 200.0);
    let s = surface(&mut b, "snow", "packed_ice", "packed_ice", 6.0, 300.0, 200.0, st);
    b.l(s, o, 1);
    let spike = b.prop("ice_spike", 0.0, 400.0);
    b.scatter(spike, 260.0, 400.0, &[("spacing", fv(26.0)), ("chance", fv(0.35)), ("any_surface", Value::B(true))], 3, None);
    let ice = b.prop("boulder_ice", 0.0, 560.0);
    b.scatter(ice, 260.0, 560.0, &[("spacing", fv(30.0)), ("chance", fv(0.3)), ("any_surface", Value::B(true)), ("seed", iv(2))], 4, None);
    b.out_params(&[("fluid", ev("Ice")), ("water_tint", Value::Color([0.3, 0.5, 0.8])), ("fog_color", Value::Color([0.85, 0.9, 1.0])), ("fog_density", fv(1.8)), ("map_color", Value::Color([0.85, 0.92, 1.0]))]);
    b.g
}

fn snowy_peaks() -> Graph {
    let mut b = B::new("snowy_peaks", GraphKind::Biome);
    let r = b.noise("ridged2d", 0.0, 0.0, 520.0, 6, 130.0, 20);
    let s2 = b.noise("simplex2d", 0.0, 150.0, 900.0, 2, 30.0, 21);
    let add = b.n("add", 240.0, 60.0, &[]);
    b.l(r, add, 0);
    b.l(s2, add, 1);
    let t = b.terrain(add, 96.0, 460.0, 60.0);
    // 3D overhangs / cliffs
    let over = b.noise("simplex3d", 240.0, 220.0, 48.0, 3, 14.0, 22);
    let wx = b.noise("simplex3d", 0.0, 320.0, 90.0, 2, 1.0, 23);
    let wz = b.noise("simplex3d", 0.0, 420.0, 90.0, 2, 1.0, 24);
    let warp = b.n("warp", 460.0, 260.0, &[("strength", fv(10.0))]);
    b.l(over, warp, 0);
    b.l(wx, warp, 1);
    b.l(wz, warp, 2);
    let sum = b.n("add", 680.0, 100.0, &[]);
    b.l(t, sum, 0);
    b.l(warp, sum, 1);
    let o = b.out();
    b.l(sum, o, 0);
    let st = b.deep(0.0, 560.0);
    let ores = b.ores(st, 260.0, 580.0);
    let s = surface(&mut b, "snowy_grass", "dirt", "gravel", 2.0, 300.0, 700.0, ores);
    let snow = b.n("block", 300.0, 800.0, &[("block", bk("snow"))]);
    let stone = b.n("block", 300.0, 880.0, &[("block", bk("stone"))]);
    let rocky = b.n("height_select", 560.0, 700.0, &[("height", fv(125.0)), ("noise_amount", fv(12.0))]);
    let hn = b.noise("simplex2d", 300.0, 960.0, 35.0, 2, 1.0, 25);
    b.l(s, rocky, 0);
    b.l(stone, rocky, 1);
    b.l(hn, rocky, 2);
    let snowcap = b.n("height_select", 780.0, 760.0, &[("height", fv(175.0)), ("noise_amount", fv(16.0))]);
    b.l(rocky, snowcap, 0);
    b.l(snow, snowcap, 1);
    b.l(hn, snowcap, 2);
    let ds = b.n("depth_select", 980.0, 760.0, &[("depth", fv(2.0))]);
    b.l(snowcap, ds, 0);
    b.l(rocky, ds, 1);
    b.l(ds, o, 1);
    let spruce = b.prop("spruce", 0.0, 1080.0);
    b.scatter(spruce, 260.0, 1080.0, &[("spacing", fv(11.0)), ("chance", fv(0.5)), ("surface", bk("snowy_grass")), ("y_max", fv(140.0))], 3, None);
    let geode = b.prop("geode", 0.0, 1220.0);
    b.scatter(geode, 260.0, 1220.0, &[("spacing", fv(96.0)), ("chance", fv(0.5)), ("jitter", fv(0.8)), ("y_min", fv(-100.0)), ("y_max", fv(60.0)), ("placement", ev("Buried"))], 4, None);
    b.out_params(&[("grass_tint", Value::Color([0.5, 0.65, 0.5])), ("fog_color", Value::Color([0.8, 0.86, 0.95])), ("map_color", Value::Color([0.95, 0.95, 0.98]))]);
    b.g
}

fn ashlands() -> Graph {
    let mut b = B::new("ashlands", GraphKind::Biome);
    let h = b.noise("billow2d", 0.0, 0.0, 220.0, 4, 12.0, 26);
    let t = b.terrain(h, 70.0, 260.0, 0.0);
    let o = b.out();
    b.l(t, o, 0);
    let basalt = b.n("block", 0.0, 200.0, &[("block", bk("basalt"))]);
    let ores = b.ores(basalt, 220.0, 200.0);
    let s = surface(&mut b, "ash", "ash", "basalt", 3.0, 300.0, 340.0, ores);
    let mag = b.n("block", 300.0, 440.0, &[("block", bk("magma"))]);
    let sel = b.noise("simplex2d", 300.0, 520.0, 30.0, 2, 1.0, 27);
    let ns = b.n("noise_select", 560.0, 380.0, &[("threshold", fv(0.62))]);
    b.l(s, ns, 0);
    b.l(mag, ns, 1);
    b.l(sel, ns, 2);
    b.l(ns, o, 1);
    let gc = b.n("ground_cover", 260.0, 640.0, &[("on", bk("ash")), ("chance", fv(0.06)), ("plant1", bk("ember_weed")), ("weight1", fv(3.0)), ("plant2", bk("dead_bush")), ("weight2", fv(2.0)), ("weight3", fv(0.0)), ("weight4", fv(0.0))]);
    b.l(gc, o, 2);
    let dead = b.prop("dead_tree", 0.0, 780.0);
    b.scatter(dead, 260.0, 780.0, &[("spacing", fv(16.0)), ("chance", fv(0.4)), ("surface", bk("ash"))], 3, None);
    let col = b.prop("basalt_column", 0.0, 920.0);
    b.scatter(col, 260.0, 920.0, &[("spacing", fv(12.0)), ("chance", fv(0.35)), ("any_surface", Value::B(true)), ("seed", iv(7))], 4, None);
    b.out_params(&[("fluid", ev("Lava")), ("grass_tint", Value::Color([0.4, 0.38, 0.32])), ("fog_color", Value::Color([0.45, 0.38, 0.35])), ("fog_density", fv(3.0)), ("map_color", Value::Color([0.35, 0.33, 0.32]))]);
    b.g
}

fn lava_fields() -> Graph {
    let mut b = B::new("lava_fields", GraphKind::Biome);
    let h = b.noise("simplex2d", 0.0, 0.0, 160.0, 4, 11.0, 28);
    let cones = b.noise("worley2d", 0.0, 150.0, 260.0, 1, 1.0, 29);
    let cc = b.n("curve", 220.0, 150.0, &[("curve", Value::Curve(vec![[0.0, 60.0], [0.12, 40.0], [0.3, 6.0], [1.0, 0.0]]))]);
    b.l(cones, cc, 0);
    let add = b.n("add", 440.0, 60.0, &[]);
    b.l(h, add, 0);
    b.l(cc, add, 1);
    let t = b.terrain(add, 58.0, 640.0, 60.0);
    let o = b.out();
    b.l(t, o, 0);
    let basalt = b.n("block", 0.0, 300.0, &[("block", bk("basalt"))]);
    let obs = b.n("block", 0.0, 380.0, &[("block", bk("obsidian"))]);
    let s = surface(&mut b, "basalt", "basalt", "obsidian", 2.0, 300.0, 300.0, basalt);
    let mag = b.n("block", 300.0, 440.0, &[("block", bk("magma"))]);
    let sel = b.noise("simplex3d", 300.0, 520.0, 20.0, 2, 1.0, 30);
    let ns = b.n("noise_select", 560.0, 380.0, &[("threshold", fv(0.4))]);
    b.l(s, ns, 0);
    b.l(mag, ns, 1);
    b.l(sel, ns, 2);
    let _ = obs;
    b.l(ns, o, 1);
    let spike = b.prop("obsidian_spike", 0.0, 640.0);
    b.scatter(spike, 260.0, 640.0, &[("spacing", fv(30.0)), ("chance", fv(0.3)), ("any_surface", Value::B(true))], 3, None);
    b.out_params(&[("fluid", ev("Lava")), ("fog_color", Value::Color([0.55, 0.3, 0.2])), ("fog_density", fv(3.5)), ("map_color", Value::Color([0.75, 0.25, 0.1]))]);
    b.g
}

/// A cave biome: only its Material (floors / ceilings) and Ground Cover (floors) are used, via a
/// zone's Cave Biomes input. `accent` sprinkles a second block over the ceilings (3D noise).
fn cave_biome(name: &str, floor: &str, ceiling: &str, accent: Option<(&str, f32)>, cover: &[(&str, f32)], chance: f32, map: [f32; 3]) -> Graph {
    let mut b = B::new(name, GraphKind::Biome);
    let o = b.out();
    let ceil = b.n("block", 0.0, 0.0, &[("block", bk(ceiling))]);
    let mut ceil_mat = ceil;
    if let Some((blk, threshold)) = accent {
        let acc = b.n("block", 0.0, 90.0, &[("block", bk(blk))]);
        let nz = b.noise("simplex3d", 0.0, 180.0, 6.0, 1, 1.0, 51);
        let ns = b.n("noise_select", 260.0, 60.0, &[("threshold", fv(threshold))]);
        b.l(ceil, ns, 0);
        b.l(acc, ns, 1);
        b.l(nz, ns, 2);
        ceil_mat = ns;
    }
    // in caves Surface Layers puts `cave_floor` on floors and its base input on ceilings
    let s = b.n("surface_layers", 520.0, 60.0, &[("cave_floor", bk(floor))]);
    b.l(ceil_mat, s, 0);
    b.l(s, o, 1);
    let mut params = vec![("on", bk(floor)), ("chance", fv(chance))];
    for (i, (plant, w)) in cover.iter().enumerate().take(4) {
        params.push((["plant1", "plant2", "plant3", "plant4"][i], bk(plant)));
        params.push((["weight1", "weight2", "weight3", "weight4"][i], fv(*w)));
    }
    for i in cover.len()..4 {
        params.push((["weight1", "weight2", "weight3", "weight4"][i], fv(0.0)));
    }
    let gc = b.n("ground_cover", 520.0, 260.0, &params);
    b.l(gc, o, 2);
    b.out_params(&[("fluid", ev("None")), ("map_color", Value::Color(map))]);
    b.g
}

fn lush_caves() -> Graph {
    cave_biome(
        "lush_caves",
        "moss",
        "moss",
        Some(("mushroom_cap_glow", 0.72)),
        &[("fern", 3.0), ("tall_grass", 3.0), ("mushroom_glow", 1.2), ("mushroom_red", 0.8)],
        0.5,
        [0.3, 0.6, 0.35],
    )
}

fn crystal_caves() -> Graph {
    cave_biome("crystal_caves", "calcite", "calcite", Some(("crystal", 0.6)), &[("crystal_cluster", 2.0), ("pebbles", 1.0)], 0.18, [0.65, 0.5, 0.85])
}

fn magma_caves() -> Graph {
    cave_biome("magma_caves", "basalt", "basalt", Some(("magma", 0.55)), &[("ember_weed", 3.0), ("dead_bush", 1.0)], 0.14, [0.55, 0.25, 0.15])
}

fn frost_caves() -> Graph {
    cave_biome("frost_caves", "snow", "packed_ice", Some(("crystal", 0.78)), &[("snow_shrub", 2.0), ("pebbles", 0.5)], 0.15, [0.7, 0.85, 0.95])
}

/// Replace a zone's Biome Voronoi with a Biome Threshold Select over a large humidity noise:
/// `bands` lists the biomes from driest to wettest, split at `thresholds`.
fn humidity_bands(mut g: Graph, bands: &[&str], thresholds: &[f32]) -> Graph {
    let out = g.output_node().map(|n| n.id).unwrap();
    if let Some(vor) = g.nodes.iter().find(|n| n.kind == "biome_voronoi").map(|n| n.id) {
        g.remove_node(vor);
    }
    let sel = g.add("biome_threshold", [300.0, 0.0]);
    let hum = g.add("simplex2d", [0.0, -150.0]);
    for (k, v) in [("size", fv(1400.0)), ("octaves", iv(3)), ("amplitude", fv(1.0)), ("seed", iv(61))] {
        g.set(hum, k, v);
    }
    g.connect(hum, 0, sel, 0);
    for (i, t) in thresholds.iter().enumerate() {
        g.set(sel, &format!("threshold{}", i + 1), fv(*t));
    }
    g.set(sel, "blend", fv(0.035));
    for (i, name) in bands.iter().enumerate() {
        let id = g.nodes.iter().find(|n| n.kind == "biome" && n.params.get("biome") == Some(&rv(name))).map(|n| n.id);
        if let Some(id) = id {
            g.connect(id, 0, sel, i as u8 + 1);
        }
    }
    g.connect(sel, 0, out, 0);
    g
}

fn zone(name: &str, biomes: &[(&str, f32)], caves: &[(&str, f32)], cell: f32, color: [f32; 3], ravines: bool) -> Graph {
    let mut b = B::new(name, GraphKind::Zone);
    let vor = b.n("biome_voronoi", 300.0, 0.0, &[("cell_size", fv(cell))]);
    for (i, (bn, w)) in biomes.iter().enumerate() {
        let n = b.n("biome", 0.0, i as f32 * 90.0, &[("biome", rv(bn))]);
        b.l(n, vor, i as u8);
        b.g.set(vor, &format!("weight{}", i + 1), fv(*w));
    }
    let o = b.out();
    b.l(vor, o, 0);
    let sp = b.n("spaghetti_caves", 0.0, 420.0, &[]);
    let ch = b.n("cheese_caves", 0.0, 560.0, &[]);
    let nd = b.n("noodle_caves", 0.0, 700.0, &[]);
    let m1 = b.n("max", 260.0, 480.0, &[]);
    b.l(sp, m1, 0);
    b.l(ch, m1, 1);
    let m2 = b.n("max", 420.0, 560.0, &[]);
    b.l(m1, m2, 0);
    b.l(nd, m2, 1);
    let mut last = m2;
    if ravines {
        let rv_ = b.n("ravines", 260.0, 760.0, &[]);
        let m3 = b.n("max", 560.0, 640.0, &[]);
        b.l(last, m3, 0);
        b.l(rv_, m3, 1);
        last = m3;
    }
    b.l(last, o, 1);
    if !caves.is_empty() {
        let sel = b.n("cave_biomes", 300.0, 860.0, &[]);
        for (i, (bn, w)) in caves.iter().enumerate() {
            let n = b.n("biome", 0.0, 860.0 + i as f32 * 90.0, &[("biome", rv(bn))]);
            b.l(n, sel, i as u8);
            b.g.set(sel, &format!("weight{}", i + 1), fv(*w));
        }
        b.l(sel, o, 2);
    }
    b.out_params(&[("map_color", Value::Color(color))]);
    b.g
}

fn world() -> Graph {
    let mut b = B::new("world", GraphKind::World);
    // surface: y -128..1024, the four surface zones, rivers at sea level
    let vor = b.n("zone_voronoi", 300.0, 0.0, &[]);
    for (i, z) in ["temperate", "arid", "frozen", "volcanic"].iter().enumerate() {
        let n = b.n("zone", 0.0, i as f32 * 90.0, &[("zone", rv(z))]);
        b.l(n, vor, i as u8);
    }
    b.g.set(vor, "weight1", fv(1.6));
    b.g.set(vor, "weight4", fv(0.7));
    let surface = b.n("layer", 520.0, 0.0, &[("kind", ev("Surface")), ("y_min", fv(-128.0)), ("y_max", fv(1024.0)), ("sea_level", fv(62.0))]);
    b.l(vor, surface, 0);
    let river = b.n("river", 0.0, 420.0, &[]);
    b.l(river, surface, 1);
    let o = b.out();
    b.l(surface, o, 0);
    // underground: upper caves -128..-1200, the deep -1200..-6000 with a lava sea
    let upper = b.n("zone", 0.0, 560.0, &[("zone", rv("upper_caves"))]);
    let deep = b.n("zone", 0.0, 650.0, &[("zone", rv("deep"))]);
    let bands = b.n("depth_bands", 520.0, 560.0, &[("sea_level_deep", fv(-5300.0))]);
    b.l(upper, bands, 0);
    b.l(deep, bands, 1);
    b.l(bands, o, 1);
    // sky: y 4000..6200, above the cloud sea
    let sky_z = b.n("zone", 0.0, 800.0, &[("zone", rv("sky"))]);
    let sky = b.n("layer", 520.0, 800.0, &[("kind", ev("Sky")), ("y_min", fv(4000.0)), ("y_max", fv(6200.0)), ("sea_level", fv(-100000.0))]);
    b.l(sky_z, sky, 0);
    b.l(sky, o, 2);
    b.g
}

fn ridge_function() -> Graph {
    // An example reusable function: sharp ridges with a falloff, f(arg0 = strength)
    let mut b = B::new("sharp_ridges", GraphKind::Function);
    let r = b.noise("ridged2d", 0.0, 0.0, 300.0, 5, 1.0, 40);
    let a = b.n("function_input", 0.0, 150.0, &[("index", iv(0))]);
    let m = b.n("multiply", 240.0, 60.0, &[]);
    b.l(r, m, 0);
    b.l(a, m, 1);
    let o = b.out();
    b.l(m, o, 0);
    b.g
}

// ---- prop graphs ------------------------------------------------------------------------------

/// A prop graph: its output's box (radius, height, below) and whether it stays on far LODs.
fn prop_graph(name: &str, radius: f32, height: f32, below: f32, far: bool) -> B {
    let mut b = B::new(name, GraphKind::Prop);
    b.out_params(&[("radius", fv(radius)), ("height", fv(height)), ("below", fv(below)), ("far", Value::B(far))]);
    b
}

/// Connect up to four shapes to the output through a Union (or directly if there is one).
fn finish(mut b: B, shapes: &[(u32, u8)]) -> Graph {
    let o = b.out();
    if let [(s, pin)] = shapes {
        b.l2(*s, *pin, o, 0);
    } else {
        let u = b.n("union", 0.0, 0.0, &[]);
        for (i, (s, pin)) in shapes.iter().enumerate() {
            b.l2(*s, *pin, u, i as u8);
        }
        b.l(u, o, 0);
    }
    b.g
}

fn trunk(b: &mut B, block: &str, hmin: f32, hmax: f32, extra: &[(&str, Value)]) -> u32 {
    let t = b.n("trunk", 0.0, 0.0, &[("block", bk(block)), ("height_min", fv(hmin)), ("height_max", fv(hmax))]);
    for (k, v) in extra {
        b.g.set(t, k, v.clone());
    }
    t
}

fn leaves(b: &mut B, skeleton: u32, block: &str, params: &[(&str, Value)]) -> u32 {
    let l = b.n("leaves", 0.0, 0.0, &[("block", bk(block))]);
    for (k, v) in params {
        b.g.set(l, k, v.clone());
    }
    b.l2(skeleton, 1, l, 0);
    l
}

fn branches(b: &mut B, parent: u32, block: &str, params: &[(&str, Value)]) -> u32 {
    let br = b.n("branches", 0.0, 0.0, &[("block", bk(block))]);
    for (k, v) in params {
        b.g.set(br, k, v.clone());
    }
    b.l2(parent, 1, br, 0);
    br
}

fn oak() -> Graph {
    let mut b = prop_graph("oak", 7.0, 15.0, 1.0, true);
    let t = trunk(&mut b, "oak_log", 5.0, 8.0, &[("segments", iv(4))]);
    let br = branches(&mut b, t, "oak_log", &[("count", iv(3)), ("start", fv(0.55)), ("end", fv(0.9)), ("angle", fv(55.0)), ("length", fv(0.45)), ("radius", fv(0.55)), ("droop", fv(0.5))]);
    let l = leaves(&mut b, br, "oak_leaves", &[("radius", fv(2.6)), ("y_scale", fv(0.8)), ("density", fv(0.88)), ("bumps", fv(0.7))]);
    finish(b, &[(t, 0), (br, 0), (l, 0)])
}

fn birch() -> Graph {
    let mut b = prop_graph("birch", 5.0, 16.0, 1.0, true);
    let t = trunk(&mut b, "birch_log", 7.0, 12.0, &[("segments", iv(4))]);
    let l = leaves(&mut b, t, "birch_leaves", &[("shape", ev("Ellipsoid")), ("at", ev("Along")), ("start", fv(0.45)), ("spacing", fv(2.0)), ("radius", fv(2.2)), ("y_scale", fv(1.2)), ("top_scale", fv(0.55)), ("density", fv(0.86)), ("bumps", fv(0.5))]);
    finish(b, &[(t, 0), (l, 0)])
}

fn spruce() -> Graph {
    let mut b = prop_graph("spruce", 5.0, 19.0, 1.0, true);
    let t = trunk(&mut b, "spruce_log", 8.0, 15.0, &[("segments", iv(4)), ("taper", fv(0.8))]);
    let l = leaves(&mut b, t, "spruce_leaves", &[("shape", ev("Cone")), ("at", ev("Along")), ("start", fv(0.2)), ("spacing", fv(1.7)), ("radius", fv(3.2)), ("y_scale", fv(0.9)), ("top_scale", fv(0.3)), ("density", fv(0.96)), ("bumps", fv(0.3)), ("var", fv(0.08))]);
    let top = leaves(&mut b, t, "spruce_leaves", &[("shape", ev("Cone")), ("radius", fv(1.3)), ("y_scale", fv(1.6)), ("y_offset", fv(-0.5)), ("density", fv(1.0)), ("bumps", fv(0.0)), ("var", fv(0.0))]);
    finish(b, &[(t, 0), (l, 0), (top, 0)])
}

fn palm() -> Graph {
    let mut b = prop_graph("palm", 8.0, 13.0, 1.0, true);
    let t = trunk(&mut b, "palm_log", 6.0, 9.0, &[("segments", iv(6)), ("bend", fv(3.0))]);
    let l = leaves(&mut b, t, "palm_leaves", &[("shape", ev("Fronds")), ("radius", fv(4.5)), ("y_scale", fv(0.9)), ("density", fv(1.0)), ("bumps", fv(0.0)), ("var", fv(0.1))]);
    finish(b, &[(t, 0), (l, 0)])
}

fn acacia() -> Graph {
    let mut b = prop_graph("acacia", 9.0, 12.0, 1.0, true);
    let t = trunk(&mut b, "acacia_log", 5.0, 7.0, &[("segments", iv(4)), ("bend", fv(2.0))]);
    let br = branches(&mut b, t, "acacia_log", &[("count", iv(2)), ("start", fv(0.6)), ("end", fv(0.9)), ("angle", fv(45.0)), ("length", fv(0.55)), ("radius", fv(0.55)), ("droop", fv(-0.5))]);
    let l = leaves(&mut b, br, "acacia_leaves", &[("shape", ev("Disc")), ("radius", fv(3.4)), ("y_scale", fv(0.6)), ("y_offset", fv(0.6)), ("density", fv(0.9)), ("bumps", fv(0.6))]);
    finish(b, &[(t, 0), (br, 0), (l, 0)])
}

fn dead_tree() -> Graph {
    let mut b = prop_graph("dead_tree", 6.0, 11.0, 1.0, true);
    let t = trunk(&mut b, "charred_log", 4.0, 8.0, &[("segments", iv(4)), ("twist", fv(0.25))]);
    let br = branches(&mut b, t, "charred_log", &[("count", iv(3)), ("levels", iv(2)), ("start", fv(0.45)), ("end", fv(0.9)), ("angle", fv(60.0)), ("length", fv(0.4)), ("radius", fv(0.55)), ("droop", fv(-0.6))]);
    finish(b, &[(t, 0), (br, 0)])
}

fn bush() -> Graph {
    let mut b = prop_graph("bush", 3.0, 4.0, 1.0, false);
    let blob = b.n("blob", 0.0, 0.0, &[("y", fv(0.6)), ("radius", fv(1.8)), ("var", fv(0.25)), ("squash", fv(1.4)), ("bumps", fv(0.6)), ("bump_size", fv(2.0)), ("block", bk("oak_leaves"))]);
    let h = b.n("holes", 0.0, 0.0, &[("amount", fv(0.15))]);
    b.l(blob, h, 0);
    finish(b, &[(h, 0)])
}

fn willow() -> Graph {
    let mut b = prop_graph("willow", 8.0, 13.0, 1.0, true);
    let t = trunk(&mut b, "oak_log", 6.0, 9.0, &[("segments", iv(4)), ("radius", fv(0.6))]);
    let br = branches(&mut b, t, "oak_log", &[("count", iv(4)), ("start", fv(0.6)), ("end", fv(0.95)), ("angle", fv(70.0)), ("length", fv(0.45)), ("radius", fv(0.55)), ("droop", fv(1.0))]);
    let l = leaves(&mut b, br, "oak_leaves", &[("shape", ev("Ellipsoid")), ("radius", fv(2.6)), ("y_scale", fv(0.7)), ("y_offset", fv(0.5)), ("density", fv(0.88)), ("bumps", fv(0.6))]);
    let v = b.n("vines", 0.0, 0.0, &[("count", iv(14)), ("spread", fv(2.8)), ("length_min", fv(2.0)), ("length_max", fv(5.0)), ("y_offset", fv(0.5)), ("block", bk("oak_leaves"))]);
    b.l2(br, 1, v, 0);
    finish(b, &[(t, 0), (br, 0), (l, 0), (v, 0)])
}

fn giant_mushroom() -> Graph {
    let mut b = prop_graph("giant_mushroom", 7.0, 14.0, 1.0, true);
    let t = trunk(&mut b, "mushroom_stem", 6.0, 11.0, &[("segments", iv(3)), ("radius", fv(1.1)), ("bend", fv(1.0))]);
    let cap = leaves(&mut b, t, "mushroom_cap_red", &[("shape", ev("Disc")), ("radius", fv(4.5)), ("y_scale", fv(0.75)), ("y_offset", fv(0.5)), ("density", fv(1.0)), ("bumps", fv(0.25)), ("var", fv(0.15))]);
    finish(b, &[(t, 0), (cap, 0)])
}

fn boulder(name: &str, block: &str, block2: &str, radius: f32) -> Graph {
    let mut b = prop_graph(name, radius * 1.6 + 2.0, radius * 1.6 + 1.0, radius + 1.0, false);
    let blob = b.n("blob", 0.0, 0.0, &[("y", fv(radius * 0.25 - 0.6)), ("radius", fv(radius)), ("var", fv(0.3)), ("squash", fv(1.25)), ("bumps", fv(0.7)), ("bump_size", fv(2.5)), ("block", bk(block))]);
    let sb = b.n("set_block", 0.0, 0.0, &[("block", bk(block)), ("block2", bk(block2)), ("mix", fv(0.4))]);
    b.l(blob, sb, 0);
    finish(b, &[(sb, 0)])
}

fn cactus() -> Graph {
    let mut b = prop_graph("cactus", 4.0, 7.0, 1.0, false);
    let t = trunk(&mut b, "cactus", 2.0, 5.0, &[("segments", iv(1))]);
    let br = branches(&mut b, t, "cactus", &[("count", iv(2)), ("start", fv(0.35)), ("end", fv(0.75)), ("angle", fv(90.0)), ("length", fv(0.3)), ("radius", fv(0.55)), ("droop", fv(-2.0)), ("elbow", Value::B(true))]);
    finish(b, &[(t, 0), (br, 0)])
}

fn spike(name: &str, block: &str, hmin: f32, hmax: f32, radius: f32) -> Graph {
    let mut b = prop_graph(name, radius + 2.0, hmax + 1.0, 4.0, true);
    let t = trunk(&mut b, block, hmin, hmax, &[("radius", fv(radius)), ("taper", fv(0.14)), ("taper_curve", fv(0.8)), ("sink", fv(3.0)), ("segments", iv(6))]);
    finish(b, &[(t, 0)])
}

fn basalt_column() -> Graph {
    let mut b = prop_graph("basalt_column", 3.0, 15.0, 2.0, true);
    let t = trunk(&mut b, "basalt", 3.0, 14.0, &[("radius", fv(1.5)), ("sink", fv(1.0)), ("segments", iv(1))]);
    finish(b, &[(t, 0)])
}

/// A hollow geode: shells of basalt, calcite and crystal around air, all sized by one random
/// scale and roughened by one displacement, so the shells stay concentric.
fn geode() -> Graph {
    let mut b = prop_graph("geode", 8.0, 8.0, 8.0, false);
    let o = b.out();
    b.g.set(o, "replace", ev("Everything"));
    let shells = [("basalt", 6.0, 1), ("calcite", 5.0, 2), ("crystal", 4.0, 3)];
    let u = b.n("union", 0.0, 0.0, &[]);
    for (i, (block, r, prio)) in shells.iter().enumerate() {
        let s = b.n("sphere", 0.0, 0.0, &[("y", fv(0.0)), ("radius", fv(*r)), ("block", bk(block)), ("priority", iv(*prio))]);
        b.l(s, u, i as u8);
    }
    let core = b.n("sphere", 0.0, 0.0, &[("y", fv(0.0)), ("radius", fv(3.0)), ("priority", iv(4))]);
    let carve = b.n("carve", 0.0, 0.0, &[]);
    b.l(core, carve, 0);
    b.l(carve, u, 3);
    let d = b.n("displace", 0.0, 0.0, &[("amount", fv(0.4)), ("size", fv(2.0))]);
    b.l(u, d, 0);
    let rnd = b.n("random", 0.0, 0.0, &[("min", fv(0.8)), ("max", fv(1.1))]);
    let tf = b.n("transform", 0.0, 0.0, &[]);
    b.l(d, tf, 0);
    b.l(rnd, tf, 1);
    finish(b, &[(tf, 0)])
}

/// Every built-in prop graph.
pub fn default_props() -> Vec<Graph> {
    vec![
        oak(),
        birch(),
        spruce(),
        palm(),
        acacia(),
        dead_tree(),
        bush(),
        willow(),
        giant_mushroom(),
        boulder("boulder", "cobblestone", "mossy_cobblestone", 1.5),
        boulder("boulder_andesite", "andesite", "mossy_cobblestone", 2.2),
        boulder("boulder_terracotta", "terracotta_red", "terracotta", 1.7),
        boulder("boulder_ice", "packed_ice", "ice", 2.2),
        cactus(),
        spike("ice_spike", "packed_ice", 10.0, 30.0, 2.5),
        spike("obsidian_spike", "obsidian", 4.0, 12.0, 1.5),
        basalt_column(),
        geode(),
    ]
}

pub fn default_project() -> Project {
    let mut p = Project {
        graphs: vec![
            world(),
            zone(
                "temperate",
                &[("meadow", 1.0), ("oak_forest", 1.4), ("birch_hills", 1.0)],
                &[("lush_caves", 1.2), ("crystal_caves", 0.5)],
                420.0,
                [0.35, 0.7, 0.3],
                true,
            ),
            // arid: badlands where it's driest, desert, savanna where it's wettest (humidity bands)
            humidity_bands(
                zone("arid", &[("desert", 1.5), ("badlands", 1.0), ("savanna", 1.0)], &[("crystal_caves", 1.0), ("lush_caves", 0.4)], 520.0, [0.9, 0.75, 0.4], true),
                &["badlands", "desert", "savanna"],
                &[-0.3, 0.2],
            ),
            zone("frozen", &[("taiga", 1.3), ("glacier", 0.8), ("snowy_peaks", 1.0)], &[("frost_caves", 1.0), ("crystal_caves", 0.6)], 480.0, [0.8, 0.9, 1.0], false),
            zone("volcanic", &[("ashlands", 1.0), ("lava_fields", 0.8)], &[("magma_caves", 1.0), ("crystal_caves", 0.3)], 380.0, [0.6, 0.2, 0.15], false),
            meadow(),
            oak_forest(),
            birch_hills(),
            floating_isles(),
            desert(),
            badlands(),
            savanna(),
            taiga(),
            glacier(),
            snowy_peaks(),
            ashlands(),
            lava_fields(),
            lush_caves(),
            crystal_caves(),
            magma_caves(),
            frost_caves(),
            single_zone("sky", "floating_isles", [0.55, 0.85, 1.0]),
            single_zone("upper_caves", "caverns", [0.35, 0.55, 0.3]),
            single_zone("deep", "magma_depths", [0.55, 0.15, 0.1]),
            underground_biome("caverns", "stone", "moss", &[("mushroom_glow", 1.0), ("fern", 2.0), ("mushroom_red", 0.5)], "Water", (-1200.0, -128.0), [0.3, 0.5, 0.3]),
            underground_biome("magma_depths", "basalt", "magma", &[("ember_weed", 1.0)], "Lava", (-6000.0, -1200.0), [0.5, 0.15, 0.1]),
            ridge_function(),
        ],
    };
    p.graphs.extend(default_props());
    // colour grading while the camera is in a biome: tint, saturation, contrast
    let grading: &[(&str, [f32; 3], f32, f32)] = &[
        ("meadow", [1.0, 1.0, 0.97], 1.08, 1.0),
        ("oak_forest", [0.98, 1.02, 0.96], 1.05, 1.0),
        ("birch_hills", [1.0, 1.0, 0.98], 1.05, 1.0),
        ("floating_isles", [0.97, 1.0, 1.04], 1.12, 1.02),
        ("desert", [1.06, 1.0, 0.9], 1.05, 1.05),
        ("badlands", [1.08, 0.98, 0.9], 1.1, 1.08),
        ("savanna", [1.05, 1.02, 0.9], 1.0, 1.02),
        ("taiga", [0.95, 1.0, 1.04], 0.92, 1.0),
        ("glacier", [0.93, 0.98, 1.08], 0.85, 1.05),
        ("snowy_peaks", [0.95, 0.98, 1.06], 0.88, 1.03),
        ("ashlands", [1.05, 0.95, 0.9], 0.7, 1.1),
        ("lava_fields", [1.1, 0.95, 0.85], 1.1, 1.12),
    ];
    // ambient particles and precipitation (rain falls with the weather, snow always a little)
    let particles: &[(&str, &str, &str)] = &[
        ("meadow", "Pollen", "Rain"),
        ("oak_forest", "Fireflies", "Rain"),
        ("birch_hills", "Fireflies", "Rain"),
        ("floating_isles", "Pollen", "Rain"),
        ("savanna", "Pollen", "Rain"),
        ("taiga", "None", "Snow"),
        ("glacier", "None", "Snow"),
        ("snowy_peaks", "None", "Snow"),
        ("ashlands", "Ash", "None"),
        ("lava_fields", "Embers", "None"),
    ];
    for g in &mut p.graphs {
        if let Some(&(_, tint, sat, contrast)) = grading.iter().find(|e| e.0 == g.name)
            && let Some(o) = g.output_node().map(|n| n.id)
        {
            g.set(o, "grade_tint", Value::Color(tint));
            g.set(o, "saturation", fv(sat));
            g.set(o, "contrast", fv(contrast));
        }
        if let Some(&(_, ambient, precip)) = particles.iter().find(|e| e.0 == g.name)
            && let Some(o) = g.output_node().map(|n| n.id)
        {
            g.set(o, "ambient", ev(ambient));
            g.set(o, "precipitation", ev(precip));
        }
        crate::layout::auto_layout(g);
        if g.kind == GraphKind::World {
            // a note around the river network (comment boxes ignore auto-layout)
            if let Some(r) = g.nodes.iter().find(|n| n.kind == "river").map(|n| n.pos) {
                let c = g.add("comment", [r[0] - 24.0, r[1] - 48.0]);
                g.set(c, "width", fv(330.0));
                g.set(c, "height", fv(crate::layout::node_size("river")[1] + 40.0));
                if let Some(n) = g.node_mut(c) {
                    n.comment = "Rivers: carved to sea level everywhere".into();
                }
            }
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile;
    use std::collections::HashMap;

    fn blocks() -> HashMap<String, u32> {
        [
            "air", "stone", "dirt", "grass", "sand", "sandstone", "gravel", "snow", "snowy_grass", "ice", "packed_ice", "water", "lava", "red_sand",
            "terracotta", "terracotta_orange", "terracotta_red", "terracotta_yellow", "terracotta_brown", "terracotta_white", "basalt", "ash", "magma",
            "obsidian", "deepslate", "calcite", "podzol", "dry_grass", "coarse_dirt", "coal_ore", "iron_ore", "gold_ore", "copper_ore", "crystal",
            "oak_log", "oak_leaves", "birch_log", "birch_leaves", "spruce_log", "spruce_leaves", "acacia_log", "acacia_leaves", "charred_log", "cactus",
            "palm_log", "palm_leaves", "mushroom_stem", "mushroom_cap_red", "cobblestone", "mossy_cobblestone", "andesite", "tall_grass", "fern",
            "flower_red", "flower_yellow", "flower_blue", "flower_white", "dead_bush", "dry_shrub", "snow_shrub", "ember_weed", "pebbles", "mushroom_red",
            "moss", "mushroom_cap_glow", "mushroom_glow", "crystal_cluster",
        ]
        .iter()
        .enumerate()
        .map(|(i, n)| (n.to_string(), i as u32))
        .collect()
    }

    fn prefabs() -> HashMap<String, (u32, [i32; 3])> {
        [("ruin".to_string(), (0u32, [7, 5, 7]))].into_iter().collect()
    }

    #[test]
    fn default_project_compiles() {
        let p = default_project();
        let c = compile::compile(&p, &blocks(), &prefabs()).unwrap();
        assert!(c.warnings.is_empty(), "warnings: {:#?}", c.warnings);
        // four generator units: the surface, the two underground bands and the sky
        let layers: Vec<(&str, u32, u32)> = c.layers.iter().map(|l| (l.name.as_str(), l.biome_count, l.zone_count)).collect();
        assert_eq!(layers, [("surface", 15, 4), ("upper", 1, 1), ("deep", 1, 1), ("sky", 1, 1)]);
        assert_eq!((c.biomes.len(), c.zones.len()), (18, 7));
        assert_eq!(c.layers[3].biome_base, 17, "biome ids are global: the sky's come last");
        assert!(c.layers.iter().all(|l| l.source.contains("void wg_column(") && l.source.contains("float gLayerBlend;")));
        // each layer reads its own block of parameters (P(i) is relative to it), so its code
        // doesn't depend on how many parameters the layers before it have
        for (i, l) in c.layers.iter().enumerate() {
            let max = l.source.split("P(").skip(1).filter_map(|t| t[..t.find('u').unwrap_or(0)].parse::<u32>().ok()).max().unwrap();
            assert!(max < l.param_count, "layer {i} reads P({max}) beyond its {} parameters", l.param_count);
            assert_eq!(l.param_base, c.layers[..i].iter().map(|l| l.param_count).sum::<u32>());
        }
        assert!(c.cloud_sea.iter().all(|&s| s as usize >= c.layers.iter().map(|l| l.param_count as usize).sum::<usize>()));
        assert_eq!(c.value(c.cloud_sea[0]), 2600.0);
        assert!(c.layers[3].source.contains("#define LAYER_BELOW -1.0") && c.layers[1].source.contains("#define LAYER_ABOVE 1.0"));
        let range = |i: usize| (c.value(c.layers[i].y_min), c.value(c.layers[i].y_max));
        assert_eq!([range(0), range(1), range(2), range(3)], [(-128.0, 1024.0), (-1200.0, -128.0), (-6000.0, -1200.0), (4000.0, 6200.0)]);
        assert_eq!(c.layer_at(70.0), Some(0));
        assert_eq!(c.layer_at(-3000.0), Some(2));
        assert_eq!(c.layer_at(2000.0), None);
        // who generates what: the sky, the surface, the upper caves, and the deep all the way down
        let own = c.owned_ranges();
        assert_eq!(own, [(-128.0, 1024.0), (-1200.0, -128.0), (f32::NEG_INFINITY, -1200.0), (4000.0, 6200.0)]);
        // between layers the cloud sea (2600..3400) decides which one the camera sees
        assert_eq!([c.layer_near(70.0), c.layer_near(2550.0), c.layer_near(3000.0), c.layer_near(3500.0), c.layer_near(-9000.0)], [0, 0, 3, 3, 2]);
        // only the first layer builds every prop graph (the previews run there)
        assert!(c.layers[3].source.matches("_build() {").count() < c.layers[0].source.matches("_build() {").count());
        let c0 = c.clone();
        let c = &c0.layers[0];
        assert!(c.source.contains("wg_column"));
        // cave biomes get flat per-voxel functions, every zone selects them, rivers are wired
        for b in ["lush_caves", "crystal_caves", "magma_caves", "frost_caves"] {
            let id = c0.biomes.iter().position(|i| i.name == b).unwrap();
            assert!(c.source.contains(&format!("uint b{id}_cmat(")), "{b} has no cave material");
        }
        assert!(c.source.contains("return NO_CAVE_BIOME;") && c.source.matches("_cave(vec3 p)").count() == 4);
        assert!(c.source.contains("river_mask("));
        assert!(c.source.contains("float th[2] = float[2]("), "arid zone uses humidity bands");
        assert!(c0.prop_layers > 10);
        // the terrain pass looks as deep as the column's biomes need: 12 blocks of sandstone
        // under the desert, 1 + 3 blocks of grass and dirt in the meadow
        let fun = &c.source[c.source.find("float wg_depth_max(").unwrap()..];
        let fun = &fun[..fun.find("\n}").unwrap()];
        let case = |b: &str| {
            let id = c0.biomes.iter().position(|i| i.name == b).unwrap();
            let line = fun.lines().find(|l| l.trim_start().starts_with(&format!("case {id}u:"))).unwrap().to_string();
            line.split("P(").skip(1).map(|t| c0.params[t[..t.find('u').unwrap()].parse::<usize>().unwrap()]).collect::<Vec<f32>>()
        };
        assert!(case("desert").contains(&12.0), "{fun}");
        assert_eq!(case("meadow"), [1.0, 3.0], "{fun}");
    }

    #[test]
    fn prop_graphs_compile_to_build_functions() {
        let p = default_project();
        let c = compile::compile(&p, &blocks(), &prefabs()).unwrap();
        let names = p.names(GraphKind::Prop);
        assert_eq!(names.len(), default_props().len());
        // one build function per prop graph, behind the props pass define
        let props = &c.source[c.source.find("#ifdef PROPS_PASS").unwrap()..];
        for k in 0..names.len() {
            assert!(props.contains(&format!("void prop{k}_build()")), "prop {k}");
        }
        let oak = names.iter().position(|n| n == "oak").unwrap();
        let body = &props[props.find(&format!("void prop{oak}_build()")).unwrap()..];
        let body = &body[..body.find("\n}").unwrap()];
        for call in ["tree_trunk(", "tree_branches(", "tree_leaves(", "tree_wood("] {
            assert!(body.contains(call), "oak: {call}\n{body}");
        }
        // the trunk is grown once although wood, branches and leaves all use it
        assert_eq!(body.matches("tree_trunk(").count(), 1, "{body}");
        let geode = names.iter().position(|n| n == "geode").unwrap();
        let body = &props[props.find(&format!("void prop{geode}_build()")).unwrap()..];
        let body = &body[..body.find("\n}").unwrap()];
        for call in ["xf_make(", "pb_displace(", "OP_CARVE", "pb_sphere("] {
            assert!(body.contains(call), "geode: {call}\n{body}");
        }
        // scatters point at prop graphs, geodes are buried
        assert!(c.source.contains("L.kind = PROP_GRAPH;") && c.source.contains("L.buried = true;"));
        // the prop's box is a live parameter of the prop graph
        assert!(c.param_index.keys().any(|k| k.graph == (GraphKind::Prop, "oak".to_string()) && k.param == "radius"));
    }

    #[test]
    fn param_change_keeps_source() {
        let mut p = default_project();
        let a = compile::compile(&p, &blocks(), &prefabs()).unwrap();
        let g = p.get_mut(GraphKind::Biome, "meadow").unwrap();
        let id = g.nodes.iter().find(|n| n.kind == "simplex2d").unwrap().id;
        g.set(id, "amplitude", Value::F(99.0));
        let b = compile::compile(&p, &blocks(), &prefabs()).unwrap();
        assert_eq!(a.source, b.source);
        assert_ne!(a.params, b.params);
    }

    #[test]
    fn roundtrip_ron() {
        let p = default_project();
        let dir = std::env::temp_dir().join(format!("wg_test_{}", std::process::id()));
        p.save_dir(&dir).unwrap();
        let q = Project::load_dir(&dir).unwrap();
        assert_eq!(q.graphs.len(), p.graphs.len());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
