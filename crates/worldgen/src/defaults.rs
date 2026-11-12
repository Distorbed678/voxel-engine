//! The built-in example world: 4 zones (temperate, arid, frozen, volcanic) with 12 biomes.
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
    let oak = b.n("tree", 0.0, 650.0, &[("species", ev("Round")), ("height_min", fv(5.0)), ("height_max", fv(7.0)), ("radius", fv(2.5))]);
    b.scatter(oak, 260.0, 650.0, &[("spacing", fv(28.0)), ("chance", fv(0.35))], 3, None);
    let rock = b.n("boulder", 0.0, 800.0, &[("radius_min", fv(1.0)), ("radius_max", fv(2.0))]);
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
    let oak = b.n("tree", 0.0, 1000.0, &[("species", ev("Round")), ("height_min", fv(5.0)), ("height_max", fv(9.0)), ("radius", fv(3.0))]);
    b.scatter(oak, 260.0, 1000.0, &[("spacing", fv(8.0)), ("chance", fv(0.75))], 3, Some(mask));
    let birch = b.n("tree", 0.0, 1150.0, &[("species", ev("Birch")), ("trunk", bk("birch_log")), ("leaves", bk("birch_leaves")), ("height_min", fv(7.0)), ("height_max", fv(11.0)), ("radius", fv(2.0))]);
    b.scatter(birch, 260.0, 1150.0, &[("spacing", fv(13.0)), ("chance", fv(0.4)), ("seed", iv(5))], 4, Some(mask));
    let bush = b.n("tree", 0.0, 1300.0, &[("species", ev("Bush")), ("height_min", fv(1.0)), ("height_max", fv(2.0)), ("radius", fv(2.0))]);
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
    let birch = b.n("tree", 0.0, 900.0, &[("species", ev("Birch")), ("trunk", bk("birch_log")), ("leaves", bk("birch_leaves")), ("height_min", fv(8.0)), ("height_max", fv(13.0)), ("radius", fv(2.5))]);
    b.scatter(birch, 260.0, 900.0, &[("spacing", fv(10.0)), ("chance", fv(0.55))], 3, None);
    let rock = b.n("boulder", 0.0, 1050.0, &[("radius_min", fv(1.5)), ("radius_max", fv(3.0)), ("block", bk("andesite")), ("block2", bk("mossy_cobblestone"))]);
    b.scatter(rock, 260.0, 1050.0, &[("spacing", fv(30.0)), ("chance", fv(0.3)), ("seed", iv(9))], 4, None);
    b.out_params(&[("grass_tint", Value::Color([0.55, 0.78, 0.35])), ("foliage_tint", Value::Color([0.5, 0.72, 0.3])), ("map_color", Value::Color([0.6, 0.75, 0.4]))]);
    b.g
}

fn floating_isles() -> Graph {
    let mut b = B::new("floating_isles", GraphKind::Biome);
    let h = b.noise("simplex2d", 0.0, 0.0, 300.0, 4, 10.0, 6);
    let t = b.terrain(h, 70.0, 240.0, 0.0);
    // islands: 3D blobs in a band around y=190
    let blob = b.noise("simplex3d", 0.0, 160.0, 70.0, 3, 40.0, 12);
    let band = b.n("y_gradient", 0.0, 300.0, &[("y_low", fv(150.0)), ("y_high", fv(230.0)), ("value_low", fv(-60.0)), ("value_high", fv(-60.0))]);
    let center = b.n("coord", 0.0, 400.0, &[("axis", ev("Y"))]);
    let dist = b.n("scale_bias", 200.0, 400.0, &[("scale", fv(1.0)), ("bias", fv(-195.0))]);
    b.l(center, dist, 0);
    let ad = b.n("abs", 380.0, 400.0, &[]);
    b.l(dist, ad, 0);
    let falloff = b.n("scale_bias", 540.0, 400.0, &[("scale", fv(-1.6)), ("bias", fv(-8.0))]);
    b.l(ad, falloff, 0);
    let isl = b.n("add", 720.0, 250.0, &[]);
    b.l(blob, isl, 0);
    b.l(falloff, isl, 1);
    let _ = band;
    let both = b.n("max", 900.0, 100.0, &[]);
    b.l(t, both, 0);
    b.l(isl, both, 1);
    let o = b.out();
    b.l(both, o, 0);
    let st = b.deep(0.0, 560.0);
    let s = surface(&mut b, "grass", "dirt", "sand", 3.0, 300.0, 560.0, st);
    // islands are "open to sky" at their tops; their undersides get the base material
    b.l(s, o, 1);
    let gc = b.n("ground_cover", 260.0, 700.0, &[("chance", fv(0.5)), ("plant2", bk("flower_white")), ("weight2", fv(2.0))]);
    b.l(gc, o, 2);
    let tree = b.n("tree", 0.0, 840.0, &[("species", ev("Willow")), ("height_min", fv(6.0)), ("height_max", fv(9.0)), ("radius", fv(3.5))]);
    b.scatter(tree, 260.0, 840.0, &[("spacing", fv(16.0)), ("chance", fv(0.5))], 3, None);
    b.out_params(&[("grass_tint", Value::Color([0.42, 0.85, 0.45])), ("fog_color", Value::Color([0.8, 0.85, 1.0])), ("map_color", Value::Color([0.4, 0.85, 0.6]))]);
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
    let cactus = b.n("column", 0.0, 760.0, &[("block", bk("cactus")), ("height_min", fv(2.0)), ("height_max", fv(5.0)), ("arms", Value::B(true))]);
    b.scatter(cactus, 260.0, 760.0, &[("spacing", fv(14.0)), ("chance", fv(0.35)), ("surface", bk("sand"))], 3, None);
    let palm = b.n("tree", 0.0, 900.0, &[("species", ev("Palm")), ("trunk", bk("palm_log")), ("leaves", bk("palm_leaves")), ("height_min", fv(6.0)), ("height_max", fv(9.0)), ("radius", fv(4.0))]);
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
    let rock = b.n("boulder", 0.0, 640.0, &[("block", bk("terracotta_red")), ("block2", bk("terracotta")), ("radius_min", fv(1.0)), ("radius_max", fv(2.5))]);
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
    let acacia = b.n("tree", 0.0, 760.0, &[("species", ev("Acacia")), ("trunk", bk("acacia_log")), ("leaves", bk("acacia_leaves")), ("height_min", fv(5.0)), ("height_max", fv(8.0)), ("radius", fv(4.0))]);
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
    let spruce = b.n("tree", 0.0, 820.0, &[("species", ev("Spruce")), ("trunk", bk("spruce_log")), ("leaves", bk("spruce_leaves")), ("height_min", fv(8.0)), ("height_max", fv(16.0)), ("radius", fv(3.5))]);
    b.scatter(spruce, 260.0, 820.0, &[("spacing", fv(7.0)), ("chance", fv(0.6)), ("any_surface", Value::B(true))], 3, None);
    let geode = b.n("geode", 260.0, 980.0, &[]);
    let o2 = b.out();
    b.l(geode, o2, 4);
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
    let spike = b.n("spike", 0.0, 400.0, &[("height_min", fv(10.0)), ("height_max", fv(30.0)), ("radius", fv(2.5))]);
    b.scatter(spike, 260.0, 400.0, &[("spacing", fv(26.0)), ("chance", fv(0.35)), ("any_surface", Value::B(true))], 3, None);
    let ice = b.n("boulder", 0.0, 560.0, &[("block", bk("packed_ice")), ("block2", bk("ice")), ("radius_min", fv(1.5)), ("radius_max", fv(3.0))]);
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
    let spruce = b.n("tree", 0.0, 1080.0, &[("species", ev("Spruce")), ("trunk", bk("spruce_log")), ("leaves", bk("spruce_leaves")), ("height_min", fv(7.0)), ("height_max", fv(12.0)), ("radius", fv(3.0))]);
    b.scatter(spruce, 260.0, 1080.0, &[("spacing", fv(11.0)), ("chance", fv(0.5)), ("surface", bk("snowy_grass")), ("y_max", fv(140.0))], 3, None);
    let geode = b.n("geode", 260.0, 1220.0, &[("y_max", fv(60.0))]);
    let o2 = b.out();
    b.l(geode, o2, 4);
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
    let dead = b.n("tree", 0.0, 780.0, &[("species", ev("Dead")), ("trunk", bk("charred_log")), ("leaves", bk("charred_log")), ("height_min", fv(4.0)), ("height_max", fv(8.0)), ("radius", fv(2.0))]);
    b.scatter(dead, 260.0, 780.0, &[("spacing", fv(16.0)), ("chance", fv(0.4)), ("surface", bk("ash"))], 3, None);
    let col = b.n("column", 0.0, 920.0, &[("block", bk("basalt")), ("height_min", fv(3.0)), ("height_max", fv(14.0)), ("radius", fv(1.5)), ("arms", Value::B(false))]);
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
    let spike = b.n("spike", 0.0, 640.0, &[("block", bk("obsidian")), ("height_min", fv(4.0)), ("height_max", fv(12.0)), ("radius", fv(1.5))]);
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
    let vor = b.n("zone_voronoi", 300.0, 0.0, &[]);
    for (i, z) in ["temperate", "arid", "frozen", "volcanic"].iter().enumerate() {
        let n = b.n("zone", 0.0, i as f32 * 90.0, &[("zone", rv(z))]);
        b.l(n, vor, i as u8);
    }
    b.g.set(vor, "weight1", fv(1.6));
    b.g.set(vor, "weight4", fv(0.7));
    let o = b.out();
    b.l(vor, o, 0);
    let river = b.n("river", 0.0, 420.0, &[]);
    b.l(river, o, 1);
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

pub fn default_project() -> Project {
    let mut p = Project {
        graphs: vec![
            world(),
            zone(
                "temperate",
                &[("meadow", 1.0), ("oak_forest", 1.4), ("birch_hills", 1.0), ("floating_isles", 0.35)],
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
            ridge_function(),
        ],
    };
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
        assert_eq!(c.biomes.len(), 16);
        assert_eq!(c.zones.len(), 4);
        assert!(c.source.contains("wg_column"));
        // cave biomes get flat per-voxel functions, every zone selects them, rivers are wired
        for b in ["lush_caves", "crystal_caves", "magma_caves", "frost_caves"] {
            let id = c.biomes.iter().position(|i| i.name == b).unwrap();
            assert!(c.source.contains(&format!("uint b{id}_cmat(")), "{b} has no cave material");
        }
        assert!(c.source.contains("return NO_CAVE_BIOME;") && c.source.matches("_cave(vec3 p)").count() == 4);
        assert!(c.source.contains("river_mask("));
        assert!(c.source.contains("float th[2] = float[2]("), "arid zone uses humidity bands");
        assert!(c.prop_layers > 10);
        // the terrain pass looks as deep as the column's biomes need: 12 blocks of sandstone
        // under the desert, 1 + 3 blocks of grass and dirt in the meadow
        let fun = &c.source[c.source.find("float wg_depth_max(").unwrap()..];
        let fun = &fun[..fun.find("\n}").unwrap()];
        let case = |b: &str| {
            let id = c.biomes.iter().position(|i| i.name == b).unwrap();
            let line = fun.lines().find(|l| l.trim_start().starts_with(&format!("case {id}u:"))).unwrap().to_string();
            line.split("P(").skip(1).map(|t| c.params[t[..t.find('u').unwrap()].parse::<usize>().unwrap()]).collect::<Vec<f32>>()
        };
        assert!(case("desert").contains(&12.0), "{fun}");
        assert_eq!(case("meadow"), [1.0, 3.0], "{fun}");
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
