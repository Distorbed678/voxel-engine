//! Procedural 32x32 block textures with height-derived normal maps.
//!
//! Every texture is generated from tileable noise / cellular patterns so it repeats seamlessly.
//! Output is sRGB albedo (+alpha for cutout/transparent) and a tangent-space normal map whose
//! alpha channel holds the height. Any texture can be overridden by dropping
//! `assets/textures/<name>.png` (32x32 RGBA) next to the executable's working directory.

pub const TEX: usize = 32;
const N: i32 = TEX as i32;

#[derive(Clone)]
pub struct TexImage {
    /// sRGB colour + alpha, 0..1
    pub color: Vec<[f32; 4]>,
    /// height 0..1 (for normal map generation)
    pub height: Vec<f32>,
    pub normal_strength: f32,
}

impl TexImage {
    fn new() -> Self {
        Self { color: vec![[0.0, 0.0, 0.0, 1.0]; TEX * TEX], height: vec![0.5; TEX * TEX], normal_strength: 2.0 }
    }
    fn set(&mut self, x: i32, y: i32, c: [f32; 3], a: f32, h: f32) {
        let i = idx(x, y);
        self.color[i] = [c[0].clamp(0.0, 1.0), c[1].clamp(0.0, 1.0), c[2].clamp(0.0, 1.0), a.clamp(0.0, 1.0)];
        self.height[i] = h.clamp(0.0, 1.0);
    }
    fn get(&self, x: i32, y: i32) -> [f32; 4] {
        self.color[idx(x, y)]
    }

    /// RGBA8 sRGB albedo
    pub fn albedo_rgba8(&self) -> Vec<u8> {
        self.color.iter().flat_map(|c| c.map(|v| (v * 255.0 + 0.5) as u8)).collect()
    }

    /// RGBA8 tangent-space normal (xyz) + height (w)
    pub fn normal_rgba8(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(TEX * TEX * 4);
        for y in 0..N {
            for x in 0..N {
                let h = |dx: i32, dy: i32| self.height[idx(x + dx, y + dy)];
                let dx = (h(1, 0) - h(-1, 0)) * self.normal_strength;
                let dy = (h(0, 1) - h(0, -1)) * self.normal_strength;
                let n = normalize([-dx, -dy, 1.0]);
                out.extend_from_slice(&[
                    ((n[0] * 0.5 + 0.5) * 255.0) as u8,
                    ((n[1] * 0.5 + 0.5) * 255.0) as u8,
                    ((n[2] * 0.5 + 0.5) * 255.0) as u8,
                    (self.height[idx(x, y)] * 255.0) as u8,
                ]);
            }
        }
        out
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn idx(x: i32, y: i32) -> usize {
    (y.rem_euclid(N) * N + x.rem_euclid(N)) as usize
}

// ---- tileable noise ----------------------------------------------------------------------

fn hash(x: i32, y: i32, seed: u32) -> u32 {
    let mut h = (x as u32).wrapping_mul(0x8da6b343) ^ (y as u32).wrapping_mul(0xd8163841) ^ seed.wrapping_mul(0xcb1ab31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1e995);
    h ^= h >> 15;
    h
}

fn rnd(x: i32, y: i32, seed: u32) -> f32 {
    (hash(x, y, seed) >> 8) as f32 / 16_777_216.0
}

/// Tileable value noise with `cells` lattice cells across the texture.
fn vnoise(x: f32, y: f32, cells: i32, seed: u32) -> f32 {
    let fx = x / N as f32 * cells as f32;
    let fy = y / N as f32 * cells as f32;
    let ix = fx.floor() as i32;
    let iy = fy.floor() as i32;
    let tx = fx - ix as f32;
    let ty = fy - iy as f32;
    let sx = tx * tx * (3.0 - 2.0 * tx);
    let sy = ty * ty * (3.0 - 2.0 * ty);
    let w = |a: i32, b: i32| rnd(a.rem_euclid(cells), b.rem_euclid(cells), seed);
    let a = w(ix, iy) + (w(ix + 1, iy) - w(ix, iy)) * sx;
    let b = w(ix, iy + 1) + (w(ix + 1, iy + 1) - w(ix, iy + 1)) * sx;
    a + (b - a) * sy
}

fn fbm(x: f32, y: f32, cells: i32, octaves: i32, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut amp = 1.0;
    let mut norm = 0.0;
    let mut c = cells;
    for o in 0..octaves {
        sum += vnoise(x, y, c, seed.wrapping_add(o as u32 * 101)) * amp;
        norm += amp;
        amp *= 0.5;
        c = (c * 2).min(N);
    }
    sum / norm
}

/// Tileable Worley noise: (F1, F2, cell id hash)
fn worley(x: f32, y: f32, cells: i32, seed: u32, jitter: f32) -> (f32, f32, u32) {
    let fx = x / N as f32 * cells as f32;
    let fy = y / N as f32 * cells as f32;
    let ix = fx.floor() as i32;
    let iy = fy.floor() as i32;
    let mut f1 = 1e9f32;
    let mut f2 = 1e9f32;
    let mut id = 0;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let cx = ix + dx;
            let cy = iy + dy;
            let wx = cx.rem_euclid(cells);
            let wy = cy.rem_euclid(cells);
            let px = cx as f32 + 0.5 + (rnd(wx, wy, seed) - 0.5) * jitter;
            let py = cy as f32 + 0.5 + (rnd(wx, wy, seed ^ 0x9e37) - 0.5) * jitter;
            let d = ((fx - px).powi(2) + (fy - py).powi(2)).sqrt();
            if d < f1 {
                f2 = f1;
                f1 = d;
                id = hash(wx, wy, seed ^ 0x51ed);
            } else if d < f2 {
                f2 = d;
            }
        }
    }
    (f1, f2, id)
}

// ---- colour helpers ----------------------------------------------------------------------

fn hex(c: u32) -> [f32; 3] {
    [((c >> 16) & 255) as f32 / 255.0, ((c >> 8) & 255) as f32 / 255.0, (c & 255) as f32 / 255.0]
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn shade(c: [f32; 3], k: f32) -> [f32; 3] {
    [c[0] * k, c[1] * k, c[2] * k]
}

/// Quantise a 0..1 value into `levels` steps: gives the chunky pixel-art look.
fn q(v: f32, levels: f32) -> f32 {
    (v * levels).floor() / levels
}

// ---- pattern families --------------------------------------------------------------------

struct Speckle {
    dark: u32,
    mid: u32,
    light: u32,
    cells: i32,
    grain: f32,
    seed: u32,
}

fn speckle(t: &mut TexImage, p: Speckle) {
    let (d, m, l) = (hex(p.dark), hex(p.mid), hex(p.light));
    for y in 0..N {
        for x in 0..N {
            let n = fbm(x as f32, y as f32, p.cells, 3, p.seed);
            let g = rnd(x, y, p.seed ^ 77) - 0.5;
            let v = q((n + g * p.grain).clamp(0.0, 1.0), 5.0);
            let c = if v < 0.5 { mix(d, m, v * 2.0) } else { mix(m, l, (v - 0.5) * 2.0) };
            t.set(x, y, c, 1.0, n * 0.7 + g * 0.3 + 0.2);
        }
    }
}

fn stone_like(t: &mut TexImage, dark: u32, mid: u32, light: u32, seed: u32, cracks: f32) {
    speckle(t, Speckle { dark, mid, light, cells: 4, grain: 0.35, seed });
    for y in 0..N {
        for x in 0..N {
            let (f1, f2, _) = worley(x as f32, y as f32, 5, seed ^ 0xabc, 0.9);
            let edge = f2 - f1;
            if edge < cracks {
                let c = t.get(x, y);
                let i = idx(x, y);
                t.color[i] = [c[0] * 0.72, c[1] * 0.72, c[2] * 0.72, 1.0];
                t.height[i] *= 0.5;
            }
        }
    }
}

fn cobble(t: &mut TexImage, base: u32, mortar: u32, seed: u32, moss: bool) {
    let b = hex(base);
    let mo = hex(mortar);
    for y in 0..N {
        for x in 0..N {
            let (f1, f2, id) = worley(x as f32, y as f32, 4, seed, 0.8);
            let edge = f2 - f1;
            let shadev = 0.75 + (id & 255) as f32 / 255.0 * 0.4;
            let n = fbm(x as f32, y as f32, 8, 2, seed ^ 3);
            let mut c = shade(b, shadev * (0.85 + n * 0.3));
            let mut h = (1.0 - f1 * 0.8).clamp(0.0, 1.0);
            if edge < 0.12 {
                c = mo;
                h = 0.05;
            }
            if moss && fbm(x as f32, y as f32, 4, 3, seed ^ 99) > 0.55 && edge < 0.35 {
                c = mix(hex(0x4f7a2a), hex(0x6c9a3a), rnd(x, y, 5));
                h += 0.1;
            }
            t.set(x, y, shade(c, 1.0), 1.0, h);
        }
    }
}

fn bricks(t: &mut TexImage, brick: u32, mortar: u32, rows: i32, cols: i32, seed: u32) {
    let bh = N / rows;
    let bw = N / cols;
    for y in 0..N {
        for x in 0..N {
            let row = y / bh;
            let off = if row % 2 == 0 { 0 } else { bw / 2 };
            let xx = (x + off).rem_euclid(N);
            let col = xx / bw;
            let mortar_px = y % bh == 0 || xx % bw == 0;
            let id = hash(col, row, seed);
            let n = fbm(x as f32, y as f32, 8, 2, seed);
            if mortar_px {
                t.set(x, y, shade(hex(mortar), 0.9 + n * 0.2), 1.0, 0.1);
            } else {
                let k = 0.82 + (id & 255) as f32 / 255.0 * 0.3 + (rnd(x, y, seed) - 0.5) * 0.12;
                t.set(x, y, shade(hex(brick), k), 1.0, 0.8 + n * 0.2);
            }
        }
    }
}

fn planks(t: &mut TexImage, base: u32, seed: u32) {
    let b = hex(base);
    for y in 0..N {
        for x in 0..N {
            let board = y / 8;
            let id = hash(board, 0, seed);
            let grain = vnoise(x as f32 * 0.25, y as f32 * 4.0, 8, seed ^ board as u32);
            let k = 0.85 + (id & 255) as f32 / 255.0 * 0.2 + (grain - 0.5) * 0.25;
            let gap = y % 8 == 0;
            let end = (x + (id as i32 & 31)).rem_euclid(N) == 0 && board % 2 == 0;
            let nail = (y % 8 == 3 || y % 8 == 5) && ((x + 2) % 16 == 0) && y % 8 == 3;
            let mut c = shade(b, q(k, 8.0));
            let mut h = 0.7 + grain * 0.2;
            if gap || end {
                c = shade(b, 0.55);
                h = 0.1;
            }
            if nail {
                c = hex(0x5a4a3a);
            }
            t.set(x, y, c, 1.0, h);
        }
    }
}

fn log_side(t: &mut TexImage, bark: u32, dark: u32, seed: u32, birch: bool) {
    let b = hex(bark);
    let d = hex(dark);
    for y in 0..N {
        for x in 0..N {
            let stripe = vnoise(x as f32 * 2.0, y as f32 * 0.25, 8, seed);
            let n = fbm(x as f32, y as f32, 8, 2, seed ^ 7);
            let mut c;
            let mut h = 0.5 + stripe * 0.5;
            if birch {
                c = mix(hex(0xe8e4d8), hex(0xcfcabb), n);
                let mark = vnoise(x as f32, y as f32 * 3.0, 4, seed ^ 55);
                if mark > 0.72 && (y % 7 < 2) {
                    c = hex(0x2d2a26);
                    h = 0.2;
                }
            } else {
                c = mix(d, b, q(stripe * 0.7 + n * 0.3, 5.0));
                if stripe < 0.25 {
                    h = 0.1;
                }
            }
            t.set(x, y, c, 1.0, h);
        }
    }
}

fn log_top(t: &mut TexImage, wood: u32, ring: u32, bark: u32, seed: u32) {
    for y in 0..N {
        for x in 0..N {
            let dx = x as f32 - 15.5;
            let dy = y as f32 - 15.5;
            let r = (dx * dx + dy * dy).sqrt() + (vnoise(x as f32, y as f32, 4, seed) - 0.5) * 2.0;
            let edge = dx.abs().max(dy.abs());
            
            
            let (c, h) = if edge > 13.5 {
                (shade(hex(bark), 0.9 + rnd(x, y, seed) * 0.2), 0.6)
            } else {
                let rings = (r * 0.8).sin() * 0.5 + 0.5;
                (mix(hex(ring), hex(wood), q(rings, 4.0)), 0.5 + rings * 0.2)
            };
            t.set(x, y, c, 1.0, h);
        }
    }
}

fn leaves(t: &mut TexImage, dark: u32, light: u32, seed: u32, hole: f32) {
    for y in 0..N {
        for x in 0..N {
            let (f1, _f2, id) = worley(x as f32, y as f32, 8, seed, 1.0);
            let n = fbm(x as f32, y as f32, 8, 2, seed ^ 5);
            let k = (id & 255) as f32 / 255.0;
            let leaf = 1.0 - f1;
            let a = if rnd(x, y, seed ^ 11) < hole && f1 > 0.45 { 0.0 } else { 1.0 };
            let c = mix(hex(dark), hex(light), q(leaf * 0.6 + k * 0.25 + n * 0.15, 5.0));
            t.set(x, y, c, a, leaf);
        }
    }
    t.normal_strength = 3.0;
}

fn ore(t: &mut TexImage, ore: u32, ore_light: u32, seed: u32) {
    stone_like(t, 0x6b6b6b, 0x7f7f7f, 0x939393, 11, 0.06);
    for y in 0..N {
        for x in 0..N {
            let (f1, _, id) = worley(x as f32, y as f32, 5, seed, 0.9);
            if f1 < 0.28 && (id & 3) != 0 {
                let c = if f1 < 0.14 { hex(ore_light) } else { hex(ore) };
                t.set(x, y, shade(c, 0.9 + rnd(x, y, seed) * 0.2), 1.0, 0.9);
            }
        }
    }
}

fn flat(t: &mut TexImage, base: u32, var: f32, seed: u32) {
    for y in 0..N {
        for x in 0..N {
            let n = fbm(x as f32, y as f32, 4, 3, seed);
            let g = rnd(x, y, seed) - 0.5;
            t.set(x, y, shade(hex(base), 1.0 + (n - 0.5) * var + g * var * 0.5), 1.0, 0.5 + n * 0.3);
        }
    }
}

fn grass_top(t: &mut TexImage, dark: u32, mid: u32, light: u32, seed: u32) {
    for y in 0..N {
        for x in 0..N {
            let n = fbm(x as f32, y as f32, 4, 3, seed);
            let blade = vnoise(x as f32 * 3.0, y as f32 * 1.0, 16, seed ^ 3);
            let g = rnd(x, y, seed) * 0.3;
            let v = q((n * 0.5 + blade * 0.35 + g).clamp(0.0, 1.0), 5.0);
            let c = if v < 0.5 { mix(hex(dark), hex(mid), v * 2.0) } else { mix(hex(mid), hex(light), v * 2.0 - 1.0) };
            t.set(x, y, c, 1.0, blade * 0.6 + n * 0.4);
        }
    }
}

fn overlay_side(t: &mut TexImage, top: u32, top_light: u32, depth: f32, seed: u32) {
    for x in 0..N {
        let fringe = depth + vnoise(x as f32 * 2.0, 0.0, 16, seed) * 5.0 + rnd(x, 0, seed) * 2.0;
        for y in 0..N {
            if (y as f32) < fringe {
                let k = rnd(x, y, seed ^ 9);
                t.set(x, y, mix(hex(top), hex(top_light), q(k, 3.0)), 1.0, 0.8);
            }
        }
    }
}

fn plant(t: &mut TexImage, stem: u32, head: Option<(u32, u32)>, seed: u32, blades: i32) {
    for i in 0..TEX * TEX {
        t.color[i] = [0.0, 0.0, 0.0, 0.0];
        t.height[i] = 0.0;
    }
    for b in 0..blades {
        let bx = 4 + (hash(b, 0, seed) % 24) as i32;
        let height = 14 + (hash(b, 1, seed) % 16) as i32;
        let lean = ((hash(b, 2, seed) % 7) as f32 - 3.0) * 0.18;
        for k in 0..height {
            let y = N - 1 - k;
            let x = bx + (lean * k as f32) as i32;
            if (0..N).contains(&x) {
                let c = shade(hex(stem), 0.75 + k as f32 / height as f32 * 0.45);
                t.set(x, y, c, 1.0, 0.6);
                if k < height / 2 && x + 1 < N {
                    t.set(x + 1, y, shade(c, 0.85), 1.0, 0.5);
                }
            }
        }
        if let Some((h1, h2)) = head {
            let hx = bx + (lean * height as f32) as i32;
            let hy = N - height;
            for dy in -2..=2 {
                for dx in -2..=2 {
                    if dx * dx + dy * dy <= 5 {
                        let x = hx + dx;
                        let y = hy + dy;
                        if (0..N).contains(&x) && (0..N).contains(&y) {
                            let c = if dx == 0 && dy == 0 { hex(h2) } else { hex(h1) };
                            t.set(x, y, shade(c, 0.9 + rnd(x, y, seed) * 0.2), 1.0, 0.8);
                        }
                    }
                }
            }
        }
    }
}

fn glass(t: &mut TexImage) {
    for y in 0..N {
        for x in 0..N {
            let border = x == 0 || y == 0 || x == N - 1 || y == N - 1;
            let streak = (x - y).rem_euclid(13) == 0 && (4..12).contains(&x);
            if border {
                t.set(x, y, hex(0xc8dde0), 1.0, 0.8);
            } else if streak {
                t.set(x, y, hex(0xf2fbff), 0.55, 0.5);
            } else {
                t.set(x, y, hex(0xd5ecf2), 0.12, 0.5);
            }
        }
    }
}

fn crystal(t: &mut TexImage, dark: u32, light: u32, seed: u32) {
    for y in 0..N {
        for x in 0..N {
            let (f1, f2, id) = worley(x as f32, y as f32, 4, seed, 1.0);
            let facet = (id & 255) as f32 / 255.0;
            let edge = f2 - f1;
            let mut c = mix(hex(dark), hex(light), q(facet * 0.7 + (1.0 - f1) * 0.3, 4.0));
            if edge < 0.08 {
                c = shade(hex(light), 1.15);
            }
            t.set(x, y, c, 1.0, 1.0 - f1 * 0.7);
        }
    }
    t.normal_strength = 4.0;
}

fn fluid(t: &mut TexImage, deep: u32, light: u32, alpha: f32, seed: u32) {
    for y in 0..N {
        for x in 0..N {
            let n = fbm(x as f32, y as f32, 4, 3, seed);
            let w = vnoise(x as f32 + n * 6.0, y as f32 * 2.0, 4, seed ^ 3);
            let c = mix(hex(deep), hex(light), q(w * 0.7 + n * 0.3, 6.0));
            t.set(x, y, c, alpha, w);
        }
    }
    t.normal_strength = 1.0;
}

fn generate_builtin(name: &str) -> TexImage {
    let mut t = TexImage::new();
    let s = hash(name.len() as i32, name.bytes().map(|b| b as i32).sum(), 1234);
    match name {
        "stone" => stone_like(&mut t, 0x6e6e6e, 0x828282, 0x989898, 1, 0.05),
        "deepslate" => stone_like(&mut t, 0x3b3b40, 0x4b4b52, 0x5c5c63, 2, 0.08),
        "granite" => speckle(&mut t, Speckle { dark: 0x7d4f3f, mid: 0x9a6a55, light: 0xb98a73, cells: 8, grain: 0.6, seed: 3 }),
        "andesite" => speckle(&mut t, Speckle { dark: 0x707070, mid: 0x858585, light: 0xa0a0a0, cells: 8, grain: 0.7, seed: 4 }),
        "calcite" => speckle(&mut t, Speckle { dark: 0xc9c9c4, mid: 0xdcdcd6, light: 0xf0f0ea, cells: 4, grain: 0.3, seed: 5 }),
        "dirt" => speckle(&mut t, Speckle { dark: 0x5a3e28, mid: 0x6f4d33, light: 0x86603f, cells: 8, grain: 0.6, seed: 6 }),
        "coarse_dirt" => {
            speckle(&mut t, Speckle { dark: 0x4f3725, mid: 0x6a4a31, light: 0x7d5b3d, cells: 8, grain: 0.7, seed: 7 });
            for y in 0..N {
                for x in 0..N {
                    if rnd(x, y, 99) > 0.88 {
                        t.set(x, y, hex(0x8a8a86), 1.0, 0.9);
                    }
                }
            }
        }
        "mud" => speckle(&mut t, Speckle { dark: 0x3b2f2a, mid: 0x4a3b33, light: 0x5a4a40, cells: 4, grain: 0.3, seed: 8 }),
        "grass_top" => grass_top(&mut t, 0x3f7d22, 0x5a9e2e, 0x78bb3e, 9),
        "dry_grass_top" => grass_top(&mut t, 0x8a8a3a, 0xa8a249, 0xc4bb62, 10),
        "podzol_top" => speckle(&mut t, Speckle { dark: 0x4a3018, mid: 0x6b4520, light: 0x8a5e2c, cells: 8, grain: 0.7, seed: 11 }),
        "moss" => grass_top(&mut t, 0x3d5e1f, 0x507a28, 0x699a34, 12),
        "grass_side" | "dry_grass_side" | "podzol_side" | "snowy_grass_side" => {
            speckle(&mut t, Speckle { dark: 0x5a3e28, mid: 0x6f4d33, light: 0x86603f, cells: 8, grain: 0.6, seed: 6 });
            match name {
                "grass_side" => overlay_side(&mut t, 0x4f9128, 0x6aaf35, 4.0, 13),
                "dry_grass_side" => overlay_side(&mut t, 0x9c9640, 0xbcb158, 4.0, 14),
                "podzol_side" => overlay_side(&mut t, 0x5c3a1c, 0x7a5126, 4.0, 15),
                _ => overlay_side(&mut t, 0xe8eef2, 0xffffff, 6.0, 16),
            }
        }
        "sand" => speckle(&mut t, Speckle { dark: 0xc8b47e, mid: 0xd8c690, light: 0xe8d8a6, cells: 8, grain: 0.6, seed: 17 }),
        "red_sand" => speckle(&mut t, Speckle { dark: 0xa45222, mid: 0xbb632b, light: 0xd07938, cells: 8, grain: 0.6, seed: 18 }),
        "sandstone" => {
            speckle(&mut t, Speckle { dark: 0xc9b47c, mid: 0xd8c48c, light: 0xe6d49e, cells: 4, grain: 0.3, seed: 19 });
            for y in 0..N {
                if y % 8 == 0 || y % 8 == 7 {
                    for x in 0..N {
                        let c = t.get(x, y);
                        t.set(x, y, [c[0] * 0.88, c[1] * 0.88, c[2] * 0.88], 1.0, 0.3);
                    }
                }
            }
        }
        "gravel" => cobble(&mut t, 0x7c7672, 0x5d5754, 20, false),
        "clay" => flat(&mut t, 0x9da3b0, 0.12, 21),
        "snow" => speckle(&mut t, Speckle { dark: 0xdfe8ef, mid: 0xeef4f8, light: 0xffffff, cells: 4, grain: 0.25, seed: 22 }),
        "ice" => {
            speckle(&mut t, Speckle { dark: 0x8fb8e8, mid: 0xa7caf0, light: 0xc6def8, cells: 4, grain: 0.2, seed: 23 });
            for i in 0..TEX * TEX {
                t.color[i][3] = 0.7;
            }
        }
        "packed_ice" => stone_like(&mut t, 0x86aee0, 0x98bce8, 0xb0cff2, 24, 0.04),
        "water" => fluid(&mut t, 0x123f86, 0x2a62b0, 0.78, 25),
        "lava" => {
            fluid(&mut t, 0xc0300a, 0xffa020, 1.0, 26);
            for y in 0..N {
                for x in 0..N {
                    let (f1, f2, _) = worley(x as f32, y as f32, 4, 27, 1.0);
                    if f2 - f1 < 0.12 {
                        t.set(x, y, hex(0xffe070), 1.0, 0.9);
                    }
                }
            }
        }
        "terracotta" => flat(&mut t, 0x985e43, 0.1, 28),
        "terracotta_orange" => flat(&mut t, 0xa1531f, 0.1, 29),
        "terracotta_red" => flat(&mut t, 0x8e3b2e, 0.1, 30),
        "terracotta_yellow" => flat(&mut t, 0xba8523, 0.1, 31),
        "terracotta_brown" => flat(&mut t, 0x4d3324, 0.1, 32),
        "terracotta_white" => flat(&mut t, 0xd1b2a1, 0.08, 33),
        "basalt_side" => {
            for y in 0..N {
                for x in 0..N {
                    let col = vnoise(x as f32 * 2.0, y as f32 * 0.2, 8, 34);
                    let c = mix(hex(0x2f2f33), hex(0x56565c), q(col, 5.0));
                    t.set(x, y, shade(c, 0.9 + rnd(x, y, 34) * 0.2), 1.0, col);
                }
            }
        }
        "basalt_top" => cobble(&mut t, 0x4a4a50, 0x27272b, 35, false),
        "ash" => speckle(&mut t, Speckle { dark: 0x5c5a58, mid: 0x6e6b68, light: 0x84807b, cells: 8, grain: 0.7, seed: 36 }),
        "magma" => {
            stone_like(&mut t, 0x2a1810, 0x3a2016, 0x4a2a1c, 37, 0.0);
            for y in 0..N {
                for x in 0..N {
                    let (f1, f2, _) = worley(x as f32, y as f32, 5, 38, 0.9);
                    let e = f2 - f1;
                    if e < 0.16 {
                        t.set(x, y, mix(hex(0xffc040), hex(0xff5a10), e / 0.16), 1.0, 0.1);
                    }
                }
            }
        }
        "obsidian" => speckle(&mut t, Speckle { dark: 0x0f0a1a, mid: 0x1a1229, light: 0x3a2a5a, cells: 8, grain: 0.3, seed: 39 }),
        "coal_ore" => ore(&mut t, 0x1e1e1e, 0x3a3a3a, 40),
        "iron_ore" => ore(&mut t, 0xc8a68a, 0xe2c8ae, 41),
        "gold_ore" => ore(&mut t, 0xe0b020, 0xfff07a, 42),
        "copper_ore" => ore(&mut t, 0xc0683a, 0x5ab89a, 43),
        "crystal" => crystal(&mut t, 0x7a4cc0, 0xd2a8ff, 44),
        "crystal_shell" => cobble(&mut t, 0x5a4a6a, 0x3a2e48, 45, false),
        "oak_log" => log_side(&mut t, 0x6b5132, 0x46341f, 46, false),
        "oak_log_top" => log_top(&mut t, 0xb08c5a, 0x8c6c42, 0x5a4329, 47),
        "birch_log" => log_side(&mut t, 0xe0dccf, 0x2d2a26, 48, true),
        "birch_log_top" => log_top(&mut t, 0xd6c79a, 0xbca97a, 0xe0dccf, 49),
        "spruce_log" => log_side(&mut t, 0x4a3420, 0x2e2014, 50, false),
        "spruce_log_top" => log_top(&mut t, 0x9a7448, 0x7a5a36, 0x3a2918, 51),
        "acacia_log" => log_side(&mut t, 0x6e6458, 0x4a433b, 52, false),
        "acacia_log_top" => log_top(&mut t, 0xc06a3a, 0xa0552c, 0x5e554b, 53),
        "palm_log" => log_side(&mut t, 0x8a7a5a, 0x5e533c, 54, false),
        "palm_log_top" => log_top(&mut t, 0xc0a878, 0xa08c60, 0x6e6248, 55),
        "charred_log" => log_side(&mut t, 0x2a2420, 0x121010, 56, false),
        "charred_log_top" => log_top(&mut t, 0x3a2a20, 0x2a1e18, 0x1a1614, 57),
        "oak_leaves" => leaves(&mut t, 0x2f6a1c, 0x5a9e32, 58, 0.25),
        "birch_leaves" => leaves(&mut t, 0x4f7f2a, 0x86b84a, 59, 0.25),
        "spruce_leaves" => leaves(&mut t, 0x1f4a2a, 0x386a40, 60, 0.18),
        "acacia_leaves" => leaves(&mut t, 0x5a7a1a, 0x8aaa2a, 61, 0.3),
        "palm_leaves" => leaves(&mut t, 0x3a7a1a, 0x6aaa2a, 62, 0.3),
        "cactus_side" => {
            for y in 0..N {
                for x in 0..N {
                    let rib = ((x as f32 / 32.0 * std::f32::consts::TAU * 4.0).cos() * 0.5 + 0.5).powf(2.0);
                    let c = mix(hex(0x2f6a2a), hex(0x5aa040), q(rib, 4.0));
                    let spine = (x % 8 == 4) && (y % 6 == (x / 8) % 6) && x > 0;
                    if spine {
                        t.set(x, y, hex(0xe8e0b0), 1.0, 1.0);
                    } else {
                        t.set(x, y, shade(c, 0.9 + rnd(x, y, 63) * 0.15), 1.0, rib);
                    }
                }
            }
        }
        "cactus_top" => log_top(&mut t, 0x6ab048, 0x4a8a34, 0x3a7a2a, 64),
        "mushroom_stem" => speckle(&mut t, Speckle { dark: 0xc8c0aa, mid: 0xdcd4be, light: 0xeee8d6, cells: 4, grain: 0.3, seed: 65 }),
        "mushroom_stem_top" => log_top(&mut t, 0xcfc7a8, 0xbab295, 0xdcd4be, 66),
        "mushroom_cap_red" => {
            flat(&mut t, 0xb02a24, 0.15, 67);
            for y in 0..N {
                for x in 0..N {
                    let (f1, _, id) = worley(x as f32, y as f32, 3, 68, 0.8);
                    if f1 < 0.22 && id & 1 == 0 {
                        t.set(x, y, hex(0xf0e8e0), 1.0, 0.9);
                    }
                }
            }
        }
        "mushroom_cap_glow" => crystal(&mut t, 0x2a8a9a, 0x8af0ff, 69),
        "cobblestone" => cobble(&mut t, 0x7a7a7a, 0x4a4a4a, 70, false),
        "mossy_cobblestone" => cobble(&mut t, 0x7a7a7a, 0x4a4a4a, 70, true),
        "planks" => planks(&mut t, 0xa07a48, 71),
        "bricks" => bricks(&mut t, 0x9a4a38, 0xb8b0a0, 4, 2, 72),
        "stone_bricks" => bricks(&mut t, 0x7e7e7e, 0x5a5a5a, 2, 1, 73),
        "glass" => glass(&mut t),
        "lamp" => {
            crystal(&mut t, 0xd09030, 0xfff0a0, 74);
            for i in 0..N {
                t.set(i, 0, hex(0x5a4a30), 1.0, 1.0);
                t.set(i, N - 1, hex(0x5a4a30), 1.0, 1.0);
                t.set(0, i, hex(0x5a4a30), 1.0, 1.0);
                t.set(N - 1, i, hex(0x5a4a30), 1.0, 1.0);
            }
        }
        "lantern" => {
            flat(&mut t, 0x3a3a40, 0.2, 75);
            for y in 8..24 {
                for x in 9..23 {
                    t.set(x, y, mix(hex(0xffb040), hex(0xfff0b0), rnd(x, y, 75)), 1.0, 0.6);
                }
            }
        }
        "tall_grass" => plant(&mut t, 0x5a9e2e, None, 76, 9),
        "fern" => plant(&mut t, 0x3f7d22, None, 77, 6),
        "flower_red" => plant(&mut t, 0x3f7d22, Some((0xd02a2a, 0xffe060)), 78, 3),
        "flower_yellow" => plant(&mut t, 0x3f7d22, Some((0xf0d020, 0xb06010)), 79, 3),
        "flower_blue" => plant(&mut t, 0x3f7d22, Some((0x4060e0, 0xf0f0ff)), 80, 3),
        "flower_white" => plant(&mut t, 0x3f7d22, Some((0xf4f4f0, 0xe0c020)), 81, 3),
        "dead_bush" => plant(&mut t, 0x7a5a32, None, 82, 7),
        "dry_shrub" => plant(&mut t, 0x9a8a4a, None, 83, 10),
        "snow_shrub" => plant(&mut t, 0xc8dce8, None, 84, 8),
        "ember_weed" => plant(&mut t, 0x3a2a20, Some((0xff6a20, 0xffd060)), 85, 4),
        _ => speckle(&mut t, Speckle { dark: 0xff00ff, mid: 0x800080, light: 0xff80ff, cells: 4, grain: 0.5, seed: s }),
    }
    t
}

/// Generate a texture, preferring a user override PNG if present.
pub fn generate(name: &str) -> TexImage {
    let path = std::path::Path::new("assets/textures").join(format!("{name}.png"));
    if let Ok(img) = image::open(&path) {
        let img = img.resize_exact(TEX as u32, TEX as u32, image::imageops::FilterType::Nearest).to_rgba8();
        let mut t = TexImage::new();
        for (i, p) in img.pixels().enumerate() {
            t.color[i] = p.0.map(|v| v as f32 / 255.0);
            t.height[i] = (t.color[i][0] + t.color[i][1] + t.color[i][2]) / 3.0;
        }
        return t;
    }
    generate_builtin(name)
}

/// Box-filtered mip chain of RGBA8 data (alpha-coverage preserving for cutouts).
pub fn mip_chain(base: &[u8], srgb: bool) -> Vec<Vec<u8>> {
    let to_lin = |v: u8| {
        let c = v as f32 / 255.0;
        if srgb { c.powf(2.2) } else { c }
    };
    let from_lin = |v: f32| {
        let c = if srgb { v.max(0.0).powf(1.0 / 2.2) } else { v };
        (c.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    };
    let coverage = |data: &[u8]| data.chunks(4).filter(|p| p[3] >= 128).count() as f32 / (data.len() / 4) as f32;
    let base_cov = coverage(base);
    let has_cutout = srgb && base.chunks(4).any(|p| p[3] < 128) && base.chunks(4).any(|p| p[3] >= 128);
    let mut mips = vec![base.to_vec()];
    let mut size = TEX;
    while size > 1 {
        let prev = mips.last().unwrap();
        let ns = size / 2;
        let mut out = vec![0u8; ns * ns * 4];
        for y in 0..ns {
            for x in 0..ns {
                for c in 0..4 {
                    let mut acc = 0.0;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let v = prev[((y * 2 + dy) * size + x * 2 + dx) * 4 + c];
                        acc += if c == 3 { v as f32 / 255.0 } else { to_lin(v) };
                    }
                    acc *= 0.25;
                    out[(y * ns + x) * 4 + c] = if c == 3 { (acc * 255.0 + 0.5) as u8 } else { from_lin(acc) };
                }
            }
        }
        if has_cutout {
            // scale alpha so that the fraction of texels passing the 0.5 test matches mip 0
            let mut lo = 0.5f32;
            let mut hi = 4.0f32;
            let alphas: Vec<f32> = out.chunks(4).map(|p| p[3] as f32 / 255.0).collect();
            for _ in 0..12 {
                let mid = (lo + hi) * 0.5;
                let cov = alphas.iter().filter(|a| *a * mid >= 0.5).count() as f32 / alphas.len() as f32;
                if cov < base_cov { lo = mid } else { hi = mid }
            }
            let scale = (lo + hi) * 0.5;
            for (i, a) in alphas.iter().enumerate() {
                out[i * 4 + 3] = ((a * scale).min(1.0) * 255.0) as u8;
            }
        }
        mips.push(out);
        size = ns;
    }
    mips
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_builtins_tile() {
        for name in ["stone", "grass_top", "oak_leaves", "planks", "water"] {
            let t = generate_builtin(name);
            assert_eq!(t.color.len(), TEX * TEX);
        }
        assert_eq!(mip_chain(&vec![255u8; TEX * TEX * 4], true).len(), 6);
    }
}
