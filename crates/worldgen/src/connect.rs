//! Connections: rare world-scale features that cross layers (sinkholes, sky pillars). Every
//! layer's generator evaluates them (`wg_connect` in the generated code); the CPU mirrors their
//! placement exactly (the same hash as `shaders/noise.glsl`) so streaming can generate the open
//! air between layers only where a pillar passes, and scripts can find them.

/// `pcg3d` from `shaders/noise.glsl`.
pub fn pcg3d(v: [u32; 3]) -> [u32; 3] {
    let mut v = v.map(|x| x.wrapping_mul(1664525).wrapping_add(1013904223));
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[2]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v = v.map(|x| x ^ (x >> 16));
    v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[2]));
    v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
    v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
    v
}

fn unit(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16777216.0)
}

/// `rand2` from `shaders/noise.glsl`.
pub fn rand2(c: [i32; 2], seed: u32) -> f32 {
    unit(pcg3d([c[0] as u32, c[1] as u32, seed])[0])
}

/// `rand2v` from `shaders/noise.glsl`.
pub fn rand2v(c: [i32; 2], seed: u32) -> [f32; 2] {
    let h = pcg3d([c[0] as u32, c[1] as u32, seed ^ 0xA511_E9B3]);
    [unit(h[0]), unit(h[1])]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnectionKind {
    Sinkhole,
    SkyPillar,
}

impl ConnectionKind {
    /// Added to the seed of each kind (the generated code uses the same).
    pub fn salt(self) -> u32 {
        match self {
            ConnectionKind::Sinkhole => 4211,
            ConnectionKind::SkyPillar => 6007,
        }
    }
}

/// One Sinkholes or Sky Pillars node, its parameters resolved.
#[derive(Clone, Debug, PartialEq)]
pub struct ConnectionSpec {
    pub kind: ConnectionKind,
    pub spacing: f32,
    pub chance: f32,
    /// distance from a site's centre to the cell edge it keeps clear (`margin` in the shader)
    pub margin: f32,
    /// how far from its centre a site reaches
    pub reach: f32,
    /// heights it spans
    pub y_min: f32,
    pub y_max: f32,
    pub seed: u32,
}

impl ConnectionSpec {
    fn seed(&self, world_seed: u32) -> u32 {
        world_seed.wrapping_add(self.seed.wrapping_mul(7919)).wrapping_add(self.kind.salt())
    }

    /// The site of grid cell `cell`, if it has one: its centre (x, z).
    pub fn site(&self, world_seed: u32, cell: [i32; 2]) -> Option<[f32; 2]> {
        let seed = self.seed(world_seed);
        if rand2(cell, seed) >= self.chance {
            return None;
        }
        let j = rand2v(cell, seed);
        let room = (self.spacing - 2.0 * self.margin).max(0.0);
        Some([cell[0] as f32 * self.spacing + self.margin + j[0] * room, cell[1] as f32 * self.spacing + self.margin + j[1] * room])
    }

    /// Sites whose reach overlaps the box [min, max] (x, y, z).
    pub fn sites_in(&self, world_seed: u32, min: [f32; 3], max: [f32; 3]) -> Vec<[f32; 2]> {
        let mut out = Vec::new();
        if self.spacing <= 0.0 || max[1] < self.y_min || min[1] > self.y_max {
            return out;
        }
        let c0 = [((min[0] - self.reach) / self.spacing).floor() as i32, ((min[2] - self.reach) / self.spacing).floor() as i32];
        let c1 = [((max[0] + self.reach) / self.spacing).floor() as i32, ((max[2] + self.reach) / self.spacing).floor() as i32];
        if (c1[0] - c0[0] + 1) * (c1[1] - c0[1] + 1) > 4096 {
            return out;
        }
        for cz in c0[1]..=c1[1] {
            for cx in c0[0]..=c1[0] {
                let Some(c) = self.site(world_seed, [cx, cz]) else { continue };
                // distance from the centre to the box (xz)
                let dx = (min[0] - c[0]).max(c[0] - max[0]).max(0.0);
                let dz = (min[2] - c[1]).max(c[1] - max[2]).max(0.0);
                if dx * dx + dz * dz <= self.reach * self.reach {
                    out.push(c);
                }
            }
        }
        out
    }

    /// The site nearest (x, z) within `cells` grid cells of it.
    pub fn nearest(&self, world_seed: u32, x: f32, z: f32, cells: i32) -> Option<[f32; 2]> {
        let cx = (x / self.spacing).floor() as i32;
        let cz = (z / self.spacing).floor() as i32;
        let mut best: Option<([f32; 2], f32)> = None;
        for dz in -cells..=cells {
            for dx in -cells..=cells {
                if let Some(c) = self.site(world_seed, [cx + dx, cz + dz]) {
                    let d = (c[0] - x).powi(2) + (c[1] - z).powi(2);
                    if best.is_none_or(|(_, b)| d < b) {
                        best = Some((c, d));
                    }
                }
            }
        }
        best.map(|(c, _)| c)
    }
}

/// Whether any connection reaches the box [min, max].
pub fn any_in(specs: &[ConnectionSpec], world_seed: u32, min: [f32; 3], max: [f32; 3]) -> bool {
    specs.iter().any(|s| !s.sites_in(world_seed, min, max).is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_in_range_and_seeded() {
        assert!((0.0..1.0).contains(&rand2([-5, 7], 99)));
        assert_ne!(rand2([-5, 7], 99), rand2([-5, 7], 100));
        assert_ne!(rand2v([3, 4], 1)[0], rand2v([3, 4], 1)[1]);
    }

    #[test]
    fn sites_stay_inside_their_cells() {
        let s = ConnectionSpec { kind: ConnectionKind::SkyPillar, spacing: 1000.0, chance: 1.0, margin: 100.0, reach: 91.0, y_min: -64.0, y_max: 5000.0, seed: 2 };
        for cz in -3..3 {
            for cx in -3..3 {
                let c = s.site(1337, [cx, cz]).unwrap();
                assert!(c[0] >= cx as f32 * 1000.0 + 100.0 && c[0] <= (cx + 1) as f32 * 1000.0 - 100.0);
                // a box around the centre finds it; one far away doesn't
                assert_eq!(s.sites_in(1337, [c[0] - 1.0, 0.0, c[1] - 1.0], [c[0] + 1.0, 10.0, c[1] + 1.0]), vec![c]);
                assert!(s.sites_in(1337, [c[0] + 200.0, 0.0, c[1]], [c[0] + 201.0, 10.0, c[1] + 1.0]).is_empty());
            }
        }
        // the nearest site is the closest of all the cells around
        let n = s.nearest(1337, 10.0, 10.0, 2).unwrap();
        let d = |c: [f32; 2]| (c[0] - 10.0).powi(2) + (c[1] - 10.0).powi(2);
        let best = (-2..=2).flat_map(|z| (-2..=2).map(move |x| [x, z])).filter_map(|c| s.site(1337, c)).min_by(|a, b| d(*a).total_cmp(&d(*b))).unwrap();
        assert_eq!(n, best);
    }
}
