//! CPU side of the light volumes: which LOD0 chunks each toroidal volume around the camera
//! covers, and which still need their blocks copied in (`crate::light` is the GPU side).
//!
//! - The **fine** volume (128³ blocks, ±64 around the camera) holds block light and sky light at
//!   block resolution.
//! - The **coarse** volume (4 blocks per cell, 512×256×512 blocks) holds block light only. It
//!   matches the LOD0 streaming box at default settings, so distant lava still lights its banks.
//!
//! Chunks get into a volume in two ways. LOD0 generation jobs for chunks inside it copy their
//! blocks as part of the batch, which covers edits and regeneration. Chunks that scroll into a
//! volume while already meshed get a light-only job: generate and copy, no meshing.

use std::collections::HashSet;

use glam::{DVec3, IVec3};

use super::CHUNK;

/// Fine volume edge length in chunks (keep `LIGHT_N` in shaders/light/light_defs.glsl in sync).
pub const LIGHT_CHUNKS: i32 = 4;
/// Fine volume edge length in blocks.
pub const LIGHT_N: i32 = LIGHT_CHUNKS * CHUNK;
/// Coarse volume size in chunks (the default LOD0 box: 2R × 2Rv × 2R with R = 8, Rv = 4).
pub const COARSE_CHUNKS: IVec3 = IVec3::new(16, 8, 16);
/// Blocks per coarse cell (keep `COARSE_CELL` in light_defs.glsl in sync).
pub const COARSE_CELL: i32 = 4;

pub struct LightTracker {
    size: IVec3,
    /// Snap the centre to even chunks, like the LOD0 streaming box.
    snap_even: bool,
    /// Min chunk of the volume.
    pub origin: Option<IVec3>,
    /// Chunks inside the volume whose blocks are not in it yet.
    pending: HashSet<IVec3>,
    /// The volume moved since the last call to `take_moved`.
    moved: bool,
}

impl LightTracker {
    pub fn fine() -> Self {
        Self::new(IVec3::splat(LIGHT_CHUNKS), false)
    }

    pub fn coarse() -> Self {
        Self::new(COARSE_CHUNKS, true)
    }

    fn new(size: IVec3, snap_even: bool) -> Self {
        Self { size, snap_even, origin: None, pending: HashSet::new(), moved: false }
    }

    /// Recentre the volume on the camera (in whole chunks).
    pub fn update(&mut self, cam: DVec3) {
        let mut c = (cam / CHUNK as f64).floor().as_ivec3();
        if self.snap_even {
            c = c.div_euclid(IVec3::splat(2)) * 2;
        }
        let origin = c - self.size / 2;
        if self.origin == Some(origin) {
            return;
        }
        let old = self.origin;
        for z in 0..self.size.z {
            for y in 0..self.size.y {
                for x in 0..self.size.x {
                    let p = origin + IVec3::new(x, y, z);
                    if old.is_none_or(|o| !self.inside(o, p)) {
                        self.pending.insert(p);
                    }
                }
            }
        }
        let size = self.size;
        self.pending.retain(|p| {
            let l = *p - origin;
            l.cmpge(IVec3::ZERO).all() && l.cmplt(size).all()
        });
        self.origin = Some(origin);
        self.moved = true;
    }

    fn inside(&self, origin: IVec3, p: IVec3) -> bool {
        let l = p - origin;
        l.cmpge(IVec3::ZERO).all() && l.cmplt(self.size).all()
    }

    /// Is LOD0 chunk `p` inside the volume?
    pub fn contains(&self, p: IVec3) -> bool {
        self.origin.is_some_and(|o| self.inside(o, p))
    }

    /// A job will copy chunk `p` into the volume.
    pub fn filled(&mut self, p: IVec3) {
        self.pending.remove(&p);
    }

    pub fn is_pending(&self, p: IVec3) -> bool {
        self.pending.contains(&p)
    }

    /// Chunks still waiting for their blocks.
    pub fn pending(&self) -> impl Iterator<Item = IVec3> + '_ {
        self.pending.iter().copied()
    }

    pub fn take_moved(&mut self) -> bool {
        std::mem::take(&mut self.moved)
    }

    /// World block coordinate of the volume's min corner.
    pub fn origin_block(&self) -> Option<IVec3> {
        self.origin.map(|o| o * CHUNK)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrolling_queues_only_new_chunks() {
        let mut t = LightTracker::fine();
        t.update(DVec3::new(10.0, 10.0, 10.0));
        assert_eq!(t.pending().count(), (LIGHT_CHUNKS * LIGHT_CHUNKS * LIGHT_CHUNKS) as usize);
        for p in t.pending().collect::<Vec<_>>() {
            t.filled(p);
        }
        // one chunk along +X: one new slab of LIGHT_CHUNKS^2 chunks
        t.update(DVec3::new(10.0 + CHUNK as f64, 10.0, 10.0));
        assert_eq!(t.pending().count(), (LIGHT_CHUNKS * LIGHT_CHUNKS) as usize);
        let o = t.origin.unwrap();
        assert!(t.pending().all(|p| p.x == o.x + LIGHT_CHUNKS - 1));
        assert!(t.take_moved());
        assert!(!t.take_moved());
    }

    #[test]
    fn coarse_volume_matches_lod0_box() {
        // the LOD0 streaming box: centre chunk snapped to even, ±R (8) and ±Rv (4) chunks
        let cam = DVec3::new(100.0, -40.0, 1000.0);
        let mut t = LightTracker::coarse();
        t.update(cam);
        let c = (cam / CHUNK as f64).floor().as_ivec3().div_euclid(IVec3::splat(2)) * 2;
        assert_eq!(t.origin, Some(c - IVec3::new(8, 4, 8)));
        // moving within the same even pair of chunks doesn't scroll it
        t.take_moved();
        t.update(cam + DVec3::new(CHUNK as f64 * 0.9, 0.0, 0.0));
        let moved = t.take_moved();
        let c2 = ((cam + DVec3::new(CHUNK as f64 * 0.9, 0.0, 0.0)) / CHUNK as f64).floor().as_ivec3().div_euclid(IVec3::splat(2)) * 2;
        assert_eq!(moved, c2 != c);
    }
}
