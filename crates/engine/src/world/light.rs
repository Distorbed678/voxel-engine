//! CPU side of the light volume: which LOD0 chunks the toroidal volume around the camera covers,
//! and which of them still need their blocks copied in (`crate::light` is the GPU side).
//!
//! Chunks get into the volume in two ways. LOD0 generation jobs for chunks inside the volume copy
//! their blocks as part of the batch, which covers edits and regeneration. Chunks that scroll into
//! the volume while already meshed get a light-only job: generate and copy, no meshing.

use std::collections::HashSet;

use glam::{DVec3, IVec3};

use super::CHUNK;

/// Volume edge length in chunks (keep `LIGHT_N` in shaders/light/light_defs.glsl in sync).
pub const LIGHT_CHUNKS: i32 = 4;
/// Volume edge length in blocks.
pub const LIGHT_N: i32 = LIGHT_CHUNKS * CHUNK;

#[derive(Default)]
pub struct LightTracker {
    /// Min chunk of the volume.
    pub origin: Option<IVec3>,
    /// Chunks inside the volume whose blocks are not in it yet.
    pending: HashSet<IVec3>,
    /// The volume moved since the last call to `take_moved`.
    moved: bool,
}

impl LightTracker {
    /// Recentre the volume on the camera (in whole chunks).
    pub fn update(&mut self, cam: DVec3) {
        let c = (cam / CHUNK as f64).floor().as_ivec3();
        let origin = c - IVec3::splat(LIGHT_CHUNKS / 2);
        if self.origin == Some(origin) {
            return;
        }
        let old = self.origin;
        for z in 0..LIGHT_CHUNKS {
            for y in 0..LIGHT_CHUNKS {
                for x in 0..LIGHT_CHUNKS {
                    let p = origin + IVec3::new(x, y, z);
                    if old.is_none_or(|o| !Self::inside(o, p)) {
                        self.pending.insert(p);
                    }
                }
            }
        }
        self.pending.retain(|p| Self::inside(origin, *p));
        self.origin = Some(origin);
        self.moved = true;
    }

    fn inside(origin: IVec3, p: IVec3) -> bool {
        let l = p - origin;
        l.cmpge(IVec3::ZERO).all() && l.cmplt(IVec3::splat(LIGHT_CHUNKS)).all()
    }

    /// Is LOD0 chunk `p` inside the volume?
    pub fn contains(&self, p: IVec3) -> bool {
        self.origin.is_some_and(|o| Self::inside(o, p))
    }

    /// A job will copy chunk `p` into the volume.
    pub fn filled(&mut self, p: IVec3) {
        self.pending.remove(&p);
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
        let mut t = LightTracker::default();
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
}
