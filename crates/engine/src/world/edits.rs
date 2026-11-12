//! Sparse edit overlay on top of procedural generation, with undo/redo.
//!
//! The world is always `generate(graph, seed) + overlay`. Regenerating a chunk therefore
//! automatically keeps every placed/broken block, and undo simply restores overlay entries.

use std::collections::HashMap;

use glam::IVec3;
use serde::{Deserialize, Serialize};

pub const CHUNK: i32 = 32;

pub fn chunk_of(p: IVec3) -> IVec3 {
    p.div_euclid(IVec3::splat(CHUNK))
}

pub fn local_index(p: IVec3) -> u16 {
    let l = p.rem_euclid(IVec3::splat(CHUNK));
    (l.x | (l.z << 5) | (l.y << 10)) as u16
}

pub fn local_from_index(i: u16) -> IVec3 {
    let i = i as i32;
    IVec3::new(i & 31, (i >> 10) & 31, (i >> 5) & 31)
}

#[derive(Clone, Debug, Default)]
pub struct EditOp {
    pub label: String,
    /// (position, overlay value before, overlay value after). `None` = procedural.
    pub changes: Vec<(IVec3, Option<u16>, Option<u16>)>,
}

#[derive(Default, Serialize, Deserialize)]
pub struct EditOverlay {
    /// LOD0 chunk -> (local index -> block id)
    pub chunks: HashMap<IVec3, HashMap<u16, u16>>,
    #[serde(skip)]
    pub undo: Vec<EditOp>,
    #[serde(skip)]
    pub redo: Vec<EditOp>,
    #[serde(skip)]
    pub revision: u64,
}

impl EditOverlay {
    pub fn get(&self, p: IVec3) -> Option<u16> {
        self.chunks.get(&chunk_of(p)).and_then(|c| c.get(&local_index(p)).copied())
    }

    fn set_raw(&mut self, p: IVec3, v: Option<u16>) {
        let c = chunk_of(p);
        match v {
            Some(b) => {
                self.chunks.entry(c).or_default().insert(local_index(p), b);
            }
            None => {
                if let Some(m) = self.chunks.get_mut(&c) {
                    m.remove(&local_index(p));
                    if m.is_empty() {
                        self.chunks.remove(&c);
                    }
                }
            }
        }
    }

    /// Write a block outside the undo history (simulated water). `None` = back to generated.
    pub fn set_untracked(&mut self, p: IVec3, v: Option<u16>) {
        self.set_raw(p, v);
        self.revision += 1;
    }

    /// Apply a batch of block writes as one undoable operation. Returns touched positions.
    pub fn apply(&mut self, label: &str, writes: &[(IVec3, u16)]) -> Vec<IVec3> {
        let mut op = EditOp { label: label.to_string(), changes: Vec::with_capacity(writes.len()) };
        let mut touched = Vec::with_capacity(writes.len());
        for &(p, b) in writes {
            let before = self.get(p);
            if before == Some(b) {
                continue;
            }
            self.set_raw(p, Some(b));
            op.changes.push((p, before, Some(b)));
            touched.push(p);
        }
        if !op.changes.is_empty() {
            self.undo.push(op);
            self.redo.clear();
            if self.undo.len() > 512 {
                self.undo.remove(0);
            }
            self.revision += 1;
        }
        touched
    }

    pub fn undo(&mut self) -> Option<(String, Vec<IVec3>)> {
        let op = self.undo.pop()?;
        let mut touched = Vec::new();
        for &(p, before, _) in op.changes.iter().rev() {
            self.set_raw(p, before);
            touched.push(p);
        }
        let label = op.label.clone();
        self.redo.push(op);
        self.revision += 1;
        Some((label, touched))
    }

    pub fn redo(&mut self) -> Option<(String, Vec<IVec3>)> {
        let op = self.redo.pop()?;
        let mut touched = Vec::new();
        for &(p, _, after) in &op.changes {
            self.set_raw(p, after);
            touched.push(p);
        }
        let label = op.label.clone();
        self.undo.push(op);
        self.revision += 1;
        Some((label, touched))
    }

    pub fn clear(&mut self) -> Vec<IVec3> {
        let touched: Vec<IVec3> = self
            .chunks
            .iter()
            .flat_map(|(c, m)| m.keys().map(move |&i| *c * CHUNK + local_from_index(i)))
            .collect();
        self.chunks.clear();
        self.undo.clear();
        self.redo.clear();
        self.revision += 1;
        touched
    }

    pub fn edit_count(&self) -> usize {
        self.chunks.values().map(|m| m.len()).sum()
    }

    /// All overlay entries inside the world-space box [min, max) (inclusive-exclusive).
    pub fn entries_in(&self, min: IVec3, max: IVec3, mut f: impl FnMut(IVec3, u16)) {
        let cmin = chunk_of(min);
        let cmax = chunk_of(max - IVec3::ONE);
        let span = (cmax - cmin + IVec3::ONE).as_i64vec3();
        let volume = span.x * span.y * span.z;
        let mut visit = |c: IVec3, m: &HashMap<u16, u16>| {
            for (&i, &b) in m {
                let p = c * CHUNK + local_from_index(i);
                if p.cmpge(min).all() && p.cmplt(max).all() {
                    f(p, b);
                }
            }
        };
        if volume as usize <= self.chunks.len() * 2 {
            for z in cmin.z..=cmax.z {
                for y in cmin.y..=cmax.y {
                    for x in cmin.x..=cmax.x {
                        let c = IVec3::new(x, y, z);
                        if let Some(m) = self.chunks.get(&c) {
                            visit(c, m);
                        }
                    }
                }
            }
        } else {
            for (c, m) in &self.chunks {
                if c.cmpge(cmin).all() && c.cmple(cmax).all() {
                    visit(*c, m);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn index_roundtrip() {
        for p in [IVec3::new(0, 0, 0), IVec3::new(31, 5, 17), IVec3::new(-1, -33, 64)] {
            let l = p.rem_euclid(IVec3::splat(32));
            assert_eq!(local_from_index(local_index(p)), l);
        }
    }
    #[test]
    fn undo_redo() {
        let mut o = EditOverlay::default();
        o.apply("a", &[(IVec3::new(1, 2, 3), 5)]);
        assert_eq!(o.get(IVec3::new(1, 2, 3)), Some(5));
        o.undo();
        assert_eq!(o.get(IVec3::new(1, 2, 3)), None);
        o.redo();
        assert_eq!(o.get(IVec3::new(1, 2, 3)), Some(5));
    }
}
