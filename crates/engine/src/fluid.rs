//! Flowing water for edits.
//!
//! Generated water is still: every water cell the generator makes is a source. When an edit opens
//! space next to water, or takes away what fed some flowing water, the cells around it become
//! active and the water spreads or recedes, a few steps per second:
//!
//! - water above a cell makes it falling water;
//! - otherwise a cell next to water that rests on something (a source, or a flowing or falling
//!   cell over ground or still water) gets that water's level + 1, up to 7 (each level a lower
//!   surface), so water runs out 7 blocks from where it lands;
//! - flowing water that nothing feeds any more dries up, and plants it washed away come back.
//!
//! The simulation runs on the CPU over block values read back from the GPU world (generation plus
//! edits, one LOD0 chunk at a time), and writes water as edits that are not on the undo stack. So
//! water survives regeneration and saves with the world, and undoing the edit that let it in
//! makes it recede. The water level lives in the block state: 0 = source, 1..7 = flowing,
//! 8 = falling (`shaders/render/terrain.mesh` lowers the surface accordingly).

use std::collections::{BTreeSet, HashMap, HashSet};

use glam::IVec3;

use crate::blocks::{BlockRegistry, STATE_SHIFT, block_id};
use crate::world::{CHUNK, RegionRead, World};

pub const FALLING: u16 = 8;
const MAX_LEVEL: u16 = 7;
/// Seconds per simulation step (water runs about 7 blocks a second).
const STEP: f32 = 0.15;
/// Cells updated per step at most (the rest wait for the next one).
const MAX_UPDATES: usize = 32768;
/// New chunk reads requested per frame at most.
const MAX_READS: usize = 8;
/// Chunks kept in the cache.
const MAX_CHUNKS: usize = 128;
/// Bigger edits only activate their boundary (water can only get in from outside).
const BOUNDARY_ONLY: usize = 4096;

const SIDES: [IVec3; 4] = [IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z];
const N6: [IVec3; 6] = [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z];

type Key = (i32, i32, i32); // (y, z, x): bottom-up, then a fixed order

fn key(p: IVec3) -> Key {
    (p.y, p.z, p.x)
}

fn pos(k: Key) -> IVec3 {
    IVec3::new(k.2, k.0, k.1)
}

fn chunk_of(p: IVec3) -> IVec3 {
    p.div_euclid(IVec3::splat(CHUNK))
}

fn index(p: IVec3) -> usize {
    let l = p.rem_euclid(IVec3::splat(CHUNK));
    ((l.y * CHUNK + l.z) * CHUNK + l.x) as usize
}

pub struct FluidSim {
    pub enabled: bool,
    water: u16,
    active: BTreeSet<Key>,
    /// block values of LOD0 chunks around the activity (x fastest, then z, then y)
    chunks: HashMap<IVec3, Vec<u16>>,
    last_used: HashMap<IVec3, u64>,
    /// region read id -> chunk
    reading: HashMap<u64, IVec3>,
    /// chunks the last step was missing (read before the next one)
    wanted: BTreeSet<(i32, i32, i32)>,
    /// cells the water took over: (edit before, block before), restored when it leaves
    origin: HashMap<IVec3, (Option<u16>, u16)>,
    timer: f32,
    steps: u64,
}

impl FluidSim {
    pub fn new(blocks: &BlockRegistry) -> Self {
        Self {
            enabled: true,
            water: blocks.id("water"),
            active: BTreeSet::new(),
            chunks: HashMap::new(),
            last_used: HashMap::new(),
            reading: HashMap::new(),
            wanted: BTreeSet::new(),
            origin: HashMap::new(),
            timer: 0.0,
            steps: 0,
        }
    }

    /// Forget everything (the world was replaced: opened, cleared, regenerated from scratch).
    pub fn reset(&mut self) {
        self.active.clear();
        self.chunks.clear();
        self.last_used.clear();
        self.reading.clear();
        self.wanted.clear();
        self.origin.clear();
    }

    /// Cells waiting for an update (for the stats panel and scripts).
    pub fn pending(&self) -> usize {
        self.active.len()
    }

    /// Blocks changed by an edit (not by the simulation): refresh what we know and wake the
    /// water around them.
    pub fn edited(&mut self, positions: &[IVec3], world: &World) {
        let set: HashSet<IVec3> = if positions.len() > BOUNDARY_ONLY { positions.iter().copied().collect() } else { HashSet::new() };
        for &p in positions {
            // the edit owns the cell now
            self.origin.remove(&p);
            let c = chunk_of(p);
            if let Some(cells) = self.chunks.get_mut(&c) {
                match world.overlay.get(p) {
                    Some(v) => cells[index(p)] = v,
                    // back to the generated block, which only a new read knows
                    None => {
                        self.chunks.remove(&c);
                    }
                }
            }
            if !self.enabled {
                continue;
            }
            if set.is_empty() || N6.iter().any(|&d| !set.contains(&(p + d))) {
                self.active.insert(key(p));
                for d in N6 {
                    self.active.insert(key(p + d));
                }
            }
        }
        // reads in flight for those chunks may predate the edit
        let stale: HashSet<IVec3> = positions.iter().map(|&p| chunk_of(p)).collect();
        self.reading.retain(|_, c| !stale.contains(c));
    }

    /// A finished region read. Returns it if it isn't ours.
    pub fn region_ready(&mut self, r: RegionRead) -> Option<RegionRead> {
        let Some(c) = self.reading.remove(&r.id) else { return Some(r) };
        if r.blocks.len() == (CHUNK * CHUNK * CHUNK) as usize {
            self.chunks.insert(c, r.blocks);
        }
        None
    }

    /// The block at `p`, or the chunk to read first.
    fn get(&self, p: IVec3) -> Result<u16, IVec3> {
        let c = chunk_of(p);
        self.chunks.get(&c).map(|cells| cells[index(p)]).ok_or(c)
    }

    fn is_water(&self, v: u16) -> bool {
        block_id(v) == self.water
    }

    fn level(v: u16) -> u16 {
        v >> STATE_SHIFT
    }

    /// Water can run into this cell (air, or a plant it washes away).
    fn open(&self, v: u16, reg: &BlockRegistry) -> bool {
        v == 0 || reg.get(v).washes_away()
    }

    /// What `p` becomes (None = unchanged), or a chunk that has to be read first.
    fn next(&self, p: IVec3, reg: &BlockRegistry) -> Result<Option<u16>, IVec3> {
        let cur = self.get(p)?;
        let water = self.is_water(cur);
        if (water && Self::level(cur) == 0) || (!water && !self.open(cur, reg)) {
            return Ok(None); // sources and solid blocks stay
        }
        let new = if self.is_water(self.get(p + IVec3::Y)?) {
            Some(self.water | (FALLING << STATE_SHIFT))
        } else {
            let mut best: Option<u16> = None;
            for s in SIDES {
                let n = self.get(p + s)?;
                if !self.is_water(n) {
                    continue;
                }
                // water spreads sideways only where it rests on something
                let below = self.get(p + s - IVec3::Y)?;
                if self.open(below, reg) || (self.is_water(below) && Self::level(below) == FALLING) {
                    continue;
                }
                let l = match Self::level(n) {
                    FALLING => 0,
                    l => l,
                };
                best = Some(best.map_or(l + 1, |b: u16| b.min(l + 1)));
            }
            best.filter(|&l| l <= MAX_LEVEL).map(|l| self.water | (l << STATE_SHIFT))
        };
        Ok(match new {
            Some(v) if v != cur => Some(v),
            Some(_) => None,
            None if water => Some(0), // dries up
            None => None,
        })
    }

    /// Read the chunks the last step was missing, and step the simulation every STEP seconds.
    pub fn update(&mut self, dt: f32, world: &mut World, reg: &BlockRegistry) {
        if !self.enabled {
            self.active.clear();
            self.wanted.clear();
            return;
        }
        if self.active.is_empty() {
            return;
        }
        let mut new_reads = 0;
        while new_reads < MAX_READS {
            let Some(k) = self.wanted.pop_first() else { break };
            let c = IVec3::new(k.2, k.0, k.1);
            if self.chunks.contains_key(&c) || self.reading.values().any(|&r| r == c) {
                continue;
            }
            let o = c * CHUNK;
            let id = world.request_region(o, o + IVec3::splat(CHUNK - 1));
            self.reading.insert(id, c);
            new_reads += 1;
        }
        self.timer += dt;
        if self.timer < STEP {
            return;
        }
        self.timer = 0.0;
        self.steps += 1;
        self.step(world, reg);
        self.evict();
    }

    fn step(&mut self, world: &mut World, reg: &BlockRegistry) {
        // every cell decides from the old state, then all changes apply together
        let mut writes = Vec::new();
        let mut waiting = Vec::new();
        let mut n = 0;
        while let Some(k) = self.active.pop_first() {
            if n == MAX_UPDATES {
                self.active.insert(k);
                break;
            }
            n += 1;
            self.last_used.insert(chunk_of(pos(k)), self.steps);
            match self.next(pos(k), reg) {
                Err(c) => {
                    waiting.push(k);
                    self.wanted.insert(key(c));
                }
                Ok(Some(v)) => writes.push((pos(k), v)),
                Ok(None) => {}
            }
        }
        self.active.extend(waiting);
        if writes.is_empty() {
            return;
        }
        let mut changed = Vec::with_capacity(writes.len());
        for (p, v) in writes {
            let cur = self.get(p).unwrap_or(0);
            let (edit, block) = if self.is_water(v) {
                self.origin.entry(p).or_insert((world.overlay.get(p), cur));
                (Some(v), v)
            } else {
                // receding: give the cell back what it held before the water came
                self.origin.remove(&p).unwrap_or((Some(0), 0))
            };
            world.overlay.set_untracked(p, edit);
            if let Some(cells) = self.chunks.get_mut(&chunk_of(p)) {
                cells[index(p)] = block;
            }
            changed.push(p);
            for d in N6 {
                self.active.insert(key(p + d));
            }
        }
        world.invalidate_blocks_quiet(&changed);
    }

    /// Drop chunks nothing has needed for a while.
    fn evict(&mut self) {
        if self.chunks.len() <= MAX_CHUNKS {
            return;
        }
        let mut by_age: Vec<(u64, IVec3)> = self.chunks.keys().map(|c| (self.last_used.get(c).copied().unwrap_or(0), *c)).collect();
        by_age.sort_by_key(|(age, c)| (*age, c.to_array()));
        for (_, c) in by_age.into_iter().take(self.chunks.len() - MAX_CHUNKS) {
            self.chunks.remove(&c);
            self.last_used.remove(&c);
        }
    }
}
