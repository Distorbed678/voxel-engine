//! Flowing fluids for edits: any block marked as a fluid (water, lava, your own), each with its
//! own flow settings (`blocks::Flow`).
//!
//! Generated fluid is still: every fluid cell the generator makes is a source. When an edit opens
//! space next to a fluid, or takes away what fed some flowing fluid, the cells around it become
//! active and the fluid spreads or recedes, a step at a time:
//!
//! - fluid above an open cell makes it falling fluid;
//! - otherwise an open cell next to fluid that rests on something (a source, or a flowing or
//!   falling cell over ground or a still fluid) gets that fluid's level + 1, up to its flow
//!   distance (each level a lower surface): water runs 7 blocks, lava 3;
//! - a cell holding one fluid only takes levels from the same fluid;
//! - with infinite sources (water), a cell between two sources on solid ground becomes one;
//! - a fluid that hardens (lava) turns into its block where it touches another fluid;
//! - flowing fluid that nothing feeds any more dries up, and plants it washed away come back.
//!
//! Each fluid moves at its own pace: the simulation steps every STEP seconds and a fluid with a
//! slower `step` only changes every few steps.
//!
//! The simulation runs on the CPU over block values read back from the GPU world (generation plus
//! edits, one LOD0 chunk at a time), and writes fluid as edits that are not on the undo stack. So
//! fluid survives regeneration and saves with the world, and undoing the edit that let it in
//! makes it recede. The level lives in the block state: 0 = source, 1..7 = flowing,
//! 8 = falling (`shaders/render/terrain.mesh` lowers the surface accordingly).

use std::collections::{BTreeSet, HashMap, HashSet};

use glam::IVec3;

use crate::blocks::{BlockRegistry, STATE_SHIFT, block_id};
use crate::world::{CHUNK, RegionRead, World};

pub const FALLING: u16 = 8;
/// Seconds per simulation step (the fastest a fluid runs: water, about 7 blocks a second).
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

/// A fluid block's flow, for the simulation.
#[derive(Clone, Copy, Debug)]
struct Fluid {
    /// flow distance (the deepest flowing level)
    distance: u16,
    /// simulation steps per block it runs
    ticks: u64,
    infinite: bool,
    /// the block it turns into where it touches another fluid
    hardens: Option<u16>,
}

pub struct FluidSim {
    pub enabled: bool,
    /// the fluid blocks by id
    fluids: HashMap<u16, Fluid>,
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
            fluids: Self::fluids_of(blocks),
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

    fn fluids_of(reg: &BlockRegistry) -> HashMap<u16, Fluid> {
        reg.live()
            .filter(|(_, d)| d.fluid)
            .map(|(id, d)| {
                let hardens = (!d.flow.hardens.is_empty()).then(|| reg.id(&d.flow.hardens)).filter(|&h| h != 0 || d.flow.hardens == "air");
                let ticks = (d.flow.step / STEP).round().max(1.0) as u64;
                (id, Fluid { distance: d.flow.distance.clamp(1, 7) as u16, ticks, infinite: d.flow.infinite, hardens })
            })
            .collect()
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

    fn fluid(&self, v: u16) -> Option<Fluid> {
        self.fluids.get(&block_id(v)).copied()
    }

    fn level(v: u16) -> u16 {
        v >> STATE_SHIFT
    }

    /// Water can run into this cell (air, or a plant it washes away).
    fn open(&self, v: u16, reg: &BlockRegistry) -> bool {
        v == 0 || reg.get(v).washes_away
    }

    /// What `p` becomes (None = unchanged) and the fluid whose pace the change follows, or a
    /// chunk that has to be read first.
    fn next(&self, p: IVec3, reg: &BlockRegistry) -> Result<Option<(u16, Fluid)>, IVec3> {
        let cur = self.get(p)?;
        let here = self.fluid(cur);
        if here.is_none() && !self.open(cur, reg) {
            return Ok(None); // solid blocks stay
        }
        // a fluid that hardens turns into its block where it touches another fluid
        if let Some(f) = here
            && let Some(h) = f.hardens
        {
            for d in N6 {
                let n = self.get(p + d)?;
                if self.fluid(n).is_some() && block_id(n) != block_id(cur) {
                    return Ok(Some((h, f)));
                }
            }
        }
        if here.is_some() && Self::level(cur) == 0 {
            return Ok(None); // sources stay
        }
        let id = |v: u16| block_id(v);
        // the fluid this cell may take: its own, or any (an open cell)
        let same = |v: u16| here.is_none() || id(v) == id(cur);
        let above = self.get(p + IVec3::Y)?;
        let new = if self.fluid(above).is_some() && same(above) {
            Some((id(above) | (FALLING << STATE_SHIFT), self.fluid(above).unwrap()))
        } else {
            let mut best: Option<(u16, u16, Fluid)> = None; // (level, block, fluid)
            let mut sources: HashMap<u16, u32> = HashMap::new();
            for s in SIDES {
                let n = self.get(p + s)?;
                let Some(f) = self.fluid(n) else { continue };
                if !same(n) {
                    continue;
                }
                if Self::level(n) == 0 {
                    *sources.entry(id(n)).or_default() += 1;
                }
                // fluid spreads sideways only where it rests on something
                let below = self.get(p + s - IVec3::Y)?;
                if self.open(below, reg) || (self.fluid(below).is_some() && Self::level(below) == FALLING) {
                    continue;
                }
                let l = match Self::level(n) {
                    FALLING => 0,
                    l => l,
                } + 1;
                if l <= f.distance && best.is_none_or(|b| (l, id(n)) < (b.0, b.1)) {
                    best = Some((l, id(n), f));
                }
            }
            // infinite sources: a gap between two sources, over solid ground or a source
            let below = self.get(p - IVec3::Y)?;
            let grounded = !self.open(below, reg) && self.fluid(below).is_none_or(|_| Self::level(below) == 0);
            let refill = sources.iter().filter(|&(&b, &n)| n >= 2 && grounded && self.fluids.get(&b).is_some_and(|f| f.infinite)).map(|(&b, _)| b).min();
            match refill {
                Some(b) => Some((b, self.fluids[&b])),
                None => best.map(|(l, b, f)| (b | (l << STATE_SHIFT), f)),
            }
        };
        Ok(match new {
            Some((v, _)) if v == cur => None,
            Some(n) => Some(n),
            None => here.map(|f| (0, f)), // dries up
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
        // the block editor may have changed a fluid's settings
        self.fluids = Self::fluids_of(reg);
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
                // slower fluids change only every few steps
                Ok(Some((_, f))) if !self.steps.is_multiple_of(f.ticks) => waiting.push(k),
                Ok(Some((v, _))) => writes.push((pos(k), v)),
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
            let (edit, block) = if self.fluid(v).is_some() {
                self.origin.entry(p).or_insert((world.overlay.get(p), cur));
                (Some(v), v)
            } else if v != 0 {
                // hardened: a block of its own now, which stays
                self.origin.remove(&p);
                (Some(v), v)
            } else {
                // receding: give the cell back what it held before the fluid came
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
