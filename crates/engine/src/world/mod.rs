//! Chunk streaming: nested LOD clipmaps of cubic 32^3 chunks, infinite in all three axes.
//!
//! Level L uses chunks of 32 * 2^L blocks. Each level covers a box of (2R x 2Rv x 2R) chunks
//! around the camera minus the box covered by level L-1, so the levels nest without overlap.
//! Chunks are (re)generated entirely on the GPU; the CPU only decides *which* chunks to build,
//! sub-allocates mesh memory and keeps the per-chunk draw records up to date.

pub mod alloc;
pub mod edits;
pub mod gpu;
pub mod light;
pub mod save;

use std::collections::HashMap;
use std::time::Instant;

use glam::{DVec3, IVec3, Vec3};

use alloc::{Range, RangeAllocator};
use edits::EditOverlay;
use gpu::{GENJOB_COARSE, GENJOB_LIGHT, GpuChunkRecord, JobDesc, JobResult};

pub const CHUNK: i32 = 32;
pub const CA: i32 = 34;
pub const RECORD_ACTIVE: u32 = 256;
pub const JOB_OVERFLOW: u32 = 1;
/// Light-only jobs per batch (chunks scrolling into the light volume).
const MAX_LIGHT_ONLY: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkKey {
    pub lod: u8,
    pub pos: IVec3,
}

impl ChunkKey {
    pub fn size(&self) -> i32 {
        CHUNK << self.lod
    }
    pub fn origin(&self) -> IVec3 {
        self.pos * self.size()
    }
    pub fn center(&self) -> DVec3 {
        self.origin().as_dvec3() + DVec3::splat(self.size() as f64 * 0.5)
    }
}

#[derive(Debug, Default)]
pub struct Chunk {
    pub record: Option<u32>,
    pub mesh: Option<Range>,
    pub dirty: u32,
    pub applied: u32,
    pub in_flight: bool,
    pub has_result: bool,
    pub desired: bool,
    pub retire_since: Option<Instant>,
    pub boost: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LodSettings {
    pub levels: u32,
    /// Horizontal radius in chunks per level (must be even).
    pub radius: i32,
    /// Vertical radius in chunks per level (must be even).
    pub radius_v: i32,
}

impl Default for LodSettings {
    fn default() -> Self {
        Self { levels: 5, radius: 8, radius_v: 4 }
    }
}

#[derive(Default, Clone, Debug)]
pub struct WorldStats {
    pub chunks: usize,
    pub meshed: usize,
    pub empty: usize,
    pub pending: usize,
    pub in_flight: usize,
    pub retiring: usize,
    pub jobs_total: u64,
    pub jobs_last_second: u32,
    pub quads_resident: u64,
    pub per_lod: [(u32, u32); 8],
    pub last_overflow: u32,
}

pub enum Deferred {
    Mesh(Range),
    Record(u32),
}

/// Copy from a batch scratch buffer into the quad pool.
#[derive(Clone, Copy, Debug)]
pub struct PoolCopy {
    pub slot: usize,
    pub src: u32,
    pub dst: u32,
    pub count: u32,
}

/// Asynchronous read of the actual blocks (generation + overlay) inside a box.
pub struct RegionRead {
    pub id: u64,
    pub min: IVec3,
    pub size: IVec3,
    pub blocks: Vec<u16>,
    pending: std::collections::HashSet<IVec3>,
    queued: Vec<IVec3>,
}

impl RegionRead {
    pub fn get(&self, p: IVec3) -> u16 {
        let l = p - self.min;
        if l.cmplt(IVec3::ZERO).any() || l.cmpge(self.size).any() {
            return 0;
        }
        self.blocks[((l.y * self.size.z + l.z) * self.size.x + l.x) as usize]
    }
    pub fn done(&self) -> bool {
        self.pending.is_empty()
    }
    pub fn requeue(&mut self, c: IVec3) {
        if self.pending.contains(&c) && !self.queued.contains(&c) {
            self.queued.push(c);
        }
    }
}

pub struct World {
    pub chunks: HashMap<ChunkKey, Chunk>,
    pub region_reads: Vec<RegionRead>,
    next_region_id: u64,
    pub settings: LodSettings,
    last_center: Option<(IVec3, LodSettings)>,
    pub overlay: EditOverlay,
    pub pool: RangeAllocator,
    free_records: Vec<u32>,
    pub record_count: u32,
    pub max_records: u32,
    deferred: Vec<(u64, Deferred)>,
    pub record_updates: Vec<(u32, GpuChunkRecord)>,
    /// Chunks that can cast sun shadows (LOD 0 and 1) whose mesh changed: (origin, size). The
    /// engine redraws a cached shadow cascade when one lies inside it.
    pub caster_changes: Vec<(IVec3, i32)>,
    /// Blocks changed by edits since the water simulation last looked.
    pub edited: Vec<IVec3>,
    pub copies: Vec<PoolCopy>,
    pub stats: WorldStats,
    jobs_window: (Instant, u32),
    pub frozen: bool,
    /// Which chunks the fine light volume covers and still needs.
    pub light: light::LightTracker,
    /// Same for the coarse (far-field) light volume.
    pub coarse_light: light::LightTracker,
    /// LOD0 chunks whose raw 34³ voxel slot (apron included) the next region read keeps (debugging).
    pub capture_raw: std::collections::HashSet<IVec3>,
    pub raw_slots: HashMap<IVec3, Vec<u16>>,
}

impl World {
    pub fn new(quad_capacity: u32, max_records: u32) -> Self {
        Self {
            chunks: HashMap::new(),
            region_reads: Vec::new(),
            next_region_id: 1,
            settings: LodSettings::default(),
            last_center: None,
            overlay: EditOverlay::default(),
            pool: RangeAllocator::new(quad_capacity),
            free_records: Vec::new(),
            record_count: 0,
            max_records,
            deferred: Vec::new(),
            record_updates: Vec::new(),
            caster_changes: Vec::new(),
            edited: Vec::new(),
            copies: Vec::new(),
            stats: WorldStats::default(),
            jobs_window: (Instant::now(), 0),
            frozen: false,
            light: light::LightTracker::fine(),
            coarse_light: light::LightTracker::coarse(),
            capture_raw: Default::default(),
            raw_slots: HashMap::new(),
        }
    }

    fn level_box(&self, cam: DVec3, lod: u32) -> (IVec3, IVec3) {
        let size = (CHUNK << lod) as f64;
        let cc = (cam / size).floor().as_ivec3();
        let c = cc.div_euclid(IVec3::splat(2)) * 2;
        let r = IVec3::new(self.settings.radius, self.settings.radius_v, self.settings.radius);
        (c - r, c + r)
    }

    /// Recompute the desired chunk set when the camera crosses a chunk boundary.
    pub fn update_desired(&mut self, cam: DVec3) {
        if self.frozen {
            return;
        }
        self.light.update(cam);
        self.coarse_light.update(cam);
        let center = (cam / CHUNK as f64).floor().as_ivec3();
        if self.last_center == Some((center, self.settings)) {
            return;
        }
        self.last_center = Some((center, self.settings));
        for c in self.chunks.values_mut() {
            c.desired = false;
        }
        let mut inner: Option<(IVec3, IVec3)> = None;
        for lod in 0..self.settings.levels {
            let (mn, mx) = self.level_box(cam, lod);
            let inner_box = inner.map(|(a, b)| (a.div_euclid(IVec3::splat(2)), b.div_euclid(IVec3::splat(2))));
            for z in mn.z..mx.z {
                for y in mn.y..mx.y {
                    for x in mn.x..mx.x {
                        let p = IVec3::new(x, y, z);
                        if let Some((a, b)) = inner_box
                            && p.cmpge(a).all() && p.cmplt(b).all() {
                                continue;
                            }
                        let key = ChunkKey { lod: lod as u8, pos: p };
                        let c = self.chunks.entry(key).or_default();
                        c.desired = true;
                        c.retire_since = None;
                    }
                }
            }
            inner = Some((mn, mx));
        }
        let now = Instant::now();
        for c in self.chunks.values_mut() {
            if !c.desired && c.retire_since.is_none() {
                c.retire_since = Some(now);
            }
        }
    }

    fn is_ready(&self, key: &ChunkKey, depth: u32) -> bool {
        match self.chunks.get(key) {
            Some(c) if c.desired => c.has_result,
            _ => {
                if depth == 0 || key.lod == 0 {
                    return true;
                }
                // not desired at this level: covered by finer levels?
                for i in 0..8 {
                    let o = IVec3::new(i & 1, (i >> 1) & 1, (i >> 2) & 1);
                    let child = ChunkKey { lod: key.lod - 1, pos: key.pos * 2 + o };
                    if self.chunks.get(&child).map(|c| c.desired && !c.has_result).unwrap_or(false) {
                        return false;
                    }
                }
                true
            }
        }
    }

    /// Remove retired chunks once the chunks replacing them are ready (no LOD-switch holes).
    pub fn process_retirement(&mut self, frame_value: u64) {
        let now = Instant::now();
        let mut remove = Vec::new();
        for (key, c) in &self.chunks {
            let Some(since) = c.retire_since else { continue };
            if c.in_flight {
                continue;
            }
            if c.mesh.is_none() || now.duration_since(since).as_secs_f32() > 4.0 {
                remove.push(*key);
                continue;
            }
            let mut covered = true;
            if key.lod > 0 {
                for i in 0..8 {
                    let o = IVec3::new(i & 1, (i >> 1) & 1, (i >> 2) & 1);
                    let child = ChunkKey { lod: key.lod - 1, pos: key.pos * 2 + o };
                    if !self.is_ready(&child, 1) {
                        covered = false;
                        break;
                    }
                }
            }
            if covered && (key.lod as u32) + 1 < self.settings.levels {
                let parent = ChunkKey { lod: key.lod + 1, pos: key.pos.div_euclid(IVec3::splat(2)) };
                if let Some(p) = self.chunks.get(&parent)
                    && p.desired && !p.has_result {
                        covered = false;
                    }
            }
            if covered {
                remove.push(*key);
            }
        }
        for key in remove {
            if let Some(c) = self.chunks.remove(&key) {
                self.release_chunk(key, c, frame_value);
            }
        }
    }

    fn note_caster_change(&mut self, key: ChunkKey) {
        if key.lod <= 1 {
            self.caster_changes.push((key.origin(), key.size()));
        }
    }

    fn release_chunk(&mut self, key: ChunkKey, c: Chunk, frame_value: u64) {
        if let Some(m) = c.mesh {
            self.deferred.push((frame_value, Deferred::Mesh(m)));
        }
        if let Some(r) = c.record {
            self.note_caster_change(key);
            self.record_updates.push((r, GpuChunkRecord::inactive()));
            self.deferred.push((frame_value, Deferred::Record(r)));
        }
    }

    pub fn free_deferred(&mut self, completed: u64) {
        let mut i = 0;
        while i < self.deferred.len() {
            if self.deferred[i].0 <= completed {
                let (_, d) = self.deferred.swap_remove(i);
                match d {
                    Deferred::Mesh(r) => self.pool.free(r),
                    Deferred::Record(r) => self.free_records.push(r),
                }
            } else {
                i += 1;
            }
        }
    }

    fn alloc_record(&mut self) -> Option<u32> {
        if let Some(r) = self.free_records.pop() {
            return Some(r);
        }
        if self.record_count < self.max_records {
            self.record_count += 1;
            Some(self.record_count - 1)
        } else {
            None
        }
    }

    /// Mark every chunk for regeneration (graph / seed changed). Old meshes stay visible
    /// until their replacements arrive.
    pub fn invalidate_all(&mut self) {
        for c in self.chunks.values_mut() {
            c.dirty = c.dirty.wrapping_add(1);
        }
    }

    /// Mark chunks affected by edits at the given block positions (and tell the water
    /// simulation, through `edited`).
    pub fn invalidate_blocks(&mut self, positions: &[IVec3]) {
        self.edited.extend_from_slice(positions);
        self.invalidate_blocks_quiet(positions);
    }

    /// `invalidate_blocks` for the water simulation's own writes.
    pub fn invalidate_blocks_quiet(&mut self, positions: &[IVec3]) {
        let mut keys = std::collections::HashSet::new();
        for &p in positions {
            for lod in 0..self.settings.levels {
                let size = CHUNK << lod;
                let scale = 1 << lod;
                let cpos = p.div_euclid(IVec3::splat(size));
                let local = p - cpos * size;
                // neighbours whose apron contains p
                let lo = local.cmplt(IVec3::splat(scale));
                let hi = local.cmpge(IVec3::splat(size - scale));
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let d = IVec3::new(dx, dy, dz);
                            let ok = (0..3).all(|a| match d[a] {
                                -1 => lo.test(a),
                                1 => hi.test(a),
                                _ => true,
                            });
                            if ok {
                                keys.insert(ChunkKey { lod: lod as u8, pos: cpos + d });
                            }
                        }
                    }
                }
            }
        }
        for k in keys {
            if let Some(c) = self.chunks.get_mut(&k) {
                c.dirty = c.dirty.wrapping_add(1);
                c.boost = true;
            }
        }
    }

    /// Request the real block contents of [min, max] (inclusive). Returns a request id.
    pub fn request_region(&mut self, a: IVec3, b: IVec3) -> u64 {
        let min = a.min(b);
        let max = a.max(b);
        let size = max - min + IVec3::ONE;
        let c0 = min.div_euclid(IVec3::splat(CHUNK));
        let c1 = max.div_euclid(IVec3::splat(CHUNK));
        let mut pending = std::collections::HashSet::new();
        for z in c0.z..=c1.z {
            for y in c0.y..=c1.y {
                for x in c0.x..=c1.x {
                    pending.insert(IVec3::new(x, y, z));
                }
            }
        }
        let id = self.next_region_id;
        self.next_region_id += 1;
        let queued = pending.iter().copied().collect();
        self.region_reads.push(RegionRead { id, min, size, blocks: vec![0; (size.x * size.y * size.z) as usize], pending, queued });
        id
    }

    /// Feed voxel data from a readback job into any region reads waiting for that chunk.
    pub fn fill_region(&mut self, key: ChunkKey, voxels: &[u16]) {
        if key.lod != 0 {
            return;
        }
        if self.capture_raw.remove(&key.pos) {
            self.raw_slots.insert(key.pos, voxels.to_vec());
        }
        let origin = key.origin();
        for r in &mut self.region_reads {
            if !r.pending.remove(&key.pos) {
                continue;
            }
            for ly in 0..CHUNK {
                for lz in 0..CHUNK {
                    for lx in 0..CHUNK {
                        let w = origin + IVec3::new(lx, ly, lz);
                        let l = w - r.min;
                        if l.cmplt(IVec3::ZERO).any() || l.cmpge(r.size).any() {
                            continue;
                        }
                        let vi = ((ly + 1) * CA * CA + (lz + 1) * CA + (lx + 1)) as usize;
                        r.blocks[((l.y * r.size.z + l.z) * r.size.x + l.x) as usize] = voxels[vi];
                    }
                }
            }
        }
    }

    /// Completed region reads (removed from the queue).
    pub fn take_finished_regions(&mut self) -> Vec<RegionRead> {
        let (done, rest): (Vec<RegionRead>, Vec<RegionRead>) = std::mem::take(&mut self.region_reads).into_iter().partition(|r| r.done());
        self.region_reads = rest;
        done
    }

    /// Pick up to `max` chunks to (re)generate, nearest / most visible first.
    pub fn pick_jobs(&mut self, cam: DVec3, forward: Vec3, max: usize) -> Vec<JobDesc> {
        let mut cands: Vec<(f32, ChunkKey)> = self
            .chunks
            .iter()
            .filter(|(_, c)| c.desired && !c.in_flight && (!c.has_result || c.applied != c.dirty))
            .map(|(k, c)| {
                let d = k.center() - cam;
                let dist = d.length() as f32;
                let dir = d.as_vec3() / dist.max(1e-3);
                let facing = dir.dot(forward);
                let mut prio = dist * if facing > 0.3 { 0.5 } else { 1.0 };
                if !c.has_result {
                    prio *= 0.8;
                }
                if c.boost {
                    prio = -1.0e6 + dist;
                }
                (prio, *k)
            })
            .collect();
        if cands.len() > max {
            cands.select_nth_unstable_by(max, |a, b| a.0.total_cmp(&b.0));
            cands.truncate(max);
        }
        cands.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut jobs = Vec::with_capacity(cands.len());
        // region-read chunks go first (they don't need to be in the streamed set)
        let mut rb: Vec<IVec3> = Vec::new();
        for r in &mut self.region_reads {
            while let Some(c) = r.queued.pop() {
                if rb.len() >= max / 2 {
                    r.queued.push(c);
                    break;
                }
                rb.push(c);
            }
        }
        for c in rb {
            let key = ChunkKey { lod: 0, pos: c };
            let origin = key.origin();
            let min = origin - IVec3::ONE;
            let maxp = origin + IVec3::splat(CHUNK + 1);
            let mut edits = Vec::new();
            self.overlay.entries_in(min, maxp, |p, b| {
                let cell = p - min;
                edits.push((((cell.y * CA + cell.z) * CA + cell.x) as u32, b));
            });
            // counter u32::MAX marks a pure read (results are not applied to the streamed chunk)
            jobs.push(JobDesc { key, counter: u32::MAX, readback: true, light: 0, edits });
        }
        // light-only jobs: chunks that scrolled into a light volume while already up to date
        // (chunks still waiting for generation fill the volumes when they are meshed)
        let ready = |p: &IVec3| {
            self.chunks
                .get(&ChunkKey { lod: 0, pos: *p })
                .is_none_or(|c| c.has_result && c.applied == c.dirty && !c.in_flight)
        };
        let mut light_only: Vec<IVec3> = self.light.pending().filter(ready).take(MAX_LIGHT_ONLY.min(max / 4)).collect();
        let room = MAX_LIGHT_ONLY.min(max / 4).saturating_sub(light_only.len());
        light_only.extend(self.coarse_light.pending().filter(|p| ready(p) && !self.light.is_pending(*p)).take(room));
        for p in light_only {
            let mut flags = 0;
            if self.light.is_pending(p) {
                flags |= GENJOB_LIGHT;
            }
            if self.coarse_light.contains(p) {
                flags |= GENJOB_COARSE;
            }
            self.light.filled(p);
            self.coarse_light.filled(p);
            let key = ChunkKey { lod: 0, pos: p };
            let min = key.origin() - IVec3::ONE;
            let mut edits = Vec::new();
            self.overlay.entries_in(min, key.origin() + IVec3::splat(CHUNK + 1), |q, b| {
                let cell = q - min;
                edits.push((((cell.y * CA + cell.z) * CA + cell.x) as u32, b));
            });
            jobs.push(JobDesc { key, counter: u32::MAX, readback: false, light: flags, edits });
        }
        let cands: Vec<(f32, ChunkKey)> = cands.into_iter().take(max.saturating_sub(jobs.len())).collect();
        for (_, key) in cands {
            let c = self.chunks.get_mut(&key).unwrap();
            c.in_flight = true;
            c.boost = false;
            let counter = c.dirty;
            // gather overlay edits that land on this chunk's sample points (incl. apron)
            let scale = 1i32 << key.lod;
            let origin = key.origin();
            let min = origin - IVec3::splat(scale);
            let max = origin + IVec3::splat(scale * (CHUNK + 1));
            let half = scale / 2;
            let mut edits = Vec::new();
            self.overlay.entries_in(min, max, |p, b| {
                let rel = p - min;
                let m = rel.rem_euclid(IVec3::splat(scale));
                if m == IVec3::splat(half) {
                    let cell = rel.div_euclid(IVec3::splat(scale));
                    let idx = (cell.y * CA + cell.z) * CA + cell.x;
                    edits.push((idx as u32, b));
                }
            });
            let mut light = 0;
            if key.lod == 0 && self.light.contains(key.pos) {
                self.light.filled(key.pos);
                light |= GENJOB_LIGHT;
            }
            if key.lod == 0 && self.coarse_light.contains(key.pos) {
                self.coarse_light.filled(key.pos);
                light |= GENJOB_COARSE;
            }
            jobs.push(JobDesc { key, counter, readback: false, light, edits });
        }
        self.stats.jobs_total += jobs.len() as u64;
        self.jobs_window.1 += jobs.len() as u32;
        if self.jobs_window.0.elapsed().as_secs_f32() >= 1.0 {
            self.stats.jobs_last_second = self.jobs_window.1;
            self.jobs_window = (Instant::now(), 0);
        }
        jobs
    }

    /// Apply finished GPU jobs: allocate mesh memory, queue pool copies and record updates.
    pub fn apply_results(&mut self, slot: usize, jobs: &[JobDesc], results: &[JobResult], frame_value: u64) {
        let mut overflow = 0;
        for (job, res) in jobs.iter().zip(results) {
            if job.counter == u32::MAX {
                if res.flags & JOB_OVERFLOW != 0 {
                    for r in &mut self.region_reads {
                        r.requeue(job.key.pos);
                    }
                }
                continue; // region read only
            }
            let Some(c) = self.chunks.get_mut(&job.key) else { continue };
            c.in_flight = false;
            if res.flags & JOB_OVERFLOW != 0 {
                overflow += 1;
                continue; // stays dirty, will be retried
            }
            c.applied = job.counter;
            c.has_result = true;
            let old_mesh = c.mesh.take();
            let old_record = c.record;
            if let Some(m) = old_mesh {
                self.deferred.push((frame_value, Deferred::Mesh(m)));
            }
            if res.total == 0 {
                if let Some(r) = old_record {
                    self.note_caster_change(job.key);
                    self.record_updates.push((r, GpuChunkRecord::inactive()));
                    self.deferred.push((frame_value, Deferred::Record(r)));
                }
                self.chunks.get_mut(&job.key).unwrap().record = None;
                continue;
            }
            let Some(range) = self.pool.alloc(res.total) else {
                log::warn!("quad pool exhausted ({} quads requested)", res.total);
                let c = self.chunks.get_mut(&job.key).unwrap();
                c.has_result = false;
                if let Some(r) = old_record {
                    self.record_updates.push((r, GpuChunkRecord::inactive()));
                    self.deferred.push((frame_value, Deferred::Record(r)));
                    c.record = None;
                    if job.key.lod <= 1 {
                        self.caster_changes.push((job.key.origin(), job.key.size()));
                    }
                }
                continue;
            };
            let record = match old_record {
                Some(r) => Some(r),
                None => self.alloc_record(),
            };
            let Some(record) = record else {
                self.pool.free(range);
                log::warn!("chunk record table full");
                continue;
            };
            self.copies.push(PoolCopy { slot, src: res.scratch_base, dst: range.offset, count: res.total });
            self.note_caster_change(job.key);
            self.record_updates.push((
                record,
                GpuChunkRecord {
                    origin: job.key.origin().to_array(),
                    lod_flags: job.key.lod as u32 | RECORD_ACTIVE,
                    quad_base: range.offset,
                    counts: res.counts,
                },
            ));
            let c = self.chunks.get_mut(&job.key).unwrap();
            c.mesh = Some(range);
            c.record = Some(record);
        }
        self.stats.last_overflow = overflow;
    }

    pub fn update_stats(&mut self) {
        let mut s = WorldStats {
            jobs_total: self.stats.jobs_total,
            jobs_last_second: self.stats.jobs_last_second,
            last_overflow: self.stats.last_overflow,
            ..Default::default()
        };
        s.chunks = self.chunks.len();
        for (k, c) in &self.chunks {
            if c.mesh.is_some() {
                s.meshed += 1;
                s.per_lod[k.lod as usize].0 += 1;
            } else if c.has_result {
                s.empty += 1;
            }
            if c.desired && (!c.has_result || c.applied != c.dirty) && !c.in_flight {
                s.pending += 1;
                s.per_lod[k.lod as usize].1 += 1;
            }
            if c.in_flight {
                s.in_flight += 1;
            }
            if c.retire_since.is_some() {
                s.retiring += 1;
            }
        }
        s.quads_resident = self.pool.used;
        self.stats = s;
    }

    /// Drop everything (e.g. LOD settings changed drastically). Meshes are freed lazily.
    pub fn reset(&mut self, frame_value: u64) {
        let chunks: Vec<(ChunkKey, Chunk)> = self.chunks.drain().collect();
        for (key, c) in chunks {
            self.release_chunk(key, c, frame_value);
        }
        self.last_center = None;
    }
}
