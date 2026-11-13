//! GPU side of chunk generation: batch ring, transient voxel slots, scratch quads, quad pool and
//! chunk draw records. Everything here is recorded into the frame command buffer.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use vkcore::sync::{compute_barrier, memory_barrier};
use vkcore::{Buffer, GpuProfiler, VkContext, vk};

use super::{ChunkKey, PoolCopy};

pub const MAX_JOBS: usize = 256;
pub const SLOT_U16: u64 = 39424;
pub const NUM_STREAMS: usize = 13;
pub const BATCH_SLOTS: usize = 4;
/// Job lists per batch slot: two per generator layer (the cells it owns, and sole-layer jobs),
/// so at most 8 layers.
pub const MAX_LAYERS: usize = 16;
/// Job list entry flag: this layer writes the job's column biomes (`JOBLIST_COLUMNS`).
const JOBLIST_COLUMNS: u32 = 0x8000_0000;
/// Voxels in a slot (34³) and column entries (34²).
const SLOT_CELLS: u64 = 34 * 34 * 34;
const SLOT_COLUMNS: u64 = 34 * 34;
/// Props pass scratch (`shaders/worldgen/props_lib.glsl`, `props_main.glsl`): per job, 16 prop
/// instances are built at a time, each with up to 192 primitives (80 bytes) and 96 skeleton
/// segments (36 bytes).
const PROP_BUILD_BATCH: u64 = 16;
const PROP_PRIM_SCRATCH: u64 = MAX_JOBS as u64 * PROP_BUILD_BATCH * 192 * 80;
const PROP_SEG_SCRATCH: u64 = MAX_JOBS as u64 * PROP_BUILD_BATCH * 96 * 36;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuChunkRecord {
    pub origin: [i32; 3],
    pub lod_flags: u32,
    pub quad_base: u32,
    pub counts: [u32; NUM_STREAMS],
}

impl GpuChunkRecord {
    pub fn inactive() -> Self {
        Self::zeroed()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GpuGenJob {
    pub origin: [i32; 3],
    pub lod: u32,
    pub edit_offset: u32,
    pub edit_count: u32,
    /// GENJOB_* flags
    pub flags: u32,
    pub pad: u32,
}

/// Copy the job's final blocks into the fine light volume.
pub const GENJOB_LIGHT: u32 = 1;
/// Summarise the job's blocks into the coarse (far-field) light volume.
pub const GENJOB_COARSE: u32 = 2;
/// A prop preview (prop editor): no terrain, one prop graph at the origin.
pub const GENJOB_PREVIEW: u32 = 4;
/// Result flag: don't mesh (region reads, light-only jobs).
const JOB_NO_MESH: u32 = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
pub struct JobResult {
    pub total: u32,
    pub scratch_base: u32,
    pub flags: u32,
    pub non_air: u32,
    pub opaque: u32,
    pub counts: [u32; NUM_STREAMS],
    pub pad: [u32; 2],
}

#[derive(Clone, Debug)]
pub struct JobDesc {
    pub key: ChunkKey,
    pub counter: u32,
    /// copy the generated voxels back to the CPU (region reads for editing tools)
    pub readback: bool,
    /// GENJOB_LIGHT / GENJOB_COARSE: copy the final blocks into the light volumes
    pub light: u32,
    /// (cell index in the 34^3 slot, block)
    pub edits: Vec<(u32, u16)>,
    /// A prop preview: (prop graph index, seed). Implies `readback`.
    pub preview: Option<(u32, u32)>,
    /// Generate every cell with this layer (the camera's), not each with the layer owning it:
    /// far chunks across a layer boundary, whose other side the camera can't see.
    pub sole_layer: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    Free,
    Generating(u64),
    /// Results read; copies (if any) are recorded this frame.
    Polled,
    Copying(u64),
}

struct BatchSlot {
    scratch: Buffer,
    readback: Buffer,
    voxel_readback: Option<Buffer>,
    jobs_buf: Buffer,
    /// every layer's job list, one after the other
    job_lists: Buffer,
    edits_buf: Buffer,
    state: SlotState,
    jobs: Vec<JobDesc>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GenPush {
    pub jobs: u64,
    pub voxels: u64,
    pub results: u64,
    pub edits: u64,
    pub params: u64,
    pub columns: u64,
    pub prefab_table: u64,
    pub prefab_data: u64,
    pub num_jobs: u32,
    pub seed: u32,
    pub edit_count: u32,
    pub pad: u32,
    /// the GPU block table (kinds and rotations for props and prefabs)
    pub blocks: u64,
    /// scratch for building prop instances (props pass)
    pub prop_prims: u64,
    pub prop_segs: u64,
    /// the layer's jobs (indices into `jobs`), and the cell heights it owns
    pub job_list: u64,
    pub own_min: f32,
    pub own_max: f32,
    /// solo preview zone and biome (global ids, -1 = off)
    pub solo_zone: f32,
    pub solo_biome: f32,
}

/// One generator layer: its passes, its block of parameters and the cell heights it owns,
/// [own.0, own.1) (`worldgen::Compiled::owned_ranges`).
#[derive(Clone, Copy, Debug)]
pub struct LayerPass {
    pub terrain: vk::Pipeline,
    pub props: vk::Pipeline,
    pub param_base: u32,
    pub own: (f32, f32),
}

impl LayerPass {
    /// Whether the layer owns any of a job's cells, and all of them.
    fn covers(&self, key: &ChunkKey) -> (bool, bool) {
        let (lo, hi) = key.cell_heights();
        (self.own.0 <= hi && self.own.1 > lo, self.own.0 <= lo && hi < self.own.1)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EditPush {
    voxels: u64,
    edits: u64,
    results: u64,
    count: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeshPush {
    voxels: u64,
    blocks: u64,
    columns: u64,
    table: u64,
    results: u64,
    scratch: u64,
    num_jobs: u32,
    scratch_capacity: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightFillPush {
    jobs: u64,
    voxels: u64,
    num_jobs: u32,
    blocks_img: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CoarseFillPush {
    jobs: u64,
    voxels: u64,
    blocks: u64,
    num_jobs: u32,
    coarse_img: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrefixPush {
    table: u64,
    results: u64,
    stats: u64,
    num_jobs: u32,
    scratch_capacity: u32,
}

pub struct WorldPipelines {
    /// the generator layers (the first one also draws prop previews)
    pub layers: Vec<LayerPass>,
    pub apply_edits: vk::Pipeline,
    pub mesh_count: vk::Pipeline,
    pub mesh_write: vk::Pipeline,
    pub prefix: vk::Pipeline,
    pub light_fill: vk::Pipeline,
    pub coarse_fill: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    /// Bindless storage index of the light volume's block image.
    pub light_blocks_img: u32,
    /// Bindless storage index of the coarse volume's cell image.
    pub coarse_cells_img: u32,
}

pub struct WorldGpu {
    pub voxels: Buffer,
    pub columns: Buffer,
    /// props pass: prop instances are built here before they are drawn
    pub prop_prims: Buffer,
    pub prop_segs: Buffer,
    pub quads: Buffer,
    pub records: Buffer,
    table: Buffer,
    results: Buffer,
    batch_stats: Buffer,
    record_staging: Vec<Buffer>,
    slots: Vec<BatchSlot>,
    pub quad_capacity: u32,
    pub scratch_capacity: u32,
    pub last_batch: (u32, u32),
    pub params: Buffer,
    params_staging: Buffer,
    params_dirty: bool,
    pub param_values: Vec<f32>,
    /// solo preview zone and biome (global ids, -1 = off), passed in the push constants
    pub solo: [f32; 2],
    pub prefab_addrs: (u64, u64),
}

const RESULTS_BYTES: u64 = (MAX_JOBS * std::mem::size_of::<JobResult>()) as u64;

impl WorldGpu {
    pub fn new(ctx: &VkContext, frames_in_flight: usize, quad_capacity: u32, max_records: u32) -> Result<Self> {
        let scratch_capacity = 2 << 20; // 2M quads (16 MiB) per batch slot
        let mut slots = Vec::new();
        for i in 0..BATCH_SLOTS {
            slots.push(BatchSlot {
                scratch: Buffer::gpu(ctx, &format!("scratch_quads_{i}"), scratch_capacity as u64 * 8)?,
                readback: Buffer::readback(ctx, &format!("batch_readback_{i}"), RESULTS_BYTES + 64)?,
                voxel_readback: None,
                jobs_buf: Buffer::upload(ctx, &format!("jobs_{i}"), (MAX_JOBS * 32) as u64)?,
                job_lists: Buffer::upload(ctx, &format!("job_lists_{i}"), (MAX_JOBS * MAX_LAYERS * 4) as u64)?,
                edits_buf: Buffer::upload(ctx, &format!("edits_{i}"), 1 << 20)?,
                state: SlotState::Free,
                jobs: Vec::new(),
            });
        }
        Ok(Self {
            voxels: Buffer::gpu(ctx, "voxel_slots", MAX_JOBS as u64 * SLOT_U16 * 2)?,
            columns: Buffer::gpu(ctx, "column_info", (MAX_JOBS * 34 * 34 * 2) as u64)?,
            prop_prims: Buffer::gpu(ctx, "prop_prim_scratch", PROP_PRIM_SCRATCH)?,
            prop_segs: Buffer::gpu(ctx, "prop_seg_scratch", PROP_SEG_SCRATCH)?,
            quads: Buffer::gpu(ctx, "quad_pool", quad_capacity as u64 * 8)?,
            records: Buffer::gpu(ctx, "chunk_records", max_records as u64 * std::mem::size_of::<GpuChunkRecord>() as u64)?,
            table: Buffer::gpu(ctx, "mesh_table", (MAX_JOBS * NUM_STREAMS * 32 * 4) as u64)?,
            results: Buffer::gpu(ctx, "job_results", RESULTS_BYTES)?,
            batch_stats: Buffer::gpu(ctx, "batch_stats", 64)?,
            record_staging: (0..frames_in_flight)
                // room for every record: after deduplication a frame writes each at most once
                .map(|i| Buffer::upload(ctx, &format!("record_staging_{i}"), max_records as u64 * std::mem::size_of::<GpuChunkRecord>() as u64))
                .collect::<Result<_>>()?,
            slots,
            quad_capacity,
            scratch_capacity,
            last_batch: (0, 0),
            params: Buffer::gpu(ctx, "graph_params", worldgen::compile::PARAM_CAPACITY as u64 * 4)?,
            params_staging: Buffer::upload(ctx, "graph_params_staging", worldgen::compile::PARAM_CAPACITY as u64 * 4)?,
            params_dirty: true,
            solo: [-1.0, -1.0],
            param_values: vec![0.0; 16],
            prefab_addrs: (0, 0),
        })
    }

    pub fn set_params(&mut self, values: &[f32]) {
        self.param_values = values.to_vec();
        self.params_dirty = true;
    }

    /// Solo preview: zone and biome (global ids) shown everywhere (None = off).
    pub fn set_solo(&mut self, zone: Option<u32>, biome: Option<u32>) {
        self.solo = [zone.map_or(-1.0, |z| z as f32), biome.map_or(-1.0, |b| b as f32)];
    }

    pub fn has_free_slot(&self) -> bool {
        self.slots.iter().any(|s| s.state == SlotState::Free)
    }

    pub fn busy_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.state != SlotState::Free).count()
    }

    /// Voxel data (34^3 u16 incl. apron) of a readback job from the last poll.
    pub fn take_voxels(&mut self, slot: usize, job_index: usize) -> Option<Vec<u16>> {
        let rb = self.slots[slot].voxel_readback.as_ref()?;
        Some(rb.read::<u16>(job_index as u64 * SLOT_U16 * 2, 34 * 34 * 34))
    }

    /// Collect finished batches. Returns (slot, jobs, results) for each.
    pub fn poll(&mut self, completed: u64) -> Vec<(usize, Vec<JobDesc>, Vec<JobResult>)> {
        let mut out = Vec::new();
        for (i, s) in self.slots.iter_mut().enumerate() {
            match s.state {
                SlotState::Copying(v) if v <= completed => s.state = SlotState::Free,
                SlotState::Generating(v) if v <= completed => {
                    let n = s.jobs.len();
                    let results: Vec<JobResult> = s.readback.read(0, n);
                    let stats: Vec<u32> = s.readback.read(RESULTS_BYTES, 2);
                    self.last_batch = (stats[0], stats[1]);
                    s.state = SlotState::Polled;
                    out.push((i, std::mem::take(&mut s.jobs), results));
                }
                _ => {}
            }
        }
        out
    }

    /// Record pool copies + chunk record updates (start of frame, before culling).
    pub fn record_uploads(
        &mut self,
        ctx: &VkContext,
        cb: vk::CommandBuffer,
        frame_slot: usize,
        frame_value: u64,
        copies: &[PoolCopy],
        updates: &[(u32, GpuChunkRecord)],
    ) {
        let d = &ctx.device;
        let mut any = false;
        if self.params_dirty {
            let n = self.param_values.len().min(worldgen::compile::PARAM_CAPACITY);
            if n > 0 {
                self.params_staging.write(0, &self.param_values[..n]);
                let region = vk::BufferCopy { src_offset: 0, dst_offset: 0, size: (n * 4) as u64 };
                unsafe { d.cmd_copy_buffer(cb, self.params_staging.raw, self.params.raw, &[region]) };
                any = true;
            }
            self.params_dirty = false;
        }
        for slot in 0..self.slots.len() {
            let regions: Vec<vk::BufferCopy> = copies
                .iter()
                .filter(|c| c.slot == slot && c.count > 0)
                .map(|c| vk::BufferCopy { src_offset: c.src as u64 * 8, dst_offset: c.dst as u64 * 8, size: c.count as u64 * 8 })
                .collect();
            if self.slots[slot].state == SlotState::Polled {
                self.slots[slot].state = if regions.is_empty() { SlotState::Free } else { SlotState::Copying(frame_value) };
            }
            if !regions.is_empty() {
                unsafe { d.cmd_copy_buffer(cb, self.slots[slot].scratch.raw, self.quads.raw, &regions) };
                any = true;
            }
        }
        if !updates.is_empty() {
            let rec_size = std::mem::size_of::<GpuChunkRecord>() as u64;
            let staging = &mut self.record_staging[frame_slot];
            let max = (staging.size / rec_size) as usize;
            // A record can change twice in a frame (freed, then reused right away). Only the last
            // write counts, and one copy must not write the same bytes twice (the order of
            // overlapping regions is undefined).
            let mut last = std::collections::HashMap::with_capacity(updates.len());
            for (i, (idx, _)) in updates.iter().enumerate() {
                last.insert(*idx, i);
            }
            let mut regions = Vec::with_capacity(last.len().min(max));
            for (i, (idx, rec)) in updates.iter().enumerate() {
                if last[idx] != i || regions.len() == max {
                    continue;
                }
                let at = regions.len() as u64 * rec_size;
                staging.write(at, std::slice::from_ref(rec));
                regions.push(vk::BufferCopy { src_offset: at, dst_offset: *idx as u64 * rec_size, size: rec_size });
            }
            unsafe { d.cmd_copy_buffer(cb, staging.raw, self.records.raw, &regions) };
            any = true;
        }
        if any {
            memory_barrier(
                ctx,
                cb,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::TASK_SHADER_EXT
                    | vk::PipelineStageFlags2::MESH_SHADER_EXT
                    | vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE,
            );
        }
    }

    /// Record one generation + meshing batch. Returns false (and records nothing) when no batch
    /// slot is free.
    #[allow(clippy::too_many_arguments)]
    pub fn record_batch(
        &mut self,
        ctx: &VkContext,
        cb: vk::CommandBuffer,
        frame_value: u64,
        jobs: Vec<JobDesc>,
        pipes: &WorldPipelines,
        block_table: u64,
        seed: u32,
        profiler: &mut GpuProfiler,
        frame_slot: usize,
    ) -> Result<(), Vec<JobDesc>> {
        let Some(slot_idx) = self.slots.iter().position(|s| s.state == SlotState::Free) else {
            return Err(jobs);
        };
        if jobs.is_empty() {
            return Ok(());
        }
        let n = jobs.len().min(MAX_JOBS);
        let d = &ctx.device;

        // ---- upload jobs and edits ----
        let total_edits: usize = jobs.iter().map(|j| j.edits.len()).sum();
        {
            let slot = &mut self.slots[slot_idx];
            if (total_edits * 8) as u64 > slot.edits_buf.size {
                let mut old = std::mem::replace(
                    &mut slot.edits_buf,
                    Buffer::upload(ctx, "edits_grown", ((total_edits * 8) as u64).next_power_of_two()).map_err(|_| jobs.clone())?,
                );
                old.destroy(ctx);
            }
            let mut gpu_jobs = Vec::with_capacity(n);
            let mut edits: Vec<[u32; 2]> = Vec::with_capacity(total_edits);
            for (i, j) in jobs.iter().take(n).enumerate() {
                gpu_jobs.push(match j.preview {
                    Some((graph, seed)) => GpuGenJob {
                        origin: j.key.origin().to_array(),
                        lod: 0,
                        edit_offset: seed,
                        edit_count: 0,
                        flags: GENJOB_PREVIEW,
                        pad: graph,
                    },
                    None => GpuGenJob {
                        origin: j.key.origin().to_array(),
                        lod: j.key.lod as u32,
                        edit_offset: edits.len() as u32,
                        edit_count: j.edits.len() as u32,
                        flags: j.light,
                        pad: 0,
                    },
                });
                for &(idx, b) in &j.edits {
                    edits.push([i as u32 | ((b as u32) << 8), idx]);
                }
            }
            slot.jobs_buf.write(0, &gpu_jobs);
            if !edits.is_empty() {
                slot.edits_buf.write(0, &edits);
            }
        }
        // ---- which layers generate which jobs ----
        // Every layer owning some of a job's cells runs on it; one of them (the owner of the
        // chunk's middle, else the first) writes its column biomes. Jobs no single layer covers
        // start from air: cells no layer owns stay empty. A job with a sole layer is generated by
        // that layer alone, owning every cell (a second dispatch of the layer: groups
        // `layers.len()..`).
        let layers = &pipes.layers[..pipes.layers.len().min(MAX_LAYERS / 2)];
        let nl = layers.len();
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nl * 2];
        let mut clear: Vec<(u32, bool)> = Vec::new();
        for (i, j) in jobs.iter().take(n).enumerate() {
            if j.preview.is_some() {
                if nl > 0 {
                    lists[0].push(i as u32 | JOBLIST_COLUMNS);
                }
                continue;
            }
            if let Some(li) = j.sole_layer.map(|l| l as usize).filter(|&l| l < nl) {
                lists[nl + li].push(i as u32 | JOBLIST_COLUMNS);
                continue;
            }
            let mid = j.key.origin().y as f32 + (16i32 << j.key.lod) as f32;
            let mut col: Option<usize> = None;
            let mut full = false;
            let mut hits = Vec::new();
            for (li, l) in layers.iter().enumerate() {
                let (any, all) = l.covers(&j.key);
                if !any {
                    continue;
                }
                full |= all;
                hits.push(li);
                if mid >= l.own.0 && mid < l.own.1 {
                    col = Some(li);
                }
            }
            let col = col.or(hits.first().copied());
            for &li in &hits {
                lists[li].push(i as u32 | if Some(li) == col { JOBLIST_COLUMNS } else { 0 });
            }
            if !full {
                clear.push((i as u32, col.is_none()));
            }
        }
        let mut list_offsets = Vec::with_capacity(lists.len());
        {
            let flat: Vec<u32> = lists.iter().flatten().copied().collect();
            let mut at = 0u64;
            for l in &lists {
                list_offsets.push(at);
                at += l.len() as u64 * 4;
            }
            if !flat.is_empty() {
                self.slots[slot_idx].job_lists.write(0, &flat);
            }
        }
        if jobs.iter().take(n).any(|j| j.readback) && self.slots[slot_idx].voxel_readback.is_none() {
            let b = Buffer::readback(ctx, "voxel_readback", MAX_JOBS as u64 * SLOT_U16 * 2).map_err(|_| jobs.clone())?;
            self.slots[slot_idx].voxel_readback = Some(b);
        }
        let slot = &self.slots[slot_idx];
        let edit_count = total_edits as u32;

        profiler.begin(ctx, cb, frame_slot, "worldgen");
        compute_barrier(ctx, cb);
        // zeroed results; jobs that are only read (region reads, light-only) skip meshing
        let mut init = vec![JobResult::default(); n];
        for (r, j) in init.iter_mut().zip(&jobs) {
            if j.counter == u32::MAX {
                r.flags = JOB_NO_MESH;
            }
        }
        unsafe {
            d.cmd_fill_buffer(cb, self.results.raw, 0, RESULTS_BYTES, 0);
            for &(i, columns) in &clear {
                d.cmd_fill_buffer(cb, self.voxels.raw, i as u64 * SLOT_U16 * 2, SLOT_CELLS * 2, 0);
                if columns {
                    d.cmd_fill_buffer(cb, self.columns.raw, i as u64 * SLOT_COLUMNS * 2, SLOT_COLUMNS * 2, 0);
                }
            }
        }
        memory_barrier(
            ctx,
            cb,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
        unsafe {
            d.cmd_update_buffer(cb, self.results.raw, 0, bytemuck::cast_slice(&init));
        }
        memory_barrier(
            ctx,
            cb,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
        let push = |p: &[u8]| unsafe { d.cmd_push_constants(cb, pipes.layout, vk::ShaderStageFlags::ALL, 0, p) };
        // group g: layer g % nl, owning its cells (g < nl) or everything
        let layer_push = |g: usize| GenPush {
            jobs: slot.jobs_buf.address,
            voxels: self.voxels.address,
            results: self.results.address,
            edits: slot.edits_buf.address,
            params: self.params.address + layers[g % nl].param_base as u64 * 4,
            columns: self.columns.address,
            prefab_table: self.prefab_addrs.0,
            prefab_data: self.prefab_addrs.1,
            num_jobs: lists[g].len() as u32,
            seed,
            edit_count,
            pad: 0,
            blocks: block_table,
            prop_prims: self.prop_prims.address,
            prop_segs: self.prop_segs.address,
            job_list: slot.job_lists.address + list_offsets[g],
            own_min: if g < nl { layers[g].own.0.max(-1.0e30) } else { -1.0e30 },
            own_max: if g < nl { layers[g].own.1.min(1.0e30) } else { 1.0e30 },
            solo_zone: self.solo[0],
            solo_biome: self.solo[1],
        };
        // terrain: the groups write disjoint cells, so they run back to back
        for g in (0..lists.len()).filter(|&g| !lists[g].is_empty()) {
            unsafe {
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, layers[g % nl].terrain);
                push(bytemuck::bytes_of(&layer_push(g)));
                d.cmd_dispatch(cb, 5, 5, lists[g].len() as u32);
            }
        }
        compute_barrier(ctx, cb);
        profiler.end(ctx, cb, frame_slot);
        profiler.begin(ctx, cb, frame_slot, "props");
        // props: a job's prop scratch is shared by its layers, so a layer that shares jobs with
        // one dispatched since the last barrier waits for it
        let mut busy = vec![false; n];
        for g in (0..lists.len()).filter(|&g| !lists[g].is_empty()) {
            if lists[g].iter().any(|&e| busy[(e & !JOBLIST_COLUMNS) as usize]) {
                compute_barrier(ctx, cb);
                busy.iter_mut().for_each(|b| *b = false);
            }
            for &e in &lists[g] {
                busy[(e & !JOBLIST_COLUMNS) as usize] = true;
            }
            unsafe {
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, layers[g % nl].props);
                push(bytemuck::bytes_of(&layer_push(g)));
                d.cmd_dispatch(cb, lists[g].len() as u32, 1, 1);
            }
        }
        compute_barrier(ctx, cb);
        if edit_count > 0 {
            unsafe {
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipes.apply_edits);
                push(bytemuck::bytes_of(&EditPush {
                    voxels: self.voxels.address,
                    edits: slot.edits_buf.address,
                    results: self.results.address,
                    count: edit_count,
                    pad: 0,
                }));
                d.cmd_dispatch(cb, edit_count.div_ceil(64), 1, 1);
            }
            compute_barrier(ctx, cb);
        }
        // light volumes: copy the final blocks of flagged LOD0 jobs
        if jobs.iter().take(n).any(|j| j.light & GENJOB_COARSE != 0) {
            unsafe {
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipes.coarse_fill);
                push(bytemuck::bytes_of(&CoarseFillPush {
                    jobs: slot.jobs_buf.address,
                    voxels: self.voxels.address,
                    blocks: block_table,
                    num_jobs: n as u32,
                    coarse_img: pipes.coarse_cells_img,
                }));
                d.cmd_dispatch(cb, n as u32, 1, 1);
            }
        }
        if jobs.iter().take(n).any(|j| j.light & GENJOB_LIGHT != 0) {
            unsafe {
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipes.light_fill);
                push(bytemuck::bytes_of(&LightFillPush {
                    jobs: slot.jobs_buf.address,
                    voxels: self.voxels.address,
                    num_jobs: n as u32,
                    blocks_img: pipes.light_blocks_img,
                }));
                d.cmd_dispatch(cb, 64, n as u32, 1);
            }
            compute_barrier(ctx, cb);
        }
        // region reads: copy the final voxels of flagged jobs to the host
        let rb_regions: Vec<vk::BufferCopy> = jobs
            .iter()
            .take(n)
            .enumerate()
            .filter(|(_, j)| j.readback)
            .map(|(i, _)| vk::BufferCopy { src_offset: i as u64 * SLOT_U16 * 2, dst_offset: i as u64 * SLOT_U16 * 2, size: SLOT_U16 * 2 })
            .collect();
        if !rb_regions.is_empty()
            && let Some(rb) = &slot.voxel_readback {
                unsafe { d.cmd_copy_buffer(cb, self.voxels.raw, rb.raw, &rb_regions) };
            }
        profiler.end(ctx, cb, frame_slot);
        profiler.begin(ctx, cb, frame_slot, "meshing");
        let mesh_push = MeshPush {
            voxels: self.voxels.address,
            blocks: block_table,
            columns: self.columns.address,
            table: self.table.address,
            results: self.results.address,
            scratch: slot.scratch.address,
            num_jobs: n as u32,
            scratch_capacity: self.scratch_capacity,
        };
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipes.mesh_count);
            push(bytemuck::bytes_of(&mesh_push));
            d.cmd_dispatch(cb, 6, n as u32, 1);
        }
        compute_barrier(ctx, cb);
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipes.prefix);
            push(bytemuck::bytes_of(&PrefixPush {
                table: self.table.address,
                results: self.results.address,
                stats: self.batch_stats.address,
                num_jobs: n as u32,
                scratch_capacity: self.scratch_capacity,
            }));
            d.cmd_dispatch(cb, 1, 1, 1);
        }
        compute_barrier(ctx, cb);
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipes.mesh_write);
            push(bytemuck::bytes_of(&mesh_push));
            d.cmd_dispatch(cb, 6, n as u32, 1);
        }
        compute_barrier(ctx, cb);
        unsafe {
            d.cmd_copy_buffer(
                cb,
                self.results.raw,
                slot.readback.raw,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: (n * std::mem::size_of::<JobResult>()) as u64 }],
            );
            d.cmd_copy_buffer(
                cb,
                self.batch_stats.raw,
                slot.readback.raw,
                &[vk::BufferCopy { src_offset: 0, dst_offset: RESULTS_BYTES, size: 8 }],
            );
        }
        memory_barrier(
            ctx,
            cb,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::HOST,
            vk::AccessFlags2::HOST_READ,
        );
        profiler.end(ctx, cb, frame_slot);

        let slot = &mut self.slots[slot_idx];
        slot.state = SlotState::Generating(frame_value);
        let mut jobs = jobs;
        let rest = if jobs.len() > n { jobs.split_off(n) } else { Vec::new() };
        slot.jobs = jobs;
        if rest.is_empty() { Ok(()) } else { Err(rest) }
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        for s in &mut self.slots {
            if let Some(mut b) = s.voxel_readback.take() {
                b.destroy(ctx);
            }
            s.scratch.destroy(ctx);
            s.readback.destroy(ctx);
            s.jobs_buf.destroy(ctx);
            s.job_lists.destroy(ctx);
            s.edits_buf.destroy(ctx);
        }
        for b in &mut self.record_staging {
            b.destroy(ctx);
        }
        self.voxels.destroy(ctx);
        self.columns.destroy(ctx);
        self.prop_prims.destroy(ctx);
        self.prop_segs.destroy(ctx);
        self.quads.destroy(ctx);
        self.records.destroy(ctx);
        self.table.destroy(ctx);
        self.results.destroy(ctx);
        self.batch_stats.destroy(ctx);
        self.params.destroy(ctx);
        self.params_staging.destroy(ctx);
    }
}
