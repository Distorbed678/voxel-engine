use anyhow::Result;
use ash::vk;

use crate::VkContext;

const MAX_SCOPES: u32 = 64;

struct FrameQueries {
    pool: vk::QueryPool,
    names: Vec<&'static str>,
    open: Vec<u32>,
    used: u32,
}

/// Per-pass GPU timestamps. One query pool per frame in flight; results are read back once
/// that frame's timeline value has been reached, so reading never stalls.
pub struct GpuProfiler {
    frames: Vec<FrameQueries>,
    period_ns: f32,
    /// (scope name, milliseconds) of the most recent completed frame.
    pub last: Vec<(&'static str, f32)>,
    pub enabled: bool,
}

impl GpuProfiler {
    pub fn new(ctx: &VkContext, frames_in_flight: usize) -> Result<Self> {
        let mut frames = Vec::new();
        for _ in 0..frames_in_flight {
            let pool = unsafe {
                ctx.device.create_query_pool(
                    &vk::QueryPoolCreateInfo::default()
                        .query_type(vk::QueryType::TIMESTAMP)
                        .query_count(MAX_SCOPES * 2),
                    None,
                )?
            };
            unsafe { ctx.device.reset_query_pool(pool, 0, MAX_SCOPES * 2) };
            frames.push(FrameQueries { pool, names: Vec::new(), open: Vec::new(), used: 0 });
        }
        Ok(Self {
            frames,
            period_ns: ctx.props.limits.timestamp_period,
            last: Vec::new(),
            enabled: true,
        })
    }

    /// Call after the frame slot's previous GPU work is known to be complete.
    pub fn begin_frame(&mut self, ctx: &VkContext, slot: usize) {
        let f = &mut self.frames[slot];
        if f.used > 0 {
            let mut data = vec![[0u64; 2]; f.used as usize * 2];
            let ok = unsafe {
                ctx.device.get_query_pool_results(
                    f.pool,
                    0,
                    &mut data,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
                )
            };
            if ok.is_ok() {
                self.last.clear();
                for (i, name) in f.names.iter().enumerate() {
                    let a = data[i * 2];
                    let b = data[i * 2 + 1];
                    if a[1] != 0 && b[1] != 0 {
                        let ms = (b[0].saturating_sub(a[0])) as f32 * self.period_ns / 1.0e6;
                        self.last.push((name, ms));
                    }
                }
            }
            unsafe { ctx.device.reset_query_pool(f.pool, 0, MAX_SCOPES * 2) };
        }
        f.names.clear();
        f.open.clear();
        f.used = 0;
    }

    pub fn begin(&mut self, ctx: &VkContext, cb: vk::CommandBuffer, slot: usize, name: &'static str) {
        ctx.begin_label(cb, name);
        if !self.enabled {
            return;
        }
        let f = &mut self.frames[slot];
        if f.used >= MAX_SCOPES {
            f.open.push(u32::MAX);
            return;
        }
        let idx = f.used;
        f.used += 1;
        f.names.push(name);
        f.open.push(idx);
        unsafe {
            ctx.device.cmd_write_timestamp2(
                cb,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                f.pool,
                idx * 2,
            );
        }
    }

    pub fn end(&mut self, ctx: &VkContext, cb: vk::CommandBuffer, slot: usize) {
        ctx.end_label(cb);
        if !self.enabled {
            return;
        }
        let f = &mut self.frames[slot];
        if let Some(idx) = f.open.pop() {
            if idx != u32::MAX {
                unsafe {
                    ctx.device.cmd_write_timestamp2(
                        cb,
                        vk::PipelineStageFlags2::ALL_COMMANDS,
                        f.pool,
                        idx * 2 + 1,
                    );
                }
            }
        }
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        for f in self.frames.drain(..) {
            unsafe { ctx.device.destroy_query_pool(f.pool, None) };
        }
    }
}
