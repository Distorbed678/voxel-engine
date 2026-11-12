//! Ambient particles and weather (`shaders/particles/`). A fixed pool of GPU particles lives in a
//! 64³ box that wraps around the camera. Each frame `sim.comp` respawns free slots with a kind picked
//! from per-kind shares, moves them, and kills those inside solid blocks (and rain and snow under a
//! roof) using the light volume. The translucent pass draws them as billboards.
//!
//! The shares come from the camera's biome (Biome Output: ambient particles, precipitation,
//! density) and the weather intensity, blended over time like fog.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use vkcore::{Buffer, GpuProfiler, VkContext, vk};

use crate::renderer::{DEPTH_FORMAT, HDR_FORMAT};
use crate::shaders::{GraphicsState, PipelineHandle, PipelineManager, PipelineSource};
use vkcore::pipeline::BlendMode;

/// Particle slots (32 bytes each).
pub const MAX_PARTICLES: u32 = 24576;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ParticlePush {
    frame: u64,
    particles: u64,
    blocks: u64,
    weights0: [f32; 4],
    weights1: [f32; 4],
    count: u32,
    light_blocks: u32,
    frame_index: u32,
    dt: f32,
}

/// Share of the particle slots per kind (sums to <= 1).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Shares {
    pub snow: f32,
    pub rain: f32,
    pub ash: f32,
    pub ember: f32,
    pub firefly: f32,
    pub pollen: f32,
}

impl Shares {
    /// Target shares for a biome's settings, the weather intensity and the time of day.
    /// `ambient`: Biome Output index (None, Ash, Embers, Fireflies, Pollen); `precipitation`:
    /// (None, Rain, Snow).
    pub fn for_biome(ambient: u32, precipitation: u32, density: f32, weather: f32, daylight: f32) -> Self {
        let mut s = Shares::default();
        let d = density.clamp(0.0, 4.0);
        match ambient {
            1 => s.ash = 0.35 * d,
            2 => {
                s.ember = 0.12 * d;
                s.ash = 0.18 * d;
            }
            3 => s.firefly = 0.05 * d * (1.0 - daylight),
            4 => s.pollen = 0.06 * d * daylight,
            _ => {}
        }
        match precipitation {
            1 => s.rain = 0.9 * weather,
            2 => s.snow = (0.2 + 0.7 * weather) * d.min(1.5),
            _ => {}
        }
        // never more than the pool
        let total = s.snow + s.rain + s.ash + s.ember + s.firefly + s.pollen;
        if total > 1.0 {
            for v in [&mut s.snow, &mut s.rain, &mut s.ash, &mut s.ember, &mut s.firefly, &mut s.pollen] {
                *v /= total;
            }
        }
        s
    }

    pub fn lerp(self, to: Shares, k: f32) -> Shares {
        let m = |a: f32, b: f32| a + (b - a) * k;
        Shares {
            snow: m(self.snow, to.snow),
            rain: m(self.rain, to.rain),
            ash: m(self.ash, to.ash),
            ember: m(self.ember, to.ember),
            firefly: m(self.firefly, to.firefly),
            pollen: m(self.pollen, to.pollen),
        }
    }

}

pub struct Particles {
    buffer: Buffer,
    sim: PipelineHandle,
    pub draw: PipelineHandle,
    pub shares: Shares,
    initialized: bool,
}

impl Particles {
    pub fn new(ctx: &VkContext, pm: &mut PipelineManager) -> Result<Self> {
        let sim = pm.create(ctx, "particles_sim", PipelineSource::Compute { path: "particles/sim.comp".into(), defines: vec![] })?;
        let draw = pm.create(
            ctx,
            "particles",
            PipelineSource::Graphics {
                stages: vec![("particles/particles.vert".into(), vec![]), ("particles/particles.frag".into(), vec![])],
                state: GraphicsState {
                    color_formats: vec![HDR_FORMAT],
                    blend: vec![BlendMode::Alpha],
                    depth_format: Some(DEPTH_FORMAT),
                    depth_test: true,
                    depth_write: false,
                    cull: vk::CullModeFlags::NONE,
                    ..Default::default()
                },
            },
        )?;
        Ok(Self {
            buffer: Buffer::gpu(ctx, "particles", MAX_PARTICLES as u64 * 32)?,
            sim,
            draw,
            shares: Shares::default(),
            initialized: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push(&self, frame: u64, blocks: u64, light_blocks: u32, wind: Vec2, frame_index: u32, dt: f32) -> ParticlePush {
        let s = self.shares;
        ParticlePush {
            frame,
            particles: self.buffer.address,
            blocks,
            weights0: [s.snow, s.rain, s.ash, s.ember],
            weights1: [s.firefly, s.pollen, wind.x, wind.y],
            count: MAX_PARTICLES,
            light_blocks,
            frame_index,
            dt,
        }
    }

    /// Simulate one step (before the scene is drawn).
    pub fn record_sim(&mut self, ctx: &VkContext, cb: vk::CommandBuffer, pm: &PipelineManager, set: vk::DescriptorSet, push: &ParticlePush, profiler: &mut GpuProfiler, slot: usize) {
        let d = &ctx.device;
        if !self.initialized {
            // all slots start free
            unsafe { d.cmd_fill_buffer(cb, self.buffer.raw, 0, vk::WHOLE_SIZE, 0) };
            self.initialized = true;
        }
        profiler.begin(ctx, cb, slot, "particles");
        vkcore::sync::compute_barrier(ctx, cb);
        unsafe {
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, pm.layout, 0, &[set], &[]);
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pm.get(self.sim));
            d.cmd_push_constants(cb, pm.layout, vk::ShaderStageFlags::ALL, 0, bytemuck::bytes_of(push));
            d.cmd_dispatch(cb, MAX_PARTICLES.div_ceil(64), 1, 1);
        }
        vkcore::sync::memory_barrier(
            ctx,
            cb,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
        profiler.end(ctx, cb, slot);
    }

    /// Free every slot on the next simulation step.
    pub fn reset(&mut self) {
        self.initialized = false;
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        self.buffer.destroy(ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shares_follow_biome_weather_and_time() {
        // snow biomes always snow a little, more with weather
        let calm = Shares::for_biome(0, 2, 1.0, 0.0, 1.0);
        let storm = Shares::for_biome(0, 2, 1.0, 1.0, 1.0);
        assert!(calm.snow > 0.0 && storm.snow > calm.snow);
        // rain only with weather
        assert_eq!(Shares::for_biome(0, 1, 1.0, 0.0, 1.0).rain, 0.0);
        // fireflies only at night, pollen only by day
        assert_eq!(Shares::for_biome(3, 0, 1.0, 0.0, 1.0).firefly, 0.0);
        assert!(Shares::for_biome(3, 0, 1.0, 0.0, 0.0).firefly > 0.0);
        assert_eq!(Shares::for_biome(4, 0, 1.0, 0.0, 0.0).pollen, 0.0);
        // never more than the pool
        let s = Shares::for_biome(2, 2, 4.0, 1.0, 0.0);
        assert!(s.snow + s.rain + s.ash + s.ember + s.firefly + s.pollen <= 1.0 + 1e-5);
    }
}
