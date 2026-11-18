//! Ambient particles and weather (`shaders/particles/`). A fixed pool of GPU particles lives in a
//! 64³ box that wraps around the camera. Each frame `sim.comp` respawns free slots with a type
//! picked from the active types' shares (where the type's spawn rules allow), moves them by the
//! type's motion, and kills those inside solid blocks (and open-sky types under a roof) using the
//! light volume. The translucent pass draws them as billboards.
//!
//! Particle types are Particle graphs (`worldgen::particles`), uploaded as a table. The shares
//! come from the camera's environment (Particles nodes: which types, a density) and the weather
//! intensity, blended over time.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use vkcore::{Buffer, GpuProfiler, VkContext, vk};

use crate::renderer::{DEPTH_FORMAT, HDR_FORMAT};
use crate::shaders::{GraphicsState, PipelineHandle, PipelineManager, PipelineSource};
use vkcore::pipeline::BlendMode;

/// Particle slots (32 bytes each).
pub const MAX_PARTICLES: u32 = 24576;
/// Particle types in the GPU table (Particle graphs).
pub const MAX_TYPES: usize = 64;
/// Types spawning at once (the camera's set, plus those fading out after a change).
pub const ACTIVE: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ParticlePush {
    frame: u64,
    particles: u64,
    blocks: u64,
    types: u64,
    /// type index + 1 of each active type (0 = unused), and its share of the pool
    active: [u32; ACTIVE],
    share: [f32; ACTIVE],
    wind: [f32; 2],
    count: u32,
    light_blocks: u32,
    frame_index: u32,
    dt: f32,
}

/// One particle type as the GPU reads it (`ParticleType` in particles_common.glsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct GpuParticleType {
    /// rgb, a = opacity
    color: [f32; 4],
    /// rgb (each particle mixes the two at random), a = glow
    color2: [f32; 4],
    /// size, blink, shape, life
    look: [f32; 4],
    /// motion, speed, sway, wind
    motion: [f32; 4],
    /// flags (1 = open sky only), near class, near block id, unused
    spawn: [u32; 4],
}

impl GpuParticleType {
    pub fn new(d: &worldgen::particles::ParticleDef, block_ids: &std::collections::HashMap<String, u32>) -> Self {
        GpuParticleType {
            color: [d.color[0], d.color[1], d.color[2], d.alpha],
            color2: [d.color2[0], d.color2[1], d.color2[2], d.glow],
            look: [d.size, d.blink, d.shape as f32, d.life],
            motion: [d.motion as f32, d.speed, d.sway, d.wind],
            spawn: [d.open_sky as u32, d.near, block_ids.get(&d.near_block).copied().unwrap_or(u32::MAX), 0],
        }
    }
}

/// The share of the particle pool each type wants (type index -> share): the environment's set,
/// each type's amount at this weather and time of day, times the set's density, and never more
/// than the whole pool.
pub fn target_shares(defs: &[worldgen::particles::ParticleDef], set: &worldgen::env::ParticlesEnv, weather: f32, daylight: f32) -> Vec<f32> {
    let mut out = vec![0.0; defs.len()];
    for name in &set.types {
        if let Some(i) = defs.iter().position(|d| &d.name == name) {
            out[i] = defs[i].share(weather, daylight) * set.density.clamp(0.0, 4.0);
        }
    }
    let total: f32 = out.iter().sum();
    if total > 1.0 {
        out.iter_mut().for_each(|v| *v /= total);
    }
    out
}

pub struct Particles {
    buffer: Buffer,
    sim: PipelineHandle,
    pub draw: PipelineHandle,
    /// the GPU table of particle types
    types: Buffer,
    /// what the table holds (the type definitions, and the block ids their spawn rules name)
    type_defs: Vec<worldgen::particles::ParticleDef>,
    table: Vec<GpuParticleType>,
    /// the current share of the pool per type (blended toward the environment's)
    pub shares: Vec<f32>,
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
            types: Buffer::upload(ctx, "particle_types", (MAX_TYPES * std::mem::size_of::<GpuParticleType>()) as u64)?,
            type_defs: Vec::new(),
            table: Vec::new(),
            shares: Vec::new(),
            initialized: false,
        })
    }

    /// Upload the particle types (after the world generator compiles). Particles of types whose
    /// definition moved to another index are cleared.
    pub fn set_types(&mut self, defs: &[worldgen::particles::ParticleDef], block_ids: &std::collections::HashMap<String, u32>) {
        let table: Vec<GpuParticleType> = defs.iter().take(MAX_TYPES).map(|d| GpuParticleType::new(d, block_ids)).collect();
        if defs == self.type_defs && bytemuck::cast_slice::<_, u8>(&table) == bytemuck::cast_slice::<_, u8>(&self.table) {
            return;
        }
        let names = |d: &[worldgen::particles::ParticleDef]| d.iter().map(|d| d.name.clone()).collect::<Vec<_>>();
        if names(defs) != names(&self.type_defs) {
            self.shares = vec![0.0; defs.len()];
            self.initialized = false;
        }
        self.types.write(0, &table);
        self.table = table;
        self.type_defs = defs.to_vec();
    }

    /// Move the shares a step `k` (0..1) toward `target`.
    pub fn blend_shares(&mut self, target: &[f32], k: f32) {
        self.shares.resize(target.len(), 0.0);
        for (s, t) in self.shares.iter_mut().zip(target) {
            *s += (t - *s) * k;
        }
    }

    /// The (up to ACTIVE) types with the largest shares.
    fn active(&self) -> ([u32; ACTIVE], [f32; ACTIVE]) {
        let mut order: Vec<usize> = (0..self.shares.len().min(MAX_TYPES)).filter(|&i| self.shares[i] > 1e-4).collect();
        order.sort_by(|&a, &b| self.shares[b].total_cmp(&self.shares[a]));
        let (mut ids, mut shares) = ([0u32; ACTIVE], [0.0f32; ACTIVE]);
        for (k, &i) in order.iter().take(ACTIVE).enumerate() {
            ids[k] = i as u32 + 1;
            shares[k] = self.shares[i];
        }
        (ids, shares)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push(&self, frame: u64, blocks: u64, light_blocks: u32, wind: Vec2, frame_index: u32, dt: f32) -> ParticlePush {
        let (active, share) = self.active();
        ParticlePush {
            frame,
            particles: self.buffer.address,
            blocks,
            types: self.types.address,
            active,
            share,
            wind: wind.to_array(),
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
        self.types.destroy(ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shares_follow_the_environment_weather_and_time() {
        let defs = worldgen::particles::particle_defs(&worldgen::defaults::default_project());
        let set = |t: &[&str]| worldgen::env::ParticlesEnv { types: t.iter().map(|s| s.to_string()).collect(), density: 1.0 };
        let share = |t: &[&str], name: &str, weather: f32, daylight: f32| {
            let i = defs.iter().position(|d| d.name == name).unwrap();
            target_shares(&defs, &set(t), weather, daylight)[i]
        };
        // snow always falls a little, more with weather; rain only with weather
        assert!(share(&["snow"], "snow", 0.0, 1.0) > 0.0 && share(&["snow"], "snow", 1.0, 1.0) > share(&["snow"], "snow", 0.0, 1.0));
        assert_eq!(share(&["rain"], "rain", 0.0, 1.0), 0.0);
        // fireflies only at night, pollen only by day; nothing outside the set
        assert_eq!(share(&["fireflies"], "fireflies", 0.0, 1.0), 0.0);
        assert!(share(&["fireflies"], "fireflies", 0.0, 0.0) > 0.0);
        assert_eq!(share(&["pollen"], "pollen", 0.0, 0.0), 0.0);
        assert_eq!(share(&["pollen"], "snow", 1.0, 1.0), 0.0);
        // never more than the pool
        let all: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        let mut dense = set(&all);
        dense.density = 4.0;
        assert!(target_shares(&defs, &dense, 1.0, 0.0).iter().sum::<f32>() <= 1.0 + 1e-5);
    }
}
