//! Temporal anti-aliasing: per-frame sub-pixel jitter of the projection, and a compute resolve
//! (`shaders/render/taa.comp`) that blends each frame into a ping-ponged history. The tonemap pass
//! reads the resolved image.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};
use vkcore::image::ImageDesc;
use vkcore::sync::{image_barrier, memory_barrier};
use vkcore::{Bindless, GpuProfiler, Image, VkContext, vk};

use crate::renderer::HDR_FORMAT;
use crate::shaders::{PipelineHandle, PipelineManager, PipelineSource};

/// Jitter pattern length (Halton 2,3).
const JITTER_SAMPLES: u64 = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TaaPush {
    frame: u64,
    history_in: u32,
    history_out: u32,
    reset: u32,
    scene_tex: u32,
    jitter: [f32; 2],
}

fn halton(mut i: u64, base: u64) -> f32 {
    let (mut f, mut r) = (1.0f32, 0.0f32);
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

/// Sub-pixel jitter for frame `n`, in pixels (-0.5..0.5).
pub fn jitter(n: u64) -> Vec2 {
    let i = n % JITTER_SAMPLES + 1;
    Vec2::new(halton(i, 2), halton(i, 3)) - 0.5
}

/// Clip-space translation that shifts the image by `jitter_px` pixels.
pub fn jitter_matrix(jitter_px: Vec2, extent: vk::Extent2D) -> Mat4 {
    Mat4::from_translation(Vec3::new(2.0 * jitter_px.x / extent.width as f32, 2.0 * jitter_px.y / extent.height as f32, 0.0))
}

pub struct Taa {
    history: [Image; 2],
    storage: [u32; 2],
    /// Bindless sampled indices of the two history images.
    pub sampled: [u32; 2],
    pipeline: PipelineHandle,
    initialized: bool,
    /// Discard the history on the next resolve (resize, teleport, toggled on).
    pub reset: bool,
}

fn create_history(ctx: &VkContext, extent: vk::Extent2D) -> Result<[Image; 2]> {
    let desc = |name| {
        ImageDesc::d2(name, HDR_FORMAT, extent.width, extent.height, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
    };
    Ok([Image::new(ctx, &desc("taa_history_a"))?, Image::new(ctx, &desc("taa_history_b"))?])
}

impl Taa {
    pub fn new(ctx: &VkContext, pm: &mut PipelineManager, bindless: &mut Bindless, extent: vk::Extent2D) -> Result<Self> {
        let history = create_history(ctx, extent)?;
        let storage = [bindless.add_storage(ctx, history[0].view), bindless.add_storage(ctx, history[1].view)];
        let sampler = bindless.samplers.linear_clamp;
        let sampled = [
            bindless.add_sampled(ctx, history[0].view, sampler, vk::ImageLayout::GENERAL),
            bindless.add_sampled(ctx, history[1].view, sampler, vk::ImageLayout::GENERAL),
        ];
        let pipeline = pm.create(ctx, "taa", PipelineSource::Compute { path: "render/taa.comp".into(), defines: vec![] })?;
        Ok(Self { history, storage, sampled, pipeline, initialized: false, reset: true })
    }

    /// History image written in frame `n` (what the tonemap pass reads).
    pub fn output_index(&self, n: u64) -> usize {
        (n % 2) as usize
    }

    pub fn resize(&mut self, ctx: &VkContext, bindless: &mut Bindless, extent: vk::Extent2D) -> Result<()> {
        for h in &mut self.history {
            h.destroy(ctx);
        }
        self.history = create_history(ctx, extent)?;
        for i in 0..2 {
            bindless.update_storage(ctx, self.storage[i], self.history[i].view);
            bindless.update_sampled(ctx, self.sampled[i], self.history[i].view, bindless.samplers.linear_clamp, vk::ImageLayout::GENERAL);
        }
        self.initialized = false;
        self.reset = true;
        Ok(())
    }

    /// Resolve frame `n` (after the scene pass, before tonemapping).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        ctx: &VkContext,
        cb: vk::CommandBuffer,
        pm: &PipelineManager,
        set: vk::DescriptorSet,
        frame_addr: u64,
        scene_tex: u32,
        extent: vk::Extent2D,
        n: u64,
        jitter_index: u64,
        profiler: &mut GpuProfiler,
        slot: usize,
    ) {
        let d = &ctx.device;
        if !self.initialized {
            for h in &self.history {
                image_barrier(
                    ctx,
                    cb,
                    h.raw,
                    vkcore::sync::color_range(),
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags2::TOP_OF_PIPE,
                    vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                );
            }
            self.initialized = true;
            self.reset = true;
        }
        profiler.begin(ctx, cb, slot, "taa");
        // previous frames' resolve / tonemap must be done with the images we touch
        memory_barrier(
            ctx,
            cb,
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
        let out = self.output_index(n);
        let push = TaaPush {
            frame: frame_addr,
            history_in: self.sampled[1 - out],
            history_out: self.storage[out],
            reset: self.reset as u32,
            scene_tex,
            jitter: jitter(jitter_index).to_array(),
        };
        unsafe {
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, pm.layout, 0, &[set], &[]);
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pm.get(self.pipeline));
            d.cmd_push_constants(cb, pm.layout, vk::ShaderStageFlags::ALL, 0, bytemuck::bytes_of(&push));
            d.cmd_dispatch(cb, extent.width.div_ceil(8), extent.height.div_ceil(8), 1);
        }
        memory_barrier(
            ctx,
            cb,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ,
        );
        self.reset = false;
        profiler.end(ctx, cb, slot);
    }

    pub fn destroy(&mut self, ctx: &VkContext, bindless: &mut Bindless) {
        for i in 0..2 {
            bindless.free_sampled(self.sampled[i]);
            bindless.free_storage(self.storage[i]);
            self.history[i].destroy(ctx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_is_centred_and_varied() {
        let js: Vec<Vec2> = (0..JITTER_SAMPLES).map(jitter).collect();
        let mean = js.iter().copied().sum::<Vec2>() / js.len() as f32;
        assert!(mean.length() < 0.1, "mean {mean}");
        assert!(js.iter().all(|j| j.x.abs() <= 0.5 && j.y.abs() <= 0.5));
        for i in 0..js.len() {
            for k in i + 1..js.len() {
                assert!((js[i] - js[k]).length() > 0.05);
            }
        }
    }
}
