//! Bloom: a half-resolution mip chain (RGBA16F). The first downsample applies a soft threshold;
//! each upsample adds a tent-filtered smaller mip onto the next larger one. The tonemap pass adds
//! mip 0 to the scene (`shaders/render/bloom_down.comp`, `bloom_up.comp`).

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use vkcore::image::ImageDesc;
use vkcore::sync::{compute_barrier, image_barrier, memory_barrier};
use vkcore::{Bindless, GpuProfiler, Image, VkContext, vk};

use crate::renderer::HDR_FORMAT;
use crate::shaders::{PipelineHandle, PipelineManager, PipelineSource};

const MAX_MIPS: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DownPush {
    src: u32,
    dst: u32,
    dst_size: [i32; 2],
    src_texel: [f32; 2],
    first: u32,
    threshold: f32,
    exposure: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UpPush {
    src: u32,
    dst: u32,
    dst_size: [i32; 2],
    src_texel: [f32; 2],
}

pub struct Bloom {
    chain: Image,
    sizes: Vec<vk::Extent2D>,
    storage: Vec<u32>,
    sampled: Vec<u32>,
    down: PipelineHandle,
    up: PipelineHandle,
    initialized: bool,
}

fn mip_sizes(extent: vk::Extent2D) -> Vec<vk::Extent2D> {
    let mut sizes = Vec::new();
    let (mut w, mut h) = ((extent.width / 2).max(1), (extent.height / 2).max(1));
    while sizes.len() < MAX_MIPS as usize && w >= 8 && h >= 8 {
        sizes.push(vk::Extent2D { width: w, height: h });
        w = (w / 2).max(1);
        h = (h / 2).max(1);
    }
    if sizes.is_empty() {
        sizes.push(vk::Extent2D { width: w, height: h });
    }
    sizes
}

fn create_chain(ctx: &VkContext, sizes: &[vk::Extent2D]) -> Result<Image> {
    Image::new(
        ctx,
        &ImageDesc {
            name: "bloom_chain",
            format: HDR_FORMAT,
            extent: vk::Extent3D { width: sizes[0].width, height: sizes[0].height, depth: 1 },
            mips: sizes.len() as u32,
            layers: 1,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            view_type: vk::ImageViewType::TYPE_2D,
            per_mip_views: true,
        },
    )
}

impl Bloom {
    pub fn new(ctx: &VkContext, pm: &mut PipelineManager, bindless: &mut Bindless, extent: vk::Extent2D) -> Result<Self> {
        let compute = |path: &str| PipelineSource::Compute { path: path.into(), defines: vec![] };
        let mut b = Self {
            chain: create_chain(ctx, &mip_sizes(extent))?,
            sizes: Vec::new(),
            storage: Vec::new(),
            sampled: Vec::new(),
            down: pm.create(ctx, "bloom_down", compute("render/bloom_down.comp"))?,
            up: pm.create(ctx, "bloom_up", compute("render/bloom_up.comp"))?,
            initialized: false,
        };
        b.sizes = mip_sizes(extent);
        b.bind(ctx, bindless);
        Ok(b)
    }

    fn bind(&mut self, ctx: &VkContext, bindless: &mut Bindless) {
        for i in self.storage.drain(..) {
            bindless.free_storage(i);
        }
        for i in self.sampled.drain(..) {
            bindless.free_sampled(i);
        }
        for &v in &self.chain.mip_views {
            self.storage.push(bindless.add_storage(ctx, v));
            self.sampled.push(bindless.add_sampled(ctx, v, bindless.samplers.linear_clamp, vk::ImageLayout::GENERAL));
        }
    }

    /// Bindless sampled index of the finished bloom (mip 0), for the tonemap pass.
    pub fn output(&self) -> u32 {
        self.sampled[0]
    }

    pub fn resize(&mut self, ctx: &VkContext, bindless: &mut Bindless, extent: vk::Extent2D) -> Result<()> {
        self.chain.destroy(ctx);
        self.sizes = mip_sizes(extent);
        self.chain = create_chain(ctx, &self.sizes)?;
        self.bind(ctx, bindless);
        self.initialized = false;
        Ok(())
    }

    /// Build the bloom chain from `src` (full-resolution sampled image), after TAA.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        ctx: &VkContext,
        cb: vk::CommandBuffer,
        pm: &PipelineManager,
        set: vk::DescriptorSet,
        src: u32,
        src_extent: vk::Extent2D,
        threshold: f32,
        exposure: f32,
        profiler: &mut GpuProfiler,
        slot: usize,
    ) {
        let d = &ctx.device;
        if !self.initialized {
            image_barrier(
                ctx,
                cb,
                self.chain.raw,
                self.chain.full_range(),
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            );
            self.initialized = true;
        }
        profiler.begin(ctx, cb, slot, "bloom");
        // the previous frame's tonemap may still be reading mip 0
        memory_barrier(
            ctx,
            cb,
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
        let push = |p: &[u8]| unsafe { d.cmd_push_constants(cb, pm.layout, vk::ShaderStageFlags::ALL, 0, p) };
        let texel = |e: vk::Extent2D| [1.0 / e.width as f32, 1.0 / e.height as f32];
        let groups = |e: vk::Extent2D| (e.width.div_ceil(8), e.height.div_ceil(8));
        unsafe {
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, pm.layout, 0, &[set], &[]);
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pm.get(self.down));
        }
        for (i, &size) in self.sizes.iter().enumerate() {
            let (src_idx, src_size) = if i == 0 { (src, src_extent) } else { (self.sampled[i - 1], self.sizes[i - 1]) };
            push(bytemuck::bytes_of(&DownPush {
                src: src_idx,
                dst: self.storage[i],
                dst_size: [size.width as i32, size.height as i32],
                src_texel: texel(src_size),
                first: (i == 0) as u32,
                threshold,
                exposure,
            }));
            let (gx, gy) = groups(size);
            unsafe { d.cmd_dispatch(cb, gx, gy, 1) };
            compute_barrier(ctx, cb);
        }
        unsafe { d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pm.get(self.up)) };
        for i in (1..self.sizes.len()).rev() {
            let size = self.sizes[i - 1];
            push(bytemuck::bytes_of(&UpPush {
                src: self.sampled[i],
                dst: self.storage[i - 1],
                dst_size: [size.width as i32, size.height as i32],
                src_texel: texel(self.sizes[i]),
            }));
            let (gx, gy) = groups(size);
            unsafe { d.cmd_dispatch(cb, gx, gy, 1) };
            compute_barrier(ctx, cb);
        }
        memory_barrier(
            ctx,
            cb,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ,
        );
        profiler.end(ctx, cb, slot);
    }

    pub fn destroy(&mut self, ctx: &VkContext, bindless: &mut Bindless) {
        for i in self.storage.drain(..) {
            bindless.free_storage(i);
        }
        for i in self.sampled.drain(..) {
            bindless.free_sampled(i);
        }
        self.chain.destroy(ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_chain_halves_and_stops_small() {
        let s = mip_sizes(vk::Extent2D { width: 1920, height: 1080 });
        assert_eq!(s.len(), MAX_MIPS as usize);
        assert_eq!((s[0].width, s[0].height), (960, 540));
        assert_eq!((s[5].width, s[5].height), (30, 16));
        let tiny = mip_sizes(vk::Extent2D { width: 40, height: 30 });
        assert!(!tiny.is_empty() && tiny.iter().all(|e| e.width >= 8 && e.height >= 8));
    }
}
