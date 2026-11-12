//! Light volume (GPU side): RGB block light and sky light at block resolution in a toroidal
//! 128³ volume around the camera (`world::light` tracks which chunks it covers).
//!
//! - `blocks` (R16_UINT): block id | sky source level << 12. LOD0 generation jobs copy their final
//!   blocks in (`shaders/light/fill.comp`, recorded inside the generation batch); `sky.comp` walks
//!   every column top-down to mark cells open to the sky.
//! - `light[0..2]` (RGBA8): ping-pong pair for `propagate.comp`, a Jacobi relaxation in which light
//!   loses 1/15 per block. An even number of steps per frame leaves the result in `light[0]`,
//!   which surfaces sample trilinearly (smooth lighting).
//!
//! The passes only run for a few frames after something changed (a fill or the volume moving),
//! so a static scene costs nothing.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::IVec3;
use vkcore::image::ImageDesc;
use vkcore::sync::{compute_barrier, image_barrier, memory_barrier};
use vkcore::{Bindless, GpuProfiler, Image, VkContext, vk};

use crate::shaders::{PipelineHandle, PipelineManager, PipelineSource};
use crate::world::light::LIGHT_N;

/// Propagation steps per active frame (even, so the result ends in `light[0]`).
const STEPS_PER_FRAME: u32 = 4;
/// Frames to keep propagating after the last change: 6 × 4 steps > the 15 needed to settle.
const SETTLE_FRAMES: u32 = 6;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SkyPush {
    blocks: u64,
    origin: [i32; 4],
    blocks_img: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PropagatePush {
    blocks: u64,
    origin: [i32; 4],
    blocks_img: u32,
    src: u32,
    dst: u32,
    pad: u32,
}

pub struct LightGpu {
    blocks: Image,
    light: [Image; 2],
    /// Bindless storage index of the block image (used by the fill pass).
    pub blocks_img: u32,
    light_img: [u32; 2],
    /// Bindless sampled index of `light[0]` (what surfaces sample).
    pub sampled: u32,
    pub fill: PipelineHandle,
    sky: PipelineHandle,
    propagate: PipelineHandle,
    active_frames: u32,
    initialized: bool,
}

impl LightGpu {
    pub fn new(ctx: &VkContext, pm: &mut PipelineManager, bindless: &mut Bindless) -> Result<Self> {
        let n = LIGHT_N as u32;
        let desc = |name, format| ImageDesc {
            name,
            format,
            extent: vk::Extent3D { width: n, height: n, depth: n },
            mips: 1,
            layers: 1,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            view_type: vk::ImageViewType::TYPE_3D,
            per_mip_views: false,
        };
        let blocks = Image::new(ctx, &desc("light_blocks", vk::Format::R16_UINT))?;
        let light = [
            Image::new(ctx, &desc("light_a", vk::Format::R8G8B8A8_UNORM))?,
            Image::new(ctx, &desc("light_b", vk::Format::R8G8B8A8_UNORM))?,
        ];
        let blocks_img = bindless.add_storage(ctx, blocks.view);
        let light_img = [bindless.add_storage(ctx, light[0].view), bindless.add_storage(ctx, light[1].view)];
        let sampled = bindless.add_sampled(ctx, light[0].view, bindless.samplers.linear_repeat, vk::ImageLayout::GENERAL);
        let compute = |path: &str| PipelineSource::Compute { path: path.into(), defines: vec![] };
        Ok(Self {
            blocks,
            light,
            blocks_img,
            light_img,
            sampled,
            fill: pm.create(ctx, "light_fill", compute("light/fill.comp"))?,
            sky: pm.create(ctx, "light_sky", compute("light/sky.comp"))?,
            propagate: pm.create(ctx, "light_propagate", compute("light/propagate.comp"))?,
            active_frames: 0,
            initialized: false,
        })
    }

    /// Something in the volume changed: propagate for the next few frames.
    pub fn mark_dirty(&mut self) {
        self.active_frames = SETTLE_FRAMES;
    }

    /// Whether the light volume holds valid data (surfaces should sample it).
    pub fn ready(&self) -> bool {
        self.initialized
    }

    /// First use: move the images to GENERAL and clear them. Must run before the generation batch
    /// (which fills `blocks`).
    pub fn record_init(&mut self, ctx: &VkContext, cb: vk::CommandBuffer) {
        if self.initialized {
            return;
        }
        let d = &ctx.device;
        for img in [&self.blocks, &self.light[0], &self.light[1]] {
            image_barrier(
                ctx,
                cb,
                img.raw,
                vkcore::sync::color_range(),
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::TRANSFER_WRITE,
            );
            unsafe {
                d.cmd_clear_color_image(
                    cb,
                    img.raw,
                    vk::ImageLayout::GENERAL,
                    &vk::ClearColorValue { uint32: [0; 4] },
                    &[vkcore::sync::color_range()],
                );
            }
        }
        memory_barrier(
            ctx,
            cb,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
        self.initialized = true;
        self.mark_dirty();
    }

    /// Sky sources + propagation steps (after the generation batch, before rendering).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        ctx: &VkContext,
        cb: vk::CommandBuffer,
        pm: &PipelineManager,
        set: vk::DescriptorSet,
        origin_block: IVec3,
        block_table: u64,
        profiler: &mut GpuProfiler,
        slot: usize,
    ) {
        if self.active_frames == 0 || !self.initialized {
            return;
        }
        self.active_frames -= 1;
        let d = &ctx.device;
        let n = LIGHT_N as u32;
        let origin = [origin_block.x, origin_block.y, origin_block.z, 0];
        let push = |p: &[u8]| unsafe { d.cmd_push_constants(cb, pm.layout, vk::ShaderStageFlags::ALL, 0, p) };
        profiler.begin(ctx, cb, slot, "light");
        compute_barrier(ctx, cb);
        unsafe {
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, pm.layout, 0, &[set], &[]);
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pm.get(self.sky));
            push(bytemuck::bytes_of(&SkyPush { blocks: block_table, origin, blocks_img: self.blocks_img, pad: 0 }));
            d.cmd_dispatch(cb, n / 8, n / 8, 1);
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pm.get(self.propagate));
        }
        for step in 0..STEPS_PER_FRAME {
            compute_barrier(ctx, cb);
            let (src, dst) = if step % 2 == 0 { (0, 1) } else { (1, 0) };
            push(bytemuck::bytes_of(&PropagatePush {
                blocks: block_table,
                origin,
                blocks_img: self.blocks_img,
                src: self.light_img[src],
                dst: self.light_img[dst],
                pad: 0,
            }));
            unsafe { d.cmd_dispatch(cb, n / 8, n / 8, n / 8) };
        }
        memory_barrier(
            ctx,
            cb,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
        profiler.end(ctx, cb, slot);
    }

    pub fn destroy(&mut self, ctx: &VkContext, bindless: &mut Bindless) {
        bindless.free_sampled(self.sampled);
        bindless.free_storage(self.blocks_img);
        for i in self.light_img {
            bindless.free_storage(i);
        }
        self.blocks.destroy(ctx);
        for l in &mut self.light {
            l.destroy(ctx);
        }
    }
}
