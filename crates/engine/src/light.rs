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
//! The coarse volume (block light only, 4 blocks per cell, 128×64×128 cells) covers the LOD0
//! streaming box: `coarse_fill.comp` keeps each cell's brightest emission and its number of
//! light-blocking blocks, `coarse_propagate.comp` spreads block light 4/15 per cell. Surfaces
//! beyond the fine volume sample it, so distant lava still lights its banks.
//!
//! The passes only run for a few frames after something changed (a fill or a volume moving),
//! so a static scene costs nothing.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::IVec3;
use vkcore::image::ImageDesc;
use vkcore::sync::{compute_barrier, image_barrier, memory_barrier};
use vkcore::{Bindless, GpuProfiler, Image, VkContext, vk};

use crate::shaders::{PipelineHandle, PipelineManager, PipelineSource};
use crate::world::light::{COARSE_CELL, COARSE_CHUNKS, LIGHT_N};
use crate::world::CHUNK;

/// Propagation steps per active frame (even, so the result ends in `light[0]`).
const STEPS_PER_FRAME: u32 = 4;
/// Frames to keep propagating after the last change: 6 × 4 steps > the 15 needed to settle.
const SETTLE_FRAMES: u32 = 6;
/// Coarse steps per active frame (light reaches 15 / 4 cells, so 6 × 2 steps settle it).
const COARSE_STEPS_PER_FRAME: u32 = 2;

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
struct CoarsePush {
    origin: [i32; 4],
    cells_img: u32,
    src: u32,
    dst: u32,
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
    /// Below this height the sky above the volume is not open (the top of the underground layers).
    pub sky_floor: i32,
    active_frames: u32,
    coarse_cells: Image,
    coarse_light: [Image; 2],
    /// Bindless storage index of the coarse cell image (used by the coarse fill pass).
    pub coarse_cells_img: u32,
    coarse_light_img: [u32; 2],
    /// Bindless sampled index of `coarse_light[0]`.
    pub coarse_sampled: u32,
    pub coarse_fill: PipelineHandle,
    coarse_propagate: PipelineHandle,
    coarse_active_frames: u32,
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
        let cd = (COARSE_CHUNKS * CHUNK / COARSE_CELL).as_uvec3();
        let cdesc = |name, format| ImageDesc {
            name,
            format,
            extent: vk::Extent3D { width: cd.x, height: cd.y, depth: cd.z },
            mips: 1,
            layers: 1,
            usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            view_type: vk::ImageViewType::TYPE_3D,
            per_mip_views: false,
        };
        let coarse_cells = Image::new(ctx, &cdesc("coarse_cells", vk::Format::R8G8B8A8_UINT))?;
        let coarse_light = [
            Image::new(ctx, &cdesc("coarse_light_a", vk::Format::R8G8B8A8_UNORM))?,
            Image::new(ctx, &cdesc("coarse_light_b", vk::Format::R8G8B8A8_UNORM))?,
        ];
        let coarse_cells_img = bindless.add_storage(ctx, coarse_cells.view);
        let coarse_light_img = [bindless.add_storage(ctx, coarse_light[0].view), bindless.add_storage(ctx, coarse_light[1].view)];
        let coarse_sampled = bindless.add_sampled(ctx, coarse_light[0].view, bindless.samplers.linear_repeat, vk::ImageLayout::GENERAL);
        let compute = |path: &str| PipelineSource::Compute { path: path.into(), defines: vec![] };
        Ok(Self {
            sky_floor: i32::MIN / 2,
            blocks,
            light,
            blocks_img,
            light_img,
            sampled,
            fill: pm.create(ctx, "light_fill", compute("light/fill.comp"))?,
            sky: pm.create(ctx, "light_sky", compute("light/sky.comp"))?,
            propagate: pm.create(ctx, "light_propagate", compute("light/propagate.comp"))?,
            active_frames: 0,
            coarse_cells,
            coarse_light,
            coarse_cells_img,
            coarse_light_img,
            coarse_sampled,
            coarse_fill: pm.create(ctx, "coarse_fill", compute("light/coarse_fill.comp"))?,
            coarse_propagate: pm.create(ctx, "coarse_propagate", compute("light/coarse_propagate.comp"))?,
            coarse_active_frames: 0,
            initialized: false,
        })
    }

    /// Something in the fine volume changed: propagate for the next few frames.
    pub fn mark_dirty(&mut self) {
        self.active_frames = SETTLE_FRAMES;
    }

    /// Something in the coarse volume changed.
    pub fn mark_coarse_dirty(&mut self) {
        self.coarse_active_frames = SETTLE_FRAMES;
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
        for img in [&self.blocks, &self.light[0], &self.light[1], &self.coarse_cells, &self.coarse_light[0], &self.coarse_light[1]] {
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
        self.mark_coarse_dirty();
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
        coarse_origin_block: Option<IVec3>,
        block_table: u64,
        profiler: &mut GpuProfiler,
        slot: usize,
    ) {
        let fine = self.active_frames > 0;
        let coarse = self.coarse_active_frames > 0 && coarse_origin_block.is_some();
        if !(fine || coarse) || !self.initialized {
            return;
        }
        let d = &ctx.device;
        let push = |p: &[u8]| unsafe { d.cmd_push_constants(cb, pm.layout, vk::ShaderStageFlags::ALL, 0, p) };
        profiler.begin(ctx, cb, slot, "light");
        compute_barrier(ctx, cb);
        unsafe { d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, pm.layout, 0, &[set], &[]) };
        if let Some(cob) = coarse_origin_block.filter(|_| coarse) {
            self.coarse_active_frames -= 1;
            let o = cob / COARSE_CELL;
            let dim = (COARSE_CHUNKS * CHUNK / COARSE_CELL).as_uvec3();
            unsafe { d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pm.get(self.coarse_propagate)) };
            for step in 0..COARSE_STEPS_PER_FRAME {
                if step > 0 {
                    compute_barrier(ctx, cb);
                }
                let (src, dst) = if step % 2 == 0 { (0, 1) } else { (1, 0) };
                push(bytemuck::bytes_of(&CoarsePush {
                    origin: [o.x, o.y, o.z, 0],
                    cells_img: self.coarse_cells_img,
                    src: self.coarse_light_img[src],
                    dst: self.coarse_light_img[dst],
                    pad: 0,
                }));
                unsafe { d.cmd_dispatch(cb, dim.x / 8, dim.y / 8, dim.z / 8) };
            }
        }
        if !fine {
            memory_barrier(
                ctx,
                cb,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            );
            profiler.end(ctx, cb, slot);
            return;
        }
        self.active_frames -= 1;
        let n = LIGHT_N as u32;
        let origin = [origin_block.x, origin_block.y, origin_block.z, self.sky_floor];
        unsafe {
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
        bindless.free_sampled(self.coarse_sampled);
        bindless.free_storage(self.coarse_cells_img);
        for i in self.coarse_light_img {
            bindless.free_storage(i);
        }
        self.blocks.destroy(ctx);
        self.coarse_cells.destroy(ctx);
        for l in self.light.iter_mut().chain(self.coarse_light.iter_mut()) {
            l.destroy(ctx);
        }
    }
}
