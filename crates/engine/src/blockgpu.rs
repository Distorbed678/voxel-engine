//! GPU side of the block registry:
//!
//! - the block table, one copy per frame in flight, so the block editor can change it every
//!   frame (dragging a slider) without waiting for the GPU;
//! - the block texture arrays (sRGB albedo, normal + height), one layer per texture frame, with
//!   spare capacity so new textures rarely need new images;
//! - a UI atlas with the same textures (mip 0, UNORM so ImGui shows the sRGB bytes as they are)
//!   for thumbnails and block previews in the editors.
//!
//! Layer uploads from the editors are recorded at the start of a frame's command buffer from a
//! per-slot staging buffer. Only growing the arrays waits for the GPU.

use std::collections::VecDeque;

use anyhow::Result;
use vkcore::sync::image_barrier;
use vkcore::{Bindless, Buffer, Image, VkContext, vk};

use crate::blocks::{BlockRegistry, GpuBlockInfo, MAX_BLOCKS};
use crate::textures::{self, TEX, TexImage};

/// Layers are allocated in steps of this many.
const CAPACITY_STEP: u32 = 128;
/// Atlas tiles per row.
pub const ATLAS_COLS: u32 = 32;
/// Staging bytes per frame slot for layer uploads (about 270 layers).
const STAGING_BYTES: u64 = 4 << 20;

/// The bytes one layer uploads: albedo and normal mip chains plus the atlas tile.
struct LayerData {
    layer: u16,
    albedo: Vec<Vec<u8>>,
    normal: Vec<Vec<u8>>,
    atlas: Vec<u8>,
}

impl LayerData {
    fn new(layer: u16, t: &TexImage) -> Self {
        let a = t.albedo_rgba8();
        Self { layer, albedo: textures::mip_chain(&a, true), normal: textures::mip_chain(&t.normal_rgba8(), false), atlas: a }
    }

    fn bytes(&self) -> u64 {
        (self.albedo.iter().map(Vec::len).sum::<usize>() + self.normal.iter().map(Vec::len).sum::<usize>() + self.atlas.len()) as u64
    }
}

/// The UI atlas registered with ImGui: tile `layer` shows the first mip of that array layer.
#[derive(Clone, Copy)]
pub struct AtlasRef {
    pub id: dear_imgui_rs::TextureId,
    pub rows: u32,
}

impl AtlasRef {
    /// UV rectangle of a layer.
    pub fn uv(&self, layer: u16) -> ([f32; 2], [f32; 2]) {
        let (c, r) = ((layer as u32 % ATLAS_COLS) as f32, (layer as u32 / ATLAS_COLS) as f32);
        let (w, h) = (1.0 / ATLAS_COLS as f32, 1.0 / self.rows.max(1) as f32);
        ([c * w, r * h], [(c + 1.0) * w, (r + 1.0) * h])
    }
}

pub struct BlockGpu {
    tables: Vec<Buffer>,
    table: Vec<GpuBlockInfo>,
    /// slots whose table copy is out of date (bit per slot)
    table_dirty: u32,
    pub albedo: Image,
    pub normal: Image,
    pub atlas: Image,
    /// bindless sampled indices of the arrays
    pub albedo_tex: u32,
    pub normal_tex: u32,
    /// layers the arrays hold
    pub capacity: u32,
    staging: Vec<Buffer>,
    pending: VecDeque<LayerData>,
    /// The atlas image changed (grown): the ImGui texture must point at the new view.
    pub atlas_replaced: bool,
}

fn round_capacity(layers: u32) -> u32 {
    layers.div_ceil(CAPACITY_STEP).max(1) * CAPACITY_STEP
}

fn create_images(ctx: &VkContext, capacity: u32) -> Result<(Image, Image, Image)> {
    let mips = (TEX as u32).ilog2() + 1;
    let array = |name, format| {
        Image::new(
            ctx,
            &vkcore::image::ImageDesc {
                name,
                format,
                extent: vk::Extent3D { width: TEX as u32, height: TEX as u32, depth: 1 },
                mips,
                layers: capacity,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                view_type: vk::ImageViewType::TYPE_2D_ARRAY,
                per_mip_views: false,
            },
        )
    };
    let albedo = array("block_albedo", vk::Format::R8G8B8A8_SRGB)?;
    let normal = array("block_normal", vk::Format::R8G8B8A8_UNORM)?;
    let rows = capacity.div_ceil(ATLAS_COLS);
    let atlas = Image::new(
        ctx,
        &vkcore::image::ImageDesc::d2(
            "block_atlas_ui",
            vk::Format::R8G8B8A8_UNORM,
            ATLAS_COLS * TEX as u32,
            rows * TEX as u32,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        ),
    )?;
    Ok((albedo, normal, atlas))
}

impl BlockGpu {
    pub fn new(ctx: &VkContext, bindless: &mut Bindless, reg: &mut BlockRegistry, fif: usize) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let capacity = round_capacity(reg.layer_count());
        let (albedo, normal, atlas) = create_images(ctx, capacity)?;
        let albedo_tex = bindless.add_sampled(ctx, albedo.view, bindless.samplers.pixel_repeat, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let normal_tex = bindless.add_sampled(ctx, normal.view, bindless.samplers.pixel_repeat, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let size = (MAX_BLOCKS * std::mem::size_of::<GpuBlockInfo>()) as u64;
        let tables = (0..fif).map(|i| Buffer::upload(ctx, &format!("block_table_{i}"), size)).collect::<Result<Vec<_>>>()?;
        let staging = (0..fif).map(|i| Buffer::upload(ctx, &format!("block_tex_staging_{i}"), STAGING_BYTES)).collect::<Result<Vec<_>>>()?;
        let mut g = Self {
            tables,
            table: reg.gpu_table(),
            table_dirty: (1 << fif) - 1,
            albedo,
            normal,
            atlas,
            albedo_tex,
            normal_tex,
            capacity,
            staging,
            pending: VecDeque::new(),
            atlas_replaced: false,
        };
        g.upload_all_now(ctx, reg, true)?;
        for slot in 0..fif {
            g.write_table(slot);
        }
        log::info!("block textures: {} layers ({} capacity) in {:.1} ms", reg.layer_count(), capacity, t0.elapsed().as_secs_f32() * 1000.0);
        Ok(g)
    }

    fn write_table(&mut self, slot: usize) {
        self.tables[slot].write(0, &self.table);
        self.table_dirty &= !(1 << slot);
    }

    /// The table this frame slot's commands read.
    pub fn table_address(&self, slot: usize) -> u64 {
        self.tables[slot].address
    }

    pub fn set_table(&mut self, table: Vec<GpuBlockInfo>) {
        self.table = table;
        self.table_dirty = (1 << self.tables.len()) - 1;
    }

    /// Bring this slot's table up to date (call after the slot's previous frame completed).
    pub fn begin_frame(&mut self, slot: usize) {
        if self.table_dirty & (1 << slot) != 0 {
            self.write_table(slot);
        }
    }

    /// Upload these layers over the next frames.
    pub fn queue_layers(&mut self, layers: Vec<(u16, TexImage)>) {
        for (l, t) in layers {
            self.pending.retain(|p| p.layer != l);
            self.pending.push_back(LayerData::new(l, &t));
        }
    }

    /// The layout changed: every layer is uploaded again; the arrays grow (waiting for the GPU)
    /// if they are too small.
    pub fn relayout(&mut self, ctx: &VkContext, bindless: &mut Bindless, reg: &mut BlockRegistry) -> Result<()> {
        self.pending.clear();
        let need = reg.layer_count();
        if need > self.capacity {
            ctx.wait_idle();
            let capacity = round_capacity(need + need / 4);
            let (albedo, normal, atlas) = create_images(ctx, capacity)?;
            let mut old = [std::mem::replace(&mut self.albedo, albedo), std::mem::replace(&mut self.normal, normal), std::mem::replace(&mut self.atlas, atlas)];
            for o in &mut old {
                o.destroy(ctx);
            }
            self.capacity = capacity;
            bindless.update_sampled(ctx, self.albedo_tex, self.albedo.view, bindless.samplers.pixel_repeat, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            bindless.update_sampled(ctx, self.normal_tex, self.normal.view, bindless.samplers.pixel_repeat, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            self.atlas_replaced = true;
            log::info!("block texture arrays grown to {capacity} layers");
            return self.upload_all_now(ctx, reg, true);
        }
        self.queue_layers(reg.layer_images());
        Ok(())
    }

    /// Upload every layer with a one-off command buffer (startup, growing).
    fn upload_all_now(&mut self, ctx: &VkContext, reg: &mut BlockRegistry, fresh: bool) -> Result<()> {
        let items: Vec<LayerData> = reg.layer_images().iter().map(|(l, t)| LayerData::new(*l, t)).collect();
        let total: u64 = items.iter().map(LayerData::bytes).sum::<u64>().max(16);
        let mut staging = Buffer::upload(ctx, "block_tex_upload", total)?;
        let images = [&self.albedo, &self.normal, &self.atlas];
        let r = ctx.immediate(|cb| record_copies(ctx, cb, images, &mut staging, &items, fresh));
        staging.destroy(ctx);
        r
    }

    /// Record queued layer uploads into this frame's command buffer (as many as fit).
    pub fn record_uploads(&mut self, ctx: &VkContext, cb: vk::CommandBuffer, slot: usize) {
        if self.pending.is_empty() {
            return;
        }
        let mut items = Vec::new();
        let mut bytes = 0;
        while let Some(p) = self.pending.front() {
            if bytes + p.bytes() > STAGING_BYTES || p.layer as u32 >= self.capacity {
                if p.layer as u32 >= self.capacity {
                    self.pending.pop_front();
                    continue;
                }
                break;
            }
            bytes += p.bytes();
            items.push(self.pending.pop_front().unwrap());
        }
        record_copies(ctx, cb, [&self.albedo, &self.normal, &self.atlas], &mut self.staging[slot], &items, false);
    }

    /// The UI atlas as ImGui sees it (`id`: its registered texture).
    pub fn atlas_ref(&self, id: dear_imgui_rs::TextureId) -> AtlasRef {
        AtlasRef { id, rows: self.capacity.div_ceil(ATLAS_COLS) }
    }

    pub fn uploads_pending(&self) -> usize {
        self.pending.len()
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        for b in self.tables.iter_mut().chain(self.staging.iter_mut()) {
            b.destroy(ctx);
        }
        self.albedo.destroy(ctx);
        self.normal.destroy(ctx);
        self.atlas.destroy(ctx);
    }
}

/// Copy layers from `staging` into the arrays and the atlas (`images`: albedo, normal, atlas).
/// `fresh`: the images have no contents yet (UNDEFINED layout).
fn record_copies(ctx: &VkContext, cb: vk::CommandBuffer, images: [&Image; 3], staging: &mut Buffer, items: &[LayerData], fresh: bool) {
    let mut albedo_regions = Vec::new();
    let mut normal_regions = Vec::new();
    let mut atlas_regions = Vec::new();
    let mut offset = 0u64;
    for it in items {
        for (mips, regions) in [(&it.albedo, &mut albedo_regions), (&it.normal, &mut normal_regions)] {
            for (m, data) in mips.iter().enumerate() {
                staging.write(offset, data);
                let size = (TEX as u32 >> m).max(1);
                regions.push(
                    vk::BufferImageCopy::default()
                        .buffer_offset(offset)
                        .image_subresource(vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: m as u32,
                            base_array_layer: it.layer as u32,
                            layer_count: 1,
                        })
                        .image_extent(vk::Extent3D { width: size, height: size, depth: 1 }),
                );
                offset += data.len() as u64;
            }
        }
        staging.write(offset, &it.atlas);
        let (tx, ty) = ((it.layer as u32 % ATLAS_COLS) * TEX as u32, (it.layer as u32 / ATLAS_COLS) * TEX as u32);
        atlas_regions.push(
            vk::BufferImageCopy::default()
                .buffer_offset(offset)
                .image_subresource(vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 })
                .image_offset(vk::Offset3D { x: tx as i32, y: ty as i32, z: 0 })
                .image_extent(vk::Extent3D { width: TEX as u32, height: TEX as u32, depth: 1 }),
        );
        offset += it.atlas.len() as u64;
    }
    let (old, src_stage) = if fresh {
        (vk::ImageLayout::UNDEFINED, vk::PipelineStageFlags2::NONE)
    } else {
        (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::PipelineStageFlags2::ALL_COMMANDS)
    };
    for (img, regions) in [(images[0], &albedo_regions), (images[1], &normal_regions), (images[2], &atlas_regions)] {
        if regions.is_empty() && !fresh {
            continue;
        }
        let range = img.full_range();
        image_barrier(ctx, cb, img.raw, range, old, vk::ImageLayout::TRANSFER_DST_OPTIMAL, src_stage, vk::AccessFlags2::NONE, vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE);
        if !regions.is_empty() {
            unsafe { ctx.device.cmd_copy_buffer_to_image(cb, staging.raw, img.raw, vk::ImageLayout::TRANSFER_DST_OPTIMAL, regions) };
        }
        image_barrier(
            ctx,
            cb,
            img.raw,
            range,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::SHADER_READ,
        );
    }
}
