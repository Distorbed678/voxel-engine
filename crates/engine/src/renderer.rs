//! Render targets, block textures and the per-frame render passes.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use vkcore::image::ImageDesc;
use vkcore::pipeline::BlendMode;
use vkcore::sync::{image_barrier, memory_barrier};
use vkcore::{Bindless, Buffer, GpuProfiler, Image, VkContext, vk};

use crate::blocks::BlockRegistry;
use crate::shaders::{GraphicsState, PipelineHandle, PipelineManager, PipelineSource};
use crate::textures;

pub const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;
pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
pub const MAX_DRAWS: u32 = 1 << 18;
pub const SHADOW_RES: u32 = 2048;
pub const CASCADES: usize = 4;
pub const SHADOW_DRAWS: u32 = 1 << 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct FrameData {
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub prev_view_proj: [[f32; 4]; 4],
    pub frustum: [[f32; 4]; 6],
    pub cam_block: [i32; 4],
    pub cam_frac: [f32; 4],
    pub sun_dir: [f32; 4],
    pub sun_color: [f32; 4],
    pub sky_zenith: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub ground_color: [f32; 4],
    pub fog_color: [f32; 4],
    pub fog_params: [f32; 4],
    pub resolution: [f32; 2],
    pub time: f32,
    pub dt: f32,
    pub frame_index: u32,
    pub debug_mode: u32,
    pub flags: u32,
    pub pick_pixel: u32,
    pub block_tex_albedo: u32,
    pub block_tex_normal: u32,
    pub hdr_tex: u32,
    pub depth_tex: u32,
    pub exposure: [f32; 4],
    pub shadow_vp: [[[f32; 4]; 4]; 4],
    pub shadow_planes: [[f32; 4]; 24],
    pub shadow_splits: [f32; 4],
    pub cam_forward: [f32; 4],
    pub shadow_tex: u32,
    pub shadow_res: u32,
    pub light_tex: u32,
    pub pad: u32,
    /// light volume min corner relative to cam_block, w = enabled
    pub light_origin: [i32; 4],
}

pub const FLAG_SHADOWS: u32 = 2;
pub const FLAG_FOG: u32 = 4;
pub const FLAG_CROSSHAIR: u32 = 8;
pub const FLAG_TAA: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullPush {
    frame: u64,
    records: u64,
    opaque: u64,
    trans: u64,
    models: u64,
    args: u64,
    num_records: u32,
    max_draws: u32,
    shadow: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TerrainPush {
    frame: u64,
    records: u64,
    draws: u64,
    quads: u64,
    blocks: u64,
    pick: u64,
    biomes: u64,
    flags: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FramePush {
    frame: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CloudPush {
    frame: u64,
    coverage: f32,
    base: f32,
    thickness: f32,
    density: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct CloudSettings {
    pub enabled: bool,
    pub coverage: f32,
    pub base: f32,
    pub thickness: f32,
    pub density: f32,
}

impl Default for CloudSettings {
    fn default() -> Self {
        Self { enabled: true, coverage: 0.45, base: 230.0, thickness: 90.0, density: 0.06 }
    }
}

pub struct FramePerSlot {
    pub frame_buf: Buffer,
    pub pick_readback: Buffer,
    pub stats_readback: Buffer,
}

pub struct Pipelines {
    pub cull: PipelineHandle,
    pub terrain: PipelineHandle,
    pub terrain_wire: PipelineHandle,
    pub terrain_trans: PipelineHandle,
    pub models: PipelineHandle,
    pub shadow: PipelineHandle,
    pub sky: PipelineHandle,
    pub clouds: PipelineHandle,
    pub tonemap: PipelineHandle,
    pub apply_edits: PipelineHandle,
    pub mesh_count: PipelineHandle,
    pub mesh_write: PipelineHandle,
    pub prefix: PipelineHandle,
}

pub struct Renderer {
    pub extent: vk::Extent2D,
    pub hdr: Image,
    pub depth: Image,
    pub hdr_tex: u32,
    pub depth_tex: u32,
    pub albedo: Image,
    pub normal: Image,
    pub albedo_tex: u32,
    pub normal_tex: u32,
    pub block_table: Buffer,
    pub per_slot: Vec<FramePerSlot>,
    pub pick: Buffer,
    pub args: Buffer,
    pub opaque_draws: Buffer,
    pub trans_draws: Buffer,
    pub model_draws: Buffer,
    pub shadow_map: Image,
    pub shadow_views: Vec<vk::ImageView>,
    pub shadow_tex: u32,
    pub shadow_draws: Buffer,
    pub shadow_args: Buffer,
    pub pipes: Pipelines,
}

fn create_targets(ctx: &VkContext, extent: vk::Extent2D) -> Result<(Image, Image)> {
    let hdr = Image::new(
        ctx,
        &ImageDesc::d2(
            "hdr",
            HDR_FORMAT,
            extent.width,
            extent.height,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
                | vk::ImageUsageFlags::TRANSFER_SRC,
        ),
    )?;
    let depth = Image::new(
        ctx,
        &ImageDesc::d2(
            "depth",
            DEPTH_FORMAT,
            extent.width,
            extent.height,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        ),
    )?;
    Ok((hdr, depth))
}

fn upload_texture_array(ctx: &VkContext, name: &str, format: vk::Format, layers: &[Vec<Vec<u8>>]) -> Result<Image> {
    let mips = layers[0].len() as u32;
    let n = layers.len() as u32;
    let size = textures::TEX as u32;
    let image = Image::new(
        ctx,
        &ImageDesc {
            name,
            format,
            extent: vk::Extent3D { width: size, height: size, depth: 1 },
            mips,
            layers: n,
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            view_type: vk::ImageViewType::TYPE_2D_ARRAY,
            per_mip_views: false,
        },
    )?;
    let total: usize = layers.iter().map(|l| l.iter().map(|m| m.len()).sum::<usize>()).sum();
    let mut staging = Buffer::upload(ctx, "tex_staging", total as u64)?;
    let mut regions = Vec::new();
    let mut offset = 0u64;
    for mip in 0..mips as usize {
        let mip_size = (size >> mip).max(1);
        let start = offset;
        for layer in layers {
            staging.write(offset, &layer[mip]);
            offset += layer[mip].len() as u64;
        }
        regions.push(
            vk::BufferImageCopy::default()
                .buffer_offset(start)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: mip as u32,
                    base_array_layer: 0,
                    layer_count: n,
                })
                .image_extent(vk::Extent3D { width: mip_size, height: mip_size, depth: 1 }),
        );
    }
    let range = image.full_range();
    ctx.immediate(|cb| {
        image_barrier(
            ctx,
            cb,
            image.raw,
            range,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
        );
        unsafe {
            ctx.device.cmd_copy_buffer_to_image(cb, staging.raw, image.raw, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions);
        }
        image_barrier(
            ctx,
            cb,
            image.raw,
            range,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::SHADER_READ,
        );
    })?;
    staging.destroy(ctx);
    Ok(image)
}

impl Renderer {
    pub fn new(
        ctx: &VkContext,
        bindless: &mut Bindless,
        pm: &mut PipelineManager,
        blocks: &BlockRegistry,
        extent: vk::Extent2D,
        output_format: vk::Format,
        frames_in_flight: usize,
    ) -> Result<Self> {
        let (hdr, depth) = create_targets(ctx, extent)?;
        let hdr_tex = bindless.add_sampled(ctx, hdr.view, bindless.samplers.linear_clamp, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let depth_tex = bindless.add_sampled(ctx, depth.view, bindless.samplers.nearest_clamp, vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL);

        // ---- block textures ----
        let t0 = std::time::Instant::now();
        let mut albedo_layers = Vec::new();
        let mut normal_layers = Vec::new();
        for name in &blocks.textures {
            let t = textures::generate(name);
            albedo_layers.push(textures::mip_chain(&t.albedo_rgba8(), true));
            normal_layers.push(textures::mip_chain(&t.normal_rgba8(), false));
        }
        let albedo = upload_texture_array(ctx, "block_albedo", vk::Format::R8G8B8A8_SRGB, &albedo_layers)?;
        let normal = upload_texture_array(ctx, "block_normal", vk::Format::R8G8B8A8_UNORM, &normal_layers)?;
        log::info!("generated {} block textures in {:.1} ms", blocks.textures.len(), t0.elapsed().as_secs_f32() * 1000.0);
        let albedo_tex = bindless.add_sampled(ctx, albedo.view, bindless.samplers.pixel_repeat, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let normal_tex = bindless.add_sampled(ctx, normal.view, bindless.samplers.pixel_repeat, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let table = blocks.gpu_table();
        let mut block_table = Buffer::upload(ctx, "block_table", std::mem::size_of_val(table.as_slice()) as u64)?;
        block_table.write(0, &table);

        let per_slot = (0..frames_in_flight)
            .map(|i| {
                Ok(FramePerSlot {
                    frame_buf: Buffer::upload(ctx, &format!("frame_data_{i}"), std::mem::size_of::<FrameData>() as u64)?,
                    pick_readback: Buffer::readback(ctx, &format!("pick_readback_{i}"), 16)?,
                    stats_readback: Buffer::readback(ctx, &format!("stats_readback_{i}"), 64)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let pipes = Self::create_pipelines(pm, ctx, output_format)?;
        let shadow_map = Image::new(
            ctx,
            &ImageDesc {
                name: "shadow_cascades",
                format: DEPTH_FORMAT,
                extent: vk::Extent3D { width: SHADOW_RES, height: SHADOW_RES, depth: 1 },
                mips: 1,
                layers: CASCADES as u32,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                view_type: vk::ImageViewType::TYPE_2D_ARRAY,
                per_mip_views: false,
            },
        )?;
        let shadow_views = (0..CASCADES as u32).map(|l| shadow_map.layer_view(ctx, l)).collect::<Result<Vec<_>>>()?;
        let shadow_tex = bindless.add_sampled(ctx, shadow_map.view, bindless.samplers.shadow, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        // the shadow map must be in a readable layout even before the first shadow pass
        let range = shadow_map.full_range();
        ctx.immediate(|cb| {
            image_barrier(
                ctx,
                cb,
                shadow_map.raw,
                range,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::PipelineStageFlags2::NONE,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::SHADER_READ,
            );
        })?;

        Ok(Self {
            extent,
            hdr,
            depth,
            hdr_tex,
            depth_tex,
            albedo,
            normal,
            albedo_tex,
            normal_tex,
            block_table,
            per_slot,
            pick: Buffer::gpu(ctx, "pick", 16)?,
            args: Buffer::gpu(ctx, "draw_args", 64)?,
            opaque_draws: Buffer::gpu(ctx, "opaque_draws", MAX_DRAWS as u64 * 16)?,
            trans_draws: Buffer::gpu(ctx, "trans_draws", MAX_DRAWS as u64 * 16)?,
            model_draws: Buffer::gpu(ctx, "model_draws", MAX_DRAWS as u64 * 16)?,
            shadow_map,
            shadow_views,
            shadow_tex,
            shadow_draws: Buffer::gpu(ctx, "shadow_draws", SHADOW_DRAWS as u64 * 16 * CASCADES as u64)?,
            shadow_args: Buffer::gpu(ctx, "shadow_args", 64 * CASCADES as u64)?,
            pipes,
        })
    }

    fn create_pipelines(pm: &mut PipelineManager, ctx: &VkContext, output_format: vk::Format) -> Result<Pipelines> {
        let compute = |path: &str, defines: &[(&str, &str)]| PipelineSource::Compute {
            path: path.into(),
            defines: defines.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect(),
        };
        let terrain_stages = |defines: Vec<(String, String)>| {
            vec![
                ("render/terrain.task".to_string(), defines.clone()),
                ("render/terrain.mesh".to_string(), defines.clone()),
                ("render/terrain.frag".to_string(), defines),
            ]
        };
        let opaque_state = GraphicsState {
            color_formats: vec![HDR_FORMAT],
            blend: vec![BlendMode::None],
            depth_format: Some(DEPTH_FORMAT),
            depth_test: true,
            depth_write: true,
            cull: vk::CullModeFlags::BACK,
            ..Default::default()
        };
        let fullscreen = |frag: &str, state: GraphicsState| PipelineSource::Graphics {
            stages: vec![("render/fullscreen.vert".into(), vec![]), (frag.into(), vec![])],
            state,
        };
        Ok(Pipelines {
            cull: pm.create(ctx, "cull", compute("render/cull.comp", &[]))?,
            terrain: pm.create(ctx, "terrain", PipelineSource::Graphics { stages: terrain_stages(vec![]), state: opaque_state.clone() })?,
            terrain_wire: pm.create(
                ctx,
                "terrain_wire",
                PipelineSource::Graphics {
                    stages: terrain_stages(vec![]),
                    state: GraphicsState { polygon_mode: vk::PolygonMode::LINE, ..opaque_state.clone() },
                },
            )?,
            terrain_trans: pm.create(
                ctx,
                "terrain_trans",
                PipelineSource::Graphics {
                    stages: terrain_stages(vec![("TRANSPARENT".into(), "1".into())]),
                    state: GraphicsState {
                        blend: vec![BlendMode::Alpha],
                        depth_write: false,
                        cull: vk::CullModeFlags::NONE,
                        ..opaque_state.clone()
                    },
                },
            )?,
            models: pm.create(
                ctx,
                "models",
                PipelineSource::Graphics {
                    stages: vec![
                        ("render/model.task".to_string(), vec![]),
                        ("render/model.mesh".to_string(), vec![]),
                        ("render/model.frag".to_string(), vec![]),
                    ],
                    state: GraphicsState { cull: vk::CullModeFlags::NONE, ..opaque_state.clone() },
                },
            )?,
            shadow: pm.create(
                ctx,
                "shadow",
                PipelineSource::Graphics {
                    stages: vec![
                        ("render/terrain.task".to_string(), vec![("SHADOW".into(), "1".into())]),
                        ("render/terrain.mesh".to_string(), vec![("SHADOW".into(), "1".into())]),
                        ("render/terrain.frag".to_string(), vec![("SHADOW".into(), "1".into())]),
                    ],
                    state: GraphicsState {
                        color_formats: vec![],
                        blend: vec![],
                        depth_format: Some(DEPTH_FORMAT),
                        depth_test: true,
                        depth_write: true,
                        cull: vk::CullModeFlags::NONE,
                        depth_bias: true,
                        depth_clamp: true,
                        ..Default::default()
                    },
                },
            )?,
            sky: pm.create(
                ctx,
                "sky",
                fullscreen(
                    "render/sky.frag",
                    GraphicsState {
                        color_formats: vec![HDR_FORMAT],
                        blend: vec![BlendMode::None],
                        depth_format: Some(DEPTH_FORMAT),
                        depth_test: true,
                        depth_write: false,
                        ..Default::default()
                    },
                ),
            )?,
            clouds: pm.create(
                ctx,
                "clouds",
                fullscreen(
                    "render/clouds.frag",
                    GraphicsState {
                        color_formats: vec![HDR_FORMAT],
                        blend: vec![BlendMode::Premultiplied],
                        depth_format: Some(DEPTH_FORMAT),
                        depth_test: false,
                        depth_write: false,
                        ..Default::default()
                    },
                ),
            )?,
            tonemap: pm.create(
                ctx,
                "tonemap",
                fullscreen("render/tonemap.frag", GraphicsState { color_formats: vec![output_format], blend: vec![BlendMode::None], ..Default::default() }),
            )?,
            apply_edits: pm.create(ctx, "apply_edits", compute("worldgen/apply_edits.comp", &[]))?,
            mesh_count: pm.create(ctx, "mesh_count", compute("mesh/greedy.comp", &[]))?,
            mesh_write: pm.create(ctx, "mesh_write", compute("mesh/greedy.comp", &[("MESH_WRITE", "1")]))?,
            prefix: pm.create(ctx, "prefix", compute("mesh/prefix.comp", &[]))?,
        })
    }

    pub fn resize(&mut self, ctx: &VkContext, bindless: &mut Bindless, extent: vk::Extent2D) -> Result<()> {
        if extent == self.extent {
            return Ok(());
        }
        ctx.wait_idle();
        self.hdr.destroy(ctx);
        self.depth.destroy(ctx);
        let (hdr, depth) = create_targets(ctx, extent)?;
        self.hdr = hdr;
        self.depth = depth;
        bindless.update_sampled(ctx, self.hdr_tex, self.hdr.view, bindless.samplers.linear_clamp, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        bindless.update_sampled(ctx, self.depth_tex, self.depth.view, bindless.samplers.nearest_clamp, vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL);
        self.extent = extent;
        Ok(())
    }

    /// GPU culling + main scene (terrain, sky, transparent) into the HDR target.
    #[allow(clippy::too_many_arguments)]
    pub fn record_scene(
        &mut self,
        ctx: &VkContext,
        cb: vk::CommandBuffer,
        slot: usize,
        pm: &PipelineManager,
        bindless: &Bindless,
        profiler: &mut GpuProfiler,
        records: &Buffer,
        quads: &Buffer,
        num_records: u32,
        wireframe: bool,
        biomes: u64,
        shadows: bool,
        clouds: CloudSettings,
    ) {
        let d = &ctx.device;
        let frame_addr = self.per_slot[slot].frame_buf.address;
        let layout = pm.layout;
        let push = |p: &[u8]| unsafe { d.cmd_push_constants(cb, layout, vk::ShaderStageFlags::ALL, 0, p) };

        profiler.begin(ctx, cb, slot, "cull");
        unsafe {
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, layout, 0, &[bindless.set], &[]);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::GRAPHICS, layout, 0, &[bindless.set], &[]);
            let args: [u32; 11] = [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1];
            d.cmd_update_buffer(cb, self.args.raw, 0, bytemuck::cast_slice(&args));
            d.cmd_fill_buffer(cb, self.pick.raw, 0, 16, 0);
        }
        memory_barrier(
            ctx,
            cb,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pm.get(self.pipes.cull));
            push(bytemuck::bytes_of(&CullPush {
                frame: frame_addr,
                records: records.address,
                opaque: self.opaque_draws.address,
                trans: self.trans_draws.address,
                models: self.model_draws.address,
                args: self.args.address,
                num_records,
                max_draws: MAX_DRAWS,
                shadow: 0,
                pad: 0,
            }));
            d.cmd_dispatch(cb, num_records.div_ceil(64).max(1), 1, 1);
        }
        memory_barrier(
            ctx,
            cb,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::DRAW_INDIRECT
                | vk::PipelineStageFlags2::TASK_SHADER_EXT
                | vk::PipelineStageFlags2::MESH_SHADER_EXT
                | vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::INDIRECT_COMMAND_READ | vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::TRANSFER_READ,
        );
        profiler.end(ctx, cb, slot);

        if shadows {
            self.record_shadows(ctx, cb, slot, pm, profiler, records, quads, num_records, biomes);
        }

        // ---- main pass ----
        profiler.begin(ctx, cb, slot, "main pass");
        image_barrier(
            ctx,
            cb,
            self.hdr.raw,
            self.hdr.full_range(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::SHADER_READ,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        );
        image_barrier(
            ctx,
            cb,
            self.depth.raw,
            self.depth.full_range(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
        let color_att = [vk::RenderingAttachmentInfo::default()
            .image_view(self.hdr.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } })];
        let depth_att = vk::RenderingAttachmentInfo::default()
            .image_view(self.depth.view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 } });
        let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: self.extent };
        let info = vk::RenderingInfo::default()
            .render_area(area)
            .layer_count(1)
            .color_attachments(&color_att)
            .depth_attachment(&depth_att);
        let terrain_push = |draws: u64| TerrainPush {
            frame: frame_addr,
            records: records.address,
            draws,
            quads: quads.address,
            blocks: self.block_table.address,
            pick: self.pick.address,
            biomes,
            flags: 0,
            pad: 0,
        };
        unsafe {
            d.cmd_begin_rendering(cb, &info);
            d.cmd_set_viewport(
                cb,
                0,
                &[vk::Viewport { x: 0.0, y: 0.0, width: self.extent.width as f32, height: self.extent.height as f32, min_depth: 0.0, max_depth: 1.0 }],
            );
            d.cmd_set_scissor(cb, 0, &[area]);

            let terrain = if wireframe { self.pipes.terrain_wire } else { self.pipes.terrain };
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pm.get(terrain));
            push(bytemuck::bytes_of(&terrain_push(self.opaque_draws.address)));
            ctx.mesh_shader.cmd_draw_mesh_tasks_indirect(cb, self.args.raw, 0, 1, 12);

            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pm.get(self.pipes.models));
            push(bytemuck::bytes_of(&terrain_push(self.model_draws.address)));
            ctx.mesh_shader.cmd_draw_mesh_tasks_indirect(cb, self.args.raw, 32, 1, 12);

            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pm.get(self.pipes.sky));
            push(bytemuck::bytes_of(&FramePush { frame: frame_addr }));
            d.cmd_draw(cb, 3, 1, 0, 0);

            d.cmd_end_rendering(cb);
        }
        // translucent pass: depth becomes read-only so water and clouds can sample it
        image_barrier(
            ctx,
            cb,
            self.depth.raw,
            self.depth.full_range(),
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
        );
        let color_load = [vk::RenderingAttachmentInfo::default()
            .image_view(self.hdr.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let depth_ro = vk::RenderingAttachmentInfo::default()
            .image_view(self.depth.view)
            .image_layout(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::NONE);
        let info2 = vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&color_load).depth_attachment(&depth_ro);
        unsafe {
            d.cmd_begin_rendering(cb, &info2);
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pm.get(self.pipes.terrain_trans));
            push(bytemuck::bytes_of(&terrain_push(self.trans_draws.address)));
            ctx.mesh_shader.cmd_draw_mesh_tasks_indirect(cb, self.args.raw, 12, 1, 12);
            if clouds.enabled {
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pm.get(self.pipes.clouds));
                push(bytemuck::bytes_of(&CloudPush {
                    frame: frame_addr,
                    coverage: clouds.coverage,
                    base: clouds.base,
                    thickness: clouds.thickness,
                    density: clouds.density,
                }));
                d.cmd_draw(cb, 3, 1, 0, 0);
            }
            d.cmd_end_rendering(cb);
        }
        image_barrier(
            ctx,
            cb,
            self.hdr.raw,
            self.hdr.full_range(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ,
        );

        // pick + stats readback
        memory_barrier(
            ctx,
            cb,
            vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::TRANSFER,
            vk::AccessFlags2::TRANSFER_READ,
        );
        unsafe {
            d.cmd_copy_buffer(cb, self.pick.raw, self.per_slot[slot].pick_readback.raw, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: 16 }]);
            d.cmd_copy_buffer(cb, self.args.raw, self.per_slot[slot].stats_readback.raw, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: 44 }]);
        }
        profiler.end(ctx, cb, slot);
    }

    #[allow(clippy::too_many_arguments)]
    fn record_shadows(
        &mut self,
        ctx: &VkContext,
        cb: vk::CommandBuffer,
        slot: usize,
        pm: &PipelineManager,
        profiler: &mut GpuProfiler,
        records: &Buffer,
        quads: &Buffer,
        num_records: u32,
        biomes: u64,
    ) {
        let d = &ctx.device;
        let frame_addr = self.per_slot[slot].frame_buf.address;
        let layout = pm.layout;
        let push = |p: &[u8]| unsafe { d.cmd_push_constants(cb, layout, vk::ShaderStageFlags::ALL, 0, p) };
        profiler.begin(ctx, cb, slot, "shadows");
        unsafe {
            for c in 0..CASCADES {
                let args: [u32; 11] = [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1];
                d.cmd_update_buffer(cb, self.shadow_args.raw, c as u64 * 64, bytemuck::cast_slice(&args));
            }
        }
        memory_barrier(
            ctx,
            cb,
            vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::TASK_SHADER_EXT,
            vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::INDIRECT_COMMAND_READ | vk::AccessFlags2::SHADER_READ,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        );
        unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pm.get(self.pipes.cull));
            for c in 0..CASCADES {
                let draws = self.shadow_draws.address + (c as u64) * SHADOW_DRAWS as u64 * 16;
                push(bytemuck::bytes_of(&CullPush {
                    frame: frame_addr,
                    records: records.address,
                    opaque: draws,
                    trans: draws,
                    models: draws,
                    args: self.shadow_args.address + c as u64 * 64,
                    num_records,
                    max_draws: SHADOW_DRAWS,
                    shadow: c as u32 + 1,
                    pad: 0,
                }));
                d.cmd_dispatch(cb, num_records.div_ceil(64).max(1), 1, 1);
            }
        }
        memory_barrier(
            ctx,
            cb,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::TASK_SHADER_EXT | vk::PipelineStageFlags2::MESH_SHADER_EXT,
            vk::AccessFlags2::INDIRECT_COMMAND_READ | vk::AccessFlags2::SHADER_READ,
        );
        image_barrier(
            ctx,
            cb,
            self.shadow_map.raw,
            self.shadow_map.full_range(),
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ,
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );
        let area = vk::Rect2D { offset: vk::Offset2D::default(), extent: vk::Extent2D { width: SHADOW_RES, height: SHADOW_RES } };
        for c in 0..CASCADES {
            let depth_att = vk::RenderingAttachmentInfo::default()
                .image_view(self.shadow_views[c])
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 } });
            let info = vk::RenderingInfo::default().render_area(area).layer_count(1).depth_attachment(&depth_att);
            unsafe {
                d.cmd_begin_rendering(cb, &info);
                d.cmd_set_viewport(cb, 0, &[vk::Viewport { x: 0.0, y: 0.0, width: SHADOW_RES as f32, height: SHADOW_RES as f32, min_depth: 0.0, max_depth: 1.0 }]);
                d.cmd_set_scissor(cb, 0, &[area]);
                d.cmd_set_depth_bias(cb, -1.5, 0.0, -2.0);
                d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pm.get(self.pipes.shadow));
                push(bytemuck::bytes_of(&TerrainPush {
                    frame: frame_addr,
                    records: records.address,
                    draws: self.shadow_draws.address + (c as u64) * SHADOW_DRAWS as u64 * 16,
                    quads: quads.address,
                    blocks: self.block_table.address,
                    pick: self.pick.address,
                    biomes,
                    flags: c as u32,
                    pad: 0,
                }));
                ctx.mesh_shader.cmd_draw_mesh_tasks_indirect(cb, self.shadow_args.raw, c as u64 * 64, 1, 12);
                d.cmd_end_rendering(cb);
            }
        }
        image_barrier(
            ctx,
            cb,
            self.shadow_map.raw,
            self.shadow_map.full_range(),
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::SHADER_READ,
        );
        profiler.end(ctx, cb, slot);
    }

    /// Tonemap the HDR image into `target_view` (which must be in COLOR_ATTACHMENT_OPTIMAL),
    /// then let `overlay` draw on top (ImGui) inside the same rendering scope.
    #[allow(clippy::too_many_arguments)]
    pub fn record_output(
        &self,
        ctx: &VkContext,
        cb: vk::CommandBuffer,
        slot: usize,
        pm: &PipelineManager,
        target_view: vk::ImageView,
        extent: vk::Extent2D,
        overlay: impl FnOnce(vk::CommandBuffer),
    ) {
        let d = &ctx.device;
        let att = [vk::RenderingAttachmentInfo::default()
            .image_view(target_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let area = vk::Rect2D { offset: vk::Offset2D::default(), extent };
        let info = vk::RenderingInfo::default().render_area(area).layer_count(1).color_attachments(&att);
        unsafe {
            d.cmd_begin_rendering(cb, &info);
            d.cmd_set_viewport(
                cb,
                0,
                &[vk::Viewport { x: 0.0, y: 0.0, width: extent.width as f32, height: extent.height as f32, min_depth: 0.0, max_depth: 1.0 }],
            );
            d.cmd_set_scissor(cb, 0, &[area]);
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pm.get(self.pipes.tonemap));
            d.cmd_push_constants(
                cb,
                pm.layout,
                vk::ShaderStageFlags::ALL,
                0,
                bytemuck::bytes_of(&FramePush { frame: self.per_slot[slot].frame_buf.address }),
            );
            d.cmd_draw(cb, 3, 1, 0, 0);
        }
        overlay(cb);
        unsafe { d.cmd_end_rendering(cb) };
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        self.hdr.destroy(ctx);
        self.depth.destroy(ctx);
        self.albedo.destroy(ctx);
        self.normal.destroy(ctx);
        self.block_table.destroy(ctx);
        for s in &mut self.per_slot {
            s.frame_buf.destroy(ctx);
            s.pick_readback.destroy(ctx);
            s.stats_readback.destroy(ctx);
        }
        self.pick.destroy(ctx);
        self.args.destroy(ctx);
        self.opaque_draws.destroy(ctx);
        self.trans_draws.destroy(ctx);
        self.model_draws.destroy(ctx);
        for v in self.shadow_views.drain(..) {
            unsafe { ctx.device.destroy_image_view(v, None) };
        }
        self.shadow_map.destroy(ctx);
        self.shadow_draws.destroy(ctx);
        self.shadow_args.destroy(ctx);
    }
}
