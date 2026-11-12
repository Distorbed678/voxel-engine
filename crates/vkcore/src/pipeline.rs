use anyhow::Result;
use ash::vk;

use crate::VkContext;

pub const PUSH_CONSTANT_SIZE: u32 = 256;

/// The single pipeline layout shared by every pipeline in the engine.
pub fn create_global_layout(
    ctx: &VkContext,
    set_layouts: &[vk::DescriptorSetLayout],
) -> Result<vk::PipelineLayout> {
    let ranges = [vk::PushConstantRange {
        stage_flags: vk::ShaderStageFlags::ALL,
        offset: 0,
        size: PUSH_CONSTANT_SIZE,
    }];
    Ok(unsafe {
        ctx.device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(set_layouts)
                .push_constant_ranges(&ranges),
            None,
        )?
    })
}

pub fn shader_module(ctx: &VkContext, spirv: &[u32]) -> Result<vk::ShaderModule> {
    Ok(unsafe {
        ctx.device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)?
    })
}

pub fn compute(
    ctx: &VkContext,
    layout: vk::PipelineLayout,
    spirv: &[u32],
    name: &str,
) -> Result<vk::Pipeline> {
    let module = shader_module(ctx, spirv)?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"main");
    let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
    let result = unsafe {
        ctx.device
            .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
    };
    unsafe { ctx.device.destroy_shader_module(module, None) };
    let pipeline = result.map_err(|(_, e)| e)?[0];
    ctx.set_name(pipeline, name);
    Ok(pipeline)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    None,
    Alpha,
    Premultiplied,
    Additive,
}

pub struct GraphicsDesc<'a> {
    pub stages: Vec<(vk::ShaderStageFlags, &'a [u32])>,
    pub color_formats: Vec<vk::Format>,
    pub blend: Vec<BlendMode>,
    pub depth_format: Option<vk::Format>,
    pub depth_test: bool,
    pub depth_write: bool,
    pub depth_compare: vk::CompareOp,
    pub cull: vk::CullModeFlags,
    pub front_face: vk::FrontFace,
    pub polygon_mode: vk::PolygonMode,
    pub depth_bias: bool,
    pub depth_clamp: bool,
    pub topology: vk::PrimitiveTopology,
}

impl<'a> Default for GraphicsDesc<'a> {
    fn default() -> Self {
        Self {
            stages: Vec::new(),
            color_formats: Vec::new(),
            blend: Vec::new(),
            depth_format: None,
            depth_test: false,
            depth_write: false,
            // Reversed-Z everywhere.
            depth_compare: vk::CompareOp::GREATER_OR_EQUAL,
            cull: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            polygon_mode: vk::PolygonMode::FILL,
            depth_bias: false,
            depth_clamp: false,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        }
    }
}

pub fn graphics(
    ctx: &VkContext,
    layout: vk::PipelineLayout,
    desc: &GraphicsDesc,
    name: &str,
) -> Result<vk::Pipeline> {
    let mut modules = Vec::new();
    let mut stages = Vec::new();
    let mut is_mesh = false;
    for (stage, code) in &desc.stages {
        let m = shader_module(ctx, code)?;
        modules.push(m);
        if stage.contains(vk::ShaderStageFlags::MESH_EXT) {
            is_mesh = true;
        }
        stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(*stage)
                .module(m)
                .name(c"main"),
        );
    }

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly =
        vk::PipelineInputAssemblyStateCreateInfo::default().topology(desc.topology);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(desc.polygon_mode)
        .cull_mode(desc.cull)
        .front_face(desc.front_face)
        .line_width(1.0)
        .depth_bias_enable(desc.depth_bias)
        .depth_clamp_enable(desc.depth_clamp);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(desc.depth_test)
        .depth_write_enable(desc.depth_write)
        .depth_compare_op(desc.depth_compare);
    let attachments: Vec<_> = desc
        .color_formats
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let mode = desc.blend.get(i).copied().unwrap_or(BlendMode::None);
            let a = vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA);
            match mode {
                BlendMode::None => a,
                BlendMode::Alpha => a
                    .blend_enable(true)
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                    .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .alpha_blend_op(vk::BlendOp::ADD),
                BlendMode::Premultiplied => a
                    .blend_enable(true)
                    .src_color_blend_factor(vk::BlendFactor::ONE)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                    .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .alpha_blend_op(vk::BlendOp::ADD),
                BlendMode::Additive => a
                    .blend_enable(true)
                    .src_color_blend_factor(vk::BlendFactor::ONE)
                    .dst_color_blend_factor(vk::BlendFactor::ONE)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                    .dst_alpha_blend_factor(vk::BlendFactor::ONE)
                    .alpha_blend_op(vk::BlendOp::ADD),
            }
        })
        .collect();
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
    let mut dyn_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    if desc.depth_bias {
        dyn_states.push(vk::DynamicState::DEPTH_BIAS);
    }
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dyn_states);
    let mut rendering = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&desc.color_formats)
        .depth_attachment_format(desc.depth_format.unwrap_or(vk::Format::UNDEFINED));

    let mut info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .push_next(&mut rendering);
    if !is_mesh {
        info = info
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly);
    }
    let result = unsafe {
        ctx.device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
    };
    for m in modules {
        unsafe { ctx.device.destroy_shader_module(m, None) };
    }
    let pipeline = result.map_err(|(_, e)| e)?[0];
    ctx.set_name(pipeline, name);
    Ok(pipeline)
}
