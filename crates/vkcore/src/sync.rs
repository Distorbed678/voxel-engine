use anyhow::Result;
use ash::vk;

use crate::VkContext;

/// Global memory barrier (buffers). Cheap and sufficient for BDA-based buffer access.
pub fn memory_barrier(
    ctx: &VkContext,
    cb: vk::CommandBuffer,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) {
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)];
    unsafe {
        ctx.device
            .cmd_pipeline_barrier2(cb, &vk::DependencyInfo::default().memory_barriers(&barrier));
    }
}

/// compute -> compute RAW/WAW barrier.
pub fn compute_barrier(ctx: &VkContext, cb: vk::CommandBuffer) {
    memory_barrier(
        ctx,
        cb,
        vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::TRANSFER_WRITE,
        vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::SHADER_READ
            | vk::AccessFlags2::SHADER_WRITE
            | vk::AccessFlags2::TRANSFER_READ
            | vk::AccessFlags2::TRANSFER_WRITE,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn image_barrier(
    ctx: &VkContext,
    cb: vk::CommandBuffer,
    image: vk::Image,
    range: vk::ImageSubresourceRange,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) {
    let barrier = [vk::ImageMemoryBarrier2::default()
        .image(image)
        .subresource_range(range)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)];
    unsafe {
        ctx.device.cmd_pipeline_barrier2(
            cb,
            &vk::DependencyInfo::default().image_memory_barriers(&barrier),
        );
    }
}

pub fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Timeline semaphore used to track frame completion.
pub struct Timeline {
    pub semaphore: vk::Semaphore,
}

impl Timeline {
    pub fn new(ctx: &VkContext) -> Result<Self> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let semaphore = unsafe {
            ctx.device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut type_info),
                None,
            )?
        };
        Ok(Self { semaphore })
    }

    pub fn value(&self, ctx: &VkContext) -> u64 {
        unsafe { ctx.device.get_semaphore_counter_value(self.semaphore).unwrap_or(0) }
    }

    pub fn wait(&self, ctx: &VkContext, value: u64) -> Result<()> {
        let sems = [self.semaphore];
        let values = [value];
        unsafe {
            ctx.device.wait_semaphores(
                &vk::SemaphoreWaitInfo::default().semaphores(&sems).values(&values),
                u64::MAX,
            )?;
        }
        Ok(())
    }

    pub fn destroy(&self, ctx: &VkContext) {
        unsafe { ctx.device.destroy_semaphore(self.semaphore, None) };
    }
}
