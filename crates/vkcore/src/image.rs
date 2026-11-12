use anyhow::Result;
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::VkContext;

pub struct Image {
    pub raw: vk::Image,
    pub view: vk::ImageView,
    pub allocation: Option<Allocation>,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mips: u32,
    pub layers: u32,
    pub aspect: vk::ImageAspectFlags,
    /// Per-mip views (only created for images flagged with `per_mip_views`).
    pub mip_views: Vec<vk::ImageView>,
}

pub struct ImageDesc<'a> {
    pub name: &'a str,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mips: u32,
    pub layers: u32,
    pub usage: vk::ImageUsageFlags,
    pub view_type: vk::ImageViewType,
    pub per_mip_views: bool,
}

impl<'a> ImageDesc<'a> {
    pub fn d2(name: &'a str, format: vk::Format, w: u32, h: u32, usage: vk::ImageUsageFlags) -> Self {
        Self {
            name,
            format,
            extent: vk::Extent3D { width: w, height: h, depth: 1 },
            mips: 1,
            layers: 1,
            usage,
            view_type: vk::ImageViewType::TYPE_2D,
            per_mip_views: false,
        }
    }
}

pub fn is_depth(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D32_SFLOAT | vk::Format::D16_UNORM | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT
    )
}

impl Image {
    pub fn new(ctx: &VkContext, desc: &ImageDesc) -> Result<Self> {
        let image_type = if desc.view_type == vk::ImageViewType::TYPE_3D {
            vk::ImageType::TYPE_3D
        } else {
            vk::ImageType::TYPE_2D
        };
        let raw = unsafe {
            ctx.device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(image_type)
                    .format(desc.format)
                    .extent(desc.extent)
                    .mip_levels(desc.mips)
                    .array_layers(desc.layers)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(desc.usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )?
        };
        let requirements = unsafe { ctx.device.get_image_memory_requirements(raw) };
        let allocation = ctx.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
            name: desc.name,
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::DedicatedImage(raw),
        })?;
        unsafe { ctx.device.bind_image_memory(raw, allocation.memory(), allocation.offset())? };
        let aspect = if is_depth(desc.format) {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let view = unsafe {
            ctx.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(raw)
                    .view_type(desc.view_type)
                    .format(desc.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: aspect,
                        base_mip_level: 0,
                        level_count: desc.mips,
                        base_array_layer: 0,
                        layer_count: desc.layers,
                    }),
                None,
            )?
        };
        let mut mip_views = Vec::new();
        if desc.per_mip_views {
            for mip in 0..desc.mips {
                mip_views.push(unsafe {
                    ctx.device.create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(raw)
                            .view_type(desc.view_type)
                            .format(desc.format)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: aspect,
                                base_mip_level: mip,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: desc.layers,
                            }),
                        None,
                    )?
                });
            }
        }
        ctx.set_name(raw, desc.name);
        Ok(Self {
            raw,
            view,
            allocation: Some(allocation),
            format: desc.format,
            extent: desc.extent,
            mips: desc.mips,
            layers: desc.layers,
            aspect,
            mip_views,
        })
    }

    /// Create an extra view of a single array layer (e.g. one shadow cascade).
    pub fn layer_view(&self, ctx: &VkContext, layer: u32) -> Result<vk::ImageView> {
        Ok(unsafe {
            ctx.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(self.raw)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(self.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: self.aspect,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: layer,
                        layer_count: 1,
                    }),
                None,
            )?
        })
    }

    pub fn full_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: self.aspect,
            base_mip_level: 0,
            level_count: self.mips,
            base_array_layer: 0,
            layer_count: self.layers,
        }
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        if let Some(a) = self.allocation.take() {
            unsafe {
                for v in self.mip_views.drain(..) {
                    ctx.device.destroy_image_view(v, None);
                }
                ctx.device.destroy_image_view(self.view, None);
                ctx.device.destroy_image(self.raw, None);
            }
            let _ = ctx.allocator.lock().unwrap().free(a);
        }
    }
}
