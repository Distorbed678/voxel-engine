use anyhow::{Result, bail};
use ash::vk;

use crate::VkContext;

pub struct Swapchain {
    pub surface: vk::SurfaceKHR,
    pub raw: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    /// One "render finished" semaphore per swapchain image (safe reuse w.r.t. present).
    pub present_semaphores: Vec<vk::Semaphore>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    pub present_mode: vk::PresentModeKHR,
    pub vsync: bool,
}

impl Swapchain {
    pub fn new(
        ctx: &VkContext,
        surface: vk::SurfaceKHR,
        width: u32,
        height: u32,
        vsync: bool,
    ) -> Result<Self> {
        let supported = unsafe {
            ctx.surface_loader.get_physical_device_surface_support(
                ctx.physical_device,
                ctx.queue_family,
                surface,
            )?
        };
        if !supported {
            bail!("graphics queue cannot present to this surface");
        }
        let mut sc = Self {
            surface,
            raw: vk::SwapchainKHR::null(),
            images: Vec::new(),
            views: Vec::new(),
            present_semaphores: Vec::new(),
            format: vk::Format::UNDEFINED,
            extent: vk::Extent2D { width, height },
            present_mode: vk::PresentModeKHR::FIFO,
            vsync,
        };
        sc.recreate(ctx, width, height, vsync)?;
        Ok(sc)
    }

    pub fn recreate(&mut self, ctx: &VkContext, width: u32, height: u32, vsync: bool) -> Result<()> {
        unsafe {
            let caps = ctx
                .surface_loader
                .get_physical_device_surface_capabilities(ctx.physical_device, self.surface)?;
            let formats = ctx
                .surface_loader
                .get_physical_device_surface_formats(ctx.physical_device, self.surface)?;
            let modes = ctx
                .surface_loader
                .get_physical_device_surface_present_modes(ctx.physical_device, self.surface)?;

            // UNORM target: the tonemapper writes sRGB-encoded values itself so that ImGui
            // (which works in sRGB space) can draw straight on top.
            let format = formats
                .iter()
                .find(|f| {
                    f.format == vk::Format::B8G8R8A8_UNORM
                        && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .or_else(|| formats.iter().find(|f| f.format == vk::Format::R8G8B8A8_UNORM))
                .copied()
                .unwrap_or(formats[0]);

            let present_mode = if vsync {
                vk::PresentModeKHR::FIFO
            } else if modes.contains(&vk::PresentModeKHR::MAILBOX) {
                vk::PresentModeKHR::MAILBOX
            } else if modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
                vk::PresentModeKHR::IMMEDIATE
            } else {
                vk::PresentModeKHR::FIFO
            };

            let extent = if caps.current_extent.width != u32::MAX {
                caps.current_extent
            } else {
                vk::Extent2D {
                    width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                    height: height
                        .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
                }
            };
            let mut image_count = caps.min_image_count.max(3);
            if caps.max_image_count > 0 {
                image_count = image_count.min(caps.max_image_count);
            }

            let old = self.raw;
            let info = vk::SwapchainCreateInfoKHR::default()
                .surface(self.surface)
                .min_image_count(image_count)
                .image_format(format.format)
                .image_color_space(format.color_space)
                .image_extent(extent)
                .image_array_layers(1)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::STORAGE,
                )
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(caps.current_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .old_swapchain(old);
            let raw = ctx.swapchain_loader.create_swapchain(&info, None)?;

            self.destroy_views(ctx);
            if old != vk::SwapchainKHR::null() {
                ctx.swapchain_loader.destroy_swapchain(old, None);
            }

            self.raw = raw;
            self.images = ctx.swapchain_loader.get_swapchain_images(raw)?;
            self.views = self
                .images
                .iter()
                .map(|&img| {
                    ctx.device.create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(img)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(format.format)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: vk::ImageAspectFlags::COLOR,
                                base_mip_level: 0,
                                level_count: 1,
                                base_array_layer: 0,
                                layer_count: 1,
                            }),
                        None,
                    )
                })
                .collect::<Result<_, _>>()?;
            self.present_semaphores = (0..self.images.len())
                .map(|_| ctx.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None))
                .collect::<Result<_, _>>()?;
            self.format = format.format;
            self.extent = extent;
            self.present_mode = present_mode;
            self.vsync = vsync;
            log::info!(
                "swapchain {}x{} {:?} {:?} ({} images)",
                extent.width,
                extent.height,
                format.format,
                present_mode,
                self.images.len()
            );
        }
        Ok(())
    }

    fn destroy_views(&mut self, ctx: &VkContext) {
        unsafe {
            for v in self.views.drain(..) {
                ctx.device.destroy_image_view(v, None);
            }
            for s in self.present_semaphores.drain(..) {
                ctx.device.destroy_semaphore(s, None);
            }
        }
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        self.destroy_views(ctx);
        unsafe {
            if self.raw != vk::SwapchainKHR::null() {
                ctx.swapchain_loader.destroy_swapchain(self.raw, None);
                self.raw = vk::SwapchainKHR::null();
            }
            ctx.surface_loader.destroy_surface(self.surface, None);
        }
    }
}
