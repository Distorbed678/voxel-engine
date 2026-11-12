use anyhow::Result;
use ash::vk;

use crate::VkContext;

pub const MAX_SAMPLED: u32 = 4096;
pub const MAX_STORAGE: u32 = 1024;

pub struct Samplers {
    /// Pixel-art friendly: nearest magnification, trilinear + 16x aniso minification, repeat.
    pub pixel_repeat: vk::Sampler,
    pub linear_repeat: vk::Sampler,
    pub linear_clamp: vk::Sampler,
    pub nearest_clamp: vk::Sampler,
    /// Depth comparison sampler for shadow maps (reversed-Z => GREATER).
    pub shadow: vk::Sampler,
    /// MIN reduction sampler for Hi-Z (reversed-Z farthest = min).
    pub min_reduce: vk::Sampler,
}

/// One global descriptor set: binding 0 = sampled images, binding 1 = storage images.
/// Buffers never go through descriptors (buffer device address).
pub struct Bindless {
    pub pool: vk::DescriptorPool,
    pub layout: vk::DescriptorSetLayout,
    pub set: vk::DescriptorSet,
    pub samplers: Samplers,
    sampled_next: u32,
    sampled_free: Vec<u32>,
    storage_next: u32,
    storage_free: Vec<u32>,
}

impl Bindless {
    pub fn new(ctx: &VkContext) -> Result<Self> {
        let d = &ctx.device;
        let flags = vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND;
        let binding_flags = [flags, flags];
        let mut flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_SAMPLED)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(MAX_STORAGE)
                .stage_flags(vk::ShaderStageFlags::ALL),
        ];
        let layout = unsafe {
            d.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&bindings)
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut flags_info),
                None,
            )?
        };
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: MAX_SAMPLED,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: MAX_STORAGE,
            },
        ];
        let pool = unsafe {
            d.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&sizes)
                    .max_sets(1)
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                None,
            )?
        };
        let layouts = [layout];
        let set = unsafe {
            d.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )?[0]
        };
        ctx.set_name(set, "bindless");

        let max_aniso = ctx.props.limits.max_sampler_anisotropy.min(16.0);
        let mk = |info: vk::SamplerCreateInfo| unsafe { d.create_sampler(&info, None) };
        let base = vk::SamplerCreateInfo::default().max_lod(vk::LOD_CLAMP_NONE);
        let pixel_repeat = mk(base
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(max_aniso))?;
        let linear_repeat = mk(base
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT))?;
        let linear_clamp = mk(base
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE))?;
        let nearest_clamp = mk(base
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE))?;
        let shadow = mk(base
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
            .compare_enable(true)
            .compare_op(vk::CompareOp::GREATER_OR_EQUAL))?;
        let mut reduction = vk::SamplerReductionModeCreateInfo::default()
            .reduction_mode(vk::SamplerReductionMode::MIN);
        let min_reduce = unsafe {
            d.create_sampler(
                &base
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .push_next(&mut reduction),
                None,
            )?
        };

        Ok(Self {
            pool,
            layout,
            set,
            samplers: Samplers {
                pixel_repeat,
                linear_repeat,
                linear_clamp,
                nearest_clamp,
                shadow,
                min_reduce,
            },
            sampled_next: 0,
            sampled_free: Vec::new(),
            storage_next: 0,
            storage_free: Vec::new(),
        })
    }

    pub fn add_sampled(
        &mut self,
        ctx: &VkContext,
        view: vk::ImageView,
        sampler: vk::Sampler,
        layout: vk::ImageLayout,
    ) -> u32 {
        let idx = self.sampled_free.pop().unwrap_or_else(|| {
            let i = self.sampled_next;
            self.sampled_next += 1;
            assert!(i < MAX_SAMPLED, "bindless sampled table full");
            i
        });
        self.update_sampled(ctx, idx, view, sampler, layout);
        idx
    }

    pub fn update_sampled(
        &self,
        ctx: &VkContext,
        idx: u32,
        view: vk::ImageView,
        sampler: vk::Sampler,
        layout: vk::ImageLayout,
    ) {
        let info = [vk::DescriptorImageInfo { sampler, image_view: view, image_layout: layout }];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(0)
            .dst_array_element(idx)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&info);
        unsafe { ctx.device.update_descriptor_sets(&[write], &[]) };
    }

    pub fn add_storage(&mut self, ctx: &VkContext, view: vk::ImageView) -> u32 {
        let idx = self.storage_free.pop().unwrap_or_else(|| {
            let i = self.storage_next;
            self.storage_next += 1;
            assert!(i < MAX_STORAGE, "bindless storage table full");
            i
        });
        self.update_storage(ctx, idx, view);
        idx
    }

    pub fn update_storage(&self, ctx: &VkContext, idx: u32, view: vk::ImageView) {
        let info = [vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: view,
            image_layout: vk::ImageLayout::GENERAL,
        }];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(1)
            .dst_array_element(idx)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&info);
        unsafe { ctx.device.update_descriptor_sets(&[write], &[]) };
    }

    pub fn free_sampled(&mut self, idx: u32) {
        self.sampled_free.push(idx);
    }

    pub fn free_storage(&mut self, idx: u32) {
        self.storage_free.push(idx);
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        unsafe {
            let s = &self.samplers;
            for sampler in [
                s.pixel_repeat,
                s.linear_repeat,
                s.linear_clamp,
                s.nearest_clamp,
                s.shadow,
                s.min_reduce,
            ] {
                ctx.device.destroy_sampler(sampler, None);
            }
            ctx.device.destroy_descriptor_pool(self.pool, None);
            ctx.device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
