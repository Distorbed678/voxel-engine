use std::ffi::{CStr, CString, c_void};
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Context as _, Result, bail};
use ash::{Device, Entry, Instance, vk};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use raw_window_handle::RawDisplayHandle;

/// Number of validation errors reported so far (used by headless self-tests).
pub static VALIDATION_ERRORS: AtomicU32 = AtomicU32::new(0);

pub struct VkContext {
    pub entry: Entry,
    pub instance: Instance,
    pub debug_utils: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    pub debug_device: Option<ash::ext::debug_utils::Device>,
    pub surface_loader: ash::khr::surface::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub mesh_shader: ash::ext::mesh_shader::Device,
    pub swapchain_loader: ash::khr::swapchain::Device,
    pub allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
    pub props: vk::PhysicalDeviceProperties,
    pub mesh_props: MeshShaderLimits,
    pub device_name: String,
    pub driver_info: String,
    pub api_version: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct MeshShaderLimits {
    pub max_mesh_output_vertices: u32,
    pub max_mesh_output_primitives: u32,
    pub max_task_payload_size: u32,
    pub max_preferred_mesh_workgroup_invocations: u32,
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    let msg = unsafe {
        let d = &*data;
        if d.p_message.is_null() {
            "<null>".to_string()
        } else {
            CStr::from_ptr(d.p_message).to_string_lossy().into_owned()
        }
    };
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
        log::error!("[vk {ty:?}] {msg}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        log::warn!("[vk {ty:?}] {msg}");
    } else {
        log::debug!("[vk {ty:?}] {msg}");
    }
    vk::FALSE
}

pub struct ContextDesc {
    pub display: Option<RawDisplayHandle>,
    pub validation: bool,
    pub app_name: &'static str,
}

impl VkContext {
    pub fn new(desc: ContextDesc) -> Result<Self> {
        let entry = unsafe { Entry::load() }.context("failed to load Vulkan loader")?;

        let loader_version = unsafe { entry.try_enumerate_instance_version()? }
            .unwrap_or(vk::API_VERSION_1_0);
        if loader_version < vk::make_api_version(0, 1, 3, 0) {
            bail!("Vulkan loader is too old (need 1.3+)");
        }
        // Request 1.4 when the loader knows about it; every 1.3 path we use is core there too.
        let api_version = if loader_version >= vk::make_api_version(0, 1, 4, 0) {
            vk::make_api_version(0, 1, 4, 0)
        } else {
            vk::make_api_version(0, 1, 3, 0)
        };

        let app_name = CString::new(desc.app_name).unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(c"voxel-engine")
            .engine_version(1)
            .api_version(api_version);

        let mut extensions: Vec<*const i8> = Vec::new();
        if let Some(display) = desc.display {
            extensions.extend_from_slice(ash_window::enumerate_required_extensions(display)?);
        }

        let available_layers = unsafe { entry.enumerate_instance_layer_properties()? };
        let has_validation = available_layers.iter().any(|l| {
            l.layer_name_as_c_str().map(|n| n == c"VK_LAYER_KHRONOS_validation").unwrap_or(false)
        });
        let validation = desc.validation && has_validation;
        if desc.validation && !has_validation {
            log::warn!("validation requested but VK_LAYER_KHRONOS_validation is not installed");
        }
        let mut layers: Vec<*const i8> = Vec::new();
        if validation {
            layers.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
        }
        // Debug utils is cheap and gives us object names in tools (RenderDoc/Nsight) always.
        extensions.push(ash::ext::debug_utils::NAME.as_ptr());

        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layers);
        let instance = unsafe { entry.create_instance(&instance_info, None) }
            .context("vkCreateInstance failed")?;

        let debug_utils = if validation {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            let messenger = unsafe { loader.create_debug_utils_messenger(&info, None)? };
            Some((loader, messenger))
        } else {
            None
        };

        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);

        // ---- physical device -------------------------------------------------------------
        let physical_devices = unsafe { instance.enumerate_physical_devices()? };
        let mut chosen = None;
        for pd in physical_devices {
            let props = unsafe { instance.get_physical_device_properties(pd) };
            if props.api_version < vk::make_api_version(0, 1, 3, 0) {
                continue;
            }
            let exts = unsafe { instance.enumerate_device_extension_properties(pd)? };
            let has = |name: &CStr| {
                exts.iter().any(|e| e.extension_name_as_c_str().map(|n| n == name).unwrap_or(false))
            };
            if !has(ash::ext::mesh_shader::NAME) {
                continue;
            }
            if desc.display.is_some() && !has(ash::khr::swapchain::NAME) {
                continue;
            }
            let qf = unsafe { instance.get_physical_device_queue_family_properties(pd) };
            let Some(family) = qf.iter().position(|q| {
                q.queue_flags.contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
            }) else {
                continue;
            };
            let score = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
                _ => 1,
            };
            if chosen.as_ref().map(|(_, _, s, _)| score > *s).unwrap_or(true) {
                chosen = Some((pd, family as u32, score, props));
            }
        }
        let Some((physical_device, queue_family, _, props)) = chosen else {
            bail!("no GPU with Vulkan 1.3 + VK_EXT_mesh_shader found");
        };
        let device_name = props
            .device_name_as_c_str()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();

        let mut driver_props = vk::PhysicalDeviceDriverProperties::default();
        let mut mesh_props_raw = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut driver_props)
            .push_next(&mut mesh_props_raw);
        unsafe { instance.get_physical_device_properties2(physical_device, &mut props2) };
        let driver_info = format!(
            "{} {}",
            driver_props.driver_name_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            driver_props.driver_info_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
        );
        let mesh_props = MeshShaderLimits {
            max_mesh_output_vertices: mesh_props_raw.max_mesh_output_vertices,
            max_mesh_output_primitives: mesh_props_raw.max_mesh_output_primitives,
            max_task_payload_size: mesh_props_raw.max_task_payload_size,
            max_preferred_mesh_workgroup_invocations: mesh_props_raw
                .max_preferred_mesh_work_group_invocations,
        };
        log::info!(
            "GPU: {device_name} ({driver_info}), API {}.{}.{}",
            vk::api_version_major(props.api_version),
            vk::api_version_minor(props.api_version),
            vk::api_version_patch(props.api_version)
        );

        // ---- logical device --------------------------------------------------------------
        let features10 = vk::PhysicalDeviceFeatures::default()
            .sampler_anisotropy(true)
            .fill_mode_non_solid(true)
            .multi_draw_indirect(true)
            .shader_int64(true)
            .shader_int16(true)
            .fragment_stores_and_atomics(true)
            .vertex_pipeline_stores_and_atomics(true)
            .independent_blend(true)
            .depth_clamp(true)
            .depth_bias_clamp(true)
            .image_cube_array(true)
            .shader_storage_image_write_without_format(true)
            .shader_storage_image_read_without_format(true);
        let mut features2 = vk::PhysicalDeviceFeatures2::default().features(features10);
        let mut v11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(true)
            .uniform_and_storage_buffer16_bit_access(true)
            .shader_draw_parameters(true);
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .descriptor_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_image_array_non_uniform_indexing(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .runtime_descriptor_array(true)
            .timeline_semaphore(true)
            .scalar_block_layout(true)
            .shader_int8(true)
            .shader_float16(true)
            .storage_buffer8_bit_access(true)
            .uniform_and_storage_buffer8_bit_access(true)
            .draw_indirect_count(true)
            .host_query_reset(true)
            .shader_buffer_int64_atomics(true)
            .sampler_filter_minmax(true)
            .vulkan_memory_model(true)
            .vulkan_memory_model_device_scope(true);
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true)
            .maintenance4(true)
            .subgroup_size_control(true)
            .compute_full_subgroups(true)
            .shader_demote_to_helper_invocation(true);
        let mut mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default()
            .task_shader(true)
            .mesh_shader(true);

        let mut device_exts = vec![ash::ext::mesh_shader::NAME.as_ptr()];
        if desc.display.is_some() {
            device_exts.push(ash::khr::swapchain::NAME.as_ptr());
        }

        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&device_exts)
            .push_next(&mut features2)
            .push_next(&mut v11)
            .push_next(&mut v12)
            .push_next(&mut v13)
            .push_next(&mut mesh);
        let device = unsafe { instance.create_device(physical_device, &device_info, None) }
            .context("vkCreateDevice failed (missing feature?)")?;
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let mesh_shader = ash::ext::mesh_shader::Device::new(&instance, &device);
        let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
        let debug_device = Some(ash::ext::debug_utils::Device::new(&instance, &device));

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        Ok(Self {
            entry,
            instance,
            debug_utils,
            debug_device,
            surface_loader,
            physical_device,
            device,
            queue,
            queue_family,
            mesh_shader,
            swapchain_loader,
            allocator: ManuallyDrop::new(Arc::new(Mutex::new(allocator))),
            props,
            mesh_props,
            device_name,
            driver_info,
            api_version,
        })
    }

    pub fn set_name<H: vk::Handle>(&self, handle: H, name: &str) {
        if let Some(dbg) = &self.debug_device {
            let Ok(cname) = CString::new(name) else { return };
            let info = vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(handle)
                .object_name(&cname);
            unsafe {
                let _ = dbg.set_debug_utils_object_name(&info);
            }
        }
    }

    pub fn begin_label(&self, cb: vk::CommandBuffer, name: &str) {
        if let Some(dbg) = &self.debug_device {
            let Ok(cname) = CString::new(name) else { return };
            let label = vk::DebugUtilsLabelEXT::default().label_name(&cname);
            unsafe { dbg.cmd_begin_debug_utils_label(cb, &label) };
        }
    }

    pub fn end_label(&self, cb: vk::CommandBuffer) {
        if let Some(dbg) = &self.debug_device {
            unsafe { dbg.cmd_end_debug_utils_label(cb) };
        }
    }

    pub fn wait_idle(&self) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
    }

    /// Record and synchronously execute a one-off command buffer (startup uploads only).
    pub fn immediate<F: FnOnce(vk::CommandBuffer)>(&self, f: F) -> Result<()> {
        unsafe {
            let pool = self.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(self.queue_family)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT),
                None,
            )?;
            let cb = self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .command_buffer_count(1),
            )?[0];
            self.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            f(cb);
            self.device.end_command_buffer(cb)?;
            let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(cb)];
            let submit = vk::SubmitInfo2::default().command_buffer_infos(&cbs);
            self.device.queue_submit2(self.queue, &[submit], fence)?;
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.destroy_fence(fence, None);
            self.device.destroy_command_pool(pool, None);
        }
        Ok(())
    }
}

impl Drop for VkContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            // The allocator must die before the device.
            let alloc = ManuallyDrop::take(&mut self.allocator);
            match Arc::try_unwrap(alloc) {
                Ok(m) => drop(m),
                Err(_) => log::warn!("allocator still shared at shutdown; leaking"),
            }
            self.device.destroy_device(None);
            if let Some((loader, messenger)) = self.debug_utils.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
