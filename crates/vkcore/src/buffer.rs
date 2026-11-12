use anyhow::Result;
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::VkContext;

/// A buffer with a device address. Host-visible buffers stay persistently mapped.
pub struct Buffer {
    pub raw: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: u64,
    pub address: vk::DeviceAddress,
    pub name: String,
}

impl Buffer {
    pub fn new(
        ctx: &VkContext,
        name: &str,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<Self> {
        let size = size.max(16);
        let usage = usage
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let raw = unsafe {
            ctx.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?
        };
        let requirements = unsafe { ctx.device.get_buffer_memory_requirements(raw) };
        let allocation = ctx.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
            allocation_scheme: if size >= 64 << 20 {
                AllocationScheme::DedicatedBuffer(raw)
            } else {
                AllocationScheme::GpuAllocatorManaged
            },
        })?;
        unsafe {
            ctx.device
                .bind_buffer_memory(raw, allocation.memory(), allocation.offset())?;
        }
        let address = unsafe {
            ctx.device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(raw))
        };
        ctx.set_name(raw, name);
        Ok(Self { raw, allocation: Some(allocation), size, address, name: name.to_string() })
    }

    /// Storage buffer in VRAM.
    pub fn gpu(ctx: &VkContext, name: &str, size: u64) -> Result<Self> {
        Self::new(
            ctx,
            name,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            MemoryLocation::GpuOnly,
        )
    }

    /// Host-visible upload buffer (write-combined; on ReBAR systems this is VRAM).
    pub fn upload(ctx: &VkContext, name: &str, size: u64) -> Result<Self> {
        Self::new(
            ctx,
            name,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            MemoryLocation::CpuToGpu,
        )
    }

    /// Host-visible readback buffer.
    pub fn readback(ctx: &VkContext, name: &str, size: u64) -> Result<Self> {
        Self::new(ctx, name, size, vk::BufferUsageFlags::STORAGE_BUFFER, MemoryLocation::GpuToCpu)
    }

    pub fn mapped(&self) -> &[u8] {
        let a = self.allocation.as_ref().expect("buffer destroyed");
        let ptr = a.mapped_ptr().expect("buffer is not host visible").as_ptr() as *const u8;
        unsafe { std::slice::from_raw_parts(ptr, self.size as usize) }
    }

    pub fn mapped_mut(&mut self) -> &mut [u8] {
        let a = self.allocation.as_ref().expect("buffer destroyed");
        let ptr = a.mapped_ptr().expect("buffer is not host visible").as_ptr() as *mut u8;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.size as usize) }
    }

    pub fn write<T: bytemuck::Pod>(&mut self, offset: u64, data: &[T]) {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let dst = self.mapped_mut();
        dst[offset as usize..offset as usize + bytes.len()].copy_from_slice(bytes);
    }

    pub fn read<T: bytemuck::Pod>(&self, offset: u64, count: usize) -> Vec<T> {
        let size = std::mem::size_of::<T>() * count;
        let src = &self.mapped()[offset as usize..offset as usize + size];
        let mut out = vec![T::zeroed(); count];
        bytemuck::cast_slice_mut::<T, u8>(&mut out).copy_from_slice(src);
        out
    }

    pub fn destroy(&mut self, ctx: &VkContext) {
        if let Some(a) = self.allocation.take() {
            let _ = ctx.allocator.lock().unwrap().free(a);
            unsafe { ctx.device.destroy_buffer(self.raw, None) };
            self.raw = vk::Buffer::null();
        }
    }
}
