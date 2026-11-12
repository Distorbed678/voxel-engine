//! Thin, explicit Vulkan 1.3/1.4 foundation used by the engine.
//!
//! Design rules:
//! * One graphics+compute queue, timeline-semaphore driven frames.
//! * Every buffer has a device address (BDA); shaders reach buffers through
//!   pointers passed in push constants, so there is a single global
//!   pipeline layout: set 0 = bindless images, 256 bytes of push constants.
//! * Dynamic rendering + synchronization2 only, no render passes.

pub mod bindless;
pub mod buffer;
pub mod context;
pub mod image;
pub mod pipeline;
pub mod profiler;
pub mod shader;
pub mod swapchain;
pub mod sync;

pub use ash;
pub use ash::vk;
pub use gpu_allocator;
pub use gpu_allocator::MemoryLocation;

pub use bindless::Bindless;
pub use buffer::Buffer;
pub use context::VkContext;
pub use image::Image;
pub use profiler::GpuProfiler;
pub use shader::{ShaderCompiler, ShaderKind, ShaderOutput};
pub use swapchain::Swapchain;
