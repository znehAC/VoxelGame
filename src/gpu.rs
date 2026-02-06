//! Headless-first GPU context wrapping wgpu Instance/Adapter/Device/Queue.

use std::sync::Arc;

/// Core GPU state that can operate headless (compute-only) or with a surface.
pub struct GpuContext {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuContext {
    /// Create a wgpu Instance with Vulkan + Metal backends.
    pub fn create_instance() -> wgpu::Instance {
        wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            ..Default::default()
        })
    }

    /// Create a headless context (no surface compatibility).
    pub async fn new_headless() -> Self {
        let instance = Self::create_instance();
        Self::from_instance(instance, None).await
    }

    /// Create a context from an existing instance, optionally compatible with a surface.
    pub async fn from_instance(
        instance: wgpu::Instance,
        surface: Option<&wgpu::Surface<'static>>,
    ) -> Self {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: surface,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find a suitable GPU adapter");

        let info = adapter.get_info();
        log::info!("GPU: {} ({:?})", info.name, info.backend);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Ara Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            }, None)
            .await
            .expect("Failed to create GPU device");

        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }

    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }

    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Create a surface for the given window, using this context's instance.
    pub fn create_surface(
        &self,
        window: Arc<winit::window::Window>,
    ) -> wgpu::Surface<'static> {
        self.instance
            .create_surface(window)
            .expect("Failed to create surface")
    }
}
