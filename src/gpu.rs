//! Headless-first GPU context wrapping wgpu Instance/Adapter/Device/Queue.



/// Core GPU state that can operate headless (compute-only) or with a surface.
pub struct GpuContext {
    /// Must outlive all surfaces created from it.
    #[allow(dead_code)]
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

        let mut limits = wgpu::Limits::default();
        limits.max_push_constant_size = 128;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Ara Device"),
                required_features: wgpu::Features::PUSH_CONSTANTS,
                required_limits: limits,
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



    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }


}
