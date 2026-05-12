//! Compute ray marcher pipeline and deferred color resolve.

use crate::brick_map::BrickMap;
use crate::clipmap::VoxelClipmaps;
use crate::gpu::GpuContext;

pub struct RayPipeline {
    march_pipeline: wgpu::ComputePipeline,
    resolve_pipeline: wgpu::ComputePipeline,
    noop_lighting: wgpu::ComputePipeline,

    visibility_buf: wgpu::Buffer,
    depth_buf: wgpu::Buffer,
    march_bind_group: wgpu::BindGroup,
    resolve_bind_group: wgpu::BindGroup,

    // Kept to recreate bind groups on resize
    march_bgl: wgpu::BindGroupLayout,
    resolve_bgl: wgpu::BindGroupLayout,

    width: u32,
    height: u32,
}

impl RayPipeline {
    pub fn new(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        brick_map: &BrickMap,
        clipmaps: &VoxelClipmaps,
        uniform_buf: &wgpu::Buffer,
        palette_tex: &wgpu::Texture,
        output_texture_view: &wgpu::TextureView,
        velocity_texture_view: &wgpu::TextureView,
    ) -> Self {
        let device = gpu.device();

        let visibility_buf = Self::create_visibility_buffer(device, width, height);
        let depth_buf = Self::create_depth_buffer(device, width, height);

        // --- Bind Group Layouts ---
        let march_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Ray March BGL"),
            entries: &[
                // globals
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // top_grid
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // brick_pool
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // brick_occupancy
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // visibility_buf
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // depth_buf
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // velocity_texture (write-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // cascade0..5 (clipmap 3D textures, Rgba8Uint)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
                    count: None,
                },
                // clip origins uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        let resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Deferred Resolve BGL"),
            entries: &[
                // globals
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // visibility_buf
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                // t_palette
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false },
                    count: None,
                },
                // s_palette
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // output_texture
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // depth_buf (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        // --- Pipelines ---
        let march_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ray March Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../assets/shaders/ray_march.wgsl").into()),
        });
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Deferred Resolve Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../assets/shaders/deferred_resolve.wgsl").into()),
        });
        let noop_light_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("NOOP Lighting Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../assets/shaders/noop_lighting.wgsl").into()),
        });

        let march_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ray March Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Ray March Pipeline Layout"),
                bind_group_layouts: &[&march_bgl],
                push_constant_ranges: &[],
            })),
            module: &march_shader,
            entry_point: Some("ray_march"),
            compilation_options: Default::default(),
            cache: None,
        });

        let resolve_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Deferred Resolve Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Deferred Resolve Pipeline Layout"),
                bind_group_layouts: &[&resolve_bgl],
                push_constant_ranges: &[],
            })),
            module: &resolve_shader,
            entry_point: Some("resolve"),
            compilation_options: Default::default(),
            cache: None,
        });

        let empty_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

        let noop_lighting = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("NOOP Lighting"),
            layout: Some(&empty_layout),
            module: &noop_light_shader,
            entry_point: Some("lighting_pass"),
            compilation_options: Default::default(),
            cache: None,
        });

        let palette_view = palette_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let palette_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Palette Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let (march_bind_group, resolve_bind_group) = Self::create_bind_groups(
            device,
            &march_bgl,
            &resolve_bgl,
            &visibility_buf,
            &depth_buf,
            brick_map,
            clipmaps,
            uniform_buf,
            &palette_view,
            &palette_sampler,
            output_texture_view,
            velocity_texture_view,
        );

        Self {
            march_pipeline,
            resolve_pipeline,
            noop_lighting,
            visibility_buf,
            depth_buf,
            march_bind_group,
            resolve_bind_group,
            march_bgl,
            resolve_bgl,
            width,
            height,
        }
    }

    fn create_visibility_buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
        let size = (width * height * 8) as u64; // 64 bits = 8 bytes per pixel
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Visibility Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    fn create_depth_buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
        let size = (width * height * 4) as u64; // f32 per pixel
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Depth Buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn create_bind_groups(
        device: &wgpu::Device,
        march_bgl: &wgpu::BindGroupLayout,
        resolve_bgl: &wgpu::BindGroupLayout,
        visibility_buf: &wgpu::Buffer,
        depth_buf: &wgpu::Buffer,
        brick_map: &BrickMap,
        clipmaps: &VoxelClipmaps,
        uniform_buf: &wgpu::Buffer,
        palette_view: &wgpu::TextureView,
        palette_sampler: &wgpu::Sampler,
        output_texture_view: &wgpu::TextureView,
        velocity_texture_view: &wgpu::TextureView,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let cascade_views = clipmaps.cascade_views();
        let march_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ray March Bind Group"),
            layout: march_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: brick_map.top_grid_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: brick_map.brick_pool_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: brick_map.brick_occupancy_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: visibility_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: depth_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(velocity_texture_view) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&cascade_views[0]) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&cascade_views[1]) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&cascade_views[2]) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&cascade_views[3]) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&cascade_views[4]) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::TextureView(&cascade_views[5]) },
                wgpu::BindGroupEntry { binding: 14, resource: clipmaps.origins_buf().as_entire_binding() },
            ],
        });

        let resolve_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Deferred Resolve Bind Group"),
            layout: resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: visibility_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(palette_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(palette_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(output_texture_view) },
                wgpu::BindGroupEntry { binding: 5, resource: depth_buf.as_entire_binding() },
            ],
        });

        (march_bind_group, resolve_bind_group)
    }

    pub fn resize(
        &mut self,
        gpu: &GpuContext,
        width: u32,
        height: u32,
        brick_map: &BrickMap,
        clipmaps: &VoxelClipmaps,
        uniform_buf: &wgpu::Buffer,
        palette_tex: &wgpu::Texture,
        output_texture_view: &wgpu::TextureView,
        velocity_texture_view: &wgpu::TextureView,
    ) {
        self.width = width;
        self.height = height;
        self.visibility_buf = Self::create_visibility_buffer(gpu.device(), width, height);
        self.depth_buf = Self::create_depth_buffer(gpu.device(), width, height);

        let palette_view = palette_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let palette_sampler = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Palette Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let (march_bind_group, resolve_bind_group) = Self::create_bind_groups(
            gpu.device(),
            &self.march_bgl,
            &self.resolve_bgl,
            &self.visibility_buf,
            &self.depth_buf,
            brick_map,
            clipmaps,
            uniform_buf,
            &palette_view,
            &palette_sampler,
            output_texture_view,
            velocity_texture_view,
        );

        self.march_bind_group = march_bind_group;
        self.resolve_bind_group = resolve_bind_group;
    }

    pub fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        let dispatch_x = (self.width + 7) / 8;
        let dispatch_y = (self.height + 7) / 8;

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ray March Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.march_pipeline);
            pass.set_bind_group(0, &self.march_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Lighting Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.noop_lighting);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Deferred Resolve Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.resolve_pipeline);
            pass.set_bind_group(0, &self.resolve_bind_group, &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
    }
}
