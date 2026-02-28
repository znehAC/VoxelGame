//! VCT pipeline — sun injection + LOD mipmap compute passes for voxel cone tracing GI.

use crate::gpu::GpuContext;
use ara_core::bytemuck;

/// Voxel Cone Tracing pipeline: sun injection + LOD mipmap compute passes.
pub struct VctPipeline {
    inject_pipeline: wgpu::ComputePipeline,
    inject_bind_group: wgpu::BindGroup,
    mipmap_pipeline: wgpu::ComputePipeline,
    mipmap_bind_group: wgpu::BindGroup,
}

impl VctPipeline {
    pub fn new(
        gpu: &GpuContext,
        top_grid_buf: &wgpu::Buffer,
        brick_pool_buf: &wgpu::Buffer,
        brick_header_buf: &wgpu::Buffer,
        radiance_pool_buf: &wgpu::Buffer,
        brick_occupancy_buf: &wgpu::Buffer,
        uniform_buf: &wgpu::Buffer,
        palette_tex: &wgpu::Texture,
    ) -> Self {
        let device = gpu.device();

        let palette_view = palette_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("VCT Palette View"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        // Sun injection BGL:
        // 0: top_grid (storage read)
        // 1: brick_pool (storage read)
        // 2: brick_headers (storage read)
        // 3: radiance_pool (storage read_write)
        // 4: uniforms
        // 5: palette texture
        // 6: brick_occupancy (storage read)
        let inject_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VCT Sun Inject BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let inject_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VCT Sun Inject Bind Group"),
            layout: &inject_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: top_grid_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: brick_pool_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: brick_header_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: radiance_pool_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&palette_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: brick_occupancy_buf.as_entire_binding(),
                },
            ],
        });

        let inject_shader_src = include_str!("../assets/shaders/sun_inject.wgsl");
        let inject_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VCT Sun Inject Shader"),
            source: wgpu::ShaderSource::Wgsl(inject_shader_src.into()),
        });

        let inject_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("VCT Sun Inject Pipeline Layout"),
                bind_group_layouts: &[&inject_bgl],
                push_constant_ranges: &[],
            });

        let inject_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VCT Sun Inject Pipeline"),
            layout: Some(&inject_pipeline_layout),
            module: &inject_shader,
            entry_point: Some("inject"),
            compilation_options: Default::default(),
            cache: None,
        });

        // LOD mipmap BGL (same buffers, read_write on brick_pool + radiance_pool)
        let mipmap_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VCT LOD Mipmap BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let mipmap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VCT LOD Mipmap Bind Group"),
            layout: &mipmap_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: top_grid_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: brick_pool_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: brick_header_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: radiance_pool_buf.as_entire_binding(),
                },
            ],
        });

        let mipmap_shader_src = include_str!("../assets/shaders/lod_mipmap.wgsl");
        let mipmap_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VCT LOD Mipmap Shader"),
            source: wgpu::ShaderSource::Wgsl(mipmap_shader_src.into()),
        });

        let mipmap_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("VCT LOD Mipmap Pipeline Layout"),
                bind_group_layouts: &[&mipmap_bgl],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..4,
                }],
            });

        let mipmap_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VCT LOD Mipmap Pipeline"),
            layout: Some(&mipmap_pipeline_layout),
            module: &mipmap_shader,
            entry_point: Some("mipmap"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            inject_pipeline,
            inject_bind_group,
            mipmap_pipeline,
            mipmap_bind_group,
        }
    }

    /// Run sun injection then LOD mipmap (LOD0→1, LOD1→2).
    pub fn inject(&self, encoder: &mut wgpu::CommandEncoder, max_bricks: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("VCT Sun Inject Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.inject_pipeline);
        pass.set_bind_group(0, &self.inject_bind_group, &[]);
        pass.dispatch_workgroups(max_bricks, 2, 1);
    }

    pub fn mipmap(&self, encoder: &mut wgpu::CommandEncoder, max_bricks: u32) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("VCT LOD Mipmap LOD1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mipmap_pipeline);
            pass.set_bind_group(0, &self.mipmap_bind_group, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&1u32));
            pass.dispatch_workgroups(max_bricks, 2, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("VCT LOD Mipmap LOD2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mipmap_pipeline);
            pass.set_bind_group(0, &self.mipmap_bind_group, &[]);
            pass.set_push_constants(0, bytemuck::bytes_of(&2u32));
            pass.dispatch_workgroups(max_bricks, 2, 1);
        }
    }
}
