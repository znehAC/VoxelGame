//! GPU-based RGB light propagation via cellular automata on 3D textures.

use crate::gpu::GpuContext;
use ara_core::LIGHT_GRID_SIZE;

/// Ping-pong light propagation system using two 64^3 RGBA8 volumes.
pub struct LightPropagation {
    compute_pipeline: wgpu::ComputePipeline,
    /// A→B: read from volume_a, write to volume_b
    compute_bind_group_ab: wgpu::BindGroup,
    /// B→A: read from volume_b, write to volume_a
    compute_bind_group_ba: wgpu::BindGroup,
    /// Raytracer samples volume_a
    read_bind_group_a: wgpu::BindGroup,
    /// Raytracer samples volume_b
    read_bind_group_b: wgpu::BindGroup,
    read_bind_group_layout: wgpu::BindGroupLayout,
    /// true = current result is in volume_a
    current_is_a: bool,
    iterations: u32,
}

impl LightPropagation {
    pub fn new(
        gpu: &GpuContext,
        voxel_buf: &wgpu::Buffer,
        uniform_buf: &wgpu::Buffer,
        palette_tex: &wgpu::Texture,
        iterations: u32,
    ) -> Self {
        let device = gpu.device();

        let volume_a = create_light_volume(device, "Ara Light Volume A");
        let volume_b = create_light_volume(device, "Ara Light Volume B");

        let view_a = volume_a.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Ara Light View A"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        let view_b = volume_b.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Ara Light View B"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });

        let palette_view = palette_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Ara Palette View (Light)"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        // Compute bind group layout
        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Ara Light Compute BGL"),
            entries: &[
                // 0: voxel SSBO
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
                // 1: light source (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // 2: light destination (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                // 3: palette texture
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // 4: Global Uniforms
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
            ],
        });

        // A→B bind group
        let compute_bind_group_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ara Light Compute A→B"),
            layout: &compute_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: voxel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&palette_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        // B→A bind group
        let compute_bind_group_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ara Light Compute B→A"),
            layout: &compute_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: voxel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&palette_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        // Raytracer read bind group layout
        let read_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Ara Light Read BGL"),
                entries: &[
                    // 0: light volume (sample)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D3,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // 1: sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let light_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Ara Light Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let read_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ara Light Read A"),
            layout: &read_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&light_sampler),
                },
            ],
        });

        let read_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ara Light Read B"),
            layout: &read_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&light_sampler),
                },
            ],
        });

        // Compute pipeline
        let shader_src = include_str!("../assets/shaders/light_propagate.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ara Light Propagation Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Ara Light Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bgl],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ara Light Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader,
            entry_point: Some("propagate"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            compute_pipeline,
            compute_bind_group_ab,
            compute_bind_group_ba,
            read_bind_group_a,
            read_bind_group_b,
            read_bind_group_layout,
            current_is_a: true,
            iterations,
        }
    }

    /// Run N iterations of light propagation with ping-pong swapping.
    pub fn propagate(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let workgroups = LIGHT_GRID_SIZE / 4;

        for _ in 0..self.iterations {
            let bind_group = if self.current_is_a {
                &self.compute_bind_group_ab
            } else {
                &self.compute_bind_group_ba
            };

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Ara Light Propagation Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.compute_pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(workgroups, workgroups, workgroups);
            }

            self.current_is_a = !self.current_is_a;
        }
    }

    /// Bind group layout for the raytracer to reference in its pipeline layout.
    pub fn read_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.read_bind_group_layout
    }

    /// Current result bind group for the raytracer to sample.
    pub fn current_read_bind_group(&self) -> &wgpu::BindGroup {
        if self.current_is_a {
            &self.read_bind_group_a
        } else {
            &self.read_bind_group_b
        }
    }
}

fn create_light_volume(device: &wgpu::Device, label: &str) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: LIGHT_GRID_SIZE,
            height: LIGHT_GRID_SIZE,
            depth_or_array_layers: LIGHT_GRID_SIZE,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}
