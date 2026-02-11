use ara_core::bytemuck;

use crate::gpu::GpuContext;

pub mod smaa;
pub use smaa::{SmaaPipeline, SmaaPreset};

pub mod taa;
pub use taa::{TaaPipeline, TaaPreset};

/// A generic fullscreen shader pass.
pub struct FullscreenPass {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
}

impl FullscreenPass {
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        shader_source: &str,
        push_constant_size: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} Shader", label)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{} Bind Group Layout", label)),
            entries: &[
                // Input texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Input sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let push_constant_ranges = if push_constant_size > 0 {
            vec![wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::FRAGMENT,
                range: 0..push_constant_size,
            }]
        } else {
            vec![]
        };

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} Pipeline Layout", label)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &push_constant_ranges,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{} Pipeline", label)),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[], // No vertex buffers - we use vertex_index
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float, // HDR intermediate format
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    // Execute method removed as we use manual pass recording in BloomPipeline::render for now
}

pub struct BloomPipeline {
    pub(crate) threshold_pass: FullscreenPass,
    pub(crate) blur_pass: FullscreenPass,
    pub(crate) composite_pipeline: wgpu::RenderPipeline,
    pub(crate) composite_bind_group_layout: wgpu::BindGroupLayout,

    pub(crate) sampler: wgpu::Sampler,

    // Intermediate textures
    hdr_texture: wgpu::Texture,
    hdr_view: wgpu::TextureView,

    pub(crate) bloom_texture: wgpu::Texture,
    pub(crate) bloom_view: wgpu::TextureView,

    pub(crate) blur_temp_texture: wgpu::Texture,
    pub(crate) blur_temp_view: wgpu::TextureView,
}

impl BloomPipeline {
    pub fn new(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let device = gpu.device();

        let threshold_shader = include_str!("../assets/shaders/brightness_threshold.wgsl");
        let blur_shader = include_str!("../assets/shaders/blur.wgsl");
        let composite_shader = include_str!("../assets/shaders/composite.wgsl");

        // 1. Threshold Pass
        let threshold_pass = FullscreenPass::new(
            device,
            "Brightness Threshold",
            threshold_shader,
            16, // f32 threshold + padding
        );

        // 2. Blur Pass
        let blur_pass = FullscreenPass::new(
            device,
            "Blur",
            blur_shader,
            8, // vec2f direction
        );

        // 3. Composite Pass (Special case: 2 inputs, output to SwapChain format)
        let composite_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(composite_shader.into()),
        });

        let composite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Composite Bind Group Layout"),
                entries: &[
                    // Scene texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Scene sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Bloom texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Bloom sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ][..],
            });

        let composite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Composite Pipeline Layout"),
                bind_group_layouts: &[&composite_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::FRAGMENT,
                    range: 0..8, // bloom_intensity (f32) + exposure (f32)
                }],
            });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Composite Pipeline"),
            layout: Some(&composite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &composite_shader_module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: output_format, // Output to swapchain
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PostProcess Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let mut pipeline = Self {
            threshold_pass,
            blur_pass,
            composite_pipeline,
            composite_bind_group_layout,
            sampler,
            // Placeholders, will be created in resize
            hdr_texture: create_texture(
                device,
                1,
                1,
                wgpu::TextureFormat::Rgba16Float,
                "Placeholder",
            ),
            hdr_view: create_texture(
                device,
                1,
                1,
                wgpu::TextureFormat::Rgba16Float,
                "Placeholder",
            )
            .create_view(&Default::default()),
            bloom_texture: create_texture(
                device,
                1,
                1,
                wgpu::TextureFormat::Rgba16Float,
                "Placeholder",
            ),
            bloom_view: create_texture(
                device,
                1,
                1,
                wgpu::TextureFormat::Rgba16Float,
                "Placeholder",
            )
            .create_view(&Default::default()),
            blur_temp_texture: create_texture(
                device,
                1,
                1,
                wgpu::TextureFormat::Rgba16Float,
                "Placeholder",
            ),
            blur_temp_view: create_texture(
                device,
                1,
                1,
                wgpu::TextureFormat::Rgba16Float,
                "Placeholder",
            )
            .create_view(&Default::default()),
        };

        pipeline.resize(gpu, width, height);
        pipeline
    }

    pub fn resize(&mut self, gpu: &GpuContext, width: u32, height: u32) {
        let device = gpu.device();

        // 1. HDR Scene Texture (Full Res)
        self.hdr_texture = create_texture(
            device,
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            "HDR Scene",
        );
        self.hdr_view = self.hdr_texture.create_view(&Default::default());

        // 2. Bloom Textures (Half Res for performance & softness)
        let bloom_width = width / 2;
        let bloom_height = height / 2;

        self.bloom_texture = create_texture(
            device,
            bloom_width,
            bloom_height,
            wgpu::TextureFormat::Rgba16Float,
            "Bloom A",
        );
        self.bloom_view = self.bloom_texture.create_view(&Default::default());

        self.blur_temp_texture = create_texture(
            device,
            bloom_width,
            bloom_height,
            wgpu::TextureFormat::Rgba16Float,
            "Bloom B",
        );
        self.blur_temp_view = self.blur_temp_texture.create_view(&Default::default());
    }

    pub fn render(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        // 1. Threshold: HDR -> Bloom A
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Threshold Bind Group"),
                layout: &self.threshold_pass.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.hdr_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Threshold Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.threshold_pass.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let threshold = 1.0f32; // Extract pixels brighter than 1.0
            pass.set_push_constants(
                wgpu::ShaderStages::FRAGMENT,
                0,
                bytemuck::bytes_of(&threshold),
            );
            pass.draw(0..3, 0..1);
        }

        // 2. Blur Horizontal: Bloom A -> Bloom B
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Blur H Bind Group"),
                layout: &self.blur_pass.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.bloom_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blur H Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.blur_temp_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.blur_pass.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let dir = [1.0f32, 0.0f32];
            pass.set_push_constants(wgpu::ShaderStages::FRAGMENT, 0, bytemuck::bytes_of(&dir));
            pass.draw(0..3, 0..1);
        }

        // 3. Blur Vertical: Bloom B -> Bloom A
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Blur V Bind Group"),
                layout: &self.blur_pass.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.blur_temp_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blur V Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.bloom_view, // Write back to Bloom A
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.blur_pass.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let dir = [0.0f32, 1.0f32];
            pass.set_push_constants(wgpu::ShaderStages::FRAGMENT, 0, bytemuck::bytes_of(&dir));
            pass.draw(0..3, 0..1);
        }

        // 4. Composite: HDR + Bloom A -> Swapchain
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Composite Bind Group"),
                layout: &self.composite_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.hdr_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.bloom_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Composite Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: view, // Swapchain view
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            struct CompositeParams {
                intensity: f32, // 0.8
                exposure: f32,  // 1.0
            }
            let params = CompositeParams {
                intensity: 0.8,
                exposure: 1.0,
            };
            pass.set_push_constants(
                wgpu::ShaderStages::FRAGMENT,
                0,
                bytemuck::bytes_of(&[params.intensity, params.exposure]),
            );
            pass.draw(0..3, 0..1);
        }
    }

    pub fn hdr_view(&self) -> &wgpu::TextureView {
        &self.hdr_view
    }
    
    pub fn hdr_texture(&self) -> &wgpu::Texture {
        &self.hdr_texture
    }
}

fn create_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

pub struct FxaaPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniform_buf: wgpu::Buffer,
}

impl FxaaPipeline {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let shader_source = include_str!("../assets/shaders/fxaa.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FXAA Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Sampler: Linear, ClampToEdge
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("FXAA Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Resolution Uniform Buffer
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FXAA Uniforms"),
            size: 16, // vec2f + padding
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initial upload (will also be done in resize, but good to have)
        // We can't upload here easily without a queue, but resize is usually called immediately after or logic handles it.
        // Actually, we can use create_buffer_init if we had the util trait, but we don't.
        // We will rely on resize() being called or queue write in resize.
        // Since resize() requires queue, we will leave it empty here or require queue in new().
        // Existing pipelines don't take queue in new(), but resize() is called in Renderer::new usually?
        // Renderer::new calls pipeline.resize() usually.

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FXAA Bind Group Layout"),
            entries: &[
                // Input Texture
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FXAA Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("FXAA Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
            uniform_buf,
        }
    }

    pub fn resize(&self, queue: &wgpu::Queue, width: u32, height: u32) {
        let resolution = [width as f32, height as f32];
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&resolution));
    }

    pub fn render(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
    ) {
        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(input_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&self.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: self.uniform_buf.as_entire_binding(),
            },
        ];

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FXAA Bind Group"),
            layout: &self.bind_group_layout,
            entries: &entries[..],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("FXAA Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
