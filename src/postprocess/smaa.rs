use crate::gpu::GpuContext;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

// --- Embedded Standard SMAA Data ---
const AREA_TEX_BYTES: &[u8] = include_bytes!("../../assets/smaa/AreaTex.bin");
const SEARCH_TEX_BYTES: &[u8] = include_bytes!("../../assets/smaa/SearchTex.bin");

/// SMAA Quality Presets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum SmaaPreset {
    /// 60% quality - fastest
    Low,
    /// 80% quality - balanced
    Medium,
    /// 95% quality - good quality (default)
    High,
    /// 99% quality - best
    Ultra,
    /// Custom configuration (not matching any preset)
    Custom,
}

impl SmaaPreset {
    /// Get configuration values for this preset
    pub fn config(&self) -> SmaaConfig {
        match self {
            SmaaPreset::Low => SmaaConfig {
                threshold: 0.15,
                max_search_steps: 4,
            },
            SmaaPreset::Medium => SmaaConfig {
                threshold: 0.10,
                max_search_steps: 8,
            },
            SmaaPreset::High => SmaaConfig {
                threshold: 0.10,
                max_search_steps: 16,
            },
            SmaaPreset::Ultra => SmaaConfig {
                threshold: 0.05,
                max_search_steps: 32,
            },
            SmaaPreset::Custom => panic!("Custom preset has no predefined configuration"),
        }
    }
}

/// SMAA Configuration Parameters
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct SmaaConfig {
    /// Edge detection threshold (0.05 - 0.15)
    pub threshold: f32,
    /// Maximum search steps for edge ends (4 - 32)
    pub max_search_steps: i32,
}

impl Default for SmaaConfig {
    fn default() -> Self {
        SmaaPreset::High.config()
    }
}

pub struct SmaaPipeline {
    edges_pipeline: wgpu::RenderPipeline,
    weights_pipeline: wgpu::RenderPipeline,
    blend_pipeline: wgpu::RenderPipeline,

    bind_group_layout: wgpu::BindGroupLayout,

    // Textures
    area_view: wgpu::TextureView,
    search_view: wgpu::TextureView,
    edges_view: wgpu::TextureView,
    weights_view: wgpu::TextureView,
    edges_tex: wgpu::Texture,
    weights_tex: wgpu::Texture,

    // Resources
    sampler_linear: wgpu::Sampler,
    sampler_point: wgpu::Sampler,
    metrics_buf: wgpu::Buffer,

    width: u32,
    height: u32,
}

impl SmaaPipeline {
    /// Create a new SMAA pipeline with the specified preset
    pub fn new(gpu: &GpuContext, width: u32, height: u32, format: wgpu::TextureFormat, preset: SmaaPreset) -> Self {
        Self::with_config(gpu, width, height, format, preset.config())
    }

    /// Create a new SMAA pipeline with custom configuration
    pub fn with_config(gpu: &GpuContext, width: u32, height: u32, format: wgpu::TextureFormat, config: SmaaConfig) -> Self {
        let device = gpu.device();
        let queue = gpu.queue();
        
        // Generate shader with config constants
        let shader_source = generate_smaa_shader(&config);
        
        // 1. Create Standard SMAA Lookup Textures

        // AreaTex: 160x560, RG8_UNORM (2 bytes/pixel)
        // This texture allows SMAA to estimate coverage for specific edge patterns
        let area_tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("SMAA Area Tex"),
                size: wgpu::Extent3d {
                    width: 160,
                    height: 560,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg8Unorm, // MUST be Rg8Unorm
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            AREA_TEX_BYTES,
        );

        // SearchTex: 64x16, R8_UNORM (1 byte/pixel)
        // This texture helps the shader quickly find the ends of edges
        let search_tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("SMAA Search Tex"),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 16,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm, // MUST be R8Unorm
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            SEARCH_TEX_BYTES,
        );

        let area_view = area_tex.create_view(&Default::default());
        let search_view = search_tex.create_view(&Default::default());

        // 2. Samplers
        // SMAA relies on Bilinear filtering for some lookups and Point for others
        let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SMAA Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let sampler_point = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SMAA Point Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // 3. Uniforms
        let metrics_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SMAA Metrics"),
            size: 32, // vec4 (16) + u32 (4) + padding (12)
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 4. Bind Group Layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SMAA Layout"),
            entries: &[
                // Binding 0: Input/Edge/Weight Texture (changes per pass)
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
                // Binding 1: Search Texture (Constant)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true }, // Must be filterable for bilinear search
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 2: Area Texture (Constant)
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
                // Binding 3: Linear Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding 4: Point Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // Binding 5: Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // 5. Load Shader (with config constants injected)
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SMAA Shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(shader_source)),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SMAA Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // 6. Create Pipelines
        let edges_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SMAA Edges"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_edges"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_edges"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rg8Unorm, // Edges stored in RG8
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

        let weights_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SMAA Weights"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_weights"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_weights"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm, // Weights stored in RGBA8
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

        let blend_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SMAA Blend"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_blend"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_blend"),
                targets: &[Some(wgpu::ColorTargetState {
                    format, // Output format
                    blend: Some(wgpu::BlendState::REPLACE),
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

        // Create Internal Targets
        let edges_tex = Self::create_target_texture(
            device,
            width,
            height,
            wgpu::TextureFormat::Rg8Unorm,
            "SMAA Edges",
        );
        let edges_view = edges_tex.create_view(&Default::default());

        let weights_tex = Self::create_target_texture(
            device,
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
            "SMAA Weights",
        );
        let weights_view = weights_tex.create_view(&Default::default());

        let pipeline = Self {
            edges_pipeline,
            weights_pipeline,
            blend_pipeline,
            bind_group_layout,
            area_view,
            search_view,
            edges_view,
            weights_view,
            edges_tex,
            weights_tex,
            sampler_linear,
            sampler_point,
            metrics_buf,
            width,
            height,
        };

        pipeline.update_buffer(queue);
        pipeline
    }

    fn create_target_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    }

    pub fn resize(&mut self, gpu: &GpuContext, width: u32, height: u32) {
        let device = gpu.device();
        let queue = gpu.queue();
        self.width = width;
        self.height = height;

        self.edges_tex = Self::create_target_texture(
            device,
            width,
            height,
            wgpu::TextureFormat::Rg8Unorm,
            "SMAA Edges",
        );
        self.edges_view = self.edges_tex.create_view(&Default::default());

        self.weights_tex = Self::create_target_texture(
            device,
            width,
            height,
            wgpu::TextureFormat::Rgba8Unorm,
            "SMAA Weights",
        );
        self.weights_view = self.weights_tex.create_view(&Default::default());

        self.update_buffer(queue);
    }

    fn update_buffer(&self, queue: &wgpu::Queue) {
        let metrics = [
            1.0 / self.width as f32,
            1.0 / self.height as f32,
            self.width as f32,
            self.height as f32,
        ];

        // WGSL uniform layout: vec4f (16 bytes) + u32 (4 bytes) + 12 bytes padding = 32 bytes
        let mut data = Vec::with_capacity(32);
        data.extend_from_slice(bytemuck::cast_slice(&metrics));
        data.extend_from_slice(bytemuck::cast_slice(&[0u32]));
        data.extend_from_slice(&[0u8; 12]);

        queue.write_buffer(&self.metrics_buf, 0, &data);
    }

    pub fn render(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
    ) {
        // Pass 1: Edge Detection
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SMAA Edges BG"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.search_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.area_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.sampler_point),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.metrics_buf.as_entire_binding(),
                    },
                ],
            });

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SMAA Edge Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.edges_view,
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
            rpass.set_pipeline(&self.edges_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        // Pass 2: Blending Weights
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SMAA Weights BG"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.edges_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.search_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.area_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.sampler_point),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.metrics_buf.as_entire_binding(),
                    },
                ],
            });

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SMAA Weights Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.weights_view,
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
            rpass.set_pipeline(&self.weights_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }

        // Pass 3: Neighborhood Blending
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SMAA Blend BG"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.search_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.weights_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.sampler_linear),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.sampler_point),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.metrics_buf.as_entire_binding(),
                    },
                ],
            });

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SMAA Blend Pass"),
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
            rpass.set_pipeline(&self.blend_pipeline);
            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.draw(0..3, 0..1);
        }
    }
}

/// Generate SMAA shader source with configurable constants
fn generate_smaa_shader(config: &SmaaConfig) -> String {
    let base_shader = include_str!("../../assets/shaders/smaa.wgsl");
    
    // Replace the config constants at the top of the shader
    let mut result = base_shader.to_string();
    
    // Replace threshold constant
    result = result.replace(
        "const SMAA_THRESHOLD: f32 = 0.1;",
        &format!("const SMAA_THRESHOLD: f32 = {};", config.threshold)
    );
    
    // Replace max search steps
    result = result.replace(
        "const SMAA_MAX_SEARCH_STEPS: i32 = 16;",
        &format!("const SMAA_MAX_SEARCH_STEPS: i32 = {};", config.max_search_steps)
    );
    
    result
}
