//! Presentation layer: fullscreen-quad pipeline with palette texture, voxel SSBO, and uniforms.

use ara_core::glam::Mat4;
use ara_core::{GlobalUniforms, LightBuffer, PackedVoxel, bytemuck};

use crate::assets;
use crate::gpu::GpuContext;
use crate::light::LightPropagation;
use crate::postprocess::{
    BloomPipeline, FxaaPipeline, SmaaPipeline, SmaaPreset, TaaPipeline, TaaPreset,
};
use crate::ui::{UiContext, UiSystem};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AaMode {
    None,
    Fxaa,
    Smaa,
    Taa,
    TaaThenSmaa,
}

/// Manages surface presentation with palette-texture + voxel-SSBO pipeline.
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    light_buf: wgpu::Buffer,
    voxel_buf: wgpu::Buffer,
    palette_tex: wgpu::Texture,
    light: LightPropagation,
    post_process: BloomPipeline,
    fxaa: FxaaPipeline,
    smaa: SmaaPipeline,
    taa: TaaPipeline,
    final_sdr_texture: wgpu::Texture,
    final_sdr_view: wgpu::TextureView,
    ui_system: UiSystem,
    pub aa_mode: AaMode,
    pub smaa_debug_mode: u32,

    // TAA-related fields
    velocity_texture: wgpu::Texture,
    velocity_view: wgpu::TextureView,
    prev_view_proj: [[f32; 4]; 4],
    frame_count: u64,
    width: u32,
    height: u32,
}

impl Renderer {
    /// Create a new renderer with palette texture and voxel data uploaded to GPU.
    pub fn new(
        gpu: &GpuContext,
        surface: wgpu::Surface<'static>,
        width: u32,
        height: u32,
        palette_data: &[u8],
        voxel_data: &[PackedVoxel],
    ) -> Self {
        let caps = surface.get_capabilities(gpu.adapter());
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(gpu.device(), &config);

        // Initialize Bloom Pipeline
        let post_process = BloomPipeline::new(gpu, width, height, format);

        // Initialize FXAA Pipeline
        let fxaa = FxaaPipeline::new(gpu.device(), &config);

        // Initialize SMAA Pipeline
        let smaa = SmaaPipeline::new(gpu, width, height, format, SmaaPreset::Ultra);

        // Initialize Final SDR Texture (Input for AA)
        let final_sdr_texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("Final SDR Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let final_sdr_view = final_sdr_texture.create_view(&Default::default());

        // Initialize UI System
        let ui_system = UiSystem::new(gpu, format);

        // Palette texture: 256x256, 2 layers, Rgba8UnormSrgb
        let palette_tex = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("Ara Palette Texture"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let layer_size: usize = 256 * 256 * 4;

        // Upload layer 0
        gpu.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &palette_tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &palette_data[..layer_size],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: Some(256),
            },
            wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
        );

        // Upload layer 1
        gpu.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &palette_tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
                aspect: wgpu::TextureAspect::All,
            },
            &palette_data[layer_size..],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: Some(256),
            },
            wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
        );

        let palette_view = palette_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Ara Palette View"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let palette_sampler = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Ara Palette Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Voxel SSBO
        let voxel_bytes = bytemuck::cast_slice::<PackedVoxel, u8>(voxel_data);
        let voxel_buf = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Voxel SSBO"),
            size: voxel_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue().write_buffer(&voxel_buf, 0, voxel_bytes);

        // Uniform buffer
        let uniform_buf = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Global Uniforms"),
            size: std::mem::size_of::<GlobalUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Light uniform buffer
        let light_buf = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Light Uniforms"),
            size: std::mem::size_of::<LightBuffer>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout
        let bind_group_layout =
            gpu.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Ara Bind Group Layout"),
                    entries: &[
                        // binding 0: GlobalUniforms
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 2: palette texture array
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2Array,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // binding 3: palette sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        // binding 4: voxel SSBO
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 5: point light buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
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

        let bind_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ara Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&palette_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&palette_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: voxel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: light_buf.as_entire_binding(),
                },
            ],
        });

        // Light propagation system
        let light = LightPropagation::new(gpu, &voxel_buf, &uniform_buf, &palette_tex, 25);

        // Load shader from assets, panic if file not found (no fallback)
        let shader_src = assets::load_shader().expect("Failed to load voxel_raytracer.wgsl");

        let shader = gpu
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Ara Fullscreen Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

        let pipeline_layout =
            gpu.device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Ara Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout, light.read_bind_group_layout()],
                    push_constant_ranges: &[],
                });

        // Initialize TAA Pipeline
        // Use TaaPreset::TestExtreme for very obvious TAA effect (50% blend, no clipping)
        let taa = TaaPipeline::new(
            gpu,
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
            TaaPreset::Medium,
        );

        // Create velocity texture (RG16Float for velocity)
        let velocity_texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("Velocity Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let velocity_view = velocity_texture.create_view(&Default::default());

        let pipeline = gpu
            .device()
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Ara Render Pipeline"),
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
                    targets: &[
                        // Location 0: HDR color
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba16Float,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        // Location 1: Velocity (RG16Float)
                        Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rg16Float,
                            blend: None,
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        Self {
            surface,
            config,
            pipeline,
            bind_group,
            uniform_buf,
            light_buf,
            voxel_buf,
            palette_tex,
            light,
            post_process,
            fxaa,
            smaa,
            taa,
            final_sdr_texture,
            final_sdr_view,
            ui_system,
            aa_mode: AaMode::TaaThenSmaa, // Default to TAA + SMAA
            smaa_debug_mode: 0,
            velocity_texture,
            velocity_view,
            prev_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            frame_count: 0,
            width: width.max(1),
            height: height.max(1),
        }
    }

    /// Reconfigure the surface after a window resize.
    pub fn resize(&mut self, gpu: &GpuContext, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.width = width;
        self.height = height;
        self.surface.configure(gpu.device(), &self.config);
        self.post_process.resize(gpu, width, height);
        self.fxaa.resize(gpu.queue(), width, height);
        self.smaa.resize(gpu, width, height);
        self.taa.resize(gpu, width, height);

        // Recreate final SDR texture
        self.final_sdr_texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("Final SDR Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.final_sdr_view = self.final_sdr_texture.create_view(&Default::default());

        // Recreate velocity texture
        self.velocity_texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("Velocity Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.velocity_view = self.velocity_texture.create_view(&Default::default());

        self.ui_system.resize(gpu, width, height);
    }

    /// Render post-processing (bloom) from a specific input view.
    /// Used to pipeline TAA output directly into bloom without copy.
    fn render_post_process_from_view(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
    ) {
        // Use the post-process pipeline but with a custom input view
        // We need to manually recreate the bind groups with the input_view

        // 1. Threshold: Input -> Bloom A (half-res)
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Threshold Bind Group (TAA Input)"),
                layout: &self.post_process.threshold_pass.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.post_process.sampler),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Threshold Pass (TAA Input)"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.post_process.bloom_view,
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

            pass.set_pipeline(&self.post_process.threshold_pass.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let threshold = 1.0f32;
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
                layout: &self.post_process.blur_pass.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&self.post_process.bloom_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.post_process.sampler),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blur H Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.post_process.blur_temp_view,
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

            pass.set_pipeline(&self.post_process.blur_pass.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let dir = [1.0f32, 0.0f32];
            pass.set_push_constants(wgpu::ShaderStages::FRAGMENT, 0, bytemuck::bytes_of(&dir));
            pass.draw(0..3, 0..1);
        }

        // 3. Blur Vertical: Bloom B -> Bloom A
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Blur V Bind Group"),
                layout: &self.post_process.blur_pass.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.post_process.blur_temp_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.post_process.sampler),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Blur V Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.post_process.bloom_view,
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

            pass.set_pipeline(&self.post_process.blur_pass.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let dir = [0.0f32, 1.0f32];
            pass.set_push_constants(wgpu::ShaderStages::FRAGMENT, 0, bytemuck::bytes_of(&dir));
            pass.draw(0..3, 0..1);
        }

        // 4. Composite: Input HDR + Bloom A -> Output SDR
        {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Composite Bind Group (TAA Input)"),
                layout: &self.post_process.composite_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(input_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.post_process.sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&self.post_process.bloom_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.post_process.sampler),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Composite Pass (TAA Input)"),
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

            pass.set_pipeline(&self.post_process.composite_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            struct CompositeParams {
                intensity: f32,
                exposure: f32,
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

    /// Simple blit from final SDR texture to screen.
    /// Used when TAA mode needs to output directly without additional AA.
    fn blit_final_to_screen(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
    ) {
        // Create a simple bind group layout for blitting
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blit Bind Group Layout"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Blit Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.final_sdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Simple pass-through shader for blitting
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0)
    );
    return vec4<f32>(pos[vertex_index], 0.0, 1.0);
}

@group(0) @binding(0)
var source_tex: texture_2d<f32>;
@group(0) @binding(1)
var source_sampler: sampler;

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coord.xy / vec2<f32>(textureDimensions(source_tex));
    return textureSample(source_tex, source_sampler, uv);
}
"#
                .into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blit Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Pipeline"),
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
                    format: self.config.format,
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

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Blit Final to Screen"),
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

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Cycle SMAA Debug Mode
    /// 0=Normal, 1=UVs, 2=Edge Deltas, 3=Edge Threshold, 4=Edges Input, 5=Search Dist,
    /// 6=Area Horiz, 7=Area Vert, 8=Weights Combined, 9=Final Weights
    pub fn cycle_smaa_debug(&mut self, gpu: &GpuContext) {
        self.smaa_debug_mode = (self.smaa_debug_mode + 1) % 10;
        self.smaa.set_debug_mode(gpu.queue(), self.smaa_debug_mode);
        let mode_name = match self.smaa_debug_mode {
            0 => "Normal",
            1 => "UV Test (gradient)",
            2 => "Edge Deltas (20x boost)",
            3 => "Edge Threshold Result",
            4 => "Edges Input to Weights Pass",
            5 => "Search Distances",
            6 => "Area Weights (Horizontal)",
            7 => "Area Weights (Vertical)",
            8 => "Combined Weights",
            9 => "Final Blend Weights",
            _ => "Unknown",
        };
        println!("SMAA Debug Mode: {} ({})", self.smaa_debug_mode, mode_name);
    }

    /// Cycle TAA Debug Mode
    /// 0=Normal, 1=Velocity, 2=Neighborhood Min, 3=Neighborhood Max,
    /// 4=Raw History, 5=Clipped History, 6=Current Only
    pub fn cycle_taa_debug(&mut self, gpu: &GpuContext) {
        self.taa.cycle_debug_mode(gpu);
    }

    /// Initialize the previous view-projection matrix with the camera's initial matrix.
    /// Call this once at startup after camera is created, before the first render.
    pub fn init_prev_view_proj(&mut self, initial_view_proj: [[f32; 4]; 4]) {
        self.prev_view_proj = initial_view_proj;
    }

    /// Update the previous view-projection matrix (call after rendering).
    /// Must be called with the UNJITTERED view-projection matrix for correct velocity calculation.
    pub fn update_prev_view_proj(&mut self, unjittered_view_proj: [[f32; 4]; 4]) {
        self.prev_view_proj = unjittered_view_proj;
    }

    /// Get the previous view-projection matrix
    pub fn prev_view_proj(&self) -> [[f32; 4]; 4] {
        self.prev_view_proj
    }

    /// Get current frame count
    #[allow(dead_code)]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Render a frame: upload uniforms, draw fullscreen quad.
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        uniforms: &GlobalUniforms,
        lights: &LightBuffer,
        ui_ctx: &UiContext,
    ) -> Result<(), wgpu::SurfaceError> {
        gpu.queue()
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniforms));
        gpu.queue()
            .write_buffer(&self.light_buf, 0, bytemuck::bytes_of(lights));

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Ara Frame Encoder"),
            });

        // Propagate light before raytracing
        self.light.propagate(&mut encoder);

        // Raytrace pass: outputs HDR color + velocity
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Ara Raytrace Pass"),
                color_attachments: &[
                    // Location 0: HDR color
                    Some(wgpu::RenderPassColorAttachment {
                        view: self.post_process.hdr_view(),
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    // Location 1: Velocity
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.velocity_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_bind_group(1, self.light.current_read_bind_group(), &[]);
            pass.draw(0..3, 0..1);
        }

        match self.aa_mode {
            AaMode::Fxaa => {
                // Bloom -> Final SDR (Intermediate)
                self.post_process
                    .render(gpu.device(), &mut encoder, &self.final_sdr_view);
                // FXAA -> Screen
                self.fxaa
                    .render(gpu.device(), &mut encoder, &self.final_sdr_view, &view);
            }
            AaMode::Smaa => {
                // Bloom -> Final SDR (Intermediate)
                self.post_process
                    .render(gpu.device(), &mut encoder, &self.final_sdr_view);
                // SMAA -> Screen
                self.smaa
                    .render(gpu.device(), &mut encoder, &self.final_sdr_view, &view);
            }
            AaMode::Taa => {
                // TAA Resolve: HDR + Velocity -> TAA Output (ping-pong)
                self.taa.resolve(
                    gpu.device(),
                    gpu.queue(),
                    &mut encoder,
                    self.post_process.hdr_view(),
                    &self.velocity_view,
                );

                // Bloom: TAA Output -> Final SDR (direct, no copy)
                let taa_output = self.taa.get_current_output_view();
                self.render_post_process_from_view(
                    gpu.device(),
                    &mut encoder,
                    taa_output,
                    &self.final_sdr_view,
                );

                // Blit Final SDR -> Screen
                self.blit_final_to_screen(gpu.device(), &mut encoder, &view);
            }
            AaMode::TaaThenSmaa => {
                // TAA Resolve: HDR + Velocity -> TAA Output (ping-pong)
                self.taa.resolve(
                    gpu.device(),
                    gpu.queue(),
                    &mut encoder,
                    self.post_process.hdr_view(),
                    &self.velocity_view,
                );

                // Bloom: TAA Output -> Final SDR (direct, no copy)
                let taa_output = self.taa.get_current_output_view();
                self.render_post_process_from_view(
                    gpu.device(),
                    &mut encoder,
                    taa_output,
                    &self.final_sdr_view,
                );

                // SMAA: Final SDR -> Screen
                self.smaa
                    .render(gpu.device(), &mut encoder, &self.final_sdr_view, &view);
            }
            AaMode::None => {
                // Bloom -> Screen (Direct)
                self.post_process.render(gpu.device(), &mut encoder, &view);
            }
        }

        // Increment frame count (for TAA frame tracking)
        self.frame_count += 1;

        // Render UI on top
        self.ui_system.render(gpu, &view, &mut encoder, ui_ctx);

        gpu.queue().submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Reload the palette texture with new data.
    ///
    /// This overwrites both layers of the existing texture using `queue.write_texture()`.
    /// No pipeline or bind group recreation is needed since they already reference this texture.
    pub fn reload_palette(&self, gpu: &GpuContext, palette_data: &[u8]) {
        const LAYER_SIZE: usize = 256 * 256 * 4;

        // Upload layer 0 (albedo)
        gpu.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.palette_tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &palette_data[..LAYER_SIZE],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: Some(256),
            },
            wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
        );

        // Upload layer 1 (material properties)
        gpu.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.palette_tex,
                mip_level: 0,
                origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
                aspect: wgpu::TextureAspect::All,
            },
            &palette_data[LAYER_SIZE..],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: Some(256),
            },
            wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Update the voxel buffer with new data.
    #[allow(dead_code)]
    pub fn update_voxels(&self, gpu: &GpuContext, voxel_data: &[PackedVoxel]) {
        let voxel_bytes = bytemuck::cast_slice::<PackedVoxel, u8>(voxel_data);
        gpu.queue().write_buffer(&self.voxel_buf, 0, voxel_bytes);
    }

    /// Write a single voxel to the GPU buffer at the given linear index.
    pub fn update_voxel_at(&self, gpu: &GpuContext, index: usize, voxel: PackedVoxel) {
        let offset = (index * std::mem::size_of::<PackedVoxel>()) as u64;
        gpu.queue()
            .write_buffer(&self.voxel_buf, offset, bytemuck::bytes_of(&voxel));
    }
}
