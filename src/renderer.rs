//! Presentation layer: fullscreen-quad pipeline with palette texture, voxel SSBO, and uniforms.

use ara_core::glam::Mat4;
use ara_core::{GlobalUniforms, LightBuffer, bytemuck};

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

const BLIT_SHADER_SRC: &str = r#"
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
"#;

/// Manages surface presentation with palette-texture + voxel-SSBO pipeline.
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    light_buf: wgpu::Buffer,
    palette_tex: wgpu::Texture,
    palette_sampler: wgpu::Sampler,
    light: LightPropagation,
    post_process: BloomPipeline,
    fxaa: FxaaPipeline,
    smaa: SmaaPipeline,
    taa: TaaPipeline,
    final_sdr_texture: wgpu::Texture,
    final_sdr_view: wgpu::TextureView,
    ui_system: UiSystem,
    aa_mode: AaMode,

    // Bloom parameters
    bloom_threshold: f32,
    bloom_intensity: f32,
    bloom_exposure: f32,

    // Cached blit pipeline resources
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,
    blit_bind_group: wgpu::BindGroup,

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
        voxel_buf: &wgpu::Buffer,
        occupancy_buf: &wgpu::Buffer,
        bloom_threshold: f32,
        bloom_intensity: f32,
        bloom_exposure: f32,
        font_atlas: &[u8],
        font_atlas_w: u32,
        font_atlas_h: u32,
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
        let ui_system = UiSystem::new(gpu, format, font_atlas, font_atlas_w, font_atlas_h);

        // Initialize Blit Pipeline (cached, not recreated per frame)
        let blit_bind_group_layout =
            gpu.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let blit_sampler = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Blit Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let blit_bind_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit Bind Group"),
            layout: &blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&final_sdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&blit_sampler),
                },
            ],
        });

        let blit_shader = gpu
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Blit Shader"),
                source: wgpu::ShaderSource::Wgsl(BLIT_SHADER_SRC.into()),
            });

        let blit_pipeline_layout =
            gpu.device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Blit Pipeline Layout"),
                    bind_group_layouts: &[&blit_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let blit_pipeline = gpu
            .device()
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Blit Pipeline"),
                layout: Some(&blit_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &blit_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &blit_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
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
                        // binding 6: occupancy buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: occupancy_buf.as_entire_binding(),
                },
            ],
        });

        // Light propagation system
        let light = LightPropagation::new(gpu, &voxel_buf, &occupancy_buf, &uniform_buf, &palette_tex, 10);

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
            bind_group_layout,
            uniform_buf,
            light_buf,
            palette_tex,
            palette_sampler,
            light,
            post_process,
            fxaa,
            smaa,
            taa,
            final_sdr_texture,
            final_sdr_view,
            ui_system,
            aa_mode: AaMode::TaaThenSmaa,
            bloom_threshold,
            bloom_intensity,
            bloom_exposure,
            blit_pipeline,
            blit_bind_group_layout,
            blit_sampler,
            blit_bind_group,
            velocity_texture,
            velocity_view,
            prev_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            frame_count: 0,
            width: width.max(1),
            height: height.max(1),
        }
    }

    /// Update the voxel buffer bind group.
    pub fn update_voxel_bind_group(
        &mut self,
        device: &wgpu::Device,
        voxel_buf: &wgpu::Buffer,
        occupancy_buf: &wgpu::Buffer,
    ) {
        let palette_view = self.palette_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Ara Palette View"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ara Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&palette_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.palette_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: voxel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.light_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: occupancy_buf.as_entire_binding(),
                },
            ],
        });
        
        // Update LightPropagation with new buffers too if needed?
        // LightPropagation holds `voxel_buf` and `occupancy_buf` in its bindings?
        // Let's check LightPropagation::new.
        // It does: `self.bind_group_layout` ... `voxel_buf` ...
        // LightPropagation needs to update its bind group as well if voxel buffer changes!
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

        // Recreate blit bind group (references final_sdr_view)
        self.blit_bind_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit Bind Group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.final_sdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
            ],
        });

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

    /// Simple blit from final SDR texture to screen using cached pipeline.
    fn blit_final_to_screen(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        output_view: &wgpu::TextureView,
    ) {
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

        pass.set_pipeline(&self.blit_pipeline);
        pass.set_bind_group(0, &self.blit_bind_group, &[]);
        pass.draw(0..3, 0..1);
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

    /// Get current anti-aliasing mode.
    pub fn aa_mode(&self) -> AaMode {
        self.aa_mode
    }

    /// Set the anti-aliasing mode.
    pub fn set_aa_mode(&mut self, mode: AaMode) {
        self.aa_mode = mode;
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

        let (bt, bi, be) = (
            self.bloom_threshold,
            self.bloom_intensity,
            self.bloom_exposure,
        );

        match self.aa_mode {
            AaMode::Fxaa => {
                self.post_process.render(
                    gpu.device(),
                    &mut encoder,
                    &self.final_sdr_view,
                    None,
                    bt,
                    bi,
                    be,
                );
                self.fxaa
                    .render(gpu.device(), &mut encoder, &self.final_sdr_view, &view);
            }
            AaMode::Smaa => {
                self.post_process.render(
                    gpu.device(),
                    &mut encoder,
                    &self.final_sdr_view,
                    None,
                    bt,
                    bi,
                    be,
                );
                self.smaa
                    .render(gpu.device(), &mut encoder, &self.final_sdr_view, &view);
            }
            AaMode::Taa => {
                self.taa.resolve(
                    gpu.device(),
                    gpu.queue(),
                    &mut encoder,
                    self.post_process.hdr_view(),
                    &self.velocity_view,
                );
                let taa_output = self.taa.get_current_output_view();
                self.post_process.render(
                    gpu.device(),
                    &mut encoder,
                    &self.final_sdr_view,
                    Some(taa_output),
                    bt,
                    bi,
                    be,
                );
                self.blit_final_to_screen(&mut encoder, &view);
            }
            AaMode::TaaThenSmaa => {
                self.taa.resolve(
                    gpu.device(),
                    gpu.queue(),
                    &mut encoder,
                    self.post_process.hdr_view(),
                    &self.velocity_view,
                );
                let taa_output = self.taa.get_current_output_view();
                self.post_process.render(
                    gpu.device(),
                    &mut encoder,
                    &self.final_sdr_view,
                    Some(taa_output),
                    bt,
                    bi,
                    be,
                );
                self.smaa
                    .render(gpu.device(), &mut encoder, &self.final_sdr_view, &view);
            }
            AaMode::None => {
                self.post_process
                    .render(gpu.device(), &mut encoder, &view, None, bt, bi, be);
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

    /// Forward picker hit test to UI system.
    pub fn picker_hit_test(
        &self,
        mouse_x: f32,
        mouse_y: f32,
        screen_w: f32,
        screen_h: f32,
        block_count: usize,
    ) -> Option<u16> {
        self.ui_system
            .picker_hit_test(mouse_x, mouse_y, screen_w, screen_h, block_count)
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

}
