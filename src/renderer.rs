//! Presentation layer: fullscreen-quad pipeline with palette texture, voxel SSBO, and uniforms.

use ara_core::{bytemuck, GlobalUniforms, PackedVoxel};

use crate::assets;
use crate::gpu::GpuContext;
use crate::light::LightPropagation;
use crate::postprocess::BloomPipeline;

/// Manages surface presentation with palette-texture + voxel-SSBO pipeline.
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    voxel_buf: wgpu::Buffer,
    palette_tex: wgpu::Texture,
    light: LightPropagation,
    post_process: BloomPipeline,
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
            ],
        });

        // Light propagation system
        let light = LightPropagation::new(gpu, &voxel_buf, &palette_tex, 25);

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
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float, // HDR target
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
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
            voxel_buf,
            palette_tex,
            light,
            post_process,
        }
    }

    /// Reconfigure the surface after a window resize.
    pub fn resize(&mut self, gpu: &GpuContext, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(gpu.device(), &self.config);
        self.post_process.resize(gpu, width, height);
    }

    /// Render a frame: upload uniforms, draw fullscreen quad.
    pub fn render(
        &mut self,
        gpu: &GpuContext,
        uniforms: &GlobalUniforms,
    ) -> Result<(), wgpu::SurfaceError> {
        gpu.queue()
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(uniforms));

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            gpu.device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Ara Frame Encoder"),
                });

        // Propagate light before raytracing
        self.light.propagate(&mut encoder);

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Ara Raytrace Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.post_process.hdr_view(), // Render to HDR texture
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
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_bind_group(1, self.light.current_read_bind_group(), &[]);
            pass.draw(0..3, 0..1);
        }

        // Run Bloom + Tone Mapping -> Swapchain
        self.post_process.render(gpu.device(), &mut encoder, &view);

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
