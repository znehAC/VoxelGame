use crate::gpu::GpuContext;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct UiVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    pub mode: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct UiUniforms {
    screen_size: [f32; 2],
    pad: [f32; 2],
}

pub struct UiBatcher {
    pub vertices: Vec<UiVertex>,
    pub indices: Vec<u16>,
}

impl UiBatcher {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    pub fn push_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        let start_idx = self.vertices.len() as u16;

        self.vertices.push(UiVertex {
            position: [x, y],
            uv: [0.0, 0.0],
            color,
            mode: 1,
        });
        self.vertices.push(UiVertex {
            position: [x + w, y],
            uv: [1.0, 0.0],
            color,
            mode: 1,
        });
        self.vertices.push(UiVertex {
            position: [x + w, y + h],
            uv: [1.0, 1.0],
            color,
            mode: 1,
        });
        self.vertices.push(UiVertex {
            position: [x, y + h],
            uv: [0.0, 1.0],
            color,
            mode: 1,
        });

        self.indices.extend_from_slice(&[
            start_idx,
            start_idx + 1,
            start_idx + 2,
            start_idx,
            start_idx + 2,
            start_idx + 3,
        ]);
    }

    // TODO: Texture support requires passing texture handle or using a bound texture atlas
    // For now, let's assume one font texture for text, and maybe icons later.
}

pub struct UiContext {
    pub screen_width: f32,
    pub screen_height: f32,
    #[allow(dead_code)]
    pub selected_block_name: String,
}

pub trait UiComponent {
    #[allow(dead_code, unused_variables)]
    fn update(&mut self, ctx: &UiContext) {}
    fn draw(&self, batch: &mut UiBatcher, ctx: &UiContext);
}

// --- Components ---

pub struct CrosshairComponent;

impl UiComponent for CrosshairComponent {
    fn draw(&self, batch: &mut UiBatcher, ctx: &UiContext) {
        let cx = ctx.screen_width / 2.0;
        let cy = ctx.screen_height / 2.0;
        let size = 10.0;
        let thickness = 2.0;
        let color = [1.0, 1.0, 1.0, 0.8];

        batch.push_rect(
            cx - size / 2.0,
            cy - thickness / 2.0,
            size,
            thickness,
            color,
        );
        batch.push_rect(
            cx - thickness / 2.0,
            cy - size / 2.0,
            thickness,
            size,
            color,
        );
    }
}

pub struct HotbarComponent;

impl UiComponent for HotbarComponent {
    fn draw(&self, batch: &mut UiBatcher, ctx: &UiContext) {
        let w = 400.0;
        let h = 50.0;
        let x = (ctx.screen_width - w) / 2.0;
        let y = ctx.screen_height - h - 20.0;

        // Background
        batch.push_rect(x, y, w, h, [0.1, 0.1, 0.1, 0.5]);

        // Selection Box (just a placeholder visual)
        batch.push_rect(x + 10.0, y + 5.0, 40.0, 40.0, [1.0, 1.0, 1.0, 0.3]);
    }
}

pub struct UiSystem {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_capacity: usize,
    index_capacity: usize,
    // Store components
    components: Vec<Box<dyn UiComponent>>,
}

impl UiSystem {
    pub fn new(gpu: &GpuContext, format: wgpu::TextureFormat) -> Self {
        let shader = gpu
            .device()
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("UI Shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "../assets/shaders/ui.wgsl"
                ))),
            });

        let uniform_size = std::mem::size_of::<UiUniforms>() as u64;
        let uniform_buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Uniforms"),
            size: uniform_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Dummy texture for now (1x1 white) to satisfy binding
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("UI Dummy Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        gpu.queue().write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255, 255, 255, 255],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group_layout =
            gpu.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("UI Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });

        let bind_group = gpu.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("UI Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let layout = gpu
            .device()
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("UI Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = gpu
            .device()
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("UI Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<UiVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 8,
                                shader_location: 1,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x4,
                                offset: 16,
                                shader_location: 2,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Uint32,
                                offset: 32,
                                shader_location: 3,
                            },
                        ],
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        // Initial buffers (dynamically resized later if needed, but for now fixed size is easier)
        let vertex_capacity = 1024;
        let index_capacity = 2048;

        let vertex_buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Vertex Buffer"),
            size: (vertex_capacity * std::mem::size_of::<UiVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Index Buffer"),
            size: (index_capacity * std::mem::size_of::<u16>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let components: Vec<Box<dyn UiComponent>> =
            vec![Box::new(CrosshairComponent), Box::new(HotbarComponent)];

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            bind_group,
            vertex_capacity,
            index_capacity,
            components,
        }
    }

    pub fn resize(&mut self, gpu: &GpuContext, width: u32, height: u32) {
        let uniforms = UiUniforms {
            screen_size: [width as f32, height as f32],
            pad: [0.0, 0.0],
        };
        gpu.queue()
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    pub fn render(
        &mut self,
        gpu: &GpuContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        ctx: &UiContext,
    ) {
        let mut batch = UiBatcher::new();

        for comp in &self.components {
            comp.draw(&mut batch, ctx);
        }

        if batch.vertices.is_empty() {
            return;
        }

        // Upload buffers (resize if needed, simplified for now: panic if too small or just clamp)
        // In real engine: reallocate buffer.
        let v_bytes = bytemuck::cast_slice(&batch.vertices);
        let i_bytes = bytemuck::cast_slice(&batch.indices);

        if batch.vertices.len() > self.vertex_capacity || batch.indices.len() > self.index_capacity
        {
            log::warn!("UI batch overflow");
            // For now, just truncate or return to avoid crash
        }

        gpu.queue().write_buffer(&self.vertex_buffer, 0, v_bytes);
        gpu.queue().write_buffer(&self.index_buffer, 0, i_bytes);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("UI Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // Load existing scene
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..batch.indices.len() as u32, 0, 0..1);
    }
}
