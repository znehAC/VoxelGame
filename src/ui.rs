//! UI system: vertex batching, text rendering, and overlay components.

use crate::font::BitmapFont;
use crate::gpu::GpuContext;
use ara_core::BlockRegistry;
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

    /// Emit a single textured quad (mode=2 for text alpha mask).
    fn push_text_quad(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        uv: (f32, f32, f32, f32),
        color: [f32; 4],
    ) {
        let start_idx = self.vertices.len() as u16;
        let (u0, v0, u1, v1) = uv;

        self.vertices.push(UiVertex {
            position: [x, y],
            uv: [u0, v0],
            color,
            mode: 2,
        });
        self.vertices.push(UiVertex {
            position: [x + w, y],
            uv: [u1, v0],
            color,
            mode: 2,
        });
        self.vertices.push(UiVertex {
            position: [x + w, y + h],
            uv: [u1, v1],
            color,
            mode: 2,
        });
        self.vertices.push(UiVertex {
            position: [x, y + h],
            uv: [u0, v1],
            color,
            mode: 2,
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

    /// Render a text string at (x, y). Returns total width in pixels.
    pub fn push_text(
        &mut self,
        text: &str,
        x: f32,
        y: f32,
        color: [f32; 4],
        scale: f32,
    ) -> f32 {
        let gw = crate::font::GLYPH_W as f32 * scale;
        let gh = crate::font::GLYPH_H as f32 * scale;
        let mut cx = x;

        for &byte in text.as_bytes() {
            let uv = BitmapFont::glyph_uv(byte);
            self.push_text_quad(cx, y, gw, gh, uv, color);
            cx += gw;
        }

        cx - x
    }
}

/// Hotbar state: 9 slots mapping to block material IDs.
pub struct HotbarState {
    pub slots: [u16; 9],
    pub selected: usize,
}

impl HotbarState {
    pub fn new() -> Self {
        Self {
            slots: [1, 2, 3, 4, 5, 6, 7, 8, 0],
            selected: 0,
        }
    }

    pub fn selected_block_id(&self) -> u16 {
        self.slots[self.selected]
    }
}

/// Block picker overlay state.
pub struct BlockPickerState {
    pub visible: bool,
}

impl BlockPickerState {
    pub fn new() -> Self {
        Self { visible: false }
    }
}

/// All data needed by UI components each frame.
pub struct UiContext<'a> {
    pub screen_width: f32,
    pub screen_height: f32,
    pub debug_visible: bool,
    pub fps: f32,
    pub frame_time_ms: f32,
    pub camera_pos: [f32; 3],
    pub world_origin: [i32; 3],
    pub aa_mode_name: &'a str,
    pub hotbar: &'a HotbarState,
    pub registry: &'a BlockRegistry,
    pub picker_visible: bool,
    pub brush_size: u32,
    pub interaction_mode: &'a str,
}

// --- Components ---

fn draw_crosshair(batch: &mut UiBatcher, ctx: &UiContext) {
    let cx = ctx.screen_width / 2.0;
    let cy = ctx.screen_height / 2.0;
    let size = 10.0;
    let thickness = 2.0;
    let color = [1.0, 1.0, 1.0, 0.8];

    batch.push_rect(cx - size / 2.0, cy - thickness / 2.0, size, thickness, color);
    batch.push_rect(cx - thickness / 2.0, cy - size / 2.0, thickness, size, color);

    // Temporary test string
    // batch.push_text("Bitmap Font Works!", cx + 10.0, cy + 10.0, [0.0, 1.0, 0.0, 1.0], 1.0);
}

fn draw_debug_hud(batch: &mut UiBatcher, ctx: &UiContext) {
    if !ctx.debug_visible {
        return;
    }

    let scale = 1.0_f32;
    let line_h = 16.0 * scale + 2.0;
    let pad = 6.0;
    let x = pad;
    let mut y = pad;
    let text_color = [1.0, 1.0, 1.0, 1.0];
    let bg_color = [0.0, 0.0, 0.0, 0.5];

    let lines: [arrayvec::ArrayString<64>; 6] = [
        fmt_line64(&format_args!(
            "FPS: {:.0} ({:.1}ms)",
            ctx.fps, ctx.frame_time_ms
        )),
        fmt_line64(&format_args!(
            "Pos: {:.1}  {:.1}  {:.1}",
            ctx.camera_pos[0], ctx.camera_pos[1], ctx.camera_pos[2]
        )),
        fmt_line64(&format_args!(
            "Origin: {}  {}  {}",
            ctx.world_origin[0], ctx.world_origin[1], ctx.world_origin[2]
        )),
        fmt_line64(&format_args!("AA: {} | Brush: {}", ctx.aa_mode_name, ctx.brush_size)),
        fmt_line64(&format_args!("Mode: {}", ctx.interaction_mode)),
        fmt_line64(&format_args!(
            "Block: {}",
            ctx.registry
                .get_block_name(ctx.hotbar.selected_block_id())
                .unwrap_or("?")
        )),
    ];

    let max_chars = lines.iter().map(|l| l.len()).max().unwrap_or(0);
    let bg_w = max_chars as f32 * 8.0 * scale + pad * 2.0;
    let bg_h = lines.len() as f32 * line_h + pad * 2.0;
    batch.push_rect(0.0, 0.0, bg_w, bg_h, bg_color);

    for line in &lines {
        batch.push_text(line, x, y, text_color, scale);
        y += line_h;
    }
}

fn draw_hotbar(batch: &mut UiBatcher, ctx: &UiContext) {
    let slot_size = 48.0;
    let gap = 4.0;
    let total_w = 9.0 * slot_size + 8.0 * gap;
    let hx = (ctx.screen_width - total_w) / 2.0;
    let hy = ctx.screen_height - slot_size - 20.0;

    // Background
    batch.push_rect(
        hx - 4.0,
        hy - 4.0,
        total_w + 8.0,
        slot_size + 8.0,
        [0.1, 0.1, 0.1, 0.6],
    );

    for i in 0..9 {
        let sx = hx + i as f32 * (slot_size + gap);
        let block_id = ctx.hotbar.slots[i];

        // Slot color from registry
        let color = block_color_f32(ctx.registry, block_id);
        batch.push_rect(sx, hy, slot_size, slot_size, color);

        // Selection border
        if i == ctx.hotbar.selected {
            let b = 2.0;
            let sc = [1.0, 1.0, 1.0, 0.9];
            batch.push_rect(sx - b, hy - b, slot_size + 2.0 * b, b, sc);
            batch.push_rect(sx - b, hy + slot_size, slot_size + 2.0 * b, b, sc);
            batch.push_rect(sx - b, hy, b, slot_size, sc);
            batch.push_rect(sx + slot_size, hy, b, slot_size, sc);
        }

        // Slot number
        let num = format!("{}", i + 1);
        batch.push_text(&num, sx + 2.0, hy + 2.0, [1.0, 1.0, 1.0, 0.6], 1.0);
    }

    // Selected block name
    let name = ctx
        .registry
        .get_block_name(ctx.hotbar.selected_block_id())
        .unwrap_or("Air");
    let name_w = name.len() as f32 * 8.0;
    let name_x = (ctx.screen_width - name_w) / 2.0;
    batch.push_text(name, name_x, hy - 20.0, [1.0, 1.0, 1.0, 0.9], 1.0);
}

fn draw_block_picker(batch: &mut UiBatcher, ctx: &UiContext) {
    if !ctx.picker_visible {
        return;
    }

    // Fullscreen overlay
    batch.push_rect(
        0.0,
        0.0,
        ctx.screen_width,
        ctx.screen_height,
        [0.0, 0.0, 0.0, 0.7],
    );

    let block_count = ctx.registry.block_count() as usize;
    let swatch_size = 56.0;
    let gap = 8.0;
    let text_h = 18.0;
    let cell_h = swatch_size + text_h + gap;
    let cols = 4usize;
    let grid_w = cols as f32 * (swatch_size + gap) - gap;
    let rows = (block_count + cols - 1) / cols;
    let grid_h = rows as f32 * cell_h;
    let ox = (ctx.screen_width - grid_w) / 2.0;
    let oy = (ctx.screen_height - grid_h) / 2.0;

    // Title
    let title = "Block Picker  [E/ESC to close]";
    let tw = title.len() as f32 * 8.0;
    batch.push_text(
        title,
        (ctx.screen_width - tw) / 2.0,
        oy - 30.0,
        [1.0, 1.0, 1.0, 1.0],
        1.0,
    );

    for idx in 0..block_count {
        let col = idx % cols;
        let row = idx / cols;
        let sx = ox + col as f32 * (swatch_size + gap);
        let sy = oy + row as f32 * cell_h;

        let block_id = idx as u16;
        let color = block_color_f32(ctx.registry, block_id);
        batch.push_rect(sx, sy, swatch_size, swatch_size, color);

        let name = ctx.registry.get_block_name(block_id).unwrap_or("?");
        let name_w = name.len() as f32 * 8.0;
        let nx = sx + (swatch_size - name_w) / 2.0;
        batch.push_text(name, nx, sy + swatch_size + 2.0, [1.0, 1.0, 1.0, 0.8], 1.0);
    }
}

fn draw_brush_indicator(batch: &mut UiBatcher, ctx: &UiContext) {
    if ctx.brush_size <= 1 {
        return;
    }
    let cx = ctx.screen_width / 2.0;
    let cy = ctx.screen_height / 2.0;
    let label = format!("{}x", ctx.brush_size);
    batch.push_text(
        &label,
        cx + 12.0,
        cy - 4.0,
        [1.0, 1.0, 0.5, 0.8],
        1.0,
    );
}

/// Helper: get block sRGB color as [f32; 4] for UI display.
fn block_color_f32(registry: &BlockRegistry, block_id: u16) -> [f32; 4] {
    if block_id == 0 {
        return [0.2, 0.2, 0.2, 0.5];
    }
    let c = registry.get_block_color_f32(block_id);
    [c[0], c[1], c[2], 1.0]
}

/// Small formatting helper (no heap alloc for common case).
fn fmt_line64(args: &std::fmt::Arguments) -> arrayvec::ArrayString<64> {
    use std::fmt::Write;
    let mut s = arrayvec::ArrayString::<64>::new();
    let _ = write!(s, "{}", args);
    s
}

// --- GPU UI System ---

pub struct UiSystem {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_capacity: usize,
    index_capacity: usize,
}

impl UiSystem {
    pub fn new(
        gpu: &GpuContext,
        format: wgpu::TextureFormat,
        atlas_data: &[u8],
        atlas_w: u32,
        atlas_h: u32,
    ) -> Self {
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

        // Font atlas texture
        let texture = gpu.device().create_texture(&wgpu::TextureDescriptor {
            label: Some("UI Font Atlas"),
            size: wgpu::Extent3d {
                width: atlas_w,
                height: atlas_h,
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
            atlas_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_w * 4),
                rows_per_image: Some(atlas_h),
            },
            wgpu::Extent3d {
                width: atlas_w,
                height: atlas_h,
                depth_or_array_layers: 1,
            },
        );

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = gpu.device().create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
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

        let vertex_capacity = 4096;
        let index_capacity = 8192;

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

        Self {
            pipeline,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            bind_group,
            vertex_capacity,
            index_capacity,
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

        draw_crosshair(&mut batch, ctx);
        draw_debug_hud(&mut batch, ctx);
        draw_brush_indicator(&mut batch, ctx);
        draw_hotbar(&mut batch, ctx);
        draw_block_picker(&mut batch, ctx);

        if batch.vertices.is_empty() {
            return;
        }

        if batch.vertices.len() > self.vertex_capacity || batch.indices.len() > self.index_capacity
        {
            log::warn!(
                "UI batch overflow: {}v/{}i (max {}v/{}i)",
                batch.vertices.len(),
                batch.indices.len(),
                self.vertex_capacity,
                self.index_capacity
            );
            batch.vertices.truncate(self.vertex_capacity);
            batch.indices.truncate(self.index_capacity);
        }

        let v_bytes = bytemuck::cast_slice(&batch.vertices);
        let i_bytes = bytemuck::cast_slice(&batch.indices);

        gpu.queue().write_buffer(&self.vertex_buffer, 0, v_bytes);
        gpu.queue().write_buffer(&self.index_buffer, 0, i_bytes);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("UI Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
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

    /// Handle a click in the block picker. Returns the selected block ID if a swatch was clicked.
    pub fn picker_hit_test(
        &self,
        mouse_x: f32,
        mouse_y: f32,
        screen_w: f32,
        screen_h: f32,
        block_count: usize,
    ) -> Option<u16> {
        let swatch_size = 56.0;
        let gap = 8.0;
        let text_h = 18.0;
        let cell_h = swatch_size + text_h + gap;
        let cols = 4usize;
        let grid_w = cols as f32 * (swatch_size + gap) - gap;
        let rows = (block_count + cols - 1) / cols;
        let grid_h = rows as f32 * cell_h;
        let ox = (screen_w - grid_w) / 2.0;
        let oy = (screen_h - grid_h) / 2.0;

        for idx in 0..block_count {
            let col = idx % cols;
            let row = idx / cols;
            let sx = ox + col as f32 * (swatch_size + gap);
            let sy = oy + row as f32 * cell_h;

            if mouse_x >= sx
                && mouse_x <= sx + swatch_size
                && mouse_y >= sy
                && mouse_y <= sy + swatch_size
            {
                return Some(idx as u16);
            }
        }

        None
    }
}
