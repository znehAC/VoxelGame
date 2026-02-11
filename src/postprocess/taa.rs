//! Temporal Anti-Aliasing (TAA) Pipeline
//!
//! Implements standard TAA with:
//! - History double-buffering (ping-pong)
//! - Neighborhood clamping in YCoCg space
//! - Velocity-based reprojection
//! - Optional sharpening

use crate::gpu::GpuContext;
use ara_core::{TaaUniforms, bytemuck};

/// TAA Quality Presets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum TaaPreset {
    /// Low quality - minimal ghosting, faster
    Low,
    Medium,
    /// High quality - better stability, slightly more blur
    High,
}

impl TaaPreset {
    /// Get configuration values for this preset
    pub fn config(&self) -> TaaConfig {
        match self {
            TaaPreset::Low => TaaConfig {
                blend_alpha: 0.6, // High responsiveness, more shimmer
                enable_sharpening: true,
                use_variance_clamp: true,
                use_ycocg: true,
            },
            TaaPreset::Medium => TaaConfig {
                blend_alpha: 0.8, // High responsiveness, more shimmer
                enable_sharpening: true,
                use_variance_clamp: true,
                use_ycocg: true,
            },
            TaaPreset::High => TaaConfig {
                blend_alpha: 0.95, // High stability, more ghosting
                enable_sharpening: true,
                use_variance_clamp: true,
                use_ycocg: true,
            },
        }
    }
}

/// TAA Configuration Parameters
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct TaaConfig {
    /// Blend factor between current and history (0.0-1.0)
    /// Lower = more temporal accumulation (less flicker, more blur)
    pub blend_alpha: f32,
    /// Enable sharpening to counteract TAA blur
    pub enable_sharpening: bool,
    /// Use variance clipping (prevents ghosting)
    pub use_variance_clamp: bool,
    /// Use YCoCg color space for neighborhood clamping
    pub use_ycocg: bool,
}

impl TaaConfig {
    /// Pack config flags into a u32 for GPU
    fn packed_flags(&self) -> u32 {
        let mut flags = 0u32;
        if self.use_variance_clamp {
            flags |= 1;
        }
        if self.use_ycocg {
            flags |= 2;
        }
        flags
    }
}

impl Default for TaaConfig {
    fn default() -> Self {
        TaaPreset::High.config()
    }
}

/// Temporal Anti-Aliasing Pipeline
///
/// Uses a ping-pong buffer strategy:
/// - Two targets: targets[0] and targets[1]
/// - Read from targets[frame % 2] (previous frame's output)
/// - Write to targets[(frame + 1) % 2] (current frame's output)
pub struct TaaPipeline {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,

    /// Ping-pong targets: each frame we read from one and write to the other
    targets: [wgpu::Texture; 2],
    target_views: [wgpu::TextureView; 2],

    /// Current frame index for ping-pong calculation
    frame_count: u64,

    // Samplers
    sampler_linear: wgpu::Sampler,
    sampler_point: wgpu::Sampler,

    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    config: TaaConfig,
    debug_mode: u32,
}

impl TaaPipeline {
    /// Create a new TAA pipeline with the specified preset
    pub fn new(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        preset: TaaPreset,
    ) -> Self {
        Self::with_config(gpu, width, height, format, preset.config())
    }

    /// Create a new TAA pipeline with custom configuration
    pub fn with_config(
        gpu: &GpuContext,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        config: TaaConfig,
    ) -> Self {
        let device = gpu.device();

        // Load TAA shader
        let shader_source = include_str!("../../assets/shaders/taa_resolve.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TAA Resolve Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create samplers
        let sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TAA Linear Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let sampler_point = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TAA Point Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // Uniform buffer for TAA parameters
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TAA Uniforms"),
            size: 64, // 4 vec4s: screen_size + params + flags + padding
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TAA Bind Group Layout"),
            entries: &[
                // Binding 0: Current color texture
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
                // Binding 1: History texture
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
                // Binding 2: Velocity texture
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Binding 3: Linear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Binding 4: Point sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // Binding 5: Uniforms (4 vec4s = 64 bytes)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(std::num::NonZeroU64::new(64).unwrap()),
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TAA Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TAA Resolve Pipeline"),
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
                    format, // HDR output format (Rgba16Float)
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

        // Create placeholder textures (will be resized)
        let targets = [
            create_texture(device, 1, 1, format, "TAA Target 0"),
            create_texture(device, 1, 1, format, "TAA Target 1"),
        ];
        let target_views = [
            targets[0].create_view(&Default::default()),
            targets[1].create_view(&Default::default()),
        ];

        let mut pipeline = Self {
            pipeline,
            bind_group_layout,
            uniform_buf,
            targets,
            target_views,
            frame_count: 0,
            sampler_linear,
            sampler_point,
            width: 1,
            height: 1,
            format,
            config,
            debug_mode: 0,
        };

        pipeline.resize(gpu, width, height);

        // Clear both history buffers to black on creation
        pipeline.clear_history(gpu);

        pipeline
    }

    /// Clear both history buffers to black (for startup and resize)
    fn clear_history(&self, gpu: &GpuContext) {
        let device = gpu.device();
        let queue = gpu.queue();

        // Create a command encoder for clearing
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("TAA Clear History"),
        });

        // Clear both ping-pong targets
        for i in 0..2 {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(&format!("TAA Clear Target {}", i)),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.target_views[i],
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
            // Render pass ends when `pass` is dropped
            drop(pass);
        }

        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Resize all TAA textures
    pub fn resize(&mut self, gpu: &GpuContext, width: u32, height: u32) {
        let device = gpu.device();

        self.width = width;
        self.height = height;

        // Recreate ping-pong targets
        self.targets = [
            create_texture(device, width, height, self.format, "TAA Target 0"),
            create_texture(device, width, height, self.format, "TAA Target 1"),
        ];
        self.target_views = [
            self.targets[0].create_view(&Default::default()),
            self.targets[1].create_view(&Default::default()),
        ];

        // Clear new textures to black (they contain garbage)
        self.clear_history(gpu);

        // Reset frame count to signal "no valid history"
        self.frame_count = 0;

        // Update uniform buffer with new screen size (no valid history after resize)
        self.update_uniforms(gpu.queue(), false);
    }

    /// Update uniform buffer
    fn update_uniforms(&self, queue: &wgpu::Queue, has_valid_history: bool) {
        let uniforms = TaaUniforms::new(
            self.width,
            self.height,
            self.config.blend_alpha,
            self.config.enable_sharpening,
            self.debug_mode,
            has_valid_history,
            self.config.use_variance_clamp,
            self.config.use_ycocg,
        );

        // Debug output when blend_alpha changes or debug mode is 8
        if self.debug_mode == 8 || (self.frame_count <= 3 && self.frame_count > 0) {
            eprintln!(
                "[TAA Debug] frame={}, blend_alpha={}, debug_mode={}, has_valid_history={}",
                self.frame_count, uniforms.params.x, uniforms.params.z, has_valid_history
            );
        }

        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Get the view for reading history (targets[frame % 2])
    fn history_read_view(&self) -> &wgpu::TextureView {
        &self.target_views[self.frame_count as usize % 2]
    }

    /// Get the view for writing output (targets[(frame + 1) % 2])
    fn output_write_view(&self) -> &wgpu::TextureView {
        &self.target_views[(self.frame_count as usize + 1) % 2]
    }

    /// Get the current output view (the one we just wrote to).
    /// Call this after resolve() to get the TAA result for downstream passes.
    pub fn get_current_output_view(&self) -> &wgpu::TextureView {
        // After resolve(), we wrote to (frame + 1) % 2, so that's our output
        &self.target_views[(self.frame_count as usize + 1) % 2]
    }

    /// Get the current output texture
    pub fn get_current_output_texture(&self) -> &wgpu::Texture {
        &self.targets[(self.frame_count as usize + 1) % 2]
    }

    /// Cycle debug mode
    pub fn cycle_debug_mode(&mut self, gpu: &GpuContext) {
        let old_mode = self.debug_mode;
        self.debug_mode = (self.debug_mode + 1) % 13; // 0-12 debug modes

        // If switching FROM a debug mode (1-6) TO normal mode (0), clear history
        // because debug visualization colors may have corrupted the history buffer
        if old_mode != 0 && self.debug_mode == 0 {
            println!("Clearing TAA history after exiting debug mode...");
            self.clear_history(gpu);
            self.frame_count = 0; // Reset frame count to invalidate history
        }

        // If entering Raw History mode (4), clear history first to get a clean view
        // This prevents confusion between old corrupted history and actual reprojection
        if old_mode != 4 && self.debug_mode == 4 {
            println!("Clearing TAA history for clean Raw History view...");
            self.clear_history(gpu);
            self.frame_count = 0;
        }

        let has_valid_history = self.frame_count >= 2;
        self.update_uniforms(gpu.queue(), has_valid_history);

        let mode_name = match self.debug_mode {
            0 => "Normal",
            1 => "Velocity",
            2 => "Neighborhood Min",
            3 => "Neighborhood Max",
            4 => "Raw History",
            5 => "Clipped History",
            6 => "Current Only (no TAA)",
            7 => "Validity (Green=OK, Red=OOB, Blue=NoHist)",
            8 => "FORCE BLEND (50/50, ignore valid_history)",
            9 => "HasValidHistory Flag (Green=True, Red=False)",
            10 => "HistorySampleValid Flag (Green=True, Red=False)",
            11 => "ValidHistory Combined (Green=True, Red=False)",
            12 => "OUTPUT RED (Test if writes work)",
            _ => "Unknown",
        };
        println!("TAA Debug Mode: {} ({})", self.debug_mode, mode_name);
    }

    /// Set debug mode directly
    #[allow(dead_code)]
    pub fn set_debug_mode(&mut self, gpu: &GpuContext, mode: u32) {
        self.debug_mode = mode;
        let has_valid_history = self.frame_count >= 2;
        self.update_uniforms(gpu.queue(), has_valid_history);
    }

    /// Get current debug mode
    #[allow(dead_code)]
    pub fn debug_mode(&self) -> u32 {
        self.debug_mode
    }

    /// Execute TAA resolve pass using ping-pong strategy
    ///
    /// Inputs:
    /// - current_view: The current frame HDR color (from raytracer)
    /// - velocity_view: The velocity buffer
    ///
    /// The result is written to targets[(frame + 1) % 2].
    /// Use `get_current_output_view()` to retrieve the result.
    pub fn resolve(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        current_view: &wgpu::TextureView,
        velocity_view: &wgpu::TextureView,
    ) {
        // Increment frame count FIRST so read/write indices are correct.
        // This ensures get_current_output_view() returns the texture we just wrote to.
        self.frame_count += 1;

        // Check if we have valid history (frame_count >= 2 means both targets have been written)
        // frame_count starts at 0, after first resolve it's 1 (wrote to target 1)
        // After second resolve it's 2 (wrote to target 0), now we have valid history
        let _has_valid_history = self.frame_count >= 2;

        // Update uniforms for this frame (IMPORTANT: must happen before creating bind group)
        self.update_uniforms(_queue, _has_valid_history);

        // Ping-pong: read from targets[frame % 2], write to targets[(frame + 1) % 2]
        let history_read = self.history_read_view();
        let output_write = self.output_write_view();

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TAA Resolve Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(current_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(history_read),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(velocity_view),
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
                    resource: self.uniform_buf.as_entire_binding(),
                },
            ],
        });

        // TAA Resolve pass - writes to output target
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("TAA Resolve Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output_write,
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

        // No copy needed! The ping-pong targets automatically serve as history for next frame.
        // Next frame, we'll read from the target we just wrote to.
    }
}

/// Create a texture suitable for TAA (render attachment + sample)
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
