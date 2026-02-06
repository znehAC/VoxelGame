//! Presentation layer: fullscreen-quad pipeline with palette texture, voxel SSBO, and uniforms.

use ara_core::{bytemuck, GlobalUniforms, PackedVoxel};

use crate::assets;
use crate::gpu::GpuContext;

/// Cornflower blue clear color.
const CLEAR_COLOR: wgpu::Color = wgpu::Color {
    r: 0.392,
    g: 0.584,
    b: 0.929,
    a: 1.0,
};

/// Fallback shader used if asset loading fails at startup.
const FALLBACK_SHADER: &str = r#"
struct GlobalUniforms {
    view_inverse: mat4x4f,
    proj_inverse: mat4x4f,
    cam_pos: vec4f,
    time: f32,
    resolution_x: f32,
    resolution_y: f32,
    _pad: f32,
}

struct VoxelData {
    id: u32,
    state: u32,
    flags: u32,
}

struct HitResult {
    hit: bool,
    pos: vec3f,
    normal: vec3f,
    voxel: VoxelData,
}

const GRID_SIZE: u32 = 64u;
const MAX_STEPS: u32 = 256u;
const SUN_DIR: vec3f = vec3f(0.318, -0.557, 0.239); // normalized(0.4, -0.7, 0.3)

@group(0) @binding(0) var<uniform> globals: GlobalUniforms;
@group(0) @binding(2) var t_palette: texture_2d_array<f32>;
@group(0) @binding(3) var s_palette: sampler;
@group(0) @binding(4) var<storage, read> voxels: array<u32>;

fn unpack_voxel(packed: u32) -> VoxelData {
    return VoxelData(
        packed & 0xFFFFu,
        (packed >> 16u) & 0xFFu,
        (packed >> 24u) & 0xFFu,
    );
}

fn voxel_index(x: i32, y: i32, z: i32) -> u32 {
    return u32(z) * GRID_SIZE * GRID_SIZE + u32(y) * GRID_SIZE + u32(x);
}

fn ray_aabb(origin: vec3f, inv_dir: vec3f, box_min: vec3f, box_max: vec3f) -> vec2f {
    let t0 = (box_min - origin) * inv_dir;
    let t1 = (box_max - origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_enter = max(max(tmin.x, tmin.y), tmin.z);
    let t_exit = min(min(tmax.x, tmax.y), tmax.z);
    return vec2f(t_enter, t_exit);
}

fn dda_march(origin: vec3f, dir: vec3f) -> HitResult {
    var result: HitResult;
    result.hit = false;

    let inv_dir = 1.0 / dir;
    let bounds = ray_aabb(origin, inv_dir, vec3f(0.0), vec3f(f32(GRID_SIZE)));
    var t_enter = bounds.x;
    let t_exit = bounds.y;

    if t_enter > t_exit || t_exit < 0.0 {
        return result;
    }

    // Advance ray to grid entry point
    t_enter = max(t_enter, 0.0);
    let entry = origin + dir * (t_enter + 0.001);

    // DDA setup
    var cell = vec3i(floor(entry));
    cell = clamp(cell, vec3i(0), vec3i(i32(GRID_SIZE) - 1));

    let step = vec3i(sign(dir));
    let t_delta = abs(1.0 / dir);

    // Distance to next cell boundary along each axis
    var t_max: vec3f;
    if dir.x > 0.0 { t_max.x = (f32(cell.x + 1) - entry.x) * abs(inv_dir.x); }
    else { t_max.x = (entry.x - f32(cell.x)) * abs(inv_dir.x); }
    if dir.y > 0.0 { t_max.y = (f32(cell.y + 1) - entry.y) * abs(inv_dir.y); }
    else { t_max.y = (entry.y - f32(cell.y)) * abs(inv_dir.y); }
    if dir.z > 0.0 { t_max.z = (f32(cell.z + 1) - entry.z) * abs(inv_dir.z); }
    else { t_max.z = (entry.z - f32(cell.z)) * abs(inv_dir.z); }

    // Check starting cell
    var packed = voxels[voxel_index(cell.x, cell.y, cell.z)];
    if (packed & 0xFFFFu) != 0u {
        result.hit = true;
        result.pos = entry;
        result.normal = -vec3f(f32(step.x), f32(step.y), f32(step.z));
        result.normal = normalize(result.normal);
        result.voxel = unpack_voxel(packed);
        return result;
    }

    var last_axis = 0u;
    for (var i = 0u; i < MAX_STEPS; i++) {
        // Step along the axis with smallest t_max
        if t_max.x < t_max.y {
            if t_max.x < t_max.z {
                cell.x += step.x;
                t_max.x += t_delta.x;
                last_axis = 0u;
            } else {
                cell.z += step.z;
                t_max.z += t_delta.z;
                last_axis = 2u;
            }
        } else {
            if t_max.y < t_max.z {
                cell.y += step.y;
                t_max.y += t_delta.y;
                last_axis = 1u;
            } else {
                cell.z += step.z;
                t_max.z += t_delta.z;
                last_axis = 2u;
            }
        }

        // Bounds check
        if cell.x < 0 || cell.x >= i32(GRID_SIZE) ||
           cell.y < 0 || cell.y >= i32(GRID_SIZE) ||
           cell.z < 0 || cell.z >= i32(GRID_SIZE) {
            break;
        }

        packed = voxels[voxel_index(cell.x, cell.y, cell.z)];
        if (packed & 0xFFFFu) != 0u {
            result.hit = true;

            // Compute normal from last axis crossed
            var normal = vec3f(0.0);
            if last_axis == 0u { normal.x = -f32(step.x); }
            else if last_axis == 1u { normal.y = -f32(step.y); }
            else { normal.z = -f32(step.z); }
            result.normal = normal;

            // Compute hit position from t_max
            var t_hit: f32;
            if last_axis == 0u { t_hit = t_max.x - t_delta.x; }
            else if last_axis == 1u { t_hit = t_max.y - t_delta.y; }
            else { t_hit = t_max.z - t_delta.z; }
            result.pos = entry + dir * t_hit;

            result.voxel = unpack_voxel(packed);
            return result;
        }
    }

    return result;
}

fn shade_pbr(hit: HitResult, dir: vec3f) -> vec3f {
    let pu = hit.voxel.id % 256u;
    let pv = hit.voxel.id / 256u;
    let coords = vec2i(i32(pu), i32(pv));

    // Palette layer 0: albedo (sRGB decoded by hardware)
    let albedo = textureLoad(t_palette, coords, 0, 0).rgb;

    // Palette layer 1: material properties
    // Hardware decodes sRGB→linear, but these values are linear data stored in sRGB texture.
    // Undo the hardware conversion: apply sRGB encoding (pow 1/2.2) to recover original values.
    let props_raw = textureLoad(t_palette, coords, 1, 0);
    let props = pow(props_raw, vec4f(1.0 / 2.2));
    let roughness = props.r;
    let emission = props.g;
    let metallic = props.a;

    let n = hit.normal;
    let l = normalize(-SUN_DIR); // toward the light
    let v = -dir;               // toward the viewer
    let h = normalize(l + v);

    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);

    let light_color = vec3f(1.3, 1.2, 1.0);

    // Diffuse: Lambertian
    let diffuse = albedo * n_dot_l * light_color;

    // Specular: Blinn-Phong with roughness-derived shininess
    let shininess = max(2.0 / (roughness * roughness + 0.001) - 2.0, 1.0);
    let spec_strength = pow(n_dot_h, shininess);
    // F0: blend between dielectric (0.04) and albedo based on metallic
    let f0 = mix(vec3f(0.04), albedo, metallic);
    let specular = f0 * spec_strength * n_dot_l * light_color;

    // Ambient: bluish sky-fill
    let ambient = albedo * vec3f(0.15, 0.17, 0.25);

    // Emission: lava glow
    let emit = albedo * emission * 4.0;

    let color = diffuse + specular + ambient + emit;

    // Reinhard tone mapping
    return color / (color + vec3f(1.0));
}

fn sky(dir: vec3f) -> vec3f {
    // Y-down: -dir.y is up
    let t = clamp(-dir.y, 0.0, 1.0);
    let horizon = vec3f(0.7, 0.75, 0.85);
    let zenith = vec3f(0.25, 0.45, 0.9);
    return mix(horizon, zenith, t);
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) % 2) * 4.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    // Pixel → NDC
    let ndc_x = (frag_coord.x / globals.resolution_x) * 2.0 - 1.0;
    let ndc_y = (frag_coord.y / globals.resolution_y) * 2.0 - 1.0;

    // NDC → clip → camera space via inverse projection
    let clip = vec4f(ndc_x, ndc_y, 1.0, 1.0);
    let cam_space = globals.proj_inverse * clip;
    let cam_dir = normalize(cam_space.xyz / cam_space.w);

    // Camera space → world space direction (w=0 for direction)
    let world_dir4 = globals.view_inverse * vec4f(cam_dir, 0.0);
    let dir = normalize(world_dir4.xyz);

    let origin = globals.cam_pos.xyz;
    let hit = dda_march(origin, dir);

    var color: vec3f;
    if hit.hit {
        color = shade_pbr(hit, dir);
    } else {
        color = sky(dir);
    }

    return vec4f(color, 1.0);
}
"#;

/// Manages surface presentation with palette-texture + voxel-SSBO pipeline.
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buf: wgpu::Buffer,
    palette_tex: wgpu::Texture,
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

        // Load shader from assets, fall back to embedded if file not found
        let shader_src = assets::load_shader().unwrap_or_else(|e| {
            log::warn!("Using fallback shader: {e}");
            FALLBACK_SHADER.to_string()
        });
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
                    bind_group_layouts: &[&bind_group_layout],
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
                        format,
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
            palette_tex,
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
    }

    /// Render a frame: upload uniforms, draw fullscreen quad.
    pub fn render(
        &self,
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

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Ara Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(CLEAR_COLOR),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        gpu.queue().submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Current surface format.
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
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
