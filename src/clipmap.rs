//! Voxel clipmap — 6 concentric toroidal 512³ cascades for far-field geometry.
//!
//! Cascade cell sizes: 2, 4, 8, 16, 32, 64 voxels → extent ~51m … 1.6km.
//! Format: Rgba8Uint — R=material_id, G=normal_idx, B=density, A=unused.

use ara_core::{bytemuck, glam::Vec3};

use crate::gpu::GpuContext;

const CASCADE_COUNT: usize = 6;
const GRID_SIZE: u32 = 512;
const HALF_GRID: i32 = 256;
pub const CELL_SIZES: [i32; CASCADE_COUNT] = [2, 4, 8, 16, 32, 64];

/// GPU-side uniform layout (must match WGSL `ClipOrigin`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ClipOrigin {
    origin: [i32; 3],
    cell_size: i32,
}

/// Push constants for clipmap_build.wgsl (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BuildParams {
    origin_x: i32,
    origin_y: i32,
    origin_z: i32,
    cell_size: i32,
    mode: u32,
    face_val: i32,
    _pad0: u32,
    _pad1: u32,
}

/// Six concentric toroidal clipmap cascades for far-field voxel geometry.
pub struct VoxelClipmaps {
    #[allow(dead_code)] // kept alive to own the textures; views borrow from them
    cascades: [wgpu::Texture; CASCADE_COUNT],
    cascade_views: [wgpu::TextureView; CASCADE_COUNT],
    centers: [[i32; 3]; CASCADE_COUNT],
    origins_buf: wgpu::Buffer,
    build_pipeline: wgpu::ComputePipeline,
    build_bgl: wgpu::BindGroupLayout,
}

impl VoxelClipmaps {
    pub fn new(gpu: &GpuContext) -> Self {
        let device = gpu.device();

        let cascades = std::array::from_fn(|i| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Clipmap Cascade {i}")),
                size: wgpu::Extent3d { width: GRID_SIZE, height: GRID_SIZE, depth_or_array_layers: GRID_SIZE },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba8Uint,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            })
        });
        let cascade_views = std::array::from_fn(|i| cascades[i].create_view(&Default::default()));

        let origins_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Clipmap Origins"),
            size: (CASCADE_COUNT * std::mem::size_of::<ClipOrigin>()) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let build_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Clipmap Build BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Uint,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
                count: None,
            }],
        });

        let build_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Clipmap Build Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../assets/shaders/clipmap_build.wgsl").into(),
            ),
        });

        let build_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Clipmap Build Layout"),
            bind_group_layouts: &[&build_bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..32,
            }],
        });

        let build_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clipmap Build Pipeline"),
            layout: Some(&build_layout),
            module: &build_shader,
            entry_point: Some("build_clipmap"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            cascades,
            cascade_views,
            centers: [[0; 3]; CASCADE_COUNT],
            origins_buf,
            build_pipeline,
            build_bgl,
        }
    }

    /// Full build of all 6 cascades. Call once at startup.
    pub fn build_full(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, cam_pos: Vec3) {
        for (i, &cs) in CELL_SIZES.iter().enumerate() {
            self.centers[i] = [
                (cam_pos.x / cs as f32).floor() as i32,
                (cam_pos.y / cs as f32).floor() as i32,
                (cam_pos.z / cs as f32).floor() as i32,
            ];
        }
        self.write_origins_buf(gpu);

        for i in 0..CASCADE_COUNT {
            self.dispatch_build(gpu.device(), encoder, i, None);
        }
    }

    /// Per-frame ring update — dispatches face updates for cascades whose origin shifted.
    pub fn update(&mut self, gpu: &GpuContext, encoder: &mut wgpu::CommandEncoder, cam_pos: Vec3) {
        let mut dirty = false;
        for i in 0..CASCADE_COUNT {
            let cs = CELL_SIZES[i];
            let new_center = [
                (cam_pos.x / cs as f32).floor() as i32,
                (cam_pos.y / cs as f32).floor() as i32,
                (cam_pos.z / cs as f32).floor() as i32,
            ];
            if new_center == self.centers[i] {
                continue;
            }

            let old_center = self.centers[i];
            let needs_full = (0..3).any(|a| (new_center[a] - old_center[a]).abs() >= HALF_GRID);
            self.centers[i] = new_center;
            dirty = true;

            if needs_full {
                self.dispatch_build(gpu.device(), encoder, i, None);
            } else {
                for axis in 0..3usize {
                    let delta = new_center[axis] - old_center[axis];
                    if delta == 0 { continue; }
                    for k in 0..delta.abs() {
                        let face_val = if delta > 0 {
                            new_center[axis] + HALF_GRID - 1 - k
                        } else {
                            new_center[axis] - HALF_GRID + k
                        };
                        self.dispatch_build(gpu.device(), encoder, i, Some((axis as u32, face_val)));
                    }
                }
            }
        }

        if dirty {
            self.write_origins_buf(gpu);
        }
    }

    pub fn cascade_views(&self) -> &[wgpu::TextureView; CASCADE_COUNT] {
        &self.cascade_views
    }

    pub fn origins_buf(&self) -> &wgpu::Buffer {
        &self.origins_buf
    }

    fn write_origins_buf(&self, gpu: &GpuContext) {
        let origins: [ClipOrigin; CASCADE_COUNT] = std::array::from_fn(|i| ClipOrigin {
            origin: self.centers[i],
            cell_size: CELL_SIZES[i],
        });
        gpu.queue().write_buffer(&self.origins_buf, 0, bytemuck::cast_slice(&origins));
    }

    /// `face`: None = full rebuild, Some((axis, face_val)) = single ring-face update.
    fn dispatch_build(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        cascade_idx: usize,
        face: Option<(u32, i32)>,
    ) {
        let cs = CELL_SIZES[cascade_idx];
        let [ox, oy, oz] = self.centers[cascade_idx];

        let (mode, face_val, dispatch) = match face {
            None => (3u32, 0i32, (64, 64, GRID_SIZE)),
            Some((axis, fv)) => {
                let d = (64, 64, 1);
                (axis, fv, d)
            }
        };

        let pc = BuildParams {
            origin_x: ox, origin_y: oy, origin_z: oz,
            cell_size: cs,
            mode,
            face_val,
            _pad0: 0, _pad1: 0,
        };

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Clipmap Build BG"),
            layout: &self.build_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&self.cascade_views[cascade_idx]),
            }],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Clipmap Build Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.build_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_push_constants(0, bytemuck::bytes_of(&pc));
        pass.dispatch_workgroups(dispatch.0, dispatch.1, dispatch.2);
    }
}
