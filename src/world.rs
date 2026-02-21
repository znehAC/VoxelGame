//! World streaming: GPU terrain generation + chunk management.

use std::sync::mpsc;
use std::sync::Arc;

use ara_core::glam::{IVec3, Vec3};
use ara_core::{CHUNK_SIZE, CHUNKS_PER_AXIS, GRID_SIZE};
use log::info;

use crate::gpu::GpuContext;

const VOXEL_BUF_SIZE: u64 = (GRID_SIZE as u64) * (GRID_SIZE as u64) * (GRID_SIZE as u64) * 4;
const OCCUPANCY_COUNT: u32 = CHUNKS_PER_AXIS * CHUNKS_PER_AXIS * CHUNKS_PER_AXIS;
const OCCUPANCY_BUF_SIZE: u64 = OCCUPANCY_COUNT as u64 * 4;

const MAX_PACK_CHUNKS: u32 = 16;
const CHUNK_VOL: u32 = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
const PACK_BUF_SIZE: u64 = MAX_PACK_CHUNKS as u64 * CHUNK_VOL as u64 * 4;

/// Push constant layout matching terrain_gen.wgsl ChunkParams.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ChunkParams {
    offset_x: i32,
    offset_y: i32,
    offset_z: i32,
    pad: i32,
}

/// Push constant layout matching brush_compute.wgsl BrushParams.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BrushParams {
    center_x: i32,
    center_y: i32,
    center_z: i32,
    radius: f32,
    material: u32,
    shape: u32,
    pad1: u32,
    pad2: u32,
}

/// Push constant layout matching chunk_pack.wgsl PackParams.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackParams {
    origin_x: u32,
    origin_y: u32,
    origin_z: u32,
    chunk_index: u32,
}

struct ReadbackTask {
    map_rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>,
    staging_buf: Arc<wgpu::Buffer>,
    data_len: usize,
    chunks: Vec<IVec3>,
}

struct ReadbackResult {
    packed_data: Vec<u32>,
    chunks: Vec<IVec3>,
}

pub struct WorldManager {
    voxel_buf_a: wgpu::Buffer,
    voxel_buf_b: wgpu::Buffer,
    occupancy_buf: wgpu::Buffer,
    dirty_chunks_buf: wgpu::Buffer,

    terrain_pipeline: wgpu::ComputePipeline,
    terrain_bind_group_a: wgpu::BindGroup,
    terrain_bind_group_b: wgpu::BindGroup,

    brush_pipeline: wgpu::ComputePipeline,

    physics_pipeline: wgpu::ComputePipeline,
    physics_bind_group_ab: wgpu::BindGroup,
    physics_bind_group_ba: wgpu::BindGroup,

    // Chunk pack readback
    pack_pipeline: wgpu::ComputePipeline,
    pack_bind_group_a: wgpu::BindGroup,
    pack_bind_group_b: wgpu::BindGroup,
    pack_buf: wgpu::Buffer,
    pack_staging_buf: Arc<wgpu::Buffer>,

    world_origin: IVec3,
    current_buffer_index: usize,

    // Worker thread communication
    worker_task_tx: mpsc::Sender<ReadbackTask>,
    worker_result_rx: mpsc::Receiver<ReadbackResult>,
    worker_handle: Option<std::thread::JoinHandle<()>>,
    readback_in_flight: bool,
    readback_queued: bool,
    dirty_readback_chunks: Vec<IVec3>,
}

impl WorldManager {
    pub fn new(gpu: &GpuContext) -> Self {
        let device = gpu.device();

        let voxel_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Voxel SSBO A"),
            size: VOXEL_BUF_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let voxel_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Voxel SSBO B"),
            size: VOXEL_BUF_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let occupancy_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Occupancy SSBO"),
            size: OCCUPANCY_BUF_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let dirty_chunks_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Dirty Chunks SSBO"),
            size: OCCUPANCY_BUF_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Terrain Gen Pipeline ---

        let terrain_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Ara Terrain Gen BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let terrain_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ara Terrain Gen BG A"),
            layout: &terrain_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: voxel_buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: occupancy_buf.as_entire_binding(),
                },
            ],
        });

        let terrain_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ara Terrain Gen BG B"),
            layout: &terrain_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: voxel_buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: occupancy_buf.as_entire_binding(),
                },
            ],
        });

        let terrain_shader_src = include_str!("../assets/shaders/terrain_gen.wgsl");
        let terrain_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ara Terrain Gen Shader"),
            source: wgpu::ShaderSource::Wgsl(terrain_shader_src.into()),
        });

        let terrain_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ara Terrain Gen Pipeline Layout"),
            bind_group_layouts: &[&terrain_bg_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<ChunkParams>() as u32,
            }],
        });

        let terrain_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ara Terrain Gen Pipeline"),
            layout: Some(&terrain_layout),
            module: &terrain_shader,
            entry_point: Some("generate"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Brush Pipeline ---
        let brush_shader_src = include_str!("../assets/shaders/brush_compute.wgsl");
        let brush_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ara Brush Shader"),
            source: wgpu::ShaderSource::Wgsl(brush_shader_src.into()),
        });

        let brush_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ara Brush Pipeline Layout"),
            bind_group_layouts: &[&terrain_bg_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<BrushParams>() as u32,
            }],
        });

        let brush_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ara Brush Pipeline"),
            layout: Some(&brush_layout),
            module: &brush_shader,
            entry_point: Some("apply_brush"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Physics Pipeline ---

        let physics_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Ara Physics BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let physics_bind_group_ab = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ara Physics BG AB"),
            layout: &physics_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: voxel_buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: voxel_buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dirty_chunks_buf.as_entire_binding(),
                },
            ],
        });

        let physics_bind_group_ba = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ara Physics BG BA"),
            layout: &physics_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: voxel_buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: voxel_buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dirty_chunks_buf.as_entire_binding(),
                },
            ],
        });

        let physics_shader_src = include_str!("../assets/shaders/physics_sim.wgsl");
        let physics_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ara Physics Shader"),
            source: wgpu::ShaderSource::Wgsl(physics_shader_src.into()),
        });

        let physics_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ara Physics Pipeline Layout"),
            bind_group_layouts: &[&physics_bg_layout],
            push_constant_ranges: &[],
        });

        let physics_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ara Physics Pipeline"),
            layout: Some(&physics_layout),
            module: &physics_shader,
            entry_point: Some("simulate"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Chunk Pack Pipeline ---

        let pack_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Ara Chunk Pack BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pack_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Chunk Pack SSBO"),
            size: PACK_BUF_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let pack_staging_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Chunk Pack Staging"),
            size: PACK_BUF_SIZE,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let pack_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ara Chunk Pack BG A"),
            layout: &pack_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: voxel_buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pack_buf.as_entire_binding(),
                },
            ],
        });

        let pack_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ara Chunk Pack BG B"),
            layout: &pack_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: voxel_buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pack_buf.as_entire_binding(),
                },
            ],
        });

        let pack_shader_src = include_str!("../assets/shaders/chunk_pack.wgsl");
        let pack_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ara Chunk Pack Shader"),
            source: wgpu::ShaderSource::Wgsl(pack_shader_src.into()),
        });

        let pack_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ara Chunk Pack Pipeline Layout"),
            bind_group_layouts: &[&pack_bg_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PackParams>() as u32,
            }],
        });

        let pack_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Ara Chunk Pack Pipeline"),
            layout: Some(&pack_layout),
            module: &pack_shader,
            entry_point: Some("pack_chunk"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- Readback Worker Thread ---

        let (task_tx, task_rx) = mpsc::channel::<ReadbackTask>();
        let (result_tx, result_rx) = mpsc::channel::<ReadbackResult>();
        let worker_device = gpu.device_arc();

        let worker_handle = std::thread::Builder::new()
            .name("readback-worker".into())
            .spawn(move || {
                readback_worker(worker_device, task_rx, result_tx);
            })
            .expect("Failed to spawn readback worker thread");

        Self {
            voxel_buf_a,
            voxel_buf_b,
            occupancy_buf,
            dirty_chunks_buf,
            terrain_pipeline,
            terrain_bind_group_a,
            terrain_bind_group_b,
            brush_pipeline,
            physics_pipeline,
            physics_bind_group_ab,
            physics_bind_group_ba,
            pack_pipeline,
            pack_bind_group_a,
            pack_bind_group_b,
            pack_buf,
            pack_staging_buf,
            world_origin: IVec3::ZERO,
            current_buffer_index: 0,
            worker_task_tx: task_tx,
            worker_result_rx: result_rx,
            worker_handle: Some(worker_handle),
            readback_in_flight: false,
            readback_queued: false,
            dirty_readback_chunks: Vec::new(),
        }
    }

    /// Generate all chunks for the initial view centered on the player.
    pub fn generate_initial_chunks(&mut self, gpu: &GpuContext, player_pos: Vec3) {
        self.world_origin = compute_origin(player_pos);

        let zeros = vec![0u8; OCCUPANCY_BUF_SIZE as usize];
        gpu.queue().write_buffer(&self.occupancy_buf, 0, &zeros);

        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Ara Initial Terrain Gen"),
            });

        let cs = CHUNK_SIZE as i32;
        let cpa = CHUNKS_PER_AXIS as i32;
        let bg = if self.current_buffer_index == 0 {
            &self.terrain_bind_group_a
        } else {
            &self.terrain_bind_group_b
        };

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ara Initial Terrain Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.terrain_pipeline);
            pass.set_bind_group(0, bg, &[]);

            for cz in 0..cpa {
                for cy in 0..cpa {
                    for cx in 0..cpa {
                        let offset = IVec3::new(
                            self.world_origin.x + cx * cs,
                            self.world_origin.y + cy * cs,
                            self.world_origin.z + cz * cs,
                        );

                        let params = ChunkParams {
                            offset_x: offset.x,
                            offset_y: offset.y,
                            offset_z: offset.z,
                            pad: 0,
                        };

                        pass.set_push_constants(0, bytemuck::bytes_of(&params));
                        pass.dispatch_workgroups(8, 8, 8);
                    }
                }
            }
        }

        gpu.queue().submit(std::iter::once(encoder.finish()));
        info!(
            "Generated initial terrain at origin {:?}",
            self.world_origin
        );
    }

    /// Update the visible region as the player moves. Returns true if origin changed.
    pub fn update_view(&mut self, gpu: &GpuContext, player_pos: Vec3) -> bool {
        let new_origin = compute_origin(player_pos);
        if new_origin == self.world_origin {
            return false;
        }

        let old_origin = self.world_origin;
        let cs = CHUNK_SIZE as i32;
        let cpa = CHUNKS_PER_AXIS as i32;

        let old_chunk_min = IVec3::new(
            floor_div(old_origin.x, cs),
            floor_div(old_origin.y, cs),
            floor_div(old_origin.z, cs),
        );
        let old_chunk_max = old_chunk_min + IVec3::splat(cpa);

        let new_chunk_min = IVec3::new(
            floor_div(new_origin.x, cs),
            floor_div(new_origin.y, cs),
            floor_div(new_origin.z, cs),
        );
        let new_chunk_max = new_chunk_min + IVec3::splat(cpa);

        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Ara Terrain Streaming"),
            });

        let mut new_chunk_count = 0u32;
        let bg = if self.current_buffer_index == 0 {
            &self.terrain_bind_group_a
        } else {
            &self.terrain_bind_group_b
        };

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ara Terrain Streaming Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.terrain_pipeline);
            pass.set_bind_group(0, bg, &[]);

            for cz in new_chunk_min.z..new_chunk_max.z {
                for cy in new_chunk_min.y..new_chunk_max.y {
                    for cx in new_chunk_min.x..new_chunk_max.x {
                        if cx >= old_chunk_min.x
                            && cx < old_chunk_max.x
                            && cy >= old_chunk_min.y
                            && cy < old_chunk_max.y
                            && cz >= old_chunk_min.z
                            && cz < old_chunk_max.z
                        {
                            continue;
                        }

                        let wcx = ((cx % cpa) + cpa) % cpa;
                        let wcy = ((cy % cpa) + cpa) % cpa;
                        let wcz = ((cz % cpa) + cpa) % cpa;
                        let occ_idx = (wcz * cpa * cpa + wcy * cpa + wcx) as u64 * 4;
                        gpu.queue()
                            .write_buffer(&self.occupancy_buf, occ_idx, &[0u8; 4]);

                        let params = ChunkParams {
                            offset_x: cx * cs,
                            offset_y: cy * cs,
                            offset_z: cz * cs,
                            pad: 0,
                        };

                        pass.set_push_constants(0, bytemuck::bytes_of(&params));
                        pass.dispatch_workgroups(8, 8, 8);
                        new_chunk_count += 1;
                        self.dirty_readback_chunks.push(IVec3::new(cx, cy, cz));
                    }
                }
            }
        }

        gpu.queue().submit(std::iter::once(encoder.finish()));
        self.world_origin = new_origin;

        if new_chunk_count > 0 {
            info!(
                "Streamed {} chunks, origin now {:?}",
                new_chunk_count, self.world_origin
            );
        }

        true
    }

    /// Dispatch physics simulation.
    pub fn dispatch_physics(&mut self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ara Physics Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.physics_pipeline);

            let bg = if self.current_buffer_index == 0 {
                &self.physics_bind_group_ab
            } else {
                &self.physics_bind_group_ba
            };

            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(0, 0, 0);
        }
    }

    pub fn voxel_buf(&self) -> &wgpu::Buffer {
        if self.current_buffer_index == 0 {
            &self.voxel_buf_a
        } else {
            &self.voxel_buf_b
        }
    }

    pub fn occupancy_buf(&self) -> &wgpu::Buffer {
        &self.occupancy_buf
    }

    /// World origin as [f32; 3] for GlobalUniforms.
    pub fn world_origin(&self) -> [f32; 3] {
        [
            self.world_origin.x as f32,
            self.world_origin.y as f32,
            self.world_origin.z as f32,
        ]
    }

    /// World origin as IVec3.
    pub fn world_origin_ivec3(&self) -> IVec3 {
        self.world_origin
    }

    /// Dispatch compute shader modification over a target volume.
    pub fn dispatch_brush(
        &mut self,
        gpu: &GpuContext,
        center: IVec3,
        radius: f32,
        material: u32,
        is_sphere: bool,
    ) {
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Ara Brush Compute Encoder"),
            });

        let bg = if self.current_buffer_index == 0 {
            &self.terrain_bind_group_a
        } else {
            &self.terrain_bind_group_b
        };

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ara Brush Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.brush_pipeline);
            pass.set_bind_group(0, bg, &[]);

            let r_int = if is_sphere {
                radius as i32
            } else {
                ((radius as i32) - 1) / 2
            };

            let width = (r_int * 2 + 1) as u32;

            let params = BrushParams {
                center_x: center.x,
                center_y: center.y,
                center_z: center.z,
                radius,
                material,
                shape: if is_sphere { 1 } else { 0 },
                pad1: 0,
                pad2: 0,
            };

            pass.set_push_constants(0, bytemuck::bytes_of(&params));

            let wg_count = (width + 3) / 4;
            pass.dispatch_workgroups(wg_count, wg_count, wg_count);
        }

        gpu.queue().submit(std::iter::once(encoder.finish()));

        let cs = CHUNK_SIZE as f32;
        let c_min = IVec3::new(
            f32::floor((center.x as f32 - radius) / cs) as i32,
            f32::floor((center.y as f32 - radius) / cs) as i32,
            f32::floor((center.z as f32 - radius) / cs) as i32,
        );
        let c_max = IVec3::new(
            f32::floor((center.x as f32 + radius) / cs) as i32,
            f32::floor((center.y as f32 + radius) / cs) as i32,
            f32::floor((center.z as f32 + radius) / cs) as i32,
        );

        for cz in c_min.z..=c_max.z {
            for cy in c_min.y..=c_max.y {
                for cx in c_min.x..=c_max.x {
                    self.dirty_readback_chunks.push(IVec3::new(cx, cy, cz));
                }
            }
        }
    }

    /// Initiate async packed readback of dirty chunks. Returns immediately.
    pub fn request_readback(&mut self, gpu: &GpuContext, full_copy: bool) {
        self.readback_queued = true;
        if self.readback_in_flight {
            return;
        }
        self.readback_queued = false;

        if full_copy {
            // Full copy uses dedicated temporary path (see initial_blocking_readback)
            return;
        }

        let chunks = std::mem::take(&mut self.dirty_readback_chunks);
        if chunks.is_empty() {
            return;
        }

        // Deduplicate chunks
        let mut unique_chunks: Vec<IVec3> = Vec::with_capacity(chunks.len());
        for c in &chunks {
            if !unique_chunks.contains(c) {
                unique_chunks.push(*c);
            }
        }

        // Cap at MAX_PACK_CHUNKS per readback; overflow stays queued for next frame
        if unique_chunks.len() > MAX_PACK_CHUNKS as usize {
            let overflow = unique_chunks.split_off(MAX_PACK_CHUNKS as usize);
            self.dirty_readback_chunks = overflow;
            self.readback_queued = true;
        }

        let chunk_count = unique_chunks.len() as u32;
        let cpa = CHUNKS_PER_AXIS as i32;

        // Dispatch pack compute shader per chunk
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Ara Pack Readback Encoder"),
            });

        let pack_bg = if self.current_buffer_index == 0 {
            &self.pack_bind_group_a
        } else {
            &self.pack_bind_group_b
        };

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ara Chunk Pack Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pack_pipeline);
            pass.set_bind_group(0, pack_bg, &[]);

            for (i, chunk) in unique_chunks.iter().enumerate() {
                let wcx = (((chunk.x % cpa) + cpa) % cpa) as u32;
                let wcy = (((chunk.y % cpa) + cpa) % cpa) as u32;
                let wcz = (((chunk.z % cpa) + cpa) % cpa) as u32;

                let params = PackParams {
                    origin_x: wcx * CHUNK_SIZE,
                    origin_y: wcy * CHUNK_SIZE,
                    origin_z: wcz * CHUNK_SIZE,
                    chunk_index: i as u32,
                };

                pass.set_push_constants(0, bytemuck::bytes_of(&params));
                pass.dispatch_workgroups(8, 8, 8);
            }
        }

        // DMA: pack_buf → pack_staging_buf (only the used portion)
        let copy_size = chunk_count as u64 * CHUNK_VOL as u64 * 4;
        encoder.copy_buffer_to_buffer(&self.pack_buf, 0, &self.pack_staging_buf, 0, copy_size);

        gpu.queue().submit(std::iter::once(encoder.finish()));

        // Map the staging buffer asynchronously
        let slice = self.pack_staging_buf.slice(..copy_size);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Send task to worker thread
        let task = ReadbackTask {
            map_rx: rx,
            staging_buf: Arc::clone(&self.pack_staging_buf),
            data_len: copy_size as usize,
            chunks: unique_chunks,
        };

        if self.worker_task_tx.send(task).is_ok() {
            self.readback_in_flight = true;
        }
    }

    /// Check if worker completed readback. Scatter packed chunks into `voxels_cpu`.
    pub fn poll_readback(&mut self, gpu: &GpuContext, voxels_cpu: &mut [u32]) -> bool {
        if let Ok(result) = self.worker_result_rx.try_recv() {
            self.readback_in_flight = false;

            let grid_size = (CHUNK_SIZE * CHUNKS_PER_AXIS) as usize;
            let cs = CHUNK_SIZE as usize;
            let cpa = CHUNKS_PER_AXIS as i32;
            let chunk_vol = CHUNK_VOL as usize;

            for (i, chunk) in result.chunks.iter().enumerate() {
                let wcx = (((chunk.x % cpa) + cpa) % cpa) as usize;
                let wcy = (((chunk.y % cpa) + cpa) % cpa) as usize;
                let wcz = (((chunk.z % cpa) + cpa) % cpa) as usize;

                let start_x = wcx * cs;
                let start_y = wcy * cs;
                let start_z = wcz * cs;

                let pack_offset = i * chunk_vol;

                for z in 0..cs {
                    for y in 0..cs {
                        let dst_idx = (start_z + z) * grid_size * grid_size
                            + (start_y + y) * grid_size
                            + start_x;
                        let src_idx = pack_offset + z * cs * cs + y * cs;

                        voxels_cpu[dst_idx..dst_idx + cs]
                            .copy_from_slice(&result.packed_data[src_idx..src_idx + cs]);
                    }
                }
            }

            if self.readback_queued {
                self.request_readback(gpu, false);
            }
            return true;
        }

        if !self.readback_in_flight && self.readback_queued {
            self.request_readback(gpu, false);
        }
        false
    }

    /// Blocking initialization readback for application start.
    pub fn initial_blocking_readback(&mut self, gpu: &GpuContext, voxels_cpu: &mut [u32]) {
        let staging_buf = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Initial Readback Staging"),
            size: VOXEL_BUF_SIZE,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buf = self.voxel_buf();
        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Ara Initial Readback Encoder"),
            });
        encoder.copy_buffer_to_buffer(buf, 0, &staging_buf, 0, VOXEL_BUF_SIZE);
        gpu.queue().submit(std::iter::once(encoder.finish()));

        let slice = staging_buf.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        gpu.device().poll(wgpu::Maintain::Wait);

        rx.recv()
            .expect("Map async channel closed")
            .expect("Map async failed");

        {
            let data = staging_buf.slice(..).get_mapped_range();
            let gpu_voxels: &[u32] = bytemuck::cast_slice(&data);
            voxels_cpu.copy_from_slice(gpu_voxels);
        }

        staging_buf.unmap();
        // staging_buf dropped here, freeing 512MB
    }
}

impl Drop for WorldManager {
    fn drop(&mut self) {
        // Drop the sender to signal the worker thread to exit
        // (done implicitly when self is dropped, but we take the handle first)
        drop(std::mem::replace(
            &mut self.worker_task_tx,
            mpsc::channel().0,
        ));
        if let Some(handle) = self.worker_handle.take() {
            let _ = handle.join();
        }
    }
}

fn readback_worker(
    device: Arc<wgpu::Device>,
    task_rx: mpsc::Receiver<ReadbackTask>,
    result_tx: mpsc::Sender<ReadbackResult>,
) {
    while let Ok(task) = task_rx.recv() {
        // Poll device until map_async completes
        loop {
            device.poll(wgpu::Maintain::Poll);
            match task.map_rx.try_recv() {
                Ok(Ok(())) => break,
                Ok(Err(e)) => {
                    log::error!("Readback map_async failed: {e}");
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }

        let slice = task.staging_buf.slice(..task.data_len as u64);
        let data = slice.get_mapped_range();
        let packed_data: Vec<u32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        task.staging_buf.unmap();

        let _ = result_tx.send(ReadbackResult {
            packed_data,
            chunks: task.chunks,
        });
    }
}

fn floor_div(a: i32, b: i32) -> i32 {
    let d = a / b;
    let r = a % b;
    if (r != 0) && ((r ^ b) < 0) { d - 1 } else { d }
}

/// Compute world_origin so the atlas is centered around the player's chunk.
fn compute_origin(pos: Vec3) -> IVec3 {
    let cs = CHUNK_SIZE as i32;
    let half = (CHUNKS_PER_AXIS as i32 / 2) * cs;
    IVec3::new(
        floor_div(pos.x as i32 - half, cs) * cs,
        floor_div(pos.y as i32 - half, cs) * cs,
        floor_div(pos.z as i32 - half, cs) * cs,
    )
}
