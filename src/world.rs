//! World streaming: GPU terrain generation + chunk management.

use ara_core::glam::{IVec3, Vec3};
use ara_core::{CHUNK_SIZE, CHUNKS_PER_AXIS, GRID_SIZE};
use log::info;

use crate::gpu::GpuContext;

const VOXEL_BUF_SIZE: u64 = (GRID_SIZE as u64) * (GRID_SIZE as u64) * (GRID_SIZE as u64) * 4;
const OCCUPANCY_COUNT: u32 = CHUNKS_PER_AXIS * CHUNKS_PER_AXIS * CHUNKS_PER_AXIS;
const OCCUPANCY_BUF_SIZE: u64 = OCCUPANCY_COUNT as u64 * 4;

/// Push constant layout matching terrain_gen.wgsl ChunkParams.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ChunkParams {
    offset_x: i32,
    offset_y: i32,
    offset_z: i32,
    pad: i32,
}

pub struct WorldManager {
    voxel_buf_a: wgpu::Buffer,
    voxel_buf_b: wgpu::Buffer,
    occupancy_buf: wgpu::Buffer,
    dirty_chunks_buf: wgpu::Buffer,
    
    terrain_pipeline: wgpu::ComputePipeline,
    terrain_bind_group_a: wgpu::BindGroup,
    terrain_bind_group_b: wgpu::BindGroup,

    physics_pipeline: wgpu::ComputePipeline,
    physics_bind_group_ab: wgpu::BindGroup,
    physics_bind_group_ba: wgpu::BindGroup,

    world_origin: IVec3,
    current_buffer_index: usize, // 0 = A, 1 = B
}

impl WorldManager {
    pub fn new(gpu: &GpuContext) -> Self {
        let device = gpu.device();

        let voxel_buf_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Voxel SSBO A"),
            size: VOXEL_BUF_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let voxel_buf_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Voxel SSBO B"),
            size: VOXEL_BUF_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let occupancy_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Occupancy SSBO"),
            size: OCCUPANCY_BUF_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Dirty chunks buffer - simple list of indices or flags
        let dirty_chunks_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Dirty Chunks SSBO"),
            size: OCCUPANCY_BUF_SIZE, // Enough for one flag/index per chunk
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- Terrain Gen Pipeline ---

        let terrain_bg_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        // --- Physics Pipeline ---

        let physics_bg_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Ara Physics BGL"),
            entries: &[
                // Binding 0: Source Voxels (Read-Only)
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
                // Binding 1: Destination Voxels (Read-Write)
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
                // Binding 2: Dirty Chunks (Read-Write)
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

        // AB: Read A, Write B
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

        // BA: Read B, Write A
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

        Self {
            voxel_buf_a,
            voxel_buf_b,
            occupancy_buf,
            dirty_chunks_buf,
            terrain_pipeline,
            terrain_bind_group_a,
            terrain_bind_group_b,
            physics_pipeline,
            physics_bind_group_ab,
            physics_bind_group_ba,
            world_origin: IVec3::ZERO,
            current_buffer_index: 0,
        }
    }

    /// Generate all chunks for the initial view centered on the player.
    pub fn generate_initial_chunks(&mut self, gpu: &GpuContext, player_pos: Vec3) {
        self.world_origin = compute_origin(player_pos);

        // Clear occupancy
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

        // Old range in chunk coordinates
        let old_chunk_min = IVec3::new(
            floor_div(old_origin.x, cs),
            floor_div(old_origin.y, cs),
            floor_div(old_origin.z, cs),
        );
        let old_chunk_max = old_chunk_min + IVec3::splat(cpa);

        // New range in chunk coordinates
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
                        // Skip chunks that were already in the old range
                        if cx >= old_chunk_min.x
                            && cx < old_chunk_max.x
                            && cy >= old_chunk_min.y
                            && cy < old_chunk_max.y
                            && cz >= old_chunk_min.z
                            && cz < old_chunk_max.z
                        {
                            continue;
                        }

                        // Clear occupancy for this chunk slot
                        let wcx = ((cx % cpa) + cpa) % cpa;
                        let wcy = ((cy % cpa) + cpa) % cpa;
                        let wcz = ((cz % cpa) + cpa) % cpa;
                        let occ_idx =
                            (wcz * cpa * cpa + wcy * cpa + wcx) as u64 * 4;
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
    ///
    /// Reads from current buffer, writes to next buffer.
    /// Swaps current buffer after dispatch.
    pub fn dispatch_physics(&mut self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ara Physics Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.physics_pipeline);
            
            // If current is A (0), we want AB (Read A, Write B).
            // If current is B (1), we want BA (Read B, Write A).
            let bg = if self.current_buffer_index == 0 {
                &self.physics_bind_group_ab
            } else {
                &self.physics_bind_group_ba
            };
            
            pass.set_bind_group(0, bg, &[]);
            
            // Dispatch 0 workgroups for now (No-op)
            pass.dispatch_workgroups(0, 0, 0);
        }

        // If we dispatched work, we would swap here.
        // For now, since we dispatch 0, we effectively do nothing.
        // But to test logic, let's NOT swap if workgroups is 0,
        // OR follow instruction "For now with 0 workgroups dispatched, buffer A stays current".
        // This implies I should NOT swap if I don't dispatch.
        // However, if I implemented the swap, I would do:
        // self.current_buffer_index = 1 - self.current_buffer_index;
        
        // Leaving swap logic commented out or guarded for now.
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
