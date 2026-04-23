//! CPU-side spatial streaming coordinator.
//!
//! Detects camera movement, evicts out-of-range bricks, and enqueues terrain
//! generation jobs for newly visible bricks. The actual voxel data is written
//! by the GPU terrain shader.

use std::collections::VecDeque;

use ara_core::glam::{IVec3, Vec3};
use ara_core::LOD_COUNT;

use crate::brick_map::BrickMap;
use crate::gpu::GpuContext;
use crate::terrain_pipeline::{TerrainGenJob, TerrainPipeline};

const HALF_GRID: i32 = 32; // TOP_GRID_SIZE / 2

/// CPU mirror of terrain_gen.wgsl height_at() — must stay identical.
fn height_at(wx: i32, wz: i32) -> i32 {
    use std::f32::consts::TAU;
    let x = wx as f32;
    let z = wz as f32;
    let h = 64.0
        + (x / 3200.0 * TAU).sin() * (z / 3200.0 * TAU).cos() * 12.0
        + ((x + z) / 800.0 * TAU).sin() * 4.0
        + (x / 160.0 * TAU).sin() * (z / 160.0 * TAU).sin() * 1.0;
    h as i32
}

/// Returns false only for bricks entirely above the terrain (pure air).
/// Underground and surface bricks are always generated.
fn brick_may_contain_terrain(lod: u32, coord: [i32; 3]) -> bool {
    let vs = 1i32 << lod as i32;
    let brick_y_min = coord[1] * 8 * vs;
    let brick_y_max = brick_y_min + 8 * vs;

    let x0 = coord[0] * 8 * vs;
    let x1 = x0 + 8 * vs - 1;
    let z0 = coord[2] * 8 * vs;
    let z1 = z0 + 8 * vs - 1;

    let heights = [
        height_at(x0, z0),
        height_at(x1, z0),
        height_at(x0, z1),
        height_at(x1, z1),
        height_at((x0 + x1) / 2, (z0 + z1) / 2),
    ];
    let h_max = heights.iter().copied().max().unwrap();
    let h_min = heights.iter().copied().min().unwrap();

    brick_y_min <= h_max + 8 * vs && brick_y_max >= h_min - 8 * vs
}

pub struct ChunkStreamer {
    pending_jobs: VecDeque<(u32, [i32; 3])>,
    max_jobs_per_frame: u32,
}

impl ChunkStreamer {
    pub fn new(max_jobs_per_frame: u32) -> Self {
        Self { pending_jobs: VecDeque::new(), max_jobs_per_frame }
    }

    /// Fill all LOD grids around `spawn_pos` and dispatch batched terrain gen passes.
    /// Intended to be called once at startup; startup latency is acceptable.
    pub fn initial_load(
        &mut self,
        spawn_pos: Vec3,
        sbm: &mut BrickMap,
        terrain: &TerrainPipeline,
        gpu: &GpuContext,
    ) {
        for lod in 0..LOD_COUNT {
            let lod_brick = lod_brick_from_world(spawn_pos, lod);
            sbm.set_lod_origin(lod, lod_brick.to_array());
            self.enqueue_full_grid(lod, lod_brick);
        }

        let mut total_submitted = 0usize;
        let mut total_allocated = 0usize;
        let batch_size = terrain.max_jobs();
        while !self.pending_jobs.is_empty() {
            let count = self.pending_jobs.len().min(batch_size);
            let batch: Vec<_> = self.pending_jobs.drain(..count).collect();
            total_submitted += batch.len();
            let gpu_jobs = self.allocate_jobs(&batch, sbm, gpu);
            total_allocated += gpu_jobs.len();
            if !gpu_jobs.is_empty() {
                let mut encoder = gpu.device().create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("Initial Load Batch") },
                );
                terrain.generate(gpu, &mut encoder, &gpu_jobs, sbm);
                gpu.queue().submit(std::iter::once(encoder.finish()));
            }
        }
        sbm.reveal_pending(gpu);
        log::info!(
            "Initial load: {}/{} bricks allocated, pool at {:.1}%",
            total_allocated, total_submitted, sbm.utilization() * 100.0
        );
    }

    /// Per-frame update: detect camera drift, evict stale bricks, enqueue new ones, dispatch GPU jobs.
    pub fn update(
        &mut self,
        cam_pos: Vec3,
        sbm: &mut BrickMap,
        terrain: &TerrainPipeline,
        encoder: &mut wgpu::CommandEncoder,
        gpu: &GpuContext,
    ) {
        // Make last frame's generated bricks visible before any evictions/allocations.
        sbm.reveal_pending(gpu);

        for lod in 0..LOD_COUNT {
            let desired = lod_brick_from_world(cam_pos, lod);
            for axis in 0..3 {
                // Re-read origin each axis so diagonal shifts use the already-updated value.
                let current = IVec3::from(sbm.lod_origin(lod));
                let d = (desired - current)[axis];
                if d == 0 {
                    continue;
                }
                let shift = d.signum();
                let new_origin = {
                    let mut o = current;
                    o[axis] += shift;
                    o
                };

                let evict_face = if shift > 0 {
                    current[axis] - HALF_GRID
                } else {
                    current[axis] + HALF_GRID - 1
                };
                let load_face = if shift > 0 {
                    new_origin[axis] + HALF_GRID - 1
                } else {
                    new_origin[axis] - HALF_GRID
                };

                self.evict_face(lod, axis, evict_face, current, sbm, gpu);
                self.enqueue_face(lod, axis, load_face, new_origin);
                sbm.set_lod_origin(lod, new_origin.to_array());
            }
        }

        // Drain pending queue up to budget
        let count = self.pending_jobs.len().min(self.max_jobs_per_frame as usize);
        let batch: Vec<_> = self.pending_jobs.drain(..count).collect();
        if !batch.is_empty() {
            let gpu_jobs = self.allocate_jobs(&batch, sbm, gpu);
            terrain.generate(gpu, encoder, &gpu_jobs, sbm);
        }

        if sbm.utilization() > 0.85 {
            log::warn!("Brick pool near capacity ({:.0}%)", sbm.utilization() * 100.0);
        }
    }

    // ---- internal helpers ----

    fn evict_face(
        &self,
        lod: u32,
        axis: usize,
        face_coord: i32,
        origin: IVec3,
        sbm: &mut BrickMap,
        gpu: &GpuContext,
    ) {
        let (u_range, v_range) = face_plane_ranges(axis, origin);
        for u in u_range.0..=u_range.1 {
            for v in v_range.0..=v_range.1 {
                let coord = make_coord(axis, face_coord, u, v);
                sbm.evict_brick(gpu, lod, coord);
            }
        }
    }

    fn enqueue_face(&mut self, lod: u32, axis: usize, face_coord: i32, origin: IVec3) {
        let (u_range, v_range) = face_plane_ranges(axis, origin);
        for u in u_range.0..=u_range.1 {
            for v in v_range.0..=v_range.1 {
                let coord = make_coord(axis, face_coord, u, v);
                if brick_may_contain_terrain(lod, coord) {
                    self.pending_jobs.push_back((lod, coord));
                }
            }
        }
    }

    fn enqueue_full_grid(&mut self, lod: u32, center: IVec3) {
        for dz in -HALF_GRID..HALF_GRID {
            for dy in -HALF_GRID..HALF_GRID {
                for dx in -HALF_GRID..HALF_GRID {
                    let coord = [center.x + dx, center.y + dy, center.z + dz];
                    if brick_may_contain_terrain(lod, coord) {
                        self.pending_jobs.push_back((lod, coord));
                    }
                }
            }
        }
    }

    /// Allocate brick slots for a batch and return the GPU job list.
    fn allocate_jobs(
        &self,
        batch: &[(u32, [i32; 3])],
        sbm: &mut BrickMap,
        gpu: &GpuContext,
    ) -> Vec<TerrainGenJob> {
        let mut gpu_jobs = Vec::with_capacity(batch.len());
        for &(lod, coord) in batch {
            if let Some(brick_idx) = sbm.begin_brick_generation(gpu, lod, coord) {
                gpu_jobs.push(TerrainGenJob {
                    brick_idx,
                    brick_x: coord[0],
                    brick_y: coord[1],
                    brick_z: coord[2],
                    lod,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                });
            }
        }
        gpu_jobs
    }
}

fn lod_brick_from_world(pos: Vec3, lod: u32) -> IVec3 {
    let voxel_size = (1u32 << lod) as f32;
    let brick_world = pos / (8.0 * voxel_size);
    brick_world.floor().as_ivec3()
}

/// Returns (u_range, v_range) for the two axes not equal to `axis`,
/// spanning ±HALF_GRID around the corresponding origin components.
fn face_plane_ranges(axis: usize, origin: IVec3) -> ((i32, i32), (i32, i32)) {
    let (u_axis, v_axis) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let u_center = origin[u_axis];
    let v_center = origin[v_axis];
    (
        (u_center - HALF_GRID, u_center + HALF_GRID - 1),
        (v_center - HALF_GRID, v_center + HALF_GRID - 1),
    )
}

fn make_coord(axis: usize, face: i32, u: i32, v: i32) -> [i32; 3] {
    match axis {
        0 => [face, u, v],
        1 => [u, face, v],
        _ => [u, v, face],
    }
}
