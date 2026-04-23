//! Sparse Brick Map — GPU buffer manager for hierarchical voxel storage.

use ara_core::glam::IVec3;
use ara_core::{
    BRICK_EMPTY, BRICK_SIZE, BRICK_VOLUME, BrickHeader, LOD_COUNT, MemoryBudget, PackedVoxel,
    TOP_GRID_SIZE, TOP_GRID_VOLUME, WORLD_EXTENT, brick_local_index, bytemuck, world_to_brick,
    world_to_local,
};

use crate::gpu::GpuContext;

/// Sparse Brick Map managing a 64³ top grid and a configurable brick pool.
pub struct BrickMap {
    top_grid_buf: wgpu::Buffer,
    brick_pool_buf: wgpu::Buffer,
    brick_header_buf: wgpu::Buffer,
    brick_occupancy_buf: wgpu::Buffer,
    radiance_pool_buf: wgpu::Buffer,
    free_list: Vec<u32>,
    world_origin: [i32; 3],
    /// Per-LOD camera center in LOD-space brick coordinates.
    lod_origins: [[i32; 3]; 8],
    max_bricks: u32,
    budget: MemoryBudget,
    /// CPU-side copy of the top grid for raycast lookups.
    top_grid: Vec<u32>,
    /// CPU-side sparse brick data for edited/loaded bricks.
    brick_data: Vec<Vec<PackedVoxel>>,
    /// Bricks generated this frame whose GPU top-grid entry is deferred to next frame.
    pending_reveals: Vec<(usize, u32)>,
}

impl BrickMap {
    /// Allocate all GPU buffers and initialize the free list.
    pub fn new(gpu: &GpuContext, budget: MemoryBudget) -> Self {
        let max_bricks = budget.max_bricks();
        let device = gpu.device();

        // Top grid: 64³ × LOD_COUNT LODs × 4 bytes, initialized to BRICK_EMPTY
        let top_grid_data = vec![BRICK_EMPTY; (TOP_GRID_VOLUME * LOD_COUNT) as usize];
        let top_grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Top Grid"),
            size: (TOP_GRID_VOLUME * LOD_COUNT * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue()
            .write_buffer(&top_grid_buf, 0, bytemuck::cast_slice(&top_grid_data));

        // Brick pool: max_bricks × 512 × 2 bytes (u16 voxels)
        let pool_size = max_bricks as u64 * BRICK_VOLUME as u64 * 2;
        let brick_pool_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Brick Pool"),
            size: pool_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Brick headers: max_bricks × 16 bytes, initialized to 0xFF for empty sentinel
        let header_init = vec![0xFFu8; max_bricks as usize * 16];
        let brick_header_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Brick Headers"),
            size: max_bricks as u64 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue().write_buffer(&brick_header_buf, 0, &header_init);

        // Brick occupancy: max_bricks × 16 × 4 bytes (512 bits per brick = 16 u32s)
        let brick_occupancy_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Brick Occupancy"),
            size: max_bricks as u64 * 16 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Free list: all brick indices available, descending for stack-like pop
        let free_list: Vec<u32> = (0..max_bricks).rev().collect();

        // Radiance pool: max_bricks × 512 × 8 bytes (vec2<u32> via pack2x16float)
        let radiance_pool_size = max_bricks as u64 * BRICK_VOLUME as u64 * 8;
        let radiance_pool_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Ara Radiance Pool"),
            size: radiance_pool_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        radiance_pool_buf.unmap();

        // CPU-side brick data storage
        let brick_data = vec![Vec::new(); max_bricks as usize];

        Self {
            top_grid_buf,
            brick_pool_buf,
            brick_header_buf,
            brick_occupancy_buf,
            radiance_pool_buf,
            free_list,
            world_origin: [0, 0, 0],
            lod_origins: [[0i32; 3]; 8],
            max_bricks,
            budget,
            top_grid: top_grid_data,
            brick_data,
            pending_reveals: Vec::new(),
        }
    }

    /// Allocate a brick for the given brick coordinate. Returns the brick pool index.
    pub fn allocate_brick(&mut self, gpu: &GpuContext, brick_coord: [i32; 3]) -> Option<u32> {
        let brick_idx = self.free_list.pop()?;

        // Write top grid entry (safe wrapping for negative coords)
        let grid_idx = top_grid_index(0, brick_coord[0], brick_coord[1], brick_coord[2]) as u32;

        self.top_grid[grid_idx as usize] = brick_idx;
        gpu.queue().write_buffer(
            &self.top_grid_buf,
            grid_idx as u64 * 4,
            bytemuck::bytes_of(&brick_idx),
        );

        // Write brick header
        let header = BrickHeader {
            world_coord_packed: BrickHeader::pack_coord(
                brick_coord[0] as u32,
                brick_coord[1] as u32,
                brick_coord[2] as u32,
                0,
            ),
            last_access_frame: 0,
            solid_count: 0,
            _pad: 0,
        };
        gpu.queue().write_buffer(
            &self.brick_header_buf,
            brick_idx as u64 * 16,
            bytemuck::bytes_of(&header),
        );

        // Initialize CPU cache so get_voxel/write_voxel stay consistent
        self.brick_data[brick_idx as usize] = vec![PackedVoxel::AIR; BRICK_VOLUME as usize];

        Some(brick_idx)
    }

    /// Allocate a brick at a specific LOD level for the given brick coordinate.
    pub fn allocate_lod_brick(
        &mut self,
        gpu: &GpuContext,
        lod: u32,
        brick_coord: [i32; 3],
    ) -> Option<u32> {
        let brick_idx = self.free_list.pop()?;

        let grid_idx = top_grid_index(lod, brick_coord[0], brick_coord[1], brick_coord[2]) as u32;

        self.top_grid[grid_idx as usize] = brick_idx;
        gpu.queue().write_buffer(
            &self.top_grid_buf,
            grid_idx as u64 * 4,
            bytemuck::bytes_of(&brick_idx),
        );

        // Write brick header with LOD level in bits 30-31
        let header = BrickHeader {
            world_coord_packed: BrickHeader::pack_coord(
                brick_coord[0] as u32,
                brick_coord[1] as u32,
                brick_coord[2] as u32,
                lod,
            ),
            last_access_frame: 0,
            solid_count: 0,
            _pad: 0,
        };
        gpu.queue().write_buffer(
            &self.brick_header_buf,
            brick_idx as u64 * 16,
            bytemuck::bytes_of(&header),
        );

        Some(brick_idx)
    }

    /// Generate LOD1–(LOD_COUNT-1) bricks for all occupied lower-LOD regions.
    pub fn generate_lod_bricks(&mut self, gpu: &GpuContext) {
        let tgs = TOP_GRID_SIZE as i32;

        for dst_lod in 1..LOD_COUNT {
            let src_lod = dst_lod - 1;
            // At src_lod, the grid extent (in src_lod brick coords) is tgs >> src_lod
            // but since the top_grid wraps toroidally we just iterate tgs >> src_lod unique coords.
            let extent = tgs >> src_lod as i32;
            for bz in (0..extent).step_by(2) {
                for by in (0..extent).step_by(2) {
                    for bx in (0..extent).step_by(2) {
                        let mut has_occupied = false;
                        'outer: for dz in 0..2i32 {
                            for dy in 0..2i32 {
                                for dx in 0..2i32 {
                                    let idx = top_grid_index(src_lod, bx + dx, by + dy, bz + dz);
                                    if self.top_grid[idx] != BRICK_EMPTY {
                                        has_occupied = true;
                                        break 'outer;
                                    }
                                }
                            }
                        }
                        if has_occupied {
                            let dst_coord = [bx / 2, by / 2, bz / 2];
                            let idx = top_grid_index(dst_lod, dst_coord[0], dst_coord[1], dst_coord[2]);
                            if self.top_grid[idx] == BRICK_EMPTY {
                                self.allocate_lod_brick(gpu, dst_lod, dst_coord);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Write a full brick's voxel data to the GPU pool.
    pub fn write_brick(&mut self, gpu: &GpuContext, brick_idx: u32, voxels: &[PackedVoxel; 512]) {
        let offset = brick_idx as u64 * BRICK_VOLUME as u64 * 2;
        gpu.queue()
            .write_buffer(&self.brick_pool_buf, offset, bytemuck::cast_slice(voxels));

        // Update CPU-side copy
        self.brick_data[brick_idx as usize] = voxels.to_vec();

        // Build occupancy bitmap (16 u32s = 512 bits)
        let mut occupancy = [0u32; 16];
        let mut solid_count = 0u32;
        for (i, v) in voxels.iter().enumerate() {
            if !v.is_air() {
                occupancy[i / 32] |= 1 << (i % 32);
                solid_count += 1;
            }
        }
        gpu.queue().write_buffer(
            &self.brick_occupancy_buf,
            brick_idx as u64 * 16 * 4,
            bytemuck::cast_slice(&occupancy),
        );

        // Update solid count in header
        gpu.queue().write_buffer(
            &self.brick_header_buf,
            brick_idx as u64 * 16 + 8,
            bytemuck::bytes_of(&solid_count),
        );
    }

    /// Convert the old room scene into the SBM format and upload all bricks.
    pub fn upload_test_scene(gpu: &GpuContext, budget: MemoryBudget) -> Self {
        let mut sbm = Self::new(gpu, budget);

        let ground_y = 10u32;
        let ceiling_y = 50u32;
        let min_xz = 10u32;
        let max_xz = 53u32;

        let mut rng_seed = 12345u32;
        let mut next_variant = || -> u8 {
            rng_seed = rng_seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng_seed >> 16) & 0x3) as u8
        };

        // Collect all voxels into a temporary flat grid, then brick-ify
        let size = WORLD_EXTENT as usize;
        let mut flat = vec![PackedVoxel::AIR; size * size * size];

        for z in 0..size {
            for x in 0..size {
                for y in 0..size {
                    let idx = z * size * size + y * size + x;
                    let (xu, yu, zu) = (x as u32, y as u32, z as u32);

                    // Floor
                    if yu == ground_y
                        && xu >= min_xz
                        && xu <= max_xz
                        && zu >= min_xz
                        && zu <= max_xz
                    {
                        let mat = if (xu + zu) % 2 == 0 { 1u16 } else { 3u16 };
                        let mut v = PackedVoxel::new(mat);
                        if mat == 1 {
                            v.set_variant(next_variant());
                        }
                        flat[idx] = v;
                        continue;
                    }

                    // Ceiling
                    if yu == ceiling_y
                        && xu >= min_xz
                        && xu <= max_xz
                        && zu >= min_xz
                        && zu <= max_xz
                    {
                        flat[idx] = PackedVoxel::new(6);
                        continue;
                    }

                    // Walls
                    if yu > ground_y && yu < ceiling_y {
                        let is_wall_x = xu == min_xz || xu == max_xz;
                        let is_wall_z = zu == min_xz || zu == max_xz;
                        if (is_wall_x && zu >= min_xz && zu <= max_xz)
                            || (is_wall_z && xu >= min_xz && xu <= max_xz)
                        {
                            let mut v = PackedVoxel::new(1);
                            v.set_variant(next_variant());
                            flat[idx] = v;
                            continue;
                        }
                    }
                }
            }
        }

        // Pillars
        for y in (ground_y + 1)..(ground_y + 5) {
            let idx1 = 20 * size * size + y as usize * size + 20;
            flat[idx1] = PackedVoxel::new(2);
            let idx2 = 40 * size * size + y as usize * size + 40;
            flat[idx2] = PackedVoxel::new(2);
        }

        // Convert flat grid to bricks
        let tgs = TOP_GRID_SIZE as i32;
        let bs = BRICK_SIZE as i32;
        for bz in 0..tgs {
            for by in 0..tgs {
                for bx in 0..tgs {
                    let mut brick = [PackedVoxel::AIR; 512];
                    let mut has_solid = false;

                    for lz in 0..bs {
                        for ly in 0..bs {
                            for lx in 0..bs {
                                let wx = bx * bs + lx;
                                let wy = by * bs + ly;
                                let wz = bz * bs + lz;
                                if wx < size as i32 && wy < size as i32 && wz < size as i32 {
                                    let flat_idx = wz as usize * size * size
                                        + wy as usize * size
                                        + wx as usize;
                                    let v = flat[flat_idx];
                                    if !v.is_air() {
                                        has_solid = true;
                                    }
                                    let li =
                                        brick_local_index(lx as u32, ly as u32, lz as u32) as usize;
                                    brick[li] = v;
                                }
                            }
                        }
                    }

                    // Allocate all bricks within the room bounding box so light can propagate
                    let in_room = bx >= 1 && bx <= 7 && bz >= 1 && bz <= 7 && by >= 1 && by <= 7;
                    if has_solid || in_room {
                        if let Some(brick_idx) = sbm.allocate_brick(gpu, [bx, by, bz]) {
                            sbm.write_brick(gpu, brick_idx, &brick);
                        }
                    }
                }
            }
        }

        // Generate LOD bricks for VCT mipmap
        sbm.generate_lod_bricks(gpu);

        sbm
    }

    /// Look up a voxel by world coordinate from CPU-side data.
    pub fn get_voxel(&self, pos: IVec3) -> PackedVoxel {
        if pos.x < 0
            || pos.y < 0
            || pos.z < 0
            || pos.x >= WORLD_EXTENT as i32
            || pos.y >= WORLD_EXTENT as i32
            || pos.z >= WORLD_EXTENT as i32
        {
            return PackedVoxel::AIR;
        }

        let bc = world_to_brick(pos.x, pos.y, pos.z);
        let grid_idx = top_grid_index(0, bc[0], bc[1], bc[2]);
        let brick_idx = self.top_grid[grid_idx];

        if brick_idx == BRICK_EMPTY {
            return PackedVoxel::AIR;
        }

        let lc = world_to_local(pos.x, pos.y, pos.z);
        let li = brick_local_index(lc[0], lc[1], lc[2]) as usize;
        self.brick_data[brick_idx as usize][li]
    }

    /// Write a single voxel to the GPU (and CPU cache) at the given world position.
    pub fn write_voxel(&mut self, gpu: &GpuContext, world_pos: IVec3, voxel: PackedVoxel) {
        if world_pos.x < 0
            || world_pos.y < 0
            || world_pos.z < 0
            || world_pos.x >= WORLD_EXTENT as i32
            || world_pos.y >= WORLD_EXTENT as i32
            || world_pos.z >= WORLD_EXTENT as i32
        {
            return;
        }

        let bc = world_to_brick(world_pos.x, world_pos.y, world_pos.z);
        let grid_idx = top_grid_index(0, bc[0], bc[1], bc[2]);
        let mut brick_idx = self.top_grid[grid_idx];

        // Allocate brick if needed
        if brick_idx == BRICK_EMPTY {
            if voxel.is_air() {
                return;
            }
            if let Some(idx) = self.allocate_brick(gpu, bc) {
                brick_idx = idx;
                let air_brick = [PackedVoxel::AIR; 512];
                self.write_brick(gpu, brick_idx, &air_brick);
            } else {
                return;
            }
        }

        let lc = world_to_local(world_pos.x, world_pos.y, world_pos.z);
        let li = brick_local_index(lc[0], lc[1], lc[2]) as usize;

        // Update CPU cache (brick_data always initialized to BRICK_VOLUME on allocation)
        self.brick_data[brick_idx as usize][li] = voxel;

        // Update GPU brick pool (u16 per voxel, must write aligned u32)
        let pair_idx = li / 2;
        let data = &self.brick_data[brick_idx as usize];
        let lo = data[pair_idx * 2].packed as u32;
        let hi = data[pair_idx * 2 + 1].packed as u32;
        let word = lo | (hi << 16);
        let offset = (brick_idx as u64 * BRICK_VOLUME as u64 * 2) + (pair_idx as u64 * 4);
        gpu.queue()
            .write_buffer(&self.brick_pool_buf, offset, bytemuck::bytes_of(&word));

        // Update occupancy bit
        let occ_idx = li / 32;
        let occ_base = brick_idx as u64 * 16 * 4;
        let mut occ_word = 0u32;
        for i in (occ_idx * 32)..((occ_idx + 1) * 32).min(512) {
            if !data[i].is_air() {
                occ_word |= 1 << (i % 32);
            }
        }
        gpu.queue().write_buffer(
            &self.brick_occupancy_buf,
            occ_base + occ_idx as u64 * 4,
            bytemuck::bytes_of(&occ_word),
        );
    }

    /// Evict a brick at the given LOD and brick coordinate, returning it to the free list.
    pub fn evict_brick(&mut self, gpu: &GpuContext, lod: u32, brick_coord: [i32; 3]) {
        let grid_idx = top_grid_index(lod, brick_coord[0], brick_coord[1], brick_coord[2]);
        let brick_idx = self.top_grid[grid_idx];
        if brick_idx == BRICK_EMPTY {
            return;
        }
        self.pending_reveals.retain(|&(gid, _)| gid != grid_idx);
        self.top_grid[grid_idx] = BRICK_EMPTY;
        self.free_list.push(brick_idx);
        self.brick_data[brick_idx as usize] = Vec::new();

        let sentinel = BRICK_EMPTY;
        gpu.queue().write_buffer(
            &self.top_grid_buf,
            grid_idx as u64 * 4,
            bytemuck::bytes_of(&sentinel),
        );
        let empty_header = [0xFFu8; 16];
        gpu.queue().write_buffer(
            &self.brick_header_buf,
            brick_idx as u64 * 16,
            &empty_header,
        );
    }

    /// Allocate a brick slot and write the top grid and header.
    /// Returns the brick pool index to use in a [`TerrainGenJob`].
    /// Voxel data is written by the GPU terrain shader, not here.
    /// The GPU top-grid entry is held back until the next call to [`reveal_pending`]
    /// so the coarser LOD remains visible while terrain gen runs.
    pub fn begin_brick_generation(
        &mut self,
        gpu: &GpuContext,
        lod: u32,
        brick_coord: [i32; 3],
    ) -> Option<u32> {
        let grid_idx = top_grid_index(lod, brick_coord[0], brick_coord[1], brick_coord[2]);
        if self.top_grid[grid_idx] != BRICK_EMPTY {
            return None;
        }
        let brick_idx = if lod == 0 {
            self.allocate_brick(gpu, brick_coord)?
        } else {
            self.allocate_lod_brick(gpu, lod, brick_coord)?
        };
        // Overwrite the immediate GPU upload with BRICK_EMPTY; real idx committed next frame.
        let sentinel = BRICK_EMPTY;
        gpu.queue().write_buffer(
            &self.top_grid_buf,
            grid_idx as u64 * 4,
            bytemuck::bytes_of(&sentinel),
        );
        self.pending_reveals.push((grid_idx, brick_idx));
        Some(brick_idx)
    }

    /// Commit pending brick reveals to the GPU top-grid.
    /// Call once per frame before terrain streaming to make last-frame bricks visible.
    pub fn reveal_pending(&mut self, gpu: &GpuContext) {
        for &(grid_idx, brick_idx) in &self.pending_reveals {
            gpu.queue().write_buffer(
                &self.top_grid_buf,
                grid_idx as u64 * 4,
                bytemuck::bytes_of(&brick_idx),
            );
        }
        self.pending_reveals.clear();
    }

    /// Return per-LOD camera center in LOD-space brick coordinates.
    pub fn lod_origin(&self, lod: u32) -> [i32; 3] {
        self.lod_origins[lod as usize]
    }

    /// Set per-LOD camera center in LOD-space brick coordinates.
    pub fn set_lod_origin(&mut self, lod: u32, origin: [i32; 3]) {
        self.lod_origins[lod as usize] = origin;
    }

    /// Pool utilization as a fraction (0.0 to 1.0).
    pub fn utilization(&self) -> f32 {
        let allocated = self.max_bricks - self.free_list.len() as u32;
        allocated as f32 / self.max_bricks as f32
    }

    pub fn top_grid_buf(&self) -> &wgpu::Buffer {
        &self.top_grid_buf
    }
    pub fn brick_pool_buf(&self) -> &wgpu::Buffer {
        &self.brick_pool_buf
    }
    pub fn brick_header_buf(&self) -> &wgpu::Buffer {
        &self.brick_header_buf
    }
    pub fn brick_occupancy_buf(&self) -> &wgpu::Buffer {
        &self.brick_occupancy_buf
    }
    pub fn radiance_pool_buf(&self) -> &wgpu::Buffer {
        &self.radiance_pool_buf
    }
    pub fn world_origin(&self) -> [i32; 3] {
        self.world_origin
    }
    pub fn max_bricks(&self) -> u32 {
        self.max_bricks
    }
    pub fn budget(&self) -> MemoryBudget {
        self.budget
    }
}

fn top_grid_index(lod: u32, bx: i32, by: i32, bz: i32) -> usize {
    let tgs = TOP_GRID_SIZE as i32;
    let x = ((bx % tgs + tgs) % tgs) as u32;
    let y = ((by % tgs + tgs) % tgs) as u32;
    let z = ((bz % tgs + tgs) % tgs) as u32;
    (lod * TOP_GRID_VOLUME + z * TOP_GRID_SIZE * TOP_GRID_SIZE + y * TOP_GRID_SIZE + x) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_grid_index_positive_coords() {
        let idx = top_grid_index(0, 5, 10, 20);
        let expected = (20 * 64 * 64 + 10 * 64 + 5) as usize;
        assert_eq!(idx, expected);
    }

    #[test]
    fn top_grid_index_wraps_negative() {
        // -1 should wrap to 63 in a 64-sized grid
        let idx_neg = top_grid_index(0, -1, 0, 0);
        let idx_pos = top_grid_index(0, 63, 0, 0);
        assert_eq!(idx_neg, idx_pos);
    }

    #[test]
    fn top_grid_index_wraps_overflow() {
        // 64 should wrap to 0
        let idx_over = top_grid_index(0, 64, 0, 0);
        let idx_zero = top_grid_index(0, 0, 0, 0);
        assert_eq!(idx_over, idx_zero);
    }

    #[test]
    fn top_grid_index_negative_all_axes() {
        let idx = top_grid_index(0, -1, -2, -3);
        let expected = top_grid_index(0, 63, 62, 61);
        assert_eq!(idx, expected);
    }

    #[test]
    fn top_grid_index_lod_offset() {
        let idx_lod0 = top_grid_index(0, 0, 0, 0);
        let idx_lod1 = top_grid_index(1, 0, 0, 0);
        assert_eq!(idx_lod0, 0);
        assert_eq!(idx_lod1, TOP_GRID_VOLUME as usize);
    }

    #[test]
    fn top_grid_index_bounds() {
        // All indices for LOD0 should be within [0, TOP_GRID_VOLUME)
        for bz in -64..128 {
            for by in -64..128 {
                for bx in -64..128 {
                    let idx = top_grid_index(0, bx, by, bz);
                    assert!(idx < TOP_GRID_VOLUME as usize,
                        "index {} out of bounds for coords ({}, {}, {})", idx, bx, by, bz);
                }
            }
        }
    }

    #[test]
    fn world_to_brick_and_local_roundtrip() {
        for wx in [0, 7, 8, 15, 63, 511] {
            for wy in [0, 1, 8, 100] {
                for wz in [0, 7, 16, 511] {
                    let bc = world_to_brick(wx, wy, wz);
                    let lc = world_to_local(wx, wy, wz);

                    // Reconstruct world from brick + local
                    let rx = bc[0] * BRICK_SIZE as i32 + lc[0] as i32;
                    let ry = bc[1] * BRICK_SIZE as i32 + lc[1] as i32;
                    let rz = bc[2] * BRICK_SIZE as i32 + lc[2] as i32;
                    assert_eq!((rx, ry, rz), (wx, wy, wz),
                        "roundtrip failed for ({}, {}, {})", wx, wy, wz);
                }
            }
        }
    }

    #[test]
    fn brick_local_index_range() {
        for lz in 0..8u32 {
            for ly in 0..8u32 {
                for lx in 0..8u32 {
                    let idx = brick_local_index(lx, ly, lz);
                    assert!(idx < BRICK_VOLUME, "local index {} >= BRICK_VOLUME", idx);
                }
            }
        }
    }
}
