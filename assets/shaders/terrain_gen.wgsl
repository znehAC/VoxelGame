const BRICK_SIZE: u32 = 8u;
const BRICK_VOLUME: u32 = 512u;
const PI2: f32 = 6.28318530718;

struct TerrainJob {
    brick_idx: u32,
    brick_x: i32,
    brick_y: i32,
    brick_z: i32,
}

@group(0) @binding(0) var<storage, read>       jobs:            array<TerrainJob>;
@group(0) @binding(1) var<storage, read_write> brick_pool:      array<u32>;
@group(0) @binding(2) var<storage, read_write> brick_occupancy: array<u32>;

fn morton_encode(x: u32, y: u32, z: u32) -> u32 {
    var vx = x & 7u;
    vx = (vx | (vx << 4u)) & 0x43u;
    vx = (vx | (vx << 2u)) & 0x49u;

    var vy = y & 7u;
    vy = (vy | (vy << 4u)) & 0x43u;
    vy = (vy | (vy << 2u)) & 0x49u;

    var vz = z & 7u;
    vz = (vz | (vz << 4u)) & 0x43u;
    vz = (vz | (vz << 2u)) & 0x49u;

    return vx | (vy << 1u) | (vz << 2u);
}

fn height_at(wx: i32, wz: i32) -> i32 {
    let x = f32(wx);
    let z = f32(wz);
    let h = 100.0
        + sin(x / 3200.0 * PI2) * cos(z / 3200.0 * PI2) * 60.0
        + sin(x / 800.0  * PI2) * sin(z / 800.0  * PI2) * 20.0
        + sin(x / 160.0  * PI2) * sin(z / 160.0  * PI2) *  5.0;
    return i32(h);
}

fn voxel_material(world_x: i32, world_y: i32, world_z: i32) -> u32 {
    let h = height_at(world_x, world_z);
    if world_y > h        { return 0u; }
    else if world_y == h  { return 1u; }  // grass
    else if world_y >= h - 3 { return 2u; }  // dirt
    else                  { return 3u; }  // stone
}

@compute @workgroup_size(4, 8, 8)
fn generate(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let job = jobs[wg.x];

    // Each thread handles 2 voxels along X (lx0=even, lx1=odd), sharing a Morton pair.
    let lx0 = lid.x * 2u;
    let lx1 = lid.x * 2u + 1u;
    let ly  = lid.y;
    let lz  = lid.z;

    let morton0 = morton_encode(lx0, ly, lz);
    // lx0 is always even → morton0 is always even → pair_idx = morton0 / 2
    let pair_idx = morton0 >> 1u;

    let wx0 = job.brick_x * i32(BRICK_SIZE) + i32(lx0);
    let wx1 = job.brick_x * i32(BRICK_SIZE) + i32(lx1);
    let wy  = job.brick_y * i32(BRICK_SIZE) + i32(ly);
    let wz  = job.brick_z * i32(BRICK_SIZE) + i32(lz);

    let mat0 = voxel_material(wx0, wy, wz);
    let mat1 = voxel_material(wx1, wy, wz);

    // Pack two u16 voxel material IDs into one u32 (low=mat0, high=mat1)
    brick_pool[job.brick_idx * 256u + pair_idx] = (mat0 & 0xFFFFu) | ((mat1 & 0xFFFFu) << 16u);

    workgroupBarrier();

    // Build occupancy bitmap from the now-filled brick_pool.
    // 16 threads each handle 16 u32 pairs = 32 voxels.
    let lid_linear = lid.z * 32u + lid.y * 4u + lid.x;
    if lid_linear < 16u {
        var occ = 0u;
        let base = lid_linear * 16u;
        for (var k = 0u; k < 16u; k++) {
            let packed = brick_pool[job.brick_idx * 256u + base + k];
            let m0 = packed & 0xFFFFu;
            let m1 = (packed >> 16u) & 0xFFFFu;
            if m0 != 0u { occ |= 1u << (k * 2u); }
            if m1 != 0u { occ |= 1u << (k * 2u + 1u); }
        }
        brick_occupancy[job.brick_idx * 16u + lid_linear] = occ;
    }
}
