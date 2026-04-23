const BRICK_SIZE: u32 = 8u;
const BRICK_VOLUME: u32 = 512u;

struct TerrainJob {
    brick_idx: u32,
    brick_x: i32,
    brick_y: i32,
    brick_z: i32,
    lod: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read>       jobs:            array<TerrainJob>;
@group(0) @binding(1) var<storage, read_write> brick_pool:      array<u32>;
@group(0) @binding(2) var<storage, read_write> brick_occupancy: array<u32>;

// Terrain height in LOD0 voxel units (5cm each), base y=64 (~3.2m).
// Gentle rolling ground appropriate for 5cm-scale gameplay:
//   Long swells:  160m period, 60cm amplitude
//   Medium rolls:  40m period, 20cm amplitude
//   Fine bumps:     8m period,  5cm amplitude
fn height_at(wx: i32, wz: i32) -> i32 {
    let x = f32(wx);
    let z = f32(wz);
    let pi2 = 6.28318530718;
    let h = 64.0
        + sin(x / 3200.0 * pi2) * cos(z / 3200.0 * pi2) * 12.0
        + sin((x + z) / 800.0 * pi2) * 4.0
        + sin(x / 160.0 * pi2) * sin(z / 160.0 * pi2) * 1.0;
    return i32(h);
}

// Sample terrain at voxel footprint center to avoid inter-LOD height bias.
fn height_at_lod(wx: i32, wz: i32, vs: i32) -> i32 {
    return height_at(wx + vs / 2, wz + vs / 2);
}

fn mat_id(world_y: i32, surface_y: i32, voxel_size: i32) -> u32 {
    let depth = surface_y - world_y;
    if depth < 0              { return 0u; } // air
    if depth < voxel_size     { return 1u; } // grass  (1 voxel thick at each LOD)
    if depth < voxel_size * 5 { return 2u; } // dirt   (4 more voxels)
    return 3u;                                // stone
}

var<workgroup> wg_solid: array<u32, 512>;

@compute @workgroup_size(4, 8, 8)
fn generate(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let job = jobs[wg.x];

    // lid.x in [0,3] covers 2 voxels each → lx0, lx1
    let lx0 = lid.x * 2u;
    let lx1 = lid.x * 2u + 1u;
    let ly  = lid.y;
    let lz  = lid.z;

    let voxel_size = i32(1u << job.lod);

    // World voxel coords (LOD0 units) for lx0 and lx1
    let wx0 = (job.brick_x * 8 + i32(lx0)) * voxel_size;
    let wx1 = (job.brick_x * 8 + i32(lx1)) * voxel_size;
    let wy  = (job.brick_y * 8 + i32(ly))  * voxel_size;
    let wz  = (job.brick_z * 8 + i32(lz))  * voxel_size;

    let h0 = height_at_lod(wx0, wz, voxel_size);
    let h1 = height_at_lod(wx1, wz, voxel_size);

    let m0 = mat_id(wy, h0, voxel_size);
    let m1 = mat_id(wy, h1, voxel_size);

    // Write packed u32 (two u16 voxels per word)
    let linear0 = lz * 64u + ly * 8u + lx0;
    brick_pool[job.brick_idx * 256u + (linear0 >> 1u)] = m0 | (m1 << 16u);

    // Stash solidity for occupancy pass
    wg_solid[linear0]      = select(0u, 1u, m0 != 0u);
    wg_solid[linear0 + 1u] = select(0u, 1u, m1 != 0u);

    workgroupBarrier();

    // First 16 threads compute one occupancy u32 each (32 voxels/word)
    let lid_linear = lid.z * 32u + lid.y * 4u + lid.x;
    if lid_linear < 16u {
        var occ = 0u;
        let base = lid_linear * 32u;
        for (var k = 0u; k < 32u; k++) {
            if wg_solid[base + k] != 0u {
                occ |= 1u << k;
            }
        }
        brick_occupancy[job.brick_idx * 16u + lid_linear] = occ;
    }
}
