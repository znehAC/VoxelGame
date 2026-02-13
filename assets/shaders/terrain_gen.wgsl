const GRID_SIZE: u32 = 512u;
const CHUNK_SIZE: u32 = 32u;
const CHUNKS_PER_AXIS: u32 = 16u;

const MAT_AIR: u32 = 0u;
const MAT_STONE: u32 = 1u;
const MAT_DIRT: u32 = 2u;
const MAT_GRASS: u32 = 3u;

struct ChunkParams {
    offset_x: i32,
    offset_y: i32,
    offset_z: i32,
    pad: i32,
}

var<push_constant> params: ChunkParams;

@group(0) @binding(0) var<storage, read_write> voxels: array<u32>;
@group(0) @binding(1) var<storage, read_write> occupancy: array<atomic<u32>>;

// Toroidal wrap for voxel coordinates
fn wrap(c: i32) -> i32 {
    let size = i32(GRID_SIZE);
    return ((c % size) + size) % size;
}

fn voxel_index(x: i32, y: i32, z: i32) -> u32 {
    return u32(wrap(z)) * GRID_SIZE * GRID_SIZE + u32(wrap(y)) * GRID_SIZE + u32(wrap(x));
}

fn chunk_wrap(c: i32) -> i32 {
    let size = i32(CHUNKS_PER_AXIS);
    return ((c % size) + size) % size;
}

fn chunk_index(cx: i32, cy: i32, cz: i32) -> u32 {
    return u32(chunk_wrap(cz)) * CHUNKS_PER_AXIS * CHUNKS_PER_AXIS
         + u32(chunk_wrap(cy)) * CHUNKS_PER_AXIS
         + u32(chunk_wrap(cx));
}

// PCG hash
fn pcg_hash(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// 2D hash -> [0, 1]
fn hash2d(x: i32, z: i32) -> f32 {
    let h = pcg_hash(u32(x) * 73856093u + u32(z) * 19349663u);
    return f32(h) / 4294967295.0;
}

// Hermite smoothstep
fn smoothstep_h(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

// Bilinear interpolation of hash values
fn smooth_noise(x: f32, z: f32) -> f32 {
    let ix = i32(floor(x));
    let iz = i32(floor(z));
    let fx = x - floor(x);
    let fz = z - floor(z);

    let sx = smoothstep_h(fx);
    let sz = smoothstep_h(fz);

    let v00 = hash2d(ix, iz);
    let v10 = hash2d(ix + 1, iz);
    let v01 = hash2d(ix, iz + 1);
    let v11 = hash2d(ix + 1, iz + 1);

    let a = mix(v00, v10, sx);
    let b = mix(v01, v11, sx);
    return mix(a, b, sz);
}

// Fractal Brownian Motion — 3 octaves
fn fbm(x: f32, z: f32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;

    for (var i = 0u; i < 3u; i++) {
        value += smooth_noise(x * frequency, z * frequency) * amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}

fn terrain_height(wx: i32, wz: i32) -> i32 {
    return i32(fbm(f32(wx) * 0.015, f32(wz) * 0.015) * 20.0);
}

@compute @workgroup_size(4, 4, 4)
fn generate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let local_pos = vec3i(gid);
    if local_pos.x >= i32(CHUNK_SIZE) || local_pos.y >= i32(CHUNK_SIZE) || local_pos.z >= i32(CHUNK_SIZE) {
        return;
    }

    let world_pos = vec3i(params.offset_x, params.offset_y, params.offset_z) + local_pos;
    let height = terrain_height(world_pos.x, world_pos.z);

    var material = MAT_AIR;
    if world_pos.y < height - 5 {
        material = MAT_STONE;
    } else if world_pos.y < height {
        material = MAT_DIRT;
    } else if world_pos.y == height {
        material = MAT_GRASS;
    }

    let idx = voxel_index(world_pos.x, world_pos.y, world_pos.z);
    
    var packed_voxel = 0u;
    if material != MAT_AIR {
        // Generate random variant (0-15) using position hash
        // We use the storage index for the hash which wraps every 512 blocks,
        // sufficient for visual noise.
        let variant = pcg_hash(idx) & 0xFu;
        
        // Pack: ID (14 bits) | Variant (4 bits @ 22)
        packed_voxel = (material & 0x3FFFu) | (variant << 22u);
        
        // Mark chunk as occupied
        let cs = i32(CHUNK_SIZE);
        let cx = i32(floor(f32(world_pos.x) / f32(cs)));
        let cy = i32(floor(f32(world_pos.y) / f32(cs)));
        let cz = i32(floor(f32(world_pos.z) / f32(cs)));
        let ci = chunk_index(cx, cy, cz);
        atomicMax(&occupancy[ci], 1u);
    }
    
    voxels[idx] = packed_voxel;
}
