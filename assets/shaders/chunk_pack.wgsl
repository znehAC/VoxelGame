// Pack dirty chunks from strided 512³ voxel buffer into dense 32³ blocks.

struct PackParams {
    origin_x: u32,
    origin_y: u32,
    origin_z: u32,
    chunk_index: u32,
}

var<push_constant> params: PackParams;

@group(0) @binding(0) var<storage, read> voxels: array<u32>;
@group(0) @binding(1) var<storage, read_write> pack_buf: array<u32>;

const GRID: u32 = 512u;
const CHUNK: u32 = 32u;
const CHUNK_VOL: u32 = 32u * 32u * 32u;

@compute @workgroup_size(4, 4, 4)
fn pack_chunk(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= CHUNK || gid.y >= CHUNK || gid.z >= CHUNK {
        return;
    }

    let src_x = params.origin_x + gid.x;
    let src_y = params.origin_y + gid.y;
    let src_z = params.origin_z + gid.z;
    let src_idx = src_z * GRID * GRID + src_y * GRID + src_x;

    let dst_idx = params.chunk_index * CHUNK_VOL + gid.z * CHUNK * CHUNK + gid.y * CHUNK + gid.x;

    pack_buf[dst_idx] = voxels[src_idx];
}
