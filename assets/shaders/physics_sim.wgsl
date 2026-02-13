@group(0) @binding(0) var<storage, read> voxels_src: array<u32>;
@group(0) @binding(1) var<storage, read_write> voxels_dst: array<u32>;
@group(0) @binding(2) var<storage, read_write> dirty_chunks: array<u32>;

@compute @workgroup_size(4, 4, 4)
fn simulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Placeholder: No-op for now
    return;
}
