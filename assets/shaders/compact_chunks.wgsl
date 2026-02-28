struct IndirectArgs {
    x: atomic<u32>,
    y: u32,
    z: u32,
}

@group(0) @binding(0) var<storage, read> dirty_chunks: array<u32>;
@group(0) @binding(1) var<storage, read_write> active_chunks: array<u32>;
@group(0) @binding(2) var<storage, read_write> indirect_args: IndirectArgs;

@compute @workgroup_size(256, 1, 1)
fn compact(@builtin(global_invocation_id) gid: vec3u) {
    let chunk_idx = gid.x;
    if (chunk_idx >= 262144u) {
        return;
    }

    let word_idx = chunk_idx >> 5u;
    let bit_idx = chunk_idx & 31u;

    let word = dirty_chunks[word_idx];
    if ((word & (1u << bit_idx)) != 0u) {
        let insert_idx = atomicAdd(&indirect_args.x, 1u);
        active_chunks[insert_idx] = chunk_idx;
    }
}
