struct IndirectArgs {
    x: atomic<u32>,
    y: u32,
    z: u32,
}

@group(0) @binding(0) var<storage, read> dirty_chunks: array<u32>;
@group(0) @binding(1) var<storage, read_write> active_chunks: array<u32>;
@group(0) @binding(2) var<storage, read_write> indirect_args: IndirectArgs;

var<workgroup> wg_count: atomic<u32>;
var<workgroup> wg_offset: u32;
var<workgroup> local_indices: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn compact(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lid: u32) {
    let chunk_idx = gid.x;

    if (lid == 0u) {
        atomicStore(&wg_count, 0u);
    }
    workgroupBarrier();

    var is_dirty = false;
    var local_slot = 0u;
    if (chunk_idx < 262144u) {
        let word_idx = chunk_idx >> 5u;
        let bit_idx = chunk_idx & 31u;
        let word = dirty_chunks[word_idx];
        is_dirty = (word & (1u << bit_idx)) != 0u;
    }

    if (is_dirty) {
        local_slot = atomicAdd(&wg_count, 1u);
        local_indices[local_slot] = chunk_idx;
    }
    workgroupBarrier();

    // One atomic per workgroup instead of per dirty chunk
    if (lid == 0u) {
        let count = atomicLoad(&wg_count);
        if (count > 0u) {
            wg_offset = atomicAdd(&indirect_args.x, count);
        }
    }
    workgroupBarrier();

    if (is_dirty) {
        active_chunks[wg_offset + local_slot] = local_indices[local_slot];
    }
}
