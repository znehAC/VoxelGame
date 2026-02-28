const BRICK_SIZE: u32 = 8u;
const BRICK_VOLUME: u32 = 512u;
const TOP_GRID_SIZE: u32 = 64u;
const BRICK_EMPTY: u32 = 0xFFFFFFFFu;

var<push_constant> target_lod: u32;

@group(0) @binding(0) var<storage, read> top_grid: array<u32>;
@group(0) @binding(1) var<storage, read_write> brick_pool: array<u32>;
@group(0) @binding(2) var<storage, read> brick_headers: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> radiance_pool: array<vec2<u32>>;

fn top_grid_index(lod: u32, bx: i32, by: i32, bz: i32) -> u32 {
    let tgs = i32(TOP_GRID_SIZE);
    let x = u32((bx % tgs + tgs) % tgs);
    let y = u32((by % tgs + tgs) % tgs);
    let z = u32((bz % tgs + tgs) % tgs);
    let vol = TOP_GRID_SIZE * TOP_GRID_SIZE * TOP_GRID_SIZE;
    return lod * vol + z * TOP_GRID_SIZE * TOP_GRID_SIZE + y * TOP_GRID_SIZE + x;
}

fn read_voxel_opacity(brick_idx: u32, lx: u32, ly: u32, lz: u32) -> f32 {
    let linear = lz * 64u + ly * 8u + lx;
    let word = brick_pool[brick_idx * 256u + (linear >> 1u)];
    return select(0.0, 1.0, ((word >> ((linear & 1u) * 16u)) & 0x1FFu) != 0u);
}

fn read_radiance(brick_idx: u32, lx: u32, ly: u32, lz: u32) -> vec3f {
    let packed = radiance_pool[brick_idx * BRICK_VOLUME + lz * 64u + ly * 8u + lx];
    return vec3f(unpack2x16float(packed.x), unpack2x16float(packed.y).x);
}

fn compute_mipmap_voxel(dst_brick_idx: u32, lx: u32, ly: u32, lz: u32, bx: i32, by: i32, bz: i32, src_lod: u32) -> vec4f {
    let src_base_brick = vec3i(bx * 2, by * 2, bz * 2);
    let src_base_local = vec3u(lx * 2u, ly * 2u, lz * 2u);

    var max_opacity = 0.0;
    var solid_count = 0.0;
    var accum_radiance = vec3f(0.0);

    for (var dz = 0u; dz < 2u; dz++) {
        for (var dy = 0u; dy < 2u; dy++) {
            for (var dx = 0u; dx < 2u; dx++) {
                let src_local = src_base_local + vec3u(dx, dy, dz);
                let src_brick_offset = vec3i(i32(src_local.x / BRICK_SIZE), i32(src_local.y / BRICK_SIZE), i32(src_local.z / BRICK_SIZE));
                let src_brick_coord = src_base_brick + src_brick_offset;
                let src_local_in_brick = vec3u(src_local.x % BRICK_SIZE, src_local.y % BRICK_SIZE, src_local.z % BRICK_SIZE);

                let src_grid_idx = top_grid_index(src_lod, src_brick_coord.x, src_brick_coord.y, src_brick_coord.z);
                let src_brick_idx = top_grid[src_grid_idx];

                if (src_brick_idx != BRICK_EMPTY) {
                    let opacity = read_voxel_opacity(src_brick_idx, src_local_in_brick.x, src_local_in_brick.y, src_local_in_brick.z);
                    max_opacity = max(max_opacity, opacity);
                    solid_count += opacity;
                    accum_radiance += read_radiance(src_brick_idx, src_local_in_brick.x, src_local_in_brick.y, src_local_in_brick.z) * opacity;
                }
            }
        }
    }

    let final_opacity = mix(max_opacity, solid_count / 8.0, 0.25);
    let final_radiance = accum_radiance / max(solid_count, 1.0);
    return vec4f(final_radiance, final_opacity);
}

@compute @workgroup_size(8, 8, 4)
fn mipmap(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    // Mask out half the threads to prevent read-modify-write data races on u32 packaging
    if ((lid.x & 1u) != 0u) { return; }

    let dst_brick_idx = wg.x;
    let header = brick_headers[dst_brick_idx];
    if (header.x == 0xFFFFFFFFu) { return; }

    let packed_coord = header.x & 0x3FFFFFFFu;
    let bx = i32(packed_coord & 0x3FFu);
    let by = i32((packed_coord >> 10u) & 0x3FFu);
    let bz = i32((packed_coord >> 20u) & 0x3FFu);

    let dst_lod = target_lod;
    if (top_grid[top_grid_index(dst_lod, bx, by, bz)] != dst_brick_idx) { return; }

    let ly = lid.y;
    let lz = wg.y * 4u + lid.z;
    let lx0 = lid.x;
    let lx1 = lid.x + 1u;

    // Process both horizontal voxels simultaneously 
    let res0 = compute_mipmap_voxel(dst_brick_idx, lx0, ly, lz, bx, by, bz, dst_lod - 1u);
    let res1 = compute_mipmap_voxel(dst_brick_idx, lx1, ly, lz, bx, by, bz, dst_lod - 1u);

    let mat_id0 = select(0u, 1u, res0.a > 0.1);
    let mat_id1 = select(0u, 1u, res1.a > 0.1);
    
    // Write full u32 word, bypassing race condition
    let linear0 = lz * 64u + ly * 8u + lx0;
    brick_pool[dst_brick_idx * 256u + (linear0 >> 1u)] = mat_id0 | (mat_id1 << 16u);

    radiance_pool[dst_brick_idx * 512u + linear0] = vec2u(pack2x16float(res0.rg), pack2x16float(vec2f(res0.b, res0.a)));
    radiance_pool[dst_brick_idx * 512u + linear0 + 1u] = vec2u(pack2x16float(res1.rg), pack2x16float(vec2f(res1.b, res1.a)));
}