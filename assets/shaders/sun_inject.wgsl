struct GlobalUniforms {
    view_inverse: mat4x4f,
    proj_inverse: mat4x4f,
    cam_pos: vec4f,
    time: f32,
    resolution_x: f32,
    resolution_y: f32,
    sun_shadow_max: f32,
    sun_dir: vec4f,
    sun_color: vec4f,
    sky_color: vec4f,
    ground_color: vec4f,
    selected_block: vec4f,
    prev_view_proj: mat4x4f,
    curr_view_proj: mat4x4f,
    world_origin: vec4f,
}

const BRICK_SIZE: u32 = 8u;
const BRICK_SHIFT: u32 = 3u;
const BRICK_VOLUME: u32 = 512u;
const TOP_GRID_SIZE: u32 = 64u;
const WORLD_EXTENT: u32 = 512u;
const BRICK_EMPTY: u32 = 0xFFFFFFFFu;
const MAX_SHADOW_STEPS: u32 = 64u;

@group(0) @binding(0) var<storage, read> top_grid: array<u32>;
@group(0) @binding(1) var<storage, read> brick_pool: array<u32>;
@group(0) @binding(2) var<storage, read> brick_headers: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> radiance_pool: array<vec2<u32>>;
@group(0) @binding(4) var<uniform> globals: GlobalUniforms;
@group(0) @binding(5) var t_palette: texture_2d_array<f32>;
@group(0) @binding(6) var<storage, read> brick_occupancy: array<u32>;

fn top_grid_index(bx: u32, by: u32, bz: u32) -> u32 {
    let tgs = TOP_GRID_SIZE;
    return (bz % tgs) * tgs * tgs + (by % tgs) * tgs + (bx % tgs);
}

// Sign-extend a 10-bit unsigned value to i32 (bit 9 is the sign bit).
fn sign10(v: u32) -> i32 {
    let n = i32(v & 0x3FFu);
    return select(n, n - 1024, (v & 0x200u) != 0u);
}

fn sbm_read_voxel(world_pos: vec3i) -> u32 {
    if (world_pos.x < 0 || world_pos.y < 0 || world_pos.z < 0 ||
        world_pos.x >= i32(WORLD_EXTENT) || world_pos.y >= i32(WORLD_EXTENT) || world_pos.z >= i32(WORLD_EXTENT)) {
        return 0u;
    }
    let bx = u32(world_pos.x) >> BRICK_SHIFT;
    let by = u32(world_pos.y) >> BRICK_SHIFT;
    let bz = u32(world_pos.z) >> BRICK_SHIFT;
    let brick_idx = top_grid[top_grid_index(bx, by, bz)];
    if (brick_idx == BRICK_EMPTY) { return 0u; }
    
    let lx = u32(world_pos.x) & 7u;
    let ly = u32(world_pos.y) & 7u;
    let lz = u32(world_pos.z) & 7u;
    let linear = brick_idx * BRICK_VOLUME + lz * 64u + ly * 8u + lx;
    let word = brick_pool[linear >> 1u];
    return (word >> ((linear & 1u) * 16u)) & 0xFFFFu;
}

fn is_opaque(pos: vec3i) -> bool {
    return (sbm_read_voxel(pos) & 0x1FFu) != 0u;
}

fn fast_neighbor_opaque(brick_idx: u32, world_pos: vec3i, lx: i32, ly: i32, lz: i32) -> bool {
    if (lx >= 0 && lx < 8 && ly >= 0 && ly < 8 && lz >= 0 && lz < 8) {
        let linear = u32(lz) * 64u + u32(ly) * 8u + u32(lx);
        let word = brick_occupancy[brick_idx * 16u + (linear >> 5u)];
        return (word & (1u << (linear & 31u))) != 0u;
    }
    return is_opaque(world_pos);
}

fn estimate_normal_fast(brick_idx: u32, w: vec3i, lx: i32, ly: i32, lz: i32) -> vec3f {
    var n = vec3f(0.0);
    if (!fast_neighbor_opaque(brick_idx, w + vec3i(1, 0, 0), lx + 1, ly, lz)) { n.x += 1.0; }
    if (!fast_neighbor_opaque(brick_idx, w - vec3i(1, 0, 0), lx - 1, ly, lz)) { n.x -= 1.0; }
    if (!fast_neighbor_opaque(brick_idx, w + vec3i(0, 1, 0), lx, ly + 1, lz)) { n.y += 1.0; }
    if (!fast_neighbor_opaque(brick_idx, w - vec3i(0, 1, 0), lx, ly - 1, lz)) { n.y -= 1.0; }
    if (!fast_neighbor_opaque(brick_idx, w + vec3i(0, 0, 1), lx, ly, lz + 1)) { n.z += 1.0; }
    if (!fast_neighbor_opaque(brick_idx, w - vec3i(0, 0, 1), lx, ly, lz - 1)) { n.z -= 1.0; }
    
    let sq = dot(n, n);
    return select(n / sqrt(sq), vec3f(0.0, 1.0, 0.0), sq < 0.001);
}

fn check_sun_path_dda(origin: vec3i, dir: vec3f) -> bool {
    let inv_dir = 1.0 / dir;
    let step = vec3i(sign(dir));
    let t_delta = abs(inv_dir);
    
    var cell = origin;
    var t_max: vec3f;
    let o_f = vec3f(origin) + 0.5;
    
    if (dir.x > 0.0) { t_max.x = (f32(cell.x + 1) - o_f.x) * inv_dir.x; } else { t_max.x = (o_f.x - f32(cell.x)) * -inv_dir.x; }
    if (dir.y > 0.0) { t_max.y = (f32(cell.y + 1) - o_f.y) * inv_dir.y; } else { t_max.y = (o_f.y - f32(cell.y)) * -inv_dir.y; }
    if (dir.z > 0.0) { t_max.z = (f32(cell.z + 1) - o_f.z) * inv_dir.z; } else { t_max.z = (o_f.z - f32(cell.z)) * -inv_dir.z; }

    for (var i = 0u; i < MAX_SHADOW_STEPS; i++) {
        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) { cell.x += step.x; t_max.x += t_delta.x; } 
            else { cell.z += step.z; t_max.z += t_delta.z; }
        } else {
            if (t_max.y < t_max.z) { cell.y += step.y; t_max.y += t_delta.y; } 
            else { cell.z += step.z; t_max.z += t_delta.z; }
        }

        if (cell.x < 0 || cell.y < 0 || cell.z < 0 || cell.x >= i32(WORLD_EXTENT) || cell.y >= i32(WORLD_EXTENT) || cell.z >= i32(WORLD_EXTENT)) { return true; }
        if (is_opaque(cell)) { return false; }
    }
    return true;
}

@compute @workgroup_size(8, 8, 4)
fn inject(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let brick_idx = wg.x + wg.z * 65535u;
    if (brick_idx >= arrayLength(&brick_headers)) { return; }
    let header = brick_headers[brick_idx];
    if (header.x == 0xFFFFFFFFu || ((header.x >> 30u) & 0x3u) != 0u) { return; }

    let packed_coord = header.x & 0x3FFFFFFFu;
    let bx: i32 = sign10(packed_coord);
    let by: i32 = sign10(packed_coord >> 10u);
    let bz: i32 = sign10(packed_coord >> 20u);

    let lx = i32(lid.x);
    let ly = i32(lid.y);
    let lz = i32(wg.y * 4u + lid.z);

    let linear = u32(lz) * 64u + u32(ly) * 8u + u32(lx);
    let occ_word = brick_occupancy[brick_idx * 16u + (linear >> 5u)];

    if ((occ_word & (1u << (linear & 31u))) == 0u) {
        radiance_pool[brick_idx * 512u + linear] = vec2u(0u, 0u);
        return;
    }

    let world_pos = vec3i(bx * 8 + lx, by * 8 + ly, bz * 8 + lz);
    let mat_id = sbm_read_voxel(world_pos) & 0x1FFu;

    let is_surface =
        !fast_neighbor_opaque(brick_idx, world_pos + vec3i(1, 0, 0), lx + 1, ly, lz) ||
        !fast_neighbor_opaque(brick_idx, world_pos - vec3i(1, 0, 0), lx - 1, ly, lz) ||
        !fast_neighbor_opaque(brick_idx, world_pos + vec3i(0, 1, 0), lx, ly + 1, lz) ||
        !fast_neighbor_opaque(brick_idx, world_pos - vec3i(0, 1, 0), lx, ly - 1, lz) ||
        !fast_neighbor_opaque(brick_idx, world_pos + vec3i(0, 0, 1), lx, ly, lz + 1) ||
        !fast_neighbor_opaque(brick_idx, world_pos - vec3i(0, 0, 1), lx, ly, lz - 1);

    let albedo = textureLoad(t_palette, vec2i(i32(mat_id % 256u), i32(mat_id / 256u)), 0, 0).rgb;
    let mat_props = textureLoad(t_palette, vec2i(i32(mat_id % 256u), i32(mat_id / 256u)), 1, 0);
    let emission = mat_props.g;

    var radiance = albedo * emission * 5.0;

    if (is_surface) {
        let normal = estimate_normal_fast(brick_idx, world_pos, lx, ly, lz);
        let ndotl = max(dot(normal, -globals.sun_dir.xyz), 0.0);
        let sun_contrib = globals.sun_color.rgb * globals.sun_dir.w * ndotl;
        let sky_contrib = globals.sky_color.rgb * globals.sky_color.w;
        radiance += albedo * (sun_contrib + sky_contrib);
    }

    radiance_pool[brick_idx * 512u + linear] = vec2u(pack2x16float(radiance.rg), pack2x16float(vec2f(radiance.b, 1.0)));
}