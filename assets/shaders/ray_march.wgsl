// Compute ray marcher — two-level DDA through the Sparse Brick Map (LOD0 active region).
//
// Dispatch: ceil(width/8) × ceil(height/8) × 1 workgroups
// Each thread: one pixel → one ray → DDA march → write VisibilityPayload + depth + velocity

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

struct VisibilityPayload {
    lo: u32,
    hi: u32,
}

struct ClipOrigin {
    x: i32,
    y: i32,
    z: i32,
    cell_size: i32,
}

struct ClipUniforms {
    origins: array<ClipOrigin, 6>,
}

const BRICK_SIZE: u32 = 8u;
const BRICK_SHIFT: u32 = 3u;
const BRICK_VOLUME: u32 = 512u;
const TOP_GRID_SIZE: u32 = 128u;
const TOP_GRID_VOLUME: u32 = 2097152u;
const WORLD_EXTENT: u32 = 1024u;
const BRICK_EMPTY: u32 = 0xFFFFFFFFu;
const MAX_OUTER_STEPS: u32 = 256u;
const MAX_INNER_STEPS: u32 = 24u;

@group(0) @binding(0) var<uniform> globals: GlobalUniforms;
@group(0) @binding(1) var<storage, read> top_grid: array<u32>;
@group(0) @binding(2) var<storage, read> brick_pool: array<u32>;
@group(0) @binding(3) var<storage, read> brick_occupancy: array<u32>;
@group(0) @binding(5) var<storage, read_write> visibility_buf: array<VisibilityPayload>;
@group(0) @binding(6) var<storage, read_write> depth_buf: array<f32>;
@group(0) @binding(7)  var velocity_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8)  var cascade0: texture_3d<u32>;
@group(0) @binding(9)  var cascade1: texture_3d<u32>;
@group(0) @binding(10) var cascade2: texture_3d<u32>;
@group(0) @binding(11) var cascade3: texture_3d<u32>;
@group(0) @binding(12) var cascade4: texture_3d<u32>;
@group(0) @binding(13) var cascade5: texture_3d<u32>;
@group(0) @binding(14) var<uniform> clip: ClipUniforms;

// ============================================================
// SBM helpers
// ============================================================

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

fn top_grid_index(bx: u32, by: u32, bz: u32) -> u32 {
    return (bz % TOP_GRID_SIZE) * TOP_GRID_SIZE * TOP_GRID_SIZE
         + (by % TOP_GRID_SIZE) * TOP_GRID_SIZE
         + (bx % TOP_GRID_SIZE);
}

fn brick_is_occupied(brick_idx: u32, lx: u32, ly: u32, lz: u32) -> bool {
    let morton = morton_encode(lx, ly, lz);
    let word = brick_occupancy[brick_idx * 16u + (morton >> 5u)];
    return (word & (1u << (morton & 31u))) != 0u;
}

fn read_voxel_material(brick_idx: u32, lx: u32, ly: u32, lz: u32) -> u32 {
    let morton = morton_encode(lx, ly, lz);
    let linear = brick_idx * BRICK_VOLUME + morton;
    let word = brick_pool[linear >> 1u];
    let shift = (linear & 1u) * 16u;
    return (word >> shift) & 0x7FFu;  // material_id(9) | variant(2)
}

fn pack_visibility(material_id: u32, normal_idx: u32, depth: f32, voxel_pos: vec3u) -> VisibilityPayload {
    let depth_fixed = u32(clamp(depth * 8.0, 0.0, 8191.0));
    let lo = (material_id & 0xFFFFu) | ((normal_idx & 0x7u) << 16u) | ((depth_fixed & 0x1FFFu) << 19u);
    let hi = (voxel_pos.x & 0x3FFu)
           | ((voxel_pos.y & 0x3FFu) << 10u)
           | ((voxel_pos.z & 0x3FFu) << 20u);
    return VisibilityPayload(lo, hi);
}

fn get_normal_index(axis: i32, step_sign: f32) -> u32 {
    let base = u32(axis) * 2u;
    return select(base, base + 1u, step_sign < 0.0);
}

// ============================================================
// Clipmap helpers
// ============================================================

fn cascade_load(idx: u32, coords: vec3i) -> vec4u {
    switch idx {
        case 0u:    { return textureLoad(cascade0, coords, 0); }
        case 1u:    { return textureLoad(cascade1, coords, 0); }
        case 2u:    { return textureLoad(cascade2, coords, 0); }
        case 3u:    { return textureLoad(cascade3, coords, 0); }
        case 4u:    { return textureLoad(cascade4, coords, 0); }
        default:    { return textureLoad(cascade5, coords, 0); }
    }
}

// Single-cascade DDA. Returns hit distance, or -1.0 on miss.
// Writes visibility_buf and depth_buf on hit; velocity is written by the caller.
fn clipmap_cell_march(
    ray_origin: vec3f,
    ray_dir:    vec3f,
    abs_dir:    vec3f,
    step:       vec3i,
    t_start:    f32,
    cascade_idx: u32,
    pixel_idx:  u32,
) -> f32 {
    let o  = clip.origins[cascade_idx];
    let cs = f32(o.cell_size);
    let cell_origin = vec3i(o.x, o.y, o.z);
    let world_center = vec3f(cell_origin) * cs;
    let half_ext = cs * 256.0;

    // Ray-AABB intersection in voxel space.
    let cmin = world_center - vec3f(half_ext);
    let cmax = world_center + vec3f(half_ext);
    let inv_dir = 1.0 / ray_dir;
    let t0 = (cmin - ray_origin) * inv_dir;
    let t1 = (cmax - ray_origin) * inv_dir;
    let t_enter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let t_exit  = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    if t_enter >= t_exit || t_exit <= t_start { return -1.0; }

    let t_begin = max(t_enter + cs * 0.001, t_start);

    // DDA in cell space. t_max values are absolute t from ray origin.
    let cell_inv = cs / max(abs_dir, vec3f(1e-10));
    let start_world  = ray_origin + ray_dir * t_begin;
    let start_cell_f = start_world / cs;
    var cell = vec3i(floor(start_cell_f));
    let frac = start_cell_f - floor(start_cell_f);
    var t_max = t_begin + vec3f(
        select(frac.x, 1.0 - frac.x, step.x > 0),
        select(frac.y, 1.0 - frac.y, step.y > 0),
        select(frac.z, 1.0 - frac.z, step.z > 0),
    ) * cell_inv;
    let t_delta = cell_inv;
    var t_cur = t_begin;

    var last_axis: i32 = -1;
    for (var i = 0u; i < 1000u; i++) {
        let rel = cell - cell_origin;
        if any(rel < vec3i(-256)) || any(rel >= vec3i(256)) { break; }

        // Toroidal lookup — matches build shader: ((global_cell % 512) + 512) % 512
        let tor = ((cell % 512) + 512) % 512;
        let texel = cascade_load(cascade_idx, tor);
        let mat = texel.r;

        if mat != 0u {
            var normal_idx: u32;
            if last_axis < 0 {
                // No prior DDA step — use stored gradient normal as fallback
                normal_idx = texel.g & 0x7u;
            } else {
                // Entry face: invert step sign (ray entering -Y face has step.y<0 → normal +Y)
                let s = select(1.0, -1.0, step[last_axis] > 0);
                normal_idx = get_normal_index(last_axis, s);
            }
            // Scale cell to world-voxel coords for consistent hash scale across cascades
            let vp = vec3u((cell * i32(o.cell_size)) & vec3i(0x3FF));
            visibility_buf[pixel_idx] = pack_visibility(mat, normal_idx, t_cur, vp);
            depth_buf[pixel_idx] = t_cur;
            return t_cur;
        }

        if t_max.x <= min(t_max.y, t_max.z) {
            t_cur = t_max.x; cell.x += step.x; t_max.x += t_delta.x; last_axis = 0;
        } else if t_max.y <= t_max.z {
            t_cur = t_max.y; cell.y += step.y; t_max.y += t_delta.y; last_axis = 1;
        } else {
            t_cur = t_max.z; cell.z += step.z; t_max.z += t_delta.z; last_axis = 2;
        }
    }
    return -1.0;
}

// Tries cascades 0→3; each cascade starts from the exit t of the previous to avoid
// re-marching inner regions at coarser resolution.
fn clipmap_march(
    ray_origin: vec3f,
    ray_dir:    vec3f,
    abs_dir:    vec3f,
    step:       vec3i,
    t_start:    f32,
    pixel_idx:  u32,
) -> f32 {
    var t = t_start;
    for (var i = 0u; i < 6u; i++) {
        let o    = clip.origins[i];
        let cs_f = f32(o.cell_size);
        let wctr = vec3f(f32(o.x), f32(o.y), f32(o.z)) * cs_f;
        let half = cs_f * 256.0;
        let inv_d = 1.0 / ray_dir;
        let t0c = (wctr - vec3f(half) - ray_origin) * inv_d;
        let t1c = (wctr + vec3f(half) - ray_origin) * inv_d;
        let t_exit_i = min(min(max(t0c.x, t1c.x), max(t0c.y, t1c.y)), max(t0c.z, t1c.z));
        let hit = clipmap_cell_march(ray_origin, ray_dir, abs_dir, step, t, i, pixel_idx);
        if hit >= 0.0 { return hit; }
        t = max(t, t_exit_i);
    }
    return -1.0;
}

// ============================================================
// Two-level DDA: outer (brick grid) + inner (voxel within brick)
// Returns hit distance, or -1.0 on miss.
// ============================================================

fn march(
    ray_origin: vec3f,
    ray_dir: vec3f,
    abs_dir: vec3f,
    step: vec3i,
    t_start: f32,
    start_pos: vec3f,
    pixel_idx: u32,
    grid_origin: vec3i,
) -> f32 {
    let brick_scale = f32(BRICK_SIZE);
    let brick_inv = 1.0 / max(abs_dir, vec3f(1e-10));

    let brick_pos = start_pos / brick_scale;
    var cell = vec3i(floor(brick_pos));
    let frac = brick_pos - floor(brick_pos);
    var t_max = vec3f(
        select(frac.x, 1.0 - frac.x, step.x > 0),
        select(frac.y, 1.0 - frac.y, step.y > 0),
        select(frac.z, 1.0 - frac.z, step.z > 0),
    ) * brick_inv * brick_scale;
    let t_delta = brick_inv * brick_scale;

    var t_current = t_start;

    for (var outer = 0u; outer < MAX_OUTER_STEPS; outer++) {
        if any(cell < grid_origin) || any(cell >= grid_origin + vec3i(i32(TOP_GRID_SIZE))) {
            break;
        }

        let brick_idx = top_grid[top_grid_index(u32(cell.x), u32(cell.y), u32(cell.z))];

        if (brick_idx != BRICK_EMPTY) {
            let brick_world_origin = vec3f(vec3i(cell)) * brick_scale;
            let entry_pos = ray_origin + ray_dir * t_current;
            let local_pos = clamp(
                entry_pos - brick_world_origin,
                vec3f(0.0),
                vec3f(brick_scale - 0.001),
            );

            var voxel = vec3i(floor(local_pos));
            let vfrac = local_pos - floor(local_pos);
            let voxel_inv = 1.0 / max(abs_dir, vec3f(1e-10));
            var vt_max = vec3f(
                select(vfrac.x, 1.0 - vfrac.x, step.x > 0),
                select(vfrac.y, 1.0 - vfrac.y, step.y > 0),
                select(vfrac.z, 1.0 - vfrac.z, step.z > 0),
            ) * voxel_inv;
            let vt_delta = voxel_inv;

            var inner_axis: i32 = -1;

            for (var inner = 0u; inner < MAX_INNER_STEPS; inner++) {
                if (voxel.x < 0 || voxel.y < 0 || voxel.z < 0 ||
                    voxel.x >= i32(BRICK_SIZE) || voxel.y >= i32(BRICK_SIZE) || voxel.z >= i32(BRICK_SIZE)) {
                    break;
                }

                if (brick_is_occupied(brick_idx, u32(voxel.x), u32(voxel.y), u32(voxel.z))) {
                    let mat_id = read_voxel_material(brick_idx, u32(voxel.x), u32(voxel.y), u32(voxel.z));
                    if (mat_id & 0x1FFu) != 0u {
                        let world_voxel = vec3u(vec3i(cell) * i32(BRICK_SIZE) + voxel);
                        let dist = length(vec3f(world_voxel) + 0.5 - ray_origin);

                        var normal_idx = 2u;
                        if (inner_axis >= 0) {
                            let s = select(1.0, -1.0, (vec3f(f32(step.x), f32(step.y), f32(step.z)))[inner_axis] > 0.0);
                            normal_idx = get_normal_index(inner_axis, s);
                        }

                        visibility_buf[pixel_idx] = pack_visibility(mat_id, normal_idx, dist, world_voxel);
                        return dist;
                    }
                }

                if (vt_max.x <= min(vt_max.y, vt_max.z)) {
                    voxel.x += step.x;
                    vt_max.x += vt_delta.x;
                    inner_axis = 0;
                } else if (vt_max.y <= vt_max.z) {
                    voxel.y += step.y;
                    vt_max.y += vt_delta.y;
                    inner_axis = 1;
                } else {
                    voxel.z += step.z;
                    vt_max.z += vt_delta.z;
                    inner_axis = 2;
                }
            }
        }

        if (t_max.x <= min(t_max.y, t_max.z)) {
            t_current = t_max.x;
            cell.x += step.x;
            t_max.x += t_delta.x;
        } else if (t_max.y <= t_max.z) {
            t_current = t_max.y;
            cell.y += step.y;
            t_max.y += t_delta.y;
        } else {
            t_current = t_max.z;
            cell.z += step.z;
            t_max.z += t_delta.z;
        }
    }

    return -1.0;
}

// ============================================================
// Entry point
// ============================================================

@compute @workgroup_size(8, 8, 1)
fn ray_march(@builtin(global_invocation_id) gid: vec3u) {
    let px = gid.x;
    let py = gid.y;
    let width = u32(globals.resolution_x);
    let height = u32(globals.resolution_y);
    if (px >= width || py >= height) { return; }

    let pixel_idx = py * width + px;
    let screen_pos = vec2i(i32(px), i32(py));

    let ndc = vec2f(
        (f32(px) + 0.5) / f32(width) * 2.0 - 1.0,
        1.0 - (f32(py) + 0.5) / f32(height) * 2.0,
    );

    let clip_near = vec4f(ndc, 0.0, 1.0);
    let clip_far  = vec4f(ndc, 1.0, 1.0);
    var world_near = globals.view_inverse * (globals.proj_inverse * clip_near);
    var world_far  = globals.view_inverse * (globals.proj_inverse * clip_far);
    world_near /= world_near.w;
    world_far  /= world_far.w;

    let ray_origin = world_near.xyz;
    let ray_dir = normalize(world_far.xyz - world_near.xyz);
    let abs_dir = abs(ray_dir);
    let step = vec3i(sign(ray_dir));

    let wo = vec3i(globals.world_origin.xyz);
    let grid_corner = wo - vec3i(i32(TOP_GRID_SIZE) / 2);
    let world_min = vec3f(grid_corner * i32(BRICK_SIZE));
    let world_max = vec3f((grid_corner + vec3i(i32(TOP_GRID_SIZE))) * i32(BRICK_SIZE));
    let inv_dir = 1.0 / ray_dir;
    let t0 = (world_min - ray_origin) * inv_dir;
    let t1 = (world_max - ray_origin) * inv_dir;
    let t_enter_v = min(t0, t1);
    let t_exit_v  = max(t0, t1);
    let t_enter = max(max(t_enter_v.x, t_enter_v.y), t_enter_v.z);
    let t_exit  = min(min(t_exit_v.x, t_exit_v.y), t_exit_v.z);

    // SBM march (skip if AABB missed entirely)
    var sbm_hit = -1.0;
    if t_enter < t_exit && t_exit >= 0.0 {
        let t_start = max(t_enter + 0.001, 0.0);
        let start_pos = ray_origin + ray_dir * t_start;
        sbm_hit = march(ray_origin, ray_dir, abs_dir, step, t_start, start_pos, pixel_idx, grid_corner);
    }

    if sbm_hit >= 0.0 {
        depth_buf[pixel_idx] = sbm_hit;
        let hit_pos = ray_origin + ray_dir * sbm_hit;
        let curr_clip = globals.curr_view_proj * vec4f(hit_pos, 1.0);
        let prev_clip = globals.prev_view_proj * vec4f(hit_pos, 1.0);
        let velocity = (curr_clip.xy / curr_clip.w - prev_clip.xy / prev_clip.w) * 0.5;
        textureStore(velocity_texture, screen_pos, vec4f(velocity, 0.0, 0.0));
        return;
    }

    // SBM miss → try clipmap cascades starting from SBM exit (or 0 if AABB not hit)
    let clip_hit = clipmap_march(ray_origin, ray_dir, abs_dir, step, max(t_exit, 0.0), pixel_idx);
    if clip_hit >= 0.0 {
        let hit_pos = ray_origin + ray_dir * clip_hit;
        let curr_clip = globals.curr_view_proj * vec4f(hit_pos, 1.0);
        let prev_clip = globals.prev_view_proj * vec4f(hit_pos, 1.0);
        let velocity = (curr_clip.xy / curr_clip.w - prev_clip.xy / prev_clip.w) * 0.5;
        textureStore(velocity_texture, screen_pos, vec4f(velocity, 0.0, 0.0));
        return;
    }

    // Sky fallback
    visibility_buf[pixel_idx] = VisibilityPayload(0u, 0u);
    depth_buf[pixel_idx] = -1.0;
    textureStore(velocity_texture, screen_pos, vec4f(0.0));
}
