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

struct VoxelData {
    id: u32,
    variant: u32,
    level: u32,
    flags: u32,
}

struct HitResult {
    hit: bool,
    pos: vec3f,
    normal: vec3f,
    voxel: VoxelData,
    voxel_coord: vec3i,
    local_pos: vec3f,
    lod: u32,
}

struct PointLight {
    position: vec4f,
    color: vec4f,
    flags: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct LightBuffer {
    count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    lights: array<PointLight, 16>,
}

const BRICK_SIZE: u32 = 8u;
const BRICK_SHIFT: u32 = 3u;
const BRICK_VOLUME: u32 = 512u;
const TOP_GRID_SIZE: u32 = 64u;
const LOD_COUNT: u32 = 3u;
const WORLD_EXTENT: u32 = 512u;
const BRICK_EMPTY: u32 = 0xFFFFFFFFu;
const MAX_STEPS: u32 = 512u;
const MAX_INNER_STEPS: u32 = 24u;
const LIGHT_GRID_MASK: i32 = 511;

@group(0) @binding(0) var<uniform> globals: GlobalUniforms;
@group(0) @binding(2) var t_palette: texture_2d_array<f32>;
@group(0) @binding(3) var s_palette: sampler;
@group(0) @binding(4) var<storage, read> top_grid: array<u32>;
@group(0) @binding(5) var<storage, read> brick_pool: array<u32>;
@group(0) @binding(6) var<storage, read> brick_occupancy: array<u32>;
@group(0) @binding(7) var<uniform> point_lights: LightBuffer;
@group(0) @binding(8) var<storage, read> radiance_pool: array<vec2<u32>>;
@group(0) @binding(9) var<storage, read> brick_headers: array<vec4<u32>>;
@group(0) @binding(10) var<storage, read> toroidal_light: array<u32>;
@group(0) @binding(11) var<uniform> toroidal_origin: vec4i;

fn unpack_rgb10(p: u32) -> vec3f {
    let r = f32((p >> 20u) & 0x3FFu);
    let g = f32((p >> 10u) & 0x3FFu);
    let b = f32(p & 0x3FFu);
    return vec3f(r, g, b) * 0.000977517 * 10.0;
}

fn toroidal_idx(world_pos: vec3i) -> u32 {
    let tx = u32(world_pos.x & LIGHT_GRID_MASK);
    let ty = u32(world_pos.y & LIGHT_GRID_MASK);
    let tz = u32(world_pos.z & LIGHT_GRID_MASK);
    return tz * 262144u + ty * 512u + tx; 
}

fn unpack_voxel(packed: u32) -> VoxelData {
    return VoxelData(
        packed & 0x1FFu,
        (packed >> 9u) & 0x3u,
        (packed >> 11u) & 0x7u,
        (packed >> 14u) & 0x3u,
    );
}

fn floor_div(a: i32, b: i32) -> i32 {
    let q = a / b;
    let r = a % b;
    return select(q, q - 1, (r != 0) && ((r < 0) != (b < 0)));
}

fn floor_mod(a: i32, b: i32) -> u32 {
    let r = a % b;
    return u32(select(r, r + b, r < 0));
}

fn top_grid_index(lod: u32, bx: i32, by: i32, bz: i32) -> u32 {
    let tgs = i32(TOP_GRID_SIZE);
    let x = u32((bx % tgs + tgs) % tgs);
    let y = u32((by % tgs + tgs) % tgs);
    let z = u32((bz % tgs + tgs) % tgs);
    let vol = TOP_GRID_SIZE * TOP_GRID_SIZE * TOP_GRID_SIZE;
    return lod * vol + z * TOP_GRID_SIZE * TOP_GRID_SIZE + y * TOP_GRID_SIZE + x;
}

fn brick_local_index(lx: u32, ly: u32, lz: u32) -> u32 {
    return lz * BRICK_SIZE * BRICK_SIZE + ly * BRICK_SIZE + lx;
}

fn read_brick_voxel(brick_idx: u32, lx: u32, ly: u32, lz: u32) -> u32 {
    let linear = brick_idx * BRICK_VOLUME + brick_local_index(lx, ly, lz);
    let word = brick_pool[linear >> 1u];
    let shift = (linear & 1u) * 16u;
    return (word >> shift) & 0xFFFFu;
}

fn sbm_read_voxel_lod(lod: u32, world_pos: vec3i) -> u32 {
    if world_pos.x >= i32(WORLD_EXTENT) || world_pos.y >= i32(WORLD_EXTENT) || world_pos.z >= i32(WORLD_EXTENT) ||
       world_pos.x < -i32(WORLD_EXTENT) || world_pos.y < -i32(WORLD_EXTENT) || world_pos.z < -i32(WORLD_EXTENT) {
        return 0u;
    }
    let bs = i32(BRICK_SIZE);
    let bx = floor_div(world_pos.x, bs);
    let by = floor_div(world_pos.y, bs);
    let bz = floor_div(world_pos.z, bs);
    let brick_idx = top_grid[top_grid_index(lod, bx, by, bz)];
    if brick_idx == BRICK_EMPTY {
        return 0u;
    }
    
    let lx = u32(world_pos.x) & 7u;
    let ly = u32(world_pos.y) & 7u;
    let lz = u32(world_pos.z) & 7u;
    let linear = brick_idx * BRICK_VOLUME + brick_local_index(lx, ly, lz);
    let word = brick_pool[linear >> 1u];
    let shift = (linear & 1u) * 16u;
    return (word >> shift) & 0xFFFFu;
}

fn get_voxel_at_lod(lod: u32, pos: vec3i) -> bool {
    return (sbm_read_voxel_lod(lod, pos) & 0x1FFu) != 0u;
}

fn vertex_ao(side1: f32, side2: f32, corner: f32) -> f32 {
    if side1 > 0.5 && side2 > 0.5 {
        return 0.0;
    }
    return (3.0 - side1 - side2 - corner) / 3.0;
}

fn get_ao(lod: u32, p: vec3i, normal: vec3f, local: vec3f) -> f32 {
    var tang: vec3i;
    var bitang: vec3i;

    var u: f32;
    var v: f32;

    if abs(normal.x) > 0.5 {
        tang = vec3i(0,1,0);
        bitang = vec3i(0,0,1);
        u = local.y;
        v = local.z;
    } else if abs(normal.y) > 0.5 {
        tang = vec3i(1,0,0);
        bitang = vec3i(0,0,1);
        u = local.x;
        v = local.z;
    } else {
        tang = vec3i(1,0,0);
        bitang = vec3i(0,1,0);
        u = local.x;
        v = local.y;
    }

    let base = p;

    let s00 = select(0.0,1.0,get_voxel_at_lod(lod, base - tang - bitang));
    let s10 = select(0.0,1.0,get_voxel_at_lod(lod, base       - bitang));
    let s20 = select(0.0,1.0,get_voxel_at_lod(lod, base + tang - bitang));
    let s01 = select(0.0,1.0,get_voxel_at_lod(lod, base - tang));
    let s21 = select(0.0,1.0,get_voxel_at_lod(lod, base + tang));
    let s02 = select(0.0,1.0,get_voxel_at_lod(lod, base - tang + bitang));
    let s12 = select(0.0,1.0,get_voxel_at_lod(lod, base       + bitang));
    let s22 = select(0.0,1.0,get_voxel_at_lod(lod, base + tang + bitang));

    let ao0 = vertex_ao(s01,s10,s00);
    let ao1 = vertex_ao(s21,s10,s20);
    let ao2 = vertex_ao(s21,s12,s22);
    let ao3 = vertex_ao(s01,s12,s02);

    return pow(mix(mix(ao0,ao1,u), mix(ao3,ao2,u), v), 1.5);
}

fn ray_aabb(origin: vec3f, inv_dir: vec3f, box_min: vec3f, box_max: vec3f) -> vec2f {
    let t0 = (box_min - origin) * inv_dir;
    let t1 = (box_max - origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_enter = max(max(tmin.x, tmin.y), tmin.z);
    let t_exit = min(min(tmax.x, tmax.y), tmax.z);
    return vec2f(t_enter, t_exit);
}

fn dda_march(origin: vec3f, dir: vec3f) -> HitResult {
    var result: HitResult;
    result.hit = false;

    let inv_dir = 1.0 / dir;
    let step = vec3i(sign(dir));
    let step_f = vec3f(step);
    let t_delta_base = abs(inv_dir);

    let cam = globals.cam_pos.xyz;
    var current_t = 0.0;

    for (var lod = 0u; lod < LOD_COUNT; lod++) {
        let voxel_size = f32(1u << lod);
        let brick_size = 8.0 * voxel_size;
        let radius = 256.0 * voxel_size;
        
        let bounds = ray_aabb(origin, inv_dir, cam - vec3f(radius), cam + vec3f(radius));
        let t_enter = max(max(bounds.x, current_t), 0.0);
        let t_exit = bounds.y;

        if (t_enter > t_exit || t_exit < 0.0) { continue; }

        let entry_pos = origin + dir * (t_enter + 0.0001);

        var macro_pos = entry_pos / brick_size;
        var macro_cell = vec3i(floor(macro_pos));

        let t_delta_macro = t_delta_base * brick_size;
        var t_max_macro = (vec3f(macro_cell) + max(step_f, vec3f(0.0)) - macro_pos) * inv_dir * brick_size + t_enter;

        var t_current = t_enter;
        var macro_mask = vec3<bool>(false, false, false);

        for (var outer = 0u; outer < MAX_STEPS; outer++) {
            if (t_current > t_exit) { break; }

            let grid_idx = top_grid_index(lod, macro_cell.x, macro_cell.y, macro_cell.z);
            let brick_idx = top_grid[grid_idx];

            if (brick_idx != BRICK_EMPTY) {
                let inner_entry = origin + dir * (t_current + 0.0001);
                var micro_pos = inner_entry / voxel_size;
                var micro_cell = vec3i(floor(micro_pos));

                let min_cell = macro_cell * 8;
                let max_cell = min_cell + vec3i(7);
                micro_cell = clamp(micro_cell, min_cell, max_cell);

                let t_delta_micro = t_delta_base * voxel_size;
                var t_max_micro = (vec3f(micro_cell) + max(step_f, vec3f(0.0)) - micro_pos) * inv_dir * voxel_size + t_current;

                var inner_mask = macro_mask;
                
                var inside_brick = true;
                while (inside_brick) {
                    let lx = u32(micro_cell.x) & 7u;
                    let ly = u32(micro_cell.y) & 7u;
                    let lz = u32(micro_cell.z) & 7u;
                    
                    let linear_idx = lz * 64u + ly * 8u + lx;
                    let word_idx = linear_idx >> 5u;
                    let bit_idx = linear_idx & 31u;
                    
                    let occ_word = brick_occupancy[brick_idx * 16u + word_idx];
                    
                    if ((occ_word & (1u << bit_idx)) != 0u) {
                        let packed = read_brick_voxel(brick_idx, lx, ly, lz);
                        
                        result.hit = true;
                        result.voxel = unpack_voxel(packed);
                        result.voxel_coord = micro_cell * i32(1u << lod);
                        result.lod = lod;

                        var normal = vec3f(0.0);
                        if (inner_mask.x) { normal.x = -step_f.x; }
                        else if (inner_mask.y) { normal.y = -step_f.y; }
                        else if (inner_mask.z) { normal.z = -step_f.z; }
                        else {
                            let a = abs(dir);
                            if (a.x >= a.y && a.x >= a.z) { normal.x = -step_f.x; }
                            else if (a.y >= a.x && a.y >= a.z) { normal.y = -step_f.y; }
                            else { normal.z = -step_f.z; }
                        }
                        result.normal = normal;

                        var t_hit = t_current;
                        if (inner_mask.x) { t_hit = t_max_micro.x - t_delta_micro.x; }
                        else if (inner_mask.y) { t_hit = t_max_micro.y - t_delta_micro.y; }
                        else if (inner_mask.z) { t_hit = t_max_micro.z - t_delta_micro.z; }

                        result.pos = origin + dir * t_hit;
                        result.local_pos = result.pos - vec3f(result.voxel_coord);
                        return result;
                    }

                    inner_mask = t_max_micro.xyz <= min(t_max_micro.yzx, t_max_micro.zxy);
                    
                    if (inner_mask.x) { t_current = t_max_micro.x; t_max_micro.x += t_delta_micro.x; micro_cell.x += step.x; }
                    else if (inner_mask.y) { t_current = t_max_micro.y; t_max_micro.y += t_delta_micro.y; micro_cell.y += step.y; }
                    else { t_current = t_max_micro.z; t_max_micro.z += t_delta_micro.z; micro_cell.z += step.z; }

                    if (micro_cell.x < min_cell.x || micro_cell.x > max_cell.x ||
                        micro_cell.y < min_cell.y || micro_cell.y > max_cell.y ||
                        micro_cell.z < min_cell.z || micro_cell.z > max_cell.z) {
                        inside_brick = false;
                    }
                }
            }

            macro_mask = t_max_macro.xyz <= min(t_max_macro.yzx, t_max_macro.zxy);
            
            if (macro_mask.x) { t_current = t_max_macro.x; t_max_macro.x += t_delta_macro.x; macro_cell.x += step.x; }
            else if (macro_mask.y) { t_current = t_max_macro.y; t_max_macro.y += t_delta_macro.y; macro_cell.y += step.y; }
            else { t_current = t_max_macro.z; t_max_macro.z += t_delta_macro.z; macro_cell.z += step.z; }
        }
        current_t = t_exit;
    }

    return result;
}

fn trace_visibility(origin: vec3f, start_cell: vec3i, dir: vec3f, max_dist: f32) -> f32 {
    let inv_dir = 1.0 / dir;
    let step = vec3i(sign(dir));
    let abs_inv = abs(inv_dir);

    var pos = origin;
    var cell = start_cell;
    let t_delta = abs_inv;

    var t_max: vec3f;
    if dir.x > 0.0 { t_max.x = (f32(cell.x + 1) - pos.x) * abs_inv.x; }
    else { t_max.x = (pos.x - f32(cell.x)) * abs_inv.x; }
    if dir.y > 0.0 { t_max.y = (f32(cell.y + 1) - pos.y) * abs_inv.y; }
    else { t_max.y = (pos.y - f32(cell.y)) * abs_inv.y; }
    if dir.z > 0.0 { t_max.z = (f32(cell.z + 1) - pos.z) * abs_inv.z; }
    else { t_max.z = (pos.z - f32(cell.z)) * abs_inv.z; }

    let max_steps = u32(max_dist) + 1u;
    for (var i = 0u; i < min(max_steps, MAX_STEPS); i++) {
        if t_max.x < t_max.y {
            if t_max.x < t_max.z {
                if t_max.x > max_dist { return 1.0; }
                cell.x += step.x; t_max.x += t_delta.x;
            } else {
                if t_max.z > max_dist { return 1.0; }
                cell.z += step.z; t_max.z += t_delta.z;
            }
        } else {
            if t_max.y < t_max.z {
                if t_max.y > max_dist { return 1.0; }
                cell.y += step.y; t_max.y += t_delta.y;
            } else {
                if t_max.z > max_dist { return 1.0; }
                cell.z += step.z; t_max.z += t_delta.z;
            }
        }

        let packed = sbm_read_voxel_lod(0u, cell);
        if (packed & 0x1FFu) != 0u {
            return 0.0;
        }
    }

    return 1.0;
}

fn hash(p: vec3f) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash33(p: vec3f) -> vec3f {
    var p3 = fract(p * vec3f(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
}

fn interleaved_gradient_noise(pixel_pos: vec2f) -> f32 {
    let magic = vec3f(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(pixel_pos, magic.xy)));
}

fn evaluate_point_light(light: PointLight, pos: vec3f, normal: vec3f) -> vec3f {
    let to_light = light.position.xyz - pos;
    let dist = length(to_light);
    let radius = light.position.w;

    if (dist > radius) {
        return vec3f(0.0);
    }

    let light_dir = normalize(to_light);
    let ndotl = max(dot(normal, light_dir), 0.0);

    if (ndotl <= 0.0) {
        return vec3f(0.0);
    }

    let intensity = light.color.w;
    let attenuation = intensity / (dist * dist + 1.0);

    var visibility = 1.0;
    if (light.flags == 1u) {
        let extent_factor = 1.0;
        let shadow_origin = pos + normal * 0.05;
        let start_cell = vec3i(floor(shadow_origin));
        visibility = trace_visibility(shadow_origin, start_cell, light_dir, dist);
    }

    return light.color.rgb * attenuation * ndotl * visibility;
}

fn apply_selection_outline(color: vec3f, hit: HitResult, dir: vec3f) -> vec3f {
    if (globals.selected_block.w < 0.5) {
        return color;
    }

    let sel_pos = vec3i(globals.selected_block.xyz);
    let voxel_pos = hit.voxel_coord;

    if (voxel_pos.x != sel_pos.x || voxel_pos.y != sel_pos.y || voxel_pos.z != sel_pos.z) {
        return color;
    }

    let rel = hit.pos - floor(hit.pos);
    var uv = vec2f(0.0);

    let n = abs(hit.normal);
    if (n.x > 0.5) {
        uv = rel.yz;
    } else if (n.y > 0.5) {
        uv = rel.xz;
    } else {
        uv = rel.xy;
    }

    let d = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    let thickness = 0.02;

    if (d < thickness) {
        return vec3f(0.0);
    }

    return color;
}

fn read_voxel_at(lod: u32, ix: i32, iy: i32, iz: i32) -> vec4f {
    let extent = i32(WORLD_EXTENT >> lod);
    if ix < 0 || iy < 0 || iz < 0 || ix >= extent || iy >= extent || iz >= extent {
        return vec4f(0.0);
    }

    let bs = i32(BRICK_SIZE);
    let bx = ix >> 3;
    let by = iy >> 3;
    let bz = iz >> 3;

    let grid_idx = top_grid_index(lod, bx, by, bz);
    let brick_idx = top_grid[grid_idx];
    if brick_idx == BRICK_EMPTY {
        return vec4f(0.0);
    }

    let lx = floor_mod(ix, bs);
    let ly = floor_mod(iy, bs);
    let lz = floor_mod(iz, bs);

    let linear = brick_idx * BRICK_VOLUME + brick_local_index(lx, ly, lz);

    let rad_packed = radiance_pool[linear];
    let rg = unpack2x16float(rad_packed.x);
    let ba = unpack2x16float(rad_packed.y);

    return vec4f(rg.x, rg.y, ba.x, ba.y);
}

fn sample_voxel_lod_exact(pos: vec3f, lod: u32) -> vec4f {
    let scale = f32(1u << lod);
    let base = vec3i(floor(pos / scale));
    let frac = (pos / scale) - vec3f(base);

    let c000 = read_voxel_at(lod, base.x,     base.y,     base.z);
    let c100 = read_voxel_at(lod, base.x + 1, base.y,     base.z);
    let c010 = read_voxel_at(lod, base.x,     base.y + 1, base.z);
    let c110 = read_voxel_at(lod, base.x + 1, base.y + 1, base.z);
    let c001 = read_voxel_at(lod, base.x,     base.y,     base.z + 1);
    let c101 = read_voxel_at(lod, base.x + 1, base.y,     base.z + 1);
    let c011 = read_voxel_at(lod, base.x,     base.y + 1, base.z + 1);
    let c111 = read_voxel_at(lod, base.x + 1, base.y + 1, base.z + 1);

    let c00 = mix(c000, c100, frac.x);
    let c10 = mix(c010, c110, frac.x);
    let c01 = mix(c001, c101, frac.x);
    let c11 = mix(c011, c111, frac.x);

    let c0 = mix(c00, c10, frac.y);
    let c1 = mix(c01, c11, frac.y);

    return mix(c0, c1, frac.z);
}

fn sample_voxel_continuous(pos: vec3f, diameter: f32) -> vec4f {
    let lod_float = clamp(log2(max(diameter, 1.0)), 0.0, f32(LOD_COUNT - 1u));
    let lod0 = u32(floor(lod_float));
    let lod1 = min(lod0 + 1u, LOD_COUNT - 1u);
    let weight = fract(lod_float);

    let s0 = sample_voxel_lod_exact(pos, lod0);
    if (weight < 0.01) { 
        return s0; 
    }
    
    let s1 = sample_voxel_lod_exact(pos, lod1);
    return mix(s0, s1, weight);
}

fn trace_cone(origin: vec3f, normal: vec3f, dir: vec3f, half_angle: f32, max_dist: f32) -> vec3f {
    let tan_half = tan(half_angle);
    var color = vec3f(0.0);
    var alpha = 0.0;

    let safe_origin = origin + normal * 0.51;
    var t = 1.0;

    for (var i = 0u; i < 64u; i++) {
        if (alpha >= 0.95 || t > max_dist) { break; }

        let diameter = max(1.0, 2.0 * t * tan_half);
        let pos = safe_origin + dir * t;
        
        let s = sample_voxel_continuous(pos, diameter);

        let occlusion_bias = select(1.5, 1.0, diameter <= 1.0);
        let opacity_step = clamp(s.a * occlusion_bias, 0.0, 1.0);
        let a = opacity_step * (1.0 - alpha);
        
        let radiance = select(vec3f(0.0), s.rgb / s.a, s.a > 0.001);
        color += radiance * a;
        alpha += a;
        
        t += max(0.5, diameter * 0.25);
    }
    return color;
}

fn cone_trace_diffuse(origin: vec3f, normal: vec3f) -> vec3f {
    var tang: vec3f;
    if abs(normal.y) > 0.9 {
        tang = normalize(cross(normal, vec3f(1.0, 0.0, 0.0)));
    } else {
        tang = normalize(cross(normal, vec3f(0.0, 1.0, 0.0)));
    }
    let bitang = cross(normal, tang);

    let half_angle = 0.5236; 
    let max_dist = 128.0;

    var result = trace_cone(origin, normal, normal, half_angle, max_dist) * 0.3;

    let tilt = 0.5236; 
    let cos_t = cos(tilt);
    let sin_t = sin(tilt);

    let d0 = normalize(normal * cos_t + tang * sin_t);
    let d1 = normalize(normal * cos_t - tang * sin_t);
    let d2 = normalize(normal * cos_t + bitang * sin_t);
    let d3 = normalize(normal * cos_t - bitang * sin_t);

    result += trace_cone(origin, normal, d0, half_angle, max_dist) * 0.175;
    result += trace_cone(origin, normal, d1, half_angle, max_dist) * 0.175;
    result += trace_cone(origin, normal, d2, half_angle, max_dist) * 0.175;
    result += trace_cone(origin, normal, d3, half_angle, max_dist) * 0.175;

    return result;
}

fn cone_trace_specular(origin: vec3f, reflect_dir: vec3f, normal: vec3f, roughness: f32) -> vec3f {
    let half_angle = max(0.035, roughness * 0.35);
    return trace_cone(origin, normal, reflect_dir, half_angle, 128.0);
}
fn sample_flood_light(pos: vec3f) -> vec3f {
    let p0 = vec3i(floor(pos - 0.5));
    let f = fract(pos - 0.5);

    let c000 = unpack_rgb10(toroidal_light[toroidal_idx(p0 + vec3i(0, 0, 0))]);
    let c100 = unpack_rgb10(toroidal_light[toroidal_idx(p0 + vec3i(1, 0, 0))]);
    let c010 = unpack_rgb10(toroidal_light[toroidal_idx(p0 + vec3i(0, 1, 0))]);
    let c110 = unpack_rgb10(toroidal_light[toroidal_idx(p0 + vec3i(1, 1, 0))]);
    let c001 = unpack_rgb10(toroidal_light[toroidal_idx(p0 + vec3i(0, 0, 1))]);
    let c101 = unpack_rgb10(toroidal_light[toroidal_idx(p0 + vec3i(1, 0, 1))]);
    let c011 = unpack_rgb10(toroidal_light[toroidal_idx(p0 + vec3i(0, 1, 1))]);
    let c111 = unpack_rgb10(toroidal_light[toroidal_idx(p0 + vec3i(1, 1, 1))]);

    let c00 = mix(c000, c100, f.x);
    let c10 = mix(c010, c110, f.x);
    let c01 = mix(c001, c101, f.x);
    let c11 = mix(c011, c111, f.x);

    let c0 = mix(c00, c10, f.y);
    let c1 = mix(c01, c11, f.y);

    return mix(c0, c1, f.z);
}

fn shade_pbr(hit: HitResult, dir: vec3f, screen_pos: vec2f) -> vec3f {
    let pu = hit.voxel.id % 256u;
    let pv = hit.voxel.id / 256u;
    let coords = vec2i(i32(pu), i32(pv));

    let albedo = textureLoad(t_palette, coords, 0, 0).rgb;
    let data = textureLoad(t_palette, coords, 1, 0);
    let roughness = data.r;
    let emission = data.g;
    let noise_strength = data.b;
    let metallic = data.a;

    let emit = albedo * emission * 2.0;

    let noise_val = hash(vec3f(f32(hit.voxel.variant), 0.0, 0.0));
    let noise_mod = 1.0 - (noise_strength * noise_val * 0.5);
    let noisy_albedo = albedo * noise_mod;

    let sample_pos = hit.pos + hit.normal * 0.5;
    let flood_light = sample_flood_light(sample_pos);

    let diffuse_gi = vec3f(0.0); // cone_trace_diffuse(hit.pos, hit.normal);

    let view_dir = normalize(globals.cam_pos.xyz - hit.pos);
    let reflect_dir = reflect(-view_dir, hit.normal);
    let specular_gi = vec3f(0.0); // cone_trace_specular(hit.pos, reflect_dir, hit.normal, roughness);

    let indirect = diffuse_gi + specular_gi * metallic + flood_light;
    
    let ao_p = vec3i(floor(hit.pos + hit.normal * 0.5));
    let ao_val = get_ao(0u, ao_p, hit.normal, fract(hit.pos));
    let ao = pow(ao_val, 1.0);

    let ndotl = max(dot(hit.normal, -globals.sun_dir.xyz), 0.0);
    var sun_light = globals.sun_color.rgb * globals.sun_dir.w * ndotl;

    if ndotl > 0.0 {
        let shadow_origin = hit.pos + hit.normal * 0.05;
        let start_cell = vec3i(floor(shadow_origin));
        let sun_vis = trace_visibility(shadow_origin, start_cell, -globals.sun_dir.xyz, globals.sun_shadow_max);
        sun_light *= sun_vis;
    }

    var dynamic_light = vec3f(0.0);
    let count = min(point_lights.count, 16u);

    for (var i = 0u; i < count; i++) {
        dynamic_light += evaluate_point_light(point_lights.lights[i], hit.pos, hit.normal);
    }

    let total_light = sun_light + indirect * ao + dynamic_light * ao;

    let noise_scale = 150.0;
    let bump_intensity = 0.08;
    let random_vec = hash33(hit.pos * noise_scale) * 2.0 - 1.0;
    let perturbation = random_vec * noise_strength * bump_intensity;
    let n = normalize(hit.normal + perturbation);

    var specular = vec3f(0.0);
    let light_intensity = max(total_light.r, max(total_light.g, total_light.b));

    if (light_intensity > 0.0001) {
        let v = -dir;
        let view_dot_n = max(dot(v, n), 0.0);

        let f0 = mix(vec3f(0.04), noisy_albedo, metallic);
        let f90 = max(vec3f(1.0 - roughness), f0);
        let fresnel = f0 + (f90 - f0) * pow(1.0 - view_dot_n, 5.0);

        specular = fresnel * total_light * (1.0 - roughness);

        if (roughness > 0.9) {
            specular = vec3f(0.0);
        }
    }

    let diffuse = noisy_albedo * (1.0 - metallic);

    var final_hdr = (diffuse * total_light + specular) + emit;
    final_hdr = apply_selection_outline(final_hdr, hit, dir);

    return final_hdr;
}

fn sky(dir: vec3f) -> vec3f {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    let horizon = globals.sky_color.rgb * 0.3;
    let zenith = globals.sky_color.rgb;
    return mix(horizon, zenith, pow(t, 0.5));
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) ndc: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) % 2) * 4.0 - 1.0;
    out.position = vec4f(x, y, 0.0, 1.0);
    out.ndc = vec2f(x, -y);
    return out;
}

fn calculate_velocity(world_pos: vec3f) -> vec2f {
    let prev_clip = globals.prev_view_proj * vec4f(world_pos, 1.0);
    let curr_clip = globals.curr_view_proj * vec4f(world_pos, 1.0);

    let prev_ndc = prev_clip.xy / prev_clip.w;
    let curr_ndc = curr_clip.xy / curr_clip.w;

    let prev_uv = vec2f(prev_ndc.x, -prev_ndc.y) * 0.5 + 0.5;
    let curr_uv = vec2f(curr_ndc.x, -curr_ndc.y) * 0.5 + 0.5;

    return (curr_uv - prev_uv);
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @location(1) velocity: vec2f,
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let clip = vec4f(in.ndc.x, in.ndc.y, 1.0, 1.0);
    let cam_space = globals.proj_inverse * clip;
    let cam_dir = normalize(cam_space.xyz / cam_space.w);

    let world_dir4 = globals.view_inverse * vec4f(cam_dir, 0.0);
    let dir = normalize(world_dir4.xyz);

    let origin = globals.cam_pos.xyz;
    let hit = dda_march(origin, dir);

    var color: vec3f;
    var world_pos: vec3f;
    var velocity = vec2f(0.0);

    if hit.hit {
        color = shade_pbr(hit, dir, in.position.xy);
        world_pos = hit.pos + hit.normal * 0.01;
        velocity = calculate_velocity(world_pos);
    } else {
        color = sky(dir);
        velocity = vec2f(0.0);
    }

    return FragmentOutput(vec4f(color, 1.0), velocity);
}