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
    selected_block: vec4f, // xyz = pos, w = active (1.0) or inactive (0.0)
    prev_view_proj: mat4x4f,
    curr_view_proj: mat4x4f,
    world_origin: vec4f,
}

struct VoxelData {
    id: u32,
    level: u32,
    temperature: u32,
    variant: u32,
    flags: u32,
}

struct HitResult {
    hit: bool,
    pos: vec3f,
    normal: vec3f,
    voxel: VoxelData,
}

struct PointLight {
    position: vec4f,       // xyz = world pos, w = radius
    color: vec4f,          // rgb = color, w = intensity
    flags: u32,            // 1 = Cast Shadows, 0 = No Shadows
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

const GRID_SIZE: u32 = 512u;
const MAX_STEPS: u32 = 1024u;
const CHUNKSi: i32 = 16;

@group(0) @binding(0) var<uniform> globals: GlobalUniforms;
@group(0) @binding(2) var t_palette: texture_2d_array<f32>;
@group(0) @binding(3) var s_palette: sampler;
@group(0) @binding(4) var<storage, read> voxels: array<u32>;
@group(0) @binding(5) var<uniform> point_lights: LightBuffer;
@group(0) @binding(6) var<storage, read> occupancy: array<u32>;

@group(1) @binding(0) var t_light: texture_3d<f32>;
@group(1) @binding(1) var s_light: sampler;

// --- Toroidal Wrapping ---

fn wrap(c: i32) -> i32 {
    let size = i32(GRID_SIZE);
    return ((c % size) + size) % size;
}

fn chunk_wrap(c: i32) -> i32 {
    let size = CHUNKSi;
    return ((c % size) + size) % size;
}

// --- Helpers ---

fn unpack_voxel(packed: u32) -> VoxelData {
    return VoxelData(
        packed & 0x3FFFu,
        (packed >> 14u) & 0xFu,
        (packed >> 18u) & 0xFu,
        (packed >> 22u) & 0xFu,
        (packed >> 26u) & 0x3Fu,
    );
}

fn voxel_index(x: i32, y: i32, z: i32) -> u32 {
    return u32(wrap(z)) * GRID_SIZE * GRID_SIZE + u32(wrap(y)) * GRID_SIZE + u32(wrap(x));
}

fn chunk_index(cx: i32, cy: i32, cz: i32) -> u32 {
    return u32(chunk_wrap(cz)) * u32(CHUNKSi) * u32(CHUNKSi) + u32(chunk_wrap(cy)) * u32(CHUNKSi) + u32(chunk_wrap(cx));
}

fn is_chunk_occupied(cx: i32, cy: i32, cz: i32) -> bool {
    // Infinite domain, just wrap lookups
    return occupancy[chunk_index(cx, cy, cz)] != 0u;
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

// --- Unified HDDA Traversal Engine ---

fn traverse_grid(origin: vec3f, dir: vec3f, max_dist: f32, shadow_mode: bool) -> HitResult {
    var result: HitResult;
    result.hit = false;

    // 1. Setup
    let inv_dir = 1.0 / dir;
    let t_delta = abs(inv_dir);
    let step = vec3i(sign(dir));
    let bound_offset = vec3f(max(sign(dir), vec3f(0.0)));
    
    // 2. Global Bound Check (Sliding Window)
    
    // Calculate bounds based on world_origin
    let origin_voxel = vec3i(i32(globals.world_origin.x), i32(globals.world_origin.y), i32(globals.world_origin.z));
    let half_grid = i32(GRID_SIZE) / 2;
    
    let window_min = origin_voxel - vec3i(half_grid);
    let window_max = origin_voxel + vec3i(half_grid); // Exclusive max is origin + half? Or window is centered at origin + half?
    // Actually, let's treat world_origin as the CORNER of the atlas in world space for now?
    // The prompt says: "world_origin uniform tells the GPU which region of the world the atlas currently represents."
    // Usually origin = min corner.
    // Let's assume origin is the (0,0,0) index of the atlas in world space.
    // So valid range is [origin, origin + 512).
    
    let grid_min = vec3f(vec3i(i32(globals.world_origin.x), i32(globals.world_origin.y), i32(globals.world_origin.z)));
    let grid_max = grid_min + 512.0;

    let bounds = ray_aabb(origin, inv_dir, grid_min, grid_max);
    
    if bounds.x > bounds.y || bounds.y < 0.0 || bounds.x > max_dist {
        return result;
    }

    // 3. Init State
    var t_curr = max(bounds.x, 0.0);
    // Epsilon offset to enter grid safely
    var curr_pos = origin + dir * (t_curr + 0.001); 
    var cell = vec3i(floor(curr_pos));
    
    // No clamping to 0..512 anymore, we can be anywhere.
    // cell = clamp(cell, vec3i(0), vec3i(i32(GRID_SIZE) - 1));

    var t_max = (vec3f(cell) + bound_offset - origin) * inv_dir;
    
    // Robust NaN handling
    if (abs(dir.x) < 0.00001) { t_max.x = 3.402823e38; }
    if (abs(dir.y) < 0.00001) { t_max.y = 3.402823e38; }
    if (abs(dir.z) < 0.00001) { t_max.z = 3.402823e38; }

    var last_axis = 0u;

    for (var i = 0u; i < MAX_STEPS; i++) {
        
        // --- A. Hierarchical Skip ---
        let chunk_idx = cell >> vec3u(5u);
        
        if (!is_chunk_occupied(chunk_idx.x, chunk_idx.y, chunk_idx.z)) {
            let min_bound = vec3f(chunk_idx << vec3u(5u));
            let max_bound = min_bound + 32.0;

            let t0 = (min_bound - origin) * inv_dir;
            let t1 = (max_bound - origin) * inv_dir;
            let t_far = max(t0, t1); 
            let dist_to_exit = min(min(t_far.x, t_far.y), t_far.z);

            // Advance Strategy: Jump to exit + epsilon
            t_curr = dist_to_exit + 0.005; 
            
            // Check distance limit (crucial for shadows)
            if (t_curr > max_dist) { break; }

            // Rebuild State
            curr_pos = origin + dir * t_curr;
            cell = vec3i(floor(curr_pos));
            // No clamp
            // cell = clamp(cell, vec3i(0), vec3i(i32(GRID_SIZE) - 1));

            t_max = (vec3f(cell) + bound_offset - origin) * inv_dir;
            if (abs(dir.x) < 0.00001) { t_max.x = 3.402823e38; }
            if (abs(dir.y) < 0.00001) { t_max.y = 3.402823e38; }
            if (abs(dir.z) < 0.00001) { t_max.z = 3.402823e38; }

            // Sanity check for precision loops
            if (all((cell >> vec3u(5u)) == chunk_idx)) {
                 if t_max.x < t_max.y {
                    if t_max.x < t_max.z { t_max.x += t_delta.x; cell.x += step.x; last_axis = 0u; }
                    else { t_max.z += t_delta.z; cell.z += step.z; last_axis = 2u; }
                } else {
                    if t_max.y < t_max.z { t_max.y += t_delta.y; cell.y += step.y; last_axis = 1u; }
                    else { t_max.z += t_delta.z; cell.z += step.z; last_axis = 2u; }
                }
            }

            if cell.x < i32(grid_min.x) || cell.x >= i32(grid_max.x) ||
               cell.y < i32(grid_min.y) || cell.y >= i32(grid_max.y) ||
               cell.z < i32(grid_min.z) || cell.z >= i32(grid_max.z) {
                break;
            }
            continue;
        }

        // --- B. Voxel Intersection ---
        let idx = voxel_index(cell.x, cell.y, cell.z);
        if (voxels[idx] & 0x3FFFu) != 0u {
             result.hit = true;
             
             // Shadow Optimization: Early return
             if (shadow_mode) { return result; }

             // Full Detail Calculation
             result.voxel = unpack_voxel(voxels[idx]);

             var t_hit = 0.0;
             if last_axis == 0u { t_hit = t_max.x - t_delta.x; }
             else if last_axis == 1u { t_hit = t_max.y - t_delta.y; }
             else { t_hit = t_max.z - t_delta.z; }
             
             result.pos = origin + dir * t_hit;
             
             var normal = vec3f(0.0);
             if last_axis == 0u { normal.x = -f32(step.x); }
             else if last_axis == 1u { normal.y = -f32(step.y); }
             else { normal.z = -f32(step.z); }
             result.normal = normal;
             
             return result;
        }

        // --- C. Standard Step ---
        if t_max.x < t_max.y {
            if t_max.x < t_max.z {
                t_max.x += t_delta.x;
                cell.x += step.x;
                last_axis = 0u;
            } else {
                t_max.z += t_delta.z;
                cell.z += step.z;
                last_axis = 2u;
            }
        } else {
            if t_max.y < t_max.z {
                t_max.y += t_delta.y;
                cell.y += step.y;
                last_axis = 1u;
            } else {
                t_max.z += t_delta.z;
                cell.z += step.z;
                last_axis = 2u;
            }
        }
        
        // Check Dist
        let t_next = min(min(t_max.x, t_max.y), t_max.z);
        if (t_next > max_dist) { break; }

        if cell.x < i32(grid_min.x) || cell.x >= i32(grid_max.x) ||
           cell.y < i32(grid_min.y) || cell.y >= i32(grid_max.y) ||
           cell.z < i32(grid_min.z) || cell.z >= i32(grid_max.z) {
            break;
        }
    }

    return result;
}

// Wrappers for modularity
fn dda_march(origin: vec3f, dir: vec3f) -> HitResult {
    return traverse_grid(origin, dir, 3.402823e38, false);
}

fn trace_visibility(origin: vec3f, dir: vec3f, max_dist: f32) -> f32 {
    let result = traverse_grid(origin, dir, max_dist, true);
    if (result.hit) { return 0.0; }
    return 1.0;
}

// --- AO & Shading ---

fn get_voxel_at(pos: vec3i) -> bool {
    // Check bounds against loaded region
    let ox = i32(globals.world_origin.x);
    let oy = i32(globals.world_origin.y);
    let oz = i32(globals.world_origin.z);
    let s = i32(GRID_SIZE);

    if (pos.x < ox || pos.x >= ox + s ||
        pos.y < oy || pos.y >= oy + s ||
        pos.z < oz || pos.z >= oz + s) {
        return false; 
    }
    // Idx is wrapped inside voxel_index
    let idx = voxel_index(pos.x, pos.y, pos.z);
    return (voxels[idx] & 0x3FFFu) != 0u;
}

fn vertex_ao(side1: f32, side2: f32, corner: f32) -> f32 {
    if side1 > 0.5 && side2 > 0.5 { return 0.0; }
    return (3.0 - side1 - side2 - corner) / 3.0;
}

fn get_ao(pos: vec3f, normal: vec3f) -> f32 {
    let p = vec3i(floor(pos + normal * 0.5));

    var tang: vec3i;
    var bitang: vec3i;
    if abs(normal.x) > 0.5 {
        tang = vec3i(0, 1, 0); bitang = vec3i(0, 0, 1);
    } else if abs(normal.y) > 0.5 {
        tang = vec3i(1, 0, 0); bitang = vec3i(0, 0, 1);
    } else {
        tang = vec3i(1, 0, 0); bitang = vec3i(0, 1, 0);
    }

    let center = vec3f(p) + 0.5;
    let rel = pos - center;
    let u = dot(rel, vec3f(tang)) + 0.5;
    let v = dot(rel, vec3f(bitang)) + 0.5;

    let s00 = select(0.0, 1.0, get_voxel_at(p - tang - bitang));
    let s10 = select(0.0, 1.0, get_voxel_at(p       - bitang));
    let s20 = select(0.0, 1.0, get_voxel_at(p + tang - bitang));
    let s01 = select(0.0, 1.0, get_voxel_at(p - tang));
    let s21 = select(0.0, 1.0, get_voxel_at(p + tang));
    let s02 = select(0.0, 1.0, get_voxel_at(p - tang + bitang));
    let s12 = select(0.0, 1.0, get_voxel_at(p       + bitang));
    let s22 = select(0.0, 1.0, get_voxel_at(p + tang + bitang));

    let ao0 = vertex_ao(s01, s10, s00);
    let ao1 = vertex_ao(s21, s10, s20);
    let ao2 = vertex_ao(s21, s12, s22);
    let ao3 = vertex_ao(s01, s12, s02);

    return pow(mix(mix(ao0, ao1, u), mix(ao3, ao2, u), v), 1.5);
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

fn evaluate_point_light(light: PointLight, pos: vec3f, normal: vec3f) -> vec3f {
    let to_light = light.position.xyz - pos;
    let dist = length(to_light);
    let radius = light.position.w;

    if (dist > radius) { return vec3f(0.0); }

    let light_dir = normalize(to_light);
    let ndotl = max(dot(normal, light_dir), 0.0);

    if (ndotl <= 0.0) { return vec3f(0.0); }

    let intensity = light.color.w;
    let attenuation = intensity / (dist * dist + 1.0);

    var visibility = 1.0;
    if (light.flags == 1u) {
        let shadow_origin = pos + normal * 0.05;
        visibility = trace_visibility(shadow_origin, light_dir, dist);
    }

    return light.color.rgb * attenuation * ndotl * visibility;
}

fn apply_selection_outline(color: vec3f, hit: HitResult, dir: vec3f) -> vec3f {
    if (globals.selected_block.w < 0.5) { return color; }

    let sel_pos = vec3i(globals.selected_block.xyz);
    let voxel_pos = vec3i(floor(hit.pos - dir * 0.001));

    if (voxel_pos.x != sel_pos.x || voxel_pos.y != sel_pos.y || voxel_pos.z != sel_pos.z) {
        return color;
    }

    let rel = hit.pos - floor(hit.pos);
    var uv = vec2f(0.0);
    let n = abs(hit.normal);
    if (n.x > 0.5) { uv = rel.yz; } 
    else if (n.y > 0.5) { uv = rel.xz; } 
    else { uv = rel.xy; }

    let d = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    if (d < 0.02) { return vec3f(0.0); } // Black outline

    return color;
}

fn shade_pbr(hit: HitResult, dir: vec3f) -> vec3f {
    let pu = hit.voxel.id % 256u;
    let pv = hit.voxel.id / 256u;
    let coords = vec2i(i32(pu), i32(pv));

    let albedo = textureLoad(t_palette, coords, 0, 0).rgb;
    let data = textureLoad(t_palette, coords, 1, 0);
    let roughness = data.r;
    let emission = data.g;
    let noise_strength = data.b;
    let metallic = data.a;

    let noise_val = hash(vec3f(f32(hit.voxel.variant), 0.0, 0.0));
    let noisy_albedo = albedo * (1.0 - (noise_strength * noise_val * 0.5));

    let light_uvw = (hit.pos + hit.normal * 0.1) / vec3f(f32(GRID_SIZE));
    // light_uvw is 0..1 in "atlas space".
    // But hit.pos is world space.
    // We need to map world pos to atlas UVW.
    // (pos - origin) / size? 
    // And handle wrapping?
    // textureSampleLevel on 3D texture wraps by default if address mode is repeat.
    // But we probably want linear mapping relative to origin.
    // The Light Volume is toroidal too.
    // So light_uvw should be (hit.pos) / GRID_SIZE.
    // If sampler is repeat, it wraps automatically.
    // Let's rely on sampler wrapping for now.
    // Wait, light_uvw calcuated here assumes origin is 0.
    // Correct UVW: (hit.pos - globals.world_origin.xyz) / 512.0?
    // No, texture is wrapped.
    // Just use hit.pos / 512.0. The fractional part is the UVW.
    // But we need to ensure the integer boundary aligns.
    // We'll stick to hit.pos / 512.0 and ensure sampler is Repeat.
    // Check sampler in renderer.rs.
    let voxel_light = textureSampleLevel(t_light, s_light, light_uvw, 0.0).rgb;
    let ao = pow(get_ao(hit.pos, hit.normal), 1.0);

    let ndotl = max(dot(hit.normal, -globals.sun_dir.xyz), 0.0);
    var sun_light = globals.sun_color.rgb * globals.sun_dir.w * ndotl;

    if ndotl > 0.0 {
        let shadow_origin = hit.pos + hit.normal * 0.05;
        let sun_vis = trace_visibility(shadow_origin, -globals.sun_dir.xyz, globals.sun_shadow_max);
        sun_light *= sun_vis;
    }

    var dynamic_light = vec3f(0.0);
    let count = min(point_lights.count, 16u);
    for (var i = 0u; i < count; i++) {
        dynamic_light += evaluate_point_light(point_lights.lights[i], hit.pos, hit.normal);
    }

    let total_light = sun_light + voxel_light * ao + dynamic_light;

    // Specular
    var specular = vec3f(0.0);
    let light_intensity = max(total_light.r, max(total_light.g, total_light.b));
    if (light_intensity > 0.0001 && roughness <= 0.9) {
        let noise_scale = 150.0;
        let bump = (hash33(hit.pos * noise_scale) * 2.0 - 1.0) * noise_strength * 0.08;
        let n = normalize(hit.normal + bump);
        
        let view_dot_n = max(dot(-dir, n), 0.0);
        let f0 = mix(vec3f(0.04), noisy_albedo, metallic);
        let f90 = max(vec3f(1.0 - roughness), f0);
        let fresnel = f0 + (f90 - f0) * pow(1.0 - view_dot_n, 5.0);
        
        specular = fresnel * total_light * (1.0 - roughness);
    }

    let emit = albedo * emission * 2.0;
    var final_hdr = (noisy_albedo * (1.0 - metallic) * total_light + specular) + emit;
    return apply_selection_outline(final_hdr, hit, dir);
}

fn sky(dir: vec3f) -> vec3f {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(globals.sky_color.rgb * 0.3, globals.sky_color.rgb, pow(t, 0.5));
}

// --- Main Pipeline ---

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
    return (curr_ndc.xy * vec2f(0.5, -0.5) + 0.5) - (prev_ndc.xy * vec2f(0.5, -0.5) + 0.5);
}

struct FragmentOutput {
    @location(0) color: vec4f,
    @location(1) velocity: vec2f,
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let clip = vec4f(in.ndc.x, in.ndc.y, 1.0, 1.0);
    let cam_space = globals.proj_inverse * clip;
    let world_dir4 = globals.view_inverse * vec4f(normalize(cam_space.xyz / cam_space.w), 0.0);
    let dir = normalize(world_dir4.xyz);
    let origin = globals.cam_pos.xyz;

    let hit = dda_march(origin, dir);

    var color = vec3f(0.0);
    var velocity = vec2f(0.0);

    if hit.hit {
        color = shade_pbr(hit, dir);
        velocity = calculate_velocity(hit.pos + hit.normal * 0.01);
    } else {
        color = sky(dir);
    }

    return FragmentOutput(vec4f(color, 1.0), velocity);
}