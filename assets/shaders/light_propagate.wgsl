const LIGHT_GRID: u32 = 128u;
const LG: i32 = 128;
const VOXEL_GRID: u32 = 512u;
const VSCALE: u32 = 4u; // VOXEL_GRID / LIGHT_GRID

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
    brush_pos_radius: vec4f,
}

const FACTOR_FACE: f32 = 0.80;
const FACTOR_EDGE: f32 = 0.81;
const FACTOR_CORNER: f32 = 0.729;
const BOUNCE_INTENSITY: f32 = 0.7;

const CHUNKSi: i32 = 16;
const CHUNK_SIZEf: f32 = 32.0;

@group(0) @binding(0) var<storage, read> voxels: array<u32>;
@group(0) @binding(1) var light_src: texture_3d<f32>;
@group(0) @binding(2) var light_dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var t_palette: texture_2d_array<f32>;
@group(0) @binding(4) var<uniform> globals: GlobalUniforms;
@group(0) @binding(5) var<storage, read> occupancy: array<u32>;

// --- Helpers ---

fn chunk_wrap(c: i32) -> i32 {
    let size = CHUNKSi;
    return ((c % size) + size) % size;
}

fn chunk_index(cx: i32, cy: i32, cz: i32) -> u32 {
    return u32(chunk_wrap(cz)) * u32(CHUNKSi) * u32(CHUNKSi) + u32(chunk_wrap(cy)) * u32(CHUNKSi) + u32(chunk_wrap(cx));
}

fn is_chunk_occupied(cx: i32, cy: i32, cz: i32) -> bool {
    return occupancy[chunk_index(cx, cy, cz)] != 0u;
}

fn voxel_index(pos: vec3<u32>) -> u32 {
    return pos.z * VOXEL_GRID * VOXEL_GRID + pos.y * VOXEL_GRID + pos.x;
}

fn voxel_idx_i(pos: vec3i) -> u32 {
    return u32(pos.z) * VOXEL_GRID * VOXEL_GRID + u32(pos.y) * VOXEL_GRID + u32(pos.x);
}

fn in_bounds(p: vec3i) -> bool {
    return p.x >= 0 && p.x < LG && p.y >= 0 && p.y < LG && p.z >= 0 && p.z < LG;
}

fn wrap_lg(c: i32) -> i32 {
    let size = i32(LIGHT_GRID);
    return ((c % size) + size) % size;
}

fn wrap_voxel_coord(c: i32) -> i32 {
    let size = i32(VOXEL_GRID);
    return ((c % size) + size) % size;
}

fn light_to_voxel_global(p: vec3i) -> vec3i {
    return vec3i(
        wrap_voxel_coord(p.x * i32(VSCALE) + i32(VSCALE) / 2),
        wrap_voxel_coord(p.y * i32(VSCALE) + i32(VSCALE) / 2),
        wrap_voxel_coord(p.z * i32(VSCALE) + i32(VSCALE) / 2)
    );
}

fn light_to_voxel(p: vec3i) -> vec3i {
    return light_to_voxel_global(p);
}

fn is_opaque_at(p: vec3i) -> bool {
    if !in_bounds(p) { return true; }
    let vp = light_to_voxel(p);
    return (voxels[voxel_idx_i(vp)] & 0x3FFFu) != 0u;
}

fn read_light(p: vec3i) -> vec3f {
    let wrapped_p = vec3i(wrap_lg(p.x), wrap_lg(p.y), wrap_lg(p.z));
    return textureLoad(light_src, wrapped_p, 0).rgb;
}

fn propagate_light(light: vec3f, factor: f32) -> vec3f {
    return light * factor;
}

// Unified HDDA check_sun_path
fn check_sun_path(start_pos: vec3i, sun_dir: vec3f) -> bool {
    let origin_voxel = light_to_voxel(start_pos);
    let origin = vec3f(vec3f(origin_voxel) + 0.5);
    let dir = normalize(sun_dir); 
    
    // Bias: Epsilon 1e-3
    let biased_origin = origin + dir * 0.001;
    
    let max_dist = 512.0; 
    
    let inv_dir = 1.0 / dir;
    let t_delta = abs(inv_dir);
    let step = vec3i(sign(dir));
    let bound_offset = vec3f(max(sign(dir), vec3f(0.0)));
    
    var t_curr = 0.0;
    var curr_pos = biased_origin;
    var cell = vec3i(floor(curr_pos));
    
    var t_max = (vec3f(cell) + bound_offset - biased_origin) * inv_dir;
    
    if (abs(dir.x) < 0.00001) { t_max.x = 3.402823e38; }
    if (abs(dir.y) < 0.00001) { t_max.y = 3.402823e38; }
    if (abs(dir.z) < 0.00001) { t_max.z = 3.402823e38; }

    let origin_y = i32(globals.world_origin.y);

    for (var i = 0u; i < 256u; i++) {
        // --- A. Hierarchical Skip ---
        let chunk_idx = cell >> vec3u(5u);
        
        if (!is_chunk_occupied(chunk_idx.x, chunk_idx.y, chunk_idx.z)) {
             let min_bound = vec3f(chunk_idx << vec3u(5u));
             let max_bound = min_bound + 32.0;

             let t0 = (min_bound - biased_origin) * inv_dir;
             let t1 = (max_bound - biased_origin) * inv_dir;
             let t_far = max(t0, t1); 
             let dist_to_exit = min(min(t_far.x, t_far.y), t_far.z);

             t_curr = dist_to_exit + 0.005; 
             if (t_curr > max_dist) { return true; } 

             curr_pos = biased_origin + dir * t_curr;
             cell = vec3i(floor(curr_pos));
             t_max = (vec3f(cell) + bound_offset - biased_origin) * inv_dir;
             if (abs(dir.x) < 0.00001) { t_max.x = 3.402823e38; }
             if (abs(dir.y) < 0.00001) { t_max.y = 3.402823e38; }
             if (abs(dir.z) < 0.00001) { t_max.z = 3.402823e38; }

             // Toroidal Safety: Terminate if we exited Active World Y Bounds
             if (cell.y < origin_y || cell.y >= origin_y + 512) {
                 return true; 
             }
             continue;
        }

        // --- B. Voxel Intersection ---
        // Manually wrap for lookup
        let idx = voxel_index(vec3u(u32(wrap_voxel_coord(cell.x)), u32(wrap_voxel_coord(cell.y)), u32(wrap_voxel_coord(cell.z))));
        if (voxels[idx] & 0x3FFFu) != 0u {
            return false; // Occluded
        }
        
        // --- C. Standard Step ---
        if t_max.x < t_max.y {
            if t_max.x < t_max.z { t_max.x += t_delta.x; cell.x += step.x; }
            else { t_max.z += t_delta.z; cell.z += step.z; }
        } else {
            if t_max.y < t_max.z { t_max.y += t_delta.y; cell.y += step.y; }
            else { t_max.z += t_delta.z; cell.z += step.z; }
        }
        
        let t_next = min(min(t_max.x, t_max.y), t_max.z);
        if (t_next > max_dist) { return true; }

        if (cell.y < origin_y || cell.y >= origin_y + 512) {
             return true; 
        }
    }
    
    return true;
}

fn is_sky_layer(y: u32) -> bool {
    let world_origin_y = i32(globals.world_origin.y);
    let top_y_world = world_origin_y + i32(VOXEL_GRID) - 1;
    let top_light_y = wrap_lg(top_y_world / i32(VSCALE));
    let second_light_y = wrap_lg(top_y_world / i32(VSCALE) - 1);
    return i32(y) == top_light_y || i32(y) == second_light_y;
}

@compute @workgroup_size(4, 4, 4)
fn propagate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= LIGHT_GRID || gid.y >= LIGHT_GRID || gid.z >= LIGHT_GRID {
        return;
    }

    let voxel_pos = vec3<u32>(light_to_voxel(vec3i(gid)));
    let packed = voxels[voxel_index(voxel_pos)];
    let material_id = packed & 0x3FFFu;

    let pu = material_id % 256u;
    let pv = material_id / 256u;
    let coords = vec2i(i32(pu), i32(pv));

    let albedo = textureLoad(t_palette, coords, 0, 0).rgb;
    let props = textureLoad(t_palette, coords, 1, 0);
    let emission = props.g;

    var emit = vec3f(0.0);
    if emission > 0.01 {
        emit = albedo * emission * 5.0;
    }

    var sky = vec3f(0.0);
    if is_sky_layer(gid.y) {
        let night_base = vec3f(0.02, 0.02, 0.05);
        sky = globals.sky_color.rgb * globals.sky_color.a + night_base;
    }

    if material_id != 0u {
        textureStore(light_dst, gid, vec4f(emit, 1.0));
        return;
    }

    let pos = vec3i(gid);
    var best = max(emit, sky);

    let op_nx = is_opaque_at(pos + vec3i(-1, 0, 0));
    let op_px = is_opaque_at(pos + vec3i( 1, 0, 0));
    let op_ny = is_opaque_at(pos + vec3i( 0,-1, 0));
    let op_py = is_opaque_at(pos + vec3i( 0, 1, 0));
    let op_nz = is_opaque_at(pos + vec3i( 0, 0,-1));
    let op_pz = is_opaque_at(pos + vec3i( 0, 0, 1));

    best = max(best, propagate_light(read_light(pos + vec3i(-1, 0, 0)), FACTOR_FACE));
    best = max(best, propagate_light(read_light(pos + vec3i( 1, 0, 0)), FACTOR_FACE));
    best = max(best, propagate_light(read_light(pos + vec3i( 0,-1, 0)), FACTOR_FACE));

    let to_sun = -globals.sun_dir.xyz;

    let dot_nx = max(dot(vec3f( 1.0, 0.0, 0.0), to_sun), 0.0);
    let dot_px = max(dot(vec3f(-1.0, 0.0, 0.0), to_sun), 0.0);
    let dot_ny = max(dot(vec3f( 0.0, 1.0, 0.0), to_sun), 0.0);
    let dot_nz = max(dot(vec3f( 0.0, 0.0, 1.0), to_sun), 0.0);
    let dot_pz = max(dot(vec3f( 0.0, 0.0,-1.0), to_sun), 0.0);
    
    var sun_visible = false;
    let needs_check = (op_nx && dot_nx > 0.0) || (op_px && dot_px > 0.0) || 
                      (op_ny && dot_ny > 0.0) || (op_nz && dot_nz > 0.0) || 
                      (op_pz && dot_pz > 0.0);
                      
    if (needs_check) {
        sun_visible = check_sun_path(pos, to_sun);
    }

    let sun_boost = globals.sun_color.rgb * BOUNCE_INTENSITY;

    if (sun_visible) {
        if (op_nx && dot_nx > 0.0) { best = max(best, sun_boost * dot_nx); }
        if (op_px && dot_px > 0.0) { best = max(best, sun_boost * dot_px); }
        if (op_ny && dot_ny > 0.0) { best = max(best, sun_boost * dot_ny); }
        if (op_nz && dot_nz > 0.0) { best = max(best, sun_boost * dot_nz); }
        if (op_pz && dot_pz > 0.0) { best = max(best, sun_boost * dot_pz); }
    }

    let light_up = read_light(pos + vec3i(0, 1, 0));
    best = max(best, propagate_light(light_up, FACTOR_FACE));
    best = max(best, propagate_light(read_light(pos + vec3i( 0, 0,-1)), FACTOR_FACE));
    best = max(best, propagate_light(read_light(pos + vec3i( 0, 0, 1)), FACTOR_FACE));

    if !op_nx && !op_ny { best = max(best, propagate_light(read_light(pos + vec3i(-1,-1, 0)), FACTOR_EDGE)); }
    if !op_px && !op_ny { best = max(best, propagate_light(read_light(pos + vec3i( 1,-1, 0)), FACTOR_EDGE)); }
    if !op_nx && !op_py { best = max(best, propagate_light(read_light(pos + vec3i(-1, 1, 0)), FACTOR_EDGE)); }
    if !op_px && !op_py { best = max(best, propagate_light(read_light(pos + vec3i( 1, 1, 0)), FACTOR_EDGE)); }
    
    if !op_nx && !op_nz { best = max(best, propagate_light(read_light(pos + vec3i(-1, 0,-1)), FACTOR_EDGE)); }
    if !op_px && !op_nz { best = max(best, propagate_light(read_light(pos + vec3i( 1, 0,-1)), FACTOR_EDGE)); }
    if !op_nx && !op_pz { best = max(best, propagate_light(read_light(pos + vec3i(-1, 0, 1)), FACTOR_EDGE)); }
    if !op_px && !op_pz { best = max(best, propagate_light(read_light(pos + vec3i( 1, 0, 1)), FACTOR_EDGE)); }
    
    if !op_ny && !op_nz { best = max(best, propagate_light(read_light(pos + vec3i( 0,-1,-1)), FACTOR_EDGE)); }
    if !op_py && !op_nz { best = max(best, propagate_light(read_light(pos + vec3i( 0, 1,-1)), FACTOR_EDGE)); }
    if !op_ny && !op_pz { best = max(best, propagate_light(read_light(pos + vec3i( 0,-1, 1)), FACTOR_EDGE)); }
    if !op_py && !op_pz { best = max(best, propagate_light(read_light(pos + vec3i( 0, 1, 1)), FACTOR_EDGE)); }

    if !op_nx && !op_ny && !op_nz { best = max(best, propagate_light(read_light(pos + vec3i(-1,-1,-1)), FACTOR_CORNER)); }
    if !op_px && !op_ny && !op_nz { best = max(best, propagate_light(read_light(pos + vec3i( 1,-1,-1)), FACTOR_CORNER)); }
    if !op_nx && !op_py && !op_nz { best = max(best, propagate_light(read_light(pos + vec3i(-1, 1,-1)), FACTOR_CORNER)); }
    if !op_px && !op_py && !op_nz { best = max(best, propagate_light(read_light(pos + vec3i( 1, 1,-1)), FACTOR_CORNER)); }
    
    if !op_nx && !op_ny && !op_pz { best = max(best, propagate_light(read_light(pos + vec3i(-1,-1, 1)), FACTOR_CORNER)); }
    if !op_px && !op_ny && !op_pz { best = max(best, propagate_light(read_light(pos + vec3i( 1,-1, 1)), FACTOR_CORNER)); }
    if !op_nx && !op_py && !op_pz { best = max(best, propagate_light(read_light(pos + vec3i(-1, 1, 1)), FACTOR_CORNER)); }
    if !op_px && !op_py && !op_pz { best = max(best, propagate_light(read_light(pos + vec3i( 1, 1, 1)), FACTOR_CORNER)); }

    textureStore(light_dst, gid, vec4f(best, 1.0));
}
