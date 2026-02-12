const LIGHT_GRID: u32 = 64u;
const LG: i32 = 64;
const VOXEL_GRID: u32 = 512u;
const VSCALE: u32 = 8u; // VOXEL_GRID / LIGHT_GRID

// Multiplicative attenuation factors
// Use float math directly. no more integer approximations.
// Multiplicative attenuation factors
// Use float math directly. no more integer approximations.
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
}

const FACTOR_FACE: f32 = 0.80; // Stricter decay for indirect light (Cave darkness)
const FACTOR_EDGE: f32 = 0.81;    // 0.9 * 0.9
const FACTOR_CORNER: f32 = 0.729; // 0.9 * 0.9 * 0.9
const BOUNCE_INTENSITY: f32 = 0.2; // How much sun light bounces off walls

@group(0) @binding(0) var<storage, read> voxels: array<u32>;
@group(0) @binding(1) var light_src: texture_3d<f32>;
@group(0) @binding(2) var light_dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var t_palette: texture_2d_array<f32>;
@group(0) @binding(4) var<uniform> globals: GlobalUniforms;

fn voxel_index(pos: vec3<u32>) -> u32 {
    return pos.z * VOXEL_GRID * VOXEL_GRID + pos.y * VOXEL_GRID + pos.x;
}

fn voxel_idx_i(pos: vec3i) -> u32 {
    return u32(pos.z) * VOXEL_GRID * VOXEL_GRID + u32(pos.y) * VOXEL_GRID + u32(pos.x);
}

fn in_bounds(p: vec3i) -> bool {
    return p.x >= 0 && p.x < LG && p.y >= 0 && p.y < LG && p.z >= 0 && p.z < LG;
}

fn light_to_voxel(p: vec3i) -> vec3i {
    return p * i32(VSCALE) + i32(VSCALE / 2u);
}

fn is_opaque_at(p: vec3i) -> bool {
    if !in_bounds(p) { return true; }
    let vp = light_to_voxel(p);
    return (voxels[voxel_idx_i(vp)] & 0x3FFFu) != 0u;
}

fn read_light(p: vec3i) -> vec3f {
    if !in_bounds(p) { return vec3f(0.0); }
    return textureLoad(light_src, p, 0).rgb;
}

fn propagate_light(light: vec3f, factor: f32) -> vec3f {
    return light * factor;
}

fn check_sun_path(start_pos: vec3i, sun_dir: vec3f) -> bool {
    let step = normalize(sun_dir);
    // Start slightly outside the voxel center to avoid self-occlusion
    var p = vec3f(vec3f(start_pos) + vec3f(0.5) + step * 0.7);
    
    // Raymarch towards the sun
    // 48 steps covers the diagonal of a 64^3 grid
    for (var i = 0; i < 48; i++) {
        let ip = vec3i(floor(p));
        
        // 1. Reached Sky (Out of bounds) -> Visible
        if (!in_bounds(ip)) { return true; }
        
        // 2. Hit Solid Block -> Occluded
        if (is_opaque_at(ip)) { return false; }
        
        // 3. Optimization: Hit an already Bright Air Voxel -> Visible
        // If we hit air that is effectively "Sky", we can assume clear path.
        let light = read_light(ip);
        if (light.g > 0.8) { return true; }
        
        p += step;
    }
    return true; // Reached end of loop without hitting solid
}

@compute @workgroup_size(4, 4, 4)
fn propagate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= LIGHT_GRID || gid.y >= LIGHT_GRID || gid.z >= LIGHT_GRID {
        return;
    }

    // Map light-grid cell to voxel-grid center sample
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
    if gid.y == LIGHT_GRID - 1u {
        let night_base = vec3f(0.02, 0.02, 0.05);
        sky = globals.sun_color.rgb * 0.5 + night_base;
    }

    if material_id != 0u {
        textureStore(light_dst, gid, vec4f(emit, 1.0));
        return;
    }

    // Air Block Logic:
    // I am air. I gather light from my neighbors.
    let pos = vec3i(gid);

    // Initial value: Any sky light falling here + any intrinsic emission (air shouldn't emit, but for completeness)
    var best = max(emit, sky);

    // Neighbor Opacity Checks for occlusion logic
    // We need to know if neighbors are opaque to block DIAGONAL propagation (leaks).
    // But for FACE neighbors, we ALWAYS read them. If a face neighbor is a solid light source, we want its light.
    let op_nx = is_opaque_at(pos + vec3i(-1, 0, 0));
    let op_px = is_opaque_at(pos + vec3i( 1, 0, 0));
    let op_ny = is_opaque_at(pos + vec3i( 0,-1, 0));
    let op_py = is_opaque_at(pos + vec3i( 0, 1, 0));
    let op_nz = is_opaque_at(pos + vec3i( 0, 0,-1));
    let op_pz = is_opaque_at(pos + vec3i( 0, 0, 1));

    // --- 6 face neighbors ---
    // Read unconditionally.
    best = max(best, propagate_light(read_light(pos + vec3i(-1, 0, 0)), FACTOR_FACE));
    best = max(best, propagate_light(read_light(pos + vec3i( 1, 0, 0)), FACTOR_FACE));
    best = max(best, propagate_light(read_light(pos + vec3i( 0,-1, 0)), FACTOR_FACE));

    // --- Raytraced Sun Injection ---
    // If I am AIR and I have a SOLID neighbor, check if the Sun hits that face.
    
    // The sun_dir vector points FROM sun TO world (e.g. Down).
    // We need the vector FROM surface TO sun (e.g. Up).
    let to_sun = -globals.sun_dir.xyz;

    // Pre-calculate sun dot products for axes
    // We want to know if the face normal aligns with the direction TO the sun.
    let dot_nx = max(dot(vec3f( 1.0, 0.0, 0.0), to_sun), 0.0);
    let dot_px = max(dot(vec3f(-1.0, 0.0, 0.0), to_sun), 0.0);
    let dot_ny = max(dot(vec3f( 0.0, 1.0, 0.0), to_sun), 0.0); // Floor (Up normal)
    let dot_nz = max(dot(vec3f( 0.0, 0.0, 1.0), to_sun), 0.0);
    let dot_pz = max(dot(vec3f( 0.0, 0.0,-1.0), to_sun), 0.0);
    
    // Optimization: Check sun path ONCE per voxel if ANY face is aligned
    var sun_visible = false;
    let needs_check = (op_nx && dot_nx > 0.0) || (op_px && dot_px > 0.0) || 
                      (op_ny && dot_ny > 0.0) || (op_nz && dot_nz > 0.0) || 
                      (op_pz && dot_pz > 0.0);
                      
    if (needs_check) {
        sun_visible = check_sun_path(pos, to_sun);
    }

    // Injection Intensity: Boost to make the sun spot act as a powerful lamp
    let sun_boost = globals.sun_color.rgb * BOUNCE_INTENSITY;

    if (sun_visible) {
        if (op_nx && dot_nx > 0.0) { best = max(best, sun_boost * dot_nx); }
        if (op_px && dot_px > 0.0) { best = max(best, sun_boost * dot_px); }
        if (op_ny && dot_ny > 0.0) { best = max(best, sun_boost * dot_ny); } // Floor lit by sun
        if (op_nz && dot_nz > 0.0) { best = max(best, sun_boost * dot_nz); }
        if (op_pz && dot_pz > 0.0) { best = max(best, sun_boost * dot_pz); }
    }

    // Vertical neighbor (no special logic anymore)
    let light_up = read_light(pos + vec3i(0, 1, 0));
    best = max(best, propagate_light(light_up, FACTOR_FACE));
    best = max(best, propagate_light(read_light(pos + vec3i( 0, 0,-1)), FACTOR_FACE));
    best = max(best, propagate_light(read_light(pos + vec3i( 0, 0, 1)), FACTOR_FACE));

    // --- 12 edge neighbors ---
    // Propagate ONLY if BOTH adjacent face directions are not opaque (open).
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

    // --- 8 corner neighbors ---
    // Propagate ONLY if ALL THREE adjacent faces are not opaque.
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
