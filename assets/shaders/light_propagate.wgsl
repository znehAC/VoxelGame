const GRID_SIZE: u32 = 64u;
const GS: i32 = 64;

// Multiplicative attenuation factors
// Use float math directly. no more integer approximations.
const FACTOR_FACE: f32 = 0.90;
const FACTOR_EDGE: f32 = 0.81;    // 0.9 * 0.9
const FACTOR_CORNER: f32 = 0.729; // 0.9 * 0.9 * 0.9

// Sky light injected at top row
const SKY_LIGHT: vec3f = vec3f(1.0, 1.0, 1.0);

@group(0) @binding(0) var<storage, read> voxels: array<u32>;
@group(0) @binding(1) var light_src: texture_3d<f32>;
@group(0) @binding(2) var light_dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var t_palette: texture_2d_array<f32>;

fn voxel_index(pos: vec3<u32>) -> u32 {
    return pos.z * GRID_SIZE * GRID_SIZE + pos.y * GRID_SIZE + pos.x;
}

fn voxel_idx_i(pos: vec3i) -> u32 {
    return u32(pos.z) * GRID_SIZE * GRID_SIZE + u32(pos.y) * GRID_SIZE + u32(pos.x);
}

fn in_bounds(p: vec3i) -> bool {
    return p.x >= 0 && p.x < GS && p.y >= 0 && p.y < GS && p.z >= 0 && p.z < GS;
}

fn is_opaque_at(p: vec3i) -> bool {
    if !in_bounds(p) { return true; }
    return (voxels[voxel_idx_i(p)] & 0xFFFFu) != 0u;
}

fn read_light(p: vec3i) -> vec3f {
    if !in_bounds(p) { return vec3f(0.0); }
    return textureLoad(light_src, p, 0).rgb;
}

fn propagate_light(light: vec3f, factor: f32) -> vec3f {
    return light * factor;
}

@compute @workgroup_size(4, 4, 4)
fn propagate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= GRID_SIZE || gid.y >= GRID_SIZE || gid.z >= GRID_SIZE {
        return;
    }

    let packed = voxels[voxel_index(gid)];
    let material_id = packed & 0xFFFFu;

    // Palette lookup for emission check
    let pu = material_id % 256u;
    let pv = material_id / 256u;
    let coords = vec2i(i32(pu), i32(pv));
    
    // Use Level 0 for palette lookups
    // Use Level 0 for palette lookups
    let albedo = textureLoad(t_palette, coords, 0, 0).rgb;
    let props = textureLoad(t_palette, coords, 1, 0);
    // Unpack props: R=Roughness, G=Emission, B=Noise, A=Metallic
    let emission = props.g;

    // Emission injection
    var emit = vec3f(0.0);
    if emission > 0.01 {
        // Boost emission for visual punch
        emit = albedo * emission * 5.0; 
    }

    // Sky light injected at top boundary
    var sky = vec3f(0.0);
    if gid.y == GRID_SIZE - 1u {
        sky = SKY_LIGHT;
    }

    // Solid Block Logic:
    // If I am solid, I just emit my own light. I do NOT receive light from neighbors.
    // This ensures walls stop light.
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
    best = max(best, propagate_light(read_light(pos + vec3i( 0, 1, 0)), FACTOR_FACE));
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
