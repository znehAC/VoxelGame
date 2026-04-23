const LIGHT_GRID_SIZE: u32 = 512u;
const LIGHT_GRID_MASK: i32 = 511;
const FALLOFF: f32 = 0.85;

const BRICK_SHIFT: u32 = 3u;
const BRICK_VOLUME: u32 = 512u;
const TOP_GRID_SIZE: u32 = 64u;
const WORLD_EXTENT: u32 = 512u;
const BRICK_EMPTY: u32 = 0xFFFFFFFFu;

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

const FACTOR_FACE: f32 = 0.93;
const FACTOR_EDGE: f32 = 0.90;
const FACTOR_CORNER: f32 = 0.88;
const BOUNCE_INTENSITY: f32 = 0.2;
const MAX_SHADOW_STEPS: u32 = 48u;

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

@group(0) @binding(0) var<storage, read> top_grid: array<u32>;
@group(0) @binding(1) var<storage, read> brick_pool: array<u32>;
@group(0) @binding(2) var<storage, read> light_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> light_out: array<u32>;
@group(0) @binding(4) var<storage, read> dirty_chunks: array<u32>;
@group(0) @binding(5) var<uniform> toroidal_origin: vec4i;
@group(0) @binding(6) var<storage, read> active_chunks: array<u32>;
@group(0) @binding(7) var t_palette: texture_2d_array<f32>;
@group(0) @binding(8) var<uniform> globals: GlobalUniforms;
@group(0) @binding(9) var<uniform> point_lights: LightBuffer;

fn pack_rgb10(c: vec3f) -> u32 {
    let p = vec3u(clamp(c * 0.1, vec3f(0.0), vec3f(1.0)) * 1023.0);
    return (p.x << 20u) | (p.y << 10u) | p.z;
}

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

fn read_light(world_pos: vec3i) -> vec3f {
    return unpack_rgb10(light_in[toroidal_idx(world_pos)]);
}

fn sbm_read_voxel(world_pos: vec3i) -> u32 {
    if (world_pos.x < 0 || world_pos.y < 0 || world_pos.z < 0 ||
        world_pos.x >= i32(WORLD_EXTENT) || world_pos.y >= i32(WORLD_EXTENT) || world_pos.z >= i32(WORLD_EXTENT)) {
        return 0u;
    }
    let bx = u32(world_pos.x) >> BRICK_SHIFT;
    let by = u32(world_pos.y) >> BRICK_SHIFT;
    let bz = u32(world_pos.z) >> BRICK_SHIFT;
    
    let tgs = TOP_GRID_SIZE;
    let grid_idx = (bz % tgs) * tgs * tgs + (by % tgs) * tgs + (bx % tgs);
    let brick_idx = top_grid[grid_idx];
    
    if (brick_idx == BRICK_EMPTY) { return 0u; }
    
    let lx = u32(world_pos.x) & 7u;
    let ly = u32(world_pos.y) & 7u;
    let lz = u32(world_pos.z) & 7u;
    let linear = lz * 64u + ly * 8u + lx;
    let word = brick_pool[brick_idx * 256u + (linear >> 1u)];
    return (word >> ((linear & 1u) * 16u)) & 0xFFFFu;
}

fn is_opaque(pos: vec3i) -> bool {
    return (sbm_read_voxel(pos) & 0x1FFu) != 0u;
}

fn trace_visibility(origin: vec3i, dir: vec3f, max_dist: f32) -> bool {
    let inv_dir = 1.0 / dir;
    let step = vec3i(sign(dir));
    let t_delta = abs(inv_dir);
    
    var cell = origin;
    var t_max: vec3f;
    let o_f = vec3f(origin) + 0.5;
    
    if (dir.x > 0.0) { t_max.x = (f32(cell.x + 1) - o_f.x) * inv_dir.x; } else { t_max.x = (o_f.x - f32(cell.x)) * -inv_dir.x; }
    if (dir.y > 0.0) { t_max.y = (f32(cell.y + 1) - o_f.y) * inv_dir.y; } else { t_max.y = (o_f.y - f32(cell.y)) * -inv_dir.y; }
    if (dir.z > 0.0) { t_max.z = (f32(cell.z + 1) - o_f.z) * inv_dir.z; } else { t_max.z = (o_f.z - f32(cell.z)) * -inv_dir.z; }

    let max_steps = min(MAX_SHADOW_STEPS, u32(max_dist) + 1u);

    for (var i = 0u; i < max_steps; i++) {
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

fn check_sun_path_dda(origin: vec3i, dir: vec3f) -> bool {
    return trace_visibility(origin, dir, 512.0);
}

fn trace_upwards(origin: vec3i) -> bool {
    var y = origin.y + 1;
    let bx = u32(origin.x) >> BRICK_SHIFT;
    let bz = u32(origin.z) >> BRICK_SHIFT;
    let tgs = TOP_GRID_SIZE;
    
    if (origin.x < 0 || origin.x >= i32(WORLD_EXTENT) || origin.z < 0 || origin.z >= i32(WORLD_EXTENT)) {
        return true; 
    }

    while (y < i32(WORLD_EXTENT)) {
        let by = u32(y) >> BRICK_SHIFT;
        let grid_idx = (bz % tgs) * tgs * tgs + (by % tgs) * tgs + (bx % tgs);
        let brick_idx = top_grid[grid_idx];
        
        if (brick_idx == BRICK_EMPTY) {
            y = i32((by + 1u) << BRICK_SHIFT);
            continue;
        }

        let max_y_for_brick = i32((by + 1u) << BRICK_SHIFT);
        let max_y = min(max_y_for_brick, i32(WORLD_EXTENT));
        
        let lx = u32(origin.x) & 7u;
        let lz = u32(origin.z) & 7u;
        
        while (y < max_y) {
            let ly = u32(y) & 7u;
            let linear = lz * 64u + ly * 8u + lx;
            let word = brick_pool[brick_idx * 256u + (linear >> 1u)];
            let voxel = (word >> ((linear & 1u) * 16u)) & 0xFFFFu;

            if ((voxel & 0x1FFu) != 0u) {
                return false;
            }
            y += 1;
        }
    }
    
    return true;
}

fn propagate_light(light: vec3f, factor: f32) -> vec3f {
    return light * factor;
}

@compute @workgroup_size(8, 8, 4)
fn propagate(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let chunk_idx = active_chunks[wg.x];
    
    // Decode chunk_idx -> cx, cy, cz
    let cx = chunk_idx & 63u;
    let cy = (chunk_idx >> 6u) & 63u;
    let cz = chunk_idx >> 12u;

    let world_base = vec3i(i32(cx), i32(cy), i32(cz)) * 8 + toroidal_origin.xyz;

    for (var z_half = 0u; z_half < 2u; z_half++) {
        let lz = lid.z + z_half * 4u;
        let world_pos = world_base + vec3i(i32(lid.x), i32(lid.y), i32(lz));
        let idx = toroidal_idx(world_pos);

        let packed_voxel = sbm_read_voxel(world_pos);
        let material_id = packed_voxel & 0x1FFu;

        if (material_id != 0u) {
            let pu = material_id % 256u;
            let pv = material_id / 256u;
            let coords = vec2i(i32(pu), i32(pv));
            
            let albedo = textureLoad(t_palette, coords, 0, 0).rgb;
            let props = textureLoad(t_palette, coords, 1, 0);
            let emission = props.g;

            var emit = vec3f(0.0);
            if (emission > 0.01) {
                emit = albedo * emission * 5.0; 
            }

            light_out[idx] = pack_rgb10(emit);
            continue;
        }

        var best = vec3f(0.0);

        var sky = vec3f(0.0);
        if (world_pos.y >= i32(WORLD_EXTENT - 1u)) {
            let night_base = vec3f(0.02, 0.02, 0.05);
            sky = globals.sun_color.rgb * 0.5 + night_base;
        }
        best = max(best, sky);

        let op_nx = is_opaque(world_pos + vec3i(-1, 0, 0));
        let op_px = is_opaque(world_pos + vec3i( 1, 0, 0));
        let op_ny = is_opaque(world_pos + vec3i( 0,-1, 0));
        let op_py = is_opaque(world_pos + vec3i( 0, 1, 0));
        let op_nz = is_opaque(world_pos + vec3i( 0, 0,-1));
        let op_pz = is_opaque(world_pos + vec3i( 0, 0, 1));
        
        var sky_visible = false;
        
        if (world_pos.y >= i32(WORLD_EXTENT - 1u)) {
            sky_visible = true;
        } else {
            if (!op_py) {
                 sky_visible = trace_upwards(world_pos);
            }
        }

        let sky_boost = globals.sky_color.rgb * globals.sky_color.w * 1.5;

        if (sky_visible) { 
             best = max(best, sky_boost); 
        }

        // --- 6 face neighbors ---
        best = max(best, propagate_light(read_light(world_pos + vec3i(-1, 0, 0)), FACTOR_FACE));
        best = max(best, propagate_light(read_light(world_pos + vec3i( 1, 0, 0)), FACTOR_FACE));
        best = max(best, propagate_light(read_light(world_pos + vec3i( 0,-1, 0)), FACTOR_FACE));
        best = max(best, propagate_light(read_light(world_pos + vec3i( 0, 1, 0)), FACTOR_FACE));
        best = max(best, propagate_light(read_light(world_pos + vec3i( 0, 0,-1)), FACTOR_FACE));
        best = max(best, propagate_light(read_light(world_pos + vec3i( 0, 0, 1)), FACTOR_FACE));

        // --- 12 edge neighbors ---
        if (!op_nx && !op_ny) { best = max(best, propagate_light(read_light(world_pos + vec3i(-1,-1, 0)), FACTOR_EDGE)); }
        if (!op_px && !op_ny) { best = max(best, propagate_light(read_light(world_pos + vec3i( 1,-1, 0)), FACTOR_EDGE)); }
        if (!op_nx && !op_py) { best = max(best, propagate_light(read_light(world_pos + vec3i(-1, 1, 0)), FACTOR_EDGE)); }
        if (!op_px && !op_py) { best = max(best, propagate_light(read_light(world_pos + vec3i( 1, 1, 0)), FACTOR_EDGE)); }
        
        if (!op_nx && !op_nz) { best = max(best, propagate_light(read_light(world_pos + vec3i(-1, 0,-1)), FACTOR_EDGE)); }
        if (!op_px && !op_nz) { best = max(best, propagate_light(read_light(world_pos + vec3i( 1, 0,-1)), FACTOR_EDGE)); }
        if (!op_nx && !op_pz) { best = max(best, propagate_light(read_light(world_pos + vec3i(-1, 0, 1)), FACTOR_EDGE)); }
        if (!op_px && !op_pz) { best = max(best, propagate_light(read_light(world_pos + vec3i( 1, 0, 1)), FACTOR_EDGE)); }
        
        if (!op_ny && !op_nz) { best = max(best, propagate_light(read_light(world_pos + vec3i( 0,-1,-1)), FACTOR_EDGE)); }
        if (!op_py && !op_nz) { best = max(best, propagate_light(read_light(world_pos + vec3i( 0, 1,-1)), FACTOR_EDGE)); }
        if (!op_ny && !op_pz) { best = max(best, propagate_light(read_light(world_pos + vec3i( 0,-1, 1)), FACTOR_EDGE)); }
        if (!op_py && !op_pz) { best = max(best, propagate_light(read_light(world_pos + vec3i( 0, 1, 1)), FACTOR_EDGE)); }

        // --- 8 corner neighbors ---
        if (!op_nx && !op_ny && !op_nz) { best = max(best, propagate_light(read_light(world_pos + vec3i(-1,-1,-1)), FACTOR_CORNER)); }
        if (!op_px && !op_ny && !op_nz) { best = max(best, propagate_light(read_light(world_pos + vec3i( 1,-1,-1)), FACTOR_CORNER)); }
        if (!op_nx && !op_py && !op_nz) { best = max(best, propagate_light(read_light(world_pos + vec3i(-1, 1,-1)), FACTOR_CORNER)); }
        if (!op_px && !op_py && !op_nz) { best = max(best, propagate_light(read_light(world_pos + vec3i( 1, 1,-1)), FACTOR_CORNER)); }
        
        if (!op_nx && !op_ny && !op_pz) { best = max(best, propagate_light(read_light(world_pos + vec3i(-1,-1, 1)), FACTOR_CORNER)); }
        if (!op_px && !op_ny && !op_pz) { best = max(best, propagate_light(read_light(world_pos + vec3i( 1,-1, 1)), FACTOR_CORNER)); }
        if (!op_nx && !op_py && !op_pz) { best = max(best, propagate_light(read_light(world_pos + vec3i(-1, 1, 1)), FACTOR_CORNER)); }
        if (!op_px && !op_py && !op_pz) { best = max(best, propagate_light(read_light(world_pos + vec3i( 1, 1, 1)), FACTOR_CORNER)); }

        light_out[idx] = pack_rgb10(best);
    }
}