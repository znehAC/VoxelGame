const GRID_SIZE: u32 = 64u;

@group(0) @binding(0) var<storage, read> voxels: array<u32>;
@group(0) @binding(1) var light_src: texture_3d<f32>;
@group(0) @binding(2) var light_dst: texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(3) var t_palette: texture_2d_array<f32>;

fn unpack_material_id(packed: u32) -> u32 {
    return packed & 0xFFFFu;
}

fn voxel_index(pos: vec3<u32>) -> u32 {
    return pos.z * GRID_SIZE * GRID_SIZE + pos.y * GRID_SIZE + pos.x;
}

@compute @workgroup_size(4, 4, 4)
fn propagate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= GRID_SIZE || gid.y >= GRID_SIZE || gid.z >= GRID_SIZE {
        return;
    }

    let packed = voxels[voxel_index(gid)];
    let material_id = unpack_material_id(packed);

    let pu = material_id % 256u;
    let pv = material_id / 256u;
    let coords = vec2i(i32(pu), i32(pv));

    let albedo = textureLoad(t_palette, coords, 0, 0).rgb;

    // Undo sRGB→linear hardware conversion to recover original linear data
    let props = textureLoad(t_palette, coords, 1, 0);
    let emission = props.g;

    // Opaque non-emissive: block light
    if material_id != 0u && emission < 0.01 {
        textureStore(light_dst, gid, vec4f(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let pos = vec3i(gid);
    // Diagonal Boost Heuristic
    // Sample neighbors by axis to determine propagation directionality
    var lx = 0.0;
    if (pos.x > 0) { lx = max(lx, textureLoad(light_src, pos - vec3i(1, 0, 0), 0).rgb.r); }
    if (pos.x < i32(GRID_SIZE) - 1) { lx = max(lx, textureLoad(light_src, pos + vec3i(1, 0, 0), 0).rgb.r); }

    var ly = 0.0;
    if (pos.y > 0) { ly = max(ly, textureLoad(light_src, pos - vec3i(0, 1, 0), 0).rgb.r); }
    if (pos.y < i32(GRID_SIZE) - 1) { ly = max(ly, textureLoad(light_src, pos + vec3i(0, 1, 0), 0).rgb.r); }

    var lz = 0.0;
    if (pos.z > 0) { lz = max(lz, textureLoad(light_src, pos - vec3i(0, 0, 1), 0).rgb.r); }
    if (pos.z < i32(GRID_SIZE) - 1) { lz = max(lz, textureLoad(light_src, pos + vec3i(0, 0, 1), 0).rgb.r); }

    // Re-gather full RGB values for actual propagation
    // We need the max color from each axis to propagate
    var color_x = vec3f(0.0);
    if (pos.x > 0) { color_x = max(color_x, textureLoad(light_src, pos - vec3i(1, 0, 0), 0).rgb); }
    if (pos.x < i32(GRID_SIZE) - 1) { color_x = max(color_x, textureLoad(light_src, pos + vec3i(1, 0, 0), 0).rgb); }

    var color_y = vec3f(0.0);
    if (pos.y > 0) { color_y = max(color_y, textureLoad(light_src, pos - vec3i(0, 1, 0), 0).rgb); }
    if (pos.y < i32(GRID_SIZE) - 1) { color_y = max(color_y, textureLoad(light_src, pos + vec3i(0, 1, 0), 0).rgb); }

    var color_z = vec3f(0.0);
    if (pos.z > 0) { color_z = max(color_z, textureLoad(light_src, pos - vec3i(0, 0, 1), 0).rgb); }
    if (pos.z < i32(GRID_SIZE) - 1) { color_z = max(color_z, textureLoad(light_src, pos + vec3i(0, 0, 1), 0).rgb); }

    // Count axes contributing significant light
    let threshold = 0.01;
    var axis_count = 0;
    if (lx > threshold) { axis_count++; }
    if (ly > threshold) { axis_count++; }
    if (lz > threshold) { axis_count++; }

    // Dynamic Decay
    var decay = 0.90; // 1 Axis (Straight)
    if (axis_count == 2) { decay = 0.955; } // 2 Axes (Diagonal)
    else if (axis_count == 3) { decay = 0.98; } // 3 Axes (Corner)

    let max_neighbor_color = max(color_x, max(color_y, color_z));
    let propagated = max_neighbor_color * decay;

    // Emission injection
    let emit_color = albedo * emission;

    // Sunlight injection
    var sunlight = vec3f(0.0);
    if (gid.y == GRID_SIZE - 1) {
        sunlight = vec3f(1.1, 1.1, 1.0); // Bright top-down light
    }

    let result = max(max(propagated, emit_color), sunlight);
    textureStore(light_dst, gid, vec4f(result, 1.0));
}
