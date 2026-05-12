// Deferred resolve pass — reads visibility buffer + depth buffer, outputs HDR color
//
// Dispatch: ceil(width/8) × ceil(height/8) × 1 workgroups

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

fn hash_u(n: u32) -> f32 {
    var h = n;
    h = (h ^ (h >> 17u)) * 0xBF324571u;
    h = (h ^ (h >> 13u)) * 0x4615243Fu;
    h ^= h >> 16u;
    return f32(h) / 4294967295.0;
}

const NORMALS: array<vec3f, 6> = array<vec3f, 6>(
    vec3f( 1.0,  0.0,  0.0), // +X
    vec3f(-1.0,  0.0,  0.0), // -X
    vec3f( 0.0,  1.0,  0.0), // +Y
    vec3f( 0.0, -1.0,  0.0), // -Y
    vec3f( 0.0,  0.0,  1.0), // +Z
    vec3f( 0.0,  0.0, -1.0), // -Z
);

@group(0) @binding(0) var<uniform> globals: GlobalUniforms;
@group(0) @binding(1) var<storage, read> visibility_buf: array<VisibilityPayload>;
@group(0) @binding(2) var t_palette: texture_2d_array<f32>;
@group(0) @binding(3) var s_palette: sampler;
@group(0) @binding(4) var output_texture: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<storage, read> depth_buf: array<f32>;

fn sky_color(ray_dir: vec3f) -> vec3f {
    let t = clamp(ray_dir.y * 0.5 + 0.5, 0.0, 1.0);
    let ground = globals.ground_color.rgb;
    let sky = globals.sky_color.rgb * globals.sky_color.w;
    let sun_color = globals.sun_color.rgb * globals.sun_dir.w;
    let sun_dir = normalize(-globals.sun_dir.xyz);
    let sun_dot = max(dot(ray_dir, sun_dir), 0.0);
    let sun_disc = pow(sun_dot, 512.0) * sun_color * 5.0;
    let sun_glow = pow(sun_dot, 8.0) * sun_color * 0.3;
    return mix(ground, sky, t) + sun_disc + sun_glow;
}

@compute @workgroup_size(8, 8, 1)
fn resolve(@builtin(global_invocation_id) gid: vec3u) {
    let px = gid.x;
    let py = gid.y;
    let width = u32(globals.resolution_x);
    let height = u32(globals.resolution_y);
    if (px >= width || py >= height) { return; }

    let pixel_idx = py * width + px;
    let vis = visibility_buf[pixel_idx];
    let mat_raw = vis.lo & 0x7FFu;
    let material_id = mat_raw & 0x1FFu;
    let variant = (mat_raw >> 9u) & 0x3u;

    // Reconstruct ray direction for sky
    let ndc = vec2f(
        (f32(px) + 0.5) / f32(width) * 2.0 - 1.0,
        1.0 - (f32(py) + 0.5) / f32(height) * 2.0,
    );
    let clip_far = vec4f(ndc, 1.0, 1.0);
    var world_far = globals.view_inverse * (globals.proj_inverse * clip_far);
    world_far /= world_far.w;
    let ray_dir = normalize(world_far.xyz - globals.cam_pos.xyz);

    if (material_id == 0u) {
        let sky = sky_color(ray_dir);
        textureStore(output_texture, vec2i(i32(px), i32(py)), vec4f(sky, 1.0));
        return;
    }

    // Unpack visibility payload
    let normal_idx = min((vis.lo >> 16u) & 0x7u, 5u);
    let normal = NORMALS[normal_idx];
    let vp = vec3u(vis.hi & 0x3FFu, (vis.hi >> 10u) & 0x3FFu, (vis.hi >> 20u) & 0x3FFu);

    // Material palette lookup
    let pu = material_id % 256u;
    let pv = material_id / 256u;
    let coords = vec2i(i32(pu), i32(pv));
    let albedo_base = textureLoad(t_palette, coords, 0, 0).rgb;
    let props = textureLoad(t_palette, coords, 1, 0);
    let roughness = props.r;
    let emission = props.g;
    let noise_strength = props.b;

    // Per-voxel brightness variation — single hash avoids hue shifts on neutral colors
    let h_seed = vp.x * 1234567u + vp.y * 7654321u + vp.z * 9999991u + variant * 127u;
    let brightness_vary = hash_u(h_seed) * 2.0 - 1.0;
    let albedo = clamp(albedo_base * (1.0 + brightness_vary * sqrt(noise_strength) * 0.35), vec3f(0.0), vec3f(1.0));

    // Lighting
    let sun_dir = normalize(-globals.sun_dir.xyz);
    let ndotl = max(dot(normal, sun_dir), 0.0);
    let sun_intensity = globals.sun_dir.w;
    let sun_color = globals.sun_color.rgb;

    // Hemisphere ambient with face AO: top=1.0, sides=0.75, bottom=0.5
    let face_ao = select(select(0.5, 0.75, normal_idx != 3u), 1.0, normal_idx == 2u);
    let sky_contrib = globals.sky_color.rgb * globals.sky_color.w * 0.3;
    let ground_contrib = globals.ground_color.rgb * 0.1;
    let ambient = mix(ground_contrib, sky_contrib, normal.y * 0.5 + 0.5) * face_ao;

    var color = albedo * (sun_color * sun_intensity * ndotl + ambient);

    // Emission
    if (emission > 0.01) {
        color += albedo * emission * 5.0;
    }

    // Specular (Blinn-Phong)
    let hit_depth = depth_buf[pixel_idx];
    let hit_pos = globals.cam_pos.xyz + ray_dir * hit_depth;
    let view_dir = normalize(globals.cam_pos.xyz - hit_pos);
    let half_vec = normalize(sun_dir + view_dir);
    let spec_power = mix(16.0, 256.0, 1.0 - roughness);
    let spec = pow(max(dot(normal, half_vec), 0.0), spec_power) * (1.0 - roughness) * 0.3;
    color += sun_color * sun_intensity * spec;

    textureStore(output_texture, vec2i(i32(px), i32(py)), vec4f(color, 1.0));
}
