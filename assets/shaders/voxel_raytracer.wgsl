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
}

struct VoxelData {
    id: u32,
    state: u32,
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

const GRID_SIZE: u32 = 64u;
const MAX_STEPS: u32 = 256u;

@group(0) @binding(0) var<uniform> globals: GlobalUniforms;
@group(0) @binding(2) var t_palette: texture_2d_array<f32>;
@group(0) @binding(3) var s_palette: sampler;
@group(0) @binding(4) var<storage, read> voxels: array<u32>;
@group(0) @binding(5) var<uniform> point_lights: LightBuffer;

@group(1) @binding(0) var t_light: texture_3d<f32>;
@group(1) @binding(1) var s_light: sampler;

fn unpack_voxel(packed: u32) -> VoxelData {
    return VoxelData(
        packed & 0xFFFFu,
        (packed >> 16u) & 0xFFu,
        (packed >> 24u) & 0xFFu,
    );
}

fn voxel_index(x: i32, y: i32, z: i32) -> u32 {
    return u32(z) * GRID_SIZE * GRID_SIZE + u32(y) * GRID_SIZE + u32(x);
}

fn get_voxel_at(pos: vec3i) -> bool {
    if (pos.x < 0 || pos.x >= i32(GRID_SIZE) ||
        pos.y < 0 || pos.y >= i32(GRID_SIZE) ||
        pos.z < 0 || pos.z >= i32(GRID_SIZE)) {
        return false;
    }
    let idx = voxel_index(pos.x, pos.y, pos.z);
    let packed = voxels[idx];
    return (packed & 0xFFFFu) != 0u;
}

fn vertex_ao(side1: f32, side2: f32, corner: f32) -> f32 {
    // Both sides solid → fully occluded regardless of corner
    if side1 > 0.5 && side2 > 0.5 {
        return 0.0;
    }
    return (3.0 - side1 - side2 - corner) / 3.0;
}

fn get_ao(pos: vec3f, normal: vec3f) -> f32 {
    // Air voxel adjacent to the hit surface
    let p = vec3i(floor(pos + normal * 0.5));

    // Tangent basis for the hit face
    var tang: vec3i;
    var bitang: vec3i;
    if abs(normal.x) > 0.5 {
        tang = vec3i(0, 1, 0);
        bitang = vec3i(0, 0, 1);
    } else if abs(normal.y) > 0.5 {
        tang = vec3i(1, 0, 0);
        bitang = vec3i(0, 0, 1);
    } else {
        tang = vec3i(1, 0, 0);
        bitang = vec3i(0, 1, 0);
    }

    // UV coordinate on the face [0..1]
    let center = vec3f(p) + 0.5;
    let rel = pos - center;
    let u = dot(rel, vec3f(tang)) + 0.5;
    let v = dot(rel, vec3f(bitang)) + 0.5;

    // 8 neighbors in the face-tangent plane: 4 edges + 4 corners
    //   02  12  22
    //   01  --  21
    //   00  10  20
    let s00 = select(0.0, 1.0, get_voxel_at(p - tang - bitang));
    let s10 = select(0.0, 1.0, get_voxel_at(p       - bitang));
    let s20 = select(0.0, 1.0, get_voxel_at(p + tang - bitang));
    let s01 = select(0.0, 1.0, get_voxel_at(p - tang));
    let s21 = select(0.0, 1.0, get_voxel_at(p + tang));
    let s02 = select(0.0, 1.0, get_voxel_at(p - tang + bitang));
    let s12 = select(0.0, 1.0, get_voxel_at(p       + bitang));
    let s22 = select(0.0, 1.0, get_voxel_at(p + tang + bitang));

    // Per-vertex AO: each vertex uses its 2 adjacent edges + 1 diagonal corner
    let ao0 = vertex_ao(s01, s10, s00);
    let ao1 = vertex_ao(s21, s10, s20);
    let ao2 = vertex_ao(s21, s12, s22);
    let ao3 = vertex_ao(s01, s12, s02);

    // Bilinear interpolation across the face
    let ao = mix(mix(ao0, ao1, u), mix(ao3, ao2, u), v);

    return pow(ao, 1.5);
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

fn trace_visibility(origin: vec3f, dir: vec3f, max_dist: f32) -> f32 {
    let inv_dir = 1.0 / dir;
    let bounds = ray_aabb(origin, inv_dir, vec3f(0.0), vec3f(f32(GRID_SIZE)));
    var t_enter = bounds.x;
    let t_exit = bounds.y;

    if t_enter > t_exit || t_exit < 0.0 {
        return 1.0;
    }

    t_enter = max(t_enter, 0.0);
    let entry = origin + dir * (t_enter + 0.001);

    var cell = vec3i(floor(entry));
    cell = clamp(cell, vec3i(0), vec3i(i32(GRID_SIZE) - 1));

    let step = vec3i(sign(dir));
    let t_delta = abs(1.0 / dir);

    var t_max: vec3f;
    if dir.x > 0.0 { t_max.x = (f32(cell.x + 1) - entry.x) * abs(inv_dir.x); }
    else { t_max.x = (entry.x - f32(cell.x)) * abs(inv_dir.x); }
    if dir.y > 0.0 { t_max.y = (f32(cell.y + 1) - entry.y) * abs(inv_dir.y); }
    else { t_max.y = (entry.y - f32(cell.y)) * abs(inv_dir.y); }
    if dir.z > 0.0 { t_max.z = (f32(cell.z + 1) - entry.z) * abs(inv_dir.z); }
    else { t_max.z = (entry.z - f32(cell.z)) * abs(inv_dir.z); }

    if get_voxel_at(cell) {
        return 0.0;
    }

    for (var i = 0u; i < MAX_STEPS; i++) {
        var t_step: f32;
        if t_max.x < t_max.y {
            if t_max.x < t_max.z {
                t_step = t_max.x;
                cell.x += step.x;
                t_max.x += t_delta.x;
            } else {
                t_step = t_max.z;
                cell.z += step.z;
                t_max.z += t_delta.z;
            }
        } else {
            if t_max.y < t_max.z {
                t_step = t_max.y;
                cell.y += step.y;
                t_max.y += t_delta.y;
            } else {
                t_step = t_max.z;
                cell.z += step.z;
                t_max.z += t_delta.z;
            }
        }

        if t_enter + t_step > max_dist {
            return 1.0;
        }

        if cell.x < 0 || cell.x >= i32(GRID_SIZE) ||
           cell.y < 0 || cell.y >= i32(GRID_SIZE) ||
           cell.z < 0 || cell.z >= i32(GRID_SIZE) {
            return 1.0;
        }

        if get_voxel_at(cell) {
            return 0.0;
        }
    }

    return 1.0;
}

fn dda_march(origin: vec3f, dir: vec3f) -> HitResult {
    var result: HitResult;
    result.hit = false;

    let inv_dir = 1.0 / dir;
    let bounds = ray_aabb(origin, inv_dir, vec3f(0.0), vec3f(f32(GRID_SIZE)));
    var t_enter = bounds.x;
    let t_exit = bounds.y;

    if t_enter > t_exit || t_exit < 0.0 {
        return result;
    }

    // Advance ray to grid entry point
    t_enter = max(t_enter, 0.0);
    let entry = origin + dir * (t_enter + 0.001);

    // DDA setup
    var cell = vec3i(floor(entry));
    cell = clamp(cell, vec3i(0), vec3i(i32(GRID_SIZE) - 1));

    let step = vec3i(sign(dir));
    let t_delta = abs(1.0 / dir);

    // Distance to next cell boundary along each axis
    var t_max: vec3f;
    if dir.x > 0.0 { t_max.x = (f32(cell.x + 1) - entry.x) * abs(inv_dir.x); }
    else { t_max.x = (entry.x - f32(cell.x)) * abs(inv_dir.x); }
    if dir.y > 0.0 { t_max.y = (f32(cell.y + 1) - entry.y) * abs(inv_dir.y); }
    else { t_max.y = (entry.y - f32(cell.y)) * abs(inv_dir.y); }
    if dir.z > 0.0 { t_max.z = (f32(cell.z + 1) - entry.z) * abs(inv_dir.z); }
    else { t_max.z = (entry.z - f32(cell.z)) * abs(inv_dir.z); }

    // Check starting cell
    var packed = voxels[voxel_index(cell.x, cell.y, cell.z)];
    if (packed & 0xFFFFu) != 0u {
        result.hit = true;
        result.pos = entry;

        // Determine axis-aligned entry face normal from AABB intersection
        var n = vec3f(0.0);
        if bounds.x > 0.0 {
            // Ray entered grid from outside: use AABB entry face
            let t0 = -origin * inv_dir;
            let t1 = (vec3f(f32(GRID_SIZE)) - origin) * inv_dir;
            let tmin = min(t0, t1);
            if tmin.x >= tmin.y && tmin.x >= tmin.z {
                n.x = -f32(step.x);
            } else if tmin.y >= tmin.x && tmin.y >= tmin.z {
                n.y = -f32(step.y);
            } else {
                n.z = -f32(step.z);
            }
        } else {
            // Camera inside solid voxel: use dominant ray axis
            let a = abs(dir);
            if a.x >= a.y && a.x >= a.z {
                n.x = -f32(step.x);
            } else if a.y >= a.x && a.y >= a.z {
                n.y = -f32(step.y);
            } else {
                n.z = -f32(step.z);
            }
        }
        result.normal = n;

        result.voxel = unpack_voxel(packed);
        return result;
    }

    var last_axis = 0u;
    for (var i = 0u; i < MAX_STEPS; i++) {
        // Step along the axis with smallest t_max
        if t_max.x < t_max.y {
            if t_max.x < t_max.z {
                cell.x += step.x;
                t_max.x += t_delta.x;
                last_axis = 0u;
            } else {
                cell.z += step.z;
                t_max.z += t_delta.z;
                last_axis = 2u;
            }
        } else {
            if t_max.y < t_max.z {
                cell.y += step.y;
                t_max.y += t_delta.y;
                last_axis = 1u;
            } else {
                cell.z += step.z;
                t_max.z += t_delta.z;
                last_axis = 2u;
            }
        }

        // Bounds check
        if cell.x < 0 || cell.x >= i32(GRID_SIZE) ||
           cell.y < 0 || cell.y >= i32(GRID_SIZE) ||
           cell.z < 0 || cell.z >= i32(GRID_SIZE) {
            break;
        }

        packed = voxels[voxel_index(cell.x, cell.y, cell.z)];
        if (packed & 0xFFFFu) != 0u {
            result.hit = true;

            // Compute normal from last axis crossed
            var normal = vec3f(0.0);
            if last_axis == 0u { normal.x = -f32(step.x); }
            else if last_axis == 1u { normal.y = -f32(step.y); }
            else { normal.z = -f32(step.z); }
            result.normal = normal;

            // Compute hit position from t_max
            var t_hit: f32;
            if last_axis == 0u { t_hit = t_max.x - t_delta.x; }
            else if last_axis == 1u { t_hit = t_max.y - t_delta.y; }
            else { t_hit = t_max.z - t_delta.z; }
            result.pos = entry + dir * t_hit;

            result.voxel = unpack_voxel(packed);
            return result;
        }
    }

    return result;
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

    // Attenuation: intensity / (distance^2 + 1.0)
    let intensity = light.color.w;
    let attenuation = intensity / (dist * dist + 1.0);
    
    // Shadow Logic
    var visibility = 1.0;
    if (light.flags == 1u) {
        // Bias start position to avoid self-shadowing
        let shadow_origin = pos + normal * 0.05;
        // Check visibility up to the light source
        visibility = trace_visibility(shadow_origin, light_dir, dist);
    }

    return light.color.rgb * attenuation * ndotl * visibility;
}

fn apply_selection_outline(color: vec3f, hit: HitResult, dir: vec3f) -> vec3f {
    // Check if selection is active
    if (globals.selected_block.w < 0.5) {
        return color;
    }

    let sel_pos = vec3i(globals.selected_block.xyz);
    
    // Identify the voxel coordinate of the hit
    // Offset slightly into the voxel to ensure we floor correctly
    let voxel_pos = vec3i(floor(hit.pos - dir * 0.001));

    if (voxel_pos.x != sel_pos.x || voxel_pos.y != sel_pos.y || voxel_pos.z != sel_pos.z) {
        return color;
    }

    // Determine local UV on the face [0..1]
    // We can use the fractional part of hit.pos, but we need to know which face.
    // normal tells us the axis.
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

    // Distance to nearest edge
    let d = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    
    // Thickness of the line
    let thickness = 0.02;

    if (d < thickness) {
        return vec3f(0.0); // Black outline
    }

    return color;
}

fn shade_pbr(hit: HitResult, dir: vec3f, screen_pos: vec2f) -> vec3f {
    let pu = hit.voxel.id % 256u;
    let pv = hit.voxel.id / 256u;
    let coords = vec2i(i32(pu), i32(pv));

    // Palette layer 0: albedo (sRGB decoded by hardware)
    let albedo = textureLoad(t_palette, coords, 0, 0).rgb;

    // Palette layer 1: material properties
    // R=Roughness, G=Emission, B=Noise, A=Metallic (packed as u8 unorm)
    let data = textureLoad(t_palette, coords, 1, 0);
    let roughness = data.r;
    let emission = data.g;
    let noise_strength = data.b;
    let metallic = data.a;

    // Emission: lava glow
    // Scale up emission to make it bloom
    let emit = albedo * emission * 2.0;

    // Apply Noise Variation to Albedo
    // Use the voxel integer coordinate to seed the hash for variation.
    let voxel_pos = floor(hit.pos - dir * 0.001);
    let noise_val = hash(voxel_pos);
    
    // Modulate albedo based on noise strength (e.g. stone has high variation).
    // Modulate albedo based on noise strength (e.g. stone has high variation).
    let noise_mod = 1.0 - (noise_strength * noise_val * 0.5);
    let noisy_albedo = albedo * noise_mod;

    // Sample Light from Voxel Grid (Indirect Lighting)
    // Offset significantly along normal to allow Linear Interpolation across the face
    let light_uvw = (hit.pos + hit.normal * 0.1) / vec3f(f32(GRID_SIZE));
    
    // Sample Level 0. Linear filtering enabled.
    // No more thresholding or clamping. Dark is dark.
    let voxel_light = textureSampleLevel(t_light, s_light, light_uvw, 0.0).rgb;

    // Calculate AO
    // AO affects Indirect (voxel) light.
    let ao_val = get_ao(hit.pos, hit.normal);
    // Soften AO slightly so it's not pitch black in corners immediately, but keep it strong.
    // Or strictly: AO * Light. Let's stick to simple multiplication for physical consistency.
    let ao = pow(ao_val, 1.0); // Simple linear usage of AO factor

    // Sun directional light (Direct Lighting)
    let ndotl = max(dot(hit.normal, -globals.sun_dir.xyz), 0.0);
    var sun_light = globals.sun_color.rgb * globals.sun_dir.w * ndotl;
    
    if ndotl > 0.0 {
        // Hard shadows for sunlight
        let shadow_origin = hit.pos + hit.normal * 0.05;
        let sun_vis = trace_visibility(shadow_origin, -globals.sun_dir.xyz, globals.sun_shadow_max);
        sun_light *= sun_vis;
    }

    // Dynamic Point Lights
    var dynamic_light = vec3f(0.0);
    let count = min(point_lights.count, 16u);
    
    for (var i = 0u; i < count; i++) {
        dynamic_light += evaluate_point_light(point_lights.lights[i], hit.pos, hit.normal);
    }

    // Total Irradiance = Direct + Indirect (occluded) + Dynamic
    let total_light = sun_light + voxel_light * ao + dynamic_light;

    // Procedural Bump Mapping
    let noise_scale = 150.0;
    let bump_intensity = 0.08;
    let random_vec = hash33(hit.pos * noise_scale) * 2.0 - 1.0;
    let perturbation = random_vec * noise_strength * bump_intensity;
    let n = normalize(hit.normal + perturbation);

    // View-dependent specular
    var specular = vec3f(0.0);
    // Calculate specular only if there is light
    let light_intensity = max(total_light.r, max(total_light.g, total_light.b));

    if (light_intensity > 0.0001) {
        let v = -dir;
        let view_dot_n = max(dot(v, n), 0.0);

        let f0 = mix(vec3f(0.04), noisy_albedo, metallic);
        let f90 = max(vec3f(1.0 - roughness), f0);
        let fresnel = f0 + (f90 - f0) * pow(1.0 - view_dot_n, 5.0);

        // Specular is added on top
        specular = fresnel * total_light * (1.0 - roughness);

        // Mask specular on very rough surfaces
        if (roughness > 0.9) {
            specular = vec3f(0.0);
        }
    }

    let diffuse = noisy_albedo * (1.0 - metallic);

    // Final composition
    // (Diffuse * Light + Specular) + Emission
    // Emission is added at the end for visual glow of the surface itself.
    var final_hdr = (diffuse * total_light + specular) + emit;
    
    // Apply Selection Outline
    final_hdr = apply_selection_outline(final_hdr, hit, dir);

    return final_hdr;
}

fn sky(dir: vec3f) -> vec3f {
    // Y-up: positive dir.y is towards zenith
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    // Mix between horizon (darker/foggy) and zenith (sky color)
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
    // WGPU uses Y-up NDC (-1 bottom, +1 top).
    // Our projection matrix is optimized for Vulkan Y-down (-1 top, +1 bottom).
    out.ndc = vec2f(x, -y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    // NDC → clip → camera space via inverse projection
    let clip = vec4f(in.ndc.x, in.ndc.y, 1.0, 1.0);
    let cam_space = globals.proj_inverse * clip;
    let cam_dir = normalize(cam_space.xyz / cam_space.w);

    // Camera space → world space direction (w=0 for direction)
    let world_dir4 = globals.view_inverse * vec4f(cam_dir, 0.0);
    let dir = normalize(world_dir4.xyz);

    let origin = globals.cam_pos.xyz;
    let hit = dda_march(origin, dir);

    var color: vec3f;
    if hit.hit {
        color = shade_pbr(hit, dir, in.position.xy);
    } else {
        color = sky(dir);
    }

    return vec4f(color, 1.0);
}
