struct GlobalUniforms {
    view_inverse: mat4x4f,
    proj_inverse: mat4x4f,
    cam_pos: vec4f,
    time: f32,
    resolution_x: f32,
    resolution_y: f32,
    _pad: f32,
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

const GRID_SIZE: u32 = 64u;
const MAX_STEPS: u32 = 256u;

@group(0) @binding(0) var<uniform> globals: GlobalUniforms;
@group(0) @binding(2) var t_palette: texture_2d_array<f32>;
@group(0) @binding(3) var s_palette: sampler;
@group(0) @binding(4) var<storage, read> voxels: array<u32>;

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

fn get_ao(pos: vec3f, normal: vec3f) -> f32 {
    let p = vec3i(floor(pos + normal * 0.5));
    
    // We want to sample the 4 neighbors around the vertex on the face
    // But since we are raymarching, we have a hit position which is anywhere on the face.
    // A simple AO valid for blocks is to check the 4 diagonal neighbors responsible for occlusion on that face.

    // Let's create a basis.
    var u = vec3f(0.0);
    var v = vec3f(0.0);

    if (abs(normal.x) > 0.5) {
        u = vec3f(0.0, 1.0, 0.0);
        v = vec3f(0.0, 0.0, 1.0);
    } else if (abs(normal.y) > 0.5) {
        u = vec3f(1.0, 0.0, 0.0);
        v = vec3f(0.0, 0.0, 1.0);
    } else {
        u = vec3f(1.0, 0.0, 0.0);
        v = vec3f(0.0, 1.0, 0.0);
    }

    // Determine uv coordinates on the face (0..1)
    let rel = pos - (vec3f(p) + vec3f(0.5));
    let uv = vec2f(dot(rel, u), dot(rel, v)) + 0.5; // 0..1

    // Neighbor offsets
    let off_u = vec3i(u);
    let off_v = vec3i(v);

    // Check 4 corners (neighbors)
    //  3 -- 2
    //  |    |
    //  0 -- 1
    
    // Neighbors in the plane perpendicular to normal
    let n0 = get_voxel_at(p - off_u - off_v);
    let n1 = get_voxel_at(p + off_u - off_v);
    let n2 = get_voxel_at(p + off_u + off_v);
    let n3 = get_voxel_at(p - off_u + off_v);

    // 0 means occluded, 1 means clear.
    let occ0 = select(1.0, 0.0, n0);
    let occ1 = select(1.0, 0.0, n1);
    let occ2 = select(1.0, 0.0, n2);
    let occ3 = select(1.0, 0.0, n3);

    // Bilinear interpolation
    let ao = mix(
        mix(occ0, occ1, uv.x),
        mix(occ3, occ2, uv.x),
        uv.y
    );
    
    // Curve it for strength
    return smoothstep(0.0, 1.0, ao * 0.5 + 0.5);
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

fn shade_pbr(hit: HitResult, dir: vec3f) -> vec3f {
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
    let emit = albedo * emission * 10.0;

    // Apply Noise Variation to Albedo
    // Use the voxel integer coordinate to seed the hash for variation.
    let voxel_pos = floor(hit.pos - dir * 0.001);
    let noise_val = hash(voxel_pos);
    
    // Modulate albedo based on noise strength (e.g. stone has high variation).
    let noise_mod = 1.0 - (noise_strength * noise_val * 0.5);
    let noisy_albedo = albedo * noise_mod;

    // Sample Light from Voxel Grid
    // Offset slightly along normal to sample the "air" block next to the surface
    let light_uvw = (hit.pos + hit.normal * 0.5) / vec3f(f32(GRID_SIZE));
    
    // Sample Level 0 to avoid mipmap issues with manual gradient
    let voxel_light = textureSampleLevel(t_light, s_light, light_uvw, 0.0).rgb;

    // Calculate AO
    let ao = get_ao(hit.pos, hit.normal);

    // Pure Voxel Lighting Model
    // No analytical sun. The "sun" is just bright voxels in the light texture.
    
    // View-dependent specular (Wet/Shiny look)
    // Only apply if the surface is receiving significant light
    var specular = vec3f(0.0);
    let light_intensity = max(voxel_light.r, max(voxel_light.g, voxel_light.b));
    
    // Procedural Bump Mapping
    let noise_scale = 150.0;
    let bump_intensity = 0.08;
    let random_vec = hash33(hit.pos * noise_scale) * 2.0 - 1.0;
    let perturbation = random_vec * noise_strength * bump_intensity;
    let n = normalize(hit.normal + perturbation);

    // Use extracted properties for specular logic
    if (light_intensity > 0.05) {
        let v = -dir;
        
        let view_dot_n = max(dot(v, n), 0.0);

        // F0: Surface reflection at 0 degrees
        let f0 = mix(vec3f(0.04), noisy_albedo, metallic);

        // Calculate Dampened F90: Reduce grazing angle reflection based on roughness
        let f90 = max(vec3f(1.0 - roughness), f0);

        // Modified Fresnel Schlick with dampened F90
        let fresnel = f0 + (f90 - f0) * pow(1.0 - view_dot_n, 5.0);
        
        // Final Specular Attenuation (The "Matte Hammer")
        // Multiply by inverse roughness to ensure rough materials are truly matte.
        specular = fresnel * voxel_light * (1.0 - roughness);
    }

    // Final Color Composition
    let diffuse = noisy_albedo * (1.0 - metallic);
    
    let final_color = diffuse * voxel_light * ao + emit + specular;

    return final_color;
}

fn sky(dir: vec3f) -> vec3f {
    // Y-up: positive dir.y is towards zenith
    return vec3f(0.0);
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
        color = shade_pbr(hit, dir);
    } else {
        color = sky(dir);
    }

    return vec4f(color, 1.0);
}
