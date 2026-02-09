@group(0) @binding(0) var t_scene: texture_2d<f32>;
@group(0) @binding(1) var s_scene: sampler;
@group(0) @binding(2) var t_bloom: texture_2d<f32>;
@group(0) @binding(3) var s_bloom: sampler;

struct PushConstants {
    bloom_intensity: f32, // e.g. 0.5
    exposure: f32,        // e.g. 1.0
}
var<push_constant> pc: PushConstants;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) % 2) * 4.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let dims = vec2f(textureDimensions(t_scene));
    let uv = frag_coord.xy / dims;

    let scene_color = textureSample(t_scene, s_scene, uv).rgb;
    let bloom_color = textureSample(t_bloom, s_bloom, uv).rgb;
    
    // Additive blending
    var result = scene_color + bloom_color * pc.bloom_intensity;
    
    // Tone mapping (exposure + Reinhard)
    result = vec3f(1.0) - exp(-result * pc.exposure);

    return vec4f(result, 1.0);
}
