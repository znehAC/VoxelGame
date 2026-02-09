@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;

struct PushConstants {
    direction: vec2f, // (1, 0) for horizontal, (0, 1) for vertical
}
var<push_constant> pc: PushConstants;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) % 2) * 4.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}

// 9-tap Gaussian blur weights
// sigma ~ 2.0
const WEIGHTS = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let dims = vec2f(textureDimensions(t_input));
    let uv = frag_coord.xy / dims;
    let texel_size = 1.0 / dims;
    
    // Center tap
    var result = textureSample(t_input, s_input, uv).rgb * WEIGHTS[0];
    
    // Offset taps
    for (var i = 1; i < 5; i++) {
        let offset = pc.direction * texel_size * f32(i);
        result += textureSample(t_input, s_input, uv + offset).rgb * WEIGHTS[i];
        result += textureSample(t_input, s_input, uv - offset).rgb * WEIGHTS[i];
    }
    
    return vec4f(result, 1.0);
}
