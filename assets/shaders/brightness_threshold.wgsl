@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;

struct PushConstants {
    threshold: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
}
var<push_constant> pc: PushConstants;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let x = f32(i32(vi) / 2) * 4.0 - 1.0;
    let y = f32(i32(vi) % 2) * 4.0 - 1.0;
    
    var out: VertexOutput;
    out.position = vec4f(x, y, 0.0, 1.0);
    out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let color = textureSample(t_input, s_input, in.uv).rgb;
    
    // Calculate luminance (perceptual)
    let luminance = dot(color, vec3f(0.2126, 0.7152, 0.0722));
    
    if luminance > pc.threshold {
        return vec4f(color, 1.0);
    } else {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
}
