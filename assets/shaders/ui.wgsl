struct VertexInput {
    @location(0) position: vec2f, // Screen position (pixels? or 0..1?) Let's use 0..width, 0..height
    @location(1) uv: vec2f,
    @location(2) color: vec4f,
    @location(3) mode: u32,       // 0=Textured, 1=Solid, 2=Text
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
    @location(2) @interpolate(flat) mode: u32,
}

struct Uniforms {
    screen_size: vec2f, // width, height
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var t_diffuse: texture_2d<f32>;
@group(0) @binding(2) var s_diffuse: sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Convert screen coordinates (0..width, 0..height) to NDC (-1..1, 1..-1)
    // x: 0 -> -1, width -> 1
    // y: 0 -> 1, height -> -1 (WGPU is Y-up in NDC? No, wait. 
    // WGPU NDC: (-1, -1) is bottom-left. (1, 1) is top-right.
    // Screen coords usually (0,0) top-left.
    
    let x = (in.position.x / uniforms.screen_size.x) * 2.0 - 1.0;
    let y = 1.0 - (in.position.y / uniforms.screen_size.y) * 2.0;
    
    out.position = vec4f(x, y, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.mode = in.mode;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    var color = in.color;
    
    if (in.mode == 0u) { // Textured
        let tex = textureSample(t_diffuse, s_diffuse, in.uv);
        color = color * tex;
    } else if (in.mode == 1u) { // Solid
        // color is already set
    } else if (in.mode == 2u) { // Text (Alpha Mask)
        let alpha = textureSample(t_diffuse, s_diffuse, in.uv).a;
        color.a *= alpha;
    }
    
    return color;
}
