// SMAA 1x Implementation (WGSL)
// Based on official SMAA by Jorge Jimenez et al.

// --- Config ---
const SMAA_THRESHOLD: f32 = 0.1;
const SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR: f32 = 2.0;
const SMAA_MAX_SEARCH_STEPS: i32 = 16;
const SMAA_MAX_SEARCH_STEPS_DIAG: i32 = 8;
const SMAA_CORNER_ROUNDING: f32 = 25.0 / 100.0;
const SMAA_AREATEX_MAX_DISTANCE: f32 = 16.0;
const SMAA_AREATEX_MAX_DISTANCE_DIAG: f32 = 20.0;

// AreaTex: 160 x 560, RG8_UNORM
const SMAA_AREATEX_PIXEL_SIZE: vec2f = vec2f(1.0 / 160.0, 1.0 / 560.0);
const SMAA_AREATEX_SUBTEX_SIZE: f32 = 1.0 / 7.0;

// SearchTex: 66 x 33 conceptual, packed into 64 x 16
const SMAA_SEARCHTEX_SIZE: vec2f = vec2f(66.0, 33.0);
const SMAA_SEARCHTEX_PACKED_SIZE: vec2f = vec2f(64.0, 16.0);

// --- Vertex I/O ---
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) pix_coord: vec2f,
    @location(2) offset0: vec4f,
    @location(3) offset1: vec4f,
    @location(4) offset2: vec4f,
};

@group(0) @binding(0) var t_color: texture_2d<f32>;
@group(0) @binding(1) var t_search: texture_2d<f32>;
@group(0) @binding(2) var t_area: texture_2d<f32>;
@group(0) @binding(3) var s_linear: sampler;
@group(0) @binding(4) var s_point: sampler;

struct SmaaUniforms {
    rt_metrics: vec4f,
    debug_mode: u32,
}
@group(0) @binding(5) var<uniform> u: SmaaUniforms;

// --- Utility Functions ---
fn luma(color: vec3f) -> f32 {
    return dot(color, vec3f(0.2126, 0.7152, 0.0722));
}

// --- Vertex Shaders ---
fn SMAAEdgeDetectionVS(texcoord: vec2f) -> VertexOutput {
    var out: VertexOutput;
    out.uv = texcoord;
    out.pix_coord = texcoord * u.rt_metrics.zw;
    out.offset0 = texcoord.xyxy + u.rt_metrics.xyxy * vec4f(-1.0, 0.0, 0.0, -1.0);
    out.offset1 = texcoord.xyxy + u.rt_metrics.xyxy * vec4f(1.0, 0.0, 0.0, 1.0);
    out.offset2 = texcoord.xyxy + u.rt_metrics.xyxy * vec4f(-2.0, 0.0, 0.0, -2.0);
    return out;
}

fn SMAABlendingWeightCalculationVS(texcoord: vec2f) -> VertexOutput {
    var out: VertexOutput;
    out.uv = texcoord;
    out.pix_coord = texcoord * u.rt_metrics.zw;
    out.offset0 = texcoord.xyxy + u.rt_metrics.xyxy * vec4f(-0.25, -0.125, 1.25, -0.125);
    out.offset1 = texcoord.xyxy + u.rt_metrics.xyxy * vec4f(-0.125, -0.25, -0.125, 1.25);
    let max_search = f32(SMAA_MAX_SEARCH_STEPS);
    out.offset2 = texcoord.xyxy + u.rt_metrics.xxyy * vec4f(-2.0, 2.0, -2.0, 2.0) * vec4f(max_search);
    return out;
}

fn SMAANeighborhoodBlendingVS(texcoord: vec2f) -> VertexOutput {
    var out: VertexOutput;
    out.uv = texcoord;
    out.pix_coord = texcoord * u.rt_metrics.zw;
    out.offset0 = texcoord.xyxy + u.rt_metrics.xyxy * vec4f(1.0, 0.0, 0.0, 1.0);
    out.offset1 = vec4f(0.0);
    out.offset2 = vec4f(0.0);
    return out;
}

@vertex
fn vs_edges(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let uv = vec2f(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    let pos = vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
    var out = SMAAEdgeDetectionVS(uv);
    out.position = pos;
    return out;
}

@vertex
fn vs_weights(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let uv = vec2f(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    let pos = vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
    var out = SMAABlendingWeightCalculationVS(uv);
    out.position = pos;
    return out;
}

@vertex
fn vs_blend(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let uv = vec2f(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    let pos = vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
    var out = SMAANeighborhoodBlendingVS(uv);
    out.position = pos;
    return out;
}

// --- Search Functions ---
fn SMAASearchLength(e: vec2f, offset: f32) -> f32 {
    var scale = SMAA_SEARCHTEX_SIZE * vec2f(0.5, -1.0);
    var bias = SMAA_SEARCHTEX_SIZE * vec2f(offset, 1.0);
    scale = scale + vec2f(-1.0, 1.0);
    bias = bias + vec2f(0.5, -0.5);
    scale = scale / SMAA_SEARCHTEX_PACKED_SIZE;
    bias = bias / SMAA_SEARCHTEX_PACKED_SIZE;
    return textureSampleLevel(t_search, s_linear, scale * e + bias, 0.0).r;
}

fn SMAASearchXLeft(edgesTex: texture_2d<f32>, texcoord: vec2f, end: f32) -> f32 {
    var e = vec2f(0.0, 1.0);
    var coord = texcoord;
    
    while (coord.x > end && e.g > 0.8281 && e.r == 0.0) {
        e = textureSampleLevel(edgesTex, s_linear, coord, 0.0).rg;
        coord.x = coord.x - u.rt_metrics.x * 2.0;
    }
    
    let offset = -(255.0 / 127.0) * SMAASearchLength(e, 0.0) + 3.25;
    return u.rt_metrics.x * offset + coord.x;
}

fn SMAASearchXRight(edgesTex: texture_2d<f32>, texcoord: vec2f, end: f32) -> f32 {
    var e = vec2f(0.0, 1.0);
    var coord = texcoord;
    
    while (coord.x < end && e.g > 0.8281 && e.r == 0.0) {
        e = textureSampleLevel(edgesTex, s_linear, coord, 0.0).rg;
        coord.x = coord.x + u.rt_metrics.x * 2.0;
    }
    
    let offset = -(255.0 / 127.0) * SMAASearchLength(e, 0.5) + 3.25;
    return -u.rt_metrics.x * offset + coord.x;
}

fn SMAASearchYUp(edgesTex: texture_2d<f32>, texcoord: vec2f, end: f32) -> f32 {
    var e = vec2f(1.0, 0.0);
    var coord = texcoord;
    
    while (coord.y > end && e.r > 0.8281 && e.g == 0.0) {
        e = textureSampleLevel(edgesTex, s_linear, coord, 0.0).rg;
        coord.y = coord.y - u.rt_metrics.y * 2.0;
    }
    
    let offset = -(255.0 / 127.0) * SMAASearchLength(e.gr, 0.0) + 3.25;
    return u.rt_metrics.y * offset + coord.y;
}

fn SMAASearchYDown(edgesTex: texture_2d<f32>, texcoord: vec2f, end: f32) -> f32 {
    var e = vec2f(1.0, 0.0);
    var coord = texcoord;
    
    while (coord.y < end && e.r > 0.8281 && e.g == 0.0) {
        e = textureSampleLevel(edgesTex, s_linear, coord, 0.0).rg;
        coord.y = coord.y + u.rt_metrics.y * 2.0;
    }
    
    let offset = -(255.0 / 127.0) * SMAASearchLength(e.gr, 0.5) + 3.25;
    return -u.rt_metrics.y * offset + coord.y;
}

// --- Area Sampling ---
fn SMAAArea(dist_sqrt: vec2f, e1: f32, e2: f32, offset: f32) -> vec2f {
    let texcoord_pixels = vec2f(SMAA_AREATEX_MAX_DISTANCE) * round(4.0 * vec2f(e1, e2)) + dist_sqrt;
    var texcoord = SMAA_AREATEX_PIXEL_SIZE * texcoord_pixels + 0.5 * SMAA_AREATEX_PIXEL_SIZE;
    texcoord.y = SMAA_AREATEX_SUBTEX_SIZE * offset + texcoord.y;
    return textureSampleLevel(t_area, s_linear, texcoord, 0.0).rg;
}

// --- Fragment: Edge Detection ---
@fragment
fn fs_edges(in: VertexOutput) -> @location(0) vec4f {
    let L = luma(textureSample(t_color, s_linear, in.uv).rgb);
    let Lleft = luma(textureSample(t_color, s_linear, in.offset0.xy).rgb);
    let Ltop = luma(textureSample(t_color, s_linear, in.offset0.zw).rgb);

    let delta_xy = abs(L - vec2f(Lleft, Ltop));
    var edges = step(vec2f(SMAA_THRESHOLD), delta_xy);

    if (dot(edges, vec2f(1.0)) == 0.0) {
        discard;
    }

    let Lright = luma(textureSample(t_color, s_linear, in.offset1.xy).rgb);
    let Lbottom = luma(textureSample(t_color, s_linear, in.offset1.zw).rgb);
    let delta_zw = abs(L - vec2f(Lright, Lbottom));

    var max_delta = max(delta_xy, delta_zw);

    let Lleftleft = luma(textureSample(t_color, s_linear, in.offset2.xy).rgb);
    let Ltoptop = luma(textureSample(t_color, s_linear, in.offset2.zw).rgb);
    let delta_zw2 = abs(vec2f(Lleft, Ltop) - vec2f(Lleftleft, Ltoptop));

    max_delta = max(max_delta, delta_zw2);
    let final_delta = max(max_delta.x, max_delta.y);

    edges = edges * step(vec2f(final_delta), SMAA_LOCAL_CONTRAST_ADAPTATION_FACTOR * delta_xy);
    
    if (u.debug_mode == 1u) { return vec4f(in.uv, 0.0, 1.0); }
    if (u.debug_mode == 2u) { return vec4f(delta_xy * 5.0, 0.0, 1.0); }
    if (u.debug_mode == 3u) { return vec4f(edges, 0.0, 1.0); }

    return vec4f(edges, 0.0, 0.0);
}

// --- Fragment: Blend Weights ---
@fragment
fn fs_weights(in: VertexOutput) -> @location(0) vec4f {
    var weights = vec4f(0.0);
    let e = textureSample(t_color, s_linear, in.uv).rg;
    
    if (u.debug_mode == 4u) { return vec4f(e, 0.0, 1.0); }

    if (e.g > 0.0) {
        var d: vec2f;
        var coords: vec3f;
        coords.x = SMAASearchXLeft(t_color, in.offset0.xy, in.offset2.x);
        coords.y = in.offset1.y;
        d.x = coords.x;

        let e1 = textureSampleLevel(t_color, s_linear, coords.xy, 0.0).r;
        coords.z = SMAASearchXRight(t_color, in.offset0.zw, in.offset2.y);
        d.y = coords.z;

        d = abs(round(d * u.rt_metrics.zz - in.pix_coord.xx));
        let sqrt_d = sqrt(d);
        let e2 = textureSampleLevel(t_color, s_linear, vec2f(coords.z, coords.y), 0.0).r;
        let area = SMAAArea(sqrt_d, e1, e2, 0.0);
        weights = vec4f(area.r, area.g, weights.b, weights.a);
        
        if (u.debug_mode == 5u) { return vec4f(vec2f(d) / 16.0, 0.0, 1.0); }
        if (u.debug_mode == 6u) { return vec4f(weights.rg, 0.0, 1.0); }
    }

    if (e.r > 0.0) {
        var d: vec2f;
        var coords: vec3f;
        coords.y = SMAASearchYUp(t_color, in.offset1.xy, in.offset2.z);
        coords.x = in.offset0.x;
        d.x = coords.y;

        let e1 = textureSampleLevel(t_color, s_linear, coords.xy, 0.0).g;
        coords.z = SMAASearchYDown(t_color, in.offset1.zw, in.offset2.w);
        d.y = coords.z;

        d = abs(round(d * u.rt_metrics.ww - in.pix_coord.yy));
        let sqrt_d = sqrt(d);
        let e2 = textureSampleLevel(t_color, s_linear, vec2f(coords.x, coords.z), 0.0).g;
        let area2 = SMAAArea(sqrt_d, e1, e2, 1.0);
        weights = vec4f(weights.r, weights.g, area2.r, area2.g);
        
        if (u.debug_mode == 7u && weights.r < 0.01 && weights.g < 0.01) {
            return vec4f(0.0, weights.b, weights.a, 1.0);
        }
    }

    if (u.debug_mode == 8u) {
        return vec4f(weights.r + weights.b, weights.g + weights.a, max(weights.b, weights.a), 1.0);
    }

    return weights;
}

// --- Fragment: Neighborhood Blending ---
@fragment
fn fs_blend(in: VertexOutput) -> @location(0) vec4f {
    if (u.debug_mode == 9u) {
        let w = textureSample(t_area, s_linear, in.uv);
        return vec4f(w.r + w.b, w.g + w.a, max(w.b, w.a), 1.0);
    }

    let a_x = textureSample(t_area, s_linear, in.offset0.xy).a;
    let a_y = textureSample(t_area, s_linear, in.offset0.zw).g;
    let a_z = textureSample(t_area, s_linear, in.uv).x;
    let a_w = textureSample(t_area, s_linear, in.uv).z;

    if ((a_x + a_y + a_z + a_w) < 1e-5) {
        return textureSample(t_color, s_linear, in.uv);
    }

    let h = max(a_x, a_z) > max(a_y, a_w);

    var blending_offset: vec4f;
    var blending_weight: vec2f;
    if (h) {
        blending_offset = vec4f(a_x, 0.0, a_z, 0.0);
        blending_weight = vec2f(a_x, a_z);
    } else {
        blending_offset = vec4f(0.0, a_y, 0.0, a_w);
        blending_weight = vec2f(a_y, a_w);
    }
    blending_weight = blending_weight / (blending_weight.x + blending_weight.y);

    let blending_coord = blending_offset * vec4f(u.rt_metrics.xy, -u.rt_metrics.xy) + in.uv.xyxy;

    let color = blending_weight.x * textureSample(t_color, s_linear, blending_coord.xy);
    return color + blending_weight.y * textureSample(t_color, s_linear, blending_coord.zw);
}
