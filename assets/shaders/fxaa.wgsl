// FXAA 3.11 Implementation
// Adapted for WGPU/WGSL

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Full-screen triangle
    let uv = vec2f(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u)
    );
    let position = vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
    // Invert Y for WGPU if needed, but standard UV (0,0 top-left) works if we sample consistently.
    // WGPU clip space is Y up, UVs are Y down usually.
    // Let's stick to standard full screen triangle UVs: (0,0) to (2,2) -> (0,0) to (1,1) visible.
    // Screen coords: (-1, -1) to (3, 3).
    // Visible quad: (-1, -1) to (1, 1).
    // UVs: (0, 1) bottom-left, (2, 1) bottom-right, (0, -1) top-left... wait.
    
    // Standard approach:
    // 0: (-1, -1), uv (0, 1) -> Bottom Left
    // 1: ( 3, -1), uv (2, 1) -> Bottom Right
    // 2: (-1,  3), uv (0, -1) -> Top Left
    // WGPU Texture coords: (0,0) is Top-Left.
    // WGPU Clip Space: (-1, -1) is Bottom-Left.
    
    // Let's use the logic from existing shaders if available, or standard:
    // positions: (-1, 1), (3, 1), (-1, -3) -> UVs (0,0), (2,0), (0,2) ?
    
    var out: VertexOutput;
    out.uv = vec2f(f32((vertex_index << 1u) & 2u), f32(vertex_index & 2u));
    out.position = vec4f(out.uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
    return out;
}

@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_linear: sampler;
@group(0) @binding(2) var<uniform> resolution: vec2f;

// FXAA Constants
const FXAA_EDGE_THRESHOLD: f32 = 0.0625; // Lower = more edges detected
const FXAA_EDGE_THRESHOLD_MIN: f32 = 0.0312;
const FXAA_SUBPIX_TRIM: f32 = 1.0/8.0; // Lower = start blending on lower contrast sub-pixels
const FXAA_SUBPIX_TRIM_SCALE: f32 = 1.0/(1.0 - FXAA_SUBPIX_TRIM);
const FXAA_SUBPIX_CAP: f32 = 0.875; // Higher = more blur on sub-pixels
const FXAA_SEARCH_STEPS: i32 = 12; // Maximum search steps

fn rgb2luma(rgb: vec3f) -> f32 {
    return dot(rgb, vec3f(0.299, 0.587, 0.114));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let inverse_screen_size = 1.0 / resolution;
    
    // 1. Local Contrast Check
    let rgbM = textureSample(t_input, s_linear, in.uv).rgb;
    let lumaM = rgb2luma(rgbM);
    
    let lumaS = rgb2luma(textureSample(t_input, s_linear, in.uv + vec2f(0.0, 1.0) * inverse_screen_size).rgb);
    let lumaN = rgb2luma(textureSample(t_input, s_linear, in.uv + vec2f(0.0, -1.0) * inverse_screen_size).rgb);
    let lumaW = rgb2luma(textureSample(t_input, s_linear, in.uv + vec2f(-1.0, 0.0) * inverse_screen_size).rgb);
    let lumaE = rgb2luma(textureSample(t_input, s_linear, in.uv + vec2f(1.0, 0.0) * inverse_screen_size).rgb);
    
    let maxLuma = max(lumaM, max(max(lumaN, lumaW), max(lumaS, lumaE)));
    let minLuma = min(lumaM, min(min(lumaN, lumaW), min(lumaS, lumaE)));
    let contrast = maxLuma - minLuma;
    
    // Early exit if contrast is too low
    if (contrast < max(FXAA_EDGE_THRESHOLD_MIN, maxLuma * FXAA_EDGE_THRESHOLD)) {
        return vec4f(rgbM, 1.0); // Normal render for non-edges
        // return vec4f(0.0, 1.0, 0.0, 1.0); // DEBUG: Green for skipped pixels
    }
    
    // 2. Edge Direction
    let lumaNW = rgb2luma(textureSample(t_input, s_linear, in.uv + vec2f(-1.0, -1.0) * inverse_screen_size).rgb);
    let lumaNE = rgb2luma(textureSample(t_input, s_linear, in.uv + vec2f(1.0, -1.0) * inverse_screen_size).rgb);
    let lumaSW = rgb2luma(textureSample(t_input, s_linear, in.uv + vec2f(-1.0, 1.0) * inverse_screen_size).rgb);
    let lumaSE = rgb2luma(textureSample(t_input, s_linear, in.uv + vec2f(1.0, 1.0) * inverse_screen_size).rgb);
    
    let g1 = (lumaNW + lumaNE + lumaSW + lumaSE);
    let g2 = (lumaN + lumaS + lumaW + lumaE);
    // This is part of the direction estimation, but standard FXAA uses:
    // Horizontal: |(NW - N) - (NE - N)| + 2|(W - M) - (E - M)| + |(SW - S) - (SE - S)|
    // Vertical:   |(NW - W) - (SW - W)| + 2|(N - M) - (S - M)| + |(NE - E) - (SE - E)|
    
    let dirSwMinusNe = lumaSW - lumaNE;
    let dirSeMinusNw = lumaSE - lumaNW;
    
    // Simple gradient calculation
    let gradientN = abs(lumaN - lumaM);
    let gradientS = abs(lumaS - lumaM);
    let gradientW = abs(lumaW - lumaM);
    let gradientE = abs(lumaE - lumaM);
    
    // More robust direction estimation
    let vertEdge = 
        abs((0.25 * lumaNW) + (-0.5 * lumaN) + (0.25 * lumaNE)) +
        abs((0.50 * lumaW ) + (-1.0 * lumaM) + (0.50 * lumaE )) +
        abs((0.25 * lumaSW) + (-0.5 * lumaS) + (0.25 * lumaSE));
        
    let horzEdge = 
        abs((0.25 * lumaNW) + (-0.5 * lumaW) + (0.25 * lumaSW)) +
        abs((0.50 * lumaN ) + (-1.0 * lumaM) + (0.50 * lumaS )) +
        abs((0.25 * lumaNE) + (-0.5 * lumaE) + (0.25 * lumaSE));
        
    let isHorz = horzEdge > vertEdge;
    
    // 3. Select Edge Pair
    // Determine if the gradient is stronger in the positive or negative direction along the chosen axis
    var luma1 = lumaN;
    var luma2 = lumaS;
    if (isHorz) {
        luma1 = lumaN;
        luma2 = lumaS;
    } else {
        luma1 = lumaW;
        luma2 = lumaE;
    }
    
    let gradient1 = abs(luma1 - lumaM);
    let gradient2 = abs(luma2 - lumaM);
    
    let is1Steepest = gradient1 >= gradient2;
    let gradientScaled = 0.25 * max(gradient1, gradient2);
    
    var stepLength = inverse_screen_size.y;
    var step = vec2f(0.0, inverse_screen_size.y);
    
    if (isHorz) {
        stepLength = inverse_screen_size.x;
        step = vec2f(inverse_screen_size.x, 0.0);
    }
    
    var lumaLocalAverage = 0.0;
    if (is1Steepest) {
        stepLength = -stepLength;
        step = -step;
        lumaLocalAverage = 0.5 * (luma1 + lumaM);
    } else {
        lumaLocalAverage = 0.5 * (luma2 + lumaM);
    }
    
    var currentUv = in.uv;
    if (isHorz) {
        currentUv.y += stepLength * 0.5;
    } else {
        currentUv.x += stepLength * 0.5;
    }
    
    // 4. Search Loop
    var uv1 = currentUv - step;
    var uv2 = currentUv + step;
    
    var lumaEnd1 = rgb2luma(textureSample(t_input, s_linear, uv1).rgb);
    var lumaEnd2 = rgb2luma(textureSample(t_input, s_linear, uv2).rgb);
    lumaEnd1 -= lumaLocalAverage;
    lumaEnd2 -= lumaLocalAverage;
    
    var reached1 = abs(lumaEnd1) >= gradientScaled;
    var reached2 = abs(lumaEnd2) >= gradientScaled;
    var reachedBoth = reached1 && reached2;
    
    if (!reached1) {
        uv1 -= step;
    }
    if (!reached2) {
        uv2 += step;
    }
    
    if (!reachedBoth) {
        for(var i = 2; i < FXAA_SEARCH_STEPS; i++) {
            if (!reached1) {
                lumaEnd1 = rgb2luma(textureSample(t_input, s_linear, uv1).rgb);
                lumaEnd1 = lumaEnd1 - lumaLocalAverage;
            }
            if (!reached2) {
                lumaEnd2 = rgb2luma(textureSample(t_input, s_linear, uv2).rgb);
                lumaEnd2 = lumaEnd2 - lumaLocalAverage;
            }
            
            reached1 = abs(lumaEnd1) >= gradientScaled;
            reached2 = abs(lumaEnd2) >= gradientScaled;
            reachedBoth = reached1 && reached2;
            
            if (reachedBoth) { break; }
            
            if (!reached1) {
                uv1 -= step;
            }
            if (!reached2) {
                uv2 += step;
            }
        }
    }
    
    // 5. Estimate Offset
    var distance1 = 0.0;
    var distance2 = 0.0;
    if (isHorz) {
        distance1 = in.uv.x - uv1.x;
        distance2 = uv2.x - in.uv.x;
    } else {
        distance1 = in.uv.y - uv1.y;
        distance2 = uv2.y - in.uv.y;
    }
    
    let isDirection1 = distance1 < distance2;
    let distanceFinal = min(distance1, distance2);
    let edgeThickness = (distance1 + distance2);
    let pixelOffset = -distanceFinal / edgeThickness + 0.5;
    
    
    // Sub-pixel Blend (Lowpass)
    let lumaL = (lumaN + lumaS + lumaE + lumaW) * 0.25;
    let rangeL = abs(lumaL - lumaM);
    let blendL = max(0.0, (rangeL / contrast) - FXAA_SUBPIX_TRIM) * FXAA_SUBPIX_TRIM_SCALE;
    let blendL_clamped = min(FXAA_SUBPIX_CAP, blendL);
    
    // Combine Edge and Sub-pixel offsets
    let finalOffset = max(pixelOffset, blendL_clamped);
    
    // Final UV
    var finalUv = in.uv;
    if (isHorz) {
        finalUv.y += finalOffset * stepLength;
    } else {
        finalUv.x += finalOffset * stepLength;
    }
    
    // Read Final
    return textureSample(t_input, s_linear, finalUv);
}
