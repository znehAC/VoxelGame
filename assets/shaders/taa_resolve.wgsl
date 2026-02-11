// TAA Resolve Shader
// Reprojects history and blends with current frame using neighborhood clamping.
// Uses reversible tonemapping to preserve HDR energy during blending.
// Includes robust defensive programming to handle NaNs and Infinities.

struct TaaUniforms {
    // (1.0 / width, 1.0 / height, width, height)
    screen_size: vec4f,
    // (blend_alpha, enable_sharpening, debug_mode, has_valid_history)
    params: vec4f,
    // (use_variance_clamp, use_ycocg, 0, 0)
    flags: vec4f,
    // Padding to ensure 16-byte alignment for array stride
    _padding: vec4f,
}

@group(0) @binding(0) var t_current: texture_2d<f32>;
@group(0) @binding(1) var t_history: texture_2d<f32>;
@group(0) @binding(2) var t_velocity: texture_2d<f32>;
@group(0) @binding(3) var s_linear: sampler;
@group(0) @binding(4) var s_point: sampler;
@group(0) @binding(5) var<uniform> uniforms: TaaUniforms;

// --- Safety Helpers ---

// Check if a float is finite (not NaN and not Inf)
fn is_finite(v: f32) -> bool {
    // varied checks to be robust across drivers
    return v == v && v > -1.0e38 && v < 1.0e38;
}

// Sanitize a color vector: replace NaN/Inf/Negatives with 0.0
fn sanitize(color: vec3f) -> vec3f {
    var c = color;
    if (!is_finite(c.x)) { c.x = 0.0; }
    if (!is_finite(c.y)) { c.y = 0.0; }
    if (!is_finite(c.z)) { c.z = 0.0; }
    return max(c, vec3f(0.0));
}

// --- Tonemapping & Color Space ---

// Simple Reinhard tonemapper: compresses HDR to [0, 1] range
fn reinhard_tonemap(color: vec3f) -> vec3f {
    // Sanitize input to prevent NaN propagation
    let safe_color = sanitize(color);
    let luma = dot(safe_color, vec3f(0.2126, 0.7152, 0.0722));
    let tonemapped_luma = luma / (1.0 + luma);
    if (luma > 0.0001) {
        return safe_color * (tonemapped_luma / luma);
    }
    return safe_color;
}

// Inverse Reinhard tonemapper: expands back to HDR linear space
fn reinhard_inverse(tonemapped: vec3f) -> vec3f {
    let safe_tm = sanitize(tonemapped);
    var luma = dot(safe_tm, vec3f(0.2126, 0.7152, 0.0722));
    
    // Clamp max luma to prevent singularity at 1.0
    // 0.99 gives max brightness of ~100.0, which is plenty for our scene
    luma = clamp(luma, 0.0, 0.99);
    
    if (luma <= 0.0001) {
        return vec3f(0.0);
    }
    
    let original_luma = luma / (1.0 - luma);
    return safe_tm * (original_luma / luma);
}

// Convert RGB to YCoCg
fn rgb_to_ycocg(rgb: vec3f) -> vec3f {
    let safe_rgb = sanitize(rgb);
    return vec3f(
        dot(safe_rgb, vec3f(0.25, 0.5, 0.25)),
        dot(safe_rgb, vec3f(0.5, 0.0, -0.5)),
        dot(safe_rgb, vec3f(-0.25, 0.5, -0.25))
    );
}

// Convert YCoCg back to RGB
fn ycocg_to_rgb(ycocg: vec3f) -> vec3f {
    // YCoCg sub-components can be negative, so we only sanitize NaNs/Infs, not signs
    var y = ycocg;
    if (!is_finite(y.x)) { y.x = 0.0; }
    if (!is_finite(y.y)) { y.y = 0.0; }
    if (!is_finite(y.z)) { y.z = 0.0; }
    
    return vec3f(
        y.x + y.y - y.z,
        y.x + y.z,
        y.x - y.y - y.z
    );
}

// --- Sampling Functions ---

fn sample_current(uv: vec2f) -> vec3f {
    let c = textureSample(t_current, s_linear, uv).rgb;
    return sanitize(c);
}

fn sample_history(uv: vec2f) -> vec3f {
    let c = textureSample(t_history, s_linear, uv).rgb;
    return sanitize(c);
}

fn sample_velocity(uv: vec2f) -> vec2f {
    let v = textureSample(t_velocity, s_point, uv).rg;
    if (!is_finite(v.x) || !is_finite(v.y)) {
        return vec2f(0.0);
    }
    return v;
}

fn sample_current_point(uv: vec2f) -> vec3f {
    let c = textureSample(t_current, s_point, uv).rgb;
    return sanitize(c);
}

fn sample_history_point(uv: vec2f) -> vec3f {
    let c = textureSample(t_history, s_point, uv).rgb;
    return sanitize(c);
}

// --- Neighborhood & Clipping ---

struct Neighborhood {
    min_val: vec3f,
    max_val: vec3f,
    avg: vec3f,
}

fn get_neighborhood(uv: vec2f, texel_size: vec2f, use_ycocg: bool) -> Neighborhood {
    // Initialize with safe values
    var min_v = vec3f(1000.0);
    var max_v = vec3f(-1000.0);
    var sum = vec3f(0.0);
    
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let offset = vec2f(f32(x), f32(y)) * texel_size;
            let hdr = sample_current_point(uv + offset);
            let tm = reinhard_tonemap(hdr);
            
            var val = tm;
            if (use_ycocg) {
                val = rgb_to_ycocg(tm);
            }
            
            min_v = min(min_v, val);
            max_v = max(max_v, val);
            sum = sum + val;
        }
    }
    
    return Neighborhood(min_v, max_v, sum / 9.0);
}

// Reprojected history sampling with bounds checking
// Returns vec4(rgb, valid_flag) where valid_flag is 1.0 (valid) or 0.0 (invalid)
fn sample_reprojected_history(uv: vec2f, velocity: vec2f) -> vec4f {
    let history_uv = uv - velocity;
    
    // Bounds check
    let epsilon = 0.0001; // Small epsilon
    if (history_uv.x < epsilon || history_uv.x > 1.0 - epsilon || 
        history_uv.y < epsilon || history_uv.y > 1.0 - epsilon) {
        return vec4f(0.0, 0.0, 0.0, 0.0); // Invalid
    }
    
    // Sample history (using point to minimize blurring from resampling)
    let c = textureSample(t_history, s_linear, history_uv).rgb;
    return vec4f(sanitize(c), 1.0);
}

// Simple sharpening filter
fn sharpen_color(center: vec3f, uv: vec2f, texel_size: vec2f) -> vec3f {
    let amount = 0.25;
    var sum = vec3f(0.0);
    sum += sample_current(uv + vec2f(-texel_size.x, 0.0));
    sum += sample_current(uv + vec2f( texel_size.x, 0.0));
    sum += sample_current(uv + vec2f(0.0, -texel_size.y));
    sum += sample_current(uv + vec2f(0.0,  texel_size.y));
    return sanitize(center + (center - sum * 0.25) * amount);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
    let x = f32(i32(vertex_index) / 2) * 4.0 - 1.0;
    let y = f32(i32(vertex_index) % 2) * 4.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4f) -> @location(0) vec4f {
    let texel_size = uniforms.screen_size.xy;
    let uv = frag_coord.xy * texel_size;
    
    let blend_alpha = uniforms.params.x;
    let enable_sharpening = uniforms.params.y > 0.5;
    let debug_mode = i32(uniforms.params.z);
    let has_valid_history = uniforms.params.w > 0.5;
    let use_variance_clamp = uniforms.flags.x > 0.5;
    let use_ycocg = uniforms.flags.y > 0.5;
    
    // 1. Current Frame Input
    let current_hdr = sample_current(uv);
    let current_tm = reinhard_tonemap(current_hdr);
    
    // 2. Velocity Input
    let velocity = sample_velocity(uv);
    
    // 3. History Input (Reprojected)
    let history_sample = sample_reprojected_history(uv, velocity);
    let history_hdr = history_sample.rgb;
    let history_valid_spatial = history_sample.a > 0.5;
    
    // 4. Resolve Logic
    var result_hdr: vec3f;
    
    // If we have history (both temporally and spatially valid)
    if (has_valid_history && history_valid_spatial) {
        let history_tm = reinhard_tonemap(history_hdr);
        var history_src = history_tm;
        
        // Variance Clipping / Clamping
        if (use_variance_clamp) {
            let n = get_neighborhood(uv, texel_size, use_ycocg);
            
            if (use_ycocg) {
                let history_ycocg = rgb_to_ycocg(history_tm);
                let clipped_ycocg = clamp(history_ycocg, n.min_val, n.max_val);
                // Safe conversion back to RGB (handles potential negative chroma)
                history_src = sanitize(ycocg_to_rgb(clipped_ycocg)); 
            } else {
                history_src = clamp(history_tm, n.min_val, n.max_val);
            }
        }
        
        // Blend in Tonemapped Space
        // Use max() to ensure no negative values creep in
        let blended_tm = mix(history_src, current_tm, blend_alpha);
        
        // Inverse Tonemap
        result_hdr = reinhard_inverse(blended_tm);
        
        // Optional Sharpening
        if (enable_sharpening) {
            result_hdr = sharpen_color(result_hdr, uv, texel_size);
        }
        
    } else {
        // No history available, pass through current
        result_hdr = current_hdr;
    }
    
    // 5. Debug Overlays
    switch debug_mode {
        case 1: { // Velocity
            let v = velocity * 100.0 + 0.5; // Scale up for visibility
            result_hdr = vec3f(v.x, v.y, 0.0);
        }
        case 2: { // Neighborhood Min (RGB or YCoCg->RGB)
            let n = get_neighborhood(uv, texel_size, use_ycocg);
            if (use_ycocg) { result_hdr = reinhard_inverse(ycocg_to_rgb(n.min_val)); }
            else { result_hdr = reinhard_inverse(n.min_val); }
        }
        case 3: { // Neighborhood Max
            let n = get_neighborhood(uv, texel_size, use_ycocg);
            if (use_ycocg) { result_hdr = reinhard_inverse(ycocg_to_rgb(n.max_val)); }
            else { result_hdr = reinhard_inverse(n.max_val); }
        }
        case 4: { // Raw History Reprojected
             if (history_valid_spatial) { result_hdr = history_hdr; }
             else { result_hdr = vec3f(1.0, 0.0, 0.0); }
        }
        case 5: { // Clipped History
            if (history_valid_spatial) {
                let n = get_neighborhood(uv, texel_size, use_ycocg);
                let h_tm = reinhard_tonemap(history_hdr);
                var h_clipped = h_tm;
                if (use_ycocg) {
                   h_clipped = ycocg_to_rgb(clamp(rgb_to_ycocg(h_tm), n.min_val, n.max_val));
                } else {
                   h_clipped = clamp(h_tm, n.min_val, n.max_val);
                }
                result_hdr = reinhard_inverse(h_clipped);
            } else {
                result_hdr = vec3f(1.0, 0.0, 0.0);
            }
        }
        case 6: { // Current Frame only
            result_hdr = current_hdr;
        }
        case 7: { // Validity: Green=OK, Red=SpatialInvalid, Blue=TemporalInvalid
            if (has_valid_history && history_valid_spatial) { result_hdr = vec3f(0.0, 1.0, 0.0); }
            else if (has_valid_history) { result_hdr = vec3f(1.0, 0.0, 0.0); }
            else { result_hdr = vec3f(0.0, 0.0, 1.0); }
        }
        default: {}
    }
    
    // Final Sanitize
    return vec4f(sanitize(result_hdr), 1.0);
}
