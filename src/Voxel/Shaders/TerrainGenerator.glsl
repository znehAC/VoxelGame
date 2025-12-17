#[compute]
#version 450

// Invocations in the (x, y, z) dimension
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

// A binding to the buffer we create in our script
layout(set = 0, binding = 0, std430) restrict writeonly buffer VoxelBuffer {
    uint voxels[];
};

// Metadata buffer
layout(set = 0, binding = 1, std430) restrict buffer Metadata {
    uint non_empty_flag;
};

// Parameters Buffer (Binding 2)
layout(set = 0, binding = 2, std430) restrict readonly buffer ParamsBuffer {
    int cx;
    int cy;
    int cz;
    int size;
    int seed;
    int sea_level;
    int mountain_level;
    int scale;
    int lod;
} params;

// --- Noise Functions ---

// Hash function
float hash(vec3 p) {
    p = fract(p * 0.3183099 + .1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    
    return mix(mix(mix( hash(i + vec3(0,0,0)), 
                        hash(i + vec3(1,0,0)),f.x),
                   mix( hash(i + vec3(0,1,0)), 
                        hash(i + vec3(1,1,0)),f.x),f.y),
               mix(mix( hash(i + vec3(0,0,1)), 
                        hash(i + vec3(1,0,1)),f.x),
                   mix( hash(i + vec3(0,1,1)), 
                        hash(i + vec3(1,1,1)),f.x),f.y),f.z);
}

float fbm(vec3 x) {
    float v = 0.0;
    float a = 0.5;
    vec3 shift = vec3(100.0);
    for (int i = 0; i < 4; ++i) {
        v += a * noise(x);
        x = x * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

void main() {
    ivec3 id = ivec3(gl_GlobalInvocationID.xyz);
    
    // Bounds check
    if (id.x >= params.size || id.y >= params.size || id.z >= params.size) {
        return;
    }
    
    // Reconstruct position
    ivec3 chunk_pos = ivec3(params.cx, params.cy, params.cz);
    
    // Global Voxel Coordinate (including LOD stride via params.scale)
    ivec3 global_pos = (chunk_pos * params.size + id) * params.scale;
    
    // --- Single Voxel Logic (No quantization) ---
    vec3 pos = vec3(global_pos);
    
    uint voxelType = 0; // Air

    // Noise Generation
    // Sample noise at standard frequency
    float height = fbm(vec3(pos.x * 0.005, 0.0, pos.z * 0.005)) * 128.0 + 32.0;
    
    if (float(global_pos.y) < height) {
        // LOD Debug Coloring
        if (params.lod == 0) voxelType = 1;      // Dirt (LOD 0)
        else if (params.lod == 1) voxelType = 2; // Stone (LOD 1)
        else if (params.lod == 2) voxelType = 3; // Grass (LOD 2)
        else if (params.lod == 3) voxelType = 6; // Sand (LOD 3)
        else if (params.lod == 4) voxelType = 5; // Snow (LOD 4)
        else voxelType = 9;                      // BlueStone (LOD 5+)
        
        // Bedrock override
        if (global_pos.y < -50) {
             voxelType = 2; // Stone
        }
    } else {
        voxelType = 0; // Air
    }

    // Write to buffer
    int index = id.z * params.size * params.size + id.y * params.size + id.x;
    voxels[index] = voxelType;

    // Optimization: Flag if non-empty
    if (voxelType != 0) {
        atomicOr(non_empty_flag, 1);
    }
}
