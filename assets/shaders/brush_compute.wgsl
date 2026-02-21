const GRID_SIZE: u32 = 512u;
const CHUNK_SIZE: u32 = 32u;
const CHUNKS_PER_AXIS: u32 = 16u;

struct BrushParams {
    center_x: i32,
    center_y: i32,
    center_z: i32,
    radius: f32,
    material: u32,
    shape: u32, // 0 = Cube/Block, 1 = Sphere
    pad1: u32,
    pad2: u32,
}

var<push_constant> params: BrushParams;

@group(0) @binding(0) var<storage, read_write> voxels: array<u32>;
@group(0) @binding(1) var<storage, read_write> occupancy: array<u32>;

fn wrap(c: i32) -> i32 {
    let size = i32(GRID_SIZE);
    return ((c % size) + size) % size;
}

fn voxel_index(x: i32, y: i32, z: i32) -> u32 {
    return u32(wrap(z)) * GRID_SIZE * GRID_SIZE + u32(wrap(y)) * GRID_SIZE + u32(wrap(x));
}

fn chunk_wrap(c: i32) -> i32 {
    let size = i32(CHUNKS_PER_AXIS);
    return ((c % size) + size) % size;
}

fn chunk_index(cx: i32, cy: i32, cz: i32) -> u32 {
    return u32(chunk_wrap(cz)) * CHUNKS_PER_AXIS * CHUNKS_PER_AXIS
         + u32(chunk_wrap(cy)) * CHUNKS_PER_AXIS
         + u32(chunk_wrap(cx));
}

@compute @workgroup_size(4, 4, 4)
fn apply_brush(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Thread grid is sized exactly to the bounding box: 
    // width = (radius * 2 + 1)
    
    // Convert gid to a relative offset in the box 
    // Using signed integer logic because the radius could be 0 (giving size 1)
    let r_int = i32(params.radius);
    if params.shape == 0u {
        // Shape Block uses floor((brush_size - 1) / 2) as logic
        // radius here is pre-calculated from (brush_size - 1) / 2
    }
    
    let offset_x = i32(gid.x) - r_int;
    let offset_y = i32(gid.y) - r_int;
    let offset_z = i32(gid.z) - r_int;

    // Bounds check to stop trailing threads when workgroup size exceeds brush size
    let width = r_int * 2 + 1;
    if i32(gid.x) >= width || i32(gid.y) >= width || i32(gid.z) >= width {
        return;
    }

    if params.shape == 1u { // Sphere
        let dist_sq = f32(offset_x * offset_x + offset_y * offset_y + offset_z * offset_z);
        if dist_sq > params.radius * params.radius {
            return;
        }
    }

    let world_pos = vec3i(params.center_x + offset_x, params.center_y + offset_y, params.center_z + offset_z);
    
    let idx = voxel_index(world_pos.x, world_pos.y, world_pos.z);
    voxels[idx] = params.material;
    
    if params.material != 0u {
        let cs = i32(CHUNK_SIZE);
        // Using euclidean division to mimic CPU behavior
        let cx = i32(floor(f32(world_pos.x) / f32(cs)));
        let cy = i32(floor(f32(world_pos.y) / f32(cs)));
        let cz = i32(floor(f32(world_pos.z) / f32(cs)));
        let ci = chunk_index(cx, cy, cz);
        occupancy[ci] = 1u;
    }
}
