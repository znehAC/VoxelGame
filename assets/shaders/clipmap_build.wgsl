// Clipmap cascade build — fills 3D texture via terrain height function.
//
// Two dispatch modes:
//   mode 3 (full): dispatch(64, 64, 512) × workgroup(8,8,1) → 512×512×512 cells
//     gid.x=X, gid.y=Z, gid.z=Y (all in local coords relative to cascade origin)
//   mode 0/1/2 (face): dispatch(64, 64, 1) × workgroup(8,8,1) → 512×512 face cells
//     axis 0 (X): gid.x=Y_local, gid.y=Z_local, face_val=global_X
//     axis 1 (Y): gid.x=X_local, gid.y=Z_local, face_val=global_Y
//     axis 2 (Z): gid.x=X_local, gid.y=Y_local, face_val=global_Z

const PI2: f32 = 6.28318530718;

struct BuildParams {
    origin_x: i32,
    origin_y: i32,
    origin_z: i32,
    cell_size: i32,
    mode:      u32,   // 3=full build, 0/1/2=face update axis
    face_val:  i32,   // global cell coord along update axis (face mode only)
    _pad0:     u32,
    _pad1:     u32,
}

var<push_constant> params: BuildParams;

@group(0) @binding(0) var cascade_tex: texture_storage_3d<rgba8uint, write>;

fn height_at(wx: i32, wz: i32) -> i32 {
    let x = f32(wx);
    let z = f32(wz);
    let h = 100.0
        + sin(x / 3200.0 * PI2) * cos(z / 3200.0 * PI2) * 60.0
        + sin(x / 800.0  * PI2) * sin(z / 800.0  * PI2) * 20.0
        + sin(x / 160.0  * PI2) * sin(z / 160.0  * PI2) *  5.0;
    return i32(h);
}

// wy = cell center in voxels, cs = cell size in voxels, h = terrain height
fn cell_material(wy: i32, cs: i32, h: i32) -> u32 {
    if wy > h          { return 0u; }  // air
    if wy > h - cs     { return 1u; }  // grass: surface cell
    if wy > h - cs * 2 { return 2u; }  // dirt: one cell below
    return 3u;                         // stone: deep
}

fn dominant_normal_idx(n: vec3f) -> u32 {
    let a = abs(n);
    if a.x >= a.y && a.x >= a.z { return select(0u, 1u, n.x < 0.0); }
    if a.y >= a.z                { return select(2u, 3u, n.y < 0.0); }
    return select(4u, 5u, n.z < 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn build_clipmap(@builtin(global_invocation_id) gid: vec3u) {
    let origin = vec3i(params.origin_x, params.origin_y, params.origin_z);
    var global_cell: vec3i;

    switch params.mode {
        case 3u: {
            // Full build: gid=(X_local, Z_local, Y_local) offset from origin center
            let local = vec3i(i32(gid.x) - 256, i32(gid.z) - 256, i32(gid.y) - 256);
            global_cell = origin + local;
        }
        case 0u: {
            // X face: gid.x=Y_local, gid.y=Z_local
            global_cell = vec3i(
                params.face_val,
                origin.y + i32(gid.x) - 256,
                origin.z + i32(gid.y) - 256,
            );
        }
        case 1u: {
            // Y face: gid.x=X_local, gid.y=Z_local
            global_cell = vec3i(
                origin.x + i32(gid.x) - 256,
                params.face_val,
                origin.z + i32(gid.y) - 256,
            );
        }
        default: {
            // Z face: gid.x=X_local, gid.y=Y_local
            global_cell = vec3i(
                origin.x + i32(gid.x) - 256,
                origin.y + i32(gid.y) - 256,
                params.face_val,
            );
        }
    }

    // Guard: full build has a 3rd dispatch dimension that must stay in bounds
    if params.mode == 3u && gid.z >= 512u { return; }

    let cs = params.cell_size;
    let world = global_cell * cs + cs / 2;
    let h = height_at(world.x, world.z);
    let mat = cell_material(world.y, cs, h);

    var normal_idx = 2u; // default +Y
    if mat != 0u {
        let dh_x = height_at(world.x + cs, world.z) - height_at(world.x - cs, world.z);
        let dh_z = height_at(world.x, world.z + cs) - height_at(world.x, world.z - cs);
        let n = normalize(vec3f(-f32(dh_x), f32(cs * 2), -f32(dh_z)));
        normal_idx = dominant_normal_idx(n);
    }

    let density = select(0u, 255u, mat != 0u);
    let tor = vec3i(((global_cell % 512) + 512) % 512);

    textureStore(cascade_tex, tor, vec4u(mat, normal_idx, density, 0u));
}
