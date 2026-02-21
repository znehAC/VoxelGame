//! CPU-side DDA raycast against a flat voxel grid.

use glam::{IVec3, Vec3};

use crate::{PackedVoxel, GRID_SIZE, MAX_STEPS};

/// Result of a successful raycast hit.
pub struct RayHit {
    /// Linear index into the voxel array.
    pub index: usize,
    /// Integer grid position of the hit voxel.
    pub grid_pos: IVec3,
    /// Face normal of the hit surface (-1, 0, or 1 per axis).
    pub normal: IVec3,
}

/// Compute the linear index for grid coordinates (z * S*S + y * S + x).
#[inline]
fn voxel_index(x: i32, y: i32, z: i32) -> usize {
    let s = GRID_SIZE as i32;
    (z * s * s + y * s + x) as usize
}

/// Cast a ray through the voxel grid using DDA traversal.
///
/// Returns the first non-air voxel hit, or `None` if the ray exceeds `max_dist`.
/// Wraps coordinates modulo `GRID_SIZE` for toroidal topology.
pub fn dda_raycast(
    voxels: &[PackedVoxel],
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
) -> Option<RayHit> {
    let inv_dir = 1.0 / dir;

    // DDA setup
    // Use floor() to get the integer cell coordinate.
    // Note: We do NOT clamp to [0, GRID_SIZE-1] because the world is infinite/toroidal.
    let mut cell = IVec3::new(
        origin.x.floor() as i32,
        origin.y.floor() as i32,
        origin.z.floor() as i32,
    );

    let step = IVec3::new(
        if dir.x > 0.0 { 1 } else { -1 },
        if dir.y > 0.0 { 1 } else { -1 },
        if dir.z > 0.0 { 1 } else { -1 },
    );

    let t_delta = Vec3::new(
        (1.0 / dir.x).abs(),
        (1.0 / dir.y).abs(),
        (1.0 / dir.z).abs(),
    );

    // t_max: distance to the next voxel boundary on each axis
    // For negative direction, we measure distance to the current cell wall (cell.x).
    // For positive direction, we measure distance to the next cell wall (cell.x + 1.0).
    // We use (cell as f32) directly, which works for any integer coordinate.
    let mut t_max = Vec3::new(
        if dir.x > 0.0 {
            (cell.x as f32 + 1.0 - origin.x) * inv_dir.x.abs()
        } else {
            (origin.x - cell.x as f32) * inv_dir.x.abs()
        },
        if dir.y > 0.0 {
            (cell.y as f32 + 1.0 - origin.y) * inv_dir.y.abs()
        } else {
            (origin.y - cell.y as f32) * inv_dir.y.abs()
        },
        if dir.z > 0.0 {
            (cell.z as f32 + 1.0 - origin.z) * inv_dir.z.abs()
        } else {
            (origin.z - cell.z as f32) * inv_dir.z.abs()
        },
    );

    // Helper for wrapped lookup
    let gs = GRID_SIZE as i32;
    let get_voxel = |c: IVec3| -> PackedVoxel {
        let x = ((c.x % gs) + gs) % gs;
        let y = ((c.y % gs) + gs) % gs;
        let z = ((c.z % gs) + gs) % gs;
        voxels[voxel_index(x, y, z)]
    };

    // Check starting cell
    let packed = get_voxel(cell);
    if !packed.is_air() {
        // Since we start inside a solid voxel, the normal is undefined or pointing back at the ray origin.
        // A reasonable fallback is to return the normal of the face we *would* have entered.
        // Or simply -direction.
        let a = dir.abs();
        let normal = if a.x >= a.y && a.x >= a.z {
            IVec3::new(-step.x, 0, 0)
        } else if a.y >= a.x && a.y >= a.z {
            IVec3::new(0, -step.y, 0)
        } else {
            IVec3::new(0, 0, -step.z)
        };

        // For the index, we must return the wrapped index, but the grid_pos is the "unwrapped" world coordinate.
        let x = ((cell.x % gs) + gs) % gs;
        let y = ((cell.y % gs) + gs) % gs;
        let z = ((cell.z % gs) + gs) % gs;

        return Some(RayHit {
            index: voxel_index(x, y, z),
            grid_pos: cell,
            normal,
        });
    }

    let mut t_curr = 0.0;
    for _ in 0..MAX_STEPS {
        // Step along axis with smallest t_max
        let last_axis;
        if t_max.x < t_max.y {
            if t_max.x < t_max.z {
                cell.x += step.x;
                t_curr = t_max.x;
                t_max.x += t_delta.x;
                last_axis = 0u32;
            } else {
                cell.z += step.z;
                t_curr = t_max.z;
                t_max.z += t_delta.z;
                last_axis = 2;
            }
        } else if t_max.y < t_max.z {
            cell.y += step.y;
            t_curr = t_max.y;
            t_max.y += t_delta.y;
            last_axis = 1;
        } else {
            cell.z += step.z;
            t_curr = t_max.z;
            t_max.z += t_delta.z;
            last_axis = 2;
        }

        if t_curr > max_dist {
            break;
        }

        let packed = get_voxel(cell);
        if !packed.is_air() {
            let normal = match last_axis {
                0 => IVec3::new(-step.x, 0, 0),
                1 => IVec3::new(0, -step.y, 0),
                _ => IVec3::new(0, 0, -step.z),
            };

            let x = ((cell.x % gs) + gs) % gs;
            let y = ((cell.y % gs) + gs) % gs;
            let z = ((cell.z % gs) + gs) % gs;

            return Some(RayHit {
                index: voxel_index(x, y, z),
                grid_pos: cell,
                normal,
            });
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_grid() -> Vec<PackedVoxel> {
        vec![PackedVoxel::AIR; (GRID_SIZE * GRID_SIZE * GRID_SIZE) as usize]
    }

    #[test]
    #[ignore] // 512³ grid allocation (~512 MB)
    fn hit_single_voxel() {
        let mut voxels = empty_grid();
        let idx = voxel_index(256, 256, 256);
        voxels[idx] = PackedVoxel::new(1);

        let origin = Vec3::new(256.5, 256.5, 0.5);
        let dir = Vec3::new(0.0, 0.0, 1.0);
        let hit = dda_raycast(&voxels, origin, dir, 512.0).expect("should hit");

        assert_eq!(hit.grid_pos, IVec3::new(256, 256, 256));
        assert_eq!(hit.normal, IVec3::new(0, 0, -1));
        assert_eq!(hit.index, idx);
    }

    #[test]
    #[ignore] // 512³ grid allocation (~512 MB)
    fn miss_empty_grid() {
        let voxels = empty_grid();
        let origin = Vec3::new(256.0, 256.0, -1.0);
        let dir = Vec3::new(0.0, 0.0, 1.0);
        assert!(dda_raycast(&voxels, origin, dir, 1024.0).is_none());
    }

    #[test]
    #[ignore] // 512³ grid allocation (~512 MB)
    fn max_dist_limits_ray() {
        let mut voxels = empty_grid();
        voxels[voxel_index(256, 256, 300)] = PackedVoxel::new(1);

        let origin = Vec3::new(256.5, 256.5, 0.5);
        let dir = Vec3::new(0.0, 0.0, 1.0);
        assert!(dda_raycast(&voxels, origin, dir, 10.0).is_none());
    }

    #[test]
    #[ignore] // 512³ grid allocation (~512 MB)
    fn normal_axes() {
        let mut voxels = empty_grid();
        voxels[voxel_index(256, 256, 256)] = PackedVoxel::new(1);

        let hit = dda_raycast(
            &voxels,
            Vec3::new(0.5, 256.5, 256.5),
            Vec3::new(1.0, 0.0, 0.0),
            512.0,
        )
        .expect("should hit from -X");
        assert_eq!(hit.normal, IVec3::new(-1, 0, 0));

        let hit = dda_raycast(
            &voxels,
            Vec3::new(256.5, 511.5, 256.5),
            Vec3::new(0.0, -1.0, 0.0),
            512.0,
        )
        .expect("should hit from +Y");
        assert_eq!(hit.normal, IVec3::new(0, 1, 0));
    }
}
