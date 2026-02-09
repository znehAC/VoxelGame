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
/// Returns the first non-air voxel hit, or `None` if the ray exits the grid.
pub fn dda_raycast(
    voxels: &[PackedVoxel],
    origin: Vec3,
    dir: Vec3,
    max_dist: f32,
) -> Option<RayHit> {
    let gs = GRID_SIZE as f32;
    let inv_dir = 1.0 / dir;

    // AABB intersection with [0, GRID_SIZE]^3
    let t0 = -origin * inv_dir;
    let t1 = (Vec3::splat(gs) - origin) * inv_dir;
    let tmin = t0.min(t1);
    let tmax = t0.max(t1);
    let t_enter = tmin.x.max(tmin.y).max(tmin.z);
    let t_exit = tmax.x.min(tmax.y).min(tmax.z);

    if t_enter > t_exit || t_exit < 0.0 || t_enter > max_dist {
        return None;
    }

    let t_enter = t_enter.max(0.0);
    let entry = origin + dir * (t_enter + 0.001);

    // DDA setup
    let gi = GRID_SIZE as i32;
    let mut cell = IVec3::new(
        (entry.x.floor() as i32).clamp(0, gi - 1),
        (entry.y.floor() as i32).clamp(0, gi - 1),
        (entry.z.floor() as i32).clamp(0, gi - 1),
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

    let mut t_max = Vec3::new(
        if dir.x > 0.0 {
            (cell.x as f32 + 1.0 - entry.x) * inv_dir.x.abs()
        } else {
            (entry.x - cell.x as f32) * inv_dir.x.abs()
        },
        if dir.y > 0.0 {
            (cell.y as f32 + 1.0 - entry.y) * inv_dir.y.abs()
        } else {
            (entry.y - cell.y as f32) * inv_dir.y.abs()
        },
        if dir.z > 0.0 {
            (cell.z as f32 + 1.0 - entry.z) * inv_dir.z.abs()
        } else {
            (entry.z - cell.z as f32) * inv_dir.z.abs()
        },
    );

    // Check starting cell
    let packed = voxels[voxel_index(cell.x, cell.y, cell.z)];
    if !packed.is_air() {
        // Entry normal from AABB face
        let normal = if t_enter > 0.0 {
            if tmin.x >= tmin.y && tmin.x >= tmin.z {
                IVec3::new(-step.x, 0, 0)
            } else if tmin.y >= tmin.x && tmin.y >= tmin.z {
                IVec3::new(0, -step.y, 0)
            } else {
                IVec3::new(0, 0, -step.z)
            }
        } else {
            // Camera inside solid voxel
            let a = dir.abs();
            if a.x >= a.y && a.x >= a.z {
                IVec3::new(-step.x, 0, 0)
            } else if a.y >= a.x && a.y >= a.z {
                IVec3::new(0, -step.y, 0)
            } else {
                IVec3::new(0, 0, -step.z)
            }
        };

        return Some(RayHit {
            index: voxel_index(cell.x, cell.y, cell.z),
            grid_pos: cell,
            normal,
        });
    }

    for _ in 0..MAX_STEPS {
        // Step along axis with smallest t_max
        let last_axis;
        if t_max.x < t_max.y {
            if t_max.x < t_max.z {
                cell.x += step.x;
                t_max.x += t_delta.x;
                last_axis = 0u32;
            } else {
                cell.z += step.z;
                t_max.z += t_delta.z;
                last_axis = 2;
            }
        } else if t_max.y < t_max.z {
            cell.y += step.y;
            t_max.y += t_delta.y;
            last_axis = 1;
        } else {
            cell.z += step.z;
            t_max.z += t_delta.z;
            last_axis = 2;
        }

        // Bounds check
        if cell.x < 0 || cell.x >= gi || cell.y < 0 || cell.y >= gi || cell.z < 0 || cell.z >= gi
        {
            break;
        }

        // Distance check: t from entry for the crossed boundary
        let t_crossed = match last_axis {
            0 => t_max.x - t_delta.x,
            1 => t_max.y - t_delta.y,
            _ => t_max.z - t_delta.z,
        };
        if t_enter + t_crossed > max_dist {
            break;
        }

        let packed = voxels[voxel_index(cell.x, cell.y, cell.z)];
        if !packed.is_air() {
            let normal = match last_axis {
                0 => IVec3::new(-step.x, 0, 0),
                1 => IVec3::new(0, -step.y, 0),
                _ => IVec3::new(0, 0, -step.z),
            };

            return Some(RayHit {
                index: voxel_index(cell.x, cell.y, cell.z),
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
    fn hit_single_voxel() {
        let mut voxels = empty_grid();
        // Place a stone block at (32, 32, 32)
        let idx = voxel_index(32, 32, 32);
        voxels[idx] = PackedVoxel::new(1);

        // Cast from (32.5, 32.5, 0.5) looking +Z
        let origin = Vec3::new(32.5, 32.5, 0.5);
        let dir = Vec3::new(0.0, 0.0, 1.0);
        let hit = dda_raycast(&voxels, origin, dir, 64.0).expect("should hit");

        assert_eq!(hit.grid_pos, IVec3::new(32, 32, 32));
        assert_eq!(hit.normal, IVec3::new(0, 0, -1)); // Hit -Z face
        assert_eq!(hit.index, idx);
    }

    #[test]
    fn miss_empty_grid() {
        let voxels = empty_grid();
        let origin = Vec3::new(32.0, 32.0, -1.0);
        let dir = Vec3::new(0.0, 0.0, 1.0);
        assert!(dda_raycast(&voxels, origin, dir, 128.0).is_none());
    }

    #[test]
    fn max_dist_limits_ray() {
        let mut voxels = empty_grid();
        voxels[voxel_index(32, 32, 50)] = PackedVoxel::new(1);

        // Ray origin at z=0, block at z=50 — max_dist=10 should miss
        let origin = Vec3::new(32.5, 32.5, 0.5);
        let dir = Vec3::new(0.0, 0.0, 1.0);
        assert!(dda_raycast(&voxels, origin, dir, 10.0).is_none());
    }

    #[test]
    fn normal_axes() {
        let mut voxels = empty_grid();
        voxels[voxel_index(32, 32, 32)] = PackedVoxel::new(1);

        // Hit from -X direction
        let hit = dda_raycast(
            &voxels,
            Vec3::new(0.5, 32.5, 32.5),
            Vec3::new(1.0, 0.0, 0.0),
            64.0,
        )
        .expect("should hit from -X");
        assert_eq!(hit.normal, IVec3::new(-1, 0, 0));

        // Hit from +Y direction
        let hit = dda_raycast(
            &voxels,
            Vec3::new(32.5, 63.5, 32.5),
            Vec3::new(0.0, -1.0, 0.0),
            64.0,
        )
        .expect("should hit from +Y");
        assert_eq!(hit.normal, IVec3::new(0, 1, 0));
    }
}
