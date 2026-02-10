//! GPU-shared data types (std430 layout).
//!
//! These types are designed for direct GPU upload with proper alignment.
//! All types use Vec4 instead of Vec3 to avoid padding issues.

use bytemuck::{Pod, Zeroable};
use glam::{UVec4, Vec4};

/// Global uniforms for raymarching.
///
/// std140 layout for uniform buffers (16-byte alignment for vec4, mat4).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GlobalUniforms {
    /// Inverse view matrix (Camera -> World), column-major.
    pub view_inverse: [Vec4; 4],
    /// Inverse projection matrix (Screen -> Camera), column-major.
    pub proj_inverse: [Vec4; 4],
    /// Camera position (.xyz = position, .w = padding).
    pub cam_pos: Vec4,
    /// Total elapsed time in seconds.
    pub time: f32,
    /// Viewport resolution (width, height).
    pub resolution: [f32; 2],
    /// Max shadow ray distance for sun occlusion.
    pub sun_shadow_max: f32,
    /// Sun direction (.xyz = normalized direction, .w = intensity).
    pub sun_dir: Vec4,
    /// Sun color (.rgb = color, .a = unused).
    pub sun_color: Vec4,
    /// Sky hemisphere color (.rgb = color, .a = sky_intensity).
    pub sky_color: Vec4,
    /// Ground hemisphere color (.rgb = color, .a = unused).
    pub ground_color: Vec4,
    /// Selected block position (.xyz = integer coords) + active flag (.w > 0.5 = active).
    pub selected_block: Vec4,
}

impl GlobalUniforms {
    /// Create new global uniforms from camera matrices.
    pub fn new(
        view_inverse: [[f32; 4]; 4],
        proj_inverse: [[f32; 4]; 4],
        cam_pos: [f32; 3],
        time: f32,
        resolution: [f32; 2],
        sun_dir: [f32; 3],
        sun_intensity: f32,
        sun_color: [f32; 3],
        sky_color: [f32; 3],
        sky_intensity: f32,
        ground_color: [f32; 3],
        sun_shadow_max: f32,
        selected_block: Option<[i32; 3]>,
    ) -> Self {
        let sd = glam::Vec3::from_array(sun_dir).normalize_or_zero();
        Self {
            view_inverse: [
                Vec4::from_array(view_inverse[0]),
                Vec4::from_array(view_inverse[1]),
                Vec4::from_array(view_inverse[2]),
                Vec4::from_array(view_inverse[3]),
            ],
            proj_inverse: [
                Vec4::from_array(proj_inverse[0]),
                Vec4::from_array(proj_inverse[1]),
                Vec4::from_array(proj_inverse[2]),
                Vec4::from_array(proj_inverse[3]),
            ],
            cam_pos: Vec4::new(cam_pos[0], cam_pos[1], cam_pos[2], 0.0),
            time,
            resolution,
            sun_shadow_max,
            sun_dir: Vec4::new(sd.x, sd.y, sd.z, sun_intensity),
            sun_color: Vec4::new(sun_color[0], sun_color[1], sun_color[2], 0.0),
            sky_color: Vec4::new(sky_color[0], sky_color[1], sky_color[2], sky_intensity),
            ground_color: Vec4::new(ground_color[0], ground_color[1], ground_color[2], 0.0),
            selected_block: if let Some(pos) = selected_block {
                Vec4::new(pos[0] as f32, pos[1] as f32, pos[2] as f32, 1.0)
            } else {
                Vec4::new(0.0, 0.0, 0.0, 0.0)
            },
        }
    }
}

/// Camera push constants for per-frame updates.
///
/// Replaces uniform buffer to avoid descriptor caching issues.
/// Exactly 128 bytes (Vulkan guaranteed minimum).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CameraPushConstants {
    /// Inverse view matrix (Camera -> World), column-major.
    pub view_inverse: [Vec4; 4],   // 64 bytes
    /// Inverse projection matrix (Screen -> Camera), column-major.
    pub proj_inverse: [Vec4; 4],   // 64 bytes
}

impl CameraPushConstants {
    /// Create from camera matrices.
    pub fn new(view_inverse: [[f32; 4]; 4], proj_inverse: [[f32; 4]; 4]) -> Self {
        Self {
            view_inverse: [
                Vec4::from_array(view_inverse[0]),
                Vec4::from_array(view_inverse[1]),
                Vec4::from_array(view_inverse[2]),
                Vec4::from_array(view_inverse[3]),
            ],
            proj_inverse: [
                Vec4::from_array(proj_inverse[0]),
                Vec4::from_array(proj_inverse[1]),
                Vec4::from_array(proj_inverse[2]),
                Vec4::from_array(proj_inverse[3]),
            ],
        }
    }
}

/// Input state for shaders.
///
/// std430 layout: Vec4 used for 16-byte alignment.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct InputState {
    /// Camera position (xyz), w unused.
    pub camera_pos: Vec4,
    /// View direction (xyz), w unused.
    pub view_dir: Vec4,
    /// Inverse view-projection matrix for ray reconstruction.
    /// Stored as 4x Vec4 (column-major) for proper GPU alignment.
    pub inv_view_proj: [Vec4; 4],
    /// Input flags packed as uvec4.
    /// x: movement flags (forward, back, left, right, up, down)
    /// y: action flags (primary, secondary, jump, crouch)
    /// z: modifier flags (shift, ctrl, alt)
    /// w: reserved
    pub input_flags: UVec4,
}

impl InputState {
    // Movement flag bits (input_flags.x)
    pub const MOVE_FORWARD: u32 = 1 << 0;
    pub const MOVE_BACK: u32 = 1 << 1;
    pub const MOVE_LEFT: u32 = 1 << 2;
    pub const MOVE_RIGHT: u32 = 1 << 3;
    pub const MOVE_UP: u32 = 1 << 4;
    pub const MOVE_DOWN: u32 = 1 << 5;

    // Action flag bits (input_flags.y)
    pub const ACTION_PRIMARY: u32 = 1 << 0;
    pub const ACTION_SECONDARY: u32 = 1 << 1;
    pub const ACTION_JUMP: u32 = 1 << 2;
    pub const ACTION_CROUCH: u32 = 1 << 3;

    // Modifier flag bits (input_flags.z)
    pub const MOD_SHIFT: u32 = 1 << 0;
    pub const MOD_CTRL: u32 = 1 << 1;
    pub const MOD_ALT: u32 = 1 << 2;

    /// Create input state with camera position, view direction, and inverse view-projection.
    pub fn new(camera_pos: [f32; 3], view_dir: [f32; 3], inv_view_proj: [[f32; 4]; 4]) -> Self {
        Self {
            camera_pos: Vec4::new(camera_pos[0], camera_pos[1], camera_pos[2], 0.0),
            view_dir: Vec4::new(view_dir[0], view_dir[1], view_dir[2], 0.0),
            inv_view_proj: [
                Vec4::from_array(inv_view_proj[0]),
                Vec4::from_array(inv_view_proj[1]),
                Vec4::from_array(inv_view_proj[2]),
                Vec4::from_array(inv_view_proj[3]),
            ],
            input_flags: UVec4::ZERO,
        }
    }

    /// Set movement flags.
    pub fn with_movement(mut self, flags: u32) -> Self {
        self.input_flags.x = flags;
        self
    }

    /// Set action flags.
    pub fn with_actions(mut self, flags: u32) -> Self {
        self.input_flags.y = flags;
        self
    }

    pub fn with_modifiers(mut self, flags: u32) -> Self {
        self.input_flags.z = flags;
        self
    }
}

/// Point light structure matching shader definition.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct PointLight {
    /// xyz = world position, w = radius
    pub position: Vec4,
    /// rgb = color, w = intensity
    pub color: Vec4,
    /// 1 = Cast Shadows, 0 = No Shadows
    pub flags: u32,
    /// 16-byte alignment padding
    pub padding: [u32; 3],
}

/// Uniform buffer for point lights.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct LightBuffer {
    pub count: u32,
    pub _pad: [u32; 3],
    pub lights: [PointLight; 16],
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn global_uniforms_layout() {
        // view_inverse (64) + proj_inverse (64) + cam_pos (16) + time+resolution+padding (16)
        // + sun_dir (16) + sun_color (16) + sky_color (16) + ground_color (16) + selected_block (16) = 240
        assert_eq!(size_of::<GlobalUniforms>(), 240);
        assert_eq!(align_of::<GlobalUniforms>(), 16);
    }

    #[test]
    fn input_state_layout() {
        // 2 * Vec4 (camera_pos, view_dir) + 4 * Vec4 (inv_view_proj) + 1 * UVec4 = 112 bytes
        assert_eq!(size_of::<InputState>(), 112);
        assert_eq!(align_of::<InputState>(), 16);
    }

    #[test]
    fn light_buffer_layout() {
        assert_eq!(size_of::<PointLight>(), 48);
        assert_eq!(align_of::<PointLight>(), 16);
        assert_eq!(size_of::<LightBuffer>(), 16 + 16 * 48);
        assert_eq!(align_of::<LightBuffer>(), 16);
    }
}

