//! First-person camera controller.
//!
//! Uses standard game coordinates: Y-up, -Z forward.
//! Vulkan's Y-flip is handled in the projection matrix.

use ara_core::glam::{Mat4, Vec3};
use ara_core::{Action, InputManager};

/// Movement speed in units per second.
const MOVE_SPEED: f32 = 10.0;
/// Sprint speed multiplier.
const SPRINT_MULTIPLIER: f32 = 3.0;
/// Mouse sensitivity (radians per pixel).
const MOUSE_SENSITIVITY: f32 = 0.002;
/// Pitch clamp to prevent gimbal lock (radians).
const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

/// First-person camera with mouse look and movement.
///
/// Coordinate system: Y-up, -Z forward (standard game coordinates).
pub struct FpsCamera {
    /// Camera position in world space.
    pub position: Vec3,
    /// Horizontal rotation in radians (0 = -Z direction).
    pub yaw: f32,
    /// Vertical rotation in radians (positive = looking up).
    pub pitch: f32,
    /// Cached view matrix.
    view_matrix: Mat4,
    /// Cached projection matrix.
    proj_matrix: Mat4,
}

impl FpsCamera {
    /// Create a new camera at the origin facing -Z.
    pub fn new(aspect_ratio: f32) -> Self {
        let position = Vec3::new(32.0, 62.0, 10.0); // Start above ground
        let yaw = 0.0;
        let pitch = -0.3; // Looking slightly down

        let proj_matrix = Self::create_projection(aspect_ratio);
        let view_matrix = Self::create_view(position, yaw, pitch);

        Self {
            position,
            yaw,
            pitch,
            view_matrix,
            proj_matrix,
        }
    }

    /// Create a perspective projection matrix for Vulkan.
    ///
    /// Vulkan has Y pointing down in NDC, so we flip Y here.
    fn create_projection(aspect_ratio: f32) -> Mat4 {
        let fov_y = std::f32::consts::FRAC_PI_4; // 45 degrees
        let near = 0.1;
        let far = 1000.0;

        let mut proj = Mat4::perspective_rh(fov_y, aspect_ratio, near, far);
        // Flip Y for Vulkan's NDC (Y-down in clip space)
        proj.y_axis.y *= -1.0;
        proj
    }

    /// Create view matrix from position and rotation.
    ///
    /// Uses standard Y-up coordinates.
    fn create_view(position: Vec3, yaw: f32, pitch: f32) -> Mat4 {
        let (sin_yaw, cos_yaw) = yaw.sin_cos();
        let (sin_pitch, cos_pitch) = pitch.sin_cos();

        // Forward vector: yaw rotates around Y, pitch tilts up/down
        let forward = Vec3::new(
            sin_yaw * cos_pitch,
            sin_pitch, // Positive pitch = looking up
            -cos_yaw * cos_pitch,
        );

        // World up is +Y
        let up = Vec3::Y;

        Mat4::look_to_rh(position, forward, up)
    }

    /// Update camera from input state.
    pub fn update(&mut self, input: &mut InputManager, dt: f32) {
        // Apply mouse look
        let (dx, dy) = input.take_mouse_delta();
        self.yaw += (dx as f32) * MOUSE_SENSITIVITY;
        self.pitch -= (dy as f32) * MOUSE_SENSITIVITY; // Invert Y for natural mouse look

        // Clamp pitch
        self.pitch = self.pitch.clamp(-PITCH_LIMIT, PITCH_LIMIT);

        // Wrap yaw to [-pi, pi]
        self.yaw = (self.yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
            - std::f32::consts::PI;

        // Movement input
        let forward_back = input.get_axis(Action::MoveBackward, Action::MoveForward);
        let strafe = input.get_axis(Action::StrafeLeft, Action::StrafeRight);
        let vertical = input.get_axis(Action::FlyDown, Action::FlyUp);

        if forward_back != 0.0 || strafe != 0.0 || vertical != 0.0 {
            // Forward/back uses yaw only (fly mode, not pitch-relative)
            let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
            let forward_dir = Vec3::new(sin_yaw, 0.0, -cos_yaw);
            let right_dir = Vec3::new(cos_yaw, 0.0, sin_yaw);

            let mut move_dir = forward_dir * forward_back + right_dir * strafe + Vec3::Y * vertical;

            if move_dir != Vec3::ZERO {
                move_dir = move_dir.normalize();
            }

            let speed = if input.is_active(Action::Sprint) {
                MOVE_SPEED * SPRINT_MULTIPLIER
            } else {
                MOVE_SPEED
            };

            self.position += move_dir * speed * dt;
        }

        // Recalculate view matrix
        self.view_matrix = Self::create_view(self.position, self.yaw, self.pitch);
    }

    /// Update aspect ratio (call on window resize).
    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.proj_matrix = Self::create_projection(aspect_ratio);
    }



    /// Get the inverse view matrix (Camera -> World transform).
    pub fn view_inverse(&self) -> Mat4 {
        self.view_matrix.inverse()
    }

    /// Get the inverse projection matrix (Screen -> Camera).
    pub fn proj_inverse(&self) -> Mat4 {
        self.proj_matrix.inverse()
    }


}
