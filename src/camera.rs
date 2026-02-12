//! First-person camera controller.
//!
//! Uses standard game coordinates: Y-up, -Z forward.
//! Vulkan's Y-flip is handled in the projection matrix.

use ara_core::glam::{Mat4, Vec2, Vec3};
use ara_core::{Action, InputManager};

/// Pitch clamp to prevent gimbal lock (radians).
const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

/// Halton(2,3) sequence for sub-pixel jitter.
const HALTON_SEQUENCE: [(f32, f32); 8] = [
    (0.5, 0.333333),
    (0.25, 0.666667),
    (0.75, 0.111111),
    (0.125, 0.444444),
    (0.625, 0.777778),
    (0.375, 0.222222),
    (0.875, 0.555556),
    (0.0625, 0.888889),
];

/// Halton sequence generator for TAA jitter.
pub struct HaltonJitter {
    sequence: Vec<Vec2>,
    index: usize,
}

impl HaltonJitter {
    /// Create a new Halton jitter generator with specified sample count.
    pub fn new(sample_count: usize) -> Self {
        let count = sample_count.clamp(4, HALTON_SEQUENCE.len());
        let sequence: Vec<Vec2> = HALTON_SEQUENCE[..count]
            .iter()
            .map(|(x, y)| Vec2::new(*x - 0.5, *y - 0.5))
            .collect();
        Self { sequence, index: 0 }
    }

    /// Get the next jitter offset (sub-pixel, in screen space [-0.5, 0.5]).
    pub fn next(&mut self) -> Vec2 {
        let jitter = self.sequence[self.index];
        self.index = (self.index + 1) % self.sequence.len();
        jitter
    }

    /// Reset the sequence to the beginning.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.index = 0;
    }

    /// Get current jitter without advancing.
    #[allow(dead_code)]
    pub fn current(&self) -> Vec2 {
        self.sequence[self.index]
    }

    /// Get the length of the sequence.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.sequence.len()
    }

    /// Check if sequence is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.sequence.is_empty()
    }
}

impl Default for HaltonJitter {
    fn default() -> Self {
        Self::new(8)
    }
}

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
    /// Cached projection matrix (without jitter).
    proj_matrix: Mat4,
    /// Current jitter offset in pixel space.
    jitter: Vec2,
    /// Aspect ratio for projection.
    aspect_ratio: f32,
    /// Field of view in radians.
    fov: f32,
    /// Movement speed in units per second.
    move_speed: f32,
    /// Sprint speed multiplier.
    sprint_multiplier: f32,
    /// Mouse sensitivity (radians per pixel).
    mouse_sensitivity: f32,
}

impl FpsCamera {
    /// Create a new camera with specified configuration.
    pub fn new(
        aspect_ratio: f32,
        start_position: [f32; 3],
        start_pitch: f32,
        fov: f32,
        move_speed: f32,
        sprint_multiplier: f32,
        mouse_sensitivity: f32,
    ) -> Self {
        let position = Vec3::from_array(start_position);
        let yaw = 0.0;
        let pitch = start_pitch;

        let proj_matrix = Self::build_projection(aspect_ratio, fov);
        let view_matrix = Self::create_view(position, yaw, pitch);

        Self {
            position,
            yaw,
            pitch,
            view_matrix,
            proj_matrix,
            jitter: Vec2::ZERO,
            aspect_ratio,
            fov,
            move_speed,
            sprint_multiplier,
            mouse_sensitivity,
        }
    }

    /// Create a perspective projection matrix for Vulkan.
    fn build_projection(aspect_ratio: f32, fov_y: f32) -> Mat4 {
        let near = 0.1;
        let far = 1000.0;

        let mut proj = Mat4::perspective_rh(fov_y, aspect_ratio, near, far);
        // Flip Y for Vulkan's NDC (Y-down in clip space)
        proj.y_axis.y *= -1.0;
        proj
    }

    /// Apply sub-pixel jitter to projection matrix.
    ///
    /// `jitter` is in pixel offset (typically [-0.5, 0.5] range).
    /// Converts to NDC offset: jitter / screen_size * 2.0
    fn create_jittered_projection(base_proj: Mat4, jitter: Vec2, width: f32, height: f32) -> Mat4 {
        let mut proj = base_proj;
        proj.z_axis.x += (jitter.x * 2.0) / width;
        proj.z_axis.y += (jitter.y * 2.0) / height;
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
        let (dx, dy) = input.take_mouse_delta();
        self.yaw += (dx as f32) * self.mouse_sensitivity;
        self.pitch -= (dy as f32) * self.mouse_sensitivity;

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
                self.move_speed * self.sprint_multiplier
            } else {
                self.move_speed
            };

            self.position += move_dir * speed * dt;
        }

        // Recalculate view matrix
        self.view_matrix = Self::create_view(self.position, self.yaw, self.pitch);
    }

    /// Unit forward vector derived from yaw and pitch.
    pub fn forward(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        Vec3::new(sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch)
    }

    /// Update aspect ratio (call on window resize).
    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.aspect_ratio = aspect_ratio;
        self.proj_matrix = Self::build_projection(aspect_ratio, self.fov);
    }

    /// Set the jitter offset for TAA.
    ///
    /// `jitter` is typically in range [-0.5, 0.5] in pixel space.
    pub fn set_jitter(&mut self, jitter: Vec2) {
        self.jitter = jitter;
    }

    /// Get the current jitter offset.
    #[allow(dead_code)]
    pub fn jitter(&self) -> Vec2 {
        self.jitter
    }

    /// Get the inverse view matrix (Camera -> World transform).
    pub fn view_inverse(&self) -> Mat4 {
        self.view_matrix.inverse()
    }

    /// Get the inverse projection matrix (Screen -> Camera) without jitter.
    #[allow(dead_code)]
    pub fn proj_inverse(&self) -> Mat4 {
        self.proj_matrix.inverse()
    }

    /// Get the inverse projection matrix with jitter applied.
    ///
    /// This is used for unprojecting screen coordinates to world space.
    pub fn proj_inverse_jittered(&self, width: f32, height: f32) -> Mat4 {
        let jittered_proj =
            Self::create_jittered_projection(self.proj_matrix, self.jitter, width, height);
        jittered_proj.inverse()
    }

    /// Get the jittered projection matrix.
    ///
    /// Used for rendering with TAA.
    pub fn proj_jittered(&self, width: f32, height: f32) -> Mat4 {
        Self::create_jittered_projection(self.proj_matrix, self.jitter, width, height)
    }

    /// Get the view-projection matrix without jitter.
    pub fn view_proj(&self) -> Mat4 {
        self.proj_matrix * self.view_matrix
    }

    /// Get the jittered view-projection matrix.
    pub fn view_proj_jittered(&self, width: f32, height: f32) -> Mat4 {
        let jittered_proj = self.proj_jittered(width, height);
        jittered_proj * self.view_matrix
    }
}
