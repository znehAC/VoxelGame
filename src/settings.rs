/// Centralized rendering and camera configuration defaults.
pub struct RenderSettings {
    pub bloom_threshold: f32,
    pub bloom_intensity: f32,
    pub bloom_exposure: f32,
    pub sun_direction: [f32; 3],
    pub sun_intensity: f32,
    pub sun_color: [f32; 3],
    pub sky_color: [f32; 3],
    pub sky_intensity: f32,
    pub ground_color: [f32; 3],
    pub light_max_distance: f32,
    pub camera_move_speed: f32,
    pub camera_sprint_multiplier: f32,
    pub camera_mouse_sensitivity: f32,
    pub camera_fov: f32,
    pub camera_start_position: [f32; 3],
    pub camera_start_pitch: f32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            bloom_threshold: 1.0,
            bloom_intensity: 0.8,
            bloom_exposure: 1.0,
            sun_direction: [0.4, -0.7, 0.3],
            sun_intensity: 2.0,
            sun_color: [1.0, 0.95, 0.85],
            sky_color: [0.4, 0.6, 0.9],
            sky_intensity: 0.15,
            ground_color: [0.15, 0.1, 0.05],
            light_max_distance: 128.0,
            camera_move_speed: 10.0,
            camera_sprint_multiplier: 3.0,
            camera_mouse_sensitivity: 0.002,
            camera_fov: std::f32::consts::FRAC_PI_4,
            camera_start_position: [32.0, 62.0, 10.0],
            camera_start_pitch: -0.3,
        }
    }
}
