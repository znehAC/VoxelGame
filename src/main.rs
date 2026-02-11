//! Turi — client application.

mod assets;
mod camera;
mod gpu;
mod light;
mod postprocess;
mod renderer;
mod settings;
mod ui;

use std::sync::Arc;
use std::time::Instant;

use log::{debug, error, info};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use ara_core::glam;
use ara_core::{
    Action, BlockRegistry, GRID_SIZE, GlobalUniforms, InputManager, LightBuffer, PackedVoxel,
    PointLight, dda_raycast,
};
use camera::{FpsCamera, HaltonJitter};
use gpu::GpuContext;
use renderer::{AaMode, Renderer};

const WINDOW_TITLE: &str = "Turi";
const INITIAL_WIDTH: u32 = 1280;
const INITIAL_HEIGHT: u32 = 720;

/// Create an InputManager with default WASD + Space/C/Shift bindings.
fn default_input_bindings() -> InputManager {
    let mut input = InputManager::new();
    input.bind(KeyCode::KeyW as u32, Action::MoveForward);
    input.bind(KeyCode::KeyS as u32, Action::MoveBackward);
    input.bind(KeyCode::KeyA as u32, Action::StrafeLeft);
    input.bind(KeyCode::KeyD as u32, Action::StrafeRight);
    input.bind(KeyCode::Space as u32, Action::FlyUp);
    input.bind(KeyCode::KeyC as u32, Action::FlyDown);
    input.bind(KeyCode::ShiftLeft as u32, Action::Sprint);
    input
}

/// Generate a 64x64x64 test voxel volume: checkerboard ground + lava pool.
/// Generate a 64x64x64 room scene with walls and a moving lava block.
fn generate_room_scene() -> Vec<PackedVoxel> {
    let size = GRID_SIZE as usize;
    let mut voxels = vec![PackedVoxel::AIR; size * size * size];

    let ground_y = 10;
    let ceiling_y = 50;
    let min_xz = 10;
    let max_xz = 53;

    // Simple pseudo-random number generator for visual variants
    let mut rng_seed = 12345u32;
    let mut next_variant = || -> u8 {
        rng_seed = rng_seed.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng_seed >> 16) & 0xF) as u8
    };

    for z in 0..size {
        for x in 0..size {
            for y in 0..size {
                let idx = z * size * size + y * size + x;

                // Floor (Checkerboard)
                if y == ground_y && x >= min_xz && x <= max_xz && z >= min_xz && z <= max_xz {
                    // Stone (1) / Grass (3) checkerboard
                    let mat = if (x + z) % 2 == 0 { 1u16 } else { 3u16 };
                    // Use random variant for stone to test noise
                    let variant = if mat == 1 { next_variant() } else { 0 };
                    voxels[idx] = PackedVoxel::with_all(mat, 0, 0, variant, 0);
                    continue;
                }

                // Ceiling (Wood)
                if y == ceiling_y && x >= min_xz && x <= max_xz && z >= min_xz && z <= max_xz {
                    voxels[idx] = PackedVoxel::new(6); // Wood
                    continue;
                }

                // Walls (Stone)
                if y > ground_y && y < ceiling_y {
                    let is_wall_x = x == min_xz || x == max_xz;
                    let is_wall_z = z == min_xz || z == max_xz;

                    if (is_wall_x && z >= min_xz && z <= max_xz)
                        || (is_wall_z && x >= min_xz && x <= max_xz)
                    {
                        // Assign random variant to wall stones too
                        voxels[idx] = PackedVoxel::with_all(1, 0, 0, next_variant(), 0);
                        continue;
                    }
                }
            }
        }
    }

    // Add some random pillars inside
    for y in ground_y + 1..ground_y + 5 {
        let idx1 = 20 * size * size + y * size + 20;
        voxels[idx1] = PackedVoxel::new(2); // Dirt

        let idx2 = 40 * size * size + y * size + 40;
        voxels[idx2] = PackedVoxel::new(2); // Dirt
    }

    voxels
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    registry: Option<BlockRegistry>,
    camera: FpsCamera,
    input: InputManager,
    settings: settings::RenderSettings,
    last_frame_time: Option<Instant>,
    start_time: Instant,
    frame_count: u32,
    fps_update_time: Instant,
    voxels: Vec<PackedVoxel>,
    selected_material: u16,
    target_block_name: Option<String>,
    jitter: HaltonJitter,
}

impl App {
    fn new() -> Self {
        let settings = settings::RenderSettings::default();
        Self {
            window: None,
            gpu: None,
            renderer: None,
            registry: None,
            camera: FpsCamera::new(
                INITIAL_WIDTH as f32 / INITIAL_HEIGHT as f32,
                settings.camera_start_position,
                settings.camera_start_pitch,
                settings.camera_fov,
                settings.camera_move_speed,
                settings.camera_sprint_multiplier,
                settings.camera_mouse_sensitivity,
            ),
            input: default_input_bindings(),
            settings,
            last_frame_time: None,
            start_time: Instant::now(),
            frame_count: 0,
            fps_update_time: Instant::now(),
            voxels: Vec::new(),
            selected_material: 1,
            target_block_name: None,
            jitter: HaltonJitter::new(8),
        }
    }

    fn grab_cursor(&mut self) {
        let Some(window) = &self.window else { return };

        let grab_result = window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));

        if grab_result.is_ok() {
            window.set_cursor_visible(false);
            self.input.cursor_grabbed = true;
            debug!("Cursor grabbed");
        } else {
            error!("Failed to grab cursor: {:?}", grab_result);
        }
    }

    fn release_cursor(&mut self) {
        self.input.cursor_grabbed = false;
        if let Some(window) = &self.window {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
        }
        debug!("Cursor released");
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window_attrs = Window::default_attributes()
            .with_title(WINDOW_TITLE)
            .with_inner_size(PhysicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT));

        let window = Arc::new(
            event_loop
                .create_window(window_attrs)
                .expect("Failed to create window"),
        );

        let instance = GpuContext::create_instance();
        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let gpu = pollster::block_on(GpuContext::from_instance(instance, Some(&surface)));

        // Load block definitions from assets and generate palette texture
        let registry = assets::load_blocks().expect("Failed to load block definitions");
        let palette_data = registry.generate_texture_data();
        info!("Loaded {} block types", registry.block_count());

        let test_voxels = generate_room_scene();
        self.voxels = test_voxels.clone();
        self.registry = Some(registry);

        let size = window.inner_size();
        let mut renderer = Renderer::new(
            &gpu,
            surface,
            size.width,
            size.height,
            &palette_data,
            &test_voxels,
            self.settings.bloom_threshold,
            self.settings.bloom_intensity,
            self.settings.bloom_exposure,
        );

        let aspect = size.width as f32 / size.height.max(1) as f32;
        self.camera.set_aspect_ratio(aspect);

        // Initialize prev_view_proj with camera's starting matrix for correct TAA on first frame
        let initial_view_proj = self.camera.view_proj().to_cols_array_2d();
        renderer.init_prev_view_proj(initial_view_proj);

        self.window = Some(window);
        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        self.last_frame_time = Some(Instant::now());
        self.start_time = Instant::now();

        info!("Application initialized");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                info!("Close requested");
                self.renderer = None;
                self.gpu = None;
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => {
                if key == KeyCode::Escape && state == ElementState::Pressed {
                    self.release_cursor();
                    return;
                }

                // F5: Hot reload assets
                if key == KeyCode::F5 && state == ElementState::Pressed {
                    if let (Some(gpu), Some(renderer)) = (&self.gpu, &self.renderer) {
                        match assets::load_blocks() {
                            Ok(registry) => {
                                let palette = registry.generate_texture_data();
                                renderer.reload_palette(gpu, &palette);
                                self.registry = Some(registry);
                                info!("Assets reloaded");
                            }
                            Err(e) => error!("Failed to reload assets: {e}"),
                        }
                    }
                    return;
                }

                // F2: Cycle Anti-Aliasing Mode
                if key == KeyCode::F2 && state == ElementState::Pressed {
                    if let Some(renderer) = &mut self.renderer {
                        let next = match renderer.aa_mode() {
                            AaMode::None => AaMode::Fxaa,
                            AaMode::Fxaa => AaMode::Smaa,
                            AaMode::Smaa => AaMode::Taa,
                            AaMode::Taa => AaMode::TaaThenSmaa,
                            AaMode::TaaThenSmaa => AaMode::None,
                        };
                        renderer.set_aa_mode(next);
                        info!("Anti-Aliasing Mode: {:?}", renderer.aa_mode());
                    }
                    return;
                }

                match state {
                    ElementState::Pressed => self.input.key_down(key as u32),
                    ElementState::Released => self.input.key_up(key as u32),
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } => {
                if !self.input.cursor_grabbed {
                    self.grab_cursor();
                    return;
                }

                let dir = self.camera.forward();
                if let Some(hit) = dda_raycast(&self.voxels, self.camera.position, dir, 64.0) {
                    if let (Some(gpu), Some(renderer)) = (&self.gpu, &self.renderer) {
                        match button {
                            MouseButton::Left => {
                                self.voxels[hit.index] = PackedVoxel::AIR;
                                renderer.update_voxel_at(gpu, hit.index, PackedVoxel::AIR);
                            }
                            MouseButton::Right => {
                                let gs = GRID_SIZE as i32;
                                let neighbor = hit.grid_pos + hit.normal;
                                if neighbor.x >= 0
                                    && neighbor.x < gs
                                    && neighbor.y >= 0
                                    && neighbor.y < gs
                                    && neighbor.z >= 0
                                    && neighbor.z < gs
                                {
                                    let idx = (neighbor.z * gs * gs + neighbor.y * gs + neighbor.x)
                                        as usize;
                                    // Prevent placing inside the camera
                                    let cam_cell = self.camera.position.floor().as_ivec3();
                                    if neighbor != cam_cell {
                                        let voxel = PackedVoxel::new(self.selected_material);
                                        self.voxels[idx] = voxel;
                                        renderer.update_voxel_at(gpu, idx, voxel);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::Focused(false) => {
                self.release_cursor();
                self.input.reset();
            }
            WindowEvent::Resized(new_size) => {
                if let Some(gpu) = &self.gpu {
                    if let Some(renderer) = &mut self.renderer {
                        renderer.resize(gpu, new_size.width, new_size.height);
                    }
                }
                let aspect = new_size.width as f32 / new_size.height.max(1) as f32;
                self.camera.set_aspect_ratio(aspect);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = self
                    .last_frame_time
                    .map(|t| now.duration_since(t).as_secs_f32())
                    .unwrap_or(0.0);
                self.last_frame_time = Some(now);

                self.camera.update(&mut self.input, dt);

                // Debug HUD: raycast to find targeted block name
                self.target_block_name = {
                    let dir = self.camera.forward();
                    dda_raycast(&self.voxels, self.camera.position, dir, 64.0).and_then(|hit| {
                        let voxel = self.voxels[hit.index];
                        self.registry
                            .as_ref()
                            .and_then(|reg| reg.get_block_name(voxel.id()))
                            .map(|s| s.to_owned())
                    })
                };

                if let (Some(gpu), Some(renderer), Some(window)) =
                    (&self.gpu, &mut self.renderer, &self.window)
                {
                    let time = self.start_time.elapsed().as_secs_f32();
                    let gs = GRID_SIZE as usize;

                    // Animated lava orb — partial writes only
                    let radius = 15.0;
                    let center_x = 32.0;
                    let center_z = 32.0;
                    let y = 25usize;

                    let prev_angle = (time - dt) * 1.0;
                    let prev_x = (center_x + radius * prev_angle.cos()) as usize;
                    let prev_z = (center_z + radius * prev_angle.sin()) as usize;

                    if prev_x < gs && prev_z < gs {
                        let idx = prev_z * gs * gs + y * gs + prev_x;
                        self.voxels[idx] = PackedVoxel::AIR;
                        renderer.update_voxel_at(gpu, idx, PackedVoxel::AIR);
                    }

                    let angle = time * 1.0;
                    let curr_x = (center_x + radius * angle.cos()) as usize;
                    let curr_z = (center_z + radius * angle.sin()) as usize;

                    if curr_x < gs && curr_z < gs {
                        let idx = curr_z * gs * gs + y * gs + curr_x;
                        self.voxels[idx] = PackedVoxel::new(8);
                        renderer.update_voxel_at(gpu, idx, PackedVoxel::new(8));
                    }

                    let size = window.inner_size();
                    let (width, height) = (size.width as f32, size.height as f32);

                    // Apply TAA jitter to camera (sub-pixel offset)
                    let jitter = if matches!(renderer.aa_mode(), AaMode::Taa | AaMode::TaaThenSmaa)
                    {
                        self.jitter.next()
                    } else {
                        ara_core::glam::Vec2::ZERO
                    };
                    self.camera.set_jitter(jitter);

                    // Get view-projection matrices for TAA
                    // For CORRECT velocity calculation, we must use UNJITTERED matrices
                    // The jitter is only for sampling different sub-pixel positions during rendering
                    let prev_view_proj_unjittered = renderer.prev_view_proj();
                    let curr_view_proj_unjittered = self.camera.view_proj().to_cols_array_2d();
                    // Note: view_proj_jittered is applied via camera.set_jitter() and used via proj_inverse_jittered
                    let _curr_view_proj_jittered = self
                        .camera
                        .view_proj_jittered(width, height)
                        .to_cols_array_2d();

                    // Use jittered inverse projection for raytracing
                    let proj_inverse_jittered = self
                        .camera
                        .proj_inverse_jittered(width, height)
                        .to_cols_array_2d();

                    let s = &self.settings;
                    let uniforms = GlobalUniforms::with_view_proj(
                        self.camera.view_inverse().to_cols_array_2d(),
                        proj_inverse_jittered,
                        self.camera.position.into(),
                        time,
                        [width, height],
                        s.sun_direction,
                        s.sun_intensity,
                        s.sun_color,
                        s.sky_color,
                        s.sky_intensity,
                        s.ground_color,
                        s.light_max_distance,
                        {
                            let dir = self.camera.forward();
                            dda_raycast(&self.voxels, self.camera.position, dir, 10.0)
                                .map(|hit| hit.grid_pos.to_array())
                        },
                        prev_view_proj_unjittered,
                        curr_view_proj_unjittered,
                    );

                    // Dynamic Lights
                    let mut lights = LightBuffer::default();

                    // 1. Player Torch (warm light)
                    lights.lights[0] = PointLight {
                        position: (self.camera.position + self.camera.forward() * 0.5).extend(8.0), // radius = 8.0
                        color: glam::Vec4::new(1.0, 0.6, 0.3, 2.0), // intensity = 2.0
                        flags: 1,                                   // Shadows enabled (maybe?)
                        padding: [0; 3],
                    };
                    lights.count += 1;

                    // 2. Lava Orb Light (red/orange)
                    if curr_x < gs && curr_z < gs {
                        lights.lights[lights.count as usize] = PointLight {
                            position: glam::Vec4::new(
                                curr_x as f32 + 0.5,
                                25.0 + 0.5,
                                curr_z as f32 + 0.5,
                                12.0,
                            ),
                            color: glam::Vec4::new(1.0, 0.2, 0.0, 3.0),
                            flags: 0, // No shadows for now to save perf or if inside block
                            padding: [0; 3],
                        };
                        lights.count += 1;
                    }

                    match renderer.render(
                        gpu,
                        &uniforms,
                        &lights,
                        &crate::ui::UiContext {
                            screen_width: size.width as f32,
                            screen_height: size.height as f32,
                            selected_block_name: self.target_block_name.clone().unwrap_or_default(),
                        },
                    ) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let size = window.inner_size();
                            renderer.resize(gpu, size.width, size.height);
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            error!("Out of GPU memory");
                            event_loop.exit();
                        }
                        Err(e) => {
                            error!("Surface error: {e}");
                        }
                    }

                    // Update prev_view_proj AFTER rendering for NEXT frame's velocity calculation.
                    // Store the CURRENT frame's UNJITTERED matrix so next frame can use it as "previous".
                    // This must be done AFTER render() so the CURRENT frame uses the correct prev/current pair.
                    renderer.update_prev_view_proj(curr_view_proj_unjittered);
                }

                // FPS counter + target block in window title
                self.frame_count += 1;
                let elapsed = self.fps_update_time.elapsed().as_secs_f32();
                if elapsed >= 1.0 {
                    let fps = self.frame_count as f32 / elapsed;
                    let frame_ms = elapsed * 1000.0 / self.frame_count as f32;
                    let block_label = self.target_block_name.as_deref().unwrap_or("---");
                    if let Some(window) = &self.window {
                        window.set_title(&format!(
                            "{WINDOW_TITLE} | {block_label} | {fps:.0} FPS ({frame_ms:.1} ms)"
                        ));
                    }
                    self.frame_count = 0;
                    self.fps_update_time = Instant::now();
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            self.input.mouse_move(dx, dy);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    info!("Starting Turi");

    let event_loop = EventLoop::new().expect("Failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Event loop error");

    info!("Turi shutdown complete");
}
