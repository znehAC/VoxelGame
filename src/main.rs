//! Turi — client application.

mod assets;
mod camera;
mod gpu;
mod light;
mod postprocess;
mod renderer;
mod settings;
mod ui;
mod world;

use std::sync::Arc;
use std::time::Instant;

use log::{debug, error, info};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use ara_core::glam;
use ara_core::{
    Action, BlockRegistry, GlobalUniforms, InputManager, LightBuffer, PointLight,
};
use camera::{FpsCamera, HaltonJitter};
use gpu::GpuContext;
use renderer::{AaMode, Renderer};
use world::WorldManager;

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

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    registry: Option<BlockRegistry>,
    world: Option<WorldManager>,
    camera: FpsCamera,
    input: InputManager,
    settings: settings::RenderSettings,
    last_frame_time: Option<Instant>,
    start_time: Instant,
    frame_count: u32,
    fps_update_time: Instant,
    jitter: HaltonJitter,
    debug_chunk_grid: bool,
    shadow_quality_high: bool,
}

impl App {
    fn new() -> Self {
        let settings = settings::RenderSettings::default();
        Self {
            window: None,
            gpu: None,
            renderer: None,
            registry: None,
            world: None,
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
            jitter: HaltonJitter::new(8),
            debug_chunk_grid: false,
            shadow_quality_high: true,
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

        // GPU terrain generation
        let mut world = WorldManager::new(&gpu);
        world.generate_initial_chunks(&gpu, self.camera.position);

        self.registry = Some(registry);

        let size = window.inner_size();
        let mut renderer = Renderer::new(
            &gpu,
            surface,
            size.width,
            size.height,
            &palette_data,
            world.voxel_buf(),
            world.occupancy_buf(),
            self.settings.bloom_threshold,
            self.settings.bloom_intensity,
            self.settings.bloom_exposure,
        );

        let aspect = size.width as f32 / size.height.max(1) as f32;
        self.camera.set_aspect_ratio(aspect);

        let initial_view_proj = self.camera.view_proj().to_cols_array_2d();
        renderer.init_prev_view_proj(initial_view_proj);

        self.window = Some(window);
        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        self.world = Some(world);
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

                // F3: Toggle chunk debug grid
                if key == KeyCode::F3 && state == ElementState::Pressed {
                    self.debug_chunk_grid = !self.debug_chunk_grid;
                    info!("Chunk debug grid: {}", if self.debug_chunk_grid { "ON" } else { "OFF" });
                    return;
                }

                // F4: Toggle Shadow Quality
                if key == KeyCode::F4 && state == ElementState::Pressed {
                    self.shadow_quality_high = !self.shadow_quality_high;
                    info!("Shadow Quality: {}", if self.shadow_quality_high { "HIGH" } else { "LOW" });
                    return;
                }

                match state {
                    ElementState::Pressed => self.input.key_down(key as u32),
                    ElementState::Released => self.input.key_up(key as u32),
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                ..
            } => {
                if !self.input.cursor_grabbed {
                    self.grab_cursor();
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

                if let (Some(gpu), Some(renderer), Some(world), Some(window)) =
                    (&self.gpu, &mut self.renderer, &mut self.world, &self.window)
                {
                    // Stream new chunks as player moves
                    world.update_view(gpu, self.camera.position);

                    // Physics Pass
                    {
                        let mut encoder = gpu
                            .device()
                            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Ara Physics Dispatch"),
                            });
                        world.dispatch_physics(&mut encoder);
                        gpu.queue().submit(std::iter::once(encoder.finish()));
                    }

                    let time = self.start_time.elapsed().as_secs_f32();

                    let size = window.inner_size();
                    let (width, height) = (size.width as f32, size.height as f32);

                    // Apply TAA jitter
                    let jitter = if matches!(renderer.aa_mode(), AaMode::Taa | AaMode::TaaThenSmaa)
                    {
                        self.jitter.next()
                    } else {
                        ara_core::glam::Vec2::ZERO
                    };
                    self.camera.set_jitter(jitter);

                    let prev_view_proj_unjittered = renderer.prev_view_proj();
                    let curr_view_proj_unjittered = self.camera.view_proj().to_cols_array_2d();
                    let _curr_view_proj_jittered = self
                        .camera
                        .view_proj_jittered(width, height)
                        .to_cols_array_2d();

                    let proj_inverse_jittered = self
                        .camera
                        .proj_inverse_jittered(width, height)
                        .to_cols_array_2d();

                    let s = &self.settings;
                    let mut uniforms = GlobalUniforms::with_view_proj(
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
                        None,
                        world.world_origin(),
                        prev_view_proj_unjittered,
                        curr_view_proj_unjittered,
                    );
                    uniforms.ground_color.w = if self.shadow_quality_high { 1.0 } else { 0.0 };

                    if self.debug_chunk_grid {
                        uniforms.world_origin.w = 1.0;
                    }

                    let mut lights = LightBuffer::default();

                    // Player torch
                    lights.lights[0] = PointLight {
                        position: (self.camera.position + self.camera.forward() * 0.5).extend(8.0),
                        color: glam::Vec4::new(1.0, 0.6, 0.3, 2.0),
                        flags: 1,
                        padding: [0; 3],
                    };
                    lights.count += 1;

                    match renderer.render(
                        gpu,
                        &uniforms,
                        &lights,
                        &crate::ui::UiContext {
                            screen_width: size.width as f32,
                            screen_height: size.height as f32,
                            selected_block_name: String::new(),
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

                    renderer.update_prev_view_proj(curr_view_proj_unjittered);
                }

                // FPS counter
                self.frame_count += 1;
                let elapsed = self.fps_update_time.elapsed().as_secs_f32();
                if elapsed >= 1.0 {
                    let fps = self.frame_count as f32 / elapsed;
                    let frame_ms = elapsed * 1000.0 / self.frame_count as f32;
                    if let Some(window) = &self.window {
                        window.set_title(&format!(
                            "{WINDOW_TITLE} | {fps:.0} FPS ({frame_ms:.1} ms)"
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
