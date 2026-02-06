//! Turi — client application.

mod assets;
mod camera;
mod gpu;
mod renderer;

use std::sync::Arc;
use std::time::Instant;

use log::{debug, error, info};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use ara_core::{Action, GlobalUniforms, InputManager, PackedVoxel};
use camera::FpsCamera;
use gpu::GpuContext;
use renderer::Renderer;

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
fn generate_test_voxels() -> Vec<PackedVoxel> {
    let size = 64;
    let mut voxels = vec![PackedVoxel::AIR; size * size * size];

    for z in 0..size {
        for x in 0..size {
            // Stone sub-surface: y=61..63
            for y in 61..63 {
                let idx = z * size * size + y * size + x;
                voxels[idx] = PackedVoxel::new(1);
            }

            // Ground surface at y=63: checkerboard stone/grass
            let ground = if (x + z) % 2 == 0 { 1u16 } else { 3u16 };
            let idx = z * size * size + 63 * size + x;
            voxels[idx] = PackedVoxel::new(ground);
        }
    }

    // Lava pool in center at y=63
    for z in 28..36 {
        for x in 28..36 {
            let idx = z * size * size + 63 * size + x;
            voxels[idx] = PackedVoxel::new(8);
        }
    }

    voxels
}

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    renderer: Option<Renderer>,
    camera: FpsCamera,
    input: InputManager,
    last_frame_time: Option<Instant>,
    start_time: Instant,
    frame_count: u32,
    fps_update_time: Instant,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            renderer: None,
            camera: FpsCamera::new(INITIAL_WIDTH as f32 / INITIAL_HEIGHT as f32),
            input: default_input_bindings(),
            last_frame_time: None,
            start_time: Instant::now(),
            frame_count: 0,
            fps_update_time: Instant::now(),
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

        let test_voxels = generate_test_voxels();

        let size = window.inner_size();
        let renderer = Renderer::new(
            &gpu,
            surface,
            size.width,
            size.height,
            &palette_data,
            &test_voxels,
        );

        let aspect = size.width as f32 / size.height.max(1) as f32;
        self.camera.set_aspect_ratio(aspect);

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
                                println!("Assets reloaded successfully.");
                            }
                            Err(e) => eprintln!("Failed to reload assets: {e}"),
                        }
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
                ..
            } => {
                self.grab_cursor();
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

                if let (Some(gpu), Some(renderer), Some(window)) =
                    (&self.gpu, &self.renderer, &self.window)
                {
                    let time = self.start_time.elapsed().as_secs_f32();
                    let size = window.inner_size();
                    let uniforms = GlobalUniforms::new(
                        self.camera.view_inverse().to_cols_array_2d(),
                        self.camera.proj_inverse().to_cols_array_2d(),
                        self.camera.position.into(),
                        time,
                        [size.width as f32, size.height as f32],
                    );

                    match renderer.render(gpu, &uniforms) {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            let size = window.inner_size();
                            if let Some(r) = &mut self.renderer {
                                r.resize(gpu, size.width, size.height);
                            }
                        }
                        Err(wgpu::SurfaceError::OutOfMemory) => {
                            error!("Out of GPU memory");
                            event_loop.exit();
                        }
                        Err(e) => {
                            error!("Surface error: {e}");
                        }
                    }
                }

                // FPS counter in window title
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
