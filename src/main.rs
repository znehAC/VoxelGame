//! Turi — client application.

mod assets;
mod brick_map;
mod camera;
mod chunk_streamer;
mod clipmap;
mod gpu;
mod postprocess;
mod ray_pipeline;
mod renderer;
mod settings;

mod terrain_pipeline;
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
    Action, BlockRegistry, GlobalUniforms, InputManager, LightBuffer, MemoryBudget, PackedVoxel,
    PointLight, WORLD_EXTENT, dda_raycast,
};
use brick_map::BrickMap;
use camera::{FpsCamera, HaltonJitter};
use chunk_streamer::ChunkStreamer;
use gpu::GpuContext;
use renderer::{AaMode, Renderer};

use terrain_pipeline::TerrainPipeline;

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
    camera: FpsCamera,
    input: InputManager,
    settings: settings::RenderSettings,
    last_frame_time: Option<Instant>,
    start_time: Instant,
    frame_count: u32,
    fps_update_time: Instant,
    brick_map: Option<BrickMap>,
    streamer: Option<ChunkStreamer>,

    terrain_pipeline: Option<TerrainPipeline>,
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
            brick_map: None,
            streamer: None,

            terrain_pipeline: None,
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

        // Build Sparse Brick Map with GPU-driven terrain generation
        let mut sbm = BrickMap::new(&gpu, MemoryBudget::High);
        let terrain = TerrainPipeline::new(&gpu, &sbm, 256);
        let mut streamer = ChunkStreamer::new(128);
        let spawn_pos = glam::Vec3::new(256.0, 220.0, 256.0);
        streamer.initial_load(spawn_pos, &mut sbm, &terrain, &gpu);
        self.registry = Some(registry);

        let size = window.inner_size();
        let mut renderer = Renderer::new(
            &gpu,
            surface,
            size.width,
            size.height,
            &palette_data,
            &sbm,
            self.settings.bloom_threshold,
            self.settings.bloom_intensity,
            self.settings.bloom_exposure,
            self.settings.vsync,
            spawn_pos,
        );

        let aspect = size.width as f32 / size.height.max(1) as f32;
        self.camera.set_aspect_ratio(aspect);

        let initial_view_proj = self.camera.view_proj().to_cols_array_2d();
        renderer.init_prev_view_proj(initial_view_proj);



        self.window = Some(window);
        self.gpu = Some(gpu);
        self.renderer = Some(renderer);
        self.brick_map = Some(sbm);
        self.streamer = Some(streamer);

        self.terrain_pipeline = Some(terrain);
        self.last_frame_time = Some(Instant::now());
        self.start_time = Instant::now();

        info!("Application initialized (Sparse Brick Map)");
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

                // F3: Toggle VSync
                if key == KeyCode::F3 && state == ElementState::Pressed {
                    if let (Some(gpu), Some(renderer)) = (&self.gpu, &mut self.renderer) {
                        self.settings.vsync = !self.settings.vsync;
                        renderer.set_vsync(gpu, self.settings.vsync);
                        info!("VSync: {}", if self.settings.vsync { "On" } else { "Off" });
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
                if let Some(sbm) = &mut self.brick_map {
                    if let Some(hit) =
                        dda_raycast(self.camera.position, dir, 64.0, |pos| sbm.get_voxel(pos))
                    {
                        if let Some(gpu) = &self.gpu {
                            match button {
                                MouseButton::Left => {
                                    sbm.write_voxel(gpu, hit.grid_pos, PackedVoxel::AIR);
                                }
                                MouseButton::Right => {
                                    let we = WORLD_EXTENT as i32;
                                    let neighbor = hit.grid_pos + hit.normal;
                                    if neighbor.x >= 0
                                        && neighbor.x < we
                                        && neighbor.y >= 0
                                        && neighbor.y < we
                                        && neighbor.z >= 0
                                        && neighbor.z < we
                                    {
                                        let cam_cell = self.camera.position.floor().as_ivec3();
                                        if neighbor != cam_cell {
                                            let voxel = PackedVoxel::new(self.selected_material);
                                            sbm.write_voxel(gpu, neighbor, voxel);
                                        }
                                    }
                                }
                                _ => {}
                            }
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
                        renderer.resize(gpu, new_size.width, new_size.height, self.brick_map.as_ref().unwrap());
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
                    if let Some(sbm) = &self.brick_map {
                        dda_raycast(self.camera.position, dir, 64.0, |pos| sbm.get_voxel(pos))
                            .and_then(|hit| {
                                let voxel = sbm.get_voxel(hit.grid_pos);
                                self.registry
                                    .as_ref()
                                    .and_then(|reg| reg.get_block_name(voxel.id()))
                                    .map(|s| s.to_owned())
                            })
                    } else {
                        None
                    }
                };

                if let (Some(gpu), Some(renderer), Some(window), Some(sbm), Some(streamer), Some(terrain)) = (
                    &self.gpu,
                    &mut self.renderer,
                    &self.window,
                    &mut self.brick_map,
                    &mut self.streamer,
                    &self.terrain_pipeline,
                ) {
                    // Stream terrain: evict/load bricks around camera, dispatch GPU terrain gen.
                    {
                        let mut enc = gpu.device().create_command_encoder(
                            &wgpu::CommandEncoderDescriptor { label: Some("Terrain Stream") },
                        );
                        streamer.update(self.camera.position, sbm, terrain, &mut enc, gpu);
                        gpu.queue().submit(std::iter::once(enc.finish()));
                    }

                    let time = self.start_time.elapsed().as_secs_f32();
                    let _time = time; // used for future dynamic objects

                    let size = window.inner_size();
                    let (width, height) = (size.width as f32, size.height as f32);

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
                    let wo = sbm.lod0_origin();
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
                            dda_raycast(self.camera.position, dir, 10.0, |pos| sbm.get_voxel(pos))
                                .map(|hit| hit.grid_pos.to_array())
                        },
                        prev_view_proj_unjittered,
                        curr_view_proj_unjittered,
                        [wo[0] as f32, wo[1] as f32, wo[2] as f32],
                        self.frame_count as f32,
                    );

                    // Dynamic Lights
                    let mut lights = LightBuffer::default();

                    // Player Torch
                    lights.lights[0] = PointLight {
                        position: (self.camera.position + self.camera.forward() * 0.5).extend(8.0),
                        color: glam::Vec4::new(1.0, 0.6, 0.3, 2.0),
                        flags: 1,
                        padding: [0; 3],
                    };
                    lights.count += 1;


                    match renderer.render(
                        gpu,
                        self.camera.position,
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
                            renderer.resize(gpu, size.width, size.height, self.brick_map.as_ref().unwrap());
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

                // FPS counter + debug info in window title
                self.frame_count += 1;
                let elapsed = self.fps_update_time.elapsed().as_secs_f32();
                if elapsed >= 1.0 {
                    let fps = self.frame_count as f32 / elapsed;
                    let frame_ms = elapsed * 1000.0 / self.frame_count as f32;
                    let pos = self.camera.position;
                    let yaw_deg = self.camera.yaw.to_degrees();
                    let pitch_deg = self.camera.pitch.to_degrees();
                    // yaw=0 → -Z (North), yaw=90 → +X (East)
                    let dir = match (((yaw_deg % 360.0) + 360.0) % 360.0) as u32 {
                        315..=360 | 0..=44  => "N",
                        45..=134            => "E",
                        135..=224           => "S",
                        _                   => "W",
                    };
                    let (lod0_orig, pool_pct) = self.brick_map.as_ref().map(|bm| {
                        let o = bm.lod0_origin();
                        (format!("[{},{},{}]", o[0], o[1], o[2]), bm.utilization() * 100.0)
                    }).unwrap_or_default();
                    if let Some(window) = &self.window {
                        window.set_title(&format!(
                            "{WINDOW_TITLE} | {fps:.0} FPS ({frame_ms:.1}ms) | \
                             pos({:.0},{:.0},{:.0}) yaw{yaw_deg:.0}°{dir} pitch{pitch_deg:.0}° | \
                             lod0:{lod0_orig} pool:{pool_pct:.0}%",
                            pos.x, pos.y, pos.z
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
