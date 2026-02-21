//! Turi — client application.

mod assets;
mod camera;
mod font;
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
use winit::event::{
    DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use ara_core::glam::{self, IVec3};
use ara_core::{
    Action, BlockRegistry, GRID_SIZE, GlobalUniforms, InputManager, LightBuffer, PackedVoxel,
    PointLight, RayHit, dda_raycast,
};
use camera::{FpsCamera, HaltonJitter};
use font::BitmapFont;
use gpu::GpuContext;
use renderer::{AaMode, Renderer};
use ui::{BlockPickerState, HotbarState, UiContext};
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

fn aa_mode_name(mode: AaMode) -> &'static str {
    match mode {
        AaMode::None => "None",
        AaMode::Fxaa => "FXAA",
        AaMode::Smaa => "SMAA",
        AaMode::Taa => "TAA",
        AaMode::TaaThenSmaa => "TAA+SMAA",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InteractionMode {
    Block,
    Sphere,
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
    jitter: HaltonJitter,
    debug_chunk_grid: bool,
    shadow_quality_high: bool,

    // FPS tracking
    fps_frame_count: u32,
    fps_update_time: Instant,
    display_fps: f32,
    display_frame_time: f32,

    // Debug HUD
    debug_hud_visible: bool,

    // Block interaction
    voxels_cpu: Vec<u32>,
    targeted_block: Option<RayHit>,
    left_click_pending: bool,
    right_click_pending: bool,
    interaction_mode: InteractionMode,
    brush_dist: f32,

    // Hotbar & picker
    hotbar: HotbarState,
    picker: BlockPickerState,
    mouse_pos: (f32, f32),

    // Brush
    brush_size: u32,
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
            jitter: HaltonJitter::new(8),
            debug_chunk_grid: false,
            shadow_quality_high: true,

            fps_frame_count: 0,
            fps_update_time: Instant::now(),
            display_fps: 0.0,
            display_frame_time: 0.0,

            debug_hud_visible: false,

            voxels_cpu: Vec::new(),
            targeted_block: None,
            left_click_pending: false,
            right_click_pending: false,
            interaction_mode: InteractionMode::Block,
            brush_dist: 10.0,

            hotbar: HotbarState::new(),
            picker: BlockPickerState::new(),
            mouse_pos: (0.0, 0.0),

            brush_size: 1,
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

    fn open_picker(&mut self) {
        self.picker.visible = true;
        self.release_cursor();
    }

    fn close_picker(&mut self) {
        self.picker.visible = false;
        self.grab_cursor();
    }

    /// Per-frame CPU raycast for block targeting.
    fn update_targeted_block(&mut self) {
        if self.voxels_cpu.is_empty() {
            self.targeted_block = None;
            return;
        }

        // Pass world position directly. dda_raycast handles toroidal wrapping internally.
        let voxels: &[PackedVoxel] = bytemuck::cast_slice(&self.voxels_cpu);
        self.targeted_block =
            dda_raycast(voxels, self.camera.position, self.camera.forward(), 100.0);
    }

    /// Apply a block modification (place or destroy) with the current brush size.
    fn apply_brush(&mut self, center: IVec3, packed: u32) {
        if let (Some(gpu), Some(world)) = (&self.gpu, &mut self.world) {
            let is_sphere = matches!(self.interaction_mode, InteractionMode::Sphere);
            let radius = if is_sphere {
                self.brush_size as f32
            } else {
                0.0
            };

            world.dispatch_brush(gpu, center, radius, packed, is_sphere);
            world.request_readback(gpu, false);
        }
    }

    fn handle_block_interaction(&mut self) {
        if self.picker.visible {
            return;
        }

        if self.left_click_pending {
            self.left_click_pending = false;
            match self.interaction_mode {
                InteractionMode::Block => {
                    if let Some(hit) = &self.targeted_block {
                        self.apply_brush(hit.grid_pos, 0);
                    }
                }
                InteractionMode::Sphere => {
                    let target = self.camera.position + self.camera.forward() * self.brush_dist;
                    let target_pos = target.floor().as_ivec3();
                    self.apply_brush(target_pos, 0);
                }
            }
        }

        if self.right_click_pending {
            self.right_click_pending = false;
            let block_id = self.hotbar.selected_block_id();
            let packed = PackedVoxel::new(block_id).packed;

            match self.interaction_mode {
                InteractionMode::Block => {
                    if let Some(hit) = &self.targeted_block {
                        let place_pos = hit.grid_pos + hit.normal;
                        self.apply_brush(place_pos, packed);
                    }
                }
                InteractionMode::Sphere => {
                    let target = self.camera.position + self.camera.forward() * self.brush_dist;
                    let target_pos = target.floor().as_ivec3();
                    self.apply_brush(target_pos, packed);
                }
            }
        }
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

        // Readback voxels for CPU-side raycast
        info!("Reading back voxel data from GPU (initially blocking)...");
        let grid_count = (GRID_SIZE as usize) * (GRID_SIZE as usize) * (GRID_SIZE as usize);
        self.voxels_cpu = vec![0; grid_count];
        world.initial_blocking_readback(&gpu, &mut self.voxels_cpu);
        info!("CPU voxel mirror ready ({} voxels)", self.voxels_cpu.len());

        // Generate font atlas
        let atlas_data = BitmapFont::generate_atlas();

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
            &atlas_data,
            font::ATLAS_W,
            font::ATLAS_H,
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
            WindowEvent::CursorMoved { position, .. } => {
                self.mouse_pos = (position.x as f32, position.y as f32);
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
                if state == ElementState::Pressed {
                    // ESC: close picker or release cursor
                    if key == KeyCode::Escape {
                        if self.picker.visible {
                            self.close_picker();
                        } else {
                            self.release_cursor();
                        }
                        return;
                    }

                    // F1: Toggle debug HUD
                    if key == KeyCode::F1 {
                        self.debug_hud_visible = !self.debug_hud_visible;
                        return;
                    }

                    // F2: Cycle Anti-Aliasing Mode
                    if key == KeyCode::F2 {
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
                    if key == KeyCode::F3 {
                        self.debug_chunk_grid = !self.debug_chunk_grid;
                        info!(
                            "Chunk debug grid: {}",
                            if self.debug_chunk_grid { "ON" } else { "OFF" }
                        );
                        return;
                    }

                    // F4: Toggle Shadow Quality
                    if key == KeyCode::F4 {
                        self.shadow_quality_high = !self.shadow_quality_high;
                        info!(
                            "Shadow Quality: {}",
                            if self.shadow_quality_high {
                                "HIGH"
                            } else {
                                "LOW"
                            }
                        );
                        return;
                    }

                    // F5: Hot reload assets
                    if key == KeyCode::F5 {
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

                    // G: Toggle Interaction Mode
                    if key == KeyCode::KeyG {
                        self.interaction_mode = match self.interaction_mode {
                            InteractionMode::Block => InteractionMode::Sphere,
                            InteractionMode::Sphere => InteractionMode::Block,
                        };
                        info!("Interaction Mode: {:?}", self.interaction_mode);
                        return;
                    }

                    // E: Toggle block picker
                    if key == KeyCode::KeyE {
                        if self.picker.visible {
                            self.close_picker();
                        } else {
                            self.open_picker();
                        }
                        return;
                    }

                    // 1-9: Hotbar slot selection
                    let slot = match key {
                        KeyCode::Digit1 => Some(0),
                        KeyCode::Digit2 => Some(1),
                        KeyCode::Digit3 => Some(2),
                        KeyCode::Digit4 => Some(3),
                        KeyCode::Digit5 => Some(4),
                        KeyCode::Digit6 => Some(5),
                        KeyCode::Digit7 => Some(6),
                        KeyCode::Digit8 => Some(7),
                        KeyCode::Digit9 => Some(8),
                        _ => None,
                    };
                    if let Some(s) = slot {
                        self.hotbar.selected = s;
                        return;
                    }
                }

                match state {
                    ElementState::Pressed => self.input.key_down(key as u32),
                    ElementState::Released => self.input.key_up(key as u32),
                }
            }
            WindowEvent::MouseInput {
                button,
                state: ElementState::Pressed,
                ..
            } => {
                if self.picker.visible {
                    // Click in picker
                    if button == MouseButton::Left {
                        if let Some(registry) = &self.registry {
                            if let Some(renderer) = &self.renderer {
                                if let Some(window) = &self.window {
                                    let size = window.inner_size();
                                    if let Some(block_id) = renderer.picker_hit_test(
                                        self.mouse_pos.0,
                                        self.mouse_pos.1,
                                        size.width as f32,
                                        size.height as f32,
                                        registry.block_count() as usize,
                                    ) {
                                        self.hotbar.slots[self.hotbar.selected] = block_id;
                                        self.close_picker();
                                    }
                                }
                            }
                        }
                    }
                } else if !self.input.cursor_grabbed {
                    self.grab_cursor();
                } else {
                    match button {
                        MouseButton::Left => self.left_click_pending = true,
                        MouseButton::Right => self.right_click_pending = true,
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if self.input.cursor_grabbed && !self.picker.visible {
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 40.0,
                    };

                    if self.input.is_active(Action::Sprint) {
                        self.brush_dist = (self.brush_dist + scroll).clamp(2.0, 100.0);
                    } else {
                        if scroll > 0.0 {
                            self.brush_size = (self.brush_size + 1).min(10);
                        } else if scroll < 0.0 {
                            self.brush_size = self.brush_size.saturating_sub(1).max(1);
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

                // CPU raycast for block targeting
                self.update_targeted_block();

                // Process pending clicks
                self.handle_block_interaction();

                if let (Some(gpu), Some(renderer), Some(world), Some(window), Some(registry)) = (
                    &self.gpu,
                    &mut self.renderer,
                    &mut self.world,
                    &self.window,
                    &self.registry,
                ) {
                    // Stream new chunks as player moves
                    if world.update_view(gpu, self.camera.position) {
                        world.request_readback(gpu, false);
                    }

                    // Poll for completed async chunk readback
                    world.poll_readback(gpu, &mut self.voxels_cpu);

                    // Physics Pass
                    {
                        let mut encoder =
                            gpu.device()
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
                    let selected_block = self
                        .targeted_block
                        .as_ref()
                        .map(|h| [h.grid_pos.x, h.grid_pos.y, h.grid_pos.z]);

                    let brush_pos_radius =
                        if matches!(self.interaction_mode, InteractionMode::Sphere) {
                            let target =
                                self.camera.position + self.camera.forward() * self.brush_dist;
                            // Snap to grid center for consistency with voxel grid
                            let snapped = target.floor() + 0.5;
                            ara_core::glam::Vec4::new(
                                snapped.x,
                                snapped.y,
                                snapped.z,
                                self.brush_size as f32,
                            )
                        } else {
                            ara_core::glam::Vec4::ZERO
                        };

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
                        selected_block,
                        world.world_origin(),
                        prev_view_proj_unjittered,
                        curr_view_proj_unjittered,
                        brush_pos_radius,
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

                    let wo = world.world_origin_ivec3();
                    let ui_ctx = UiContext {
                        screen_width: width,
                        screen_height: height,
                        debug_visible: self.debug_hud_visible,
                        fps: self.display_fps,
                        frame_time_ms: self.display_frame_time,
                        camera_pos: self.camera.position.into(),
                        world_origin: [wo.x, wo.y, wo.z],
                        aa_mode_name: aa_mode_name(renderer.aa_mode()),
                        hotbar: &self.hotbar,
                        registry,
                        picker_visible: self.picker.visible,
                        brush_size: self.brush_size,
                        interaction_mode: match self.interaction_mode {
                            InteractionMode::Block => "Block",
                            InteractionMode::Sphere => "Sphere",
                        },
                    };

                    match renderer.render(gpu, &uniforms, &lights, &ui_ctx) {
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
                self.fps_frame_count += 1;
                let elapsed = self.fps_update_time.elapsed().as_secs_f32();
                if elapsed >= 0.5 {
                    self.display_fps = self.fps_frame_count as f32 / elapsed;
                    self.display_frame_time = elapsed * 1000.0 / self.fps_frame_count as f32;
                    if let Some(window) = &self.window {
                        window.set_title(&format!(
                            "{WINDOW_TITLE} | {:.0} FPS ({:.1} ms)",
                            self.display_fps, self.display_frame_time
                        ));
                    }
                    self.fps_frame_count = 0;
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
