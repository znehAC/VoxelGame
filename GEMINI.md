# Turi (game) / Ara (engine)

## Build
- `cargo build` - build all crates
- `cargo run` - run client with window
- `cargo check` - fast type checking

## Architecture

```

crates/ara-core/     Pure data types (PackedVoxel, BlockRegistry, InputState) + serde/toml
src/
main.rs            App + winit event loop (ControlFlow::Poll game loop)
gpu.rs             GpuContext - headless-first Device/Queue (wgpu)
renderer.rs        Renderer - surface presentation, HDR pipeline (wgpu)
camera.rs          FpsCamera - first-person camera (pure glam math)
light.rs           LightPropagation - flood fill via compute ping-pong
postprocess.rs     BloomPipeline - threshold/blur/composite passes
ui.rs              UiRenderer - 2D overlay pass (text/textured quads)
assets/shaders/
voxel_raytracer.wgsl       DDA raymarcher + PBR shading + Hard Shadows
light_propagate.wgsl       Cellular automata GI + Sun Injection (compute)
brightness_threshold.wgsl  Extract bright pixels for bloom
blur.wgsl                  Separable Gaussian blur
composite.wgsl             Combine scene + bloom, Reinhard tone mapping
ui.wgsl                    2D vertex/fragment shader for Interface

```

> THIS RULE MUST NEVER BE BROKEN, BE TOTALLY STRICT WITH THIS.
## **IMPORTANT**
 - NEVER EVER write conversational commentary in code, only technical comments if needed
> END OF RULE THAT MUST BE FOLLOWED NO MATTER WHAT

## GPU Context (src/gpu.rs)
- `GpuContext`: Wraps wgpu Instance/Adapter/Device/Queue
- `create_instance()` → wgpu::Instance (Vulkan + Metal backends)
- `new_headless()` → headless compute (no surface)
- `from_instance(instance, Option<&Surface>)` → full init with optional surface compat

## Renderer (src/renderer.rs)
- Owns surface + config, LightPropagation, BloomPipeline
- `new(gpu, surface, width, height, palette, voxels)` → configure surface, init subsystems
- `resize(gpu, width, height)` → reconfigure surface + post-process textures
- `render(gpu, uniforms)` → light propagate → raytrace to HDR → bloom → composite → ui → present
- Handles SurfaceError::Lost (reconfigure) and OutOfMemory (exit)

### Frame pipeline
1. `LightPropagation::propagate()` — N compute passes (ping-pong flood fill)
2. Raytrace pass → HDR Rgba16Float texture (PBR + voxel light sampling)
3. Brightness threshold → half-res bloom texture
4. Separable Gaussian blur (H+V) on bloom
5. Composite: HDR + bloom → swapchain (Reinhard tone mapping)
6. UI Overlay: Textured/Text quads drawn over final composite

## Light Propagation (src/light.rs)
- **Hybrid System**: Cellular automata + Raytraced Sun Injection
- Uses two 64^3 Rgba16Float 3D textures (ping-pong)
- **Sun Injection**: Air voxels adjacent to solid surfaces trace rays towards the sun; if unblocked, they become light sources (simulating first bounce).
- **Propagation**: 
  - Seeds from emissive blocks and sun-injected air.
  - Spreads to neighbors (Face/Edge/Corner) with specific decay factors.
  - Blocked by opaque voxels (prevents light leak).
- Raytracer samples result via `textureSampleLevel` (trilinear) at `pos + normal * 0.1`.

## Post-Processing (src/postprocess.rs)
- `BloomPipeline`: threshold → blur → composite pipeline
- HDR scene rendered to Rgba16Float texture (full resolution)
- Bloom at half resolution for performance and natural softness
- Composite pass: additive bloom + Reinhard tone mapping
- Push constants for threshold, blur direction, bloom intensity, exposure

## Data-Driven Blocks (assets/blocks.toml)
- `BlockRegistry` (ara-core): loads `[[block]]` TOML
- **Palette Generation**: Uploads a `texture_2d_array<f32>` (Binding 2)
  - **Layer 0**: sRGB Albedo Color
  - **Layer 1**: Material Properties (Packed)
    - `R`: Roughness
    - `G`: Emission (Glow strength)
    - `B`: Noise Strength (Albedo variation)
    - `A`: Metallic
- **Format**: `PackedVoxel` u32 (bits 0-15=material ID, 16-23=state, 24-31=flags)

## Rust 2024 Edition
- Explicit `unsafe {}` blocks required inside `unsafe fn`
- Raw strings with `#` in content (e.g. hex colors `"#FF0000"`) need `r##"..."##` not `r#"..."#`

## Code Style
- Doc comments (`///`) for public API - generates rustdoc
- No conversational comments (`// We...`, `// This proves...`)
- Technical inline comments OK (`// Bounds check`)

## Dependencies
- winit 0.30: `ApplicationHandler` trait, `ControlFlow::Poll`
- wgpu 24: Cross-platform GPU abstraction (Vulkan/Metal/DX12)
- pollster 0.4: Blocking async executor for wgpu init
- glam 0.29: Math (Vec3, Mat4)
- bytemuck 1.21: Pod/Zeroable derives