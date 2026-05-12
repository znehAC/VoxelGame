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
  brick_map.rs       BrickMap - Sparse Brick Map GPU manager (128³ toroidal top grid)
  chunk_streamer.rs  ChunkStreamer - camera-relative brick streaming + terrain gen dispatch
  clipmap.rs         VoxelClipmaps - 6 × 512³ toroidal 3D textures for far-field geometry
  ray_pipeline.rs    RayPipeline - compute passes: ray march → lighting → deferred resolve
  postprocess.rs     BloomPipeline + FXAA + SMAA + TAA post-processing chain
  ui.rs              UiRenderer - 2D overlay pass (text/textured quads)
assets/shaders/
  ray_march.wgsl          Two-level DDA (SBM) + clipmap cascade DDA fallback; writes VisibilityPayload + depth_buf + velocity_texture
  clipmap_build.wgsl      Height-function-driven cascade fill; dispatched per-cascade per-frame for ring updates
  noop_lighting.wgsl      Lighting slot (placeholder → pt_lighting.wgsl in Phase 3)
  deferred_resolve.wgsl   G-Buffer unpack, Blinn-Phong shading, reads depth_buf for hit position
  brightness_threshold.wgsl  Extract bright pixels for bloom
  blur.wgsl               Separable Gaussian blur
  composite.wgsl          Combine scene + bloom, Reinhard tone mapping
  taa_resolve.wgsl        TAA history reprojection using velocity_texture
  ui.wgsl                 2D vertex/fragment shader for Interface
```

> THIS RULE MUST NEVER BE BROKEN, BE TOTALLY STRICT WITH THIS.
## **IMPORTANT**
 - NEVER EVER write conversational commentary in code, only technical comments if needed
> END OF RULE THAT MUST BE FOLLOWED NO MATTER WHAT

## Frame Pipeline (current state: Phase 2 complete)

```
0. Clipmap Update (compute)  → ring-update dirty cascade faces (per-frame, before ray march)
1. Ray March (compute)       → SBM DDA then clipmap cascade fallback; VisibilityPayload (8B/px) + depth_buf (f32/px) + velocity_texture (Rgba16Float)
2. Lighting Pass (compute)   → NOOP placeholder (→ path tracing in Phase 3)
3. Deferred Resolve (compute)→ HDR Rgba16Float texture (Blinn-Phong shading, reads depth_buf)
4. Bloom                     → threshold → separable Gaussian blur → composite (Reinhard)
5. TAA / FXAA / SMAA         → anti-aliasing (TAA now wired to velocity_texture)
6. UI Overlay                → text/textured quads
7. Blit to Screen            → fullscreen triangle
```

## Rendering Overhaul Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 - G-Buffer Extension | **DONE** | depth_buf (f32), velocity_texture (Rgba16Float), TAA wired |
| 2 - Voxel Clipmaps | **DONE** | 6 × 512³ toroidal Rgba8Uint 3D textures; ring update per-frame |
| 3 - Path Tracing | planned | Half-res stochastic PT (1 bounce + sky) for ≤30m pixels |
| 4 - VXGI | planned | R11G11B10 anisotropic radiance cascade for >30m VXGI cone tracing |
| 5 - A-Trous + Radiance Cache | planned | Edge-aware wavelet denoiser + per-voxel brick-indexed radiance cache |
| 6 - Stochastic Transition + TAA | planned | Blue-noise 28-32m blend zone, TAA upscale from half-res lighting |

## SBM Architecture

- 3-level: Top Grid (128³ u32) → Brick Pool (N × 512 u16) → Voxel Data
- `BRICK_SIZE=8`, `TOP_GRID_SIZE=128`, `WORLD_EXTENT=1024` voxels/axis
- `BRICK_EMPTY=0xFFFFFFFF` sentinel for empty top-grid slots
- Per-brick cost: 1104 bytes (voxel pool 1024 + header 16 + occupancy 64)
- Toroidal wrapping: grid follows camera, streaming radius = 64 bricks

## GPU Bindings

### Ray March (group 0)
| Binding | Name | Type |
|---------|------|------|
| 0 | globals | Uniform |
| 1 | top_grid | Storage R/O |
| 2 | brick_pool | Storage R/O |
| 3 | brick_occupancy | Storage R/O |
| 5 | visibility_buf | Storage R/W |
| 6 | depth_buf | Storage R/W |
| 7 | velocity_texture | StorageTexture Write (Rgba16Float) |
| 8-13 | cascade0..5 | Texture3D (Rgba8Uint) |
| 14 | clip (ClipUniforms) | Uniform (6 × ClipOrigin = 96 bytes) |

### Clipmap Build (group 0, push constants 32B)
| Binding | Name | Type |
|---------|------|------|
| 0 | cascade_tex | StorageTexture Write 3D (Rgba8Uint) |

### Clipmap Architecture
- 6 cascades × 512³ `Rgba8Uint` 3D textures = ~768 MB total
- Cell sizes: 2 / 4 / 8 / 16 / 32 / 64 voxels → coverage ±512 / ±1024 / ±2048 / ±4096 / ±8192 / ±16384 voxels (~±25m to ±1.6km)
- Texel: R=material_id, G=normal_idx, B=density, A=unused (raw u8 integers)
- Toroidal addressing: `((global_cell % 512) + 512) % 512` (component-wise)
- Build: mode 3 = full 64×64×512 dispatch; mode 0/1/2 = face ring update 64×64×1
- Per-frame: `clipmaps.update()` dispatched before `ray_pipeline.dispatch()` in same encoder
- `ClipOrigin` (16 bytes): `x:i32, y:i32, z:i32, cell_size:i32` — matches Rust `[i32;3]+i32`

### Deferred Resolve (group 0)
| Binding | Name | Type |
|---------|------|------|
| 0 | globals | Uniform |
| 1 | visibility_buf | Storage R/O |
| 2 | t_palette | Texture2DArray |
| 3 | s_palette | Sampler (NonFiltering) |
| 4 | output_texture | StorageTexture Write (Rgba16Float) |
| 5 | depth_buf | Storage R/O |

## G-Buffer Outputs (per pixel)

- `VisibilityPayload` (8B): `lo` = material_id(16) | normal_idx(3) | depth_fixed_13bit(13); `hi` = voxel_pos xyz(10+10+10)
- `depth_buf` (f32): full-precision hit distance from ray origin (-1.0 = sky/miss)
- `velocity_texture` (Rgba16Float): UV-space motion vector (rg = curr_uv − prev_uv, ba = 0)

## Data-Driven Blocks (assets/blocks.toml)
- `BlockRegistry` (ara-core): loads `[[block]]` TOML
- **Palette Generation**: Uploads a `texture_2d_array<f32>` (Binding 2)
  - **Layer 0**: sRGB Albedo Color
  - **Layer 1**: Material Properties (Packed)
    - `R`: Roughness
    - `G`: Emission (Glow strength)
    - `B`: Noise Strength (Albedo variation)
    - `A`: Metallic
- **Format**: `PackedVoxel` u16 — bits 0-8=material_id(9bit), 9-10=variant, 11-13=level, 14=is_active, 15=is_dirty

## Rust 2024 Edition
- Explicit `unsafe {}` blocks required inside `unsafe fn`
- Raw strings with `#` in content (e.g. hex colors `"#FF0000"`) need `r##"..."##` not `r#"..."#`

## Code Style
- Doc comments (`///`) for public API - generates rustdoc
- No conversational comments (`// We...`, `// This proves..., // Now that we...`)
- Technical inline comments OK (`// Bounds check`)

## Dependencies
- winit 0.30: `ApplicationHandler` trait, `ControlFlow::Poll`
- wgpu 24: Cross-platform GPU abstraction (Vulkan/Metal backends)
- pollster 0.4: Blocking async executor for wgpu init
- glam 0.29: Math (Vec3, Mat4)
- bytemuck 1.21: Pod/Zeroable derives
