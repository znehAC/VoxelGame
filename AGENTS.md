# Ara Voxel Engine / Turi Game

## Project Overview

This is a **real-time voxel engine** written in Rust, featuring GPU-accelerated raytracing, global illumination, and post-processing effects. The project consists of:

- **Ara** - The core voxel engine (library + application)
- **Turi** - The game client application (main executable)

The engine uses a **64³ voxel grid** with DDA raymarching, cellular automata lighting, and PBR shading.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Rust 2024 Edition (requires Rust 1.85+) |
| GPU API | wgpu 24 (cross-platform: Vulkan, Metal, DX12) |
| Windowing | winit 0.30 |
| Math | glam 0.29 |
| Serialization | serde + toml |
| Image Loading | image 0.25 |
| Async | pollster 0.4 (blocking executor for GPU init) |

---

## Project Structure

```
voxel-engine/
├── Cargo.toml              # Workspace definition
├── Cargo.lock              # Dependency lock file
├── src/                    # Main application (Turi client)
│   ├── main.rs             # App + winit event loop
│   ├── gpu.rs              # GpuContext - wgpu device/queue wrapper
│   ├── renderer.rs         # Renderer - surface presentation, HDR pipeline
│   ├── camera.rs           # FpsCamera - first-person camera
│   ├── light.rs            # LightPropagation - GPU light propagation
│   ├── postprocess.rs      # BloomPipeline + FXAA + SMAA
│   ├── postprocess/
│   │   └── smaa.rs         # SMAA anti-aliasing implementation
│   ├── ui.rs               # UiSystem - 2D overlay rendering
│   └── assets.rs           # Asset loading utilities
├── crates/
│   └── ara-core/           # Core library (pure data, no GPU deps)
│       ├── src/
│       │   ├── lib.rs      # Crate exports
│       │   ├── voxel.rs    # PackedVoxel type
│       │   ├── types.rs    # GPU uniform types (GlobalUniforms, etc.)
│       │   ├── registry.rs # BlockRegistry - TOML-driven block definitions
│       │   ├── input.rs    # InputManager - action-mapped input
│       │   └── raycast.rs  # CPU-side DDA raycasting
│       └── Cargo.toml
└── assets/
    ├── blocks.toml         # Block definitions (color, roughness, emission)
    ├── shaders/            # WGSL shaders
    │   ├── voxel_raytracer.wgsl   # Main raymarcher + PBR
    │   ├── light_propagate.wgsl   # Cellular automata GI
    │   ├── brightness_threshold.wgsl
    │   ├── blur.wgsl
    │   ├── composite.wgsl         # HDR + bloom + tone mapping
    │   ├── fxaa.wgsl
    │   ├── smaa.wgsl
    │   └── ui.wgsl
    └── textures/
        └── smaa/           # SMAA lookup textures
            ├── AreaTex.png
            └── SearchTex.png
```

---

## Build and Run Commands

```bash
# Build all crates
cargo build

# Run the game client (windowed)
cargo run

# Fast type checking
cargo check

# Run tests
cargo test

# Build release
cargo build --release
```

---

## Architecture Details

### Voxel Data Format

- **PackedVoxel**: 32-bit packed format
  - Bits 0-15: Material ID (u16) - up to 65536 block types
  - Bits 16-23: State/metadata (u8)
  - Bits 24-31: Flags/lighting (u8)
- **Grid Size**: 64³ voxels (constant `GRID_SIZE`)
- **Storage**: GPU SSBO (Storage Buffer)

### Rendering Pipeline

1. **Light Propagation** (Compute) - Cellular automata on 64³ RGBA16F 3D texture
2. **Raytrace Pass** - DDA raymarcher → HDR Rgba16Float texture
3. **Brightness Threshold** - Extract bright pixels for bloom (half-res)
4. **Gaussian Blur** - Separable H+V blur on bloom texture
5. **Composite** - HDR + bloom + Reinhard tone mapping → SDR
6. **Anti-Aliasing** - FXAA or SMAA (toggleable at runtime)
7. **UI Overlay** - 2D quads drawn on top

### Lighting System

- **Sun**: Directional light with hard shadows (raytraced)
- **Voxel GI**: Cellular automata propagation from emissive blocks + sun injection
- **Dynamic Lights**: Up to 16 point lights with optional shadows
- **AO**: Vertex-style ambient occlusion calculated in shader

### Input Controls

| Key | Action |
|-----|--------|
| W/S/A/D | Move |
| Space | Fly up |
| C | Fly down |
| Shift | Sprint |
| Mouse | Look around |
| Left Click | Break block |
| Right Click | Place block |
| Escape | Release cursor |
| F2 | Cycle AA mode (None → FXAA → SMAA) |
| F4 | Cycle SMAA debug mode |
| F5 | Hot reload assets (blocks.toml) |

---

## Code Style Guidelines

### Comments
- **Doc comments (`///`)** for public API - generates rustdoc
- **No conversational comments** (`// We...`, `// This proves...`)
- **Technical inline comments OK** (`// Bounds check`)

### Rust 2024 Edition Specifics
- Explicit `unsafe {}` blocks required inside `unsafe fn`
- Raw strings with `#` in content need `r##"..."##` not `r#"..."#`

### Naming Conventions
- Types: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Shader files: `snake_case.wgsl`

---

## Testing

Unit tests are embedded in source files using `#[cfg(test)]`:

```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p ara-core
```

Key test modules:
- `voxel.rs`: Layout, packing, field manipulation tests
- `types.rs`: GPU uniform buffer layout verification
- `registry.rs`: Block loading, texture generation tests
- `raycast.rs`: DDA raycast correctness tests

---

## Asset System

### blocks.toml Format

```toml
[[block]]
id = "ara:stone"
name = "Stone"
color = "#808080"
roughness = 0.9
noise = 0.7

[[block]]
id = "ara:lava"
name = "Lava"
color = "#FF5722"
emission = 1.0  # Makes block glow
roughness = 0.3
metallic = 0.0
```

### Block Properties
- `color`: Hex color (sRGB)
- `roughness`: 0.0-1.0 (affects specular)
- `emission`: 0.0-1.0 (makes block a light source)
- `metallic`: 0.0-1.0 (F0 for Fresnel)
- `noise`: 0.0-1.0 (albedo variation strength)

### Palette Texture
- 256×256 pixels, 2 layers
- **Layer 0**: sRGB albedo colors
- **Layer 1**: Packed material properties (R=roughness, G=emission, B=noise, A=metallic)

---

## Development Conventions

### Adding New Blocks
1. Edit `assets/blocks.toml`
2. Add `[[block]]` entry with unique ID
3. Press F5 in-game to hot reload

### Shader Development
- Shaders are loaded at runtime (except `light_propagate.wgsl` which is embedded)
- Edit `assets/shaders/voxel_raytracer.wgsl` for visual changes
- No hot reload for shaders yet - restart required

### GPU Resource Management
- `GpuContext` wraps Instance/Adapter/Device/Queue
- Buffers/textures recreated on window resize
- Push constants used for per-frame data (where applicable)

---

## Dependencies Between Crates

```
ara (main crate)
├── ara-core (workspace dependency)
│   ├── glam
│   ├── bytemuck
│   ├── serde
│   └── toml
├── winit
├── wgpu
├── pollster
├── log
├── env_logger
├── bytemuck
└── image
```

**Important**: `ara-core` must NOT depend on any GPU crates (wgpu, winit). It contains pure data types used by both simulation and rendering.

---

## Performance Considerations

- **Fixed 64³ grid**: Optimized for this size
- **HDR rendering**: Rgba16Float intermediate target
- **Half-res bloom**: Performance/quality tradeoff
- **Light iterations**: Configurable (default 25 compute passes)
- **AA options**: None (fastest), FXAA (balanced), SMAA (quality)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails with edition error | Update to Rust 1.85+ |
| GPU not found | Check Vulkan/Metal drivers |
| Assets not loading | Ensure working directory contains `assets/` folder |
| Window black | Check GPU memory - HDR textures are large |

---

## License

MIT License (see workspace `Cargo.toml`)
