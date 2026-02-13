  **Stage 1: Hierarchical DDA**
  Implement hierarchical (2-level) DDA raytracing on the existing fixed 512³ voxel grid.
  **COMPLETE**

  ---
  **Stage 2: Toroidal Ring Buffer**
  Add toroidal (wrapping) addressing to the 512³ voxel atlas so the grid can represent any region of infinite world space.
  **COMPLETE**

  ---
  **Stage 3: GPU Terrain Generation + World Streaming**
  Add GPU-side terrain generation and a WorldManager that streams chunks as the player moves through the infinite world.
  Context
  **COMPLETE**

  ---
  **Stage 4: Physics Pass Placeholder**
  Add a no-op physics simulation compute pass as a pipeline stub for future implementation.
  Context


  The engine needs a physics simulation pass (sand/water/gravity) that runs between terrain gen and rendering. For now, implement the pipeline infrastructure with a shader that does

  nothing, so the dispatch flow is established.
  What to implement


  1. Physics compute shader (assets/shaders/physics_sim.wgsl)
  - Bindings: voxels_src (read), voxels_dst (read_write), dirty_chunks (read_write)
  - @compute @workgroup_size(4, 4, 4) fn simulate() — empty body, just return

  - This establishes the ping-pong pattern: read from buffer A, write to buffer B


  2. Physics pipeline in WorldManager


  - Create the compute pipeline + bind group layout + bind group

  - Add dispatch_physics(encoder) method that dispatches 0 workgroups (pipeline exists but does no work)
  - Call it from main.rs each frame after update_view(), before rendering


  3. Ping-pong buffer swap


  - WorldManager already owns voxel_buf_a and voxel_buf_b

  - After physics dispatch, swap which buffer is "current" for rendering

  - For now with 0 workgroups dispatched, buffer A stays current


  What NOT to change


  - Do NOT implement actual physics logic

  - Do NOT modify rendering or lighting


  Verification


  1. cargo build — no errors

  2. cargo run — identical to Stage 4 (physics is a no-op)
  3. No GPU validation errors in the console 