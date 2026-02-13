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


  Stages 1-2 established hierarchical DDA with toroidal addressing on a fixed atlas. Now we move voxel generation to the GPU and add runtime chunk streaming.
  What to implement


  1. Terrain generation compute shader (assets/shaders/terrain_gen.wgsl)
  - Input: push constant ChunkParams { offset_x, offset_y, offset_z, pad } (world-space chunk origin)
  - Bindings: voxels (read_write), dirty_chunks (read_write), chunk_occupancy (read_write)
  - Workgroup size: @workgroup_size(8, 8, 8), dispatch (4, 4, 4) per chunk

  - For each voxel in the 32³ chunk:
    - Compute world position = offset + local_id

    - Generate terrain height using FBM noise (3-4 octaves)
    - Assign materials: stone below height-5, dirt below height-1, grass below height, air above

    - Write packed voxel to atlas using toroidal wrap() addressing

    - If solid: set chunk_occupancy[ci] = 1 and dirty_chunks[ci] = 1


  2. WorldManager (src/world.rs)
  - Owns: voxel_buf_a, voxel_buf_b (ping-pong), dirty_buf, occupancy_buf

  - Owns: terrain gen compute pipeline + bind group

  - world_origin: IVec3 — current center of loaded region

  - generate_initial_chunks(gpu): clear occupancy buffer, dispatch terrain gen for all 16³ chunks

  - update_view(gpu, encoder, player_pos):
    - Compute new_origin using floor division: floor_div(pos as i32, cs) * cs

    - If origin changed, iterate all 16³ chunk slots; for each world chunk NOT in the old loaded range, clear its occupancy slot (encoder.clear_buffer) and dispatch terrain gen

    - No per-frame cap — regenerate all boundary chunks immediately


  3. Remove CPU voxel data


  - Delete the CPU-side voxel array and upload from main.rs

  - Renderer::new() takes &wgpu::Buffer references from WorldManager instead of owned voxel data

  - Camera starts at a position above the terrain surface (e.g. y=100)
  4. Integration in main.rs


  - Create WorldManager before Renderer

  - Call world.generate_initial_chunks() at startup

  - Each frame: call world.update_view(gpu, encoder, camera.position) before rendering

  - Pass world.world_origin() to GlobalUniforms


  Important: floor division


  Rust integer division truncates toward zero. Use proper floor division for negative coordinates:
  fn floor_div(a: i32, b: i32) -> i32 {
      let d = a / b;
      let r = a % b;
      if (r != 0) && ((r ^ b) < 0) { d - 1 } else { d }
  }
  What NOT to change


  - Do NOT modify the hierarchical DDA or toroidal addressing (Stages 1-2)
  - Do NOT modify the light propagation system yet

  - Do NOT add physics simulation


  Verification


  1. cargo build — no errors

  2. cargo run — procedural terrain visible, no black holes or missing chunks

  3. Walk in all directions — new terrain streams in smoothly

  4. Walk to negative coordinates — terrain remains seamless (no misalignment)
  5. Look up — should see sky, NOT a duplicate terrain layer


  ---
  **Stage 4: Fix Light Propagation for 512³ Atlas**
  Adapt the light propagation system to work correctly with the 512³ toroidal atlas and world streaming.
  Context


  The light propagation system was designed for a 64³ grid. With the 512³ atlas and toroidal addressing from previous stages, it needs updates to:
  - Operate on the full 512³ atlas (or a downsampled representation)
  - Use toroidal wrapping for neighbor lookups

  - Inject sky light at the correct world-relative position

  - Handle the sun path check within the loaded region bounds


  Current state


  - LightPropagation in src/light.rs: ping-pong between two 3D textures, N compute iterations

  - light_propagate.wgsl: reads voxels + palette, propagates light to 26 neighbors with occlusion, injects sky + sun bounce

  - Light volume is sampled in the raytracer via sample_light() using textureLoad


  What to implement


  1. Scale light textures to 512³ or choose a resolution strategy


  Option A: Full 512³ R32Uint textures (2× 512MB — expensive but accurate)
  Option B: Downsampled (e.g. 128³ or 256³) with coordinate scaling in sampling

  Recommend Option A for correctness first, optimize later.
  2. Toroidal wrapping in light shader


  All neighbor lookups in light_propagate.wgsl must use wrap():
  fn wrap(c: i32) -> i32 { return ((c % AS) + AS) % AS; }
  Apply to is_opaque_at(), read_light(), and voxel_index().
  3. Sky light injection using world_origin


  Replace if gid.y == ATLAS_SIZE - 1u with:
  let top_world_y = i32(globals.world_origin.y) + AS / 2 - 1;
  let top_atlas_y = u32(((top_world_y % AS) + AS) % AS);
  if gid.y == top_atlas_y { ... }
  4. Sun path check bounds


  The check_sun_path() function marches a ray toward the sun. It must not wrap around the torus. Either:
  - Cap the march distance to AS / 2 (256 voxels)
  - Or check if the ray has exited the loaded region and return true (reached sky)
  5. Dirty chunk optimization (optional)
  Only re-propagate light in chunks marked dirty by terrain gen. This is a performance optimization — skip if time is limited.
  What NOT to change


  - Do NOT modify the raytracer DDA or world streaming

  - Do NOT add physics


  Verification


  1. cargo build — no errors

  2. cargo run — terrain is lit (not all black)
  3. Sky light comes from above, underground is dark

  4. Emissive blocks (lava, etc.) cast colored light

  5. Walk around — lighting updates correctly as new chunks stream in


  ---
  **Stage 5: Physics Pass Placeholder**
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