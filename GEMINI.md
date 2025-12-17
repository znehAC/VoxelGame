# Voxel Powder Simulation

## Project Overview
This project is a high-performance Voxel Engine and Powder Simulation built with **Godot 4.5.1 (C# / .NET 8.0)**. It features an infinite procedural world, dynamic LOD (Level of Detail) using an Octree system, and GPU-accelerated terrain generation via Compute Shaders.

The engine is designed to handle large-scale voxel modifications and physics (powder simulation), though the current implementation focuses on the robust terrain generation and meshing pipeline.

### Key Technologies
*   **Godot 4.5.1 (.NET Edition):** Core engine and rendering.
*   **C# / .NET 8.0:** Game logic, threading, and data management.
*   **Compute Shaders (GLSL):** procedural terrain generation (Noise) runs entirely on the GPU.
*   **RenderingDevice (Local):** Usage of local `RenderingDevice` instances to allow thread-safe compute shader execution on background threads.
*   **Greedy Meshing:** Optimized mesh generation to reduce vertex count.
*   **Octree LOD:** Dynamic level-of-detail system managing chunk loading and visibility based on player distance.

## Architecture

### 1. World & Data Management
*   **`World.cs`:** The central manager. It initializes the Octree, handles the job scheduler, and processes the main loop (chunk state transitions, job results).
    *   *Recent Fix:* Relaxed neighbor checking to allow edge chunks to mesh (treating missing neighbors as Air) and reduced strict safety brakes to prevent stalling.
*   **`Chunk.cs`:** Represents a single 32x32x32 voxel volume. Manages its own state (`Idle` -> `AwaitingData` -> `Ready`), mesh instance, and physics body.
*   **`ChunkOctree.cs`:** Manages the spatial partitioning. It handles splitting (higher detail) and merging (lower detail) of chunks based on distance.
    *   *Recent Fix:* Strict visibility propagation. The Octree explicitly sets `isVisible` on chunks to prevent Z-fighting between Parent (LOD N) and Children (LOD N-1) chunks during transitions.

### 2. Generation Pipeline (Multithreaded)
*   **`VoxelJobScheduler.cs`:** Manages a pool of worker threads.
    *   **Compute Threads:** Dedicated threads with their own `RenderingDevice` context to generate voxel data.
    *   **Mesh Threads:** CPU threads running the Greedy Meshing algorithm.
*   **`TerrainGenerator.glsl`:** The Compute Shader. Currently configured to generate a **1-meter (10x10x10 voxel)** blocky terrain with a checkerboard pattern (Stone/Dirt) for scale verification.
*   **`TerrainCompute.cs`:** C# wrapper for the Compute Shader. Handles buffer creation (StorageBuffers), dispatch, and data retrieval.

### 3. Meshing
*   **`GreedyMesher.cs`:** Implements the greedy meshing algorithm to combine adjacent faces of the same type into single quads, significantly improving rendering performance.
*   **`VoxelTypes.cs`:** Defines block types (Air, Dirt, Stone, Grass, Water) and their properties (Color, Solidity).

## Building and Running

### Prerequisites
*   **Godot Engine 4.5.1 (.NET version)**
*   **.NET SDK 8.0**

### Commands
*   **Build Project:**
    ```bash
    dotnet build
    ```
*   **Run Project:**
    Open the `project.godot` file in the Godot Editor and press Play (F5), or use the Godot CLI (if configured).

## Development Conventions

*   **Thread Safety:**
    *   Godot Nodes (`Node3D`, `MeshInstance3D`) are **not** thread-safe and must only be manipulated on the Main Thread.
    *   `RenderingDevice` is thread-safe **only** if using a local instance created via `RenderingServer.CreateLocalRenderingDevice()`, not the global singleton.
*   **Visual Debugging:**
    *   The terrain is currently set to "Debug Mode" (1m blocks). To restore natural terrain, modify `TerrainGenerator.glsl`.
    *   Wireframe mode can be toggled (input action: `toggle_debug_wireframe`).
*   **Coordinate System:**
    *   World uses standard Godot 3D coordinates (Y-up).
    *   Chunk Size is **32**.
    *   1 Unit = 1 Voxel (0.1m scale effectively, but visually debugged as 1m blocks).

## Current State & Known Issues
*   **Scale:** Terrain generation is currently hardcoded to 10x10x10 blocks to visualize 1-meter scaling.
*   **Visibility:** Z-fighting has been resolved via the strict Octree visibility logic.
*   **Performance:** A "Safety Brake" in `World.cs` pauses loading if FPS drops below 10 or memory exceeds 4.5GB.
