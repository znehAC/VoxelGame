//! Ara core — shared voxel types and utilities.
//!
//! This crate contains shared types used by both simulation and rendering.
//! No GPU dependencies.

pub mod input;
pub mod raycast;
pub mod registry;
pub mod types;
pub mod voxel;

pub use bytemuck;
pub use glam;

pub use input::{Action, InputManager};
pub use raycast::{RayHit, dda_raycast};
pub use registry::BlockRegistry;
pub use types::{
    CameraPushConstants, GlobalUniforms, InputState, LightBuffer, PointLight, TaaUniforms,
};
pub use voxel::PackedVoxel;

/// Voxel material ID.
pub type VoxelId = u16;

/// Air voxel (empty space).
pub const VOXEL_AIR: VoxelId = 0;

/// Voxel grid size (cubic).
pub const GRID_SIZE: u32 = 512;

/// Chunk size for hierarchical DDA traversal.
pub const CHUNK_SIZE: u32 = 32;

/// Number of chunks per axis (GRID_SIZE / CHUNK_SIZE).
pub const CHUNKS_PER_AXIS: u32 = 16;

/// Light volume resolution (cubic, independent of voxel grid).
pub const LIGHT_GRID_SIZE: u32 = 128;

/// Max raymarching steps (512 * sqrt(3) ≈ 886).
pub const MAX_STEPS: u32 = 1024;
