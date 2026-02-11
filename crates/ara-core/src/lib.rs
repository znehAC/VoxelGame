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
pub use raycast::{dda_raycast, RayHit};
pub use registry::BlockRegistry;
pub use types::{CameraPushConstants, GlobalUniforms, InputState, LightBuffer, PointLight, TaaUniforms};
pub use voxel::PackedVoxel;

/// Voxel material ID.
pub type VoxelId = u16;

/// Air voxel (empty space).
pub const VOXEL_AIR: VoxelId = 0;

/// Grid size (cubic).
pub const GRID_SIZE: u32 = 64;

/// Max raymarching steps.
pub const MAX_STEPS: u32 = 256;
