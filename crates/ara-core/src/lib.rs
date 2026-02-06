//! Ara core — shared voxel types and utilities.
//!
//! This crate contains shared types used by both simulation and rendering.
//! No GPU dependencies.

pub mod input;
pub mod registry;
pub mod types;
pub mod voxel;

pub use bytemuck;
pub use glam;

pub use input::{Action, InputManager};
pub use registry::BlockRegistry;
pub use types::{CameraPushConstants, GlobalUniforms, InputState};
pub use voxel::PackedVoxel;

/// Voxel material ID.
pub type VoxelId = u16;

/// Air voxel (empty space).
pub const VOXEL_AIR: VoxelId = 0;
