//! Ara core — shared voxel types and utilities.
//!
//! This crate contains shared types used by both simulation and rendering.
//! No GPU dependencies.

pub mod input;
pub mod morton;
pub mod raycast;
pub mod registry;

pub mod types;
pub mod visibility;
pub mod voxel;

pub use bytemuck;
pub use glam;

pub use input::{Action, InputManager};
pub use morton::{morton_decode, morton_encode};
pub use raycast::{RayHit, dda_raycast};
pub use registry::BlockRegistry;

pub use types::{
    CameraPushConstants, GlobalUniforms, InputState, LightBuffer, PointLight, TaaUniforms,
};
pub use visibility::{NormalIndex, VisibilityPayload};
pub use voxel::{BrickHeader, PackedVoxel};
pub use voxel::{brick_local_index, world_to_brick, world_to_local};

/// Voxel material ID.
pub type VoxelId = u16;

/// Air voxel (empty space).
pub const VOXEL_AIR: VoxelId = 0;

/// Voxel scale in meters (5cm).
pub const VOXEL_SCALE: f32 = 0.05;

/// Side length of a single brick in voxels.
pub const BRICK_SIZE: u32 = 8;

/// log2(BRICK_SIZE) — used for bit-shift addressing.
pub const BRICK_SHIFT: u32 = 3;

/// Total voxels per brick (8³ = 512).
pub const BRICK_VOLUME: u32 = 512;

/// Side length of the top-level brick grid.
pub const TOP_GRID_SIZE: u32 = 128;

/// Total entries in the top grid (128³ = 2,097,152).
pub const TOP_GRID_VOLUME: u32 = 2_097_152;

/// Total voxels per axis (TOP_GRID_SIZE * BRICK_SIZE = 1024).
pub const WORLD_EXTENT: u32 = 1024;

/// Sentinel value indicating an empty top-grid slot.
pub const BRICK_EMPTY: u32 = 0xFFFFFFFF;

/// Max raymarching steps for the outer (brick-level) DDA.
pub const MAX_STEPS: u32 = 512;


/// Pool-based memory budget derived from a total byte allocation.
#[derive(Debug, Clone, Copy)]
pub struct PoolBudget {
    pub total_bytes: u64,
}

impl PoolBudget {
    pub fn new(total_mb: u32) -> Self {
        Self {
            total_bytes: total_mb as u64 * 1024 * 1024,
        }
    }

    /// Per-brick cost in bytes: voxel pool (1024) + header (16) + occupancy (64) = 1104
    const BYTES_PER_BRICK: u64 = 1024 + 16 + 64;

    /// Top grid overhead: 128³ × 4 bytes = 8 MB
    const TOP_GRID_OVERHEAD: u64 = TOP_GRID_VOLUME as u64 * 4;

    pub fn max_bricks(&self) -> u32 {
        let available = self.total_bytes.saturating_sub(Self::TOP_GRID_OVERHEAD);
        let from_budget = (available / Self::BYTES_PER_BRICK) as u32;
        // Radiance pool (largest buffer) = max_bricks × 512 × 8 = max_bricks × 4096.
        // wgpu max buffer size is 1 GB → cap at 262,144 bricks.
        from_budget.min(262_144)
    }

    pub fn render_distance_voxels(&self) -> f32 {
        (self.max_bricks() as f32).cbrt() * BRICK_SIZE as f32
    }

    pub fn render_distance_meters(&self, voxel_m: f32) -> f32 {
        self.render_distance_voxels() * voxel_m
    }
}

/// GPU memory budget presets for the brick pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryBudget {
    Low,
    Medium,
    High,
}

impl MemoryBudget {
    /// Maximum number of bricks for this budget.
    pub fn max_bricks(self) -> u32 {
        self.pool_budget().max_bricks()
    }

    /// Voxel pool size in bytes (max_bricks × 512 × 2).
    pub fn voxel_pool_bytes(self) -> u64 {
        self.max_bricks() as u64 * BRICK_VOLUME as u64 * 2
    }

    pub fn pool_budget(self) -> PoolBudget {
        match self {
            Self::Low => PoolBudget::new(256),
            Self::Medium => PoolBudget::new(512),
            Self::High => PoolBudget::new(2048),
        }
    }
}
