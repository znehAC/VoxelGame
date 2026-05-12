//! 64-bit Visibility Buffer payload for the compute ray marcher.
//!
//! Each pixel in the visibility buffer stores a packed 64-bit value encoding
//! the ray-surface intersection result. A deferred resolve pass reads these
//! payloads to produce final shaded color.
//!
//! # Bit Layout (two u32 words)
//!
//! ## Word 0 (lo)
//! - Bits 0..15:  Material ID (u16, 0 = miss/sky)
//! - Bits 16..18: Normal axis (0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z)
//! - Bits 19..31: Depth (13 bits, fixed-point distance × 8)
//!
//! ## Word 1 (hi)
//! - Bits 0..9:   Voxel X within world (10 bits, 0..1023)
//! - Bits 10..19:  Voxel Y within world (10 bits, 0..1023)
//! - Bits 20..29:  Voxel Z within world (10 bits, 0..1023)
//! - Bits 30..31:  LOD level (2 bits, 0..3)

use bytemuck::{Pod, Zeroable};

/// 64-bit visibility buffer entry. Two u32 words per pixel.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct VisibilityPayload {
    /// Material ID (16 bits) | Normal index (3 bits) | Depth (13 bits)
    pub lo: u32,
    /// Voxel position packed (10+10+10 bits) | LOD (2 bits)
    pub hi: u32,
}

/// Normal direction index for the visibility buffer.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormalIndex {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl VisibilityPayload {
    /// Sentinel for a miss (no intersection).
    pub const MISS: Self = Self { lo: 0, hi: 0 };

    /// Pack a visibility buffer entry.
    #[inline]
    pub fn pack(
        material_id: u16,
        normal: NormalIndex,
        depth_fixed: u16,
        voxel_x: u32,
        voxel_y: u32,
        voxel_z: u32,
        lod: u32,
    ) -> Self {
        let lo = (material_id as u32)
            | ((normal as u32 & 0x7) << 16)
            | (((depth_fixed as u32) & 0x1FFF) << 19);
        let hi = (voxel_x & 0x3FF)
            | ((voxel_y & 0x3FF) << 10)
            | ((voxel_z & 0x3FF) << 20)
            | ((lod & 0x3) << 30);
        Self { lo, hi }
    }

    /// Extract material ID.
    #[inline]
    pub const fn material_id(self) -> u16 {
        (self.lo & 0xFFFF) as u16
    }

    /// Extract normal index.
    #[inline]
    pub const fn normal_index(self) -> u32 {
        (self.lo >> 16) & 0x7
    }

    /// Extract fixed-point depth.
    #[inline]
    pub const fn depth_fixed(self) -> u16 {
        ((self.lo >> 19) & 0x1FFF) as u16
    }

    /// Extract voxel world X.
    #[inline]
    pub const fn voxel_x(self) -> u32 {
        self.hi & 0x3FF
    }

    /// Extract voxel world Y.
    #[inline]
    pub const fn voxel_y(self) -> u32 {
        (self.hi >> 10) & 0x3FF
    }

    /// Extract voxel world Z.
    #[inline]
    pub const fn voxel_z(self) -> u32 {
        (self.hi >> 20) & 0x3FF
    }

    /// Extract LOD level.
    #[inline]
    pub const fn lod(self) -> u32 {
        (self.hi >> 30) & 0x3
    }

    /// Check if this is a miss (no intersection).
    #[inline]
    pub const fn is_miss(self) -> bool {
        self.lo == 0 && self.hi == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::size_of;

    #[test]
    fn layout() {
        assert_eq!(size_of::<VisibilityPayload>(), 8);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let p = VisibilityPayload::pack(42, NormalIndex::PosY, 1000, 500, 200, 800, 2);
        assert_eq!(p.material_id(), 42);
        assert_eq!(p.normal_index(), NormalIndex::PosY as u32);
        assert_eq!(p.depth_fixed(), 1000);
        assert_eq!(p.voxel_x(), 500);
        assert_eq!(p.voxel_y(), 200);
        assert_eq!(p.voxel_z(), 800);
        assert_eq!(p.lod(), 2);
    }

    #[test]
    fn miss_sentinel() {
        assert!(VisibilityPayload::MISS.is_miss());
        assert_eq!(VisibilityPayload::MISS.material_id(), 0);
    }

    #[test]
    fn max_values() {
        let p = VisibilityPayload::pack(511, NormalIndex::NegZ, 8191, 1023, 1023, 1023, 3);
        assert_eq!(p.material_id(), 511);
        assert_eq!(p.normal_index(), NormalIndex::NegZ as u32);
        assert_eq!(p.depth_fixed(), 8191);
        assert_eq!(p.voxel_x(), 1023);
        assert_eq!(p.voxel_y(), 1023);
        assert_eq!(p.voxel_z(), 1023);
        assert_eq!(p.lod(), 3);
    }

    #[test]
    fn all_normals() {
        for (i, n) in [
            NormalIndex::PosX,
            NormalIndex::NegX,
            NormalIndex::PosY,
            NormalIndex::NegY,
            NormalIndex::PosZ,
            NormalIndex::NegZ,
        ]
        .iter()
        .enumerate()
        {
            let p = VisibilityPayload::pack(1, *n, 0, 0, 0, 0, 0);
            assert_eq!(p.normal_index(), i as u32);
        }
    }
}
