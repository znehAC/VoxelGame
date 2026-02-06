//! Packed voxel type with 16-bit material IDs.
//!
//! Bit layout: `id (16) | state (8) | flags (8)`
//! - Bits  0-15: Material ID (u16) — up to 65536 block types
//! - Bits 16-23: State/metadata (u8)
//! - Bits 24-31: Flags/lighting (u8)

use bytemuck::{Pod, Zeroable};

/// Packed voxel data stored as a single u32.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct PackedVoxel {
    pub packed: u32,
}

impl PackedVoxel {
    pub const AIR: Self = Self { packed: 0 };

    /// Create a voxel with the given material ID.
    #[inline]
    pub const fn new(id: u16) -> Self {
        Self { packed: id as u32 }
    }

    /// Create a voxel with material, state, and flags.
    #[inline]
    pub const fn with_state(id: u16, state: u8, flags: u8) -> Self {
        Self {
            packed: (id as u32) | ((state as u32) << 16) | ((flags as u32) << 24),
        }
    }

    /// Extract the material ID (bits 0-15).
    #[inline]
    pub const fn id(self) -> u16 {
        self.packed as u16
    }

    /// Set the material ID (bits 0-15).
    #[inline]
    pub fn set_id(&mut self, id: u16) {
        self.packed = (self.packed & 0xFFFF_0000) | (id as u32);
    }

    /// Extract state/metadata (bits 16-23).
    #[inline]
    pub const fn state(self) -> u8 {
        (self.packed >> 16) as u8
    }

    /// Set state/metadata (bits 16-23).
    #[inline]
    pub fn set_state(&mut self, state: u8) {
        self.packed = (self.packed & 0xFF00_FFFF) | ((state as u32) << 16);
    }

    /// Extract flags/lighting (bits 24-31).
    #[inline]
    pub const fn flags(self) -> u8 {
        (self.packed >> 24) as u8
    }

    /// Set flags/lighting (bits 24-31).
    #[inline]
    pub fn set_flags(&mut self, flags: u8) {
        self.packed = (self.packed & 0x00FF_FFFF) | ((flags as u32) << 24);
    }

    /// Check if this voxel is air (empty).
    #[inline]
    pub const fn is_air(self) -> bool {
        self.packed == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn layout() {
        assert_eq!(size_of::<PackedVoxel>(), 4);
        assert_eq!(align_of::<PackedVoxel>(), 4);
    }

    #[test]
    fn packing() {
        let v = PackedVoxel::with_state(0x1234, 0xAB, 0xCD);
        assert_eq!(v.id(), 0x1234);
        assert_eq!(v.state(), 0xAB);
        assert_eq!(v.flags(), 0xCD);
    }

    #[test]
    fn set_fields() {
        let mut v = PackedVoxel::new(42);
        assert_eq!(v.id(), 42);
        assert_eq!(v.state(), 0);
        assert_eq!(v.flags(), 0);

        v.set_state(0xFF);
        assert_eq!(v.id(), 42);
        assert_eq!(v.state(), 0xFF);

        v.set_flags(0xAA);
        assert_eq!(v.flags(), 0xAA);
        assert_eq!(v.id(), 42);
        assert_eq!(v.state(), 0xFF);

        v.set_id(100);
        assert_eq!(v.id(), 100);
        assert_eq!(v.state(), 0xFF);
        assert_eq!(v.flags(), 0xAA);
    }

    #[test]
    fn air() {
        assert!(PackedVoxel::AIR.is_air());
        assert!(!PackedVoxel::new(1).is_air());
    }
}
