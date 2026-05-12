//! Packed voxel type with 9-bit material IDs and simulation state.
//!
//! Bit layout (u16):
//! - Bits 0-8:   Material ID (9 bits, 0-511)
//! - Bits 9-10:  Variant (2 bits, 0-3)
//! - Bits 11-13: Level (3 bits, 0-7)
//! - Bit  14:    Flag - is_active
//! - Bit  15:    Flag - is_dirty

use bytemuck::{Pod, Zeroable};

/// Packed voxel data stored as a single u16.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct PackedVoxel {
    pub packed: u16,
}

impl PackedVoxel {
    pub const AIR: Self = Self { packed: 0 };

    const ID_MASK: u16 = 0x1FF; // 9 bits
    const VARIANT_MASK: u16 = 0x3; // 2 bits
    const VARIANT_SHIFT: u32 = 9;
    const LEVEL_MASK: u16 = 0x7; // 3 bits
    const LEVEL_SHIFT: u32 = 11;
    const ACTIVE_BIT: u16 = 1 << 14;
    const DIRTY_BIT: u16 = 1 << 15;

    /// Create a voxel with the given material ID.
    #[inline]
    pub const fn new(id: u16) -> Self {
        Self {
            packed: id & Self::ID_MASK,
        }
    }

    /// Extract the material ID (bits 0-8).
    #[inline]
    pub const fn id(self) -> u16 {
        self.packed & Self::ID_MASK
    }

    /// Set the material ID (bits 0-8).
    #[inline]
    pub fn set_id(&mut self, id: u16) {
        self.packed = (self.packed & !Self::ID_MASK) | (id & Self::ID_MASK);
    }

    /// Extract variant (bits 9-10).
    #[inline]
    pub const fn variant(self) -> u8 {
        ((self.packed >> Self::VARIANT_SHIFT) & Self::VARIANT_MASK) as u8
    }

    /// Set variant (bits 9-10).
    #[inline]
    pub fn set_variant(&mut self, variant: u8) {
        let mask = Self::VARIANT_MASK << Self::VARIANT_SHIFT;
        let val = ((variant as u16) & Self::VARIANT_MASK) << Self::VARIANT_SHIFT;
        self.packed = (self.packed & !mask) | val;
    }

    /// Extract level (bits 11-13).
    #[inline]
    pub const fn level(self) -> u8 {
        ((self.packed >> Self::LEVEL_SHIFT) & Self::LEVEL_MASK) as u8
    }

    /// Set level (bits 11-13).
    #[inline]
    pub fn set_level(&mut self, level: u8) {
        let mask = Self::LEVEL_MASK << Self::LEVEL_SHIFT;
        let val = ((level as u16) & Self::LEVEL_MASK) << Self::LEVEL_SHIFT;
        self.packed = (self.packed & !mask) | val;
    }

    /// Check if this voxel has an active metadata entry (bit 14).
    #[inline]
    pub const fn is_active(self) -> bool {
        (self.packed & Self::ACTIVE_BIT) != 0
    }

    /// Set the active metadata flag (bit 14).
    #[inline]
    pub fn set_active(&mut self, active: bool) {
        if active {
            self.packed |= Self::ACTIVE_BIT;
        } else {
            self.packed &= !Self::ACTIVE_BIT;
        }
    }

    /// Check if this voxel is dirty (bit 15).
    #[inline]
    pub const fn is_dirty(self) -> bool {
        (self.packed & Self::DIRTY_BIT) != 0
    }

    /// Set the dirty flag (bit 15).
    #[inline]
    pub fn set_dirty(&mut self, dirty: bool) {
        if dirty {
            self.packed |= Self::DIRTY_BIT;
        } else {
            self.packed &= !Self::DIRTY_BIT;
        }
    }

    /// Check if this voxel is air (empty).
    #[inline]
    pub const fn is_air(self) -> bool {
        self.id() == 0
    }
}

/// Brick metadata stored in a parallel array alongside the brick pool.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct BrickHeader {
    /// Packed world coordinate: 10 bits x, 10 bits y, 10 bits z, 2 bits flags.
    pub world_coord_packed: u32,
    /// Frame number when this brick was last accessed.
    pub last_access_frame: u32,
    /// Number of non-air voxels in this brick.
    pub solid_count: u32,
    pub _pad: u32,
}

impl BrickHeader {
    /// Pack a world brick coordinate into 9+9+9+3 bits (LOD in bits 27-29).
    pub fn pack_coord(x: u32, y: u32, z: u32, lod: u32) -> u32 {
        (x & 0x1FF) | ((y & 0x1FF) << 9) | ((z & 0x1FF) << 18) | ((lod & 0x7) << 27)
    }

    /// Unpack world brick coordinate from packed u32 (masks out LOD bits 27-29).
    pub fn unpack_coord(packed: u32) -> [u32; 3] {
        let p = packed & 0x7FFFFFF;
        [p & 0x1FF, (p >> 9) & 0x1FF, (p >> 18) & 0x1FF]
    }
}

/// Convert a world voxel coordinate to brick coordinate.
#[inline]
pub fn world_to_brick(x: i32, y: i32, z: i32) -> [i32; 3] {
    [x >> crate::BRICK_SHIFT as i32, y >> crate::BRICK_SHIFT as i32, z >> crate::BRICK_SHIFT as i32]
}

/// Convert a world voxel coordinate to the local offset within its brick.
#[inline]
pub fn world_to_local(x: i32, y: i32, z: i32) -> [u32; 3] {
    let mask = crate::BRICK_SIZE as i32 - 1;
    [(x & mask) as u32, (y & mask) as u32, (z & mask) as u32]
}

/// Compute the Morton Z-order index within a brick from local coordinates.
///
/// Morton encoding interleaves bits of (x, y, z) for 3D cache locality.
/// Each coordinate must be 0..7.
#[inline]
pub fn brick_local_index(lx: u32, ly: u32, lz: u32) -> u32 {
    crate::morton::morton_encode(lx, ly, lz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn layout() {
        assert_eq!(size_of::<PackedVoxel>(), 2);
        assert_eq!(align_of::<PackedVoxel>(), 2);
    }

    #[test]
    fn packing() {
        let mut v = PackedVoxel::new(300);
        v.set_variant(3);
        v.set_level(5);
        v.set_active(true);
        v.set_dirty(true);
        assert_eq!(v.id(), 300);
        assert_eq!(v.variant(), 3);
        assert_eq!(v.level(), 5);
        assert!(v.is_active());
        assert!(v.is_dirty());
    }

    #[test]
    fn max_values() {
        let mut v = PackedVoxel::new(511);
        v.set_variant(3);
        v.set_level(7);
        v.set_active(true);
        v.set_dirty(true);
        assert_eq!(v.id(), 511);
        assert_eq!(v.variant(), 3);
        assert_eq!(v.level(), 7);
        assert!(v.is_active());
        assert!(v.is_dirty());
    }

    #[test]
    fn set_fields() {
        let mut v = PackedVoxel::new(42);
        assert_eq!(v.id(), 42);
        assert_eq!(v.variant(), 0);
        assert_eq!(v.level(), 0);
        assert!(!v.is_active());
        assert!(!v.is_dirty());

        v.set_variant(2);
        assert_eq!(v.id(), 42);
        assert_eq!(v.variant(), 2);

        v.set_level(6);
        assert_eq!(v.level(), 6);
        assert_eq!(v.variant(), 2);

        v.set_active(true);
        assert!(v.is_active());

        v.set_dirty(true);
        assert!(v.is_dirty());

        v.set_id(500);
        assert_eq!(v.id(), 500);
        assert_eq!(v.variant(), 2);
        assert_eq!(v.level(), 6);
        assert!(v.is_active());
        assert!(v.is_dirty());
    }

    #[test]
    fn overflow_truncation() {
        let mut v = PackedVoxel::new(0);
        v.set_level(0xFF); // Should be 0x7 (7)
        assert_eq!(v.level(), 7);

        v.set_variant(0xFF); // Should be 0x3 (3)
        assert_eq!(v.variant(), 3);

        // ID overflow: 1024 & 0x1FF = 0
        let v2 = PackedVoxel::new(1024);
        assert_eq!(v2.id(), 0);
    }

    #[test]
    fn air() {
        assert!(PackedVoxel::AIR.is_air());
        let mut dirty_air = PackedVoxel::AIR;
        dirty_air.set_level(5);
        assert!(dirty_air.is_air());

        assert!(!PackedVoxel::new(1).is_air());
    }
}
