//! Packed voxel type with 14-bit material IDs and simulation state.
//!
//! Bit layout: `id (14) | level (4) | temp (4) | variant (4) | flags (6)`
//! - Bits  0-13: Material ID (u16) — up to 16,384 block types
//! - Bits 14-17: Level/Damage (u8) — 0-15
//! - Bits 18-21: Temperature (u8) — 0-15
//! - Bits 22-25: Variant/Noise (u8) — 0-15
//! - Bits 26-31: Flags (u8) — 0-63

use bytemuck::{Pod, Zeroable};

/// Packed voxel data stored as a single u32.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct PackedVoxel {
    pub packed: u32,
}

impl PackedVoxel {
    pub const AIR: Self = Self { packed: 0 };

    // Bit masks and shifts
    const ID_MASK: u32 = 0x3FFF; // 14 bits
    const LEVEL_MASK: u32 = 0xF; // 4 bits
    const TEMP_MASK: u32 = 0xF; // 4 bits
    const VARIANT_MASK: u32 = 0xF; // 4 bits
    const FLAGS_MASK: u32 = 0x3F; // 6 bits

    const LEVEL_SHIFT: u32 = 14;
    const TEMP_SHIFT: u32 = 18;
    const VARIANT_SHIFT: u32 = 22;
    const FLAGS_SHIFT: u32 = 26;

    /// Create a voxel with the given material ID.
    #[inline]
    pub const fn new(id: u16) -> Self {
        Self {
            packed: (id as u32) & Self::ID_MASK,
        }
    }

    /// Create a voxel with all fields.
    #[inline]
    pub const fn with_all(id: u16, level: u8, temp: u8, variant: u8, flags: u8) -> Self {
        let id_part = (id as u32) & Self::ID_MASK;
        let level_part = ((level as u32) & Self::LEVEL_MASK) << Self::LEVEL_SHIFT;
        let temp_part = ((temp as u32) & Self::TEMP_MASK) << Self::TEMP_SHIFT;
        let variant_part = ((variant as u32) & Self::VARIANT_MASK) << Self::VARIANT_SHIFT;
        let flags_part = ((flags as u32) & Self::FLAGS_MASK) << Self::FLAGS_SHIFT;

        Self {
            packed: id_part | level_part | temp_part | variant_part | flags_part,
        }
    }

    /// Extract the material ID (bits 0-13).
    #[inline]
    pub const fn id(self) -> u16 {
        (self.packed & Self::ID_MASK) as u16
    }

    /// Set the material ID (bits 0-13).
    #[inline]
    pub fn set_id(&mut self, id: u16) {
        self.packed = (self.packed & !Self::ID_MASK) | ((id as u32) & Self::ID_MASK);
    }

    /// Extract level/damage (bits 14-17).
    #[inline]
    pub const fn level(self) -> u8 {
        ((self.packed >> Self::LEVEL_SHIFT) & Self::LEVEL_MASK) as u8
    }

    /// Set level/damage (bits 14-17).
    #[inline]
    pub fn set_level(&mut self, level: u8) {
        let mask = Self::LEVEL_MASK << Self::LEVEL_SHIFT;
        let val = ((level as u32) & Self::LEVEL_MASK) << Self::LEVEL_SHIFT;
        self.packed = (self.packed & !mask) | val;
    }

    /// Extract temperature (bits 18-21).
    #[inline]
    pub const fn temperature(self) -> u8 {
        ((self.packed >> Self::TEMP_SHIFT) & Self::TEMP_MASK) as u8
    }

    /// Set temperature (bits 18-21).
    #[inline]
    pub fn set_temperature(&mut self, temp: u8) {
        let mask = Self::TEMP_MASK << Self::TEMP_SHIFT;
        let val = ((temp as u32) & Self::TEMP_MASK) << Self::TEMP_SHIFT;
        self.packed = (self.packed & !mask) | val;
    }

    /// Extract variant/noise seed (bits 22-25).
    #[inline]
    pub const fn variant(self) -> u8 {
        ((self.packed >> Self::VARIANT_SHIFT) & Self::VARIANT_MASK) as u8
    }

    /// Set variant/noise seed (bits 22-25).
    #[inline]
    pub fn set_variant(&mut self, variant: u8) {
        let mask = Self::VARIANT_MASK << Self::VARIANT_SHIFT;
        let val = ((variant as u32) & Self::VARIANT_MASK) << Self::VARIANT_SHIFT;
        self.packed = (self.packed & !mask) | val;
    }

    /// Extract flags (bits 26-31).
    #[inline]
    pub const fn flags(self) -> u8 {
        ((self.packed >> Self::FLAGS_SHIFT) & Self::FLAGS_MASK) as u8
    }

    /// Set flags (bits 26-31).
    #[inline]
    pub fn set_flags(&mut self, flags: u8) {
        let mask = Self::FLAGS_MASK << Self::FLAGS_SHIFT;
        let val = ((flags as u32) & Self::FLAGS_MASK) << Self::FLAGS_SHIFT;
        self.packed = (self.packed & !mask) | val;
    }

    /// Helper to set flags directly without retrieving first.
    #[inline]
    pub fn with_flags(mut self, flags: u8) -> Self {
        self.set_flags(flags);
        self
    }

    /// Check if this voxel is air (empty).
    /// Note: Air is 0 ID. We might have metadata on air (e.g. temperature),
    /// so we check ID, not the whole packed value.
    #[inline]
    pub const fn is_air(self) -> bool {
        self.id() == 0
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
        // ID: 1000, Level: 10, Temp: 5, Variant: 7, Flags: 30
        let v = PackedVoxel::with_all(1000, 10, 5, 7, 30);
        assert_eq!(v.id(), 1000);
        assert_eq!(v.level(), 10);
        assert_eq!(v.temperature(), 5);
        assert_eq!(v.variant(), 7);
        assert_eq!(v.flags(), 30);
    }

    #[test]
    fn max_values() {
        // Test max values for each field
        // ID: 14 bits -> 16383
        // Level: 4 bits -> 15
        // Temp: 4 bits -> 15
        // Variant: 4 bits -> 15
        // Flags: 6 bits -> 63
        let v = PackedVoxel::with_all(16383, 15, 15, 15, 63);
        assert_eq!(v.id(), 16383);
        assert_eq!(v.level(), 15);
        assert_eq!(v.temperature(), 15);
        assert_eq!(v.variant(), 15);
        assert_eq!(v.flags(), 63);
    }

    #[test]
    fn set_fields() {
        let mut v = PackedVoxel::new(42);
        assert_eq!(v.id(), 42);
        assert_eq!(v.level(), 0);

        v.set_level(12);
        assert_eq!(v.id(), 42);
        assert_eq!(v.level(), 12);

        v.set_temperature(8);
        assert_eq!(v.temperature(), 8);
        assert_eq!(v.level(), 12);

        v.set_variant(3);
        assert_eq!(v.variant(), 3);

        v.set_flags(50);
        assert_eq!(v.flags(), 50);

        // Ensure changing ID doesn't mess up other fields
        v.set_id(9999);
        assert_eq!(v.id(), 9999);
        assert_eq!(v.level(), 12);
        assert_eq!(v.temperature(), 8);
        assert_eq!(v.variant(), 3);
        assert_eq!(v.flags(), 50);
    }

    #[test]
    fn overflow_truncation() {
        // Test that values larger than the bit width are truncated masked correctly
        let mut v = PackedVoxel::new(0);

        v.set_level(0xFF); // Should be 0xF (15)
        assert_eq!(v.level(), 15);

        v.set_temperature(0xAB); // Should be 0xB (11)
        assert_eq!(v.temperature(), 11);
    }

    #[test]
    fn air() {
        assert!(PackedVoxel::AIR.is_air());
        // Air with metadata is still air
        let mut dirty_air = PackedVoxel::AIR;
        dirty_air.set_temperature(10);
        assert!(dirty_air.is_air());

        assert!(!PackedVoxel::new(1).is_air());
    }
}
