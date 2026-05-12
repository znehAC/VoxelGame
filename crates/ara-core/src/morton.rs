//! Morton Z-order curve encoding/decoding for 8³ brick-local coordinates.
//!
//! Morton codes interleave the bits of (x, y, z) coordinates to produce a
//! single linear index that preserves 3D spatial locality. This improves
//! GPU cache hit rates during DDA traversal within a brick.
//!
//! For 8³ bricks (3 bits per axis), the Morton code is 9 bits (0..511).

/// Spread 3 bits of a value into every-third-bit positions.
/// Input:  0b00000abc (bits 0,1,2)
/// Output: 0b00a00b00c (bits 0,3,6)
#[inline]
const fn spread_bits_3(mut v: u32) -> u32 {
    v &= 0x7;
    // abc -> 00a0bc : shift bit 2 away from bits 0,1
    v = (v | (v << 4)) & 0x43; // 0b0100_0011
    // 00a0bc -> 00a00b00c : shift bit 1 away from bit 0
    v = (v | (v << 2)) & 0x49; // 0b0100_1001
    v
}

/// Compact every-third-bit back to contiguous 3 bits.
/// Input:  0b00a00b00c (bits 0,3,6)
/// Output: 0b00000abc (bits 0,1,2)
#[inline]
const fn compact_bits_3(mut v: u32) -> u32 {
    v &= 0x49; // 0b0100_1001
    v = (v | (v >> 2)) & 0x43; // 0b0100_0011
    v = (v | (v >> 4)) & 0x7;
    v
}

/// Encode 3D coordinates (each 0..7) into a 9-bit Morton code.
///
/// Bit interleaving: z2 y2 x2 z1 y1 x1 z0 y0 x0
#[inline]
pub const fn morton_encode(x: u32, y: u32, z: u32) -> u32 {
    spread_bits_3(x) | (spread_bits_3(y) << 1) | (spread_bits_3(z) << 2)
}

/// Decode a 9-bit Morton code back to (x, y, z), each 0..7.
#[inline]
pub const fn morton_decode(code: u32) -> (u32, u32, u32) {
    (
        compact_bits_3(code),
        compact_bits_3(code >> 1),
        compact_bits_3(code >> 2),
    )
}

/// Precomputed Morton encode table for 8³ (512 entries).
/// Index: `z * 64 + y * 8 + x` → Value: Morton code
pub static ENCODE_LUT: [u16; 512] = {
    let mut lut = [0u16; 512];
    let mut i = 0u32;
    while i < 512 {
        let x = i & 7;
        let y = (i >> 3) & 7;
        let z = (i >> 6) & 7;
        lut[i as usize] = morton_encode(x, y, z) as u16;
        i += 1;
    }
    lut
};

/// Precomputed Morton decode table for 9-bit codes (512 entries).
/// Index: Morton code → Value: packed `(x | y << 3 | z << 6)`
pub static DECODE_LUT: [u16; 512] = {
    let mut lut = [0u16; 512];
    let mut code = 0u32;
    while code < 512 {
        let (x, y, z) = morton_decode(code);
        lut[code as usize] = (x | (y << 3) | (z << 6)) as u16;
        code += 1;
    }
    lut
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        for z in 0..8u32 {
            for y in 0..8u32 {
                for x in 0..8u32 {
                    let code = morton_encode(x, y, z);
                    assert!(code < 512, "code {code} >= 512 for ({x},{y},{z})");
                    let (dx, dy, dz) = morton_decode(code);
                    assert_eq!((dx, dy, dz), (x, y, z), "roundtrip failed for ({x},{y},{z})");
                }
            }
        }
    }

    #[test]
    fn lut_matches_encode() {
        for z in 0..8u32 {
            for y in 0..8u32 {
                for x in 0..8u32 {
                    let linear = z * 64 + y * 8 + x;
                    let expected = morton_encode(x, y, z);
                    assert_eq!(ENCODE_LUT[linear as usize] as u32, expected);
                }
            }
        }
    }

    #[test]
    fn lut_decode_matches() {
        for code in 0..512u32 {
            let (x, y, z) = morton_decode(code);
            let packed = DECODE_LUT[code as usize];
            assert_eq!((packed & 7) as u32, x);
            assert_eq!(((packed >> 3) & 7) as u32, y);
            assert_eq!(((packed >> 6) & 7) as u32, z);
        }
    }

    #[test]
    fn origin_is_zero() {
        assert_eq!(morton_encode(0, 0, 0), 0);
    }

    #[test]
    fn all_codes_unique() {
        let mut seen = [false; 512];
        for z in 0..8u32 {
            for y in 0..8u32 {
                for x in 0..8u32 {
                    let code = morton_encode(x, y, z) as usize;
                    assert!(!seen[code], "duplicate Morton code {code}");
                    seen[code] = true;
                }
            }
        }
    }
}
