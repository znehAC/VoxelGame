//! Block registry: TOML-driven block definitions with palette texture generation.

use std::collections::HashMap;

use serde::Deserialize;


fn default_roughness() -> f32 {
    0.9
}
fn default_metallic() -> f32 {
    0.0 
}

/// Block definition loaded from TOML.
#[derive(Deserialize, Clone, Debug)]
pub struct BlockDef {
    pub id: String,
    pub name: String,
    pub color: String,
    #[serde(default = "default_roughness")]
    pub roughness: f32,
    #[serde(default)]
    pub emission: f32,
    #[serde(default)]
    pub noise: f32,
    #[serde(default = "default_metallic")]
    pub metallic: f32,
}

/// TOML file structure: `[[block]]` array.
#[derive(Deserialize)]
struct BlockFile {
    block: Vec<BlockDef>,
}

/// Registry mapping string IDs to sequential u16 indices.
///
/// Index 0 is always reserved for Air.
pub struct BlockRegistry {
    blocks: Vec<BlockDef>,
    name_to_id: HashMap<String, u16>,
}


impl BlockRegistry {
    /// Create an empty registry with Air at index 0.
    pub fn new() -> Self {
        let air = BlockDef {
            id: "engine:air".into(),
            name: "Air".into(),
            color: "#000000".into(),
            roughness: 0.0,
            emission: 0.0,
            noise: 0.0,
            metallic: 0.0,
        };

        let mut name_to_id = HashMap::new();
        name_to_id.insert("engine:air".into(), 0);

        Self {
            blocks: vec![air],
            name_to_id,
        }
    }

    /// Load block definitions from a TOML string.
    ///
    /// Each `[[block]]` entry is assigned the next sequential ID.
    pub fn load_from_string(&mut self, toml_data: &str) -> Result<(), String> {
        let file: BlockFile =
            toml::from_str(toml_data).map_err(|e| format!("TOML parse error: {e}"))?;

        for def in file.block {
            let idx = self.blocks.len() as u16;
            if self.name_to_id.contains_key(&def.id) {
                return Err(format!("Duplicate block ID: {}", def.id));
            }
            self.name_to_id.insert(def.id.clone(), idx);
            self.blocks.push(def);
        }

        Ok(())
    }

    /// Look up the numeric ID for a block name.
    pub fn get_id(&self, name: &str) -> Option<u16> {
        self.name_to_id.get(name).copied()
    }

    /// Look up the display name for a block by its sequential index.
    pub fn get_block_name(&self, index: u16) -> Option<&str> {
        self.blocks.get(index as usize).map(|b| b.name.as_str())
    }

    /// Get a block definition by its sequential index.
    pub fn get_block(&self, idx: u16) -> Option<&BlockDef> {
        self.blocks.get(idx as usize)
    }

    /// Get the sRGB color of a block as [f32; 4] (0.0-1.0).
    pub fn get_block_color_f32(&self, idx: u16) -> [f32; 4] {
        if idx == 0 {
            return [0.0, 0.0, 0.0, 0.0];
        }
        match self.blocks.get(idx as usize) {
            Some(block) => {
                let [r, g, b] = parse_hex_color(&block.color);
                [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
            }
            None => [1.0, 0.0, 1.0, 1.0],
        }
    }

    /// Total number of registered blocks (including Air).
    pub fn block_count(&self) -> u16 {
        self.blocks.len() as u16
    }

    /// Generate a 256x256 2-layer RGBA8 texture for the palette.
    ///
    /// - Layer 0: albedo (R, G, B, 255)
    /// - Layer 1: material properties (roughness, emission, noise, metallic)
    ///
    /// Total size: 256 * 256 * 4 * 2 = 524,288 bytes.
    pub fn generate_texture_data(&self) -> Vec<u8> {
        const W: usize = 256;
        const H: usize = 256;
        const LAYER_SIZE: usize = W * H * 4;

        let mut data = vec![0u8; LAYER_SIZE * 2];

        // Fill unregistered slots with magenta (layer 0)
        for i in 0..(W * H) {
            let off = i * 4;
            data[off] = 255;
            data[off + 1] = 0;
            data[off + 2] = 255;
            data[off + 3] = 255;
        }

        // Air (index 0): transparent black on both layers
        data[0] = 0;
        data[1] = 0;
        data[2] = 0;
        data[3] = 0;

        // Write each registered block
        for (i, block) in self.blocks.iter().enumerate() {
            let u = i % W;
            let v = i / W;
            let pixel = (v * W + u) * 4;

            // Layer 0: color
            if i == 0 {
                // Air already zeroed
            } else {
                let [r, g, b] = parse_hex_color(&block.color);
                data[pixel] = r;
                data[pixel + 1] = g;
                data[pixel + 2] = b;
                data[pixel + 3] = 255;
            }

            // Layer 1: material properties
            let l1 = LAYER_SIZE + pixel;
            data[l1] = (block.roughness.clamp(0.0, 1.0) * 255.0) as u8;
            data[l1 + 1] = (block.emission.clamp(0.0, 1.0) * 255.0) as u8;
            data[l1 + 2] = (block.noise.clamp(0.0, 1.0) * 255.0) as u8;
            data[l1 + 3] = (block.metallic.clamp(0.0, 1.0) * 255.0) as u8;
        }

        data
    }
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse "#RRGGBB" or "#RRGGBBAA" hex color to [R, G, B].
fn parse_hex_color(s: &str) -> [u8; 3] {
    let hex = s.strip_prefix('#').unwrap_or(s);
    // Support both 6-char (RRGGBB) and 8-char (RRGGBBAA) formats
    if hex.len() != 6 && hex.len() != 8 {
        return [255, 0, 255]; // magenta fallback
    }
    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(255);
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0);
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(255);
    [r, g, b]
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TOML: &str = r##"
[[block]]
id = "game:stone"
name = "Stone"
color = "#808080"
roughness = 0.9

[[block]]
id = "game:dirt"
name = "Dirt"
color = "#8B4513"
roughness = 0.95
noise = 0.3

[[block]]
id = "game:lava"
name = "Lava"
color = "#FF4500"
emission = 1.0
"##;

    #[test]
    fn load_blocks() {
        let mut reg = BlockRegistry::new();
        reg.load_from_string(TEST_TOML).unwrap();

        assert_eq!(reg.block_count(), 4); // air + 3
        assert_eq!(reg.get_id("engine:air"), Some(0));
        assert_eq!(reg.get_id("game:stone"), Some(1));
        assert_eq!(reg.get_id("game:dirt"), Some(2));
        assert_eq!(reg.get_id("game:lava"), Some(3));
        assert_eq!(reg.get_id("game:missing"), None);
    }

    #[test]
    fn block_name_lookup() {
        let mut reg = BlockRegistry::new();
        reg.load_from_string(TEST_TOML).unwrap();
        assert_eq!(reg.get_block_name(0), Some("Air"));
        assert_eq!(reg.get_block_name(1), Some("Stone"));
        assert_eq!(reg.get_block_name(2), Some("Dirt"));
        assert_eq!(reg.get_block_name(3), Some("Lava"));
        assert_eq!(reg.get_block_name(99), None);
    }

    #[test]
    fn duplicate_id_rejected() {
        let mut reg = BlockRegistry::new();
        let dupe = r##"
[[block]]
id = "game:stone"
name = "Stone"
color = "#808080"

[[block]]
id = "game:stone"
name = "Stone Again"
color = "#808080"
"##;
        assert!(reg.load_from_string(dupe).is_err());
    }

    #[test]
    fn hex_parsing() {
        assert_eq!(parse_hex_color("#FF0000"), [255, 0, 0]);
        assert_eq!(parse_hex_color("#00FF00"), [0, 255, 0]);
        assert_eq!(parse_hex_color("#0000FF"), [0, 0, 255]);
        assert_eq!(parse_hex_color("#808080"), [128, 128, 128]);
        assert_eq!(parse_hex_color("AABBCC"), [170, 187, 204]);
    }

    #[test]
    fn texture_data_size() {
        let mut reg = BlockRegistry::new();
        reg.load_from_string(TEST_TOML).unwrap();
        let data = reg.generate_texture_data();
        assert_eq!(data.len(), 256 * 256 * 4 * 2);
    }

    #[test]
    fn texture_air_is_zero() {
        let reg = BlockRegistry::new();
        let data = reg.generate_texture_data();
        // Index 0 (Air) — layer 0 should be [0,0,0,0]
        assert_eq!(&data[0..4], &[0, 0, 0, 0]);
        // Layer 1 offset
        let l1 = 256 * 256 * 4;
        assert_eq!(&data[l1..l1 + 4], &[0, 0, 0, 0]);
    }

    #[test]
    fn texture_stone_color() {
        let mut reg = BlockRegistry::new();
        reg.load_from_string(TEST_TOML).unwrap();
        let data = reg.generate_texture_data();
        // Stone is index 1 => pixel (1, 0) => offset 1*4 = 4
        assert_eq!(data[4], 0x80); // R
        assert_eq!(data[5], 0x80); // G
        assert_eq!(data[6], 0x80); // B
        assert_eq!(data[7], 255);  // A
    }
}
