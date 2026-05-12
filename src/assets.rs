//! Asset loading utilities with fail-safe error handling.

use std::path::Path;

use ara_core::BlockRegistry;

/// Base path for all assets, relative to the executable's working directory.
const ASSETS_DIR: &str = "assets";

/// Load block definitions from `assets/blocks.toml`.
///
/// Returns a fully populated `BlockRegistry` on success, or an error message on failure.
/// This function never panics — all errors are returned as `Err(String)`.
pub fn load_blocks() -> Result<BlockRegistry, String> {
    let path = Path::new(ASSETS_DIR).join("blocks.toml");

    let contents = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let mut registry = BlockRegistry::new();
    registry.load_from_string(&contents)?;

    Ok(registry)
}
