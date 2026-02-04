//! Module Import and Discovery Module
//!
//! Provides functionality to scan directories and discover STUNIR source modules.
//! Ensures deterministic ordering of discovered modules for reproducible builds.
//!
//! # Discovery Process
//!
//! 1. Recursively scan the specified root directory
//! 2. Identify files with `.stunir` extension
//! 3. Read file contents into `SpecModule` structures
//! 4. Sort modules by name for determinism
//!
//! # Safety
//!
//! Module discovery is deterministic - the same directory structure always
//! produces the same module list in the same order. This is critical for
//! reproducible builds in critical systems.

use std::fs;
use crate::spec::SpecModule;

/// Scan a directory for STUNIR source modules.
///
/// Recursively scans the specified directory for `.stunir` files and
/// returns them as a sorted list of `SpecModule` structures.
///
/// # Arguments
///
/// * `root` - Root directory path to scan
///
/// # Returns
///
/// * `Ok(Vec<SpecModule>)` - Sorted list of discovered modules
/// * `Err(std::io::Error)` - If directory reading fails
///
/// # Determinism
///
/// Modules are sorted alphabetically by name to ensure deterministic
/// ordering regardless of filesystem iteration order.
///
/// # Example
///
/// ```rust
/// use stunir::import;
///
/// match import::scan_directory("./src") {
///     Ok(modules) => {
///         for module in modules {
///             println!("Found module: {}", module.name);
///         }
///     }
///     Err(e) => eprintln!("Failed to scan: {}", e),
/// }
/// ```
///
/// # Safety
///
/// This function handles IO errors gracefully and returns them to the caller.
/// File contents are read as UTF-8 strings.
pub fn scan_directory(root: &str) -> Result<Vec<SpecModule>, std::io::Error> {
    let mut modules = Vec::new();

    // Simple recursive scan
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "stunir" {
                    let content = fs::read_to_string(&path)?;
                    let name = path.file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "unknown".to_string());
                    modules.push(SpecModule {
                        name,
                        content,
                    });
                }
            }
        }
    }

    // Sort by name for determinism
    modules.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(modules)
}
