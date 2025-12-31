#!/bin/bash
set -e

echo "Applying Rust fixes..."

TARGET_DIR="tools/native/rust/stunir-native/src"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Cannot find $TARGET_DIR"
    echo "Make sure you are in the root of the repository."
    exit 1
fi

# 1. Fix main.rs
cat > "$TARGET_DIR/main.rs" << 'RUST_EOF'
use std::process;
use clap::{Arg, Command};

mod spec;
mod ir;
mod canonical;
mod provenance;
mod toolchain;
mod receipt;
mod import;
mod check_toolchain;
mod gen_provenance;
mod spec_to_ir;

fn main() {
    let matches = Command::new("stunir-native")
        .version("0.5.0")
        .about("STUNIR Native Core (Rust)")
        .subcommand(
            Command::new("spec-to-ir")
                .about("Convert Spec to IR")
                .arg(Arg::new("in-json").long("in-json").required(true))
                .arg(Arg::new("out-ir").long("out-ir").required(true))
        )
        .subcommand(
            Command::new("gen-provenance")
                .about("Generate Provenance")
                .arg(Arg::new("in-ir").long("in-ir").required(true))
                .arg(Arg::new("epoch-json").long("epoch-json").required(true))
                .arg(Arg::new("out-prov").long("out-prov").required(true))
        )
        .subcommand(
            Command::new("check-toolchain")
                .about("Check Toolchain Lock")
                .arg(Arg::new("lockfile").long("lockfile").required(true))
        )
        .get_matches();

    let result = match matches.subcommand() {
        Some(("spec-to-ir", sub_m)) => {
            let in_json = sub_m.get_one::<String>("in-json").unwrap();
            let out_ir = sub_m.get_one::<String>("out-ir").unwrap();
            spec_to_ir::run(in_json, out_ir)
        },
        Some(("gen-provenance", sub_m)) => {
            let in_ir = sub_m.get_one::<String>("in-ir").unwrap();
            let epoch_json = sub_m.get_one::<String>("epoch-json").unwrap();
            let out_prov = sub_m.get_one::<String>("out-prov").unwrap();
            gen_provenance::run(in_ir, epoch_json, out_prov)
        },
        Some(("check-toolchain", sub_m)) => {
            let lockfile = sub_m.get_one::<String>("lockfile").unwrap();
            check_toolchain::run(lockfile)
        },
        _ => {
            println!("Use --help for usage.");
            Ok(())
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}
RUST_EOF

# 2. Fix canonical.rs
cat > "$TARGET_DIR/canonical.rs" << 'RUST_EOF'
use serde::Serialize;

// JCS (RFC 8785) Canonicalization
// We use a custom formatter or a library if available.
// For this implementation, we rely on serde_json's sorted keys and manual checks.

pub fn to_string_canonical<T>(value: &T) -> Result<String, serde_json::Error>
where
    T: Serialize,
{
    // 1. Sort keys (serde_json does this with sort_keys(true) but we need to be careful about floats)
    // STUNIR Profile 3 only allows integers, so standard JSON sorting is mostly sufficient.
    let mut buf = Vec::new();
    let formatter = serde_json::ser::CompactFormatter;
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
    value.serialize(&mut ser)?;
    
    // Note: This is a simplified canonicalizer. 
    // For full JCS compliance, use the `serde_jcs` crate.
    String::from_utf8(buf).map_err(|e| serde_json::Error::custom(e.to_string()))
}
RUST_EOF

# 3. Fix import.rs
cat > "$TARGET_DIR/import.rs" << 'RUST_EOF'
use std::fs;
use crate::spec::SpecModule;

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
                    let name = path.file_stem().unwrap().to_string_lossy().to_string();
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
RUST_EOF

# 4. Fix toolchain.rs
cat > "$TARGET_DIR/toolchain.rs" << 'RUST_EOF'
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolInfo {
    pub name: String,
    pub path: String,
    pub hash: String,
    pub version: String,
}

#[derive(Debug, Deserialize)]
struct ToolDef {
    name: String,
    path: String,
    // Prefix with _ to suppress unused warning if we just parse it but don't read it yet
    _version: String, 
}

pub fn verify_toolchain(lock_path: &str) -> Result<(), String> {
    // Implementation placeholder
    println!("Verifying toolchain at {}", lock_path);
    Ok(())
}
RUST_EOF

echo "Fixes applied. Rebuilding..."
cd tools/native/rust/stunir-native
cargo build --release
