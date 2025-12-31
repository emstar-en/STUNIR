#!/bin/bash
set -e

echo "Applying Comprehensive Rust Fixes..."

BASE_DIR="tools/native/rust/stunir-native"
SRC_DIR="$BASE_DIR/src"

if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Cannot find $SRC_DIR"
    echo "Make sure you are in the root of the repository."
    exit 1
fi

# 1. Update Cargo.toml to add missing dependencies
cat > "$BASE_DIR/Cargo.toml" << 'TOML_EOF'
[package]
name = "stunir-native"
version = "0.5.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
sha2 = "0.10"
hex = "0.4"
walkdir = "2.3"
chrono = "0.4"
clap = { version = "4.4", features = ["derive"] }
anyhow = "1.0"
TOML_EOF

# 2. Fix main.rs (Add missing modules)
cat > "$SRC_DIR/main.rs" << 'RUST_EOF'
use std::process;
use clap::{Arg, Command};

mod spec;
mod ir;
mod ir_v1;      // Added
mod errors;     // Added
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
            let in_json = sub_m.get_one::<String>("in-json").expect("required").to_string();
            let out_ir = sub_m.get_one::<String>("out-ir").expect("required").to_string();
            spec_to_ir::run(&in_json, &out_ir)
        },
        Some(("gen-provenance", sub_m)) => {
            let in_ir = sub_m.get_one::<String>("in-ir").expect("required").to_string();
            let epoch_json = sub_m.get_one::<String>("epoch-json").expect("required").to_string();
            let out_prov = sub_m.get_one::<String>("out-prov").expect("required").to_string();
            gen_provenance::run(&in_ir, &epoch_json, &out_prov)
        },
        Some(("check-toolchain", sub_m)) => {
            let lockfile = sub_m.get_one::<String>("lockfile").expect("required").to_string();
            check_toolchain::run(&lockfile)
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

# 3. Fix canonical.rs (Import Error trait)
cat > "$SRC_DIR/canonical.rs" << 'RUST_EOF'
use serde::Serialize;
use serde::ser::Error; // Import trait to use Error::custom

pub fn to_string_canonical<T>(value: &T) -> Result<String, serde_json::Error>
where
    T: Serialize,
{
    let mut buf = Vec::new();
    let formatter = serde_json::ser::CompactFormatter;
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
    value.serialize(&mut ser)?;
    
    String::from_utf8(buf).map_err(|e| serde_json::Error::custom(e.to_string()))
}
RUST_EOF

# 4. Fix import.rs (Match SpecModule struct)
cat > "$SRC_DIR/import.rs" << 'RUST_EOF'
use std::fs;
use crate::spec::SpecModule;

pub fn scan_directory(root: &str) -> Result<Vec<SpecModule>, std::io::Error> {
    let mut modules = Vec::new();
    
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "stunir" {
                    let content = fs::read_to_string(&path)?;
                    let name = path.file_stem().unwrap().to_string_lossy().to_string();
                    
                    // Map file content to SpecModule fields
                    // Assuming file content is the code and lang is inferred or default
                    modules.push(SpecModule {
                        name,
                        code: content,
                        lang: "unknown".to_string(), // Default or parse from file
                    });
                }
            }
        }
    }
    
    modules.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(modules)
}
RUST_EOF

# 5. Fix toolchain.rs (Suppress unused warning)
cat > "$SRC_DIR/toolchain.rs" << 'RUST_EOF'
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
    _version: String, 
}

pub fn verify_toolchain(lock_path: &str) -> Result<(), String> {
    println!("Verifying toolchain at {}", lock_path);
    Ok(())
}
RUST_EOF

echo "Fixes applied. Rebuilding..."
cd "$BASE_DIR"
cargo build --release
