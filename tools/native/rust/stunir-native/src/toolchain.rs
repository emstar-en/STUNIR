use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::exit;
use crate::canonical::sha256_hex;

#[derive(Deserialize, Debug)]
struct ToolDef {
    path: String,
    sha256: String,
    version: String,
}

type ToolchainLock = BTreeMap<String, ToolDef>;

pub fn verify_toolchain(lock_path: &str) {
    if !Path::new(lock_path).exists() {
        eprintln!("Error: Lockfile not found at {}", lock_path);
        exit(1);
    }

    let content = fs::read_to_string(lock_path).expect("Failed to read lockfile");
    let tools: ToolchainLock = match serde_json::from_str(&content) {
        Ok(t) => t,
        Err(_) => {
            eprintln!("Error: Failed to parse toolchain lockfile.");
            exit(1);
        }
    };

    println!("Verifying {} tools from {}", tools.len(), lock_path);

    for (name, def) in tools {
        verify_tool(&name, &def);
    }
    println!("Toolchain verification passed.");
}

fn verify_tool(name: &str, def: &ToolDef) {
    if !Path::new(&def.path).exists() {
        eprintln!("Error: Tool '{}' not found at {}", name, def.path);
        exit(1);
    }

    let content = fs::read(&def.path).expect("Failed to read tool binary");
    let actual_hash = sha256_hex(&content);

    if actual_hash != def.sha256 {
        eprintln!("Error: Hash mismatch for tool '{}'", name);
        eprintln!("  Path: {}", def.path);
        eprintln!("  Expected: {}", def.sha256);
        eprintln!("  Actual:   {}", actual_hash);
        exit(1);
    } else {
        println!("  [OK] {}", name);
    }
}
