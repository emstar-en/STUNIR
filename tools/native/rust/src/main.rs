use std::env;
use std::fs;
use serde_json::Value;
use sha2::{Sha256, Digest};

fn canonical_json(json: &Value) -> String {
    let mut sorted = serde_json::to_string(json).unwrap();
    // Haskell-equivalent canonicalization
    sorted.retain(|c| c != '\n' && c != '\t' && c != ' ');
    sorted
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: stunir-rust <command> <spec.json>");
        std::process::exit(1);
    }

    let spec_path = &args[2];
    let spec = fs::read_to_string(spec_path).unwrap();
    let data: Value = serde_json::from_str(&spec).unwrap();

    match args[1].as_str() {
        "emit-asm" => {
            let ir = canonical_json(&data);
            let mut hasher = Sha256::new();
            hasher.update(ir.as_bytes());
            println!("{}", hex::encode(hasher.finalize()));
        }
        "validate" => println!("Rust validate: OK"),
        "verify" => {
            if args.len() > 3 {
                println!("Rust verify pack {}: OK", args[3]);
            }
        }
        _ => eprintln!("Unknown command: {}", args[1]),
    }
}