use std::env;
use std::fs;
use std::process;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};

// --- Data Structures ---

#[derive(Debug, Serialize, Deserialize)]
struct StunirSpec {
    schema: String,
    id: String,
    name: String,
    stages: Vec<String>,
    targets: Vec<String>,
    profile: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct IrInstruction {
    args: Vec<String>,
    op: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct IrFunction {
    body: Vec<IrInstruction>,
    name: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct StunirIR {
    functions: Vec<IrFunction>,
    version: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Provenance {
    epoch: serde_json::Value,
    ir_hash: String,
    schema: String,
    status: String,
}

// --- Commands ---

fn cmd_spec_to_ir(in_json: &str, out_ir: &str) -> Result<(), String> {
    let _spec_content = fs::read_to_string(in_json)
        .map_err(|e| format!("Failed to read spec: {}", e))?;

    let main_fn = IrFunction {
        name: "main".to_string(),
        body: vec![
            IrInstruction { op: "LOAD".to_string(), args: vec!["r1".to_string(), "0".to_string()] },
            IrInstruction { op: "STORE".to_string(), args: vec!["r1".to_string(), "result".to_string()] },
        ],
    };

    let ir = StunirIR {
        version: "1.0.0".to_string(),
        functions: vec![main_fn],
    };

    let json_out = serde_json::to_string_pretty(&ir)
        .map_err(|e| format!("Serialization error: {}", e))?;

    fs::write(out_ir, json_out)
        .map_err(|e| format!("Failed to write IR: {}", e))?;

    println!("Generated IR at: {}", out_ir);
    Ok(())
}

fn cmd_gen_provenance(in_ir: &str, epoch_json: &str, out_prov: &str) -> Result<(), String> {
    // 1. Read IR
    let ir_content = fs::read_to_string(in_ir)
        .map_err(|e| format!("Failed to read IR: {}", e))?;

    // 2. Hash IR (SHA256)
    let mut hasher = Sha256::new();
    hasher.update(ir_content.as_bytes());
    let hash_result = hasher.finalize();
    let hash_hex = hex::encode(hash_result);

    // 3. Read Epoch
    let epoch_content = fs::read_to_string(epoch_json)
        .map_err(|e| format!("Failed to read Epoch: {}", e))?;
    let epoch_data: serde_json::Value = serde_json::from_str(&epoch_content)
        .map_err(|e| format!("Failed to parse Epoch: {}", e))?;

    // 4. Create Provenance
    let prov = Provenance {
        schema: "stunir.provenance.v1".to_string(),
        ir_hash: format!("sha256:{}", hash_hex),
        epoch: epoch_data,
        status: "SUCCESS".to_string(),
    };

    // 5. Write Output
    let json_out = serde_json::to_string_pretty(&prov)
        .map_err(|e| format!("Serialization error: {}", e))?;

    fs::write(out_prov, json_out)
        .map_err(|e| format!("Failed to write Provenance: {}", e))?;

    println!("Generated Provenance at: {}", out_prov);
    Ok(())
}

fn cmd_check_toolchain(_lockfile: &str) -> Result<(), String> {
    println!("CheckToolchain not implemented yet");
    Ok(())
}

// --- CLI Dispatch ---

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        process::exit(1);
    }

    let command = &args[1];
    let result = match command.as_str() {
        "spec-to-ir" => {
            let in_json = find_arg(&args, "--in-json");
            let out_ir = find_arg(&args, "--out-ir");
            match (in_json, out_ir) {
                (Some(i), Some(o)) => cmd_spec_to_ir(&i, &o),
                _ => Err("Missing arguments for spec-to-ir".to_string()),
            }
        },
        "gen-provenance" => {
            let in_ir = find_arg(&args, "--in-ir");
            let epoch = find_arg(&args, "--epoch-json");
            let out_prov = find_arg(&args, "--out-prov");
            match (in_ir, epoch, out_prov) {
                (Some(i), Some(e), Some(o)) => cmd_gen_provenance(&i, &e, &o),
                _ => Err("Missing arguments for gen-provenance".to_string()),
            }
        },
        "check-toolchain" => {
            let lock = find_arg(&args, "--lockfile");
            match lock {
                Some(l) => cmd_check_toolchain(&l),
                _ => Err("Missing arguments for check-toolchain".to_string()),
            }
        },
        _ => Err(format!("Unknown command: {}", command)),
    };

    if let Err(e) = result {
        eprintln!("ERROR: {}", e);
        print_usage();
        process::exit(1);
    }
}

fn find_arg(args: &[String], key: &str) -> Option<String> {
    for i in 0..args.len() - 1 {
        if args[i] == key {
            return Some(args[i+1].clone());
        }
    }
    None
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  stunir-rust spec-to-ir --in-json <file> --out-ir <file>");
    eprintln!("  stunir-rust gen-provenance --in-ir <file> --epoch-json <file> --out-prov <file>");
    eprintln!("  stunir-rust check-toolchain --lockfile <file>");
}
