use std::env;
use std::process::exit;
use std::path::Path;
use std::fs;
use std::io::{self, Write};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

mod toolchain;
mod canonical;
mod receipt;
mod spec;
mod ir;
mod import;
mod provenance;

use canonical::write_json;
use spec::Spec;
use ir::{IR, IRSource};
use provenance::Provenance;
use receipt::{Receipt, ToolInfo};
use std::collections::BTreeMap;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Error: Missing command.");
        exit(1);
    }

    match args[1].as_str() {
        "version" => println!("stunir-native-rs v0.5.0"),

        "check-toolchain" => {
            if args.len() < 4 || args[2] != "--lockfile" {
                die("Usage: check-toolchain --lockfile <path>");
            }
            toolchain::verify_toolchain(&args[3]);
        }

        "epoch" => {
            if args.len() < 5 || args[2] != "--out-json" || args[4] != "--print-epoch" {
                die("Usage: epoch --out-json <path> --print-epoch");
            }
            let start = SystemTime::now();
            let since_the_epoch = start.duration_since(UNIX_EPOCH).expect("Time went backwards");
            let epoch = since_the_epoch.as_secs();

            let mut map = BTreeMap::new();
            map.insert("epoch".to_string(), epoch);
            write_json(&args[3], &map).expect("Failed to write epoch json");
            println!("{}", epoch);
        }

        "import-code" => {
            // import-code --input-root <root> --out-spec <path>
            if args.len() < 6 || args[2] != "--input-root" || args[4] != "--out-spec" {
                die("Usage: import-code --input-root <root> --out-spec <path>");
            }
            let modules = import::scan_directory(&args[3]).expect("Failed to scan directory");
            let spec = Spec {
                kind: "spec".to_string(),
                modules,
            };
            write_json(&args[5], &spec).expect("Failed to write spec");
            println!("Imported {} modules to {}", spec.modules.len(), args[5]);
        }

        "spec-to-ir" => {
            // spec-to-ir --spec-root <root> --out <path>
            if args.len() < 6 || args[2] != "--spec-root" || args[4] != "--out" {
                die("Usage: spec-to-ir --spec-root <root> --out <path>");
            }
            let spec_path = format!("{}/spec.json", args[3]);
            if !Path::new(&spec_path).exists() {
                die(&format!("Spec file not found: {}", spec_path));
            }

            let spec_content = fs::read(&spec_path).expect("Failed to read spec");
            let spec_hash = canonical::sha256_hex(&spec_content);

            let spec: Spec = serde_json::from_slice(&spec_content).expect("Failed to parse spec.json");

            let ir = IR {
                ir_version: "v1".to_string(),
                module_name: "stunir_module".to_string(),
                types: vec![],
                functions: vec![],
                spec_sha256: spec_hash.clone(),
                source: IRSource {
                    spec_sha256: spec_hash,
                    spec_path: spec_path,
                },
                source_modules: spec.modules,
            };
            write_json(&args[5], &ir).expect("Failed to write IR");
            println!("Generated IR at {}", args[5]);
        }

        "gen-provenance" => {
            // gen-provenance --epoch <epoch> --spec-root <root> --asm-root <root> --out-json <path> --out-header <path>
            // args indices: 0=bin, 1=cmd, 2=--epoch, 3=val, 4=--spec-root, 5=val, 6=--asm-root, 7=val, 8=--out-json, 9=val, 10=--out-header, 11=val
            if args.len() < 12 {
                die("Usage: gen-provenance --epoch <epoch> --spec-root <root> --asm-root <root> --out-json <path> --out-header <path>");
            }
            let epoch: u64 = args[3].parse().expect("Invalid epoch");
            let spec_root = &args[5];
            let out_json = &args[9];
            let out_header = &args[11];

            let spec_path = format!("{}/spec.json", spec_root);
            if !Path::new(&spec_path).exists() {
                die(&format!("Spec file not found: {}", spec_path));
            }
            let spec_content = fs::read(&spec_path).expect("Failed to read spec");
            let spec_hash = canonical::sha256_hex(&spec_content);

            let spec: Spec = serde_json::from_slice(&spec_content).expect("Failed to parse spec.json");
            let mod_names: Vec<String> = spec.modules.iter().map(|m| m.name.clone()).collect();

            let prov = Provenance {
                epoch,
                spec_sha256: spec_hash,
                modules: mod_names,
            };

            write_json(out_json, &prov).expect("Failed to write provenance json");

            let header_content = provenance::generate_c_header(&prov);
            fs::write(out_header, header_content).expect("Failed to write header");

            println!("Generated Provenance: {}, {}", out_json, out_header);
        }

        "compile-provenance" => {
            // compile-provenance --in-prov <path> --out-bin <path>
            if args.len() < 6 || args[2] != "--in-prov" || args[4] != "--out-bin" {
                die("Usage: compile-provenance --in-prov <path> --out-bin <path>");
            }
            let c_file = "tools/prov_emit.c";
            if !Path::new(c_file).exists() {
                die(&format!("Missing C source: {}", c_file));
            }
            let out_bin = &args[5];

            let gcc_cmd = env::var("STUNIR_TOOL_GCC").unwrap_or_else(|_| "gcc".to_string());

            let status = Command::new(&gcc_cmd)
                .args(&[c_file, "-o", out_bin, "-Ibuild"])
                .status()
                .expect("Failed to execute gcc");

            if !status.success() {
                die("Compilation failed");
            }
            println!("Compiled provenance binary to {} using {}", out_bin, gcc_cmd);
        }

        "gen-receipt" => {
            // gen-receipt target status epoch tName tPath tHash tVer [argv...]
            if args.len() < 8 {
                die("Usage: gen-receipt target status epoch tName tPath tHash tVer [argv...]");
            }
            let target = &args[2];
            let status = &args[3];
            let epoch: u64 = args[4].parse().expect("Invalid epoch");
            let t_name = &args[5];
            let t_path = &args[6];
            let t_hash = &args[7];
            let t_ver = &args[8];
            let argv = if args.len() > 9 { args[9..].to_vec() } else { vec![] };

            let tool = ToolInfo {
                name: t_name.clone(),
                path: t_path.clone(),
                sha256: t_hash.clone(),
                version: t_ver.clone(),
            };

            let receipt = Receipt {
                schema: "stunir.receipt.build.v1".to_string(),
                target: target.clone(),
                status: status.clone(),
                build_epoch: epoch,
                epoch_json: "build/epoch.json".to_string(),
                inputs: BTreeMap::new(),
                tool,
                argv,
            };

            let json = canonical::to_string_canonical(&receipt).expect("Failed to serialize receipt");
            println!("{}", json);
        }

        _ => die("Unknown command."),
    }
}

fn die(msg: &str) -> ! {
    eprintln!("Error: {}", msg);
    exit(1);
}
