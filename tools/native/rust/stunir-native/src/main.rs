use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "stunir-native")]
#[command(about = "STUNIR Deterministic Core", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Epoch {
        #[arg(long)]
        out_json: Option<PathBuf>,
        #[arg(long)]
        print_epoch: bool,
    },
    ImportCode {
        #[arg(long)]
        input_root: PathBuf,
        #[arg(long)]
        out_spec: PathBuf,
    },
    SpecToIr {
        #[arg(long)]
        spec_root: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    GenProvenance {
        #[arg(long, default_value_t = 0)]
        epoch: u64,
        #[arg(long, default_value = "unknown")]
        epoch_source: String,
        #[arg(long)]
        out_json: PathBuf,
        #[arg(long)]
        out_header: PathBuf,
        #[arg(long)]
        spec_root: Option<PathBuf>,
        #[arg(long)]
        asm_root: Option<PathBuf>,
    },
    CompileProvenance {
        #[arg(long)]
        in_prov: PathBuf,
        #[arg(long)]
        out_bin: PathBuf,
    },
    GenReceipt {
        #[arg(long)]
        target: PathBuf,
        #[arg(long)]
        toolchain: PathBuf,
        #[arg(long)]
        epoch: u64,
        #[arg(long)]
        out: PathBuf,
    },
    Receipt,
}

#[derive(Serialize)]
struct EpochOutput {
    selected_epoch: u64,
    source: String,
}

#[derive(Serialize, Deserialize)]
struct ProvenanceData {
    build_epoch: u64,
    epoch_source: String,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Epoch { out_json, print_epoch } => {
            let (epoch, source) = match std::env::var("SOURCE_DATE_EPOCH") {
                Ok(val) => (val.parse::<u64>().unwrap_or(0), "env".to_string()),
                Err(_) => (0, "default".to_string()),
            };

            if print_epoch {
                println!("{}", epoch);
            }

            if let Some(path) = out_json {
                let output = EpochOutput { selected_epoch: epoch, source };
                let json_bytes = serde_json::to_vec(&output)?;
                let mut file = File::create(path)?;
                file.write_all(&json_bytes)?;
                file.write_all(b"\n")?;
            }
        }
        Commands::ImportCode { input_root, out_spec } => {
            let mut modules = Vec::new();
            for entry in WalkDir::new(&input_root).sort_by_file_name() {
                let entry = entry?;
                if entry.file_type().is_file() {
                    let path = entry.path();
                    let rel_path = path.strip_prefix(&input_root)?.to_string_lossy().replace("\\", "/");
                    let mut file = File::open(path)?;
                    let mut hasher = Sha256::new();
                    std::io::copy(&mut file, &mut hasher)?;
                    let hash = hex::encode(hasher.finalize());
                    modules.push(json!({ "path": rel_path, "sha256": hash }));
                }
            }
            let spec = json!({ "kind": "spec", "modules": modules });
            let mut file = File::create(out_spec)?;
            let json_bytes = serde_json::to_vec(&spec)?;
            file.write_all(&json_bytes)?;
            file.write_all(b"\n")?;
        }
        Commands::SpecToIr { spec_root, out } => {
            let spec_path = spec_root.join("spec.json");
            let mut file = File::open(&spec_path)?;
            let mut content = Vec::new();
            file.read_to_end(&mut content)?;
            
            let mut hasher = Sha256::new();
            hasher.update(&content);
            let spec_hash = hex::encode(hasher.finalize());

            let ir = json!({
                "ir_version": "v1",
                "module_name": "stunir_bootstrap",
                "docstring": "Bootstrap IR generated from source manifest.",
                "types": [],
                "functions": [],
                "source": { "spec_sha256": spec_hash.clone(), "spec_logical_path": "spec.json" },
                "spec_sha256": spec_hash,
                "determinism": { "requires_utf8": true, "requires_lf_newlines": true, "requires_stable_ordering": true }
            });

            let mut out_file = File::create(out)?;
            let json_bytes = serde_json::to_vec(&ir)?;
            out_file.write_all(&json_bytes)?;
            out_file.write_all(b"\n")?;
        }
        Commands::GenProvenance { epoch, epoch_source, out_json, out_header, .. } => {
            let prov = ProvenanceData { build_epoch: epoch, epoch_source };
            let json_bytes = serde_json::to_vec(&prov)?;
            let mut f_json = File::create(out_json)?;
            f_json.write_all(&json_bytes)?;
            f_json.write_all(b"\n")?;

            let header_content = format!("#define STUNIR_EPOCH {}\n", epoch);
            let mut f_header = File::create(out_header)?;
            f_header.write_all(header_content.as_bytes())?;
        }
        Commands::CompileProvenance { in_prov, out_bin } => {
            let f = File::open(in_prov)?;
            let prov: ProvenanceData = serde_json::from_reader(f)?;
            let magic = b"STUNIR\0\0";
            let epoch_bytes = prov.build_epoch.to_le_bytes();
            let mut f_bin = File::create(out_bin)?;
            f_bin.write_all(magic)?;
            f_bin.write_all(&epoch_bytes)?;
        }
        Commands::GenReceipt { target, toolchain, epoch, out } => {
            // 1. Hash Target
            let mut file = File::open(&target)?;
            let mut hasher = Sha256::new();
            std::io::copy(&mut file, &mut hasher)?;
            let target_hash = hex::encode(hasher.finalize());

            // 2. Hash Toolchain
            let mut file = File::open(&toolchain)?;
            let mut hasher = Sha256::new();
            std::io::copy(&mut file, &mut hasher)?;
            let toolchain_hash = hex::encode(hasher.finalize());

            // 3. Construct Receipt (as BTreeMap for sorted keys)
            let mut receipt_map = BTreeMap::new();
            receipt_map.insert("schema".to_string(), json!("stunir.receipt.build.v1"));
            receipt_map.insert("status".to_string(), json!("success"));
            receipt_map.insert("build_epoch".to_string(), json!(epoch));
            receipt_map.insert("epoch".to_string(), json!({
                "selected_epoch": epoch,
                "source": "default"
            }));
            receipt_map.insert("target".to_string(), json!(target.to_string_lossy()));
            receipt_map.insert("sha256".to_string(), json!(target_hash));
            receipt_map.insert("toolchain_sha256".to_string(), json!(toolchain_hash));
            receipt_map.insert("argv".to_string(), json!(["native_generated"]));
            receipt_map.insert("inputs".to_string(), json!([]));
            receipt_map.insert("tool".to_string(), Value::Null);

            // 4. Calculate Core ID (Hash of COMPACT + Sorted JSON)
            // Use to_vec (compact) instead of to_vec_pretty
            let receipt_bytes = serde_json::to_vec(&receipt_map)?;
            let mut hasher = Sha256::new();
            hasher.update(&receipt_bytes);
            // No newline for compact hash usually, but let's check if verifier expects one.
            // Standard canonicalization usually implies NO trailing newline in the hash input.
            // hasher.update(b"\n"); // Removed
            let core_id = hex::encode(hasher.finalize());
            
            // 5. Add ID and Write
            receipt_map.insert("receipt_core_id_sha256".to_string(), json!(core_id));
            
            // Write COMPACT JSON to file (with a single trailing newline for POSIX compliance)
            let final_json = serde_json::to_vec(&receipt_map)?;
            let mut out_file = File::create(out)?;
            out_file.write_all(&final_json)?;
            out_file.write_all(b"\n")?;
        }
        _ => {
            println!("Command not yet implemented.");
        }
    }
    Ok(())
}
