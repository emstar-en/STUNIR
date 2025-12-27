use clap::{Parser, Subcommand};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "stunir-native")]
#[command(about = "STUNIR Deterministic Core", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Determine the build epoch
    Epoch {
        #[arg(long)]
        out_json: Option<PathBuf>,
        #[arg(long)]
        print_epoch: bool,
    },
    /// Import source code into a spec
    ImportCode {
        #[arg(long)]
        input_root: PathBuf,
        #[arg(long)]
        out_spec: PathBuf,
    },
    // Placeholders
    SpecToIr,
    Receipt,
}

#[derive(Serialize)]
struct EpochOutput {
    selected_epoch: u64,
    source: String,
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
                let output = EpochOutput {
                    selected_epoch: epoch,
                    source,
                };
                // Canonical-ish JSON (Compact, Sorted Keys via Serde default)
                let json_bytes = serde_json::to_vec(&output)?;
                let mut file = File::create(path)?;
                file.write_all(&json_bytes)?;
                file.write_all(b"\n")?;
            }
        }
        Commands::ImportCode { input_root, out_spec } => {
            let mut modules = Vec::new();
            
            // Deterministic walk (sorted by filename)
            for entry in walkdir::WalkDir::new(&input_root).sort_by_file_name() {
                let entry = entry?;
                if entry.file_type().is_file() {
                    let path = entry.path();
                    // Normalize path separators to forward slash
                    let rel_path = path.strip_prefix(&input_root)?.to_string_lossy().replace("\\", "/");
                    
                    // Hash content
                    let mut file = File::open(path)?;
                    let mut hasher = Sha256::new();
                    std::io::copy(&mut file, &mut hasher)?;
                    let hash = hex::encode(hasher.finalize());
                    
                    modules.push(serde_json::json!({
                        "path": rel_path,
                        "sha256": hash
                    }));
                }
            }
            
            let spec = serde_json::json!({
                "kind": "spec",
                "modules": modules
            });
            
            let mut file = File::create(out_spec)?;
            // serde_json::to_vec produces compact JSON with sorted keys (BTreeMap default)
            let json_bytes = serde_json::to_vec(&spec)?;
            file.write_all(&json_bytes)?;
            file.write_all(b"\n")?;
        }
        _ => {
            println!("Command not yet implemented in native binary.");
        }
    }

    Ok(())
}
