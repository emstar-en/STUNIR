use clap::{Parser, Subcommand};
use serde::Serialize;
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
    // Placeholders for future commands
    ImportCode,
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
            // Logic: Read SOURCE_DATE_EPOCH or default to 0
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
                // Canonical JSON generation
                let json_bytes = serde_json::to_vec_pretty(&output)?; // TODO: Make strict canonical later
                let mut file = File::create(path)?;
                file.write_all(&json_bytes)?;
                // Shell compatibility: Trailing newline
                file.write_all(b"\n")?;
            }
        }
        _ => {
            println!("Command not yet implemented in native binary.");
        }
    }

    Ok(())
}
