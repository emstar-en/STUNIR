use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::fs;
use sha2::{Sha256, Digest};

#[derive(Parser)]
#[command(name = "stunir-native")]
#[command(version = "0.1.0")]
#[command(about = "STUNIR Native Core (Enterprise)", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Calculate SHA-256 hash of a file
    Hash {
        #[arg(short, long)]
        file: PathBuf,
    },
    /// Canonicalize JSON (JCS-lite)
    Canon {
        #[arg(short, long)]
        file: PathBuf,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Hash { file } => {
            let mut file = fs::File::open(file)?;
            let mut hasher = Sha256::new();
            std::io::copy(&mut file, &mut hasher)?;
            let hash = hasher.finalize();
            println!("{}", hex::encode(hash));
        }
        Commands::Canon { file } => {
            let content = fs::read_to_string(file)?;
            let json: serde_json::Value = serde_json::from_str(&content)?;
            // Simple pretty print for now, JCS to be implemented
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
    }
    Ok(())
}
