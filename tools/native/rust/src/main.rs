use clap::{Parser, Subcommand};
use anyhow::Result;
use std::path::PathBuf;
use stunir_native::{commands, spec_to_ir, emit};

#[derive(Parser)]
#[command(name = "stunir-native")]
#[command(about = "STUNIR Deterministic Build Core", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Canonicalize a JSON file (RFC 8785 / JCS subset)
    Canon {
        /// Input JSON file
        #[arg(short, long)]
        input: PathBuf,
        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Calculate SHA-256 Merkle hash of a file or directory
    Hash {
        /// Input path
        #[arg(short, long)]
        path: PathBuf,
    },
    /// Verify a receipt against a manifest
    Verify {
        /// Receipt file
        #[arg(short, long)]
        receipt: PathBuf,
    },
    /// Convert Spec to IR
    SpecToIr {
        /// Input Spec JSON
        #[arg(short, long)]
        input: String,
        /// Output IR JSON
        #[arg(short, long)]
        output: String,
    },
    /// Emit code from IR
    Emit {
        /// Input IR JSON
        #[arg(short, long)]
        input: String,
        /// Target language (python)
        #[arg(short, long)]
        target: String,
        /// Output file
        #[arg(short, long)]
        output: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Canon { input, output } => {
            commands::canon::execute(input, output.as_deref())?;
        }
        Commands::Hash { path } => {
            commands::hash::execute(path)?;
        }
        Commands::Verify { receipt } => {
            commands::verify::execute(receipt)?;
        }
        Commands::SpecToIr { input, output } => {
            spec_to_ir::run(input, output)?;
        }
        Commands::Emit { input, target, output } => {
            emit::run(input, target, output)?;
        }
    }

    Ok(())
}
