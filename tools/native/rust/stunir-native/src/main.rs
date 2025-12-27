mod errors;
mod hash;
mod ir_v1;
mod jcs;
mod path_policy;
mod validate;
mod verify_emit;
mod verify_pack;
mod spec_to_ir;
mod gen_provenance;
mod check_toolchain;
mod emit;

use clap::{Parser, Subcommand};
use std::process;
use crate::spec_to_ir::run as run_spec_to_ir;
use crate::gen_provenance::run as run_gen_provenance;
use crate::check_toolchain::run as run_check_toolchain;
use crate::emit::run as run_emit;

#[derive(Parser)]
#[command(name = "stunir_native")]
#[command(about = "STUNIR Native Core (Rust)", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Validate { #[arg(long)] in_json: String },
    Verify { #[arg(long)] receipt: String },
    SpecToIr { #[arg(long)] in_json: String, #[arg(long)] out_ir: String },
    GenProvenance { #[arg(long)] in_ir: String, #[arg(long)] epoch_json: String, #[arg(long)] out_prov: String },
    CheckToolchain { #[arg(long)] lockfile: String },
    Emit { 
        #[arg(long)] in_ir: String, 
        #[arg(long)] target: String, 
        #[arg(long)] out_file: String 
    },
}

fn main() {
    let cli = Cli::parse();
    let result = match &cli.command {
        Commands::Validate { in_json } => validate::run(in_json),
        Commands::Verify { receipt } => verify_pack::run(receipt),
        Commands::SpecToIr { in_json, out_ir } => run_spec_to_ir(in_json, out_ir),
        Commands::GenProvenance { in_ir, epoch_json, out_prov } => run_gen_provenance(in_ir, epoch_json, out_prov),
        Commands::CheckToolchain { lockfile } => run_check_toolchain(lockfile),
        Commands::Emit { in_ir, target, out_file } => run_emit(in_ir, target, out_file),
    };
    if let Err(e) = result {
        eprintln!("Error: {:?}", e);
        process::exit(1);
    }
}
