use std::env;
use std::process;
use clap::{Arg, Command};

mod spec;
mod ir;
mod canonical;
mod provenance;
mod toolchain;
mod receipt;
mod import;
mod check_toolchain;
mod gen_provenance;
mod spec_to_ir;

fn main() {
    let matches = Command::new("stunir-native")
        .version("0.5.0")
        .about("STUNIR Native Core (Rust)")
        .subcommand(
            Command::new("spec-to-ir")
                .about("Convert Spec to IR")
                .arg(Arg::new("in-json").long("in-json").required(true))
                .arg(Arg::new("out-ir").long("out-ir").required(true))
        )
        .subcommand(
            Command::new("gen-provenance")
                .about("Generate Provenance")
                .arg(Arg::new("in-ir").long("in-ir").required(true))
                .arg(Arg::new("epoch-json").long("epoch-json").required(true))
                .arg(Arg::new("out-prov").long("out-prov").required(true))
        )
        .subcommand(
            Command::new("check-toolchain")
                .about("Check Toolchain Lock")
                .arg(Arg::new("lockfile").long("lockfile").required(true))
        )
        .get_matches();

    let result = match matches.subcommand() {
        Some(("spec-to-ir", sub_m)) => {
            let in_json = sub_m.get_one::<String>("in-json").unwrap();
            let out_ir = sub_m.get_one::<String>("out-ir").unwrap();
            spec_to_ir::run(in_json, out_ir)
        },
        Some(("gen-provenance", sub_m)) => {
            let in_ir = sub_m.get_one::<String>("in-ir").unwrap();
            let epoch_json = sub_m.get_one::<String>("epoch-json").unwrap();
            let out_prov = sub_m.get_one::<String>("out-prov").unwrap();
            gen_provenance::run(in_ir, epoch_json, out_prov)
        },
        Some(("check-toolchain", sub_m)) => {
            let lockfile = sub_m.get_one::<String>("lockfile").unwrap();
            check_toolchain::run(lockfile)
        },
        _ => {
            println!("Use --help for usage.");
            Ok(())
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}
