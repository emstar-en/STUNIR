use clap::{Args, Parser, Subcommand};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "stunir-native")]
#[command(about = "STUNIR Deterministic Core (native)", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Determine the build epoch.
    ///
    /// Compatible with the shell pipeline:
    ///   epoch --out-json build/epoch.json --print-epoch
    #[command(name = "epoch")]
    Epoch(EpochArgs),

    /// Deterministically import source code into a spec.json
    #[command(name = "import_code", alias = "import-code")]
    ImportCode(ImportCodeArgs),

    /// Generate IR v1 from spec.json and bind spec_sha256 (top-level + nested)
    #[command(name = "spec_to_ir", alias = "spec-to-ir")]
    SpecToIr(SpecToIrArgs),
}

#[derive(Args)]
struct EpochArgs {
    #[arg(long)]
    out_json: Option<PathBuf>,

    /// Ignored flag; exists for compatibility with scripts that want stdout + json.
    #[arg(long, default_value_t = false)]
    print_epoch: bool,
}

#[derive(Args)]
struct ImportCodeArgs {
    #[arg(long)]
    input_root: PathBuf,

    #[arg(long)]
    out_spec: PathBuf,

    /// Ignored; import is always stable-ordered in native.
    #[arg(long, default_value_t = false)]
    requires_stable_ordering: bool,
}

#[derive(Args)]
struct SpecToIrArgs {
    #[arg(long)]
    spec_root: PathBuf,

    #[arg(long)]
    out: PathBuf,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

fn canon_value(v: Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut keys: Vec<String> = map.keys().cloned().collect();
            keys.sort();
            let mut out = serde_json::Map::new();
            for k in keys {
                let vv = map.get(&k).unwrap().clone();
                out.insert(k, canon_value(vv));
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(canon_value).collect()),
        other => other,
    }
}

fn write_canon_json(path: &Path, v: &Value) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let v = canon_value(v.clone());
    let s = serde_json::to_string(&v)? + "\n";
    let mut f = fs::File::create(path)?;
    f.write_all(s.as_bytes())?;
    Ok(())
}

fn cmd_epoch(args: EpochArgs) -> anyhow::Result<()> {
    let epoch: i64 = 0;

    let out = args
        .out_json
        .unwrap_or_else(|| PathBuf::from("build/epoch.json"));

    write_canon_json(&out, &json!({ "epoch": epoch }))?;

    // print epoch for scripts that capture stdout
    println!("{epoch}");
    Ok(())
}

fn cmd_import_code(args: ImportCodeArgs) -> anyhow::Result<()> {
    let mut modules: Vec<(String, String)> = Vec::new();

    for entry in WalkDir::new(&args.input_root).follow_links(false) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }

        let full = entry.path();
        let rel = full.strip_prefix(&args.input_root).unwrap_or(full);
        let rel_s = rel.to_string_lossy().replace("\\", "/");

        let bytes = fs::read(full)?;
        let h = sha256_hex(&bytes);
        modules.push((rel_s, h));
    }

    modules.sort_by(|a, b| a.0.cmp(&b.0));

    let mods_json: Vec<Value> = modules
        .into_iter()
        .map(|(path, sha256)| json!({ "path": path, "sha256": sha256 }))
        .collect();

    let spec = json!({
        "kind": "spec",
        "modules": mods_json
    });

    write_canon_json(&args.out_spec, &spec)?;
    Ok(())
}

fn cmd_spec_to_ir(args: SpecToIrArgs) -> anyhow::Result<()> {
    let spec_path = args.spec_root.join("spec.json");
    let spec_bytes = fs::read(&spec_path)?;
    let spec_sha256 = sha256_hex(&spec_bytes);

    let spec_val: Value = serde_json::from_slice(&spec_bytes)?;

    let module_name = spec_val
        .get("module_name")
        .and_then(|v| v.as_str())
        .or_else(|| spec_val.get("name").and_then(|v| v.as_str()))
        .unwrap_or("stunir_module");

    let mut ir = json!({
        "ir_version": "v1",
        "module_name": module_name,
        "types": [],
        "functions": [],
        "spec_sha256": spec_sha256,
        "source": {
            "spec_sha256": spec_sha256,
            "spec_path": spec_path.to_string_lossy().replace("\\", "/"),
        }
    });

    if let Some(mods) = spec_val.get("modules") {
        ir.as_object_mut()
            .unwrap()
            .insert("source_modules".to_string(), mods.clone());
    }

    write_canon_json(&args.out, &ir)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Epoch(args) => cmd_epoch(args),
        Commands::ImportCode(args) => cmd_import_code(args),
        Commands::SpecToIr(args) => cmd_spec_to_ir(args),
    }
}
