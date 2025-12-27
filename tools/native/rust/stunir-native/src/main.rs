use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
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
    /// Determine the build epoch (stdout prints epoch as an integer)
    Epoch {
        #[arg(long)]
        out_json: Option<PathBuf>,
    },

    /// Deterministically import source code into a spec.json (kind=spec, modules=[{path,sha256}])
    ImportCode {
        #[arg(long)]
        input_root: PathBuf,
        #[arg(long)]
        out_spec: PathBuf,
    },

    /// Generate IR v1 from spec.json (and bind spec_sha256)
    SpecToIr {
        #[arg(long)]
        spec_root: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

fn read_bytes(p: &Path) -> anyhow::Result<Vec<u8>> {
    Ok(fs::read(p)?)
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
    let v = canon_value(v.clone());
    let s = serde_json::to_string(&v)? + "\n";
    let mut f = fs::File::create(path)?;
    f.write_all(s.as_bytes())?;
    Ok(())
}

fn cmd_epoch(out_json: Option<PathBuf>) -> anyhow::Result<()> {
    let epoch: i64 = 0;
    if let Some(p) = out_json {
        write_canon_json(&p, &json!({ "epoch": epoch }))?;
    }
    println!("{epoch}");
    Ok(())
}

fn cmd_import_code(input_root: &Path, out_spec: &Path) -> anyhow::Result<()> {
    let mut modules: Vec<(String, String)> = Vec::new();

    for entry in WalkDir::new(input_root).follow_links(false) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let full_path = entry.path();

        // Relative path with forward slashes
        let rel = full_path.strip_prefix(input_root).unwrap_or(full_path);
        let rel_s = rel.to_string_lossy().replace("\\", "/");

        let bytes = read_bytes(full_path)?;
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

    write_canon_json(out_spec, &spec)?;
    Ok(())
}

fn cmd_spec_to_ir(spec_root: &Path, out: &Path) -> anyhow::Result<()> {
    let spec_path = spec_root.join("spec.json");
    let spec_bytes = read_bytes(&spec_path)?;
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

        // Canonical binding (what CI expects)
        "spec_sha256": spec_sha256,

        // Nested binding (what your check expects)
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

    write_canon_json(out, &ir)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Epoch { out_json } => cmd_epoch(out_json),
        Commands::ImportCode { input_root, out_spec } => cmd_import_code(&input_root, &out_spec),
        Commands::SpecToIr { spec_root, out } => cmd_spec_to_ir(&spec_root, &out),
    }
}
