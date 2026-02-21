use anyhow::{Context, Result};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use crate::canonical;

pub fn execute(input: &Path, output: Option<&Path>) -> Result<()> {
    let mut file = File::open(input).with_context(|| format!("Failed to open input file: {:?}", input))?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;

    let canonical_json = canonical::normalize(&content)?;

    match output {
        Some(path) => {
            let mut out_file = File::create(path)?;
            out_file.write_all(canonical_json.as_bytes())?;
        }
        None => {
            io::stdout().write_all(canonical_json.as_bytes())?;
        }
    }

    Ok(())
}
