use anyhow::Result;
use std::path::Path;
use crate::crypto;

pub fn execute(path: &Path) -> Result<()> {
    let hash = crypto::hash_path(path)?;
    println!("{}", hash);
    Ok(())
}
