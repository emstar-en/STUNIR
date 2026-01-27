use anyhow::{Context, Result};
use sha2::{Sha256, Digest};
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

pub fn hash_path(path: &Path) -> Result<String> {
    if path.is_dir() {
        // TODO: Implement Merkle tree hashing for directories
        Ok("DIR_HASH_TODO".to_string())
    } else {
        hash_file(path)
    }
}

pub fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("Failed to open file: {:?}", path))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 1024];

    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }

    Ok(hex::encode(hasher.finalize()))
}
