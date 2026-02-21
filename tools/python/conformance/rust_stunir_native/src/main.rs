use std::env;
use std::fs;
use std::io::{self, Write};
use serde::{Deserialize, Serialize};
use serde_cbor::{from_slice, to_vec};
use sha2::{Sha256, Digest};

#[derive(Serialize, Deserialize)]
struct Spec { spec: serde_json::Value }

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    match args[1].as_str() {
        "canonicalize" => {
            let stdin = io::read_to_string(io::stdin())?;
            let spec: Spec = serde_json::from_str(&stdin)?;
            let cbor = to_vec(&spec.spec).unwrap();
            let mut hasher = Sha256::new();
            hasher.update(&cbor);
            println!("IR: {:x}", hasher.finalize());
            io::stdout().write_all(&cbor)?;
        }
        _ => println!("stunir-native-rs canonicalize"),
    }
    Ok(())
}
