use crate::errors::StunirError;
use crate::hash::sha256_hex;
use crate::path_policy::check_relpath_safe;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmitReceipt {
    receipt_version: String,
    outputs: Vec<EmitOutput>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmitOutput {
    relpath: String,
    sha256: String,
}

pub fn cmd_verify_emit(receipt_json: &str, root: &str) -> Result<(), StunirError> {
    let bytes = std::fs::read(receipt_json)
        .map_err(|_| StunirError::VerifyFailed("missing_object"))?;
    let receipt: EmitReceipt = serde_json::from_slice(&bytes)
        .map_err(|_| StunirError::VerifyFailed("bad_digest"))?;

    if receipt.receipt_version != "stunir.emit.v1" {
        return Err(StunirError::VerifyFailed("bad_digest"));
    }

    let rootp = PathBuf::from(root);
    for out in &receipt.outputs {
        check_relpath_safe(&out.relpath)?;
        let want = out.sha256.to_lowercase();
        if want.len() != 64 || !want.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f')) {
            return Err(StunirError::VerifyFailed("bad_digest"));
        }
        let fp = rootp.join(&out.relpath);
        let bytes = std::fs::read(&fp)
            .map_err(|_| StunirError::VerifyFailed("missing_object"))?;
        let digest = sha256_hex(&bytes);
        if digest != want {
            return Err(StunirError::VerifyFailed("bad_digest"));
        }
    }

    println!("OK verify.emit");
    Ok(())
}
