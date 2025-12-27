use crate::errors::StunirError;
use crate::hash::sha256_hex;
use crate::ir_v1::{check_module_name, IrV1};
use crate::jcs;

pub fn cmd_validate(path: &str, allow_trailing_lf: bool) -> Result<(), StunirError> {
    let bytes = std::fs::read(path).map_err(|_| StunirError::Io("bad_digest"))?;

    // IR validation is LF-only.
    if bytes.iter().any(|b| *b == b'\r') {
        return Err(StunirError::VerifyFailed("bad_digest"));
    }

    let bytes_no_trailing: &[u8] = if allow_trailing_lf {
        if bytes.ends_with(b"\n") {
            &bytes[..bytes.len() - 1]
        } else {
            &bytes[..]
        }
    } else {
        &bytes[..]
    };

    std::str::from_utf8(bytes_no_trailing).map_err(|_| StunirError::Utf8("bad_digest"))?;

    let value = jcs::parse_json(bytes_no_trailing)?;
    let canonical = jcs::canonicalize_to_vec(&value)?;

    if canonical != bytes_no_trailing {
        return Err(StunirError::VerifyFailed("bad_digest"));
    }

    let ir: IrV1 = serde_json::from_value(value).map_err(|_| StunirError::VerifyFailed("bad_digest"))?;

    if ir.ir_version != "v1" {
        return Err(StunirError::VerifyFailed("bad_digest"));
    }

    if !check_module_name(&ir.module_name) {
        return Err(StunirError::VerifyFailed("unsafe_filename"));
    }

    // Enforce steps.value types (string/number/bool/object) per IR schema.
    for f in &ir.functions {
        if let Some(steps) = &f.steps {
            for s in steps {
                if let Some(v) = &s.value {
                    match v {
                        serde_json::Value::String(_) => {}
                        serde_json::Value::Number(_) => {}
                        serde_json::Value::Bool(_) => {}
                        serde_json::Value::Object(_) => {}
                        _ => return Err(StunirError::VerifyFailed("bad_digest")),
                    }
                }
            }
        }
    }

    let digest = sha256_hex(&canonical);
    println!("OK ir_sha256_jcs={}", digest);
    Ok(())
}
