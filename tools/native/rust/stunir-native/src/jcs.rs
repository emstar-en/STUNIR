use crate::errors::StunirError;
use serde_json::Value;
use std::collections::BTreeMap;

pub fn canonicalize_to_vec(value: &Value) -> Result<Vec<u8>, StunirError> {
    let mut out = Vec::with_capacity(1024);
    write_value(&mut out, value)?;
    Ok(out)
}

fn write_value(out: &mut Vec<u8>, v: &Value) -> Result<(), StunirError> {
    match v {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(n) => {
            if n.is_i64() {
                out.extend_from_slice(n.as_i64().unwrap().to_string().as_bytes());
            } else if n.is_u64() {
                out.extend_from_slice(n.as_u64().unwrap().to_string().as_bytes());
            } else {
                return Err(StunirError::VerifyFailed("bad_digest"));
            }
        }
        Value::String(s) => write_string(out, s),
        Value::Array(arr) => {
            out.push(b'[');
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_value(out, item)?;
            }
            out.push(b']');
        }
        Value::Object(map) => {
            let mut ordered: BTreeMap<&str, &Value> = BTreeMap::new();
            for (k, v) in map.iter() {
                ordered.insert(k.as_str(), v);
            }

            out.push(b'{');
            let mut first = true;
            for (k, v) in ordered.iter() {
                if !first {
                    out.push(b',');
                }
                first = false;
                write_string(out, k);
                out.push(b':');
                write_value(out, v)?;
            }
            out.push(b'}');
        }
    }
    Ok(())
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.push(b'"');
    for ch in s.chars() {
        match ch {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\u{08}' => out.extend_from_slice(b"\\b"),
            '\u{0C}' => out.extend_from_slice(b"\\f"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            c if (c as u32) <= 0x1F => {
                let code = c as u32;
                let hex = format!("\\u{:04x}", code);
                out.extend_from_slice(hex.as_bytes());
            }
            c => {
                let mut buf = [0u8; 4];
                let encoded = c.encode_utf8(&mut buf);
                out.extend_from_slice(encoded.as_bytes());
            }
        }
    }
    out.push(b'"');
}

pub fn parse_json(bytes: &[u8]) -> Result<Value, StunirError> {
    serde_json::from_slice(bytes).map_err(|_| StunirError::Json("bad_digest"))
}
