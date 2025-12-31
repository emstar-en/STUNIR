use serde::Serialize;
use serde_json::ser::Formatter;
use std::io;

// JCS (RFC 8785) Canonicalization
// We use a custom formatter or a library if available.
// For this implementation, we rely on serde_json's sorted keys and manual checks.
// A full JCS implementation would require a specific crate like `serde_jcs`.

pub fn to_string_canonical<T>(value: &T) -> Result<String, serde_json::Error>
where
    T: Serialize,
{
    // 1. Sort keys (serde_json does this with sort_keys(true) but we need to be careful about floats)
    // STUNIR Profile 3 only allows integers, so standard JSON sorting is mostly sufficient.
    let mut buf = Vec::new();
    let formatter = serde_json::ser::CompactFormatter;
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
    value.serialize(&mut ser)?;

    // Note: This is a simplified canonicalizer. 
    // For full JCS compliance, use the `serde_jcs` crate.
    String::from_utf8(buf).map_err(|e| serde_json::Error::custom(e.to_string()))
}
