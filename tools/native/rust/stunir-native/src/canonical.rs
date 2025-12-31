use serde::Serialize;
use serde::ser::Error;

#[allow(dead_code)]
pub fn to_string_canonical<T>(value: &T) -> Result<String, serde_json::Error>
where
    T: Serialize,
{
    let mut buf = Vec::new();
    let formatter = serde_json::ser::CompactFormatter;
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
    value.serialize(&mut ser)?;
    
    String::from_utf8(buf).map_err(|e| serde_json::Error::custom(e.to_string()))
}
