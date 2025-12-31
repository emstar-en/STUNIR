use serde::{Deserialize, Serialize};

#[allow(dead_code)]
#[derive(Serialize, Deserialize, Debug)]
pub struct Provenance {
    pub tool_id: String,
    pub timestamp: String,
}

#[allow(dead_code)]
pub fn generate_c_header(_prov: &Provenance) -> String {
    "// Provenance Header".to_string()
}
