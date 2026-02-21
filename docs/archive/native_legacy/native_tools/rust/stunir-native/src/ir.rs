use serde::{Deserialize, Serialize};

#[allow(dead_code)]
#[derive(Serialize, Deserialize, Debug)]
pub struct IRSource {
    pub file: String,
    pub line: u32,
}

#[allow(dead_code)]
#[derive(Serialize, Deserialize, Debug)]
pub struct IR {
    pub version: String,
    pub sources: Vec<IRSource>,
}
