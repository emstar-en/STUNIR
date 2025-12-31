use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Serialize, Debug)]
pub struct ToolInfo {
    pub name: String,
    pub path: String,
    pub sha256: String,
    pub version: String,
}

#[derive(Serialize, Debug)]
pub struct Receipt {
    #[serde(rename = "receipt_schema")]
    pub schema: String,
    #[serde(rename = "receipt_target")]
    pub target: String,
    #[serde(rename = "receipt_status")]
    pub status: String,
    #[serde(rename = "receipt_build_epoch")]
    pub build_epoch: u64,
    #[serde(rename = "receipt_epoch_json")]
    pub epoch_json: String,
    #[serde(rename = "receipt_inputs")]
    pub inputs: BTreeMap<String, String>,
    #[serde(rename = "receipt_tool")]
    pub tool: ToolInfo,
    #[serde(rename = "receipt_argv")]
    pub argv: Vec<String>,
}
