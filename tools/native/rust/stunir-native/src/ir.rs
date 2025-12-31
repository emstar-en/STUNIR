use serde::Serialize;
use crate::spec::SpecModule;

#[derive(Serialize, Debug)]
pub struct IRSource {
    pub spec_sha256: String,
    pub spec_path: String,
}

#[derive(Serialize, Debug)]
pub struct IR {
    pub ir_version: String,
    pub module_name: String,
    pub types: Vec<String>,
    pub functions: Vec<String>,
    pub spec_sha256: String,
    pub source: IRSource,
    pub source_modules: Vec<SpecModule>,
}
