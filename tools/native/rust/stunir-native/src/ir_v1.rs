use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IrV1 {
    pub ir_version: String,
    pub module_name: String,
    #[serde(default)]
    pub docstring: Option<String>,
    pub types: Vec<TypeDecl>,
    pub functions: Vec<FnDecl>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypeDecl {
    pub name: String,
    #[serde(default)]
    pub docstring: Option<String>,
    pub fields: Vec<FieldDecl>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldDecl {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
    #[serde(default)]
    pub optional: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FnDecl {
    pub name: String,
    #[serde(default)]
    pub docstring: Option<String>,
    pub args: Vec<ArgDecl>,
    pub return_type: String,
    #[serde(default)]
    pub steps: Option<Vec<Step>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArgDecl {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Step {
    pub op: Op,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub value: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Op {
    Return,
    Call,
    Assign,
    Error,
}

pub fn check_module_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else { return false; };
    if !first.is_ascii_alphabetic() {
        return false;
    }
    for c in chars {
        if !(c.is_ascii_alphanumeric() || c == '_') {
            return false;
        }
    }
    true
}
