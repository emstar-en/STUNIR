use serde::{Deserialize, Serialize};

#[allow(dead_code)]
#[derive(Serialize, Deserialize, Debug)]
pub struct SpecModule {
    pub name: String,
    pub code: String,
    pub lang: String,
}

#[allow(dead_code)]
#[derive(Serialize, Deserialize, Debug)]
pub struct Spec {
    pub kind: String,
    pub modules: Vec<SpecModule>,
}
