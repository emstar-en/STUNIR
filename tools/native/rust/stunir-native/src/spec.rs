use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct SpecModule {
    pub name: String,
    pub code: String,
    pub lang: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Spec {
    pub kind: String,
    pub modules: Vec<SpecModule>,
}
