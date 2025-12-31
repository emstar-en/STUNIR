use serde::Serialize;

#[derive(Serialize, Debug)]
pub struct Provenance {
    pub epoch: u64,
    pub spec_sha256: String,
    pub modules: Vec<String>,
}

pub fn generate_c_header(prov: &Provenance) -> String {
    format!(
        "#ifndef STUNIR_PROVENANCE_H\n#define STUNIR_PROVENANCE_H\n\n#define STUNIR_EPOCH {}\n#define STUNIR_SPEC_SHA256 \"{}\"\n\n#endif // STUNIR_PROVENANCE_H\n",
        prov.epoch, prov.spec_sha256
    )
}
