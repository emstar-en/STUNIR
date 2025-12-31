use std::fs;
use crate::spec::SpecModule;

pub fn scan_directory(root: &str) -> Result<Vec<SpecModule>, std::io::Error> {
    let mut modules = Vec::new();

    // Simple recursive scan
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "stunir" {
                    let content = fs::read_to_string(&path)?;
                    let name = path.file_stem().unwrap().to_string_lossy().to_string();
                    modules.push(SpecModule {
                        name,
                        content,
                    });
                }
            }
        }
    }

    // Sort by name for determinism
    modules.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(modules)
}
