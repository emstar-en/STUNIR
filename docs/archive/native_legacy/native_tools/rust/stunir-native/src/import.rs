use std::fs;
use crate::spec::SpecModule;

#[allow(dead_code)]
pub fn scan_directory(root: &str) -> Result<Vec<SpecModule>, std::io::Error> {
    let mut modules = Vec::new();
    
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "stunir" {
                    let content = fs::read_to_string(&path)?;
                    let name = path.file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "unknown".to_string());
                    
                    modules.push(SpecModule {
                        name,
                        code: content,
                        lang: "unknown".to_string(),
                    });
                }
            }
        }
    }
    
    modules.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(modules)
}
