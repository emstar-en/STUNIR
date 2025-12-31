use walkdir::WalkDir;
use std::path::Path;
use std::fs;
use crate::spec::SpecModule;

pub fn scan_directory(root: &str) -> std::io::Result<Vec<SpecModule>> {
    let mut modules = Vec::new();
    let ignored_dirs = vec![".git", "build", "node_modules", "dist", "target", "__pycache__"];

    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        let path = entry.path();

        // Check ignore list
        if path.components().any(|c| ignored_dirs.contains(&c.as_os_str().to_str().unwrap_or(""))) {
            continue;
        }

        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                let lang = match ext {
                    "py" => Some("python"),
                    "js" => Some("javascript"),
                    "ts" => Some("typescript"),
                    "go" => Some("go"),
                    "rs" => Some("rust"),
                    "c" => Some("c"),
                    "cpp" => Some("cpp"),
                    "java" => Some("java"),
                    "rb" => Some("ruby"),
                    "php" => Some("php"),
                    "sh" => Some("bash"),
                    _ => None,
                };

                if let Some(l) = lang {
                    let content = fs::read_to_string(path)?;
                    let name = path.file_stem().unwrap().to_string_lossy().to_string();
                    modules.push(SpecModule {
                        name,
                        code: content,
                        lang: l.to_string(),
                    });
                }
            }
        }
    }
    Ok(modules)
}
