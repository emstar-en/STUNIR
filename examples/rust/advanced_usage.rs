//! STUNIR Advanced Usage Example - Rust
//!
//! This example demonstrates advanced STUNIR features in Rust:
//! - Multi-target code generation
//! - Type-safe IR handling
//! - Manifest generation
//! - Builder pattern for configuration
//!
//! # Usage
//! ```bash
//! rustc advanced_usage.rs -o advanced_usage
//! ./advanced_usage
//! ```

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

// =============================================================================
// Target Language Enumeration
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TargetLanguage {
    Python,
    Rust,
    C89,
    C99,
    Go,
    TypeScript,
}

impl TargetLanguage {
    fn name(&self) -> &'static str {
        match self {
            TargetLanguage::Python => "Python",
            TargetLanguage::Rust => "Rust",
            TargetLanguage::C89 => "C89",
            TargetLanguage::C99 => "C99",
            TargetLanguage::Go => "Go",
            TargetLanguage::TypeScript => "TypeScript",
        }
    }
    
    fn extension(&self) -> &'static str {
        match self {
            TargetLanguage::Python => "py",
            TargetLanguage::Rust => "rs",
            TargetLanguage::C89 | TargetLanguage::C99 => "c",
            TargetLanguage::Go => "go",
            TargetLanguage::TypeScript => "ts",
        }
    }
}

// =============================================================================
// Type System
// =============================================================================

#[derive(Debug, Clone, PartialEq)]
enum IRType {
    I32,
    I64,
    F32,
    F64,
    Bool,
    Str,
    Void,
    Custom(String),
}

impl IRType {
    fn from_str(s: &str) -> IRType {
        match s {
            "i32" => IRType::I32,
            "i64" => IRType::I64,
            "f32" => IRType::F32,
            "f64" => IRType::F64,
            "bool" => IRType::Bool,
            "str" => IRType::Str,
            "void" => IRType::Void,
            other => IRType::Custom(other.to_string()),
        }
    }
    
    fn to_target(&self, target: TargetLanguage) -> String {
        match target {
            TargetLanguage::Python => match self {
                IRType::I32 | IRType::I64 => "int".to_string(),
                IRType::F32 | IRType::F64 => "float".to_string(),
                IRType::Bool => "bool".to_string(),
                IRType::Str => "str".to_string(),
                IRType::Void => "None".to_string(),
                IRType::Custom(s) => s.clone(),
            },
            TargetLanguage::Rust => match self {
                IRType::I32 => "i32".to_string(),
                IRType::I64 => "i64".to_string(),
                IRType::F32 => "f32".to_string(),
                IRType::F64 => "f64".to_string(),
                IRType::Bool => "bool".to_string(),
                IRType::Str => "String".to_string(),
                IRType::Void => "()".to_string(),
                IRType::Custom(s) => s.clone(),
            },
            TargetLanguage::C89 | TargetLanguage::C99 => match self {
                IRType::I32 => "int".to_string(),
                IRType::I64 => "long long".to_string(),
                IRType::F32 => "float".to_string(),
                IRType::F64 => "double".to_string(),
                IRType::Bool => if target == TargetLanguage::C99 { "bool" } else { "int" }.to_string(),
                IRType::Str => "char*".to_string(),
                IRType::Void => "void".to_string(),
                IRType::Custom(s) => s.clone(),
            },
            TargetLanguage::Go => match self {
                IRType::I32 => "int32".to_string(),
                IRType::I64 => "int64".to_string(),
                IRType::F32 => "float32".to_string(),
                IRType::F64 => "float64".to_string(),
                IRType::Bool => "bool".to_string(),
                IRType::Str => "string".to_string(),
                IRType::Void => "".to_string(),
                IRType::Custom(s) => s.clone(),
            },
            TargetLanguage::TypeScript => match self {
                IRType::I32 | IRType::I64 | IRType::F32 | IRType::F64 => "number".to_string(),
                IRType::Bool => "boolean".to_string(),
                IRType::Str => "string".to_string(),
                IRType::Void => "void".to_string(),
                IRType::Custom(s) => s.clone(),
            },
        }
    }
}

// =============================================================================
// IR Structures
// =============================================================================

#[derive(Debug, Clone)]
struct IRParam {
    name: String,
    param_type: IRType,
}

#[derive(Debug, Clone)]
struct IRFunction {
    name: String,
    params: Vec<IRParam>,
    returns: IRType,
    is_exported: bool,
}

#[derive(Debug, Clone)]
struct IRModule {
    name: String,
    version: String,
    functions: Vec<IRFunction>,
}

// =============================================================================
// Code Emitter Trait
// =============================================================================

trait CodeEmitter {
    fn emit(&self, module: &IRModule) -> String;
    fn language(&self) -> TargetLanguage;
    
    fn emit_header(&self, module: &IRModule) -> String {
        format!("// Generated {} module: {}\n// Version: {}\n\n",
                self.language().name(), module.name, module.version)
    }
}

// =============================================================================
// Python Emitter
// =============================================================================

struct PythonEmitter;

impl CodeEmitter for PythonEmitter {
    fn language(&self) -> TargetLanguage { TargetLanguage::Python }
    
    fn emit(&self, module: &IRModule) -> String {
        let mut code = format!(
            "\"\"\"Generated Python module: {}\nVersion: {}\n\"\"\"\n\nfrom typing import Any\n\n",
            module.name, module.version
        );
        
        for func in &module.functions {
            let params: Vec<String> = func.params.iter()
                .map(|p| format!("{}: {}", p.name, p.param_type.to_target(TargetLanguage::Python)))
                .collect();
            let ret_type = func.returns.to_target(TargetLanguage::Python);
            
            code.push_str(&format!(
                "def {}({}) -> {}:\n    \"\"\"STUNIR generated function.\"\"\"\n    pass\n\n",
                func.name, params.join(", "), ret_type
            ));
        }
        
        code
    }
}

// =============================================================================
// Rust Emitter
// =============================================================================

struct RustEmitter;

impl CodeEmitter for RustEmitter {
    fn language(&self) -> TargetLanguage { TargetLanguage::Rust }
    
    fn emit(&self, module: &IRModule) -> String {
        let mut code = format!(
            "//! Generated Rust module: {}\n//! Version: {}\n\n",
            module.name, module.version
        );
        
        for func in &module.functions {
            let vis = if func.is_exported { "pub " } else { "" };
            let params: Vec<String> = func.params.iter()
                .map(|p| format!("{}: {}", p.name, p.param_type.to_target(TargetLanguage::Rust)))
                .collect();
            let ret_type = func.returns.to_target(TargetLanguage::Rust);
            
            code.push_str(&format!(
                "{}fn {}({}) -> {} {{\n    todo!(\"Implementation placeholder\")\n}}\n\n",
                vis, func.name, params.join(", "), ret_type
            ));
        }
        
        code
    }
}

// =============================================================================
// Multi-Target Emitter
// =============================================================================

struct MultiTargetEmitter {
    emitters: Vec<Box<dyn CodeEmitter>>,
}

impl MultiTargetEmitter {
    fn new() -> Self {
        Self { emitters: Vec::new() }
    }
    
    fn add_target(mut self, emitter: Box<dyn CodeEmitter>) -> Self {
        self.emitters.push(emitter);
        self
    }
    
    fn emit_all(&self, module: &IRModule) -> BTreeMap<String, String> {
        let mut outputs = BTreeMap::new();
        
        for emitter in &self.emitters {
            let code = emitter.emit(module);
            let filename = format!("{}.{}", module.name, emitter.language().extension());
            outputs.insert(filename, code);
        }
        
        outputs
    }
}

// =============================================================================
// Configuration Builder
// =============================================================================

#[derive(Debug, Clone)]
struct EmitterConfig {
    targets: Vec<TargetLanguage>,
    output_dir: String,
    generate_receipts: bool,
    verify_determinism: bool,
}

impl Default for EmitterConfig {
    fn default() -> Self {
        Self {
            targets: vec![TargetLanguage::Python, TargetLanguage::Rust],
            output_dir: "output".to_string(),
            generate_receipts: true,
            verify_determinism: true,
        }
    }
}

struct EmitterConfigBuilder {
    config: EmitterConfig,
}

impl EmitterConfigBuilder {
    fn new() -> Self {
        Self { config: EmitterConfig::default() }
    }
    
    fn targets(mut self, targets: Vec<TargetLanguage>) -> Self {
        self.config.targets = targets;
        self
    }
    
    fn output_dir(mut self, dir: &str) -> Self {
        self.config.output_dir = dir.to_string();
        self
    }
    
    fn receipts(mut self, enabled: bool) -> Self {
        self.config.generate_receipts = enabled;
        self
    }
    
    fn verify(mut self, enabled: bool) -> Self {
        self.config.verify_determinism = enabled;
        self
    }
    
    fn build(self) -> EmitterConfig {
        self.config
    }
}

// =============================================================================
// Manifest Generation
// =============================================================================

fn compute_hash(data: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in data.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}{:016x}{:016x}{:016x}", hash, hash ^ 0x12345678, hash ^ 0x9abcdef0, hash ^ 0xdeadbeef)
}

fn generate_manifest(outputs: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let mut manifest = BTreeMap::new();
    
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    manifest.insert("schema".to_string(), "stunir.manifest.targets.v1".to_string());
    manifest.insert("manifest_epoch".to_string(), epoch.to_string());
    manifest.insert("entry_count".to_string(), outputs.len().to_string());
    
    // Hash each output
    let mut entries = Vec::new();
    for (filename, code) in outputs {
        let hash = compute_hash(code);
        entries.push(format!("{}:{}", filename, &hash[..16]));
    }
    manifest.insert("entries".to_string(), entries.join(";"));
    
    // Compute manifest hash
    let content: String = manifest.iter()
        .filter(|(k, _)| *k != "manifest_hash")
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join("&");
    manifest.insert("manifest_hash".to_string(), compute_hash(&content));
    
    manifest
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    println!("============================================================");
    println!("STUNIR Advanced Usage Example - Rust");
    println!("============================================================\n");
    
    // Create sample module
    let module = IRModule {
        name: "calculator".to_string(),
        version: "2.0.0".to_string(),
        functions: vec![
            IRFunction {
                name: "add".to_string(),
                params: vec![
                    IRParam { name: "a".to_string(), param_type: IRType::I32 },
                    IRParam { name: "b".to_string(), param_type: IRType::I32 },
                ],
                returns: IRType::I32,
                is_exported: true,
            },
            IRFunction {
                name: "divide".to_string(),
                params: vec![
                    IRParam { name: "x".to_string(), param_type: IRType::F64 },
                    IRParam { name: "y".to_string(), param_type: IRType::F64 },
                ],
                returns: IRType::F64,
                is_exported: true,
            },
        ],
    };
    
    // Build configuration
    println!("🛠️  Building configuration...");
    let config = EmitterConfigBuilder::new()
        .targets(vec![TargetLanguage::Python, TargetLanguage::Rust])
        .output_dir("generated")
        .receipts(true)
        .verify(true)
        .build();
    println!("   Targets: {:?}", config.targets);
    println!("   Output dir: {}", config.output_dir);
    println!();
    
    // Create multi-target emitter
    println!("🎯 Emitting code for multiple targets...");
    let emitter = MultiTargetEmitter::new()
        .add_target(Box::new(PythonEmitter))
        .add_target(Box::new(RustEmitter));
    
    let outputs = emitter.emit_all(&module);
    
    for (filename, code) in &outputs {
        println!("\n--- {} ({} bytes) ---", filename, code.len());
        println!("{}", code);
    }
    
    // Generate manifest
    println!("\n📋 Generating manifest...");
    let manifest = generate_manifest(&outputs);
    println!("   Schema: {}", manifest.get("schema").unwrap());
    println!("   Entries: {}", manifest.get("entry_count").unwrap());
    println!("   Hash: {}...", &manifest.get("manifest_hash").unwrap()[..16]);
    
    // Summary
    println!("\n============================================================");
    println!("Summary");
    println!("============================================================");
    println!("Module:     {}", module.name);
    println!("Functions:  {}", module.functions.len());
    println!("Targets:    {}", outputs.len());
    println!("\n✅ Advanced usage example completed!");
}
