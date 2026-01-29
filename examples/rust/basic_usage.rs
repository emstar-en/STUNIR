//! STUNIR Basic Usage Example - Rust
//!
//! This example demonstrates fundamental STUNIR operations in Rust:
//! - Loading a spec file
//! - Converting spec to IR
//! - Generating receipts
//! - Verifying determinism
//!
//! # Usage
//! ```bash
//! rustc basic_usage.rs -o basic_usage
//! ./basic_usage
//! ```

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

// =============================================================================
// Core Types
// =============================================================================

/// Represents a function parameter
#[derive(Debug, Clone)]
struct Param {
    name: String,
    param_type: String,
}

/// Represents a function in the spec
#[derive(Debug, Clone)]
struct Function {
    name: String,
    params: Vec<Param>,
    returns: String,
}

/// Represents a STUNIR spec
#[derive(Debug, Clone)]
struct Spec {
    name: String,
    version: String,
    functions: Vec<Function>,
    exports: Vec<String>,
}

/// Represents the IR (Intermediate Representation)
#[derive(Debug, Clone)]
struct IR {
    ir_version: String,
    ir_epoch: u64,
    ir_spec_hash: String,
    module_name: String,
    module_version: String,
    functions: Vec<Function>,
    exports: Vec<String>,
}

/// Represents a receipt
#[derive(Debug, Clone)]
struct Receipt {
    receipt_version: String,
    receipt_epoch: u64,
    module_name: String,
    ir_hash: String,
    spec_hash: String,
    function_count: usize,
    receipt_hash: String,
}

// =============================================================================
// Canonical JSON Implementation
// =============================================================================

/// Generate canonical JSON for a key-value map
/// Uses BTreeMap for deterministic key ordering
fn canonical_json_map(map: &BTreeMap<String, String>) -> String {
    let pairs: Vec<String> = map
        .iter()
        .map(|(k, v)| format!("\"{}\":\"{}\"", k, v))
        .collect();
    format!("{{{}}}", pairs.join(","))
}

/// Simple SHA-256 implementation (educational purposes)
/// In production, use the sha2 crate
fn compute_sha256(data: &str) -> String {
    // Simplified hash for demonstration
    // In production: use sha2::Sha256
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in data.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}{:016x}{:016x}{:016x}", hash, hash ^ 0x12345678, hash ^ 0x9abcdef0, hash ^ 0xdeadbeef)
}

/// Get current epoch timestamp
fn get_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// =============================================================================
// Spec Operations
// =============================================================================

/// Create a sample spec for demonstration
fn create_sample_spec() -> Spec {
    Spec {
        name: "example_module".to_string(),
        version: "1.0.0".to_string(),
        functions: vec![
            Function {
                name: "add".to_string(),
                params: vec![
                    Param { name: "a".to_string(), param_type: "i32".to_string() },
                    Param { name: "b".to_string(), param_type: "i32".to_string() },
                ],
                returns: "i32".to_string(),
            },
            Function {
                name: "multiply".to_string(),
                params: vec![
                    Param { name: "x".to_string(), param_type: "i32".to_string() },
                    Param { name: "y".to_string(), param_type: "i32".to_string() },
                ],
                returns: "i32".to_string(),
            },
        ],
        exports: vec!["add".to_string(), "multiply".to_string()],
    }
}

/// Compute hash of a spec
fn hash_spec(spec: &Spec) -> String {
    let mut map = BTreeMap::new();
    map.insert("name".to_string(), spec.name.clone());
    map.insert("version".to_string(), spec.version.clone());
    map.insert("function_count".to_string(), spec.functions.len().to_string());
    compute_sha256(&canonical_json_map(&map))
}

// =============================================================================
// IR Generation
// =============================================================================

/// Convert spec to IR
fn spec_to_ir(spec: &Spec) -> IR {
    println!("🔄 Converting spec to IR...");
    
    let spec_hash = hash_spec(spec);
    
    let ir = IR {
        ir_version: "1.0.0".to_string(),
        ir_epoch: get_epoch(),
        ir_spec_hash: spec_hash,
        module_name: spec.name.clone(),
        module_version: spec.version.clone(),
        functions: spec.functions.clone(),
        exports: spec.exports.clone(),
    };
    
    println!("   ✅ Generated IR with {} functions", ir.functions.len());
    ir
}

/// Compute hash of IR
fn hash_ir(ir: &IR) -> String {
    let mut map = BTreeMap::new();
    map.insert("ir_version".to_string(), ir.ir_version.clone());
    map.insert("ir_spec_hash".to_string(), ir.ir_spec_hash.clone());
    map.insert("module_name".to_string(), ir.module_name.clone());
    map.insert("function_count".to_string(), ir.functions.len().to_string());
    compute_sha256(&canonical_json_map(&map))
}

// =============================================================================
// Receipt Generation
// =============================================================================

/// Generate a receipt from IR
fn generate_receipt(ir: &IR) -> Receipt {
    println!("📝 Generating receipt...");
    
    let ir_hash = hash_ir(ir);
    
    let mut receipt = Receipt {
        receipt_version: "1.0.0".to_string(),
        receipt_epoch: get_epoch(),
        module_name: ir.module_name.clone(),
        ir_hash: ir_hash.clone(),
        spec_hash: ir.ir_spec_hash.clone(),
        function_count: ir.functions.len(),
        receipt_hash: String::new(),
    };
    
    // Compute receipt hash
    let mut map = BTreeMap::new();
    map.insert("ir_hash".to_string(), receipt.ir_hash.clone());
    map.insert("module_name".to_string(), receipt.module_name.clone());
    map.insert("spec_hash".to_string(), receipt.spec_hash.clone());
    receipt.receipt_hash = compute_sha256(&canonical_json_map(&map));
    
    println!("   ✅ Receipt generated: {}...", &receipt.receipt_hash[..16]);
    receipt
}

// =============================================================================
// Determinism Verification
// =============================================================================

/// Verify determinism by computing multiple hashes
fn verify_determinism(spec: &Spec, iterations: u32) -> bool {
    println!("🔍 Verifying determinism ({} iterations)...", iterations);
    
    let mut hashes = Vec::new();
    
    for i in 0..iterations {
        let hash = hash_spec(spec);
        println!("   Round {}: {}...", i + 1, &hash[..16]);
        hashes.push(hash);
    }
    
    // Check all hashes are identical
    let first = &hashes[0];
    let is_deterministic = hashes.iter().all(|h| h == first);
    
    if is_deterministic {
        println!("   ✅ Determinism verified!");
    } else {
        println!("   ❌ DETERMINISM FAILURE - hashes differ!");
    }
    
    is_deterministic
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    println!("============================================================");
    println!("STUNIR Basic Usage Example - Rust");
    println!("============================================================\n");
    
    // Step 1: Create sample spec
    println!("📄 Creating sample spec...");
    let spec = create_sample_spec();
    println!("   Module: {}", spec.name);
    println!("   Version: {}", spec.version);
    println!("   Functions: {}\n", spec.functions.len());
    
    // Step 2: Convert to IR
    let ir = spec_to_ir(&spec);
    println!();
    
    // Step 3: Generate receipt
    let receipt = generate_receipt(&ir);
    println!();
    
    // Step 4: Verify determinism
    verify_determinism(&spec, 3);
    println!();
    
    // Step 5: Display results
    println!("============================================================");
    println!("Results Summary");
    println!("============================================================");
    println!("Module Name:    {}", ir.module_name);
    println!("IR Hash:        {}", receipt.ir_hash);
    println!("Receipt Hash:   {}", receipt.receipt_hash);
    println!("Functions:      {}", receipt.function_count);
    println!("Exports:        {}\n", spec.exports.join(", "));
    
    println!("✅ Basic usage example completed successfully!");
}
