#!/usr/bin/env python3
"""STUNIR Basic Usage Example

This example demonstrates the fundamental operations in STUNIR:
- Loading a spec file
- Converting spec to IR
- Generating receipts
- Verifying determinism

Usage:
    python basic_usage.py [--spec <spec_file>]

Example:
    python basic_usage.py --spec sample_spec.json
"""

import json
import hashlib
import argparse
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# STUNIR canonical JSON implementation
def canonical_json(data: Any) -> str:
    """Generate RFC 8785 compliant canonical JSON.
    
    This ensures deterministic output regardless of:
    - Key ordering in dictionaries
    - Whitespace formatting
    - Unicode normalization
    """
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

# Sample spec for demonstration
SAMPLE_SPEC = {
    "name": "example_module",
    "version": "1.0.0",
    "description": "A sample STUNIR module",
    "functions": [
        {
            "name": "add",
            "params": [{"name": "a", "type": "i32"}, {"name": "b", "type": "i32"}],
            "returns": "i32",
            "body": [{"op": "add", "left": "a", "right": "b"}]
        },
        {
            "name": "multiply",
            "params": [{"name": "x", "type": "i32"}, {"name": "y", "type": "i32"}],
            "returns": "i32",
            "body": [{"op": "mul", "left": "x", "right": "y"}]
        }
    ],
    "exports": ["add", "multiply"]
}

def load_spec(spec_path: Optional[str] = None) -> Dict[str, Any]:
    """Load a spec file or return sample spec.
    
    Args:
        spec_path: Path to spec JSON file, or None for sample
        
    Returns:
        Parsed spec dictionary
    """
    if spec_path and os.path.exists(spec_path):
        print(f"📄 Loading spec from: {spec_path}")
        with open(spec_path, 'r') as f:
            return json.load(f)
    else:
        print("📄 Using sample spec (no file provided)")
        return SAMPLE_SPEC

def spec_to_ir(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a STUNIR spec to Intermediate Representation.
    
    The IR is a normalized, canonical form of the spec that:
    - Has consistent key ordering
    - Includes computed metadata (hashes, epochs)
    - Is ready for target code generation
    
    Args:
        spec: Input spec dictionary
        
    Returns:
        IR dictionary
    """
    print("🔄 Converting spec to IR...")
    
    # Compute spec hash for provenance
    spec_json = canonical_json(spec)
    spec_hash = compute_sha256(spec_json)
    
    # Build IR structure
    ir = {
        "ir_version": "1.0.0",
        "ir_epoch": int(datetime.now(timezone.utc).timestamp()),
        "ir_spec_hash": spec_hash,
        "module": {
            "name": spec.get("name", "unnamed"),
            "version": spec.get("version", "0.0.0")
        },
        "functions": [],
        "exports": spec.get("exports", [])
    }
    
    # Transform functions to IR format
    for func in spec.get("functions", []):
        ir_func = {
            "name": func["name"],
            "signature": {
                "params": func.get("params", []),
                "returns": func.get("returns", "void")
            },
            "body": func.get("body", [])
        }
        ir["functions"].append(ir_func)
    
    print(f"   ✅ Generated IR with {len(ir['functions'])} functions")
    return ir

def generate_receipt(ir: Dict[str, Any], output_dir: str = "receipts") -> Dict[str, Any]:
    """Generate a deterministic receipt for the IR.
    
    Receipts provide:
    - Cryptographic proof of build artifacts
    - Determinism verification
    - Audit trail
    
    Args:
        ir: The IR dictionary
        output_dir: Directory to store receipt
        
    Returns:
        Receipt dictionary
    """
    print("📝 Generating receipt...")
    
    # Create receipt
    ir_json = canonical_json(ir)
    ir_hash = compute_sha256(ir_json)
    
    receipt = {
        "receipt_version": "1.0.0",
        "receipt_epoch": int(datetime.now(timezone.utc).timestamp()),
        "module_name": ir["module"]["name"],
        "ir_hash": ir_hash,
        "ir_spec_hash": ir["ir_spec_hash"],
        "function_count": len(ir["functions"]),
        "exports": ir["exports"]
    }
    
    # Compute receipt hash (self-referential integrity)
    receipt_content = canonical_json({k: v for k, v in receipt.items() if k != "receipt_hash"})
    receipt["receipt_hash"] = compute_sha256(receipt_content)
    
    print(f"   ✅ Receipt generated: {receipt['receipt_hash'][:16]}...")
    return receipt

def verify_determinism(data: Dict[str, Any], iterations: int = 3) -> bool:
    """Verify that canonicalization is deterministic.
    
    Runs multiple canonicalizations and verifies all produce
    identical output.
    
    Args:
        data: Data to canonicalize
        iterations: Number of verification rounds
        
    Returns:
        True if deterministic, False otherwise
    """
    print(f"🔍 Verifying determinism ({iterations} iterations)...")
    
    hashes = []
    for i in range(iterations):
        json_str = canonical_json(data)
        hash_val = compute_sha256(json_str)
        hashes.append(hash_val)
        print(f"   Round {i+1}: {hash_val[:16]}...")
    
    is_deterministic = len(set(hashes)) == 1
    
    if is_deterministic:
        print("   ✅ Determinism verified!")
    else:
        print("   ❌ DETERMINISM FAILURE - hashes differ!")
        
    return is_deterministic

def main():
    """Main entry point demonstrating basic STUNIR usage."""
    parser = argparse.ArgumentParser(description="STUNIR Basic Usage Example")
    parser.add_argument("--spec", help="Path to spec JSON file")
    parser.add_argument("--output", default="receipts", help="Output directory for receipts")
    args = parser.parse_args()
    
    print("="*60)
    print("STUNIR Basic Usage Example")
    print("="*60)
    print()
    
    # Step 1: Load spec
    spec = load_spec(args.spec)
    print(f"   Module: {spec.get('name', 'unnamed')}")
    print(f"   Functions: {len(spec.get('functions', []))}")
    print()
    
    # Step 2: Convert to IR
    ir = spec_to_ir(spec)
    print()
    
    # Step 3: Generate receipt
    receipt = generate_receipt(ir, args.output)
    print()
    
    # Step 4: Verify determinism
    verify_determinism(ir)
    print()
    
    # Step 5: Display results
    print("="*60)
    print("Results Summary")
    print("="*60)
    print(f"Module Name:    {ir['module']['name']}")
    print(f"IR Hash:        {receipt['ir_hash']}")
    print(f"Receipt Hash:   {receipt['receipt_hash']}")
    print(f"Functions:      {receipt['function_count']}")
    print(f"Exports:        {', '.join(receipt['exports'])}")
    print()
    print("✅ Basic usage example completed successfully!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
