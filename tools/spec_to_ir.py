#!/usr/bin/env python3
"""
===============================================================================
STUNIR Spec to IR Converter - Python REFERENCE Implementation
===============================================================================

WARNING: This is a REFERENCE IMPLEMENTATION for readability purposes only.
         DO NOT use this file for production, verification, or safety-critical
         applications.

PRIMARY IMPLEMENTATION: Ada SPARK
    Location: tools/spark/bin/stunir_spec_to_ir_main
    Build:    cd tools/spark && gprbuild -P stunir_tools.gpr

This Python version exists to:
1. Provide a readable reference for understanding the algorithm
2. Serve as a fallback when Ada SPARK tools are not available
3. Enable quick prototyping and testing

For all production use cases, use the Ada SPARK implementation which provides:
- Formal verification guarantees
- Deterministic execution
- DO-178C compliance support
- Absence of runtime errors (proven via SPARK)

===============================================================================
"""
import argparse
import hashlib
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Add local tools dir to path to allow importing lib
sys.path.insert(0, str(Path(__file__).parent))

try:
    from lib import toolchain
except ImportError:
    # Fallback if running from root without setting PYTHONPATH
    sys.path.insert(0, "tools")
    from lib import toolchain


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canon(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def convert_type(type_str: str) -> str:
    """Convert spec type to IR type."""
    type_map = {
        "u8": "u8",
        "u16": "u16",
        "u32": "u32",
        "u64": "u64",
        "i8": "i8",
        "i16": "i16",
        "i32": "i32",
        "i64": "i64",
        "f32": "f32",
        "f64": "f64",
        "bool": "bool",
        "string": "string",
        "byte[]": "bytes",
        "void": "void",
    }
    return type_map.get(type_str, type_str)


def convert_spec_to_ir(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert spec JSON to semantic IR format."""
    
    # Extract module information
    module_field = spec.get("module", "unknown")
    if isinstance(module_field, dict):
        # Full module spec with nested structure
        module_name = module_field.get("name", "unknown")
        docstring = spec.get("description", "")
    else:
        # Simple string module name
        module_name = module_field
        docstring = spec.get("description", "")
    
    # Build types list (empty for now, can be enhanced)
    types = []
    
    # Build functions list
    functions = []
    for func_spec in spec.get("functions", []):
        func_name = func_spec.get("name", "")
        
        # Convert parameters
        args = []
        for param in func_spec.get("params", []):
            args.append({
                "name": param.get("name", ""),
                "type": convert_type(param.get("type", "void"))
            })
        
        # Convert return type
        return_type = convert_type(func_spec.get("returns", "void"))
        
        # Convert body (simplified for now)
        steps = []
        for stmt in func_spec.get("body", []):
            steps.append({
                "kind": stmt.get("type", "nop"),
                "data": str(stmt)
            })
        
        functions.append({
            "name": func_name,
            "args": args,
            "return_type": return_type,
            "steps": steps
        })
    
    # Build semantic IR
    ir = {
        "schema": "stunir_ir_v1",
        "ir_version": "v1",
        "module_name": module_name,
        "docstring": docstring,
        "types": types,
        "functions": functions,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    
    return ir


def process_spec_file(spec_path: Path) -> Dict[str, Any]:
    """Process a single spec file and convert to IR."""
    print(f"[INFO] Processing spec file: {spec_path}")
    
    with open(spec_path, 'r', encoding='utf-8') as f:
        spec = json.load(f)
    
    # Convert spec to semantic IR
    ir = convert_spec_to_ir(spec)
    
    return ir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-root", required=True, help="Path to spec root directory")
    ap.add_argument("--out", required=True, help="Output IR JSON file path")
    ap.add_argument("--lockfile", default="local_toolchain.lock.json", help="Path to toolchain lockfile")
    a = ap.parse_args()

    # 1. Enforce Toolchain Lock
    print(f"[INFO] Loading toolchain from {a.lockfile}...")
    try:
        toolchain.load(a.lockfile)
        py_path = toolchain.get_tool("python")
        if py_path:
            print(f"[INFO] Verified Python runtime: {py_path}")
    except Exception as e:
        print(f"[ERROR] Toolchain verification failed: {e}")
        sys.exit(1)

    spec_root = Path(a.spec_root)
    out_path = Path(a.out)

    if not spec_root.exists():
        print(f"[ERROR] Spec root not found: {spec_root}")
        sys.exit(1)

    # 2. Process Specs and generate semantic IR
    print(f"[INFO] Processing specs from {spec_root}...")

    # Collect all spec files
    spec_files = []
    for root, dirs, files in os.walk(spec_root):
        dirs.sort()
        files.sort()
        for f in files:
            if f.endswith(".json"):
                spec_files.append(Path(root) / f)
    
    if not spec_files:
        print(f"[ERROR] No spec files found in {spec_root}")
        sys.exit(1)
    
    # Process the first spec file (or merge multiple if needed)
    # For now, we'll process the first one
    ir = process_spec_file(spec_files[0])
    
    # If there are multiple spec files, merge them
    if len(spec_files) > 1:
        print(f"[INFO] Found {len(spec_files)} spec files, merging...")
        all_functions = list(ir["functions"])
        
        for spec_file in spec_files[1:]:
            additional_ir = process_spec_file(spec_file)
            all_functions.extend(additional_ir.get("functions", []))
        
        ir["functions"] = all_functions

    # 3. Write semantic IR output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding='utf-8', newline='\n') as f:
        json.dump(ir, f, indent=2, sort_keys=True)
        f.write('\n')

    print(f"[INFO] Generated semantic IR with {len(ir['functions'])} functions")
    print(f"[INFO] Wrote semantic IR to {out_path}")


if __name__ == "__main__":
    main()
