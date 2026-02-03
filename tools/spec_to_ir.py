#!/usr/bin/env python3
"""
STUNIR Spec to IR Converter - Python REFERENCE Implementation

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
"""

import argparse
import hashlib
import json
import sys
import os
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Configure logging for debugging and audit trails
# Format includes timestamp, level, and component name for traceability
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [spec_to_ir] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for importing local modules
sys.path.insert(0, str(Path(__file__).parent))

# Import toolchain utilities - handles path resolution and environment setup
try:
    from lib import toolchain
except ImportError:
    sys.path.insert(0, "tools")
    from lib import toolchain

# Import IR conversion utilities
# ir_converter: handles nested-to-flat IR transformations
try:
    import ir_converter
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    import ir_converter

# emit_ir: generates canonical IR from specs with deterministic output
try:
    from ir_emitter import emit_ir
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent / "ir_emitter"))
    import emit_ir


def sha256_bytes(b: bytes) -> str:
    """Compute SHA256 hash for content addressing and verification.
    
    Used for generating content-based identifiers and receipts.
    """
    return hashlib.sha256(b).hexdigest()


def canon(obj) -> str:
    """Generate canonical JSON representation for deterministic hashing.
    
    Uses sorted keys and compact separators to ensure identical output
    for identical objects regardless of insertion order.
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def convert_type(type_str: str) -> str:
    """Map specification types to IR type system.
    
    Handles primitive types and preserves unknown types as-is.
    This is a pass-through for most cases but provides a hook
    for type system evolution.
    """
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
        "byte[]": "byte[]",
        "void": "void",
    }
    return type_map.get(type_str, type_str)


def convert_type_ref(type_ref: Any) -> Any:
    """Recursively convert type references in specifications.
    
    Handles complex types including:
    - Arrays: element_type, size
    - Maps: key_type, value_type
    - Sets: element_type
    - Optionals: inner type
    - Generics: base type with type arguments
    
    Returns normalized type structure for IR consumption.
    """
    if isinstance(type_ref, str):
        return convert_type(type_ref)

    if isinstance(type_ref, dict):
        kind = type_ref.get("kind")
        result = {"kind": kind}

        if kind == "array":
            if "element_type" in type_ref:
                result["element_type"] = convert_type_ref(type_ref["element_type"])
            if "size" in type_ref:
                result["size"] = type_ref["size"]
        elif kind == "map":
            if "key_type" in type_ref:
                result["key_type"] = convert_type_ref(type_ref["key_type"])
            if "value_type" in type_ref:
                result["value_type"] = convert_type_ref(type_ref["value_type"])
        elif kind == "set":
            if "element_type" in type_ref:
                result["element_type"] = convert_type_ref(type_ref["element_type"])
        elif kind == "optional":
            if "inner" in type_ref:
                result["inner"] = convert_type_ref(type_ref["inner"])
        elif kind == "generic":
            if "base" in type_ref:
                result["base"] = convert_type_ref(type_ref["base"])
            if "args" in type_ref and isinstance(type_ref["args"], list):
                result["args"] = [convert_type_ref(a) for a in type_ref["args"]]

        return result

    return type_ref


def load_spec_dir(spec_root: str) -> List[Dict[str, Any]]:
    """Load all JSON specification files from a directory tree.
    
    Recursively searches for .json files and parses them as specs.
    Files are processed in sorted order for deterministic output.
    
    Args:
        spec_root: Root directory containing spec files
        
    Returns:
        List of parsed specification dictionaries
        
    Raises:
        FileNotFoundError: If directory doesn't exist or contains no JSON files
    """
    if not os.path.isdir(spec_root):
        raise FileNotFoundError(f"Spec directory not found: {spec_root}")

    spec_files = []
    for root, _, files in os.walk(spec_root):
        for f in files:
            if f.endswith('.json'):
                spec_files.append(os.path.join(root, f))

    if not spec_files:
        raise FileNotFoundError(f"No JSON specs found in {spec_root}")

    specs = []
    for path in sorted(spec_files):
        with open(path, 'r', encoding='utf-8') as fh:
            specs.append(json.load(fh))

    return specs


def convert_spec_to_ir(specs: List[Dict[str, Any]], emit_comments: bool = True) -> Dict[str, Any]:
    """Convert specification(s) to intermediate representation.
    
    For single spec: converts directly using emit_ir.spec_to_ir
    For multiple specs: merges all specs into single IR module
    
    Merging strategy:
    - Functions: concatenated (preserving order)
    - Types: concatenated (caller must handle duplicates)
    - Imports: concatenated
    - Exports: concatenated
    
    Args:
        specs: List of specification dictionaries
        emit_comments: Whether to include comments in output (unused in Python)
        
    Returns:
        IR module dictionary with ir_functions, ir_types, ir_imports, ir_exports
    """
    if not specs:
        return {}

    # Single spec: direct conversion
    if len(specs) == 1:
        return emit_ir.spec_to_ir(specs[0])

    # Multiple specs: merge into single module
    merged = emit_ir.spec_to_ir(specs[0])
    merged_functions = merged.get("ir_functions", [])
    merged_types = merged.get("ir_types", [])
    merged_imports = merged.get("ir_imports", [])
    merged_exports = merged.get("ir_exports", [])

    for spec in specs[1:]:
        ir_part = emit_ir.spec_to_ir(spec)
        merged_functions.extend(ir_part.get("ir_functions", []))
        merged_types.extend(ir_part.get("ir_types", []))
        merged_imports.extend(ir_part.get("ir_imports", []))
        merged_exports.extend(ir_part.get("ir_exports", []))

    merged["ir_functions"] = merged_functions
    merged["ir_types"] = merged_types
    merged["ir_imports"] = merged_imports
    merged["ir_exports"] = merged_exports

    return merged


def write_output(out_path: str, ir_obj: Dict[str, Any]) -> None:
    """Write IR output to file with canonical formatting.
    
    Creates parent directories if needed.
    Uses Unix line endings for consistency across platforms.
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(canon(ir_obj))


def main() -> None:
    """CLI entry point for spec_to_ir conversion tool.
    
    Supports:
    - --spec-root: Directory containing spec files
    - --out: Output IR JSON file path
    - --emit-comments: Include documentation comments
    - --emit-receipt: Generate verification receipt
    """
    p = argparse.ArgumentParser(description='STUNIR Spec to IR Converter')
    p.add_argument('--spec-root', required=True, help='Root directory containing spec JSON files')
    p.add_argument('--out', required=True, help='Path to output IR JSON file')
    p.add_argument('--emit-comments', action='store_true', help='Include docstrings and comments in IR')
    p.add_argument('--emit-receipt', action='store_true', help='Generate stunir.emit.v1.json receipt')

    args = p.parse_args()

    specs = load_spec_dir(args.spec_root)
    ir_obj = convert_spec_to_ir(specs, emit_comments=args.emit_comments)

    write_output(args.out, ir_obj)
    logger.info(f'IR written: {args.out}')

    if args.emit_receipt:
        receipt = {
            'tool': 'stunir_spec_to_ir',
            'version': '0.2.0',
            'inputs': {
                'spec_root': args.spec_root,
                'spec_count': len(specs),
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            },
            'outputs': {
                os.path.basename(args.out): sha256_bytes(canon(ir_obj).encode('utf-8'))
            }
        }
        receipt_path = os.path.join(os.path.dirname(args.out) or '.', 'stunir.emit.v1.json')
        with open(receipt_path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(receipt, f, indent=2, sort_keys=True)
            f.write('\n')
        logger.info(f'Receipt: {receipt_path}')


if __name__ == '__main__':
    main()
