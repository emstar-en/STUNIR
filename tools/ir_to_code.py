#!/usr/bin/env python3
"""
STUNIR IR to Code Generator - Python REFERENCE Implementation

WARNING: This is a REFERENCE IMPLEMENTATION for readability purposes only.
         DO NOT use this file for production, verification, or safety-critical
         applications.

PRIMARY IMPLEMENTATION: Ada SPARK
    Location: tools/spark/bin/stunir_ir_to_code_main
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
import json
import os
from typing import Any, Dict, List

# Type mapping from IR types to C types
# Used for generating function signatures and variable declarations
IR_TO_C_TYPE_MAP = {
    # Signed integers
    "i8": "int8_t",
    "int8": "int8_t",
    "int8_t": "int8_t",
    "i16": "int16_t",
    "int16": "int16_t",
    "int16_t": "int16_t",
    "i32": "int32_t",
    "int": "int32_t",
    "int32": "int32_t",
    "int32_t": "int32_t",
    "i64": "int64_t",
    "int64": "int64_t",
    "int64_t": "int64_t",
    # Unsigned integers
    "u8": "uint8_t",
    "uint8": "uint8_t",
    "uint8_t": "uint8_t",
    "u16": "uint16_t",
    "uint16": "uint16_t",
    "uint16_t": "uint16_t",
    "u32": "uint32_t",
    "uint32": "uint32_t",
    "uint32_t": "uint32_t",
    "u64": "uint64_t",
    "uint64": "uint64_t",
    "uint64_t": "uint64_t",
    # Floating point
    "f32": "float",
    "float": "float",
    "f64": "double",
    "double": "double",
    # Other primitives
    "bool": "bool",
    "boolean": "bool",
    "string": "const char*",
    "char*": "const char*",
    "cstring": "const char*",
    "void": "void",
}


def c_type(type_str: str) -> str:
    """Convert IR type string to C type declaration.
    
    Args:
        type_str: IR type identifier (e.g., "i32", "bool", "string")
        
    Returns:
        C type string (e.g., "int32_t", "bool", "const char*")
        
    Note:
        Unknown types default to "void" for safety.
    """
    return IR_TO_C_TYPE_MAP.get(type_str, "void")


def c_default_return(type_str: str) -> str:
    """Generate default return value for a given type.
    
    Used for generating stub function bodies that compile
    but indicate unimplemented logic.
    
    Args:
        type_str: IR type identifier
        
    Returns:
        C literal for default value of that type
    """
    if type_str in ("i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", 
                    "int", "int32", "int64", "uint32", "uint64"):
        return "0"
    if type_str in ("f32", "f64", "float", "double"):
        return "0.0"
    if type_str in ("bool", "boolean"):
        return "false"
    if type_str in ("string", "char*", "cstring"):
        return "NULL"
    return "0"


def normalize_functions(ir: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract function list from IR module with multiple key support.
    
    Handles both "functions" and "ir_functions" keys for backward
    compatibility with different IR generation tools.
    
    Args:
        ir: IR module dictionary
        
    Returns:
        List of function definition dictionaries
    """
    if "functions" in ir and isinstance(ir["functions"], list):
        return ir["functions"]
    if "ir_functions" in ir and isinstance(ir["ir_functions"], list):
        return ir["ir_functions"]
    return []


def normalize_args(func: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract argument list from function definition.
    
    Handles both "args" and "params" keys for compatibility
    with different spec formats.
    
    Args:
        func: Function definition dictionary
        
    Returns:
        List of argument dictionaries with "name" and "type" keys
    """
    args = func.get("args")
    if isinstance(args, list):
        return args
    params = func.get("params")
    if isinstance(params, list):
        return params
    return []


def normalize_return_type(func: Dict[str, Any]) -> str:
    """Extract return type from function definition.
    
    Handles both "return_type" and "returns" keys for compatibility.
    Defaults to "void" if not specified.
    
    Args:
        func: Function definition dictionary
        
    Returns:
        IR type string for return type
    """
    if "return_type" in func:
        return str(func.get("return_type") or "void")
    if "returns" in func:
        return str(func.get("returns") or "void")
    return "void"


def emit_c(ir: Dict[str, Any]) -> str:
    """Generate C code from IR module.
    
    Generates stub implementations for all functions in the IR.
    Each function includes:
    - Proper C type declarations
    - Parameter list with types
    - TODO comment indicating unimplemented status
    - Default return value
    
    Args:
        ir: IR module dictionary containing functions
        
    Returns:
        Complete C source code as string
        
    Note:
        This generates stubs only. Full implementation would
        require processing IR steps (load, store, call, etc.)
    """
    # Get module name for header comment
    module_name = ir.get("module_name") or ir.get("ir_module") or "module"
    
    lines = [
        "/* STUNIR Generated Code",
        " * Generated by: stunir_ir_to_code_python v0.2.0",
        f" * Module: {module_name}",
        " *",
        " * WARNING: This is a REFERENCE IMPLEMENTATION.",
        " * For production use, use the Ada SPARK implementation.",
        " */",
        "",
        "#include <stdint.h>",
        "#include <stdbool.h>",
        "#include <stddef.h>",
        "",
    ]
    
    for func in normalize_functions(ir):
        name = func.get("name") or "func"
        ret_type = normalize_return_type(func)
        args = normalize_args(func)
        
        # Build parameter list
        arg_parts = []
        for arg in args:
            if isinstance(arg, dict):
                arg_name = arg.get("name") or "arg"
                arg_type = c_type(str(arg.get("type") or arg.get("arg_type") or "void"))
                arg_parts.append(f"{arg_type} {arg_name}")
            else:
                arg_parts.append("void")
        
        args_str = ", ".join(arg_parts) if arg_parts else "void"
        
        # Function signature
        lines.append(f"{c_type(ret_type)} {name}({args_str}) {{")
        
        # Stub body with TODO marker for downstream processing
        lines.append("    /* TODO: Implement - stub generated from IR */")
        
        if c_type(ret_type) != "void":
            lines.append(f"    return {c_default_return(ret_type)};")
        
        lines.append("}")
        lines.append("")
    
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    """CLI entry point for IR to code generation.
    
    Supports:
    - --ir: Path to IR JSON file
    - --lang: Target language (must be "c")
    - --templates: Optional template directory (unused in Python)
    - --out: Output directory for generated code
    """
    parser = argparse.ArgumentParser(
        description="STUNIR IR to Code Generator (Python Reference)"
    )
    parser.add_argument("--ir", required=True, help="Path to IR JSON file")
    parser.add_argument("--lang", required=True, help="Target language (c)")
    parser.add_argument("--templates", required=False, help="Template directory (unused)")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    if args.lang.lower() != "c":
        print(f"Error: Language '{args.lang}' not supported. Use 'c'.", file=os.sys.stderr)
        raise SystemExit(1)

    # Load IR from JSON
    with open(args.ir, "r", encoding="utf-8") as f:
        ir = json.load(f)

    # Ensure output directory exists
    output_dir = args.out
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "output.c")

    # Write generated code
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(emit_c(ir))
    
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
