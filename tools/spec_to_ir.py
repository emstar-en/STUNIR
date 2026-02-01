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
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [spec_to_ir] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add local tools dir to path to allow importing lib
sys.path.insert(0, str(Path(__file__).parent))

try:
    from lib import toolchain
except ImportError:
    # Fallback if running from root without setting PYTHONPATH
    sys.path.insert(0, "tools")
    from lib import toolchain

try:
    import ir_converter
except ImportError:
    # Try importing from tools directory
    sys.path.insert(0, str(Path(__file__).parent))
    import ir_converter


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
        "byte[]": "byte[]",  # Fixed: keep original type
        "void": "void",
    }
    return type_map.get(type_str, type_str)


def convert_type_ref(type_ref: Any) -> Any:
    """Convert spec type reference to IR type reference (v0.8.8+).
    
    Handles both simple types and complex types (array, map, set, optional).
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
        
        return result
    
    return type_ref


def convert_spec_to_ir(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert spec JSON to semantic IR format."""
    
    # Extract module information
    module_field = spec.get("module", "unknown")
    if isinstance(module_field, dict):
        # Full module spec with nested structure
        module_name = module_field.get("name", "unknown")
        docstring = spec.get("description", "")
        module_dict = module_field
    else:
        # Simple string module name
        module_name = module_field
        docstring = spec.get("description", "")
        module_dict = {}
    
    # Build types list from module.types or spec.types
    types = []
    type_specs = module_dict.get("types", spec.get("types", []))
    for type_spec in type_specs:
        type_entry = {
            "name": type_spec.get("name", ""),
            "fields": []
        }
        if "docstring" in type_spec:
            type_entry["docstring"] = type_spec["docstring"]
        
        # Convert fields
        for field in type_spec.get("fields", []):
            field_entry = {
                "name": field.get("name", ""),
                "type": convert_type_ref(field.get("type", "void"))
            }
            if "optional" in field:
                field_entry["optional"] = field["optional"]
            type_entry["fields"].append(field_entry)
        
        types.append(type_entry)
    
    # Build functions list from module.functions or spec.functions
    functions = []
    func_specs = module_dict.get("functions", spec.get("functions", []))
    for func_spec in func_specs:
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
        
        # Convert body to stunir_ir_v1 step format (with control flow support for v0.6.1)
        def convert_statements(stmts):
            """Recursively convert statements to IR steps."""
            result_steps = []
            for stmt in stmts:
                if "type" not in stmt:
                    continue
                
                stmt_type = stmt.get("type", "nop")
                
                # Handle control flow statements (v0.6.1)
                if stmt_type == "if":
                    step = {
                        "op": "if",
                        "condition": stmt.get("condition", "true"),
                        "then_block": convert_statements(stmt.get("then", [])),
                    }
                    if "else" in stmt:
                        step["else_block"] = convert_statements(stmt.get("else", []))
                    result_steps.append(step)
                
                elif stmt_type == "while":
                    step = {
                        "op": "while",
                        "condition": stmt.get("condition", "true"),
                        "body": convert_statements(stmt.get("body", []))
                    }
                    result_steps.append(step)
                
                elif stmt_type == "for":
                    step = {
                        "op": "for",
                        "init": stmt.get("init", ""),
                        "condition": stmt.get("condition", "true"),
                        "increment": stmt.get("increment", ""),
                        "body": convert_statements(stmt.get("body", []))
                    }
                    result_steps.append(step)
                
                elif stmt_type == "switch":
                    # v0.9.0: switch/case statement
                    step = {
                        "op": "switch",
                        "expr": stmt.get("expr", "0")
                    }
                    
                    # Process cases
                    cases = []
                    for case in stmt.get("cases", []):
                        case_entry = {
                            "value": case.get("value", 0),
                            "body": convert_statements(case.get("body", []))
                        }
                        cases.append(case_entry)
                    
                    if cases:
                        step["cases"] = cases
                    
                    # Process default case
                    if "default" in stmt:
                        step["default"] = convert_statements(stmt.get("default", []))
                    
                    result_steps.append(step)
                
                # Handle regular statements
                elif stmt_type == "call":
                    # Build function call expression: func_name(arg1, arg2, ...)
                    called_func = stmt.get("func", "unknown")
                    call_args = stmt.get("args", [])
                    if isinstance(call_args, list):
                        args_str = ", ".join(str(arg) for arg in call_args)
                    else:
                        args_str = str(call_args)
                    
                    step = {"op": "call", "value": f"{called_func}({args_str})"}
                    
                    # Handle optional assignment
                    if "assign_to" in stmt:
                        step["target"] = stmt["assign_to"]
                    
                    result_steps.append(step)
                
                elif stmt_type == "var_decl":
                    step = {"op": "assign"}
                    if "var_name" in stmt:
                        step["target"] = stmt["var_name"]
                    if "init" in stmt:
                        step["value"] = stmt["init"]
                    result_steps.append(step)
                
                elif stmt_type == "assign":
                    step = {"op": "assign"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "value" in stmt:
                        step["value"] = stmt["value"]
                    result_steps.append(step)
                
                elif stmt_type == "return":
                    step = {"op": "return"}
                    if "value" in stmt:
                        step["value"] = stmt["value"]
                    result_steps.append(step)
                
                elif stmt_type == "break":
                    # v0.9.0: break statement
                    step = {"op": "break"}
                    result_steps.append(step)
                
                elif stmt_type == "continue":
                    # v0.9.0: continue statement
                    step = {"op": "continue"}
                    result_steps.append(step)
                
                elif stmt_type == "try":
                    # v0.8.7: try/catch/finally exception handling
                    step = {"op": "try"}
                    
                    # Parse try block
                    if "try" in stmt:
                        step["try_block"] = convert_statements(stmt.get("try", []))
                    elif "body" in stmt:
                        step["try_block"] = convert_statements(stmt.get("body", []))
                    else:
                        step["try_block"] = []
                    
                    # Parse catch blocks
                    catches = stmt.get("catch", stmt.get("catches", []))
                    if catches:
                        catch_blocks = []
                        # Handle single catch or list of catches
                        if isinstance(catches, dict):
                            catches = [catches]
                        for catch in catches:
                            catch_entry = {
                                "exception_type": catch.get("exception_type", catch.get("type", "*")),
                                "body": convert_statements(catch.get("body", []))
                            }
                            if "exception_var" in catch or "var" in catch:
                                catch_entry["exception_var"] = catch.get("exception_var", catch.get("var", "e"))
                            catch_blocks.append(catch_entry)
                        step["catch_blocks"] = catch_blocks
                    
                    # Parse finally block
                    if "finally" in stmt:
                        step["finally_block"] = convert_statements(stmt.get("finally", []))
                    
                    result_steps.append(step)
                
                elif stmt_type == "throw":
                    # v0.8.7: throw exception
                    step = {"op": "throw"}
                    if "exception_type" in stmt or "type" in stmt:
                        step["exception_type"] = stmt.get("exception_type", stmt.get("type", "Exception"))
                    if "message" in stmt or "exception_message" in stmt:
                        step["exception_message"] = stmt.get("exception_message", stmt.get("message", ""))
                    result_steps.append(step)
                
                elif stmt_type == "comment":
                    step = {"op": "nop"}
                    result_steps.append(step)
                
                # v0.8.8: Data structure operations - Arrays
                elif stmt_type == "array_new":
                    step = {"op": "array_new"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "element_type" in stmt:
                        step["element_type"] = convert_type_ref(stmt["element_type"])
                    if "size" in stmt:
                        step["size"] = stmt["size"]
                    if "value" in stmt:
                        step["value"] = stmt["value"]
                    result_steps.append(step)
                
                elif stmt_type == "array_get":
                    step = {"op": "array_get"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "source" in stmt:
                        step["source"] = stmt["source"]
                    if "index" in stmt:
                        step["index"] = stmt["index"]
                    result_steps.append(step)
                
                elif stmt_type == "array_set":
                    step = {"op": "array_set"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "index" in stmt:
                        step["index"] = stmt["index"]
                    if "value" in stmt:
                        step["value"] = stmt["value"]
                    result_steps.append(step)
                
                elif stmt_type == "array_push":
                    step = {"op": "array_push"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "value" in stmt:
                        step["value"] = stmt["value"]
                    result_steps.append(step)
                
                elif stmt_type == "array_pop":
                    step = {"op": "array_pop"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "source" in stmt:
                        step["source"] = stmt["source"]
                    result_steps.append(step)
                
                elif stmt_type == "array_len":
                    step = {"op": "array_len"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "source" in stmt:
                        step["source"] = stmt["source"]
                    result_steps.append(step)
                
                # v0.8.8: Data structure operations - Maps
                elif stmt_type == "map_new":
                    step = {"op": "map_new"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "key_type" in stmt:
                        step["key_type"] = convert_type_ref(stmt["key_type"])
                    if "value_type" in stmt:
                        step["value_type"] = convert_type_ref(stmt["value_type"])
                    if "value" in stmt:
                        step["value"] = stmt["value"]
                    result_steps.append(step)
                
                elif stmt_type == "map_get":
                    step = {"op": "map_get"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "source" in stmt:
                        step["source"] = stmt["source"]
                    if "key" in stmt:
                        step["key"] = stmt["key"]
                    result_steps.append(step)
                
                elif stmt_type == "map_set":
                    step = {"op": "map_set"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "key" in stmt:
                        step["key"] = stmt["key"]
                    if "value" in stmt:
                        step["value"] = stmt["value"]
                    result_steps.append(step)
                
                elif stmt_type == "map_delete":
                    step = {"op": "map_delete"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "key" in stmt:
                        step["key"] = stmt["key"]
                    result_steps.append(step)
                
                elif stmt_type == "map_has":
                    step = {"op": "map_has"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "source" in stmt:
                        step["source"] = stmt["source"]
                    if "key" in stmt:
                        step["key"] = stmt["key"]
                    result_steps.append(step)
                
                elif stmt_type == "map_keys":
                    step = {"op": "map_keys"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "source" in stmt:
                        step["source"] = stmt["source"]
                    result_steps.append(step)
                
                # v0.8.8: Data structure operations - Sets
                elif stmt_type == "set_new":
                    step = {"op": "set_new"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "element_type" in stmt:
                        step["element_type"] = convert_type_ref(stmt["element_type"])
                    if "value" in stmt:
                        step["value"] = stmt["value"]
                    result_steps.append(step)
                
                elif stmt_type == "set_add":
                    step = {"op": "set_add"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "value" in stmt:
                        step["value"] = stmt["value"]
                    result_steps.append(step)
                
                elif stmt_type == "set_remove":
                    step = {"op": "set_remove"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "value" in stmt:
                        step["value"] = stmt["value"]
                    result_steps.append(step)
                
                elif stmt_type == "set_has":
                    step = {"op": "set_has"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "source" in stmt:
                        step["source"] = stmt["source"]
                    if "value" in stmt:
                        step["value"] = stmt["value"]
                    result_steps.append(step)
                
                elif stmt_type == "set_union":
                    step = {"op": "set_union"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "source" in stmt:
                        step["source"] = stmt["source"]
                    if "source2" in stmt:
                        step["source2"] = stmt["source2"]
                    result_steps.append(step)
                
                elif stmt_type == "set_intersect":
                    step = {"op": "set_intersect"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "source" in stmt:
                        step["source"] = stmt["source"]
                    if "source2" in stmt:
                        step["source2"] = stmt["source2"]
                    result_steps.append(step)
                
                # v0.8.8: Data structure operations - Structs
                elif stmt_type == "struct_new":
                    step = {"op": "struct_new"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "struct_type" in stmt:
                        step["struct_type"] = stmt["struct_type"]
                    if "fields" in stmt:
                        step["fields"] = stmt["fields"]
                    result_steps.append(step)
                
                elif stmt_type == "struct_get":
                    step = {"op": "struct_get"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "source" in stmt:
                        step["source"] = stmt["source"]
                    if "field" in stmt:
                        step["field"] = stmt["field"]
                    result_steps.append(step)
                
                elif stmt_type == "struct_set":
                    step = {"op": "struct_set"}
                    if "target" in stmt:
                        step["target"] = stmt["target"]
                    if "field" in stmt:
                        step["field"] = stmt["field"]
                    if "value" in stmt:
                        step["value"] = stmt["value"]
                    result_steps.append(step)
                
                else:
                    # Unknown statement type, convert to nop
                    result_steps.append({"op": "nop"})
            
            return result_steps
        
        steps = convert_statements(func_spec.get("body", []))
        
        func_entry = {
            "name": func_name,
            "args": args,
            "return_type": return_type,
        }
        if "docstring" in func_spec:
            func_entry["docstring"] = func_spec["docstring"]
        if steps:
            func_entry["steps"] = steps
        
        functions.append(func_entry)
    
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
    logger.info(f"Processing spec file: {spec_path}")
    
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
    ap.add_argument("--flat-ir", action="store_true", 
                    help="Generate flattened IR for SPARK compatibility (single-level nesting)")
    ap.add_argument("--flat-out", default=None,
                    help="Output path for flattened IR (default: <out>_flat.json)")
    a = ap.parse_args()

    # 1. Enforce Toolchain Lock
    logger.info(f"Loading toolchain from {a.lockfile}...")
    try:
        toolchain.load(a.lockfile)
        py_path = toolchain.get_tool("python")
        if py_path:
            logger.info(f"Verified Python runtime: {py_path}")
    except Exception as e:
        logger.error(f"Toolchain verification failed: {e}")
        sys.exit(1)

    spec_root = Path(a.spec_root)
    out_path = Path(a.out)

    if not spec_root.exists():
        logger.error(f"Spec root not found: {spec_root}")
        sys.exit(1)

    # 2. Process Specs and generate semantic IR
    logger.info(f"Processing specs from {spec_root}...")

    # Collect all spec files
    spec_files = []
    for root, dirs, files in os.walk(spec_root):
        dirs.sort()
        files.sort()
        for f in files:
            if f.endswith(".json"):
                spec_files.append(Path(root) / f)
    
    if not spec_files:
        logger.error(f"No spec files found in {spec_root}")
        sys.exit(1)
    
    # Process the first spec file (or merge multiple if needed)
    # For now, we'll process the first one
    ir = process_spec_file(spec_files[0])
    
    # If there are multiple spec files, merge them
    if len(spec_files) > 1:
        logger.info(f"Found {len(spec_files)} spec files, merging...")
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

    logger.info(f"Generated semantic IR with {len(ir['functions'])} functions")
    logger.info(f"Wrote semantic IR to {out_path}")
    
    # 4. Generate flattened IR if requested (for SPARK compatibility)
    if a.flat_ir:
        logger.info("Generating flattened IR for SPARK compatibility...")
        flat_ir = ir_converter.convert_ir_module(ir)
        
        # Determine output path for flattened IR
        if a.flat_out:
            flat_out_path = Path(a.flat_out)
        else:
            # Default: add _flat suffix before extension
            flat_out_path = out_path.parent / (out_path.stem + "_flat" + out_path.suffix)
        
        flat_out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(flat_out_path, "w", encoding='utf-8', newline='\n') as f:
            json.dump(flat_ir, f, indent=2, sort_keys=False)
            f.write('\n')
        
        logger.info(f"Wrote flattened IR (schema: {flat_ir['schema']}) to {flat_out_path}")
        logger.info(f"Flattened IR contains {sum(len(f.get('steps', [])) for f in flat_ir['functions'])} total steps")


if __name__ == "__main__":
    main()
