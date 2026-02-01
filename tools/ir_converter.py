#!/usr/bin/env python3
"""
STUNIR IR Converter: Nested to Flattened IR Transformation

This module converts nested STUNIR IR (used by Python/Rust pipelines) 
into flattened IR (used by SPARK pipeline).

Why: Ada SPARK cannot dynamically parse nested JSON arrays due to static typing.
Solution: Flatten control flow blocks into a single array with block indices.

Copyright (c) 2026 STUNIR Project
License: MIT
"""

import json
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path


def convert_nested_to_flat(nested_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert nested IR steps to flattened IR with block indices.
    
    Args:
        nested_steps: List of IR steps with nested then_block/else_block/body arrays
    
    Returns:
        Flattened list of IR steps with block_start/block_count indices
    
    Example:
        Input (nested):
            [{"op": "if", "condition": "x > 0", 
              "then_block": [{"op": "assign", "target": "y", "value": "1"}],
              "else_block": [{"op": "assign", "target": "y", "value": "0"}]}]
        
        Output (flat):
            [{"op": "if", "condition": "x > 0", "block_start": 1, "block_count": 1,
              "else_start": 2, "else_count": 1},
             {"op": "assign", "target": "y", "value": "1"},
             {"op": "assign", "target": "y", "value": "0"}]
    """
    flat_steps = []
    
    def flatten_recursive(steps: List[Dict[str, Any]]) -> None:
        """
        Recursively flatten steps, appending to flat_steps.
        
        For v0.8.2: Full multi-level nesting support (2-5 levels).
        Nested control flow is now recursively flattened.
        """
        for step in steps:
            op = step.get("op", "")
            
            if op == "if":
                # Reserve slot for if statement
                if_index = len(flat_steps)
                flat_steps.append(None)  # Placeholder
                
                # Flatten then block (RECURSIVE)
                then_block = step.get("then_block", [])
                then_start = len(flat_steps)
                
                # v0.8.2: Recursively flatten then_block
                flatten_recursive(then_block)  # RECURSIVE CALL
                
                then_count = len(flat_steps) - then_start
                
                # Flatten else block if present (RECURSIVE)
                else_block = step.get("else_block", [])
                else_start = 0
                else_count = 0
                
                if else_block:
                    else_start = len(flat_steps)
                    
                    # v0.8.2: Recursively flatten else_block
                    flatten_recursive(else_block)  # RECURSIVE CALL
                    
                    else_count = len(flat_steps) - else_start
                
                # Fill in the if statement with indices (1-based for Ada compatibility)
                flat_steps[if_index] = {
                    "op": "if",
                    "condition": step.get("condition", ""),
                    "block_start": then_start + 1,  # Convert to 1-based indexing
                    "block_count": then_count,
                    "else_start": else_start + 1 if else_start > 0 else 0,  # Convert to 1-based
                    "else_count": else_count
                }
            
            elif op == "while":
                # Reserve slot for while statement
                while_index = len(flat_steps)
                flat_steps.append(None)
                
                # Flatten body (RECURSIVE)
                body = step.get("body", [])
                body_start = len(flat_steps)
                
                # v0.8.2: Recursively flatten body
                flatten_recursive(body)  # RECURSIVE CALL
                
                body_count = len(flat_steps) - body_start
                
                # Fill in the while statement (1-based for Ada compatibility)
                flat_steps[while_index] = {
                    "op": "while",
                    "condition": step.get("condition", ""),
                    "block_start": body_start + 1,  # Convert to 1-based indexing
                    "block_count": body_count
                }
            
            elif op == "for":
                # Reserve slot for for statement
                for_index = len(flat_steps)
                flat_steps.append(None)
                
                # Flatten body (RECURSIVE)
                body = step.get("body", [])
                body_start = len(flat_steps)
                
                # v0.8.2: Recursively flatten body
                flatten_recursive(body)  # RECURSIVE CALL
                
                body_count = len(flat_steps) - body_start
                
                # Fill in the for statement (1-based for Ada compatibility)
                flat_steps[for_index] = {
                    "op": "for",
                    "init": step.get("init", ""),
                    "condition": step.get("condition", ""),
                    "increment": step.get("increment", ""),
                    "block_start": body_start + 1,  # Convert to 1-based indexing
                    "block_count": body_count
                }
            
            elif op == "switch":
                # v0.8.5: Add switch statement support
                # Reserve slot for switch statement
                switch_index = len(flat_steps)
                flat_steps.append(None)
                
                # Flatten each case body (RECURSIVE)
                cases = step.get("cases", [])
                flat_cases = []
                
                for case in cases:
                    case_body = case.get("body", [])
                    case_start = len(flat_steps)
                    
                    # Recursively flatten case body
                    flatten_recursive(case_body)
                    
                    case_count = len(flat_steps) - case_start
                    
                    # v0.8.5: Convert case value to string for SPARK compatibility
                    case_value = case.get("value", "")
                    if isinstance(case_value, (int, float)):
                        case_value = str(case_value)
                    
                    flat_cases.append({
                        "value": case_value,
                        "block_start": case_start + 1,  # Convert to 1-based indexing
                        "block_count": case_count
                    })
                
                # Flatten default block if present (RECURSIVE)
                default_block = step.get("default", [])
                default_start = 0
                default_count = 0
                
                if default_block:
                    default_start = len(flat_steps)
                    
                    # Recursively flatten default block
                    flatten_recursive(default_block)
                    
                    default_count = len(flat_steps) - default_start
                
                # Fill in the switch statement (1-based for Ada compatibility)
                flat_steps[switch_index] = {
                    "op": "switch",
                    "expr": step.get("expr", ""),
                    "cases": flat_cases,
                    "default_start": default_start + 1 if default_start > 0 else 0,  # Convert to 1-based
                    "default_count": default_count
                }
            
            else:
                # Regular step (assign, return, call, nop, break, continue, etc.)
                flat_steps.append(step)
    
    flatten_recursive(nested_steps)
    return flat_steps


def convert_ir_module(nested_ir: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an entire IR module from nested to flattened format.
    
    Args:
        nested_ir: IR module with nested control flow
    
    Returns:
        IR module with flattened control flow
    """
    flat_ir = nested_ir.copy()
    flat_ir["schema"] = "stunir_flat_ir_v1"  # Mark as flattened format
    
    # Convert each function's steps
    if "functions" in flat_ir:
        for func in flat_ir["functions"]:
            if "steps" in func:
                func["steps"] = convert_nested_to_flat(func["steps"])
    
    return flat_ir


def load_ir(ir_path: Path) -> Dict[str, Any]:
    """Load IR from JSON file."""
    with open(ir_path, 'r') as f:
        return json.load(f)


def save_ir(ir_data: Dict[str, Any], output_path: Path) -> None:
    """Save IR to JSON file with pretty formatting."""
    with open(output_path, 'w') as f:
        json.dump(ir_data, f, indent=2, sort_keys=False)


def main():
    """
    Command-line interface for IR conversion.
    
    Usage:
        python ir_converter.py <input_ir.json> <output_flat_ir.json>
    """
    if len(sys.argv) != 3:
        print("Usage: python ir_converter.py <input_ir.json> <output_flat_ir.json>")
        print("\nConverts nested STUNIR IR to flattened IR for SPARK compatibility.")
        print("\nExample:")
        print("  python ir_converter.py ir.json ir_flat.json")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[INFO] Loading nested IR from: {input_path}")
    nested_ir = load_ir(input_path)
    
    print(f"[INFO] Converting to flattened IR format...")
    flat_ir = convert_ir_module(nested_ir)
    
    print(f"[INFO] Saving flattened IR to: {output_path}")
    save_ir(flat_ir, output_path)
    
    # Print summary
    if "functions" in flat_ir:
        total_steps = sum(len(f.get("steps", [])) for f in flat_ir["functions"])
        print(f"[INFO] Converted {len(flat_ir['functions'])} functions, "
              f"{total_steps} total steps")
    
    print(f"[INFO] Conversion complete!")
    print(f"[INFO] Output schema: {flat_ir.get('schema', 'unknown')}")


if __name__ == "__main__":
    main()
