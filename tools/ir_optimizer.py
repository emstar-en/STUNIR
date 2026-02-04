#!/usr/bin/env python3
"""
STUNIR IR Optimizer - Python Implementation

Aligns with Ada SPARK optimizer in tools/spark/src/optimizer/

This module provides IR optimization passes that match the SPARK
implementation's functionality and output.

Usage:
    python tools/ir_optimizer.py --ir input.ir.json --out output.ir.json --level 2
"""

import argparse
import json
import re
from typing import Any, Dict, List, Tuple


def is_numeric(value: str) -> bool:
    """Check if a string represents a numeric constant."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_boolean(value: str) -> bool:
    """Check if a string represents a boolean constant."""
    return value.lower() in ("true", "false")


def is_constant(value: str) -> bool:
    """Check if a value is a compile-time constant."""
    return is_numeric(value) or is_boolean(value)


def fold_constant(expr: str) -> Tuple[str, bool]:
    """
    Fold constant expressions.
    
    Returns:
        Tuple of (folded_expression, was_folded)
    """
    # Simple constant folding for arithmetic
    # Pattern: number op number
    patterns = [
        (r'^(\d+)\s*\+\s*(\d+)$', lambda a, b: str(int(a) + int(b))),
        (r'^(\d+)\s*-\s*(\d+)$', lambda a, b: str(int(a) - int(b))),
        (r'^(\d+)\s*\*\s*(\d+)$', lambda a, b: str(int(a) * int(b))),
        (r'^(\d+)\s*/\s*(\d+)$', lambda a, b: str(int(a) // int(b)) if int(b) != 0 else expr),
    ]
    
    for pattern, op in patterns:
        match = re.match(pattern, expr.strip())
        if match:
            try:
                result = op(match.group(1), match.group(2))
                return result, True
            except (ValueError, ZeroDivisionError):
                continue
    
    # Boolean folding
    if expr.strip().lower() == "true and true":
        return "true", True
    if expr.strip().lower() == "false or false":
        return "false", True
    if expr.strip().lower() in ("true or false", "false or true"):
        return "true", True
    if expr.strip().lower() in ("true and false", "false and true"):
        return "false", True
    
    return expr, False


def constant_folding(ir: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    Fold constant expressions in IR.
    
    Returns:
        Tuple of (optimized_ir, fold_count)
    """
    ir = json.loads(json.dumps(ir))  # Deep copy
    fold_count = 0
    
    functions = ir.get("ir_functions", ir.get("functions", []))
    
    for func in functions:
        statements = func.get("statements", func.get("body", []))
        
        for stmt in statements:
            if not isinstance(stmt, dict):
                continue
                
            # Fold in assignments
            if stmt.get("kind") == "assign":
                value = stmt.get("value", "")
                if isinstance(value, str) and is_constant(value):
                    folded, was_folded = fold_constant(value)
                    if was_folded:
                        stmt["value"] = folded
                        fold_count += 1
            
            # Fold in return statements
            if stmt.get("kind") == "return":
                value = stmt.get("value", "")
                if isinstance(value, str) and is_constant(value):
                    folded, was_folded = fold_constant(value)
                    if was_folded:
                        stmt["value"] = folded
                        fold_count += 1
    
    return ir, fold_count


def constant_propagation(ir: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    Propagate constants through the IR.
    
    Replaces variable references with their constant values.
    
    Returns:
        Tuple of (optimized_ir, propagation_count)
    """
    ir = json.loads(json.dumps(ir))  # Deep copy
    propagation_count = 0
    
    functions = ir.get("ir_functions", ir.get("functions", []))
    
    for func in functions:
        statements = func.get("statements", func.get("body", []))
        constants: Dict[str, str] = {}
        
        for stmt in statements:
            if not isinstance(stmt, dict):
                continue
            
            # Track constant assignments
            if stmt.get("kind") == "assign":
                target = stmt.get("target", "")
                value = stmt.get("value", "")
                if isinstance(value, str) and is_constant(value):
                    constants[target] = value
            
            # Propagate in return statements
            if stmt.get("kind") == "return":
                value = stmt.get("value", "")
                if isinstance(value, str) and value in constants:
                    stmt["value"] = constants[value]
                    propagation_count += 1
    
    return ir, propagation_count


def dead_code_elimination(ir: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    Remove dead code (assignments to unused variables).
    
    Returns:
        Tuple of (optimized_ir, elimination_count)
    """
    ir = json.loads(json.dumps(ir))  # Deep copy
    elimination_count = 0
    
    functions = ir.get("ir_functions", ir.get("functions", []))
    
    for func in functions:
        statements = func.get("statements", func.get("body", []))
        
        # Find all used variables
        used_vars: set = set()
        for stmt in statements:
            if not isinstance(stmt, dict):
                continue
            
            # Check return values
            if stmt.get("kind") == "return":
                value = stmt.get("value", "")
                if isinstance(value, str):
                    used_vars.add(value)
            
            # Check conditions
            if stmt.get("kind") in ("if", "while"):
                condition = stmt.get("condition", "")
                if isinstance(condition, str):
                    # Extract variable names from condition
                    for word in condition.split():
                        if word.isidentifier():
                            used_vars.add(word)
        
        # Remove unused assignments
        new_statements = []
        for stmt in statements:
            if not isinstance(stmt, dict):
                new_statements.append(stmt)
                continue
            
            if stmt.get("kind") == "assign":
                target = stmt.get("target", "")
                if target not in used_vars:
                    elimination_count += 1
                    continue  # Skip this statement
            
            new_statements.append(stmt)
        
        if "statements" in func:
            func["statements"] = new_statements
        elif "body" in func:
            func["body"] = new_statements
    
    return ir, elimination_count


def unreachable_code_elimination(ir: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
    """
    Remove unreachable code (after return statements).
    
    Returns:
        Tuple of (optimized_ir, elimination_count)
    """
    ir = json.loads(json.dumps(ir))  # Deep copy
    elimination_count = 0
    
    functions = ir.get("ir_functions", ir.get("functions", []))
    
    for func in functions:
        statements = func.get("statements", func.get("body", []))
        
        new_statements = []
        has_returned = False
        
        for stmt in statements:
            if not isinstance(stmt, dict):
                if not has_returned:
                    new_statements.append(stmt)
                continue
            
            if has_returned:
                elimination_count += 1
                continue
            
            new_statements.append(stmt)
            
            if stmt.get("kind") == "return":
                has_returned = True
        
        if "statements" in func:
            func["statements"] = new_statements
        elif "body" in func:
            func["body"] = new_statements
    
    return ir, elimination_count


def optimize_ir(ir: Dict[str, Any], level: int = 2) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    Apply optimization passes based on level.
    
    Args:
        ir: Input IR dictionary
        level: Optimization level (0-3)
            0: No optimization
            1: Basic (constant folding)
            2: Standard (folding + propagation + DCE)
            3: Aggressive (all passes)
    
    Returns:
        Tuple of (optimized_ir, stats)
    """
    if level == 0:
        return ir, {"total": 0}
    
    stats = {}
    
    # Level 1+: Constant folding
    ir, fold_count = constant_folding(ir)
    stats["constant_folding"] = fold_count
    
    if level >= 2:
        # Level 2+: Constant propagation
        ir, prop_count = constant_propagation(ir)
        stats["constant_propagation"] = prop_count
        
        # Level 2+: Dead code elimination
        ir, dce_count = dead_code_elimination(ir)
        stats["dead_code_elimination"] = dce_count
    
    if level >= 3:
        # Level 3+: Unreachable code elimination
        ir, uce_count = unreachable_code_elimination(ir)
        stats["unreachable_code_elimination"] = uce_count
    
    stats["total"] = sum(stats.values())
    
    return ir, stats


def main() -> None:
    """CLI entry point for IR optimization."""
    parser = argparse.ArgumentParser(
        description="STUNIR IR Optimizer (Python Implementation)"
    )
    parser.add_argument("--ir", required=True, help="Path to input IR JSON file")
    parser.add_argument("--out", required=True, help="Path to output IR JSON file")
    parser.add_argument("--level", type=int, default=2,
                       help="Optimization level (0-3, default: 2)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Print optimization statistics")
    
    args = parser.parse_args()
    
    # Load IR
    with open(args.ir, "r", encoding="utf-8") as f:
        ir = json.load(f)
    
    # Optimize
    optimized_ir, stats = optimize_ir(ir, level=args.level)
    
    # Write output
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(optimized_ir, f, indent=2, sort_keys=True)
        f.write("\n")
    
    # Print stats
    if args.verbose:
        print(f"Optimization complete: {args.out}")
        print(f"  Total optimizations: {stats['total']}")
        for pass_name, count in stats.items():
            if pass_name != "total":
                print(f"  - {pass_name}: {count}")
    else:
        print(f"Optimized: {args.out} ({stats['total']} optimizations)")


if __name__ == "__main__":
    main()