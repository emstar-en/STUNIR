#!/usr/bin/env python3
"""
===============================================================================
STUNIR IR Optimizer - Python REFERENCE Implementation
===============================================================================

Python implementation of the STUNIR IR optimization framework (v0.8.9+).

Supported optimization passes:
- Dead code elimination
- Constant folding
- Constant propagation
- Unreachable code removal

===============================================================================
"""

import argparse
import copy
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [optimizer] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

VERSION = "0.8.9"


class OptimizationPass:
    """Base class for optimization passes."""
    
    name: str = "base"
    
    def optimize(self, ir: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Run the optimization pass.
        
        Returns:
            Tuple of (optimized_ir, changes_count)
        """
        raise NotImplementedError


class DeadCodeElimination(OptimizationPass):
    """Remove unreferenced variables and unreachable code."""
    
    name = "dead_code_elimination"
    
    def optimize(self, ir: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        ir = copy.deepcopy(ir)
        changes = 0
        
        for func in ir.get("functions", []):
            steps = func.get("steps", [])
            new_steps, func_changes = self._eliminate_dead_code(steps)
            func["steps"] = new_steps
            changes += func_changes
        
        return ir, changes
    
    def _eliminate_dead_code(self, steps: List[Dict]) -> Tuple[List[Dict], int]:
        """Remove dead code from a list of steps."""
        if not steps:
            return steps, 0
        
        changes = 0
        new_steps = []
        seen_return = False
        
        for step in steps:
            # Skip steps after unconditional return
            if seen_return:
                changes += 1
                continue
            
            # Skip nop operations
            if step.get("op") == "nop":
                changes += 1
                continue
            
            # Skip steps marked as dead code
            if step.get("optimization", {}).get("dead_code"):
                changes += 1
                continue
            
            # Process nested blocks
            new_step = copy.deepcopy(step)
            
            if "then_block" in new_step:
                new_step["then_block"], c = self._eliminate_dead_code(new_step["then_block"])
                changes += c
            
            if "else_block" in new_step:
                new_step["else_block"], c = self._eliminate_dead_code(new_step["else_block"])
                changes += c
            
            if "body" in new_step:
                new_step["body"], c = self._eliminate_dead_code(new_step["body"])
                changes += c
            
            if "try_block" in new_step:
                new_step["try_block"], c = self._eliminate_dead_code(new_step["try_block"])
                changes += c
            
            if "catch_blocks" in new_step:
                for catch in new_step["catch_blocks"]:
                    if "body" in catch:
                        catch["body"], c = self._eliminate_dead_code(catch["body"])
                        changes += c
            
            if "finally_block" in new_step:
                new_step["finally_block"], c = self._eliminate_dead_code(new_step["finally_block"])
                changes += c
            
            if "cases" in new_step:
                for case in new_step["cases"]:
                    if "body" in case:
                        case["body"], c = self._eliminate_dead_code(case["body"])
                        changes += c
            
            if "default" in new_step:
                new_step["default"], c = self._eliminate_dead_code(new_step["default"])
                changes += c
            
            new_steps.append(new_step)
            
            # Mark that we've seen an unconditional return
            if step.get("op") == "return":
                seen_return = True
        
        return new_steps, changes


class ConstantFolding(OptimizationPass):
    """Evaluate constant expressions at compile time."""
    
    name = "constant_folding"
    
    def optimize(self, ir: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        ir = copy.deepcopy(ir)
        changes = 0
        
        for func in ir.get("functions", []):
            steps = func.get("steps", [])
            new_steps, func_changes = self._fold_constants(steps)
            func["steps"] = new_steps
            changes += func_changes
        
        return ir, changes
    
    def _fold_constants(self, steps: List[Dict]) -> Tuple[List[Dict], int]:
        """Fold constants in a list of steps."""
        if not steps:
            return steps, 0
        
        changes = 0
        new_steps = []
        
        for step in steps:
            new_step = copy.deepcopy(step)
            
            # Fold constant assignments
            if step.get("op") == "assign" and "value" in step:
                value = step["value"]
                if isinstance(value, str):
                    folded, did_fold = self._try_fold_expression(value)
                    if did_fold:
                        new_step["value"] = folded
                        new_step.setdefault("optimization", {})["const_eval"] = True
                        new_step["optimization"]["constant_value"] = folded
                        changes += 1
            
            # Fold constant conditions
            if "condition" in step:
                cond = step["condition"]
                folded, did_fold = self._try_fold_expression(cond)
                if did_fold:
                    new_step["condition"] = str(folded).lower() if isinstance(folded, bool) else str(folded)
                    changes += 1
            
            # Process nested blocks recursively
            if "then_block" in new_step:
                new_step["then_block"], c = self._fold_constants(new_step["then_block"])
                changes += c
            
            if "else_block" in new_step:
                new_step["else_block"], c = self._fold_constants(new_step["else_block"])
                changes += c
            
            if "body" in new_step:
                new_step["body"], c = self._fold_constants(new_step["body"])
                changes += c
            
            new_steps.append(new_step)
        
        return new_steps, changes
    
    def _try_fold_expression(self, expr: str) -> Tuple[Any, bool]:
        """Try to evaluate a constant expression.
        
        Returns:
            Tuple of (result, did_fold)
        """
        expr = expr.strip()
        
        # Try simple arithmetic: "1 + 2", "3 * 4", etc.
        arith_match = re.match(r'^(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)$', expr)
        if arith_match:
            left = float(arith_match.group(1))
            op = arith_match.group(2)
            right = float(arith_match.group(3))
            
            try:
                if op == '+':
                    result = left + right
                elif op == '-':
                    result = left - right
                elif op == '*':
                    result = left * right
                elif op == '/' and right != 0:
                    result = left / right
                else:
                    return expr, False
                
                # Return int if possible
                if result == int(result):
                    return int(result), True
                return result, True
            except:
                return expr, False
        
        # Try boolean expressions: "true && false", "1 > 0", etc.
        bool_match = re.match(r'^(true|false)\s*(&&|\|\|)\s*(true|false)$', expr.lower())
        if bool_match:
            left = bool_match.group(1) == 'true'
            op = bool_match.group(2)
            right = bool_match.group(3) == 'true'
            
            if op == '&&':
                return left and right, True
            elif op == '||':
                return left or right, True
        
        # Try comparison: "1 > 0", "2 == 2", etc.
        cmp_match = re.match(r'^(-?\d+)\s*(==|!=|<|>|<=|>=)\s*(-?\d+)$', expr)
        if cmp_match:
            left = int(cmp_match.group(1))
            op = cmp_match.group(2)
            right = int(cmp_match.group(3))
            
            if op == '==':
                return left == right, True
            elif op == '!=':
                return left != right, True
            elif op == '<':
                return left < right, True
            elif op == '>':
                return left > right, True
            elif op == '<=':
                return left <= right, True
            elif op == '>=':
                return left >= right, True
        
        return expr, False


class ConstantPropagation(OptimizationPass):
    """Propagate known constant values through the code."""
    
    name = "constant_propagation"
    
    def optimize(self, ir: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        ir = copy.deepcopy(ir)
        changes = 0
        
        for func in ir.get("functions", []):
            steps = func.get("steps", [])
            constants: Dict[str, Any] = {}
            new_steps, func_changes = self._propagate_constants(steps, constants)
            func["steps"] = new_steps
            changes += func_changes
        
        return ir, changes
    
    def _propagate_constants(self, steps: List[Dict], constants: Dict[str, Any]) -> Tuple[List[Dict], int]:
        """Propagate constants through steps."""
        if not steps:
            return steps, 0
        
        changes = 0
        new_steps = []
        
        for step in steps:
            new_step = copy.deepcopy(step)
            
            # Track constant assignments
            if step.get("op") == "assign":
                target = step.get("target")
                value = step.get("value")
                if target and isinstance(value, (int, float, bool, str)):
                    # Check if it's a literal (not a variable reference)
                    if isinstance(value, (int, float, bool)) or (isinstance(value, str) and value.startswith('"')):
                        constants[target] = value
            
            # Propagate constants in expressions
            if "value" in new_step and isinstance(new_step["value"], str):
                for var, const_val in constants.items():
                    if var in new_step["value"]:
                        # Simple variable replacement
                        pattern = r'\b' + re.escape(var) + r'\b'
                        new_val = re.sub(pattern, str(const_val), new_step["value"])
                        if new_val != new_step["value"]:
                            new_step["value"] = new_val
                            changes += 1
            
            new_steps.append(new_step)
        
        return new_steps, changes


class UnreachableCodeElimination(OptimizationPass):
    """Remove code that can never be reached."""
    
    name = "unreachable_code_elimination"
    
    def optimize(self, ir: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        ir = copy.deepcopy(ir)
        changes = 0
        
        for func in ir.get("functions", []):
            steps = func.get("steps", [])
            new_steps, func_changes = self._eliminate_unreachable(steps)
            func["steps"] = new_steps
            changes += func_changes
        
        return ir, changes
    
    def _eliminate_unreachable(self, steps: List[Dict]) -> Tuple[List[Dict], int]:
        """Remove unreachable code."""
        if not steps:
            return steps, 0
        
        changes = 0
        new_steps = []
        
        for step in steps:
            new_step = copy.deepcopy(step)
            
            # Check for if with constant false condition
            if step.get("op") == "if":
                cond = step.get("condition", "")
                if cond.lower() == "false":
                    # Replace with else block if exists
                    if "else_block" in step and step["else_block"]:
                        new_steps.extend(copy.deepcopy(step["else_block"]))
                    changes += 1
                    continue
                elif cond.lower() == "true":
                    # Replace with then block
                    if "then_block" in step and step["then_block"]:
                        new_steps.extend(copy.deepcopy(step["then_block"]))
                    changes += 1
                    continue
            
            # Check for while with constant false condition
            if step.get("op") == "while":
                cond = step.get("condition", "")
                if cond.lower() == "false":
                    changes += 1
                    continue
            
            # Process nested blocks
            if "then_block" in new_step:
                new_step["then_block"], c = self._eliminate_unreachable(new_step["then_block"])
                changes += c
            
            if "else_block" in new_step:
                new_step["else_block"], c = self._eliminate_unreachable(new_step["else_block"])
                changes += c
            
            if "body" in new_step:
                new_step["body"], c = self._eliminate_unreachable(new_step["body"])
                changes += c
            
            new_steps.append(new_step)
        
        return new_steps, changes


class Optimizer:
    """Main optimizer class that runs multiple passes."""
    
    PASS_CLASSES = {
        "dead_code_elimination": DeadCodeElimination,
        "constant_folding": ConstantFolding,
        "constant_propagation": ConstantPropagation,
        "unreachable_code_elimination": UnreachableCodeElimination,
    }
    
    def __init__(self, passes: Optional[List[str]] = None, max_iterations: int = 10):
        """Initialize optimizer with specified passes.
        
        Args:
            passes: List of pass names to run. If None, runs all passes.
            max_iterations: Maximum optimization iterations.
        """
        if passes is None:
            passes = list(self.PASS_CLASSES.keys())
        
        self.passes = [self.PASS_CLASSES[p]() for p in passes if p in self.PASS_CLASSES]
        self.max_iterations = max_iterations
    
    def optimize(self, ir: Dict[str, Any]) -> Dict[str, Any]:
        """Run all optimization passes until fixed point."""
        current_ir = ir
        total_changes = 0
        
        for iteration in range(self.max_iterations):
            iteration_changes = 0
            
            for opt_pass in self.passes:
                current_ir, changes = opt_pass.optimize(current_ir)
                iteration_changes += changes
                if changes > 0:
                    logger.debug(f"Pass {opt_pass.name}: {changes} changes")
            
            total_changes += iteration_changes
            
            if iteration_changes == 0:
                logger.info(f"Optimization converged after {iteration + 1} iterations")
                break
        
        logger.info(f"Total optimizations applied: {total_changes}")
        return current_ir


def optimize_ir(ir: Dict[str, Any], level: int = 1, passes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Convenience function to optimize IR.
    
    Args:
        ir: Input IR dictionary
        level: Optimization level (0=none, 1=basic, 2=standard, 3=aggressive)
        passes: Specific passes to run (overrides level)
    
    Returns:
        Optimized IR dictionary
    """
    if level == 0:
        return ir
    
    if passes is None:
        if level >= 3:
            passes = list(Optimizer.PASS_CLASSES.keys())
        elif level >= 2:
            passes = ["dead_code_elimination", "constant_folding", "unreachable_code_elimination"]
        else:
            passes = ["dead_code_elimination", "constant_folding"]
    
    optimizer = Optimizer(passes=passes)
    return optimizer.optimize(ir)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="STUNIR IR Optimizer - Dead code elimination, constant folding, and more"
    )
    parser.add_argument("input", help="Input IR file (JSON)")
    parser.add_argument("-o", "--output", required=True, help="Output optimized IR file")
    parser.add_argument(
        "-O", "--level",
        type=int,
        choices=[0, 1, 2, 3],
        default=1,
        help="Optimization level (0=none, 1=basic, 2=standard, 3=aggressive)"
    )
    parser.add_argument(
        "--passes",
        nargs="+",
        choices=list(Optimizer.PASS_CLASSES.keys()),
        help="Specific optimization passes to run"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    
    args = parser.parse_args()
    
    # Read input IR
    with open(args.input) as f:
        ir = json.load(f)
    
    logger.info(f"Optimizing IR from {args.input} (level={args.level})")
    
    # Optimize
    optimized_ir = optimize_ir(ir, level=args.level, passes=args.passes)
    
    # Write output
    with open(args.output, 'w') as f:
        json.dump(optimized_ir, f, indent=2)
    
    logger.info(f"Optimized IR written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
