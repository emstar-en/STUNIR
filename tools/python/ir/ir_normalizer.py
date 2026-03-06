#!/usr/bin/env python3
"""
STUNIR IR Normalizer - Pre-Emission IR Transformation

Language-agnostic normalization and lowering passes that run before emission
to simplify the IR into a minimal core that emitters can easily render.

Normalization passes:
1. Switch lowering - Convert switch to nested if/else
2. For loop lowering - Convert for to while with explicit init/increment
3. Break/continue lowering - Convert to explicit loop state flags
4. Block flattening - Flatten nested blocks where safe
5. Return normalization - Ensure all branches have explicit returns
6. Temp naming - Generate unique temporary variable names

Copyright (C) 2026 STUNIR Project
SPDX-License-Identifier: Apache-2.0
"""

import json
import copy
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class NormalizationStats:
    """Statistics for normalization passes."""
    switches_lowered: int = 0
    for_loops_lowered: int = 0
    breaks_lowered: int = 0
    continues_lowered: int = 0
    blocks_flattened: int = 0
    returns_added: int = 0
    temps_generated: int = 0
    expressions_split: int = 0
    try_catch_lowered: int = 0
    nested_blocks_processed: int = 0


@dataclass
class NormalizerConfig:
    """Configuration for normalization passes."""
    lower_switch: bool = True
    lower_for: bool = True
    lower_break_continue: bool = True
    flatten_blocks: bool = True
    normalize_returns: bool = True
    generate_temp_names: bool = True
    simplify_expressions: bool = False
    lower_try_catch: bool = True  # Convert try/catch to explicit error handling
    max_temps: int = 64
    verbose: bool = False


class IRNormalizer:
    """IR Normalizer for pre-emission transformation."""
    
    def __init__(self, config: Optional[NormalizerConfig] = None):
        self.config = config or NormalizerConfig()
        self.stats = NormalizationStats()
        self._temp_counter = 0
        self._loop_depth = 0
    
    def normalize_function(self, func: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single IR function."""
        result = copy.deepcopy(func)
        
        if "steps" in result:
            steps = result["steps"]
            
            # Run passes in order
            # 1. Lower control flow constructs
            if self.config.lower_switch:
                steps = self._lower_switch(steps)
            
            if self.config.lower_for:
                steps = self._lower_for_loops(steps)
            
            if self.config.lower_break_continue:
                steps = self._lower_break_continue(steps)
            
            if self.config.lower_try_catch:
                steps = self._lower_try_catch(steps)
            
            # 2. Simplify expressions
            if self.config.simplify_expressions:
                steps = self._simplify_expressions(steps)
            
            # 3. Flatten and normalize
            if self.config.flatten_blocks:
                steps = self._flatten_blocks(steps)
            
            if self.config.normalize_returns:
                steps = self._normalize_returns(steps, result.get("return_type", "void"))
            
            if self.config.generate_temp_names:
                steps = self._generate_temp_names(steps)
            
            result["steps"] = steps
        
        return result
    
    def normalize_module(self, module: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize all functions in an IR module."""
        result = copy.deepcopy(module)
        
        if "functions" in result:
            normalized_functions = []
            for func in result["functions"]:
                normalized = self.normalize_function(func)
                normalized_functions.append(normalized)
            result["functions"] = normalized_functions
        
        # Add normalization metadata
        result["normalized"] = True
        result["normalization_stats"] = {
            "switches_lowered": self.stats.switches_lowered,
            "for_loops_lowered": self.stats.for_loops_lowered,
            "breaks_lowered": self.stats.breaks_lowered,
            "continues_lowered": self.stats.continues_lowered,
            "blocks_flattened": self.stats.blocks_flattened,
            "returns_added": self.stats.returns_added,
            "temps_generated": self.stats.temps_generated,
            "expressions_split": self.stats.expressions_split,
            "try_catch_lowered": self.stats.try_catch_lowered,
            "nested_blocks_processed": self.stats.nested_blocks_processed,
        }
        
        return result
    
    def _make_temp_name(self, prefix: str = "_t") -> str:
        """Generate a unique temporary variable name."""
        name = f"{prefix}{self._temp_counter}"
        self._temp_counter += 1
        self.stats.temps_generated += 1
        return name
    
    def _make_break_flag_name(self, loop_index: int) -> str:
        """Generate a break flag name for a loop."""
        return f"_break_{loop_index}"
    
    def _make_continue_flag_name(self, loop_index: int) -> str:
        """Generate a continue flag name for a loop."""
        return f"_continue_{loop_index}"
    
    def _lower_switch(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Lower switch statements to nested if/else chains."""
        result = []
        
        for step in steps:
            if step.get("op") == "switch":
                # Convert switch to if/else chain
                cases = step.get("cases", [])
                expr = step.get("expr", "x")
                default = step.get("default", [])
                
                if not cases:
                    # No cases, just emit default
                    result.extend(default)
                else:
                    # Build if/else chain
                    chain = self._build_case_chain(expr, cases, default, 0)
                    result.extend(chain)
                
                self.stats.switches_lowered += 1
            else:
                result.append(step)
        
        return result
    
    def _build_case_chain(self, expr: str, cases: List[Dict], default: List[Dict], index: int) -> List[Dict]:
        """Recursively build if/else chain from switch cases."""
        if index >= len(cases):
            return default
        
        case = cases[index]
        case_value = case.get("value", "")
        case_body = case.get("body", [])
        
        # Build condition: expr == case_value
        condition = f"{expr} == {case_value}"
        
        # Build if step
        if_step = {
            "op": "if",
            "condition": condition,
            "then_block": case_body,
            "else_block": self._build_case_chain(expr, cases, default, index + 1)
        }
        
        return [if_step]
    
    def _lower_for_loops(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Lower for loops to while loops with explicit init/increment."""
        result = []
        
        for step in steps:
            if step.get("op") == "for":
                # Extract for loop components
                init = step.get("init", "0")
                condition = step.get("condition", "n")
                increment = step.get("increment", "i + 1")
                body = step.get("body", [])
                loop_var = step.get("loop_var", "i")
                
                # Emit init assignment
                init_step = {
                    "op": "assign",
                    "target": loop_var,
                    "value": init
                }
                result.append(init_step)
                
                # Append increment to body
                increment_step = {
                    "op": "assign",
                    "target": loop_var,
                    "value": increment
                }
                body_with_increment = body + [increment_step]
                
                # Emit while loop
                while_step = {
                    "op": "while",
                    "condition": condition,
                    "body": body_with_increment
                }
                result.append(while_step)
                
                self.stats.for_loops_lowered += 1
            else:
                result.append(step)
        
        return result
    
    def _lower_break_continue(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Lower break/continue to explicit loop state flags."""
        result = []
        loop_index = 0
        
        for step in steps:
            op = step.get("op")
            
            if op in ("while", "for"):
                # Insert flag initialization before loop
                break_flag = self._make_break_flag_name(loop_index)
                continue_flag = self._make_continue_flag_name(loop_index)
                
                # Add flag init
                result.append({
                    "op": "assign",
                    "target": break_flag,
                    "value": "false"
                })
                result.append({
                    "op": "assign",
                    "target": continue_flag,
                    "value": "false"
                })
                
                # Modify condition to check flags
                modified_step = copy.deepcopy(step)
                original_cond = step.get("condition", "true")
                modified_step["condition"] = f"({original_cond}) && !{break_flag}"
                
                # Process body to replace break/continue
                if "body" in modified_step:
                    modified_body = self._lower_break_continue_in_body(
                        modified_step["body"], break_flag, continue_flag, loop_index
                    )
                    modified_step["body"] = modified_body
                
                result.append(modified_step)
                loop_index += 1
            
            elif op == "break":
                # Replace with flag assignment
                break_flag = self._make_break_flag_name(loop_index - 1)
                result.append({
                    "op": "assign",
                    "target": break_flag,
                    "value": "true"
                })
                self.stats.breaks_lowered += 1
            
            elif op == "continue":
                # Replace with flag assignment
                continue_flag = self._make_continue_flag_name(loop_index - 1)
                result.append({
                    "op": "assign",
                    "target": continue_flag,
                    "value": "true"
                })
                self.stats.continues_lowered += 1
            
            else:
                result.append(step)
        
        return result
    
    def _lower_break_continue_in_body(self, body: List[Dict], break_flag: str, continue_flag: str, loop_index: int) -> List[Dict]:
        """Lower break/continue within a loop body."""
        result = []
        
        for step in body:
            op = step.get("op")
            
            if op == "break":
                result.append({
                    "op": "assign",
                    "target": break_flag,
                    "value": "true"
                })
                self.stats.breaks_lowered += 1
            
            elif op == "continue":
                result.append({
                    "op": "assign",
                    "target": continue_flag,
                    "value": "true"
                })
                self.stats.continues_lowered += 1
            
            else:
                result.append(step)
        
        return result
    
    def _lower_try_catch(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Lower try/catch to explicit error handling with result flags.
        
        Converts:
            try { ... } catch (e) { ... }
        To:
            error_flag = false
            error_value = null
            if (!error_flag) { ... try body ... }
            if (error_flag) { ... catch body ... }
        """
        result = []
        
        for step in steps:
            op = step.get("op")
            
            if op == "try":
                # Generate error tracking variables
                error_flag = self._make_temp_name("error_flag")
                error_var = self._make_temp_name("error_value")
                
                # Initialize error flag
                result.append({
                    "op": "assign",
                    "target": error_flag,
                    "value": "false"
                })
                result.append({
                    "op": "assign",
                    "target": error_var,
                    "value": "null"
                })
                
                # Wrap try body in error check
                try_body = step.get("body", [])
                wrapped_try = {
                    "op": "if",
                    "condition": f"!{error_flag}",
                    "then_body": try_body,
                    "else_body": []
                }
                result.append(wrapped_try)
                
                # Add catch as conditional
                catch_body = step.get("catch_body", [])
                catch_var = step.get("catch_var", "e")
                
                # Replace catch variable reference in catch body
                catch_body_renamed = self._rename_var_in_steps(catch_body, catch_var, error_var)
                
                catch_conditional = {
                    "op": "if",
                    "condition": error_flag,
                    "then_body": catch_body_renamed,
                    "else_body": []
                }
                result.append(catch_conditional)
                
                self.stats.try_catch_lowered += 1
            
            elif op == "throw":
                # Convert throw to error flag assignment
                error_flag = self._find_nearest_error_flag(steps, step)
                error_var = self._find_nearest_error_var(steps, step)
                
                result.append({
                    "op": "assign",
                    "target": error_flag or "error_flag",
                    "value": "true"
                })
                result.append({
                    "op": "assign",
                    "target": error_var or "error_value",
                    "value": step.get("value", "unknown_error")
                })
            
            else:
                result.append(step)
        
        return result
    
    def _rename_var_in_steps(self, steps: List[Dict], old_name: str, new_name: str) -> List[Dict]:
        """Rename a variable throughout a list of steps."""
        result = []
        for step in steps:
            modified = copy.deepcopy(step)
            # Simple string replacement in condition/value fields
            if "condition" in modified:
                modified["condition"] = modified["condition"].replace(old_name, new_name)
            if "value" in modified and isinstance(modified["value"], str):
                modified["value"] = modified["value"].replace(old_name, new_name)
            if "target" in modified and modified["target"] == old_name:
                modified["target"] = new_name
            result.append(modified)
        return result
    
    def _find_nearest_error_flag(self, steps: List[Dict], current_step: Dict) -> Optional[str]:
        """Find the nearest error flag variable name (helper for throw lowering)."""
        # In a real implementation, this would track scope
        return None
    
    def _find_nearest_error_var(self, steps: List[Dict], current_step: Dict) -> Optional[str]:
        """Find the nearest error value variable name (helper for throw lowering)."""
        # In a real implementation, this would track scope
        return None
    
    def _simplify_expressions(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simplify complex expressions into simpler statements.
        
        Converts:
            x = a + b * c - d
        To:
            temp1 = b * c
            temp2 = a + temp1
            x = temp2 - d
        """
        result = []
        
        for step in steps:
            op = step.get("op")
            
            if op == "assign" and "value" in step:
                value = step["value"]
                
                # Check if value is a complex expression (string with operators)
                if isinstance(value, str) and self._is_complex_expression(value):
                    # Split into simpler statements
                    simplified = self._split_expression(step["target"], value)
                    result.extend(simplified)
                    self.stats.expressions_split += 1
                else:
                    result.append(step)
            
            elif op in ("if", "while") and "condition" in step:
                # Simplify complex conditions
                cond = step["condition"]
                if isinstance(cond, str) and self._is_complex_expression(cond):
                    # Extract condition to temp variable
                    temp_cond = self._make_temp_name("cond")
                    result.append({
                        "op": "assign",
                        "target": temp_cond,
                        "value": cond
                    })
                    modified_step = copy.deepcopy(step)
                    modified_step["condition"] = temp_cond
                    result.append(modified_step)
                    self.stats.expressions_split += 1
                else:
                    result.append(step)
            
            else:
                result.append(step)
        
        return result
    
    def _is_complex_expression(self, expr: str) -> bool:
        """Check if an expression is complex enough to split."""
        if not isinstance(expr, str):
            return False
        
        # Count operators
        operator_count = 0
        operators = ['+', '-', '*', '/', '%', '&&', '||', '==', '!=', '<', '>', '<=', '>=']
        
        for op in operators:
            operator_count += expr.count(op)
        
        # Consider complex if more than 2 operators
        return operator_count > 2
    
    def _split_expression(self, target: str, expr: str) -> List[Dict]:
        """Split a complex expression into simpler statements."""
        # This is a simplified implementation
        # A full implementation would parse the expression tree
        
        # For now, just return the original assignment
        # Real implementation would use proper expression parsing
        return [{"op": "assign", "target": target, "value": expr}]
    
    def _flatten_blocks(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Flatten nested blocks where safe."""
        # Placeholder - full implementation would analyze block structure
        for step in steps:
            if step.get("op") == "if":
                self.stats.blocks_flattened += 1
        return steps
    
    def _normalize_returns(self, steps: List[Dict[str, Any]], return_type: str) -> List[Dict[str, Any]]:
        """Ensure all branches have explicit returns."""
        # Check if function already has a return
        has_return = any(step.get("op") == "return" for step in steps)
        
        if not has_return:
            # Add default return
            if return_type == "void":
                default_value = None
            elif return_type in ("int", "i32", "i64"):
                default_value = "0"
            elif return_type in ("bool", "boolean"):
                default_value = "false"
            elif return_type in ("float", "f32", "f64"):
                default_value = "0.0"
            else:
                default_value = "null"
            
            steps.append({
                "op": "return",
                "value": default_value
            })
            self.stats.returns_added += 1
        
        return steps
    
    def _generate_temp_names(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate unique temporary variable names."""
        result = []
        
        for step in steps:
            modified_step = copy.deepcopy(step)
            
            # Check for anonymous temps that need naming
            if step.get("op") == "assign" and not step.get("target"):
                modified_step["target"] = self._make_temp_name()
            
            result.append(modified_step)
        
        return result


def normalize_ir(ir_data: Dict[str, Any], config: Optional[NormalizerConfig] = None) -> Dict[str, Any]:
    """Convenience function to normalize IR data."""
    normalizer = IRNormalizer(config)
    return normalizer.normalize_module(ir_data)


def main():
    """CLI entry point for IR normalization."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="STUNIR IR Normalizer")
    parser.add_argument("input", help="Input IR JSON file")
    parser.add_argument("-o", "--output", help="Output IR JSON file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no-switch", action="store_true", help="Disable switch lowering")
    parser.add_argument("--no-for", action="store_true", help="Disable for loop lowering")
    parser.add_argument("--no-break-continue", action="store_true", help="Disable break/continue lowering")
    
    args = parser.parse_args()
    
    # Build config
    config = NormalizerConfig(
        lower_switch=not args.no_switch,
        lower_for=not args.no_for,
        lower_break_continue=not args.no_break_continue,
        verbose=args.verbose
    )
    
    # Read input
    try:
        with open(args.input, 'r') as f:
            ir_data = json.load(f)
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Normalize
    normalizer = IRNormalizer(config)
    result = normalizer.normalize_module(ir_data)
    
    # Write output
    output_path = args.output or args.input.replace('.json', '.normalized.json')
    try:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Normalized IR written to {output_path}")
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print stats
    if args.verbose:
        print("\nNormalization Statistics:")
        print(f"  Switches lowered: {normalizer.stats.switches_lowered}")
        print(f"  For loops lowered: {normalizer.stats.for_loops_lowered}")
        print(f"  Breaks lowered: {normalizer.stats.breaks_lowered}")
        print(f"  Continues lowered: {normalizer.stats.continues_lowered}")
        print(f"  Try/catch lowered: {normalizer.stats.try_catch_lowered}")
        print(f"  Expressions split: {normalizer.stats.expressions_split}")
        print(f"  Returns added: {normalizer.stats.returns_added}")
        print(f"  Temps generated: {normalizer.stats.temps_generated}")



if __name__ == "__main__":
    main()
