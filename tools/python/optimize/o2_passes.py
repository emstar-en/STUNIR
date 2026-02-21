#!/usr/bin/env python3
"""STUNIR O2 Optimization Passes.

Standard optimizations (includes all O1 optimizations):
- Common subexpression elimination (CSE)
- Loop invariant code motion
- Copy propagation
- Strength reduction (replace expensive ops with cheaper ones)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
import copy

from .optimization_pass import OptimizationPass, OptimizationLevel


class CommonSubexpressionEliminationPass(OptimizationPass):
    """Eliminates common subexpressions by introducing temp variables."""
    
    @property
    def name(self) -> str:
        return "common_subexpression_elimination"
    
    @property
    def min_level(self) -> OptimizationLevel:
        return OptimizationLevel.O2
    
    def transform(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        for func in ir_data.get('ir_functions', []):
            self._eliminate_cse(func)
        return ir_data
    
    def _eliminate_cse(self, func: Dict) -> None:
        body = func.get('body', [])
        expr_map: Dict[str, str] = {}  # expr_hash -> temp_var
        temp_counter = [0]  # mutable counter
        new_body = []
        insertions = []
        
        for stmt in body:
            new_stmt, new_temps = self._process_stmt(stmt, expr_map, temp_counter)
            insertions.extend(new_temps)
            new_body.append(new_stmt)
        
        # Insert temp declarations at the beginning
        for temp_name, temp_expr in insertions:
            new_body.insert(0, {
                'type': 'var_decl',
                'var_name': temp_name,
                'init': temp_expr,
                '_cse_temp': True
            })
        
        func['body'] = new_body
    
    def _process_stmt(self, stmt: Any, expr_map: Dict, 
                     counter: List[int]) -> Tuple[Any, List[Tuple[str, Any]]]:
        insertions = []
        
        if not isinstance(stmt, dict):
            return stmt, insertions
        
        stmt = copy.copy(stmt)
        
        for key in ('init', 'value'):
            if key in stmt:
                new_expr, new_temps = self._process_expr(stmt[key], expr_map, counter)
                stmt[key] = new_expr
                insertions.extend(new_temps)
        
        return stmt, insertions
    
    def _process_expr(self, expr: Any, expr_map: Dict,
                     counter: List[int]) -> Tuple[Any, List[Tuple[str, Any]]]:
        insertions = []
        
        if not isinstance(expr, dict):
            return expr, insertions
        
        if expr.get('type') == 'binary':
            # Process children first
            left, left_ins = self._process_expr(expr.get('left'), expr_map, counter)
            right, right_ins = self._process_expr(expr.get('right'), expr_map, counter)
            insertions.extend(left_ins)
            insertions.extend(right_ins)
            
            new_expr = {'type': 'binary', 'op': expr.get('op'),
                       'left': left, 'right': right}
            
            # Check if this expression was seen before
            expr_hash = self._hash_expr(new_expr)
            if expr_hash in expr_map:
                self.stats.subexpressions_eliminated += 1
                self.stats.ir_changes += 1
                return {'type': 'var', 'name': expr_map[expr_hash]}, insertions
            
            # Check if expression is complex enough to cache
            if self._is_complex(new_expr):
                temp_name = f'_cse_{counter[0]}'
                counter[0] += 1
                expr_map[expr_hash] = temp_name
                insertions.append((temp_name, new_expr))
                return {'type': 'var', 'name': temp_name}, insertions
            
            return new_expr, insertions
        
        return expr, insertions
    
    def _hash_expr(self, expr: Any) -> str:
        if isinstance(expr, dict):
            if expr.get('type') == 'binary':
                return f"bin:{expr.get('op')}:{self._hash_expr(expr.get('left'))}:{self._hash_expr(expr.get('right'))}"
            elif expr.get('type') == 'var':
                return f"var:{expr.get('name')}"
            elif expr.get('type') == 'literal':
                return f"lit:{expr.get('value')}"
        return str(expr)
    
    def _is_complex(self, expr: Dict) -> bool:
        """Check if expression is complex enough to benefit from CSE."""
        # Only cache binary expressions with non-trivial operands
        if expr.get('type') != 'binary':
            return False
        left = expr.get('left')
        right = expr.get('right')
        left_complex = isinstance(left, dict) and left.get('type') == 'binary'
        right_complex = isinstance(right, dict) and right.get('type') == 'binary'
        return left_complex or right_complex


class LoopInvariantCodeMotionPass(OptimizationPass):
    """Moves loop-invariant code outside of loops."""
    
    @property
    def name(self) -> str:
        return "loop_invariant_code_motion"
    
    @property
    def min_level(self) -> OptimizationLevel:
        return OptimizationLevel.O2
    
    def transform(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        for func in ir_data.get('ir_functions', []):
            func['body'] = self._process_body(func.get('body', []))
        return ir_data
    
    def _process_body(self, body: List) -> List:
        new_body = []
        for stmt in body:
            result = self._process_stmt(stmt)
            if isinstance(result, list):
                new_body.extend(result)
            else:
                new_body.append(result)
        return new_body
    
    def _process_stmt(self, stmt: Any) -> Any:
        if not isinstance(stmt, dict):
            return stmt
        
        stmt_type = stmt.get('type', '')
        
        if stmt_type in ('while', 'for'):
            return self._hoist_invariants(stmt)
        elif stmt_type == 'if':
            stmt['then'] = self._process_body(stmt.get('then', []))
            if 'else' in stmt:
                stmt['else'] = self._process_body(stmt.get('else', []))
        
        return stmt
    
    def _hoist_invariants(self, loop: Dict) -> Any:
        body = loop.get('body', [])
        modified_vars = self._get_modified_vars(body)
        
        invariant = []
        new_body = []
        
        for stmt in body:
            if self._is_invariant(stmt, modified_vars):
                invariant.append(stmt)
                self.stats.loops_optimized += 1
                self.stats.ir_changes += 1
            else:
                new_body.append(self._process_stmt(stmt))
        
        loop = copy.copy(loop)
        loop['body'] = new_body
        
        if invariant:
            # Return hoisted statements followed by loop
            return invariant + [loop]
        
        return loop
    
    def _get_modified_vars(self, body: List) -> Set[str]:
        modified = set()
        for stmt in body:
            if isinstance(stmt, dict):
                if stmt.get('type') == 'assign':
                    modified.add(stmt.get('target', ''))
                elif stmt.get('type') in ('var_decl', 'let'):
                    modified.add(stmt.get('var_name', stmt.get('name', '')))
                # Recurse into nested bodies
                for key in ('then', 'else', 'body'):
                    if key in stmt:
                        modified.update(self._get_modified_vars(stmt[key]))
        return modified
    
    def _is_invariant(self, stmt: Any, modified_vars: Set[str]) -> bool:
        if not isinstance(stmt, dict):
            return True
        
        stmt_type = stmt.get('type', '')
        
        # Only hoist simple assignments
        if stmt_type not in ('var_decl', 'let'):
            return False
        
        # Variable being assigned must not be in modified set (from later)
        var_name = stmt.get('var_name', stmt.get('name', ''))
        if var_name in modified_vars:
            return False
        
        # Check if uses only invariant vars
        used_vars = self._get_used_vars(stmt)
        return not (used_vars & modified_vars)
    
    def _get_used_vars(self, stmt: Any) -> Set[str]:
        used = set()
        if isinstance(stmt, dict):
            if stmt.get('type') == 'var':
                used.add(stmt.get('name', ''))
            for key in ('left', 'right', 'operand', 'value', 'init', 'cond'):
                if key in stmt:
                    used.update(self._get_used_vars(stmt[key]))
            for key in ('args', 'then', 'else', 'body'):
                if key in stmt:
                    for child in stmt[key]:
                        used.update(self._get_used_vars(child))
        return used


class CopyPropagationPass(OptimizationPass):
    """Propagates copies (x = y) to eliminate redundant variables."""
    
    @property
    def name(self) -> str:
        return "copy_propagation"
    
    @property
    def min_level(self) -> OptimizationLevel:
        return OptimizationLevel.O2
    
    def transform(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        for func in ir_data.get('ir_functions', []):
            self._propagate_copies(func)
        return ir_data
    
    def _propagate_copies(self, func: Dict) -> None:
        body = func.get('body', [])
        copies = {}  # x -> y (x is a copy of y)
        new_body = []
        
        for stmt in body:
            # Find new copies
            if isinstance(stmt, dict):
                stmt_type = stmt.get('type', '')
                
                if stmt_type in ('var_decl', 'let'):
                    var_name = stmt.get('var_name', stmt.get('name', ''))
                    init = stmt.get('init', stmt.get('value'))
                    if isinstance(init, dict) and init.get('type') == 'var':
                        copies[var_name] = init.get('name')
                elif stmt_type == 'assign':
                    target = stmt.get('target', '')
                    value = stmt.get('value')
                    if isinstance(value, dict) and value.get('type') == 'var':
                        copies[target] = value.get('name')
                    elif target in copies:
                        del copies[target]  # Invalidate
            
            # Apply copy propagation
            new_stmt = self._apply_copies(stmt, copies)
            new_body.append(new_stmt)
        
        func['body'] = new_body
    
    def _apply_copies(self, stmt: Any, copies: Dict[str, str]) -> Any:
        if not isinstance(stmt, dict):
            return stmt
        
        stmt = copy.copy(stmt)
        
        for key in ('init', 'value', 'cond'):
            if key in stmt:
                stmt[key] = self._apply_copies_expr(stmt[key], copies)
        
        for key in ('args',):
            if key in stmt:
                stmt[key] = [self._apply_copies_expr(a, copies) for a in stmt[key]]
        
        return stmt
    
    def _apply_copies_expr(self, expr: Any, copies: Dict[str, str]) -> Any:
        if not isinstance(expr, dict):
            return expr
        
        if expr.get('type') == 'var':
            var_name = expr.get('name', '')
            if var_name in copies:
                self.stats.copies_propagated += 1
                self.stats.ir_changes += 1
                return {'type': 'var', 'name': copies[var_name]}
        elif expr.get('type') == 'binary':
            return {
                'type': 'binary',
                'op': expr.get('op'),
                'left': self._apply_copies_expr(expr.get('left'), copies),
                'right': self._apply_copies_expr(expr.get('right'), copies)
            }
        elif expr.get('type') == 'unary':
            return {
                'type': 'unary',
                'op': expr.get('op'),
                'operand': self._apply_copies_expr(expr.get('operand'), copies)
            }
        
        return expr


class StrengthReductionPass(OptimizationPass):
    """Replaces expensive operations with cheaper ones.
    
    - x * 2 -> x << 1
    - x * 4 -> x << 2
    - x / 2 -> x >> 1 (for unsigned)
    - x % 2 -> x & 1
    """
    
    @property
    def name(self) -> str:
        return "strength_reduction"
    
    @property
    def min_level(self) -> OptimizationLevel:
        return OptimizationLevel.O2
    
    def transform(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        for func in ir_data.get('ir_functions', []):
            func['body'] = [self._reduce_stmt(stmt) for stmt in func.get('body', [])]
        return ir_data
    
    def _reduce_stmt(self, stmt: Any) -> Any:
        if not isinstance(stmt, dict):
            return stmt
        
        for key in ('init', 'value', 'cond'):
            if key in stmt:
                stmt[key] = self._reduce_expr(stmt[key])
        
        for key in ('then', 'else', 'body'):
            if key in stmt:
                stmt[key] = [self._reduce_stmt(s) for s in stmt[key]]
        
        return stmt
    
    def _reduce_expr(self, expr: Any) -> Any:
        if not isinstance(expr, dict):
            return expr
        
        if expr.get('type') != 'binary':
            # Recurse into children
            if expr.get('type') == 'unary':
                expr['operand'] = self._reduce_expr(expr.get('operand'))
            return expr
        
        op = expr.get('op', '')
        left = self._reduce_expr(expr.get('left'))
        right = self._reduce_expr(expr.get('right'))
        
        right_val = self._get_const(right)
        
        # x * 2^n -> x << n
        if op == '*' and right_val is not None and isinstance(right_val, int):
            shift = self._is_power_of_two(right_val)
            if shift is not None:
                self.stats.strength_reductions += 1
                self.stats.ir_changes += 1
                return {'type': 'binary', 'op': '<<', 'left': left,
                       'right': {'type': 'literal', 'value': shift}}
        
        # x / 2^n -> x >> n (assumes unsigned/positive)
        if op == '/' and right_val is not None and isinstance(right_val, int):
            shift = self._is_power_of_two(right_val)
            if shift is not None:
                self.stats.strength_reductions += 1
                self.stats.ir_changes += 1
                return {'type': 'binary', 'op': '>>', 'left': left,
                       'right': {'type': 'literal', 'value': shift}}
        
        # x % 2^n -> x & (2^n - 1)
        if op == '%' and right_val is not None and isinstance(right_val, int):
            shift = self._is_power_of_two(right_val)
            if shift is not None:
                self.stats.strength_reductions += 1
                self.stats.ir_changes += 1
                return {'type': 'binary', 'op': '&', 'left': left,
                       'right': {'type': 'literal', 'value': right_val - 1}}
        
        return {'type': 'binary', 'op': op, 'left': left, 'right': right}
    
    def _get_const(self, expr: Any) -> Optional[int]:
        if isinstance(expr, int):
            return expr
        if isinstance(expr, dict) and expr.get('type') == 'literal':
            val = expr.get('value')
            if isinstance(val, int):
                return val
        return None
    
    def _is_power_of_two(self, n: int) -> Optional[int]:
        """Return log2(n) if n is a power of 2, else None."""
        if n <= 0:
            return None
        if n & (n - 1) == 0:
            shift = 0
            while (1 << shift) < n:
                shift += 1
            return shift
        return None


def get_o2_passes() -> List[OptimizationPass]:
    """Get all O2 optimization passes (excludes O1 which are added separately)."""
    return [
        CommonSubexpressionEliminationPass(),
        LoopInvariantCodeMotionPass(),
        CopyPropagationPass(),
        StrengthReductionPass(),
    ]


__all__ = [
    'CommonSubexpressionEliminationPass',
    'LoopInvariantCodeMotionPass',
    'CopyPropagationPass',
    'StrengthReductionPass',
    'get_o2_passes'
]
