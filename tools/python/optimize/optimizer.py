#!/usr/bin/env python3
"""STUNIR Optimization Framework.

Provides a multi-pass optimization pipeline with configurable optimization
levels (-O0, -O1, -O2, -O3) and support for various optimization passes.

This module is part of the STUNIR code generation enhancement suite.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable, Type
from enum import Enum, auto
from abc import ABC, abstractmethod
import time
import copy


class OptimizationLevel(Enum):
    """Optimization levels."""
    O0 = 0  # No optimization
    O1 = 1  # Basic optimizations
    O2 = 2  # Standard optimizations
    O3 = 3  # Aggressive optimizations
    Os = 4  # Size optimization
    Oz = 5  # Minimum size


class PassKind(Enum):
    """Kinds of optimization passes."""
    ANALYSIS = auto()       # Analysis-only pass
    TRANSFORM = auto()      # Transformation pass
    CLEANUP = auto()        # Cleanup pass
    TARGET_SPECIFIC = auto()  # Target-specific pass


@dataclass
class OptimizationStats:
    """Statistics from an optimization pass."""
    pass_name: str
    ir_changes: int = 0
    statements_removed: int = 0
    statements_modified: int = 0
    constants_folded: int = 0
    subexpressions_eliminated: int = 0
    functions_inlined: int = 0
    loops_optimized: int = 0
    time_ms: float = 0.0
    
    def __str__(self) -> str:
        return (f"{self.pass_name}: "
                f"changes={self.ir_changes}, "
                f"removed={self.statements_removed}, "
                f"modified={self.statements_modified}, "
                f"time={self.time_ms:.2f}ms")


class OptimizationPass(ABC):
    """Abstract base class for optimization passes."""
    
    def __init__(self):
        self.stats = OptimizationStats(pass_name=self.name)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this pass."""
        pass
    
    @property
    def kind(self) -> PassKind:
        """Return the kind of this pass."""
        return PassKind.TRANSFORM
    
    @property
    def min_level(self) -> OptimizationLevel:
        """Minimum optimization level to run this pass."""
        return OptimizationLevel.O1
    
    @abstractmethod
    def run(self, ir_data: Dict) -> Dict:
        """Run the optimization pass on IR data.
        
        Args:
            ir_data: The IR data to optimize
            
        Returns:
            Optimized IR data
        """
        pass
    
    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = OptimizationStats(pass_name=self.name)


class DeadCodeEliminationPass(OptimizationPass):
    """Eliminates dead (unreachable) code."""
    
    @property
    def name(self) -> str:
        return "dead_code_elimination"
    
    @property
    def min_level(self) -> OptimizationLevel:
        return OptimizationLevel.O1
    
    def run(self, ir_data: Dict) -> Dict:
        start_time = time.time()
        result = copy.deepcopy(ir_data)
        
        for func in result.get('ir_functions', []):
            self._eliminate_dead_code(func)
        
        self.stats.time_ms = (time.time() - start_time) * 1000
        return result
    
    def _eliminate_dead_code(self, func: Dict) -> None:
        """Eliminate dead code in a function."""
        body = func.get('body', [])
        new_body = []
        reachable = True
        
        for stmt in body:
            if not reachable:
                self.stats.statements_removed += 1
                continue
            
            new_body.append(stmt)
            
            if self._is_terminating(stmt):
                reachable = False
        
        if len(new_body) != len(body):
            self.stats.ir_changes += 1
        
        func['body'] = new_body
    
    def _is_terminating(self, stmt: Any) -> bool:
        """Check if statement terminates control flow."""
        if not isinstance(stmt, dict):
            return False
        return stmt.get('type') in ('return', 'break', 'continue', 'goto', 'throw')


class ConstantFoldingPass(OptimizationPass):
    """Folds constant expressions at compile time."""
    
    @property
    def name(self) -> str:
        return "constant_folding"
    
    @property
    def min_level(self) -> OptimizationLevel:
        return OptimizationLevel.O1
    
    def run(self, ir_data: Dict) -> Dict:
        start_time = time.time()
        result = copy.deepcopy(ir_data)
        
        for func in result.get('ir_functions', []):
            self._fold_constants(func)
        
        self.stats.time_ms = (time.time() - start_time) * 1000
        return result
    
    def _fold_constants(self, func: Dict) -> None:
        """Fold constants in a function."""
        body = func.get('body', [])
        func['body'] = [self._fold_stmt(stmt) for stmt in body]
    
    def _fold_stmt(self, stmt: Any) -> Any:
        """Fold constants in a statement."""
        if not isinstance(stmt, dict):
            return stmt
        
        stmt_type = stmt.get('type', '')
        
        if stmt_type in ('var_decl', 'let'):
            if 'init' in stmt:
                stmt['init'] = self._fold_expr(stmt['init'])
            if 'value' in stmt:
                stmt['value'] = self._fold_expr(stmt['value'])
        elif stmt_type == 'assign':
            stmt['value'] = self._fold_expr(stmt.get('value'))
        elif stmt_type == 'return':
            if 'value' in stmt:
                stmt['value'] = self._fold_expr(stmt['value'])
        elif stmt_type == 'if':
            stmt['cond'] = self._fold_expr(stmt.get('cond'))
            stmt['then'] = [self._fold_stmt(s) for s in stmt.get('then', [])]
            if 'else' in stmt:
                stmt['else'] = [self._fold_stmt(s) for s in stmt.get('else', [])]
        elif stmt_type in ('while', 'for'):
            stmt['cond'] = self._fold_expr(stmt.get('cond'))
            stmt['body'] = [self._fold_stmt(s) for s in stmt.get('body', [])]
        elif stmt_type == 'call':
            stmt['args'] = [self._fold_expr(a) for a in stmt.get('args', [])]
        
        return stmt
    
    def _fold_expr(self, expr: Any) -> Any:
        """Fold constants in an expression."""
        if not isinstance(expr, dict):
            return expr
        
        expr_type = expr.get('type', '')
        
        if expr_type == 'binary':
            left = self._fold_expr(expr.get('left'))
            right = self._fold_expr(expr.get('right'))
            op = expr.get('op', '')
            
            # Try to evaluate
            left_val = self._get_const_value(left)
            right_val = self._get_const_value(right)
            
            if left_val is not None and right_val is not None:
                result = self._eval_binary(op, left_val, right_val)
                if result is not None:
                    self.stats.constants_folded += 1
                    self.stats.ir_changes += 1
                    return {'type': 'literal', 'value': result}
            
            return {'type': 'binary', 'op': op, 'left': left, 'right': right}
        
        elif expr_type == 'unary':
            operand = self._fold_expr(expr.get('operand'))
            op = expr.get('op', '')
            
            operand_val = self._get_const_value(operand)
            if operand_val is not None:
                result = self._eval_unary(op, operand_val)
                if result is not None:
                    self.stats.constants_folded += 1
                    self.stats.ir_changes += 1
                    return {'type': 'literal', 'value': result}
            
            return {'type': 'unary', 'op': op, 'operand': operand}
        
        return expr
    
    def _get_const_value(self, expr: Any) -> Optional[Any]:
        """Get constant value from expression if possible."""
        if isinstance(expr, (int, float, bool)):
            return expr
        if isinstance(expr, dict):
            if expr.get('type') == 'literal':
                return expr.get('value')
        return None
    
    def _eval_binary(self, op: str, left: Any, right: Any) -> Optional[Any]:
        """Evaluate binary operation."""
        try:
            if op == '+':
                return left + right
            elif op == '-':
                return left - right
            elif op == '*':
                return left * right
            elif op == '/':
                if right == 0:
                    return None
                if isinstance(left, int) and isinstance(right, int):
                    return left // right
                return left / right
            elif op == '%':
                if right == 0:
                    return None
                return left % right
            elif op == '&':
                return left & right
            elif op == '|':
                return left | right
            elif op == '^':
                return left ^ right
            elif op == '<<':
                return left << right
            elif op == '>>':
                return left >> right
            elif op == '==':
                return left == right
            elif op == '!=':
                return left != right
            elif op == '<':
                return left < right
            elif op == '>':
                return left > right
            elif op == '<=':
                return left <= right
            elif op == '>=':
                return left >= right
            elif op in ('&&', 'and'):
                return left and right
            elif op in ('||', 'or'):
                return left or right
        except (TypeError, OverflowError):
            pass
        return None
    
    def _eval_unary(self, op: str, operand: Any) -> Optional[Any]:
        """Evaluate unary operation."""
        try:
            if op == '-':
                return -operand
            elif op == '+':
                return +operand
            elif op in ('!', 'not'):
                return not operand
            elif op == '~':
                return ~operand
        except (TypeError, OverflowError):
            pass
        return None


class CommonSubexpressionEliminationPass(OptimizationPass):
    """Eliminates common subexpressions."""
    
    @property
    def name(self) -> str:
        return "common_subexpression_elimination"
    
    @property
    def min_level(self) -> OptimizationLevel:
        return OptimizationLevel.O2
    
    def run(self, ir_data: Dict) -> Dict:
        start_time = time.time()
        result = copy.deepcopy(ir_data)
        
        for func in result.get('ir_functions', []):
            self._eliminate_cse(func)
        
        self.stats.time_ms = (time.time() - start_time) * 1000
        return result
    
    def _eliminate_cse(self, func: Dict) -> None:
        """Eliminate common subexpressions in a function."""
        body = func.get('body', [])
        expr_map: Dict[str, str] = {}  # expr_hash -> temp_var
        temp_counter = 0
        new_body = []
        insertions = []
        
        for stmt in body:
            new_stmt, temps, temp_counter = self._process_stmt(
                stmt, expr_map, temp_counter)
            insertions.extend(temps)
            new_body.append(new_stmt)
        
        # Insert temp declarations at the beginning
        for temp_name, temp_expr in insertions:
            temp_decl = {
                'type': 'var_decl',
                'var_name': temp_name,
                'init': temp_expr
            }
            new_body.insert(0, temp_decl)
        
        func['body'] = new_body
    
    def _process_stmt(self, stmt: Any, expr_map: Dict[str, str],
                     temp_counter: int) -> Tuple[Any, List[Tuple[str, Any]], int]:
        """Process a statement for CSE."""
        insertions = []
        
        if not isinstance(stmt, dict):
            return stmt, insertions, temp_counter
        
        stmt_type = stmt.get('type', '')
        
        if stmt_type in ('var_decl', 'let'):
            if 'init' in stmt:
                stmt['init'], new_insertions, temp_counter = self._process_expr(
                    stmt['init'], expr_map, temp_counter)
                insertions.extend(new_insertions)
        elif stmt_type == 'assign':
            stmt['value'], new_insertions, temp_counter = self._process_expr(
                stmt.get('value'), expr_map, temp_counter)
            insertions.extend(new_insertions)
        elif stmt_type == 'return':
            if 'value' in stmt:
                stmt['value'], new_insertions, temp_counter = self._process_expr(
                    stmt['value'], expr_map, temp_counter)
                insertions.extend(new_insertions)
        
        return stmt, insertions, temp_counter
    
    def _process_expr(self, expr: Any, expr_map: Dict[str, str],
                     temp_counter: int) -> Tuple[Any, List[Tuple[str, Any]], int]:
        """Process an expression for CSE."""
        insertions = []
        
        if not isinstance(expr, dict):
            return expr, insertions, temp_counter
        
        expr_type = expr.get('type', '')
        
        if expr_type == 'binary':
            # Process children first
            left, left_ins, temp_counter = self._process_expr(
                expr.get('left'), expr_map, temp_counter)
            right, right_ins, temp_counter = self._process_expr(
                expr.get('right'), expr_map, temp_counter)
            insertions.extend(left_ins)
            insertions.extend(right_ins)
            
            new_expr = {'type': 'binary', 'op': expr.get('op'), 
                       'left': left, 'right': right}
            
            # Check if this expression was seen before
            expr_hash = self._hash_expr(new_expr)
            if expr_hash in expr_map:
                self.stats.subexpressions_eliminated += 1
                self.stats.ir_changes += 1
                return {'type': 'var', 'name': expr_map[expr_hash]}, insertions, temp_counter
            
            # Add to map
            temp_name = f'_cse_{temp_counter}'
            temp_counter += 1
            expr_map[expr_hash] = temp_name
            insertions.append((temp_name, new_expr))
            
            return {'type': 'var', 'name': temp_name}, insertions, temp_counter
        
        return expr, insertions, temp_counter
    
    def _hash_expr(self, expr: Any) -> str:
        """Create hash string for expression."""
        if isinstance(expr, dict):
            expr_type = expr.get('type', '')
            if expr_type == 'binary':
                left_hash = self._hash_expr(expr.get('left'))
                right_hash = self._hash_expr(expr.get('right'))
                return f"bin:{expr.get('op')}:{left_hash}:{right_hash}"
            elif expr_type == 'var':
                return f"var:{expr.get('name')}"
            elif expr_type == 'literal':
                return f"lit:{expr.get('value')}"
        return str(expr)


class FunctionInliningPass(OptimizationPass):
    """Inlines small functions."""
    
    def __init__(self, max_statements: int = 10):
        super().__init__()
        self.max_statements = max_statements
    
    @property
    def name(self) -> str:
        return "function_inlining"
    
    @property
    def min_level(self) -> OptimizationLevel:
        return OptimizationLevel.O2
    
    def run(self, ir_data: Dict) -> Dict:
        start_time = time.time()
        result = copy.deepcopy(ir_data)
        
        # Build function map
        func_map = {}
        for func in result.get('ir_functions', []):
            name = func.get('name', '')
            if self._should_inline(func):
                func_map[name] = func
        
        # Inline in all functions
        for func in result.get('ir_functions', []):
            self._inline_calls(func, func_map)
        
        self.stats.time_ms = (time.time() - start_time) * 1000
        return result
    
    def _should_inline(self, func: Dict) -> bool:
        """Check if function should be inlined."""
        body = func.get('body', [])
        if len(body) > self.max_statements:
            return False
        # Don't inline recursive functions
        name = func.get('name', '')
        for stmt in body:
            if self._has_call_to(stmt, name):
                return False
        return True
    
    def _has_call_to(self, stmt: Any, func_name: str) -> bool:
        """Check if statement has call to function."""
        if not isinstance(stmt, dict):
            return False
        if stmt.get('type') == 'call' and stmt.get('func') == func_name:
            return True
        for key in ('then', 'else', 'body', 'args'):
            if key in stmt:
                for child in stmt[key]:
                    if self._has_call_to(child, func_name):
                        return True
        return False
    
    def _inline_calls(self, func: Dict, func_map: Dict[str, Dict]) -> None:
        """Inline calls in a function."""
        body = func.get('body', [])
        new_body = []
        
        for stmt in body:
            inlined = self._try_inline(stmt, func_map)
            if inlined:
                new_body.extend(inlined)
            else:
                new_body.append(stmt)
        
        func['body'] = new_body
    
    def _try_inline(self, stmt: Any, func_map: Dict[str, Dict]) -> Optional[List[Any]]:
        """Try to inline a call statement."""
        if not isinstance(stmt, dict):
            return None
        
        if stmt.get('type') != 'call':
            return None
        
        func_name = stmt.get('func', '')
        if func_name not in func_map:
            return None
        
        callee = func_map[func_name]
        args = stmt.get('args', [])
        params = callee.get('params', [])
        
        # Create inlined body with parameter substitution
        inlined = []
        
        # Bind arguments to parameters
        for i, param in enumerate(params):
            param_name = param.get('name', param) if isinstance(param, dict) else param
            arg = args[i] if i < len(args) else {'type': 'literal', 'value': 0}
            inlined.append({
                'type': 'var_decl',
                'var_name': f'_inline_{param_name}',
                'init': arg
            })
        
        # Copy body with parameter substitution
        for body_stmt in callee.get('body', []):
            inlined.append(self._substitute_params(body_stmt, params))
        
        self.stats.functions_inlined += 1
        self.stats.ir_changes += 1
        
        return inlined
    
    def _substitute_params(self, stmt: Any, params: List) -> Any:
        """Substitute parameters in inlined statement."""
        # Simple implementation - just copy
        return copy.deepcopy(stmt)


class LoopOptimizationPass(OptimizationPass):
    """Optimizes loops (invariant code motion, unrolling)."""
    
    def __init__(self, unroll_threshold: int = 4):
        super().__init__()
        self.unroll_threshold = unroll_threshold
    
    @property
    def name(self) -> str:
        return "loop_optimization"
    
    @property
    def min_level(self) -> OptimizationLevel:
        return OptimizationLevel.O2
    
    def run(self, ir_data: Dict) -> Dict:
        start_time = time.time()
        result = copy.deepcopy(ir_data)
        
        for func in result.get('ir_functions', []):
            self._optimize_loops(func)
        
        self.stats.time_ms = (time.time() - start_time) * 1000
        return result
    
    def _optimize_loops(self, func: Dict) -> None:
        """Optimize loops in a function."""
        body = func.get('body', [])
        func['body'] = [self._optimize_stmt(stmt) for stmt in body]
    
    def _optimize_stmt(self, stmt: Any) -> Any:
        """Optimize a statement."""
        if not isinstance(stmt, dict):
            return stmt
        
        stmt_type = stmt.get('type', '')
        
        if stmt_type in ('while', 'for'):
            return self._optimize_loop(stmt)
        elif stmt_type == 'if':
            stmt['then'] = [self._optimize_stmt(s) for s in stmt.get('then', [])]
            if 'else' in stmt:
                stmt['else'] = [self._optimize_stmt(s) for s in stmt.get('else', [])]
        
        return stmt
    
    def _optimize_loop(self, loop: Dict) -> Any:
        """Optimize a loop."""
        # Try loop invariant code motion
        loop = self._hoist_invariants(loop)
        
        # Try loop unrolling for small constant bounds
        unrolled = self._try_unroll(loop)
        if unrolled:
            return unrolled
        
        return loop
    
    def _hoist_invariants(self, loop: Dict) -> Dict:
        """Move loop-invariant code outside the loop."""
        body = loop.get('body', [])
        modified_vars = self._get_modified_vars(body)
        
        invariant = []
        new_body = []
        
        for stmt in body:
            if self._is_invariant(stmt, modified_vars):
                invariant.append(stmt)
            else:
                new_body.append(stmt)
        
        if invariant:
            self.stats.loops_optimized += 1
            self.stats.ir_changes += 1
        
        loop['body'] = new_body
        
        # Return block with hoisted code before loop
        if invariant:
            return {
                'type': 'block',
                'body': invariant + [loop]
            }
        return loop
    
    def _get_modified_vars(self, body: List) -> Set[str]:
        """Get variables modified in loop body."""
        modified = set()
        for stmt in body:
            if isinstance(stmt, dict):
                if stmt.get('type') == 'assign':
                    modified.add(stmt.get('target', ''))
                elif stmt.get('type') in ('var_decl', 'let'):
                    modified.add(stmt.get('var_name', stmt.get('name', '')))
        return modified
    
    def _is_invariant(self, stmt: Any, modified_vars: Set[str]) -> bool:
        """Check if statement is loop-invariant."""
        if not isinstance(stmt, dict):
            return True
        
        stmt_type = stmt.get('type', '')
        
        # Assignments to modified vars are not invariant
        if stmt_type == 'assign' and stmt.get('target') in modified_vars:
            return False
        
        # Check if uses only invariant vars
        used_vars = self._get_used_vars(stmt)
        return not (used_vars & modified_vars)
    
    def _get_used_vars(self, stmt: Any) -> Set[str]:
        """Get variables used in statement."""
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
        elif isinstance(stmt, str):
            used.add(stmt)
        
        return used
    
    def _try_unroll(self, loop: Dict) -> Optional[Dict]:
        """Try to unroll a loop with constant bounds."""
        loop_type = loop.get('type', '')
        
        if loop_type != 'for':
            return None
        
        # Try to get loop bounds
        init = loop.get('init', {})
        cond = loop.get('cond', {})
        
        # Simple pattern: for (i = 0; i < N; i++)
        if not isinstance(cond, dict) or cond.get('type') != 'binary':
            return None
        
        if cond.get('op') not in ('<', '<='):
            return None
        
        right = cond.get('right')
        bound = None
        if isinstance(right, (int, float)):
            bound = int(right)
        elif isinstance(right, dict) and right.get('type') == 'literal':
            bound = right.get('value')
        
        if bound is None or bound > self.unroll_threshold:
            return None
        
        # Unroll the loop
        body = loop.get('body', [])
        unrolled = []
        
        for i in range(bound):
            for stmt in body:
                unrolled.append(copy.deepcopy(stmt))
        
        self.stats.loops_optimized += 1
        self.stats.ir_changes += 1
        
        return {'type': 'block', 'body': unrolled}


class Optimizer:
    """Main optimizer coordinating multiple passes."""
    
    def __init__(self, level: OptimizationLevel = OptimizationLevel.O2):
        self.level = level
        self.passes: List[OptimizationPass] = []
        self.stats: List[OptimizationStats] = []
        self._register_default_passes()
    
    def _register_default_passes(self) -> None:
        """Register default optimization passes."""
        self.passes = [
            DeadCodeEliminationPass(),
            ConstantFoldingPass(),
            CommonSubexpressionEliminationPass(),
            FunctionInliningPass(),
            LoopOptimizationPass(),
        ]
    
    def register_pass(self, pass_: OptimizationPass) -> None:
        """Register a custom optimization pass."""
        self.passes.append(pass_)
    
    def optimize(self, ir_data: Dict) -> Dict:
        """Run all applicable optimization passes."""
        self.stats.clear()
        result = ir_data
        
        for pass_ in self.passes:
            if pass_.min_level.value <= self.level.value:
                pass_.reset_stats()
                result = pass_.run(result)
                self.stats.append(pass_.stats)
        
        return result
    
    def get_stats_summary(self) -> str:
        """Get summary of optimization statistics."""
        lines = [f"Optimization Level: {self.level.name}"]
        lines.append("-" * 40)
        
        total_changes = 0
        total_time = 0.0
        
        for stat in self.stats:
            lines.append(str(stat))
            total_changes += stat.ir_changes
            total_time += stat.time_ms
        
        lines.append("-" * 40)
        lines.append(f"Total IR changes: {total_changes}")
        lines.append(f"Total time: {total_time:.2f}ms")
        
        return '\n'.join(lines)


def create_optimizer(level: str = 'O2') -> Optimizer:
    """Factory function to create an optimizer."""
    level_map = {
        'O0': OptimizationLevel.O0,
        'O1': OptimizationLevel.O1,
        'O2': OptimizationLevel.O2,
        'O3': OptimizationLevel.O3,
        'Os': OptimizationLevel.Os,
        'Oz': OptimizationLevel.Oz,
    }
    opt_level = level_map.get(level.upper(), OptimizationLevel.O2)
    return Optimizer(level=opt_level)


__all__ = [
    'OptimizationLevel', 'PassKind', 'OptimizationStats',
    'OptimizationPass', 'DeadCodeEliminationPass', 'ConstantFoldingPass',
    'CommonSubexpressionEliminationPass', 'FunctionInliningPass',
    'LoopOptimizationPass', 'Optimizer', 'create_optimizer'
]
