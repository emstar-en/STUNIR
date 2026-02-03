#!/usr/bin/env python3
"""STUNIR O3 Optimization Passes.

Aggressive optimizations (includes all O1 and O2 optimizations):
- Function inlining (inline small functions)
- Loop unrolling (unroll small loops)
- Aggressive constant propagation (interprocedural)
- Dead store elimination
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
import copy

from .optimization_pass import OptimizationPass, OptimizationLevel


class FunctionInliningPass(OptimizationPass):
    """Inlines small functions at call sites."""
    
    def __init__(self, max_statements: int = 10, max_inline_depth: int = 2):
        super().__init__()
        self.max_statements = max_statements
        self.max_inline_depth = max_inline_depth
    
    @property
    def name(self) -> str:
        return "function_inlining"
    
    @property
    def min_level(self) -> OptimizationLevel:
        return OptimizationLevel.O3
    
    def analyze(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify functions suitable for inlining."""
        inlinable = {}
        for func in ir_data.get('ir_functions', []):
            name = func.get('name', '')
            if self._should_inline(func):
                inlinable[name] = func
        return {'inlinable': inlinable}
    
    def _should_inline(self, func: Dict) -> bool:
        body = func.get('body', [])
        name = func.get('name', '')
        
        # Size check
        if len(body) > self.max_statements:
            return False
        
        # No recursive functions
        if self._has_recursive_call(func, name):
            return False
        
        # Mark functions with 'inline' hint
        if func.get('inline', False):
            return True
        
        # Default: inline if small enough
        return len(body) <= 3
    
    def _has_recursive_call(self, func: Dict, name: str) -> bool:
        return self._contains_call_to(func.get('body', []), name)
    
    def _contains_call_to(self, body: List, name: str) -> bool:
        for stmt in body:
            if isinstance(stmt, dict):
                if stmt.get('type') == 'call' and stmt.get('func') == name:
                    return True
                for key in ('then', 'else', 'body', 'args'):
                    if key in stmt:
                        items = stmt[key]
                        if isinstance(items, list):
                            if self._contains_call_to(items, name):
                                return True
        return False
    
    def transform(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        inlinable = self._analysis_cache.get('inlinable', {})
        
        for func in ir_data.get('ir_functions', []):
            func['body'] = self._inline_in_body(func.get('body', []), inlinable, 0)
        
        return ir_data
    
    def _inline_in_body(self, body: List, inlinable: Dict, depth: int) -> List:
        if depth >= self.max_inline_depth:
            return body
        
        new_body = []
        for stmt in body:
            if isinstance(stmt, dict) and stmt.get('type') == 'call':
                func_name = stmt.get('func', '')
                if func_name in inlinable:
                    inlined = self._inline_call(stmt, inlinable[func_name])
                    # Recursively inline in the inlined body
                    inlined = self._inline_in_body(inlined, inlinable, depth + 1)
                    new_body.extend(inlined)
                    self.stats.functions_inlined += 1
                    self.stats.ir_changes += 1
                    continue
            
            # Process nested blocks
            if isinstance(stmt, dict):
                stmt = copy.copy(stmt)
                for key in ('then', 'else', 'body'):
                    if key in stmt:
                        stmt[key] = self._inline_in_body(stmt[key], inlinable, depth)
            
            new_body.append(stmt)
        
        return new_body
    
    def _inline_call(self, call_stmt: Dict, callee: Dict) -> List:
        args = call_stmt.get('args', [])
        params = callee.get('params', [])
        result_var = call_stmt.get('result_var')  # Where to store result
        
        inlined = []
        
        # Bind arguments to parameters
        for i, param in enumerate(params):
            param_name = param.get('name', param) if isinstance(param, dict) else str(param)
            arg = args[i] if i < len(args) else {'type': 'literal', 'value': 0}
            inlined.append({
                'type': 'var_decl',
                'var_name': f'_inline_{param_name}_{id(call_stmt)}',
                'init': copy.deepcopy(arg),
                '_inlined': True
            })
        
        # Copy body with parameter substitution
        param_map = {}
        for i, param in enumerate(params):
            param_name = param.get('name', param) if isinstance(param, dict) else str(param)
            param_map[param_name] = f'_inline_{param_name}_{id(call_stmt)}'
        
        for stmt in callee.get('body', []):
            substituted = self._substitute_vars(copy.deepcopy(stmt), param_map)
            # Transform return into assignment if result_var is set
            if isinstance(substituted, dict) and substituted.get('type') == 'return':
                if result_var and 'value' in substituted:
                    inlined.append({
                        'type': 'assign',
                        'target': result_var,
                        'value': substituted['value']
                    })
            else:
                inlined.append(substituted)
        
        return inlined
    
    def _substitute_vars(self, stmt: Any, var_map: Dict[str, str]) -> Any:
        if isinstance(stmt, dict):
            if stmt.get('type') == 'var':
                name = stmt.get('name', '')
                if name in var_map:
                    return {'type': 'var', 'name': var_map[name]}
            
            result = copy.copy(stmt)
            for key, val in stmt.items():
                if isinstance(val, dict):
                    result[key] = self._substitute_vars(val, var_map)
                elif isinstance(val, list):
                    result[key] = [self._substitute_vars(v, var_map) for v in val]
            return result
        return stmt


class LoopUnrollingPass(OptimizationPass):
    """Unrolls small loops with constant bounds."""
    
    def __init__(self, unroll_threshold: int = 8, min_unroll_factor: int = 2):
        super().__init__()
        self.unroll_threshold = unroll_threshold
        self.min_unroll_factor = min_unroll_factor
    
    @property
    def name(self) -> str:
        return "loop_unrolling"
    
    @property
    def min_level(self) -> OptimizationLevel:
        return OptimizationLevel.O3
    
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
        
        if stmt_type == 'for':
            unrolled = self._try_unroll_for(stmt)
            if unrolled:
                return unrolled
        elif stmt_type == 'while':
            unrolled = self._try_unroll_while(stmt)
            if unrolled:
                return unrolled
        
        # Process nested
        stmt = copy.copy(stmt)
        for key in ('then', 'else', 'body'):
            if key in stmt:
                stmt[key] = self._process_body(stmt[key])
        
        return stmt
    
    def _try_unroll_for(self, loop: Dict) -> Optional[List]:
        """Try to fully unroll a for loop."""
        cond = loop.get('cond', {})
        
        if not isinstance(cond, dict) or cond.get('type') != 'binary':
            return None
        
        if cond.get('op') not in ('<', '<='):
            return None
        
        # Get bound
        right = cond.get('right')
        bound = self._get_const(right)
        
        if bound is None or bound > self.unroll_threshold:
            return None
        
        if cond.get('op') == '<=':
            bound += 1
        
        # Unroll
        body = loop.get('body', [])
        unrolled = []
        
        for i in range(bound):
            for stmt in body:
                unrolled.append(copy.deepcopy(stmt))
        
        self.stats.loops_optimized += 1
        self.stats.ir_changes += 1
        
        return unrolled
    
    def _try_unroll_while(self, loop: Dict) -> Optional[List]:
        """Try to partially unroll a while loop."""
        body = loop.get('body', [])
        
        # Only unroll small loops
        if len(body) > 3:
            return None
        
        # Partial unroll by factor of 2
        unrolled_body = []
        for _ in range(self.min_unroll_factor):
            for stmt in body:
                unrolled_body.append(copy.deepcopy(stmt))
        
        self.stats.loops_optimized += 1
        self.stats.ir_changes += 1
        
        return [{
            'type': 'while',
            'cond': copy.deepcopy(loop.get('cond')),
            'body': unrolled_body,
            '_unrolled': True
        }]
    
    def _get_const(self, expr: Any) -> Optional[int]:
        if isinstance(expr, int):
            return expr
        if isinstance(expr, dict) and expr.get('type') == 'literal':
            val = expr.get('value')
            if isinstance(val, int):
                return val
        return None


class AggressiveConstantPropagationPass(OptimizationPass):
    """Aggressive constant propagation with interprocedural analysis."""
    
    @property
    def name(self) -> str:
        return "aggressive_constant_propagation"
    
    @property
    def min_level(self) -> OptimizationLevel:
        return OptimizationLevel.O3
    
    def analyze(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build global constant map."""
        global_consts = {}
        
        # Find global constant values
        for var in ir_data.get('ir_globals', []):
            if isinstance(var, dict):
                name = var.get('name', '')
                value = var.get('value')
                if self._is_const(value):
                    global_consts[name] = self._get_const_val(value)
        
        return {'global_consts': global_consts}
    
    def _is_const(self, expr: Any) -> bool:
        if isinstance(expr, (int, float, bool, str)):
            return True
        if isinstance(expr, dict) and expr.get('type') == 'literal':
            return True
        return False
    
    def _get_const_val(self, expr: Any) -> Any:
        if isinstance(expr, (int, float, bool, str)):
            return expr
        if isinstance(expr, dict) and expr.get('type') == 'literal':
            return expr.get('value')
        return None
    
    def transform(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        global_consts = self._analysis_cache.get('global_consts', {})
        
        for func in ir_data.get('ir_functions', []):
            # Combine global and local constants
            local_consts = dict(global_consts)
            func['body'] = self._propagate_body(func.get('body', []), local_consts)
        
        return ir_data
    
    def _propagate_body(self, body: List, consts: Dict) -> List:
        new_body = []
        
        for stmt in body:
            if isinstance(stmt, dict):
                stmt_type = stmt.get('type', '')
                
                # Track new constants
                if stmt_type in ('var_decl', 'let'):
                    var_name = stmt.get('var_name', stmt.get('name', ''))
                    init = stmt.get('init', stmt.get('value'))
                    if self._is_const(init):
                        consts[var_name] = self._get_const_val(init)
                    elif var_name in consts:
                        del consts[var_name]
                
                # Apply propagation
                new_stmt = self._propagate_stmt(stmt, consts)
                new_body.append(new_stmt)
            else:
                new_body.append(stmt)
        
        return new_body
    
    def _propagate_stmt(self, stmt: Any, consts: Dict) -> Any:
        if not isinstance(stmt, dict):
            return stmt
        
        stmt = copy.copy(stmt)
        
        for key in ('init', 'value', 'cond'):
            if key in stmt:
                stmt[key] = self._propagate_expr(stmt[key], consts)
        
        for key in ('args',):
            if key in stmt:
                stmt[key] = [self._propagate_expr(a, consts) for a in stmt[key]]
        
        for key in ('then', 'else', 'body'):
            if key in stmt:
                stmt[key] = self._propagate_body(stmt[key], dict(consts))
        
        return stmt
    
    def _propagate_expr(self, expr: Any, consts: Dict) -> Any:
        if not isinstance(expr, dict):
            return expr
        
        if expr.get('type') == 'var':
            name = expr.get('name', '')
            if name in consts:
                self.stats.ir_changes += 1
                return {'type': 'literal', 'value': consts[name]}
        elif expr.get('type') == 'binary':
            return {
                'type': 'binary',
                'op': expr.get('op'),
                'left': self._propagate_expr(expr.get('left'), consts),
                'right': self._propagate_expr(expr.get('right'), consts)
            }
        elif expr.get('type') == 'unary':
            return {
                'type': 'unary',
                'op': expr.get('op'),
                'operand': self._propagate_expr(expr.get('operand'), consts)
            }
        
        return expr


class DeadStoreEliminationPass(OptimizationPass):
    """Eliminates dead stores (writes that are never read)."""
    
    @property
    def name(self) -> str:
        return "dead_store_elimination"
    
    @property
    def min_level(self) -> OptimizationLevel:
        return OptimizationLevel.O3
    
    def analyze(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find variables that are used."""
        used_vars = set()
        for func in ir_data.get('ir_functions', []):
            used_vars.update(self._get_used_vars(func.get('body', [])))
        return {'used_vars': used_vars}
    
    def _get_used_vars(self, body: List) -> Set[str]:
        used = set()
        for stmt in body:
            if isinstance(stmt, dict):
                stmt_type = stmt.get('type', '')
                
                # Collect reads
                if stmt_type == 'return' and 'value' in stmt:
                    used.update(self._get_vars_in_expr(stmt['value']))
                elif stmt_type == 'assign':
                    used.update(self._get_vars_in_expr(stmt.get('value')))
                elif stmt_type == 'call':
                    for arg in stmt.get('args', []):
                        used.update(self._get_vars_in_expr(arg))
                
                # Recurse
                for key in ('then', 'else', 'body'):
                    if key in stmt:
                        used.update(self._get_used_vars(stmt[key]))
                
                # Conditions are also reads
                if 'cond' in stmt:
                    used.update(self._get_vars_in_expr(stmt['cond']))
        
        return used
    
    def _get_vars_in_expr(self, expr: Any) -> Set[str]:
        if not isinstance(expr, dict):
            return set()
        
        if expr.get('type') == 'var':
            return {expr.get('name', '')}
        
        vars_ = set()
        for key in ('left', 'right', 'operand', 'init', 'value'):
            if key in expr:
                vars_.update(self._get_vars_in_expr(expr[key]))
        for key in ('args',):
            if key in expr:
                for a in expr[key]:
                    vars_.update(self._get_vars_in_expr(a))
        
        return vars_
    
    def transform(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        for func in ir_data.get('ir_functions', []):
            func['body'] = self._eliminate_dead_stores(func.get('body', []))
        return ir_data
    
    def _eliminate_dead_stores(self, body: List) -> List:
        # Backward pass to find live variables
        live = set()
        
        # First, collect all vars used in returns (always live)
        for stmt in body:
            if isinstance(stmt, dict) and stmt.get('type') == 'return':
                if 'value' in stmt:
                    live.update(self._get_vars_in_expr(stmt['value']))
        
        # Mark stores that feed into live vars
        changed = True
        while changed:
            changed = False
            for stmt in body:
                if isinstance(stmt, dict):
                    stmt_type = stmt.get('type', '')
                    if stmt_type in ('var_decl', 'let'):
                        var_name = stmt.get('var_name', stmt.get('name', ''))
                        if var_name in live:
                            init = stmt.get('init', stmt.get('value'))
                            new_vars = self._get_vars_in_expr(init)
                            if not new_vars.issubset(live):
                                live.update(new_vars)
                                changed = True
                    elif stmt_type == 'assign':
                        target = stmt.get('target', '')
                        if target in live:
                            new_vars = self._get_vars_in_expr(stmt.get('value'))
                            if not new_vars.issubset(live):
                                live.update(new_vars)
                                changed = True
        
        # Remove dead stores
        new_body = []
        for stmt in body:
            if isinstance(stmt, dict):
                stmt_type = stmt.get('type', '')
                
                if stmt_type in ('var_decl', 'let'):
                    var_name = stmt.get('var_name', stmt.get('name', ''))
                    if var_name not in live and not stmt.get('_cse_temp') and not stmt.get('_inlined'):
                        self.stats.dead_stores_removed += 1
                        self.stats.ir_changes += 1
                        continue
                
                # Process nested
                stmt = copy.copy(stmt)
                for key in ('then', 'else', 'body'):
                    if key in stmt:
                        stmt[key] = self._eliminate_dead_stores(stmt[key])
            
            new_body.append(stmt)
        
        return new_body


def get_o3_passes() -> List[OptimizationPass]:
    """Get all O3 optimization passes (excludes O1/O2 which are added separately)."""
    return [
        FunctionInliningPass(),
        LoopUnrollingPass(),
        AggressiveConstantPropagationPass(),
        DeadStoreEliminationPass(),
    ]


__all__ = [
    'FunctionInliningPass',
    'LoopUnrollingPass',
    'AggressiveConstantPropagationPass',
    'DeadStoreEliminationPass',
    'get_o3_passes'
]
