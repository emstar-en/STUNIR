#!/usr/bin/env python3
"""STUNIR O1 Optimization Passes.

Basic optimizations:
- Dead code elimination (remove unreachable code)
- Constant folding (evaluate constant expressions at compile time)
- Constant propagation (replace variables with known constant values)
- Simple algebraic simplifications (x * 1 = x, x + 0 = x)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set
import copy

from .optimization_pass import OptimizationPass, OptimizationLevel


class DeadCodeEliminationPass(OptimizationPass):
    """Eliminates dead (unreachable) code after return/break/continue."""
    
    @property
    def name(self) -> str:
        return "dead_code_elimination"
    
    @property
    def min_level(self) -> OptimizationLevel:
        return OptimizationLevel.O1
    
    def analyze(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze to find unreachable code."""
        unreachable = []
        for func in ir_data.get('ir_functions', []):
            body = func.get('body', [])
            found_term = False
            for i, stmt in enumerate(body):
                if found_term:
                    unreachable.append((func.get('name'), i))
                if self._is_terminating(stmt):
                    found_term = True
        return {'unreachable': unreachable}
    
    def transform(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove unreachable code."""
        for func in ir_data.get('ir_functions', []):
            body = func.get('body', [])
            new_body = []
            reachable = True
            
            for stmt in body:
                if not reachable:
                    self.stats.statements_removed += 1
                    self.stats.ir_changes += 1
                    continue
                
                # Process nested blocks
                stmt = self._process_nested(stmt)
                new_body.append(stmt)
                
                if self._is_terminating(stmt):
                    reachable = False
            
            func['body'] = new_body
        
        return ir_data
    
    def _process_nested(self, stmt: Any) -> Any:
        """Process nested blocks for dead code."""
        if not isinstance(stmt, dict):
            return stmt
        
        stmt_type = stmt.get('type', '')
        
        if stmt_type == 'if':
            stmt['then'] = self._process_block(stmt.get('then', []))
            if 'else' in stmt:
                stmt['else'] = self._process_block(stmt.get('else', []))
        elif stmt_type in ('while', 'for', 'block'):
            stmt['body'] = self._process_block(stmt.get('body', []))
        
        return stmt
    
    def _process_block(self, block: List) -> List:
        """Process a block for dead code."""
        new_block = []
        reachable = True
        
        for stmt in block:
            if not reachable:
                self.stats.statements_removed += 1
                self.stats.ir_changes += 1
                continue
            
            stmt = self._process_nested(stmt)
            new_block.append(stmt)
            
            if self._is_terminating(stmt):
                reachable = False
        
        return new_block
    
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
    
    def transform(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fold constant expressions."""
        for func in ir_data.get('ir_functions', []):
            func['body'] = [self._fold_stmt(stmt) for stmt in func.get('body', [])]
        return ir_data
    
    def _fold_stmt(self, stmt: Any) -> Any:
        """Fold constants in a statement."""
        if not isinstance(stmt, dict):
            return stmt
        
        stmt_type = stmt.get('type', '')
        
        if stmt_type in ('var_decl', 'let'):
            for key in ('init', 'value'):
                if key in stmt:
                    stmt[key] = self._fold_expr(stmt[key])
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
            
            left_val = self._get_const(left)
            right_val = self._get_const(right)
            
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
            
            val = self._get_const(operand)
            if val is not None:
                result = self._eval_unary(op, val)
                if result is not None:
                    self.stats.constants_folded += 1
                    self.stats.ir_changes += 1
                    return {'type': 'literal', 'value': result}
            
            return {'type': 'unary', 'op': op, 'operand': operand}
        
        return expr
    
    def _get_const(self, expr: Any) -> Optional[Any]:
        """Get constant value from expression."""
        if isinstance(expr, (int, float, bool)):
            return expr
        if isinstance(expr, dict) and expr.get('type') == 'literal':
            return expr.get('value')
        return None
    
    def _eval_binary(self, op: str, left: Any, right: Any) -> Optional[Any]:
        """Evaluate binary operation."""
        try:
            ops = {
                '+': lambda a, b: a + b,
                '-': lambda a, b: a - b,
                '*': lambda a, b: a * b,
                '/': lambda a, b: a // b if isinstance(a, int) and isinstance(b, int) and b != 0 else (a / b if b != 0 else None),
                '%': lambda a, b: a % b if b != 0 else None,
                '&': lambda a, b: a & b,
                '|': lambda a, b: a | b,
                '^': lambda a, b: a ^ b,
                '<<': lambda a, b: a << b,
                '>>': lambda a, b: a >> b,
                '==': lambda a, b: a == b,
                '!=': lambda a, b: a != b,
                '<': lambda a, b: a < b,
                '>': lambda a, b: a > b,
                '<=': lambda a, b: a <= b,
                '>=': lambda a, b: a >= b,
                '&&': lambda a, b: a and b,
                '||': lambda a, b: a or b,
                'and': lambda a, b: a and b,
                'or': lambda a, b: a or b,
            }
            if op in ops:
                return ops[op](left, right)
        except (TypeError, OverflowError, ZeroDivisionError):
            pass
        return None
    
    def _eval_unary(self, op: str, val: Any) -> Optional[Any]:
        """Evaluate unary operation."""
        try:
            if op == '-': return -val
            if op == '+': return +val
            if op in ('!', 'not'): return not val
            if op == '~': return ~val
        except (TypeError, OverflowError):
            pass
        return None


class ConstantPropagationPass(OptimizationPass):
    """Propagates constants to their usage sites."""
    
    @property
    def name(self) -> str:
        return "constant_propagation"
    
    @property
    def min_level(self) -> OptimizationLevel:
        return OptimizationLevel.O1
    
    def analyze(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find constant assignments."""
        constants = {}
        for func in ir_data.get('ir_functions', []):
            func_consts = {}
            for stmt in func.get('body', []):
                self._find_constants(stmt, func_consts)
            constants[func.get('name', '')] = func_consts
        return {'constants': constants}
    
    def _find_constants(self, stmt: Any, consts: Dict[str, Any]) -> None:
        """Find constant assignments in statement."""
        if not isinstance(stmt, dict):
            return
        
        stmt_type = stmt.get('type', '')
        
        if stmt_type in ('var_decl', 'let'):
            var_name = stmt.get('var_name', stmt.get('name', ''))
            init = stmt.get('init', stmt.get('value'))
            if init is not None:
                val = self._get_const(init)
                if val is not None:
                    consts[var_name] = val
        elif stmt_type == 'assign':
            target = stmt.get('target', '')
            val = self._get_const(stmt.get('value'))
            if val is not None:
                consts[target] = val
            elif target in consts:
                # Reassigned to non-constant, remove
                del consts[target]
    
    def _get_const(self, expr: Any) -> Optional[Any]:
        if isinstance(expr, (int, float, bool, str)):
            return expr
        if isinstance(expr, dict) and expr.get('type') == 'literal':
            return expr.get('value')
        return None
    
    def transform(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Replace variables with known constants."""
        for func in ir_data.get('ir_functions', []):
            consts = {}
            new_body = []
            
            for stmt in func.get('body', []):
                # Update constants
                self._find_constants(stmt, consts)
                # Propagate
                new_stmt = self._propagate(stmt, consts)
                new_body.append(new_stmt)
            
            func['body'] = new_body
        
        return ir_data
    
    def _propagate(self, stmt: Any, consts: Dict[str, Any]) -> Any:
        """Propagate constants in statement."""
        if not isinstance(stmt, dict):
            return stmt
        
        stmt = copy.copy(stmt)
        stmt_type = stmt.get('type', '')
        
        if stmt_type in ('var_decl', 'let'):
            for key in ('init', 'value'):
                if key in stmt:
                    stmt[key] = self._propagate_expr(stmt[key], consts)
        elif stmt_type == 'assign':
            stmt['value'] = self._propagate_expr(stmt.get('value'), consts)
        elif stmt_type == 'return':
            if 'value' in stmt:
                stmt['value'] = self._propagate_expr(stmt['value'], consts)
        elif stmt_type == 'call':
            stmt['args'] = [self._propagate_expr(a, consts) for a in stmt.get('args', [])]
        
        return stmt
    
    def _propagate_expr(self, expr: Any, consts: Dict[str, Any]) -> Any:
        """Propagate constants in expression."""
        if not isinstance(expr, dict):
            return expr
        
        expr_type = expr.get('type', '')
        
        if expr_type == 'var':
            var_name = expr.get('name', '')
            if var_name in consts:
                self.stats.ir_changes += 1
                return {'type': 'literal', 'value': consts[var_name]}
        elif expr_type == 'binary':
            return {
                'type': 'binary',
                'op': expr.get('op'),
                'left': self._propagate_expr(expr.get('left'), consts),
                'right': self._propagate_expr(expr.get('right'), consts)
            }
        elif expr_type == 'unary':
            return {
                'type': 'unary',
                'op': expr.get('op'),
                'operand': self._propagate_expr(expr.get('operand'), consts)
            }
        
        return expr


class AlgebraicSimplificationPass(OptimizationPass):
    """Performs simple algebraic simplifications.
    
    - x * 1 = x
    - x * 0 = 0
    - x + 0 = x
    - x - 0 = x
    - x / 1 = x
    - x && true = x
    - x || false = x
    """
    
    @property
    def name(self) -> str:
        return "algebraic_simplification"
    
    @property
    def min_level(self) -> OptimizationLevel:
        return OptimizationLevel.O1
    
    def transform(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        for func in ir_data.get('ir_functions', []):
            func['body'] = [self._simplify_stmt(stmt) for stmt in func.get('body', [])]
        return ir_data
    
    def _simplify_stmt(self, stmt: Any) -> Any:
        """Simplify expressions in statement."""
        if not isinstance(stmt, dict):
            return stmt
        
        for key in ('init', 'value', 'cond'):
            if key in stmt:
                stmt[key] = self._simplify_expr(stmt[key])
        
        for key in ('then', 'else', 'body', 'args'):
            if key in stmt and isinstance(stmt[key], list):
                stmt[key] = [self._simplify_stmt(s) if key in ('then', 'else', 'body') 
                             else self._simplify_expr(s) for s in stmt[key]]
        
        return stmt
    
    def _simplify_expr(self, expr: Any) -> Any:
        """Apply algebraic simplifications."""
        if not isinstance(expr, dict):
            return expr
        
        if expr.get('type') != 'binary':
            return expr
        
        op = expr.get('op', '')
        left = self._simplify_expr(expr.get('left'))
        right = self._simplify_expr(expr.get('right'))
        
        left_val = self._get_const(left)
        right_val = self._get_const(right)
        
        # x + 0 = x, 0 + x = x
        if op == '+':
            if right_val == 0: return self._simplified(left)
            if left_val == 0: return self._simplified(right)
        
        # x - 0 = x
        if op == '-':
            if right_val == 0: return self._simplified(left)
        
        # x * 1 = x, 1 * x = x
        if op == '*':
            if right_val == 1: return self._simplified(left)
            if left_val == 1: return self._simplified(right)
            if right_val == 0 or left_val == 0:
                self.stats.ir_changes += 1
                return {'type': 'literal', 'value': 0}
        
        # x / 1 = x
        if op == '/':
            if right_val == 1: return self._simplified(left)
        
        # x && true = x, x || false = x
        if op in ('&&', 'and'):
            if right_val is True: return self._simplified(left)
            if left_val is True: return self._simplified(right)
        
        if op in ('||', 'or'):
            if right_val is False: return self._simplified(left)
            if left_val is False: return self._simplified(right)
        
        return {'type': 'binary', 'op': op, 'left': left, 'right': right}
    
    def _get_const(self, expr: Any) -> Optional[Any]:
        if isinstance(expr, (int, float, bool)):
            return expr
        if isinstance(expr, dict) and expr.get('type') == 'literal':
            return expr.get('value')
        return None
    
    def _simplified(self, expr: Any) -> Any:
        self.stats.ir_changes += 1
        return expr


def get_o1_passes() -> List[OptimizationPass]:
    """Get all O1 optimization passes."""
    return [
        DeadCodeEliminationPass(),
        ConstantFoldingPass(),
        ConstantPropagationPass(),
        AlgebraicSimplificationPass(),
    ]


__all__ = [
    'DeadCodeEliminationPass',
    'ConstantFoldingPass', 
    'ConstantPropagationPass',
    'AlgebraicSimplificationPass',
    'get_o1_passes'
]
