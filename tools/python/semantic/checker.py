#!/usr/bin/env python3
"""STUNIR Semantic Checker Module.

Provides semantic checks including type compatibility, dead code detection,
unreachable code detection, and constant expression evaluation.

This module is part of the STUNIR code generation enhancement suite.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum, auto

from .analyzer import SemanticAnalyzer, SemanticIssue, WarningSeverity, AnalysisErrorKind


class CheckKind(Enum):
    """Kinds of semantic checks."""
    TYPE_COMPATIBILITY = auto()
    NULL_SAFETY = auto()
    BOUNDS_CHECK = auto()
    OVERFLOW_CHECK = auto()
    DEAD_CODE = auto()
    UNREACHABLE_CODE = auto()
    RESOURCE_LEAK = auto()
    CONSTANT_EXPRESSION = auto()
    PURITY = auto()
    SIDE_EFFECTS = auto()


@dataclass
class CheckResult:
    """Result of a semantic check."""
    kind: CheckKind
    passed: bool
    message: str
    location: Optional[str] = None
    severity: WarningSeverity = WarningSeverity.INFO
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_issue(self) -> Optional[SemanticIssue]:
        """Convert to SemanticIssue if check failed."""
        if self.passed:
            return None
        
        kind_map = {
            CheckKind.TYPE_COMPATIBILITY: AnalysisErrorKind.UNDEFINED_TYPE,
            CheckKind.DEAD_CODE: AnalysisErrorKind.DEAD_CODE,
            CheckKind.UNREACHABLE_CODE: AnalysisErrorKind.UNREACHABLE_CODE,
        }
        
        return SemanticIssue(
            kind=kind_map.get(self.kind, AnalysisErrorKind.DEAD_CODE),
            message=self.message,
            severity=self.severity,
            location=self.location
        )


class DeadCodeDetector:
    """Detects dead code in IR."""
    
    def __init__(self):
        self.dead_statements: List[Any] = []
        self.used_variables: Set[str] = set()
        self.used_functions: Set[str] = set()
        self.assigned_variables: Dict[str, int] = {}  # var -> assignment count
    
    def analyze(self, ir_data: Dict) -> List[CheckResult]:
        """Analyze IR for dead code."""
        results = []
        
        # First pass: collect usage
        for func in ir_data.get('ir_functions', []):
            self._collect_usage(func)
        
        # Second pass: detect dead code
        for func in ir_data.get('ir_functions', []):
            results.extend(self._detect_dead_code(func))
        
        # Check for unused functions
        all_functions = {f.get('name') for f in ir_data.get('ir_functions', [])}
        for func_name in all_functions:
            if func_name not in self.used_functions and func_name != 'main':
                results.append(CheckResult(
                    kind=CheckKind.DEAD_CODE,
                    passed=False,
                    message=f"Function '{func_name}' is defined but never called",
                    severity=WarningSeverity.WARNING
                ))
        
        # Check for variables assigned but never read
        for var_name, count in self.assigned_variables.items():
            if var_name not in self.used_variables and count > 0:
                results.append(CheckResult(
                    kind=CheckKind.DEAD_CODE,
                    passed=False,
                    message=f"Variable '{var_name}' is assigned but never used",
                    severity=WarningSeverity.WARNING
                ))
        
        return results
    
    def _collect_usage(self, func: Dict) -> None:
        """Collect variable and function usage."""
        body = func.get('body', [])
        for stmt in body:
            self._collect_usage_stmt(stmt)
    
    def _collect_usage_stmt(self, stmt: Any) -> None:
        """Collect usage from a statement."""
        if not isinstance(stmt, dict):
            if isinstance(stmt, str):
                self.used_variables.add(stmt)
            return
        
        stmt_type = stmt.get('type', '')
        
        if stmt_type == 'call':
            self.used_functions.add(stmt.get('func', ''))
            for arg in stmt.get('args', []):
                self._collect_usage_stmt(arg)
        
        elif stmt_type == 'var':
            self.used_variables.add(stmt.get('name', ''))
        
        elif stmt_type in ('var_decl', 'let'):
            var_name = stmt.get('var_name', stmt.get('name', ''))
            self.assigned_variables[var_name] = self.assigned_variables.get(var_name, 0) + 1
            init = stmt.get('init', stmt.get('value'))
            if init:
                self._collect_usage_stmt(init)
        
        elif stmt_type == 'assign':
            target = stmt.get('target', '')
            self.assigned_variables[target] = self.assigned_variables.get(target, 0) + 1
            self._collect_usage_stmt(stmt.get('value'))
        
        elif stmt_type == 'binary':
            self._collect_usage_stmt(stmt.get('left'))
            self._collect_usage_stmt(stmt.get('right'))
        
        elif stmt_type == 'unary':
            self._collect_usage_stmt(stmt.get('operand'))
        
        elif stmt_type in ('if', 'while', 'for'):
            self._collect_usage_stmt(stmt.get('cond'))
            for s in stmt.get('then', []):
                self._collect_usage_stmt(s)
            for s in stmt.get('else', []):
                self._collect_usage_stmt(s)
            for s in stmt.get('body', []):
                self._collect_usage_stmt(s)
        
        elif stmt_type == 'return':
            self._collect_usage_stmt(stmt.get('value'))
        
        elif 'value' in stmt:
            self._collect_usage_stmt(stmt['value'])
    
    def _detect_dead_code(self, func: Dict) -> List[CheckResult]:
        """Detect dead code in a function."""
        results = []
        body = func.get('body', [])
        
        reachable = True
        for i, stmt in enumerate(body):
            if not reachable:
                results.append(CheckResult(
                    kind=CheckKind.UNREACHABLE_CODE,
                    passed=False,
                    message=f"Unreachable statement in function '{func.get('name', 'unknown')}'",
                    severity=WarningSeverity.WARNING,
                    details={'statement_index': i}
                ))
                self.dead_statements.append(stmt)
            
            if self._is_terminating(stmt):
                reachable = False
        
        return results
    
    def _is_terminating(self, stmt: Any) -> bool:
        """Check if statement terminates control flow."""
        if not isinstance(stmt, dict):
            return False
        
        stmt_type = stmt.get('type', '')
        return stmt_type in ('return', 'break', 'continue', 'goto', 'throw')


class UnreachableCodeDetector:
    """Detects unreachable code based on control flow analysis."""
    
    def __init__(self):
        self.reachable_blocks: Set[int] = set()
    
    def analyze(self, ir_data: Dict) -> List[CheckResult]:
        """Analyze IR for unreachable code."""
        results = []
        
        for func in ir_data.get('ir_functions', []):
            func_results = self._analyze_function(func)
            results.extend(func_results)
        
        return results
    
    def _analyze_function(self, func: Dict) -> List[CheckResult]:
        """Analyze a function for unreachable code."""
        results = []
        body = func.get('body', [])
        
        # Track condition states for if-else branches
        for i, stmt in enumerate(body):
            results.extend(self._analyze_statement(stmt, func.get('name', '')))
        
        return results
    
    def _analyze_statement(self, stmt: Any, func_name: str) -> List[CheckResult]:
        """Analyze a statement for unreachable paths."""
        results = []
        
        if not isinstance(stmt, dict):
            return results
        
        stmt_type = stmt.get('type', '')
        
        if stmt_type == 'if':
            cond = stmt.get('cond')
            
            # Check for constant conditions
            const_val = self._evaluate_constant(cond)
            if const_val is not None:
                if const_val:
                    # Else branch is unreachable
                    if 'else' in stmt and stmt['else']:
                        results.append(CheckResult(
                            kind=CheckKind.UNREACHABLE_CODE,
                            passed=False,
                            message=f"Else branch is unreachable (condition always true) in '{func_name}'",
                            severity=WarningSeverity.WARNING
                        ))
                else:
                    # Then branch is unreachable
                    if stmt.get('then'):
                        results.append(CheckResult(
                            kind=CheckKind.UNREACHABLE_CODE,
                            passed=False,
                            message=f"Then branch is unreachable (condition always false) in '{func_name}'",
                            severity=WarningSeverity.WARNING
                        ))
            
            # Recursively check branches
            for s in stmt.get('then', []):
                results.extend(self._analyze_statement(s, func_name))
            for s in stmt.get('else', []):
                results.extend(self._analyze_statement(s, func_name))
        
        elif stmt_type in ('while', 'for'):
            cond = stmt.get('cond')
            
            # Check for constant false condition
            const_val = self._evaluate_constant(cond)
            if const_val is False:
                results.append(CheckResult(
                    kind=CheckKind.UNREACHABLE_CODE,
                    passed=False,
                    message=f"Loop body is unreachable (condition always false) in '{func_name}'",
                    severity=WarningSeverity.WARNING
                ))
            
            # Recursively check body
            for s in stmt.get('body', []):
                results.extend(self._analyze_statement(s, func_name))
        
        return results
    
    def _evaluate_constant(self, expr: Any) -> Optional[bool]:
        """Try to evaluate an expression as a constant boolean."""
        if expr is None:
            return None
        
        if isinstance(expr, bool):
            return expr
        
        if isinstance(expr, (int, float)):
            return bool(expr)
        
        if isinstance(expr, dict):
            expr_type = expr.get('type', '')
            
            if expr_type == 'literal':
                value = expr.get('value')
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return bool(value)
            
            elif expr_type == 'binary':
                op = expr.get('op', '')
                left_val = self._evaluate_constant(expr.get('left'))
                right_val = self._evaluate_constant(expr.get('right'))
                
                if left_val is not None and right_val is not None:
                    try:
                        if op == '==':
                            return left_val == right_val
                        elif op == '!=':
                            return left_val != right_val
                        elif op == '&&' or op == 'and':
                            return left_val and right_val
                        elif op == '||' or op == 'or':
                            return left_val or right_val
                        elif op == '<':
                            return left_val < right_val
                        elif op == '>':
                            return left_val > right_val
                        elif op == '<=':
                            return left_val <= right_val
                        elif op == '>=':
                            return left_val >= right_val
                    except TypeError:
                        pass
            
            elif expr_type == 'unary':
                op = expr.get('op', '')
                operand_val = self._evaluate_constant(expr.get('operand'))
                
                if operand_val is not None:
                    if op in ('!', 'not'):
                        return not operand_val
        
        return None


class ConstantExpressionEvaluator:
    """Evaluates constant expressions at compile time."""
    
    def __init__(self):
        self.constants: Dict[str, Any] = {}
    
    def evaluate(self, expr: Any) -> Tuple[bool, Any]:
        """Evaluate an expression if possible.
        
        Returns (is_constant, value).
        """
        if expr is None:
            return True, None
        
        if isinstance(expr, (bool, int, float, str)):
            return True, expr
        
        if isinstance(expr, dict):
            return self._evaluate_dict(expr)
        
        return False, None
    
    def _evaluate_dict(self, expr: Dict) -> Tuple[bool, Any]:
        """Evaluate a dictionary expression."""
        expr_type = expr.get('type', '')
        
        if expr_type == 'literal':
            return True, expr.get('value')
        
        if expr_type == 'var':
            name = expr.get('name', '')
            if name in self.constants:
                return True, self.constants[name]
            return False, None
        
        if expr_type == 'binary':
            return self._evaluate_binary(expr)
        
        if expr_type == 'unary':
            return self._evaluate_unary(expr)
        
        if expr_type == 'ternary':
            return self._evaluate_ternary(expr)
        
        if expr_type == 'cast':
            return self._evaluate_cast(expr)
        
        return False, None
    
    def _evaluate_binary(self, expr: Dict) -> Tuple[bool, Any]:
        """Evaluate a binary expression."""
        op = expr.get('op', '')
        
        left_const, left_val = self.evaluate(expr.get('left'))
        right_const, right_val = self.evaluate(expr.get('right'))
        
        if not (left_const and right_const):
            return False, None
        
        try:
            if op == '+':
                return True, left_val + right_val
            elif op == '-':
                return True, left_val - right_val
            elif op == '*':
                return True, left_val * right_val
            elif op == '/':
                if right_val == 0:
                    return False, None
                if isinstance(left_val, int) and isinstance(right_val, int):
                    return True, left_val // right_val
                return True, left_val / right_val
            elif op == '%':
                if right_val == 0:
                    return False, None
                return True, left_val % right_val
            elif op == '**':
                return True, left_val ** right_val
            elif op == '&':
                return True, left_val & right_val
            elif op == '|':
                return True, left_val | right_val
            elif op == '^':
                return True, left_val ^ right_val
            elif op == '<<':
                return True, left_val << right_val
            elif op == '>>':
                return True, left_val >> right_val
            elif op == '==':
                return True, left_val == right_val
            elif op == '!=':
                return True, left_val != right_val
            elif op == '<':
                return True, left_val < right_val
            elif op == '>':
                return True, left_val > right_val
            elif op == '<=':
                return True, left_val <= right_val
            elif op == '>=':
                return True, left_val >= right_val
            elif op in ('&&', 'and'):
                return True, left_val and right_val
            elif op in ('||', 'or'):
                return True, left_val or right_val
        except (TypeError, ZeroDivisionError, OverflowError):
            pass
        
        return False, None
    
    def _evaluate_unary(self, expr: Dict) -> Tuple[bool, Any]:
        """Evaluate a unary expression."""
        op = expr.get('op', '')
        
        operand_const, operand_val = self.evaluate(expr.get('operand'))
        
        if not operand_const:
            return False, None
        
        try:
            if op == '-':
                return True, -operand_val
            elif op == '+':
                return True, +operand_val
            elif op in ('!', 'not'):
                return True, not operand_val
            elif op == '~':
                return True, ~operand_val
        except TypeError:
            pass
        
        return False, None
    
    def _evaluate_ternary(self, expr: Dict) -> Tuple[bool, Any]:
        """Evaluate a ternary expression."""
        cond_const, cond_val = self.evaluate(expr.get('cond'))
        
        if not cond_const:
            return False, None
        
        if cond_val:
            return self.evaluate(expr.get('then'))
        else:
            return self.evaluate(expr.get('else'))
    
    def _evaluate_cast(self, expr: Dict) -> Tuple[bool, Any]:
        """Evaluate a cast expression."""
        inner_const, inner_val = self.evaluate(expr.get('value'))
        
        if not inner_const:
            return False, None
        
        target = expr.get('target_type', 'i32')
        
        try:
            if target in ('i8', 'i16', 'i32', 'i64', 'int', 'long'):
                return True, int(inner_val)
            elif target in ('u8', 'u16', 'u32', 'u64'):
                return True, int(inner_val) & ((1 << int(target[1:])) - 1)
            elif target in ('f32', 'f64', 'float', 'double'):
                return True, float(inner_val)
            elif target == 'bool':
                return True, bool(inner_val)
            elif target in ('string', 'str'):
                return True, str(inner_val)
        except (TypeError, ValueError):
            pass
        
        return False, None
    
    def register_constant(self, name: str, value: Any) -> None:
        """Register a compile-time constant."""
        self.constants[name] = value


class SemanticChecker:
    """Comprehensive semantic checker combining all checks."""
    
    def __init__(self, strict: bool = False):
        self.analyzer = SemanticAnalyzer(strict=strict)
        self.dead_code_detector = DeadCodeDetector()
        self.unreachable_detector = UnreachableCodeDetector()
        self.const_evaluator = ConstantExpressionEvaluator()
        self.results: List[CheckResult] = []
    
    def check(self, ir_data: Dict) -> List[SemanticIssue]:
        """Run all semantic checks on IR data."""
        issues = []
        
        # Run semantic analysis
        analysis_issues = self.analyzer.analyze(ir_data)
        issues.extend(analysis_issues)
        
        # Run dead code detection
        dead_code_results = self.dead_code_detector.analyze(ir_data)
        for result in dead_code_results:
            issue = result.to_issue()
            if issue:
                issues.append(issue)
        self.results.extend(dead_code_results)
        
        # Run unreachable code detection
        unreachable_results = self.unreachable_detector.analyze(ir_data)
        for result in unreachable_results:
            issue = result.to_issue()
            if issue:
                issues.append(issue)
        self.results.extend(unreachable_results)
        
        return issues
    
    def evaluate_constant(self, expr: Any) -> Tuple[bool, Any]:
        """Evaluate an expression as a constant."""
        return self.const_evaluator.evaluate(expr)
    
    def get_all_results(self) -> List[CheckResult]:
        """Get all check results."""
        return self.results
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return self.analyzer.has_errors()
    
    def get_summary(self) -> Dict[str, int]:
        """Get summary of issues by severity."""
        summary = {
            'fatal': 0,
            'error': 0,
            'warning': 0,
            'info': 0
        }
        
        for issue in self.analyzer.issues:
            if issue.severity == WarningSeverity.FATAL:
                summary['fatal'] += 1
            elif issue.severity == WarningSeverity.ERROR:
                summary['error'] += 1
            elif issue.severity == WarningSeverity.WARNING:
                summary['warning'] += 1
            else:
                summary['info'] += 1
        
        return summary


__all__ = [
    'CheckKind', 'CheckResult',
    'DeadCodeDetector', 'UnreachableCodeDetector', 'ConstantExpressionEvaluator',
    'SemanticChecker'
]
