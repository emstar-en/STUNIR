#!/usr/bin/env python3
"""STUNIR Semantic Analyzer Module.

Provides comprehensive semantic analysis including variable tracking,
function analysis, dead code detection, and unreachable code detection.

This module is part of the STUNIR code generation enhancement suite.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum, auto
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from types.type_system import STUNIRType, TypeRegistry, parse_type
    from types.type_inference import TypeScope, TypeInferenceEngine
except ImportError:
    # Fallback imports
    STUNIRType = Any
    TypeRegistry = None
    parse_type = lambda x, y=None: x
    TypeScope = None
    TypeInferenceEngine = None


class AnalysisErrorKind(Enum):
    """Kinds of semantic analysis errors."""
    UNDEFINED_VARIABLE = auto()
    UNDEFINED_FUNCTION = auto()
    UNDEFINED_TYPE = auto()
    REDEFINITION = auto()
    DEAD_CODE = auto()
    UNREACHABLE_CODE = auto()
    UNINITIALIZED_VARIABLE = auto()
    UNUSED_VARIABLE = auto()
    UNUSED_FUNCTION = auto()
    UNUSED_PARAMETER = auto()
    BREAK_OUTSIDE_LOOP = auto()
    CONTINUE_OUTSIDE_LOOP = auto()
    RETURN_OUTSIDE_FUNCTION = auto()
    INVALID_LVALUE = auto()
    CONSTANT_MODIFICATION = auto()
    SHADOWING = auto()


class WarningSeverity(Enum):
    """Severity levels for warnings."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    FATAL = auto()


@dataclass
class SemanticIssue:
    """Represents a semantic analysis issue."""
    kind: AnalysisErrorKind
    message: str
    severity: WarningSeverity = WarningSeverity.ERROR
    location: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        sev = self.severity.name.lower()
        loc = f"{self.location}:" if self.location else ""
        line = f"{self.line}:" if self.line else ""
        col = f"{self.column}:" if self.column else ""
        msg = f"[{sev}] {loc}{line}{col} {self.message}"
        if self.suggestion:
            msg += f"\n  suggestion: {self.suggestion}"
        return msg


@dataclass
class VariableInfo:
    """Information about a variable."""
    name: str
    type: Any  # STUNIRType
    is_initialized: bool = False
    is_used: bool = False
    is_mutable: bool = True
    is_parameter: bool = False
    definition_location: Optional[str] = None
    definition_line: Optional[int] = None


@dataclass
class FunctionInfo:
    """Information about a function."""
    name: str
    params: List[VariableInfo] = field(default_factory=list)
    return_type: Any = None
    is_used: bool = False
    is_recursive: bool = False
    called_functions: Set[str] = field(default_factory=set)
    definition_location: Optional[str] = None


class SymbolTable:
    """Symbol table for tracking definitions."""
    
    def __init__(self, parent: Optional[SymbolTable] = None, name: str = "global"):
        self.parent = parent
        self.name = name
        self.variables: Dict[str, VariableInfo] = {}
        self.functions: Dict[str, FunctionInfo] = {}
        self.types: Dict[str, Any] = {}
        self.labels: Set[str] = set()
        self.children: List[SymbolTable] = []
        
        if parent:
            parent.children.append(self)
    
    def define_variable(self, var: VariableInfo) -> Optional[VariableInfo]:
        """Define a variable. Returns existing if redefinition."""
        existing = self.variables.get(var.name)
        self.variables[var.name] = var
        return existing
    
    def lookup_variable(self, name: str) -> Optional[VariableInfo]:
        """Look up a variable in this scope and parents."""
        if name in self.variables:
            return self.variables[name]
        if self.parent:
            return self.parent.lookup_variable(name)
        return None
    
    def lookup_variable_local(self, name: str) -> Optional[VariableInfo]:
        """Look up a variable only in this scope."""
        return self.variables.get(name)
    
    def define_function(self, func: FunctionInfo) -> Optional[FunctionInfo]:
        """Define a function. Returns existing if redefinition."""
        existing = self.functions.get(func.name)
        self.functions[func.name] = func
        return existing
    
    def lookup_function(self, name: str) -> Optional[FunctionInfo]:
        """Look up a function."""
        if name in self.functions:
            return self.functions[name]
        if self.parent:
            return self.parent.lookup_function(name)
        return None
    
    def define_type(self, name: str, typ: Any) -> None:
        """Define a type."""
        self.types[name] = typ
    
    def lookup_type(self, name: str) -> Optional[Any]:
        """Look up a type."""
        if name in self.types:
            return self.types[name]
        if self.parent:
            return self.parent.lookup_type(name)
        return None
    
    def add_label(self, label: str) -> bool:
        """Add a label. Returns False if already exists."""
        if label in self.labels:
            return False
        self.labels.add(label)
        return True
    
    def has_label(self, label: str) -> bool:
        """Check if label exists."""
        if label in self.labels:
            return True
        if self.parent:
            return self.parent.has_label(label)
        return False
    
    def child_scope(self, name: str = "block") -> SymbolTable:
        """Create a child scope."""
        return SymbolTable(parent=self, name=name)
    
    def all_variables(self) -> Dict[str, VariableInfo]:
        """Get all variables including from parent scopes."""
        result = {}
        if self.parent:
            result.update(self.parent.all_variables())
        result.update(self.variables)
        return result
    
    def get_unused_variables(self) -> List[VariableInfo]:
        """Get list of unused variables in this scope."""
        return [v for v in self.variables.values() if not v.is_used]
    
    def get_uninitialized_variables(self) -> List[VariableInfo]:
        """Get list of uninitialized variables in this scope."""
        return [v for v in self.variables.values() 
                if not v.is_initialized and v.is_used]


class SemanticAnalyzer:
    """Performs semantic analysis on STUNIR IR."""
    
    def __init__(self, strict: bool = False):
        self.strict = strict
        self.issues: List[SemanticIssue] = []
        self.global_scope = SymbolTable(name="global")
        self.current_scope: SymbolTable = self.global_scope
        self.current_function: Optional[FunctionInfo] = None
        self.loop_depth = 0
        self.reachable = True
        self._init_builtins()
    
    def _init_builtins(self) -> None:
        """Initialize built-in functions and types."""
        # Common built-in functions
        builtins = [
            ('print', ['any'], 'void'),
            ('println', ['any'], 'void'),
            ('printf', ['str'], 'i32'),
            ('malloc', ['u64'], '*void'),
            ('free', ['*void'], 'void'),
            ('memcpy', ['*void', '*void', 'u64'], '*void'),
            ('memset', ['*void', 'i32', 'u64'], '*void'),
            ('strlen', ['*char'], 'u64'),
            ('strcmp', ['*char', '*char'], 'i32'),
        ]
        
        for name, params, ret in builtins:
            func_info = FunctionInfo(
                name=name,
                params=[VariableInfo(name=f'arg{i}', type=p, is_parameter=True) 
                       for i, p in enumerate(params)],
                return_type=ret,
                is_used=True  # Built-ins don't need to be marked unused
            )
            self.global_scope.define_function(func_info)
    
    def analyze(self, ir_data: Dict) -> List[SemanticIssue]:
        """Analyze IR data and return issues."""
        self.issues.clear()
        
        # First pass: collect all type and function definitions
        self._collect_definitions(ir_data)
        
        # Second pass: analyze function bodies
        for func in ir_data.get('ir_functions', []):
            self._analyze_function(func)
        
        # Check for unused definitions
        self._check_unused()
        
        return self.issues
    
    def _collect_definitions(self, ir_data: Dict) -> None:
        """Collect all type and function definitions."""
        # Types
        for type_def in ir_data.get('ir_types', []):
            name = type_def.get('name', '')
            if name:
                self.global_scope.define_type(name, type_def)
        
        # Functions (just signatures first)
        for func in ir_data.get('ir_functions', []):
            name = func.get('name', '')
            params = func.get('params', [])
            returns = func.get('returns', 'void')
            
            param_infos = []
            for p in params:
                if isinstance(p, dict):
                    param_info = VariableInfo(
                        name=p.get('name', ''),
                        type=p.get('type', 'i32'),
                        is_parameter=True,
                        is_initialized=True
                    )
                else:
                    param_info = VariableInfo(
                        name=str(p),
                        type='i32',
                        is_parameter=True,
                        is_initialized=True
                    )
                param_infos.append(param_info)
            
            func_info = FunctionInfo(
                name=name,
                params=param_infos,
                return_type=returns
            )
            
            existing = self.global_scope.define_function(func_info)
            if existing and self.strict:
                self.issues.append(SemanticIssue(
                    kind=AnalysisErrorKind.REDEFINITION,
                    message=f"Function '{name}' is redefined",
                    severity=WarningSeverity.WARNING
                ))
    
    def _analyze_function(self, func: Dict) -> None:
        """Analyze a function definition."""
        name = func.get('name', '')
        params = func.get('params', [])
        body = func.get('body', [])
        
        # Get function info
        func_info = self.global_scope.lookup_function(name)
        self.current_function = func_info
        
        # Create function scope
        func_scope = self.current_scope.child_scope(f"function:{name}")
        self.current_scope = func_scope
        self.reachable = True
        
        # Add parameters to scope
        for p in params:
            if isinstance(p, dict):
                var_info = VariableInfo(
                    name=p.get('name', ''),
                    type=p.get('type', 'i32'),
                    is_parameter=True,
                    is_initialized=True,
                    is_mutable=p.get('mutable', True)
                )
            else:
                var_info = VariableInfo(
                    name=str(p),
                    type='i32',
                    is_parameter=True,
                    is_initialized=True
                )
            func_scope.define_variable(var_info)
        
        # Analyze body
        for stmt in body:
            self._analyze_statement(stmt)
        
        # Check for unused parameters
        for var in func_scope.variables.values():
            if var.is_parameter and not var.is_used:
                self.issues.append(SemanticIssue(
                    kind=AnalysisErrorKind.UNUSED_PARAMETER,
                    message=f"Parameter '{var.name}' is unused in function '{name}'",
                    severity=WarningSeverity.WARNING,
                    suggestion=f"Consider removing parameter '{var.name}' or prefixing with _"
                ))
        
        # Check for unused local variables
        for var in func_scope.get_unused_variables():
            if not var.is_parameter:
                self.issues.append(SemanticIssue(
                    kind=AnalysisErrorKind.UNUSED_VARIABLE,
                    message=f"Variable '{var.name}' is declared but never used",
                    severity=WarningSeverity.WARNING
                ))
        
        # Check for uninitialized variables that were used
        for var in func_scope.get_uninitialized_variables():
            self.issues.append(SemanticIssue(
                kind=AnalysisErrorKind.UNINITIALIZED_VARIABLE,
                message=f"Variable '{var.name}' may be used before initialization",
                severity=WarningSeverity.WARNING
            ))
        
        # Restore scope
        self.current_scope = self.global_scope
        self.current_function = None
    
    def _analyze_statement(self, stmt: Any) -> None:
        """Analyze a statement."""
        if not self.reachable:
            self.issues.append(SemanticIssue(
                kind=AnalysisErrorKind.UNREACHABLE_CODE,
                message="Unreachable code detected",
                severity=WarningSeverity.WARNING
            ))
            return
        
        if isinstance(stmt, dict):
            stmt_type = stmt.get('type', '')
            
            if stmt_type in ('var_decl', 'let'):
                self._analyze_var_decl(stmt)
            elif stmt_type == 'assign':
                self._analyze_assign(stmt)
            elif stmt_type == 'return':
                self._analyze_return(stmt)
                self.reachable = False
            elif stmt_type == 'if':
                self._analyze_if(stmt)
            elif stmt_type in ('while', 'for', 'do_while', 'loop'):
                self._analyze_loop(stmt)
            elif stmt_type == 'break':
                self._analyze_break(stmt)
            elif stmt_type == 'continue':
                self._analyze_continue(stmt)
            elif stmt_type == 'call':
                self._analyze_call(stmt)
            elif stmt_type == 'switch':
                self._analyze_switch(stmt)
            elif stmt_type == 'try':
                self._analyze_try(stmt)
            elif stmt_type == 'label':
                self._analyze_label(stmt)
            elif stmt_type == 'goto':
                self._analyze_goto(stmt)
            elif stmt_type == 'block':
                self._analyze_block(stmt)
            else:
                # Expression statement
                self._analyze_expression(stmt)
        elif isinstance(stmt, str):
            # Simple expression or variable reference
            self._analyze_expression({'type': 'expr', 'value': stmt})
    
    def _analyze_var_decl(self, stmt: Dict) -> None:
        """Analyze variable declaration."""
        name = stmt.get('var_name', stmt.get('name', ''))
        var_type = stmt.get('var_type', stmt.get('type', 'auto'))
        init = stmt.get('init', stmt.get('value'))
        mutable = stmt.get('mutable', True)
        
        # Check for shadowing
        existing = self.current_scope.lookup_variable_local(name)
        if existing:
            self.issues.append(SemanticIssue(
                kind=AnalysisErrorKind.REDEFINITION,
                message=f"Variable '{name}' is already defined in this scope",
                severity=WarningSeverity.ERROR
            ))
            return
        
        # Check for shadowing from parent scope
        parent_var = self.current_scope.parent.lookup_variable(name) if self.current_scope.parent else None
        if parent_var and self.strict:
            self.issues.append(SemanticIssue(
                kind=AnalysisErrorKind.SHADOWING,
                message=f"Variable '{name}' shadows a variable from outer scope",
                severity=WarningSeverity.WARNING
            ))
        
        # Create variable info
        var_info = VariableInfo(
            name=name,
            type=var_type,
            is_initialized=init is not None,
            is_mutable=mutable
        )
        
        # Analyze initializer
        if init is not None:
            self._analyze_expression(init)
        
        self.current_scope.define_variable(var_info)
    
    def _analyze_assign(self, stmt: Dict) -> None:
        """Analyze assignment statement."""
        target = stmt.get('target', '')
        value = stmt.get('value')
        
        # Check target is defined
        var_info = self.current_scope.lookup_variable(target)
        if var_info is None:
            self.issues.append(SemanticIssue(
                kind=AnalysisErrorKind.UNDEFINED_VARIABLE,
                message=f"Assignment to undefined variable '{target}'",
                severity=WarningSeverity.ERROR,
                suggestion=f"Declare '{target}' before use"
            ))
        elif not var_info.is_mutable:
            self.issues.append(SemanticIssue(
                kind=AnalysisErrorKind.CONSTANT_MODIFICATION,
                message=f"Cannot assign to immutable variable '{target}'",
                severity=WarningSeverity.ERROR
            ))
        else:
            var_info.is_initialized = True
        
        # Analyze value
        self._analyze_expression(value)
    
    def _analyze_return(self, stmt: Dict) -> None:
        """Analyze return statement."""
        if self.current_function is None:
            self.issues.append(SemanticIssue(
                kind=AnalysisErrorKind.RETURN_OUTSIDE_FUNCTION,
                message="Return statement outside of function",
                severity=WarningSeverity.ERROR
            ))
            return
        
        value = stmt.get('value')
        if value is not None:
            self._analyze_expression(value)
    
    def _analyze_if(self, stmt: Dict) -> None:
        """Analyze if statement."""
        # Analyze condition
        cond = stmt.get('cond')
        self._analyze_expression(cond)
        
        # Analyze then branch
        then_scope = self.current_scope.child_scope("then")
        self.current_scope = then_scope
        saved_reachable = self.reachable
        
        for s in stmt.get('then', []):
            self._analyze_statement(s)
        
        then_reachable = self.reachable
        self.current_scope = then_scope.parent
        self.reachable = saved_reachable
        
        # Analyze elif branches
        elif_reachable = True
        for elif_clause in stmt.get('elif', []):
            self._analyze_expression(elif_clause.get('cond'))
            elif_scope = self.current_scope.child_scope("elif")
            self.current_scope = elif_scope
            
            for s in elif_clause.get('then', []):
                self._analyze_statement(s)
            
            elif_reachable = elif_reachable and self.reachable
            self.current_scope = elif_scope.parent
            self.reachable = saved_reachable
        
        # Analyze else branch
        else_reachable = True
        if 'else' in stmt:
            else_scope = self.current_scope.child_scope("else")
            self.current_scope = else_scope
            
            for s in stmt.get('else', []):
                self._analyze_statement(s)
            
            else_reachable = self.reachable
            self.current_scope = else_scope.parent
            self.reachable = saved_reachable
        
        # Code after if is reachable if any branch is reachable
        self.reachable = then_reachable or elif_reachable or else_reachable
    
    def _analyze_loop(self, stmt: Dict) -> None:
        """Analyze loop statement."""
        self.loop_depth += 1
        
        # Analyze init (for loops)
        if 'init' in stmt:
            self._analyze_statement(stmt['init'])
        
        # Analyze condition
        if 'cond' in stmt:
            self._analyze_expression(stmt['cond'])
        
        # Analyze body
        loop_scope = self.current_scope.child_scope("loop")
        self.current_scope = loop_scope
        
        for s in stmt.get('body', []):
            self._analyze_statement(s)
        
        # Analyze update (for loops)
        if 'update' in stmt:
            self._analyze_statement(stmt['update'])
        
        self.current_scope = loop_scope.parent
        self.loop_depth -= 1
    
    def _analyze_break(self, stmt: Dict) -> None:
        """Analyze break statement."""
        if self.loop_depth == 0:
            self.issues.append(SemanticIssue(
                kind=AnalysisErrorKind.BREAK_OUTSIDE_LOOP,
                message="Break statement outside of loop",
                severity=WarningSeverity.ERROR
            ))
    
    def _analyze_continue(self, stmt: Dict) -> None:
        """Analyze continue statement."""
        if self.loop_depth == 0:
            self.issues.append(SemanticIssue(
                kind=AnalysisErrorKind.CONTINUE_OUTSIDE_LOOP,
                message="Continue statement outside of loop",
                severity=WarningSeverity.ERROR
            ))
    
    def _analyze_call(self, stmt: Dict) -> None:
        """Analyze function call."""
        func_name = stmt.get('func', '')
        args = stmt.get('args', [])
        
        # Mark function as used
        func_info = self.global_scope.lookup_function(func_name)
        if func_info is None:
            self.issues.append(SemanticIssue(
                kind=AnalysisErrorKind.UNDEFINED_FUNCTION,
                message=f"Call to undefined function '{func_name}'",
                severity=WarningSeverity.ERROR
            ))
        else:
            func_info.is_used = True
            
            # Track if current function calls this one (for recursion detection)
            if self.current_function:
                self.current_function.called_functions.add(func_name)
                if func_name == self.current_function.name:
                    self.current_function.is_recursive = True
        
        # Analyze arguments
        for arg in args:
            self._analyze_expression(arg)
    
    def _analyze_switch(self, stmt: Dict) -> None:
        """Analyze switch statement."""
        self._analyze_expression(stmt.get('value'))
        
        for case in stmt.get('cases', []):
            case_scope = self.current_scope.child_scope("case")
            self.current_scope = case_scope
            
            for s in case.get('body', []):
                self._analyze_statement(s)
            
            self.current_scope = case_scope.parent
        
        if 'default' in stmt:
            default_scope = self.current_scope.child_scope("default")
            self.current_scope = default_scope
            
            for s in stmt.get('default', []):
                self._analyze_statement(s)
            
            self.current_scope = default_scope.parent
    
    def _analyze_try(self, stmt: Dict) -> None:
        """Analyze try/catch statement."""
        # Analyze try block
        try_scope = self.current_scope.child_scope("try")
        self.current_scope = try_scope
        
        for s in stmt.get('try', []):
            self._analyze_statement(s)
        
        self.current_scope = try_scope.parent
        
        # Analyze catch blocks
        for catch in stmt.get('catch', []):
            catch_scope = self.current_scope.child_scope("catch")
            self.current_scope = catch_scope
            
            # Add exception variable
            exc_name = catch.get('exception', 'e')
            exc_var = VariableInfo(name=exc_name, type='exception', is_initialized=True)
            catch_scope.define_variable(exc_var)
            
            for s in catch.get('body', []):
                self._analyze_statement(s)
            
            self.current_scope = catch_scope.parent
        
        # Analyze finally block
        if 'finally' in stmt:
            finally_scope = self.current_scope.child_scope("finally")
            self.current_scope = finally_scope
            
            for s in stmt.get('finally', []):
                self._analyze_statement(s)
            
            self.current_scope = finally_scope.parent
    
    def _analyze_label(self, stmt: Dict) -> None:
        """Analyze label statement."""
        label = stmt.get('name', '')
        if not self.current_scope.add_label(label):
            self.issues.append(SemanticIssue(
                kind=AnalysisErrorKind.REDEFINITION,
                message=f"Label '{label}' is already defined",
                severity=WarningSeverity.ERROR
            ))
    
    def _analyze_goto(self, stmt: Dict) -> None:
        """Analyze goto statement."""
        label = stmt.get('label', '')
        # Note: We can't always check if label exists because it might be forward reference
        # This would need a two-pass approach for full checking
    
    def _analyze_block(self, stmt: Dict) -> None:
        """Analyze a block of statements."""
        block_scope = self.current_scope.child_scope("block")
        self.current_scope = block_scope
        
        for s in stmt.get('body', stmt.get('statements', [])):
            self._analyze_statement(s)
        
        self.current_scope = block_scope.parent
    
    def _analyze_expression(self, expr: Any) -> None:
        """Analyze an expression."""
        if expr is None:
            return
        
        if isinstance(expr, dict):
            expr_type = expr.get('type', '')
            
            if expr_type == 'var':
                name = expr.get('name', '')
                var_info = self.current_scope.lookup_variable(name)
                if var_info is None:
                    self.issues.append(SemanticIssue(
                        kind=AnalysisErrorKind.UNDEFINED_VARIABLE,
                        message=f"Use of undefined variable '{name}'",
                        severity=WarningSeverity.ERROR
                    ))
                else:
                    var_info.is_used = True
                    
            elif expr_type == 'binary':
                self._analyze_expression(expr.get('left'))
                self._analyze_expression(expr.get('right'))
                
            elif expr_type == 'unary':
                self._analyze_expression(expr.get('operand'))
                
            elif expr_type == 'call':
                self._analyze_call(expr)
                
            elif expr_type == 'index':
                self._analyze_expression(expr.get('base'))
                self._analyze_expression(expr.get('index'))
                
            elif expr_type == 'member':
                self._analyze_expression(expr.get('base'))
                
            elif expr_type == 'cast':
                self._analyze_expression(expr.get('value'))
                
            elif expr_type == 'array':
                for elem in expr.get('elements', []):
                    self._analyze_expression(elem)
                    
            elif expr_type == 'struct':
                for field_val in expr.get('fields', {}).values():
                    self._analyze_expression(field_val)
                    
            elif expr_type == 'tuple':
                for elem in expr.get('elements', []):
                    self._analyze_expression(elem)
                    
            elif expr_type == 'lambda':
                # Lambda creates its own scope
                lambda_scope = self.current_scope.child_scope("lambda")
                self.current_scope = lambda_scope
                
                for param in expr.get('params', []):
                    if isinstance(param, dict):
                        param_var = VariableInfo(
                            name=param.get('name', ''),
                            type=param.get('type', 'auto'),
                            is_parameter=True,
                            is_initialized=True
                        )
                    else:
                        param_var = VariableInfo(
                            name=str(param),
                            type='auto',
                            is_parameter=True,
                            is_initialized=True
                        )
                    lambda_scope.define_variable(param_var)
                
                self._analyze_expression(expr.get('body'))
                self.current_scope = lambda_scope.parent
                
            elif 'value' in expr:
                self._analyze_expression(expr['value'])
                
        elif isinstance(expr, str):
            # Could be a variable name
            var_info = self.current_scope.lookup_variable(expr)
            if var_info:
                var_info.is_used = True
    
    def _check_unused(self) -> None:
        """Check for unused functions."""
        for name, func_info in self.global_scope.functions.items():
            if not func_info.is_used and name != 'main':
                self.issues.append(SemanticIssue(
                    kind=AnalysisErrorKind.UNUSED_FUNCTION,
                    message=f"Function '{name}' is defined but never called",
                    severity=WarningSeverity.WARNING
                ))
    
    def get_errors(self) -> List[SemanticIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity in (WarningSeverity.ERROR, WarningSeverity.FATAL)]
    
    def get_warnings(self) -> List[SemanticIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == WarningSeverity.WARNING]
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(i.severity in (WarningSeverity.ERROR, WarningSeverity.FATAL) 
                  for i in self.issues)


__all__ = [
    'AnalysisErrorKind', 'WarningSeverity', 'SemanticIssue',
    'VariableInfo', 'FunctionInfo', 'SymbolTable', 'SemanticAnalyzer'
]
