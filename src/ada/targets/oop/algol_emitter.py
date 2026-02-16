#!/usr/bin/env python3
"""STUNIR ALGOL Emitter - Generate ALGOL-60 code from OOP IR.

This emitter generates ALGOL code supporting:
- Block structure (begin...end)
- Procedure and function declarations
- Call-by-name and call-by-value parameters
- Arrays with dynamic bounds
- For loops with step clause
- Switch statements (computed goto)
- Own variables (static)

Usage:
    from targets.oop.algol_emitter import ALGOLEmitter
    
    emitter = ALGOLEmitter()
    result = emitter.emit(ir)
    print(result.code)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set
import json
import hashlib


# =============================================================================
# Emitter Result
# =============================================================================

@dataclass
class EmitterResult:
    """Result of code emission."""
    code: str = ''
    manifest: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    file_extension: str = '.alg'


# =============================================================================
# Exceptions
# =============================================================================

class ALGOLEmitterError(Exception):
    """Base error for ALGOL emitter."""
    pass


class UnsupportedFeatureError(ALGOLEmitterError):
    """Feature not supported by ALGOL emitter."""
    pass


# =============================================================================
# ALGOL Emitter
# =============================================================================

class ALGOLEmitter:
    """Emit ALGOL code from OOP IR."""
    
    DIALECT = 'algol'
    FILE_EXTENSION = '.alg'
    
    # ALGOL-60 keywords
    KEYWORDS: Set[str] = {
        'begin', 'end', 'if', 'then', 'else', 'for', 'do', 'while',
        'step', 'until', 'procedure', 'function', 'integer', 'real',
        'Boolean', 'array', 'switch', 'goto', 'own', 'value', 'string',
        'label', 'comment', 'true', 'false', 'and', 'or', 'not', 'equiv',
        'impl'
    }
    
    # Type mapping from IR types to ALGOL types
    TYPE_MAP: Dict[str, str] = {
        'i8': 'integer',
        'i16': 'integer',
        'i32': 'integer',
        'i64': 'integer',
        'f32': 'real',
        'f64': 'real',
        'bool': 'Boolean',
        'boolean': 'Boolean',
        'string': 'string',
        'int': 'integer',
        'float': 'real',
        'double': 'real',
    }
    
    # Operator mapping
    OPERATOR_MAP: Dict[str, str] = {
        '&&': 'and',
        '||': 'or',
        '!': 'not',
        '==': '=',
        '!=': '≠',
        '<=': '≤',
        '>=': '≥',
        '->': '⊃',  # implies
        '<->': '≡',  # equivalent
        '^': '↑',   # exponentiation
        '**': '↑',
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the emitter.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._indent = 0
        self._warnings: List[str] = []
        self._use_ascii = config.get('use_ascii', True) if config else True
    
    def emit(self, ir: Dict[str, Any]) -> EmitterResult:
        """Emit ALGOL code from IR.
        
        Args:
            ir: IR dictionary (program, procedure, or block)
            
        Returns:
            EmitterResult with code and manifest
        """
        self._warnings = []
        self._indent = 0
        
        kind = ir.get('kind', '')
        
        if kind == 'algol_program':
            code = self._emit_program(ir)
        elif kind == 'algol_procedure':
            code = self._emit_procedure(ir)
        elif kind == 'algol_block':
            code = self._emit_block(ir)
        else:
            # Try to emit as expression/statement
            code = self._emit_statement(ir)
        
        manifest = self._generate_manifest(ir, code)
        
        return EmitterResult(
            code=code,
            manifest=manifest,
            warnings=self._warnings,
            file_extension=self.FILE_EXTENSION
        )
    
    def emit_procedure(self, ir: Dict[str, Any]) -> str:
        """Emit a single procedure definition."""
        return self._emit_procedure(ir)
    
    def emit_block(self, ir: Dict[str, Any]) -> str:
        """Emit a single block."""
        return self._emit_block(ir)
    
    # =========================================================================
    # Program Emission
    # =========================================================================
    
    def _emit_program(self, prog: Dict[str, Any]) -> str:
        """Emit complete ALGOL program."""
        parts = []
        
        # Comment header
        name = prog.get('name', 'program')
        parts.append(f"comment STUNIR Generated ALGOL Program: {name};")
        parts.append('')
        
        # Main block
        main_block = prog.get('main_block')
        if main_block:
            # Procedures are typically declared inside the main block
            procedures = prog.get('procedures', [])
            if procedures:
                # Add procedures to block declarations
                if 'declarations' not in main_block:
                    main_block['declarations'] = []
                for proc in procedures:
                    main_block['declarations'].append(proc)
            
            parts.append(self._emit_block(main_block))
        else:
            # Just procedures
            for proc in prog.get('procedures', []):
                parts.append(self._emit_procedure(proc))
                parts.append('')
        
        return '\n'.join(parts)
    
    # =========================================================================
    # Block Structure
    # =========================================================================
    
    def _emit_block(self, block: Dict[str, Any]) -> str:
        """Emit ALGOL block: begin declarations; statements end."""
        decls = block.get('declarations', [])
        stmts = block.get('statements', [])
        label = block.get('label')
        
        lines = []
        
        # Label
        if label:
            lines.append(f"{label}:")
        
        lines.append(self._indent_str() + 'begin')
        self._indent += 1
        
        # Declarations
        for decl in decls:
            decl_code = self._emit_declaration(decl)
            if decl_code:
                lines.append(self._indent_str() + decl_code + ';')
        
        # Empty line between declarations and statements
        if decls and stmts:
            lines.append('')
        
        # Statements
        for i, stmt in enumerate(stmts):
            stmt_str = self._emit_statement(stmt)
            # Last statement without semicolon before 'end'
            if i < len(stmts) - 1:
                lines.append(self._indent_str() + stmt_str + ';')
            else:
                lines.append(self._indent_str() + stmt_str)
        
        self._indent -= 1
        lines.append(self._indent_str() + 'end')
        
        return '\n'.join(lines)
    
    # =========================================================================
    # Declarations
    # =========================================================================
    
    def _emit_declaration(self, decl: Dict[str, Any]) -> str:
        """Emit a declaration."""
        kind = decl.get('kind', '')
        
        if kind == 'algol_procedure':
            return self._emit_procedure(decl)
        elif kind == 'algol_var_decl':
            return self._emit_var_declaration(decl)
        elif kind == 'algol_array':
            return self._emit_array_declaration(decl)
        elif kind == 'algol_switch':
            return self._emit_switch(decl)
        elif kind == 'own_variable':
            return self._emit_own_variable(decl)
        elif kind == 'var_decl':
            # Generic variable declaration
            return self._emit_var_declaration(decl)
        else:
            self._warnings.append(f"Unknown declaration kind: {kind}")
            return f"comment unknown declaration: {kind}"
    
    def _emit_var_declaration(self, decl: Dict[str, Any]) -> str:
        """Emit variable declaration."""
        name = decl.get('name', 'var')
        var_type = decl.get('var_type', decl.get('type', 'integer'))
        algol_type = self.TYPE_MAP.get(var_type, var_type)
        
        init = decl.get('initial_value')
        if init:
            init_str = f" := {self._emit_expression(init)}"
        else:
            init_str = ''
        
        return f"{algol_type} {name}{init_str}"
    
    def _emit_own_variable(self, own: Dict[str, Any]) -> str:
        """Emit own variable declaration."""
        name = own.get('name', 'var')
        var_type = own.get('var_type', 'integer')
        algol_type = self.TYPE_MAP.get(var_type, var_type)
        
        init = own.get('initial_value')
        if init:
            init_str = f" := {self._emit_expression(init)}"
        else:
            init_str = ''
        
        return f"own {algol_type} {name}{init_str}"
    
    # =========================================================================
    # Procedure Declaration
    # =========================================================================
    
    def _emit_procedure(self, proc: Dict[str, Any]) -> str:
        """Emit ALGOL procedure.
        
        Format:
            procedure name(param1, param2); value param1; integer param1; real param2;
            begin
                ...
            end
        
        Or function:
            real procedure func(x); value x; real x;
            begin
                func := ...
            end
        """
        name = proc.get('name', 'proc')
        params = proc.get('parameters', [])
        result_type = proc.get('result_type')
        body = proc.get('body', {})
        own_vars = proc.get('own_variables', [])
        
        lines = []
        
        # Function/procedure header
        if result_type:
            type_str = self.TYPE_MAP.get(result_type, result_type)
            header = f"{type_str} procedure {name}"
        else:
            header = f"procedure {name}"
        
        # Parameters
        if params:
            param_names = [self._get_param_name(p) for p in params]
            header += f"({', '.join(param_names)})"
        header += ';'
        lines.append(self._indent_str() + header)
        
        self._indent += 1
        
        # Value specification (call-by-value params)
        value_params = [
            self._get_param_name(p) for p in params 
            if self._get_param_mode(p) == 'value'
        ]
        if value_params:
            lines.append(self._indent_str() + f"value {', '.join(value_params)};")
        
        # Type specifications for all parameters
        for param in params:
            param_name = self._get_param_name(param)
            param_type = self._get_param_type(param)
            algol_type = self.TYPE_MAP.get(param_type, param_type)
            lines.append(self._indent_str() + f"{algol_type} {param_name};")
        
        # Own variables
        for own in own_vars:
            own_decl = self._emit_own_variable(own)
            lines.append(self._indent_str() + own_decl + ';')
        
        self._indent -= 1
        
        # Body
        if body:
            lines.append(self._emit_block(body))
        else:
            lines.append(self._indent_str() + 'begin end')
        
        return '\n'.join(lines)
    
    def _get_param_name(self, param: Any) -> str:
        """Get parameter name."""
        if isinstance(param, dict):
            return param.get('name', 'param')
        return str(param)
    
    def _get_param_type(self, param: Any) -> str:
        """Get parameter type."""
        if isinstance(param, dict):
            return param.get('param_type', param.get('type', 'integer'))
        return 'integer'
    
    def _get_param_mode(self, param: Any) -> str:
        """Get parameter mode (value or name)."""
        if isinstance(param, dict):
            mode = param.get('mode', 'value')
            if isinstance(mode, str):
                return mode
            return mode.value if hasattr(mode, 'value') else str(mode)
        return 'value'
    
    # =========================================================================
    # Arrays
    # =========================================================================
    
    def _emit_array_declaration(self, arr: Dict[str, Any]) -> str:
        """Emit array declaration with potentially dynamic bounds.
        
        Format: array A[1:n, 1:m]
        """
        name = arr.get('name', 'arr')
        elem_type = arr.get('element_type', 'real')
        algol_type = self.TYPE_MAP.get(elem_type, elem_type)
        bounds = arr.get('bounds', [])
        
        bound_strs = []
        for bound in bounds:
            if isinstance(bound, (list, tuple)) and len(bound) == 2:
                low = bound[0]
                high = bound[1]
            else:
                low = {'kind': 'literal', 'value': 1}
                high = bound
            
            low_str = self._emit_expression(low)
            high_str = self._emit_expression(high)
            bound_strs.append(f"{low_str}:{high_str}")
        
        if algol_type != 'real':
            return f"{algol_type} array {name}[{', '.join(bound_strs)}]"
        return f"array {name}[{', '.join(bound_strs)}]"
    
    # =========================================================================
    # Switch
    # =========================================================================
    
    def _emit_switch(self, switch: Dict[str, Any]) -> str:
        """Emit ALGOL switch declaration.
        
        Format: switch S := L1, L2, L3, L4
        """
        name = switch.get('name', 'S')
        labels = switch.get('labels', [])
        return f"switch {name} := {', '.join(labels)}"
    
    # =========================================================================
    # Statements
    # =========================================================================
    
    def _emit_statement(self, stmt: Dict[str, Any]) -> str:
        """Emit a statement."""
        if stmt is None:
            return ''
        
        kind = stmt.get('kind', '')
        
        if kind == 'algol_block':
            return self._emit_block(stmt)
        elif kind == 'algol_for':
            return self._emit_for_loop(stmt)
        elif kind == 'algol_if':
            return self._emit_if_statement(stmt)
        elif kind == 'algol_goto':
            return self._emit_goto(stmt)
        elif kind == 'algol_call':
            return self._emit_procedure_call(stmt)
        elif kind == 'algol_comment':
            return self._emit_comment(stmt)
        elif kind == 'assignment':
            return self._emit_assignment(stmt)
        elif kind == 'return':
            return self._emit_return(stmt)
        elif kind == 'var_decl' or kind == 'algol_var_decl':
            return self._emit_var_declaration(stmt)
        elif kind in ('literal', 'variable', 'binary_op', 'unary_op'):
            return self._emit_expression(stmt)
        else:
            self._warnings.append(f"Unknown statement kind: {kind}")
            return f"comment unknown: {kind}"
    
    def _emit_for_loop(self, loop: Dict[str, Any]) -> str:
        """Emit ALGOL for loop.
        
        Formats:
            for i := 1 step 1 until n do statement
            for i := 1, 2, 3, 10 step 5 until 50 do statement
            for i := 1 while condition do statement
        """
        var = loop.get('variable', 'i')
        init = loop.get('init_value')
        body = loop.get('body')
        
        step = loop.get('step')
        until = loop.get('until_value')
        while_cond = loop.get('while_condition')
        
        init_str = self._emit_expression(init) if init else '1'
        
        # Emit body
        if body:
            body_str = self._emit_statement(body)
        else:
            body_str = 'begin end'
        
        if while_cond:
            # for-while variant
            cond = self._emit_expression(while_cond)
            return f"for {var} := {init_str} while {cond} do\n{self._indent_str()}\t{body_str}"
        elif step and until:
            # Standard step-until
            step_str = self._emit_expression(step)
            until_str = self._emit_expression(until)
            return f"for {var} := {init_str} step {step_str} until {until_str} do\n{self._indent_str()}\t{body_str}"
        elif until:
            # Step defaults to 1
            until_str = self._emit_expression(until)
            return f"for {var} := {init_str} step 1 until {until_str} do\n{self._indent_str()}\t{body_str}"
        else:
            # Simple for (just initialization)
            return f"for {var} := {init_str} do\n{self._indent_str()}\t{body_str}"
    
    def _emit_if_statement(self, if_stmt: Dict[str, Any]) -> str:
        """Emit ALGOL if statement."""
        condition = self._emit_expression(if_stmt.get('condition'))
        then_stmt = if_stmt.get('then_branch')
        else_stmt = if_stmt.get('else_branch')
        
        then_str = self._emit_statement(then_stmt) if then_stmt else 'begin end'
        
        if else_stmt:
            else_str = self._emit_statement(else_stmt)
            return f"if {condition} then {then_str} else {else_str}"
        return f"if {condition} then {then_str}"
    
    def _emit_goto(self, goto: Dict[str, Any]) -> str:
        """Emit goto statement."""
        target = goto.get('target', '')
        switch_name = goto.get('switch_name')
        switch_index = goto.get('switch_index')
        
        if switch_name and switch_index:
            # Switch-based goto: goto S[i]
            idx = self._emit_expression(switch_index)
            return f"goto {switch_name}[{idx}]"
        return f"goto {target}"
    
    def _emit_procedure_call(self, call: Dict[str, Any]) -> str:
        """Emit procedure call."""
        name = call.get('name', '')
        args = call.get('arguments', [])
        
        if not args:
            return name
        
        arg_strs = [self._emit_expression(arg) for arg in args]
        return f"{name}({', '.join(arg_strs)})"
    
    def _emit_comment(self, cmt: Dict[str, Any]) -> str:
        """Emit ALGOL comment."""
        text = cmt.get('text', '')
        return f"comment {text}"
    
    def _emit_assignment(self, assign: Dict[str, Any]) -> str:
        """Emit assignment: target := value."""
        target = assign.get('target', '')
        value = self._emit_expression(assign.get('value'))
        return f"{target} := {value}"
    
    def _emit_return(self, ret: Dict[str, Any]) -> str:
        """Emit return (function result assignment in ALGOL)."""
        value = ret.get('value')
        # In ALGOL, function result is assigned to function name
        if value:
            return f"result := {self._emit_expression(value)}"
        return ''
    
    # =========================================================================
    # Expressions
    # =========================================================================
    
    def _emit_expression(self, expr: Any) -> str:
        """Emit expression."""
        if expr is None:
            return ''
        
        if isinstance(expr, (int, float)):
            return str(expr)
        if isinstance(expr, str):
            return expr
        if isinstance(expr, bool):
            return 'true' if expr else 'false'
        
        if not isinstance(expr, dict):
            return str(expr)
        
        kind = expr.get('kind', '')
        
        if kind == 'literal':
            return self._emit_literal(expr)
        elif kind == 'variable':
            return expr.get('name', '')
        elif kind == 'binary_op':
            return self._emit_binary_op(expr)
        elif kind == 'unary_op':
            return self._emit_unary_op(expr)
        elif kind == 'algol_call':
            return self._emit_procedure_call(expr)
        elif kind == 'array_access':
            return self._emit_array_access(expr)
        else:
            # Try to get a name or value
            if 'name' in expr:
                return expr['name']
            if 'value' in expr:
                return str(expr['value'])
            self._warnings.append(f"Unknown expression kind: {kind}")
            return f"0 comment unknown: {kind}"
    
    def _emit_literal(self, lit: Dict[str, Any]) -> str:
        """Emit literal value."""
        value = lit.get('value')
        lit_type = lit.get('literal_type', '')
        
        if lit_type == 'string':
            return f'"{value}"'
        elif value is True:
            return 'true'
        elif value is False:
            return 'false'
        elif isinstance(value, float):
            # ALGOL uses 'e' for scientific notation
            return str(value)
        return str(value)
    
    def _emit_binary_op(self, op: Dict[str, Any]) -> str:
        """Emit binary operation."""
        left = self._emit_expression(op.get('left'))
        right = self._emit_expression(op.get('right'))
        operator = op.get('operator', '+')
        
        # Map operator if needed
        if self._use_ascii:
            # Use ASCII versions
            algol_op = {
                '≠': '/=',
                '≤': '<=',
                '≥': '>=',
                '⊃': 'impl',
                '≡': 'equiv',
                '↑': '^',
            }.get(operator, operator)
        else:
            algol_op = self.OPERATOR_MAP.get(operator, operator)
        
        # Handle special ALGOL operators
        if algol_op in ('and', 'or', 'impl', 'equiv'):
            return f"({left} {algol_op} {right})"
        
        return f"({left} {algol_op} {right})"
    
    def _emit_unary_op(self, op: Dict[str, Any]) -> str:
        """Emit unary operation."""
        operator = op.get('operator', '-')
        operand = self._emit_expression(op.get('operand'))
        
        # Map operator
        if operator == '!' or operator == 'not':
            return f"not {operand}"
        elif operator == '-':
            return f"-{operand}"
        elif operator == '+':
            return f"+{operand}"
        
        return f"{operator}{operand}"
    
    def _emit_array_access(self, access: Dict[str, Any]) -> str:
        """Emit array element access."""
        array = access.get('array', '')
        if isinstance(array, dict):
            array = self._emit_expression(array)
        indices = access.get('indices', [])
        
        idx_strs = [self._emit_expression(idx) for idx in indices]
        return f"{array}[{', '.join(idx_strs)}]"
    
    # =========================================================================
    # Manifest Generation
    # =========================================================================
    
    def _generate_manifest(self, ir: Dict[str, Any], code: str) -> Dict[str, Any]:
        """Generate deterministic manifest."""
        code_hash = hashlib.sha256(code.encode('utf-8')).hexdigest()
        ir_hash = hashlib.sha256(
            json.dumps(ir, sort_keys=True, default=str).encode('utf-8')
        ).hexdigest()
        
        manifest = {
            'schema': 'stunir.manifest.targets.v1',
            'generator': 'stunir.algol.emitter',
            'dialect': self.DIALECT,
            'ir_hash': ir_hash,
            'output': {
                'hash': code_hash,
                'size': len(code),
                'format': 'algol',
                'extension': self.FILE_EXTENSION,
            },
        }
        
        return manifest
    
    # =========================================================================
    # Utility Functions
    # =========================================================================
    
    def _indent_str(self) -> str:
        """Get current indentation string."""
        return '    ' * self._indent
    
    def _map_type(self, ir_type: str) -> str:
        """Map IR type to ALGOL type."""
        return self.TYPE_MAP.get(ir_type, ir_type)
