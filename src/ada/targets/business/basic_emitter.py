#!/usr/bin/env python3
"""STUNIR BASIC Emitter - Generate BASIC code from Business IR.

This emitter generates classic BASIC code including:
- Line-numbered statements (10, 20, 30, ...)
- Variable assignments (LET)
- Control flow (GOTO, GOSUB, RETURN, FOR...NEXT, WHILE...WEND)
- I/O statements (INPUT, PRINT, READ, DATA)
- Array declarations (DIM)
- Function definitions (DEF FN)
- REM comments

Usage:
    from targets.business.basic_emitter import BASICEmitter
    from ir.business import BusinessProgram, Assignment
    
    emitter = BASICEmitter()
    result = emitter.emit(ir_dict)
    print(result.code)
"""

import sys
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class EmitterResult:
    """Result of code emission."""
    code: str
    manifest: dict


def canonical_json(obj: Any) -> str:
    """Generate canonical JSON (sorted keys)."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash of data."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


class BASICEmitter:
    """BASIC code emitter.
    
    Generates classic BASIC code with line numbers.
    Supports multiple dialects: standard, qbasic, gwbasic.
    """
    
    BASIC_DIALECT = 'standard'
    
    def __init__(self, dialect: str = 'standard', line_increment: int = 10):
        """Initialize BASIC emitter.
        
        Args:
            dialect: BASIC dialect ('standard', 'qbasic', 'gwbasic')
            line_increment: Increment between line numbers
        """
        self.dialect = dialect
        self.line_number = 10
        self.line_increment = line_increment
        self._errors: List[str] = []
        self._warnings: List[str] = []
        self._use_line_numbers = True
        self._subroutine_labels: Dict[str, int] = {}
    
    def emit(self, ir: dict) -> EmitterResult:
        """Generate BASIC code from Business IR.
        
        Args:
            ir: Business IR dictionary
            
        Returns:
            EmitterResult with generated code and manifest
        """
        self._errors = []
        self._warnings = []
        self.line_number = 10
        self._subroutine_labels = {}
        
        # Check if using line numbers
        self._use_line_numbers = ir.get('line_numbers', True)
        
        self._validate(ir)
        
        lines = []
        
        # Emit REM header
        lines.extend(self._emit_header(ir))
        
        # Emit DIM statements for arrays
        lines.extend(self._emit_dim_statements(ir))
        
        # Emit DEF FN statements
        lines.extend(self._emit_def_functions(ir))
        
        # Emit DATA statements
        lines.extend(self._emit_data_statements(ir))
        
        # Emit main program statements
        lines.extend(self._emit_statements(ir))
        
        # Emit END
        if ir.get('statements') or ir.get('procedures'):
            lines.append(self._line('END'))
        
        # Emit subroutines
        lines.extend(self._emit_subroutines(ir))
        
        code = '\n'.join(lines)
        manifest = self._generate_manifest(ir, code)
        
        return EmitterResult(code=code, manifest=manifest)
    
    def _validate(self, ir: dict) -> None:
        """Validate IR structure."""
        if not ir.get('name') and not ir.get('statements'):
            self._warnings.append('Program has no name and no statements')
    
    def _line(self, content: str) -> str:
        """Generate line-numbered statement."""
        if self._use_line_numbers:
            line = f'{self.line_number} {content}'
            self.line_number += self.line_increment
            return line
        return content
    
    # =========================================================================
    # Header and Declarations
    # =========================================================================
    
    def _emit_header(self, ir: dict) -> List[str]:
        """Emit program header with REM comments."""
        lines = []
        
        if ir.get('name'):
            lines.append(self._line(f'REM {ir["name"]}'))
        
        if ir.get('author'):
            lines.append(self._line(f'REM Author: {ir["author"]}'))
        
        if ir.get('date_written'):
            lines.append(self._line(f'REM Date: {ir["date_written"]}'))
        
        if lines:
            lines.append(self._line('REM'))
        
        return lines
    
    def _emit_dim_statements(self, ir: dict) -> List[str]:
        """Emit DIM statements for arrays."""
        lines = []
        
        for dim in ir.get('dim_statements', []):
            var = dim.get('variable', '')
            dims = dim.get('dimensions', [])
            dims_str = ', '.join(str(d) for d in dims)
            lines.append(self._line(f'DIM {var}({dims_str})'))
        
        return lines
    
    def _emit_def_functions(self, ir: dict) -> List[str]:
        """Emit DEF FN statements."""
        lines = []
        
        for func in ir.get('def_functions', []):
            name = func.get('name', 'A')
            param = func.get('parameter', '')
            expr = self._emit_expr(func.get('expression', {}))
            
            if param:
                lines.append(self._line(f'DEF FN{name}({param}) = {expr}'))
            else:
                lines.append(self._line(f'DEF FN{name} = {expr}'))
        
        return lines
    
    def _emit_data_statements(self, ir: dict) -> List[str]:
        """Emit DATA statements."""
        lines = []
        
        for data in ir.get('data_statements', []):
            values = data.get('values', [])
            formatted_values = []
            
            for v in values:
                if isinstance(v, dict):
                    formatted_values.append(self._format_literal(v.get('value')))
                else:
                    formatted_values.append(self._format_literal(v))
            
            lines.append(self._line(f'DATA {", ".join(formatted_values)}'))
        
        return lines
    
    # =========================================================================
    # Statement Emission
    # =========================================================================
    
    def _emit_statements(self, ir: dict) -> List[str]:
        """Emit BASIC statements."""
        lines = []
        
        for stmt in ir.get('statements', []):
            lines.extend(self._emit_statement(stmt))
        
        return lines
    
    def _emit_subroutines(self, ir: dict) -> List[str]:
        """Emit subroutines (procedures)."""
        lines = []
        
        for proc in ir.get('procedures', []):
            # Store the start line number for GOSUB references
            self._subroutine_labels[proc.get('name', '')] = self.line_number
            
            lines.append(self._line(f'REM SUBROUTINE: {proc.get("name", "")}'))
            
            for stmt in proc.get('statements', []):
                lines.extend(self._emit_statement(stmt))
            
            lines.append(self._line('RETURN'))
        
        return lines
    
    def _emit_statement(self, stmt: dict) -> List[str]:
        """Emit single BASIC statement."""
        kind = stmt.get('kind', '')
        
        # Dispatch to specific handler
        method_name = f'_emit_{kind}'
        method = getattr(self, method_name, self._emit_unknown_statement)
        return method(stmt)
    
    def _emit_unknown_statement(self, stmt: dict) -> List[str]:
        """Handle unknown statement types."""
        kind = stmt.get('kind', 'unknown')
        self._warnings.append(f'Unknown statement kind: {kind}')
        return [self._line(f'REM Unknown: {kind}')]
    
    def _emit_rem_statement(self, stmt: dict) -> List[str]:
        """Emit REM comment."""
        return [self._line(f'REM {stmt.get("text", "")}')]
    
    def _emit_assignment(self, stmt: dict) -> List[str]:
        """Emit LET statement (assignment)."""
        var = stmt.get('variable', '')
        expr = self._emit_expr(stmt.get('value', {}))
        return [self._line(f'LET {var} = {expr}')]
    
    def _emit_if_statement(self, stmt: dict) -> List[str]:
        """Emit IF statement."""
        cond = self._emit_expr(stmt.get('condition', {}))
        lines = []
        
        then_stmts = stmt.get('then_statements', [])
        else_stmts = stmt.get('else_statements', [])
        
        # QBasic-style block IF or single-line IF
        if self.dialect == 'qbasic' and (len(then_stmts) > 1 or else_stmts):
            # Block IF...THEN...ELSE...END IF
            lines.append(self._line(f'IF {cond} THEN'))
            for s in then_stmts:
                lines.extend(self._emit_statement(s))
            if else_stmts:
                lines.append(self._line('ELSE'))
                for s in else_stmts:
                    lines.extend(self._emit_statement(s))
            lines.append(self._line('END IF'))
        elif len(then_stmts) == 1 and not else_stmts:
            # Single-line IF
            then_code = self._emit_statement_inline(then_stmts[0])
            lines.append(self._line(f'IF {cond} THEN {then_code}'))
        else:
            # Classic BASIC: use GOTO for complex IF
            then_line = self.line_number + self.line_increment
            lines.append(self._line(f'IF NOT ({cond}) THEN {self._calculate_skip_line(then_stmts, else_stmts)}'))
            for s in then_stmts:
                lines.extend(self._emit_statement(s))
            if else_stmts:
                skip_else = self.line_number + self.line_increment * (len(else_stmts) + 1)
                lines.append(self._line(f'GOTO {skip_else}'))
                for s in else_stmts:
                    lines.extend(self._emit_statement(s))
        
        return lines
    
    def _calculate_skip_line(self, then_stmts: list, else_stmts: list) -> int:
        """Calculate line number to skip to for IF statement."""
        count = len(then_stmts) + 1
        if else_stmts:
            count += len(else_stmts) + 1
        return self.line_number + self.line_increment * count
    
    def _emit_statement_inline(self, stmt: dict) -> str:
        """Emit a statement inline (without line number)."""
        old_use_ln = self._use_line_numbers
        self._use_line_numbers = False
        result = self._emit_statement(stmt)
        self._use_line_numbers = old_use_ln
        return result[0] if result else ''
    
    def _emit_for_loop(self, stmt: dict) -> List[str]:
        """Emit FOR...NEXT loop."""
        var = stmt.get('variable', 'I')
        start = self._emit_expr(stmt.get('start', {}))
        end = self._emit_expr(stmt.get('end', {}))
        
        lines = []
        step_clause = ''
        
        if stmt.get('step'):
            step = self._emit_expr(stmt['step'])
            step_clause = f' STEP {step}'
        
        lines.append(self._line(f'FOR {var} = {start} TO {end}{step_clause}'))
        
        for s in stmt.get('statements', []):
            lines.extend(self._emit_statement(s))
        
        lines.append(self._line(f'NEXT {var}'))
        return lines
    
    def _emit_while_loop(self, stmt: dict) -> List[str]:
        """Emit WHILE...WEND loop."""
        cond = self._emit_expr(stmt.get('condition', {}))
        lines = [self._line(f'WHILE {cond}')]
        
        for s in stmt.get('statements', []):
            lines.extend(self._emit_statement(s))
        
        lines.append(self._line('WEND'))
        return lines
    
    def _emit_goto_statement(self, stmt: dict) -> List[str]:
        """Emit GOTO statement."""
        target = stmt.get('target', '0')
        return [self._line(f'GOTO {target}')]
    
    def _emit_gosub_statement(self, stmt: dict) -> List[str]:
        """Emit GOSUB statement."""
        target = stmt.get('target', 0)
        return [self._line(f'GOSUB {target}')]
    
    def _emit_return_statement(self, stmt: dict) -> List[str]:
        """Emit RETURN statement."""
        return [self._line('RETURN')]
    
    def _emit_end_statement(self, stmt: dict) -> List[str]:
        """Emit END statement."""
        return [self._line('END')]
    
    def _emit_stop_statement(self, stmt: dict) -> List[str]:
        """Emit STOP statement."""
        return [self._line('STOP')]
    
    # =========================================================================
    # I/O Statements
    # =========================================================================
    
    def _emit_basic_print_screen(self, stmt: dict) -> List[str]:
        """Emit PRINT statement."""
        items = []
        
        for item in stmt.get('items', []):
            if isinstance(item, dict):
                expr = self._emit_expr(item.get('value', {}))
                sep = item.get('separator', '')
                items.append(expr + sep)
            else:
                items.append(self._emit_expr(item))
        
        if items:
            return [self._line(f'PRINT {" ".join(items)}')]
        return [self._line('PRINT')]
    
    def _emit_basic_input_user(self, stmt: dict) -> List[str]:
        """Emit INPUT statement."""
        prompt = ''
        if stmt.get('prompt'):
            prompt = f'"{stmt["prompt"]}"; '
        
        vars_str = ', '.join(stmt.get('variables', []))
        return [self._line(f'INPUT {prompt}{vars_str}')]
    
    def _emit_read_data_statement(self, stmt: dict) -> List[str]:
        """Emit READ statement (from DATA)."""
        vars_str = ', '.join(stmt.get('variables', []))
        return [self._line(f'READ {vars_str}')]
    
    def _emit_restore_statement(self, stmt: dict) -> List[str]:
        """Emit RESTORE statement."""
        if stmt.get('target_line'):
            return [self._line(f'RESTORE {stmt["target_line"]}')]
        return [self._line('RESTORE')]
    
    def _emit_data_statement(self, stmt: dict) -> List[str]:
        """Emit DATA statement."""
        values = []
        for v in stmt.get('values', []):
            if isinstance(v, dict):
                values.append(self._format_literal(v.get('value')))
            else:
                values.append(self._format_literal(v))
        return [self._line(f'DATA {", ".join(values)}')]
    
    # =========================================================================
    # File Operations
    # =========================================================================
    
    def _emit_basic_open(self, stmt: dict) -> List[str]:
        """Emit OPEN statement for files."""
        file_num = stmt.get('file_number', 1)
        filename = stmt.get('filename', '')
        mode = stmt.get('mode', 'input').upper()
        
        mode_map = {
            'INPUT': 'FOR INPUT',
            'OUTPUT': 'FOR OUTPUT',
            'APPEND': 'FOR APPEND',
            'RANDOM': 'FOR RANDOM',
            'BINARY': 'FOR BINARY',
        }
        mode_str = mode_map.get(mode, 'FOR INPUT')
        
        line = f'OPEN "{filename}" {mode_str} AS #{file_num}'
        
        if stmt.get('record_length'):
            line += f' LEN = {stmt["record_length"]}'
        
        return [self._line(line)]
    
    def _emit_basic_close(self, stmt: dict) -> List[str]:
        """Emit CLOSE statement."""
        file_nums = stmt.get('file_numbers', [])
        
        if file_nums:
            nums_str = ', '.join(f'#{n}' for n in file_nums)
            return [self._line(f'CLOSE {nums_str}')]
        return [self._line('CLOSE')]
    
    def _emit_basic_input_file(self, stmt: dict) -> List[str]:
        """Emit INPUT# statement."""
        file_num = stmt.get('file_number', 1)
        vars_str = ', '.join(stmt.get('variables', []))
        return [self._line(f'INPUT #{file_num}, {vars_str}')]
    
    def _emit_basic_print_file(self, stmt: dict) -> List[str]:
        """Emit PRINT# statement."""
        file_num = stmt.get('file_number', 1)
        items = ', '.join(self._emit_expr(e) for e in stmt.get('expressions', []))
        return [self._line(f'PRINT #{file_num}, {items}')]
    
    def _emit_basic_line_input(self, stmt: dict) -> List[str]:
        """Emit LINE INPUT# statement."""
        file_num = stmt.get('file_number', 1)
        var = stmt.get('variable', '')
        return [self._line(f'LINE INPUT #{file_num}, {var}')]
    
    def _emit_basic_get(self, stmt: dict) -> List[str]:
        """Emit GET# statement."""
        file_num = stmt.get('file_number', 1)
        rec_num = stmt.get('record_number')
        
        if rec_num:
            return [self._line(f'GET #{file_num}, {rec_num}')]
        return [self._line(f'GET #{file_num}')]
    
    def _emit_basic_put(self, stmt: dict) -> List[str]:
        """Emit PUT# statement."""
        file_num = stmt.get('file_number', 1)
        rec_num = stmt.get('record_number')
        
        if rec_num:
            return [self._line(f'PUT #{file_num}, {rec_num}')]
        return [self._line(f'PUT #{file_num}')]
    
    # =========================================================================
    # Expression Emission
    # =========================================================================
    
    def _emit_expr(self, expr: Any) -> str:
        """Emit BASIC expression."""
        if expr is None:
            return '0'
        
        if isinstance(expr, str):
            return expr
        
        if isinstance(expr, (int, float)):
            return str(expr)
        
        if not isinstance(expr, dict):
            return str(expr)
        
        kind = expr.get('kind', '')
        
        if kind == 'literal':
            return self._format_literal(expr.get('value'))
        
        elif kind == 'identifier':
            name = expr.get('name', '')
            
            # Handle array subscripts
            if expr.get('subscripts'):
                subs = ', '.join(self._emit_expr(s) for s in expr['subscripts'])
                return f'{name}({subs})'
            
            return name
        
        elif kind == 'binary_expr':
            left = self._emit_expr(expr.get('left', {}))
            right = self._emit_expr(expr.get('right', {}))
            op = self._map_operator(expr.get('op', ''))
            return f'({left} {op} {right})'
        
        elif kind == 'unary_expr':
            operand = self._emit_expr(expr.get('operand', {}))
            op = expr.get('op', '')
            if op.upper() == 'NOT':
                return f'NOT {operand}'
            return f'{op}{operand}'
        
        elif kind == 'condition':
            left = self._emit_expr(expr.get('left', {}))
            right = self._emit_expr(expr.get('right', {}))
            op = self._map_condition_op(expr.get('op', ''))
            
            if expr.get('negated'):
                return f'NOT ({left} {op} {right})'
            return f'{left} {op} {right}'
        
        elif kind == 'function_call':
            name = expr.get('name', '')
            args = ', '.join(self._emit_expr(a) for a in expr.get('arguments', []))
            
            # Handle DEF FN functions
            if name.startswith('FN'):
                return f'{name}({args})'
            
            # Standard functions
            return f'{name.upper()}({args})'
        
        else:
            return str(expr)
    
    def _map_operator(self, op: str) -> str:
        """Map operator to BASIC syntax."""
        op_map = {
            'add': '+',
            'sub': '-',
            'mul': '*',
            'div': '/',
            'mod': 'MOD',
            'pow': '^',
            'and': 'AND',
            'or': 'OR',
            'xor': 'XOR',
            'not': 'NOT',
            '+': '+',
            '-': '-',
            '*': '*',
            '/': '/',
            '^': '^',
            '**': '^',
        }
        return op_map.get(op.lower(), op.upper())
    
    def _map_condition_op(self, op: str) -> str:
        """Map condition operator to BASIC syntax."""
        op_map = {
            'eq': '=',
            'ne': '<>',
            'lt': '<',
            'le': '<=',
            'gt': '>',
            'ge': '>=',
            '=': '=',
            '<>': '<>',
            '!=': '<>',
            '<': '<',
            '<=': '<=',
            '>': '>',
            '>=': '>=',
        }
        return op_map.get(op.lower(), op)
    
    def _format_literal(self, value: Any) -> str:
        """Format literal value for BASIC."""
        if value is None:
            return '0'
        
        if isinstance(value, str):
            # Escape quotes
            escaped = value.replace('"', '""')
            return f'"{escaped}"'
        
        if isinstance(value, bool):
            return '-1' if value else '0'  # BASIC TRUE is -1
        
        if isinstance(value, float):
            # Use scientific notation for very large/small numbers
            if abs(value) > 1e10 or (0 < abs(value) < 1e-10):
                return f'{value:.6E}'
            return str(value)
        
        if isinstance(value, int):
            return str(value)
        
        return str(value)
    
    # =========================================================================
    # Manifest Generation
    # =========================================================================
    
    def _generate_manifest(self, ir: dict, code: str) -> dict:
        """Generate manifest for emitted code."""
        return {
            'schema': 'stunir.codegen.basic.v1',
            'timestamp': int(time.time()),
            'program_name': ir.get('name', 'UNTITLED'),
            'dialect': self.dialect,
            'use_line_numbers': self._use_line_numbers,
            'line_increment': self.line_increment,
            'code_hash': compute_sha256(code),
            'code_lines': len(code.split('\n')),
            'dim_count': len(ir.get('dim_statements', [])),
            'def_fn_count': len(ir.get('def_functions', [])),
            'data_count': len(ir.get('data_statements', [])),
            'statement_count': len(ir.get('statements', [])),
            'procedure_count': len(ir.get('procedures', [])),
            'errors': self._errors,
            'warnings': self._warnings,
        }


# =============================================================================
# Main entry point for testing
# =============================================================================

def main():
    """Test the BASIC emitter."""
    # Sample inventory program IR
    ir = {
        'name': 'INVENTORY',
        'line_numbers': True,
        'dim_statements': [
            {'variable': 'ITEM$', 'dimensions': [100]},
            {'variable': 'QTY', 'dimensions': [100]},
            {'variable': 'PRICE', 'dimensions': [100]},
        ],
        'def_functions': [
            {
                'name': 'VALUE',
                'parameter': 'I',
                'expression': {
                    'kind': 'binary_expr',
                    'op': 'mul',
                    'left': {'kind': 'identifier', 'name': 'QTY', 'subscripts': [{'kind': 'identifier', 'name': 'I'}]},
                    'right': {'kind': 'identifier', 'name': 'PRICE', 'subscripts': [{'kind': 'identifier', 'name': 'I'}]}
                }
            }
        ],
        'statements': [
            {'kind': 'assignment', 'variable': 'COUNT', 'value': {'kind': 'literal', 'value': 0}},
            {'kind': 'rem_statement', 'text': 'MAIN MENU'},
            {'kind': 'basic_print_screen', 'items': [{'value': {'kind': 'literal', 'value': 'INVENTORY SYSTEM'}}]},
            {'kind': 'basic_print_screen', 'items': [{'value': {'kind': 'literal', 'value': '1. ADD ITEM'}}]},
            {'kind': 'basic_print_screen', 'items': [{'value': {'kind': 'literal', 'value': '2. LIST ITEMS'}}]},
            {'kind': 'basic_print_screen', 'items': [{'value': {'kind': 'literal', 'value': '3. TOTAL VALUE'}}]},
            {'kind': 'basic_print_screen', 'items': [{'value': {'kind': 'literal', 'value': '4. EXIT'}}]},
            {'kind': 'basic_input_user', 'prompt': 'CHOICE', 'variables': ['C']},
            {'kind': 'if_statement',
             'condition': {'kind': 'condition', 'left': {'kind': 'identifier', 'name': 'C'},
                           'op': '=', 'right': {'kind': 'literal', 'value': 1}},
             'then_statements': [{'kind': 'gosub_statement', 'target': 500}]},
            {'kind': 'if_statement',
             'condition': {'kind': 'condition', 'left': {'kind': 'identifier', 'name': 'C'},
                           'op': '=', 'right': {'kind': 'literal', 'value': 2}},
             'then_statements': [{'kind': 'gosub_statement', 'target': 600}]},
            {'kind': 'if_statement',
             'condition': {'kind': 'condition', 'left': {'kind': 'identifier', 'name': 'C'},
                           'op': '=', 'right': {'kind': 'literal', 'value': 3}},
             'then_statements': [{'kind': 'gosub_statement', 'target': 700}]},
            {'kind': 'if_statement',
             'condition': {'kind': 'condition', 'left': {'kind': 'identifier', 'name': 'C'},
                           'op': '=', 'right': {'kind': 'literal', 'value': 4}},
             'then_statements': [{'kind': 'end_statement'}]},
            {'kind': 'goto_statement', 'target': '30'},
        ],
        'procedures': []
    }
    
    emitter = BASICEmitter()
    result = emitter.emit(ir)
    print(result.code)
    print('\n--- Manifest ---')
    print(json.dumps(result.manifest, indent=2))


if __name__ == '__main__':
    main()
