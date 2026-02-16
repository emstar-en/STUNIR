#!/usr/bin/env python3
"""STUNIR COBOL Emitter - Generate COBOL code from Business IR.

This emitter generates COBOL-85/2002 code including:
- Four divisions (IDENTIFICATION, ENVIRONMENT, DATA, PROCEDURE)
- Record structures with level numbers (01-49)
- File handling (OPEN, READ, WRITE, CLOSE)
- PICTURE clauses for data types
- PERFORM loops (simple, UNTIL, VARYING)
- Conditional statements (IF, EVALUATE)
- MOVE and COMPUTE statements

Usage:
    from targets.business.cobol_emitter import COBOLEmitter
    from ir.business import BusinessProgram, Division
    
    emitter = COBOLEmitter()
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


class COBOLEmitter:
    """COBOL code emitter.
    
    Generates standard COBOL-85 code with proper formatting:
    - Columns 1-6: Sequence number area (optional)
    - Column 7: Indicator area (*, /, -)
    - Columns 8-11: Area A (division/section/paragraph names)
    - Columns 12-72: Area B (statements)
    - Columns 73-80: Identification area (optional)
    """
    
    COBOL_VERSION = '85'
    
    # COBOL reserved words that need special handling
    RESERVED_WORDS = {
        'ACCEPT', 'ADD', 'CALL', 'CLOSE', 'COMPUTE', 'DELETE',
        'DISPLAY', 'DIVIDE', 'EVALUATE', 'IF', 'INITIALIZE',
        'INSPECT', 'MERGE', 'MOVE', 'MULTIPLY', 'OPEN',
        'PERFORM', 'READ', 'RELEASE', 'RETURN', 'REWRITE',
        'SEARCH', 'SET', 'SORT', 'START', 'STOP', 'STRING',
        'SUBTRACT', 'UNSTRING', 'WRITE',
    }
    
    def __init__(self, dialect: str = 'standard'):
        """Initialize COBOL emitter.
        
        Args:
            dialect: COBOL dialect ('standard', 'ibm', 'microfocus', 'gnu')
        """
        self.dialect = dialect
        self.indent_size = 3
        self._errors: List[str] = []
        self._warnings: List[str] = []
    
    def emit(self, ir: dict) -> EmitterResult:
        """Generate COBOL code from Business IR.
        
        Args:
            ir: Business IR dictionary
            
        Returns:
            EmitterResult with generated code and manifest
        """
        self._errors = []
        self._warnings = []
        
        self._validate(ir)
        
        # Generate four divisions
        id_div = self._emit_identification_division(ir)
        env_div = self._emit_environment_division(ir)
        data_div = self._emit_data_division(ir)
        proc_div = self._emit_procedure_division(ir)
        
        code = '\n'.join([id_div, env_div, data_div, proc_div])
        manifest = self._generate_manifest(ir, code)
        
        return EmitterResult(code=code, manifest=manifest)
    
    def _validate(self, ir: dict) -> None:
        """Validate IR structure."""
        if not ir.get('name'):
            self._errors.append('Program name is required')
        
        name = ir.get('name', '')
        if len(name) > 30:
            self._warnings.append(f'Program name "{name}" exceeds 30 characters')
    
    # =========================================================================
    # Division Emission
    # =========================================================================
    
    def _emit_identification_division(self, ir: dict) -> str:
        """Emit IDENTIFICATION DIVISION."""
        lines = [
            '       IDENTIFICATION DIVISION.',
            f'       PROGRAM-ID. {ir.get("name", "UNNAMED")}.',
        ]
        
        if ir.get('author'):
            lines.append(f'       AUTHOR. {ir["author"]}.')
        
        if ir.get('date_written'):
            lines.append(f'       DATE-WRITTEN. {ir["date_written"]}.')
        
        if ir.get('date_compiled'):
            lines.append(f'       DATE-COMPILED. {ir["date_compiled"]}.')
        
        if ir.get('installation'):
            lines.append(f'       INSTALLATION. {ir["installation"]}.')
        
        if ir.get('security'):
            lines.append(f'       SECURITY. {ir["security"]}.')
        
        return '\n'.join(lines)
    
    def _emit_environment_division(self, ir: dict) -> str:
        """Emit ENVIRONMENT DIVISION."""
        lines = ['       ENVIRONMENT DIVISION.']
        
        # Configuration section
        if ir.get('special_names') or ir.get('source_computer') or ir.get('object_computer'):
            lines.append('       CONFIGURATION SECTION.')
            
            if ir.get('source_computer'):
                lines.append(f'       SOURCE-COMPUTER. {ir["source_computer"]}.')
            
            if ir.get('object_computer'):
                lines.append(f'       OBJECT-COMPUTER. {ir["object_computer"]}.')
            
            if ir.get('special_names'):
                lines.append('       SPECIAL-NAMES.')
                for name, value in ir['special_names'].items():
                    lines.append(f'           {name} IS {value}.')
        
        # Input-Output section
        if ir.get('files'):
            lines.append('       INPUT-OUTPUT SECTION.')
            lines.append('       FILE-CONTROL.')
            for file in ir['files']:
                lines.extend(self._emit_file_control(file))
        
        return '\n'.join(lines)
    
    def _emit_data_division(self, ir: dict) -> str:
        """Emit DATA DIVISION."""
        lines = ['       DATA DIVISION.']
        
        # File section
        if ir.get('files'):
            lines.append('       FILE SECTION.')
            for file in ir['files']:
                lines.extend(self._emit_file_description(file))
        
        # Working-storage section
        lines.append('       WORKING-STORAGE SECTION.')
        for item in ir.get('data_items', []):
            lines.extend(self._emit_data_item(item))
        
        # Linkage section (if needed)
        if ir.get('linkage_items'):
            lines.append('       LINKAGE SECTION.')
            for item in ir['linkage_items']:
                lines.extend(self._emit_data_item(item))
        
        return '\n'.join(lines)
    
    def _emit_procedure_division(self, ir: dict) -> str:
        """Emit PROCEDURE DIVISION."""
        lines = []
        
        # Check for USING clause (for called programs)
        if ir.get('using_params'):
            params = ' '.join(ir['using_params'])
            lines.append(f'       PROCEDURE DIVISION USING {params}.')
        else:
            lines.append('       PROCEDURE DIVISION.')
        
        # Emit paragraphs
        for paragraph in ir.get('paragraphs', []):
            lines.append(f'       {paragraph["name"]}.')
            for stmt in paragraph.get('statements', []):
                lines.extend(self._emit_statement(stmt))
        
        # Add STOP RUN if not already present
        if ir.get('paragraphs'):
            last_para = ir['paragraphs'][-1]
            last_stmts = last_para.get('statements', [])
            if not last_stmts or last_stmts[-1].get('kind') != 'stop_statement':
                lines.append('           STOP RUN.')
        else:
            lines.append('           STOP RUN.')
        
        return '\n'.join(lines)
    
    # =========================================================================
    # File Handling
    # =========================================================================
    
    def _emit_file_control(self, file: dict) -> List[str]:
        """Emit FILE-CONTROL SELECT entry."""
        lines = [f'           SELECT {file["name"]}']
        
        assign_to = file.get('assign_to', file['name'])
        lines.append(f'               ASSIGN TO "{assign_to}"')
        
        org = file.get('organization', 'sequential')
        if org != 'sequential':
            lines.append(f'               ORGANIZATION IS {org.upper()}')
        
        access = file.get('access', 'sequential')
        if access != 'sequential':
            lines.append(f'               ACCESS MODE IS {access.upper()}')
        
        if file.get('record_key'):
            lines.append(f'               RECORD KEY IS {file["record_key"]}')
        
        for alt_key in file.get('alternate_keys', []):
            dup = ' WITH DUPLICATES' if alt_key.get('with_duplicates') else ''
            lines.append(f'               ALTERNATE RECORD KEY IS {alt_key["name"]}{dup}')
        
        if file.get('relative_key'):
            lines.append(f'               RELATIVE KEY IS {file["relative_key"]}')
        
        if file.get('file_status'):
            lines.append(f'               FILE STATUS IS {file["file_status"]}')
        
        lines[-1] += '.'
        return lines
    
    def _emit_file_description(self, file: dict) -> List[str]:
        """Emit FD (File Description) entry."""
        lines = [f'       FD  {file["name"]}']
        
        if file.get('block_contains'):
            lines.append(f'           BLOCK CONTAINS {file["block_contains"]} RECORDS')
        
        if file.get('record_contains'):
            rc = file['record_contains']
            if rc.get('max_chars'):
                lines.append(f'           RECORD CONTAINS {rc["min_chars"]} TO {rc["max_chars"]} CHARACTERS')
            else:
                lines.append(f'           RECORD CONTAINS {rc["min_chars"]} CHARACTERS')
        
        label = file.get('label_records', 'standard')
        lines.append(f'           LABEL RECORDS ARE {label.upper()}')
        
        lines[-1] += '.'
        
        # Emit record structure
        if file.get('record'):
            lines.extend(self._emit_data_item(file['record']))
        
        for record in file.get('records', []):
            lines.extend(self._emit_data_item(record))
        
        return lines
    
    # =========================================================================
    # Data Item Emission
    # =========================================================================
    
    def _emit_data_item(self, item: dict) -> List[str]:
        """Emit COBOL data item with level number."""
        lines = []
        level = str(item.get('level', 1)).zfill(2)
        name = item.get('name', 'FILLER')
        
        # Build data description entry
        entry = f'       {level}  {name}'
        
        # REDEFINES must come immediately after name
        if item.get('redefines'):
            entry += f' REDEFINES {item["redefines"]}'
        
        # PICTURE clause
        if item.get('picture'):
            pic = item['picture']
            pattern = pic.get('pattern', pic) if isinstance(pic, dict) else pic
            entry += f' PIC {pattern}'
        
        # USAGE clause
        usage = item.get('usage', 'display')
        if usage and usage != 'display':
            entry += f' USAGE {usage.upper()}'
        
        # VALUE clause
        if item.get('value') is not None:
            entry += f' VALUE {self._format_literal(item["value"])}'
        
        # OCCURS clause
        if item.get('occurs'):
            occ = item['occurs']
            if occ.get('depending_on'):
                entry += f' OCCURS {occ.get("min_times", 1)} TO {occ.get("max_times", occ.get("times", 1))}'
                entry += f' TIMES DEPENDING ON {occ["depending_on"]}'
            else:
                entry += f' OCCURS {occ.get("times", 1)} TIMES'
            
            if occ.get('indexed_by'):
                idx_list = occ['indexed_by']
                if isinstance(idx_list, list):
                    entry += f' INDEXED BY {", ".join(idx_list)}'
                else:
                    entry += f' INDEXED BY {idx_list}'
            
            for key in occ.get('keys', []):
                order = 'ASCENDING' if key.get('ascending', True) else 'DESCENDING'
                entry += f' {order} KEY IS {key["name"]}'
        
        # Other clauses
        if item.get('justified'):
            entry += ' JUSTIFIED RIGHT'
        
        if item.get('blank_when_zero'):
            entry += ' BLANK WHEN ZERO'
        
        if item.get('synchronized'):
            entry += ' SYNCHRONIZED'
        
        if item.get('sign_leading'):
            sign_sep = ' SEPARATE' if item.get('sign_separate') else ''
            entry += f' SIGN LEADING{sign_sep}'
        elif item.get('sign_separate'):
            entry += ' SIGN TRAILING SEPARATE'
        
        entry += '.'
        lines.append(entry)
        
        # Emit child items (for group items)
        for child in item.get('children', []):
            lines.extend(self._emit_data_item(child))
        
        # Emit condition names (88-level)
        for cond in item.get('condition_names', []):
            values = ' '.join(self._format_literal(v) for v in cond.get('values', []))
            lines.append(f'       88  {cond["name"]} VALUE IS {values}.')
        
        return lines
    
    # =========================================================================
    # Statement Emission
    # =========================================================================
    
    def _emit_statement(self, stmt: dict) -> List[str]:
        """Emit COBOL statement."""
        kind = stmt.get('kind', '')
        method_name = f'_emit_{kind}'
        method = getattr(self, method_name, self._emit_unknown_statement)
        return method(stmt)
    
    def _emit_unknown_statement(self, stmt: dict) -> List[str]:
        """Handle unknown statement types."""
        kind = stmt.get('kind', 'unknown')
        self._warnings.append(f'Unknown statement kind: {kind}')
        return [f'      * Unknown statement: {kind}']
    
    def _emit_move_statement(self, stmt: dict) -> List[str]:
        """Emit MOVE statement."""
        source = self._emit_expr(stmt.get('source', {}))
        dests = ' '.join(stmt.get('destinations', []))
        
        if stmt.get('corresponding'):
            return [f'           MOVE CORRESPONDING {source} TO {dests}.']
        return [f'           MOVE {source} TO {dests}.']
    
    def _emit_compute_statement(self, stmt: dict) -> List[str]:
        """Emit COMPUTE statement."""
        target = stmt.get('target', '')
        expr = self._emit_expr(stmt.get('expression', {}))
        rounded = ' ROUNDED' if stmt.get('rounded') else ''
        
        lines = [f'           COMPUTE {target}{rounded} = {expr}']
        
        if stmt.get('on_size_error'):
            lines.append('               ON SIZE ERROR')
            for s in stmt['on_size_error']:
                lines.extend(self._emit_statement(s))
        
        if stmt.get('not_on_size_error'):
            lines.append('               NOT ON SIZE ERROR')
            for s in stmt['not_on_size_error']:
                lines.extend(self._emit_statement(s))
        
        lines[-1] += '.'
        return lines
    
    def _emit_add_statement(self, stmt: dict) -> List[str]:
        """Emit ADD statement."""
        values = ' '.join(self._emit_expr(v) for v in stmt.get('values', []))
        
        if stmt.get('giving'):
            target = stmt['giving']
            if stmt.get('to_value'):
                return [f'           ADD {values} TO {stmt["to_value"]} GIVING {target}.']
            return [f'           ADD {values} GIVING {target}.']
        elif stmt.get('to_value'):
            return [f'           ADD {values} TO {stmt["to_value"]}.']
        return [f'           ADD {values}.']
    
    def _emit_subtract_statement(self, stmt: dict) -> List[str]:
        """Emit SUBTRACT statement."""
        values = ' '.join(self._emit_expr(v) for v in stmt.get('values', []))
        from_val = stmt.get('from_value', '')
        
        if stmt.get('giving'):
            return [f'           SUBTRACT {values} FROM {from_val} GIVING {stmt["giving"]}.']
        return [f'           SUBTRACT {values} FROM {from_val}.']
    
    def _emit_multiply_statement(self, stmt: dict) -> List[str]:
        """Emit MULTIPLY statement."""
        val1 = self._emit_expr(stmt.get('value1', {}))
        by_val = self._emit_expr(stmt.get('by_value', {}))
        
        if stmt.get('giving'):
            return [f'           MULTIPLY {val1} BY {by_val} GIVING {stmt["giving"]}.']
        return [f'           MULTIPLY {val1} BY {by_val}.']
    
    def _emit_divide_statement(self, stmt: dict) -> List[str]:
        """Emit DIVIDE statement."""
        val1 = self._emit_expr(stmt.get('value1', {}))
        into_val = self._emit_expr(stmt.get('into_value', {}))
        
        line = f'           DIVIDE {val1} INTO {into_val}'
        
        if stmt.get('giving'):
            line += f' GIVING {stmt["giving"]}'
        
        if stmt.get('remainder'):
            line += f' REMAINDER {stmt["remainder"]}'
        
        return [line + '.']
    
    def _emit_perform_statement(self, stmt: dict) -> List[str]:
        """Emit PERFORM statement."""
        lines = ['           PERFORM']
        
        if stmt.get('inline_statements'):
            # Inline PERFORM
            for s in stmt['inline_statements']:
                lines.extend(self._emit_statement(s))
            lines.append('           END-PERFORM')
        else:
            # Out-of-line PERFORM
            para = stmt.get('paragraph_name', '')
            if stmt.get('through'):
                lines[-1] += f' {para} THRU {stmt["through"]}'
            else:
                lines[-1] += f' {para}'
            
            if stmt.get('times'):
                lines[-1] += f' {self._emit_expr(stmt["times"])} TIMES'
            elif stmt.get('until'):
                test = stmt.get('with_test', 'before')
                if test == 'after':
                    lines[-1] += ' WITH TEST AFTER'
                lines[-1] += f' UNTIL {self._emit_expr(stmt["until"])}'
            elif stmt.get('varying'):
                v = stmt['varying']
                lines[-1] += f' VARYING {v["identifier"]}'
                lines[-1] += f' FROM {self._emit_expr(v.get("from_value", {}))}'
                lines[-1] += f' BY {self._emit_expr(v.get("by_value", {}))}'
                lines[-1] += f' UNTIL {self._emit_expr(v.get("until_value", {}))}'
                
                # Handle AFTER clauses
                for after in v.get('after_clauses', []):
                    lines[-1] += f' AFTER {after["identifier"]}'
                    lines[-1] += f' FROM {self._emit_expr(after.get("from_value", {}))}'
                    lines[-1] += f' BY {self._emit_expr(after.get("by_value", {}))}'
                    lines[-1] += f' UNTIL {self._emit_expr(after.get("until_value", {}))}'
        
        lines[-1] += '.'
        return lines
    
    def _emit_if_statement(self, stmt: dict) -> List[str]:
        """Emit IF statement."""
        cond = self._emit_expr(stmt.get('condition', {}))
        lines = [f'           IF {cond}']
        
        for s in stmt.get('then_statements', []):
            lines.extend(self._emit_statement(s))
        
        if stmt.get('else_statements'):
            lines.append('           ELSE')
            for s in stmt['else_statements']:
                lines.extend(self._emit_statement(s))
        
        lines.append('           END-IF.')
        return lines
    
    def _emit_evaluate_statement(self, stmt: dict) -> List[str]:
        """Emit EVALUATE statement."""
        subjects = ' ALSO '.join(self._emit_expr(s) for s in stmt.get('subjects', []))
        if not subjects:
            subjects = 'TRUE'
        
        lines = [f'           EVALUATE {subjects}']
        
        for when in stmt.get('when_clauses', []):
            conds = ' ALSO '.join(self._emit_when_condition(c) for c in when.get('conditions', []))
            if not conds:
                conds = 'OTHER'
            lines.append(f'               WHEN {conds}')
            for s in when.get('statements', []):
                lines.extend(self._emit_statement(s))
        
        if stmt.get('when_other'):
            lines.append('               WHEN OTHER')
            for s in stmt['when_other']:
                lines.extend(self._emit_statement(s))
        
        lines.append('           END-EVALUATE.')
        return lines
    
    def _emit_when_condition(self, cond: dict) -> str:
        """Emit WHEN condition."""
        if cond.get('is_any'):
            return 'ANY'
        if cond.get('is_true'):
            return 'TRUE'
        if cond.get('is_false'):
            return 'FALSE'
        
        value = self._emit_expr(cond.get('value', {}))
        if cond.get('thru_value'):
            return f'{value} THRU {self._emit_expr(cond["thru_value"])}'
        return value
    
    def _emit_goto_statement(self, stmt: dict) -> List[str]:
        """Emit GO TO statement."""
        if stmt.get('depending_on'):
            targets = ' '.join(stmt.get('targets', []))
            return [f'           GO TO {targets} DEPENDING ON {stmt["depending_on"]}.']
        return [f'           GO TO {stmt.get("target", "")}.']
    
    def _emit_stop_statement(self, stmt: dict) -> List[str]:
        """Emit STOP statement."""
        if stmt.get('literal'):
            return [f'           STOP {self._format_literal(stmt["literal"])}.']
        return ['           STOP RUN.']
    
    # =========================================================================
    # File I/O Statements
    # =========================================================================
    
    def _emit_open_statement(self, stmt: dict) -> List[str]:
        """Emit OPEN statement."""
        lines = []
        
        # Group files by mode
        for file_entry in stmt.get('files', []):
            if isinstance(file_entry, dict):
                mode = file_entry.get('mode', 'input').upper()
                name = file_entry.get('name', '')
            else:
                # Simple list of file names
                mode = stmt.get('mode', 'INPUT').upper()
                name = file_entry
            lines.append(f'           OPEN {mode} {name}.')
        
        return lines if lines else [f'           OPEN {stmt.get("mode", "INPUT").upper()} {" ".join(stmt.get("files", []))}.']
    
    def _emit_close_statement(self, stmt: dict) -> List[str]:
        """Emit CLOSE statement."""
        files = []
        for f in stmt.get('files', []):
            if isinstance(f, dict):
                name = f.get('name', '')
                if f.get('with_lock'):
                    name += ' WITH LOCK'
            else:
                name = f
            files.append(name)
        
        return [f'           CLOSE {" ".join(files)}.']
    
    def _emit_read_statement(self, stmt: dict) -> List[str]:
        """Emit READ statement."""
        lines = [f'           READ {stmt.get("file_name", "")}']
        
        if stmt.get('next_record'):
            lines[-1] += ' NEXT RECORD'
        
        if stmt.get('into'):
            lines[-1] += f' INTO {stmt["into"]}'
        
        if stmt.get('key_is'):
            lines[-1] += f' KEY IS {stmt["key_is"]}'
        
        if stmt.get('at_end'):
            lines.append('               AT END')
            for s in stmt['at_end']:
                lines.extend(self._emit_statement(s))
        
        if stmt.get('not_at_end'):
            lines.append('               NOT AT END')
            for s in stmt['not_at_end']:
                lines.extend(self._emit_statement(s))
        
        if stmt.get('invalid_key'):
            lines.append('               INVALID KEY')
            for s in stmt['invalid_key']:
                lines.extend(self._emit_statement(s))
        
        if stmt.get('not_invalid_key'):
            lines.append('               NOT INVALID KEY')
            for s in stmt['not_invalid_key']:
                lines.extend(self._emit_statement(s))
        
        lines.append('           END-READ.')
        return lines
    
    def _emit_write_statement(self, stmt: dict) -> List[str]:
        """Emit WRITE statement."""
        lines = [f'           WRITE {stmt.get("record_name", "")}']
        
        if stmt.get('from_value'):
            lines[-1] += f' FROM {stmt["from_value"]}'
        
        if stmt.get('after_advancing'):
            adv = stmt['after_advancing']
            lines[-1] += f' AFTER ADVANCING {self._emit_advance_spec(adv)}'
        elif stmt.get('before_advancing'):
            adv = stmt['before_advancing']
            lines[-1] += f' BEFORE ADVANCING {self._emit_advance_spec(adv)}'
        
        if stmt.get('invalid_key'):
            lines.append('               INVALID KEY')
            for s in stmt['invalid_key']:
                lines.extend(self._emit_statement(s))
            lines.append('           END-WRITE.')
        else:
            lines[-1] += '.'
        
        return lines
    
    def _emit_advance_spec(self, adv: dict) -> str:
        """Emit advancing specification."""
        if adv.get('mnemonic_name'):
            return adv['mnemonic_name']
        if adv.get('identifier'):
            return f'{adv["identifier"]} LINES'
        if adv.get('lines'):
            return f'{adv["lines"]} LINES'
        return '1 LINE'
    
    def _emit_rewrite_statement(self, stmt: dict) -> List[str]:
        """Emit REWRITE statement."""
        lines = [f'           REWRITE {stmt.get("record_name", "")}']
        
        if stmt.get('from_value'):
            lines[-1] += f' FROM {stmt["from_value"]}'
        
        if stmt.get('invalid_key'):
            lines.append('               INVALID KEY')
            for s in stmt['invalid_key']:
                lines.extend(self._emit_statement(s))
            lines.append('           END-REWRITE.')
        else:
            lines[-1] += '.'
        
        return lines
    
    def _emit_delete_statement(self, stmt: dict) -> List[str]:
        """Emit DELETE statement."""
        lines = [f'           DELETE {stmt.get("file_name", "")}']
        
        if stmt.get('invalid_key'):
            lines.append('               INVALID KEY')
            for s in stmt['invalid_key']:
                lines.extend(self._emit_statement(s))
            lines.append('           END-DELETE.')
        else:
            lines[-1] += '.'
        
        return lines
    
    def _emit_start_statement(self, stmt: dict) -> List[str]:
        """Emit START statement."""
        lines = [f'           START {stmt.get("file_name", "")}']
        
        if stmt.get('key_name'):
            cond = stmt.get('key_condition', 'equal').upper().replace('-', ' ')
            cond_map = {
                'EQUAL': 'EQUAL TO',
                'GREATER': 'GREATER THAN',
                'NOT LESS': 'NOT LESS THAN',
                'LESS': 'LESS THAN',
                'NOT GREATER': 'NOT GREATER THAN',
            }
            cond = cond_map.get(cond, cond)
            lines[-1] += f' KEY IS {cond} {stmt["key_name"]}'
        
        if stmt.get('invalid_key'):
            lines.append('               INVALID KEY')
            for s in stmt['invalid_key']:
                lines.extend(self._emit_statement(s))
            lines.append('           END-START.')
        else:
            lines[-1] += '.'
        
        return lines
    
    # =========================================================================
    # String Handling
    # =========================================================================
    
    def _emit_string_statement(self, stmt: dict) -> List[str]:
        """Emit STRING statement."""
        lines = ['           STRING']
        
        for source in stmt.get('sources', []):
            val = self._emit_expr(source.get('value', {}))
            delim = source.get('delimited_by')
            if delim:
                if delim == 'SIZE':
                    lines.append(f'               {val} DELIMITED BY SIZE')
                else:
                    lines.append(f'               {val} DELIMITED BY {self._emit_expr(delim)}')
            else:
                lines.append(f'               {val}')
        
        lines.append(f'               INTO {stmt.get("into", "")}')
        
        if stmt.get('pointer'):
            lines.append(f'               WITH POINTER {stmt["pointer"]}')
        
        if stmt.get('on_overflow'):
            lines.append('               ON OVERFLOW')
            for s in stmt['on_overflow']:
                lines.extend(self._emit_statement(s))
        
        lines.append('           END-STRING.')
        return lines
    
    def _emit_unstring_statement(self, stmt: dict) -> List[str]:
        """Emit UNSTRING statement."""
        lines = [f'           UNSTRING {stmt.get("source", "")}']
        
        if stmt.get('delimiters'):
            delim_strs = []
            for d in stmt['delimiters']:
                val = self._emit_expr(d.get('value', {}))
                if d.get('all_delimiters'):
                    delim_strs.append(f'ALL {val}')
                else:
                    delim_strs.append(val)
            lines.append(f'               DELIMITED BY {" OR ".join(delim_strs)}')
        
        lines.append('               INTO')
        for field in stmt.get('into_fields', []):
            field_str = f'                   {field.get("name", "")}'
            if field.get('delimiter_in'):
                field_str += f' DELIMITER IN {field["delimiter_in"]}'
            if field.get('count_in'):
                field_str += f' COUNT IN {field["count_in"]}'
            lines.append(field_str)
        
        if stmt.get('pointer'):
            lines.append(f'               WITH POINTER {stmt["pointer"]}')
        
        if stmt.get('tallying'):
            lines.append(f'               TALLYING IN {stmt["tallying"]}')
        
        if stmt.get('on_overflow'):
            lines.append('               ON OVERFLOW')
            for s in stmt['on_overflow']:
                lines.extend(self._emit_statement(s))
        
        lines.append('           END-UNSTRING.')
        return lines
    
    def _emit_inspect_statement(self, stmt: dict) -> List[str]:
        """Emit INSPECT statement."""
        lines = [f'           INSPECT {stmt.get("identifier", "")}']
        
        if stmt.get('tallying'):
            tally = stmt['tallying']
            lines.append(f'               TALLYING {tally.get("counter", "")}')
            for item in tally.get('for_items', []):
                ttype = item.get('tally_type', 'characters').upper()
                if ttype == 'CHARACTERS':
                    lines.append(f'                   FOR CHARACTERS')
                else:
                    val = self._emit_expr(item.get('value', {}))
                    lines.append(f'                   FOR {ttype} {val}')
        
        if stmt.get('replacing'):
            replace = stmt['replacing']
            lines.append('               REPLACING')
            for item in replace.get('items', []):
                rtype = item.get('replace_type', 'all').upper()
                from_val = self._emit_expr(item.get('from_value', {}))
                to_val = self._emit_expr(item.get('to_value', {}))
                lines.append(f'                   {rtype} {from_val} BY {to_val}')
        
        if stmt.get('converting'):
            conv = stmt['converting']
            lines.append(f'               CONVERTING {self._format_literal(conv.get("from_chars", ""))}')
            lines.append(f'                   TO {self._format_literal(conv.get("to_chars", ""))}')
        
        lines[-1] += '.'
        return lines
    
    # =========================================================================
    # Display/Accept
    # =========================================================================
    
    def _emit_display_statement(self, stmt: dict) -> List[str]:
        """Emit DISPLAY statement."""
        items = ' '.join(self._emit_expr(i) for i in stmt.get('items', []))
        
        line = f'           DISPLAY {items}'
        
        if stmt.get('upon'):
            line += f' UPON {stmt["upon"]}'
        
        if stmt.get('with_no_advancing'):
            line += ' WITH NO ADVANCING'
        
        return [line + '.']
    
    def _emit_accept_statement(self, stmt: dict) -> List[str]:
        """Emit ACCEPT statement."""
        line = f'           ACCEPT {stmt.get("identifier", "")}'
        
        if stmt.get('from_source'):
            line += f' FROM {stmt["from_source"]}'
        
        return [line + '.']
    
    # =========================================================================
    # Expression Emission
    # =========================================================================
    
    def _emit_expr(self, expr: dict) -> str:
        """Emit COBOL expression."""
        if expr is None or not expr:
            return ''
        
        if isinstance(expr, str):
            return expr
        
        if isinstance(expr, (int, float)):
            return str(expr)
        
        kind = expr.get('kind', '')
        
        if kind == 'literal':
            return self._format_literal(expr.get('value'))
        
        elif kind == 'identifier':
            name = expr.get('name', '')
            
            # Handle qualifiers (OF/IN)
            for qual in expr.get('qualifiers', []):
                name += f' OF {qual}'
            
            # Handle subscripts
            if expr.get('subscripts'):
                subs = ', '.join(self._emit_expr(s) for s in expr['subscripts'])
                name += f'({subs})'
            
            # Handle reference modification
            if expr.get('reference_mod'):
                rm = expr['reference_mod']
                start = self._emit_expr(rm.get('start', {}))
                length = self._emit_expr(rm.get('length', {})) if rm.get('length') else ''
                name += f'({start}:{length})'
            
            return name
        
        elif kind == 'binary_expr':
            left = self._emit_expr(expr.get('left', {}))
            right = self._emit_expr(expr.get('right', {}))
            op = self._map_operator(expr.get('op', ''))
            return f'{left} {op} {right}'
        
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
            name = expr.get('name', '').upper()
            args = ', '.join(self._emit_expr(a) for a in expr.get('arguments', []))
            return f'FUNCTION {name}({args})'
        
        else:
            return str(expr)
    
    def _map_operator(self, op: str) -> str:
        """Map operator to COBOL syntax."""
        op_map = {
            'add': '+',
            'sub': '-',
            'mul': '*',
            'div': '/',
            'pow': '**',
            'and': 'AND',
            'or': 'OR',
            '+': '+',
            '-': '-',
            '*': '*',
            '/': '/',
            '**': '**',
        }
        return op_map.get(op.lower(), op.upper())
    
    def _map_condition_op(self, op: str) -> str:
        """Map condition operator to COBOL syntax."""
        op_map = {
            'eq': '=',
            'ne': 'NOT =',
            'lt': '<',
            'le': '<=',
            'gt': '>',
            'ge': '>=',
            '=': '=',
            '<>': 'NOT =',
            '<': '<',
            '<=': '<=',
            '>': '>',
            '>=': '>=',
        }
        return op_map.get(op.lower(), op)
    
    def _format_literal(self, value: Any) -> str:
        """Format literal value for COBOL."""
        if value is None:
            return 'SPACES'
        
        if isinstance(value, str):
            # Handle figurative constants
            fig_constants = {
                'ZERO': 'ZERO', 'ZEROS': 'ZEROS', 'ZEROES': 'ZEROES',
                'SPACE': 'SPACE', 'SPACES': 'SPACES',
                'HIGH-VALUE': 'HIGH-VALUE', 'HIGH-VALUES': 'HIGH-VALUES',
                'LOW-VALUE': 'LOW-VALUE', 'LOW-VALUES': 'LOW-VALUES',
                'QUOTE': 'QUOTE', 'QUOTES': 'QUOTES',
                'NULL': 'NULL', 'NULLS': 'NULLS',
            }
            if value.upper() in fig_constants:
                return fig_constants[value.upper()]
            
            # Escape quotes
            escaped = value.replace('"', '""')
            return f'"{escaped}"'
        
        if isinstance(value, bool):
            return 'TRUE' if value else 'FALSE'
        
        if isinstance(value, (int, float)):
            return str(value)
        
        return str(value)
    
    # =========================================================================
    # Manifest Generation
    # =========================================================================
    
    def _generate_manifest(self, ir: dict, code: str) -> dict:
        """Generate manifest for emitted code."""
        return {
            'schema': 'stunir.codegen.cobol.v1',
            'timestamp': int(time.time()),
            'program_name': ir.get('name', 'UNNAMED'),
            'dialect': self.dialect,
            'cobol_version': self.COBOL_VERSION,
            'code_hash': compute_sha256(code),
            'code_lines': len(code.split('\n')),
            'files_count': len(ir.get('files', [])),
            'paragraphs_count': len(ir.get('paragraphs', [])),
            'data_items_count': len(ir.get('data_items', [])),
            'errors': self._errors,
            'warnings': self._warnings,
        }


# =============================================================================
# Main entry point for testing
# =============================================================================

def main():
    """Test the COBOL emitter."""
    # Sample payroll program IR
    ir = {
        'name': 'PAYROLL',
        'author': 'STUNIR',
        'files': [
            {
                'name': 'EMPLOYEE-FILE',
                'assign_to': 'EMPLOYEE.DAT',
                'organization': 'indexed',
                'access': 'sequential',
                'record_key': 'EMP-ID',
                'file_status': 'WS-FILE-STATUS',
                'record': {
                    'name': 'EMPLOYEE-RECORD',
                    'level': 1,
                    'children': [
                        {'name': 'EMP-ID', 'level': 5, 'picture': {'pattern': '9(5)'}},
                        {'name': 'EMP-NAME', 'level': 5, 'picture': {'pattern': 'X(30)'}},
                        {'name': 'EMP-SALARY', 'level': 5, 'picture': {'pattern': '9(7)V99'}},
                        {'name': 'EMP-DEPT', 'level': 5, 'picture': {'pattern': 'X(10)'}},
                    ]
                }
            }
        ],
        'data_items': [
            {'name': 'WS-FILE-STATUS', 'level': 1, 'picture': {'pattern': 'XX'}},
            {'name': 'WS-EOF', 'level': 1, 'picture': {'pattern': '9'}, 'value': 0},
            {'name': 'WS-TOTAL-SALARY', 'level': 1, 'picture': {'pattern': '9(10)V99'}, 'value': 0},
        ],
        'paragraphs': [
            {
                'name': 'MAIN-PARA',
                'statements': [
                    {'kind': 'open_statement', 'files': [{'name': 'EMPLOYEE-FILE', 'mode': 'input'}]},
                    {'kind': 'perform_statement', 'paragraph_name': 'PROCESS-RECORDS',
                     'until': {'kind': 'condition', 'left': {'kind': 'identifier', 'name': 'WS-EOF'},
                               'op': '=', 'right': {'kind': 'literal', 'value': 1}}},
                    {'kind': 'close_statement', 'files': ['EMPLOYEE-FILE']},
                    {'kind': 'display_statement', 'items': [
                        {'kind': 'literal', 'value': 'TOTAL SALARY: '},
                        {'kind': 'identifier', 'name': 'WS-TOTAL-SALARY'}
                    ]},
                    {'kind': 'stop_statement', 'run': True}
                ]
            },
            {
                'name': 'PROCESS-RECORDS',
                'statements': [
                    {'kind': 'read_statement', 'file_name': 'EMPLOYEE-FILE',
                     'at_end': [
                         {'kind': 'move_statement', 'source': {'kind': 'literal', 'value': 1},
                          'destinations': ['WS-EOF']}
                     ],
                     'not_at_end': [
                         {'kind': 'add_statement', 'values': [{'kind': 'identifier', 'name': 'EMP-SALARY'}],
                          'to_value': 'WS-TOTAL-SALARY'}
                     ]}
                ]
            }
        ]
    }
    
    emitter = COBOLEmitter()
    result = emitter.emit(ir)
    print(result.code)
    print('\n--- Manifest ---')
    print(json.dumps(result.manifest, indent=2))


if __name__ == '__main__':
    main()
