#!/usr/bin/env python3
"""
===============================================================================
STUNIR IR to Code Emitter - Python REFERENCE Implementation
===============================================================================

WARNING: This is a REFERENCE IMPLEMENTATION for readability purposes only.
         DO NOT use this file for production, verification, or safety-critical
         applications.

PRIMARY IMPLEMENTATION: Ada SPARK
    Location: tools/spark/bin/stunir_ir_to_code_main
    Build:    cd tools/spark && gprbuild -P stunir_tools.gpr

This Python version exists to:
1. Provide a readable reference for understanding the algorithm
2. Serve as a fallback when Ada SPARK tools are not available
3. Enable quick prototyping and testing

For all production use cases, use the Ada SPARK implementation which provides:
- Formal verification guarantees
- Deterministic execution
- DO-178C compliance support
- Absence of runtime errors (proven via SPARK)

===============================================================================

This tool reads a STUNIR Canonical IR (JSON) and a template pack directory,
and generates deterministic source code for a target language.

Design principles:
- Determinism: output depends ONLY on IR + template pack files.
- Hermeticity: stdlib only; no network.

NOTE:
This is a minimal emitter used for bringing up new language targets.
It emits stubs (not full semantics) for low-level raw targets.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [ir_to_code] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

VERSION = "0.2.0"
TOOL_ID = "stunir_ir_to_code"

class TemplateError(Exception):
    pass

class Context:
    def __init__(self, data: Dict[str, Any], parent: Optional['Context'] = None):
        self.data = data
        self.parent = parent

    def get(self, key: str) -> Any:
        if key in self.data:
            return self.data[key]
        if self.parent:
            return self.parent.get(key)
        return None

def _resolve_token(value: Any, token: str) -> Any:
    # Support simple dotted lookups and optional list indexing token like name[0]
    # Examples: t.name, fn.args[0].name
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(\[(\d+)\])?$", token)
    if not m:
        return None
    key = m.group(1)
    idx = m.group(3)

    if isinstance(value, dict):
        value = value.get(key)
    else:
        # allow attribute access for simple objects
        value = getattr(value, key, None)

    if idx is not None:
        if isinstance(value, list):
            i = int(idx)
            if 0 <= i < len(value):
                return value[i]
            return None
        return None

    return value

def resolve_var(ctx: Context, expr: str) -> Any:
    expr = expr.strip()
    if not expr:
        return None

    parts = expr.split('.')
    base = ctx.get(parts[0])
    if base is None:
        return None

    cur = base
    for token in parts[1:]:
        cur = _resolve_token(cur, token)
        if cur is None:
            return None
    return cur

class Node:
    def render(self, ctx: Context) -> str:
        raise NotImplementedError

class TextNode(Node):
    def __init__(self, text: str):
        self.text = text
    def render(self, ctx: Context) -> str:
        return self.text

class VarNode(Node):
    def __init__(self, var_expr: str):
        self.var_expr = var_expr.strip()
    def render(self, ctx: Context) -> str:
        val = resolve_var(ctx, self.var_expr)
        return str(val) if val is not None else ""

class ForNode(Node):
    def __init__(self, loop_var: str, list_expr: str, body: List[Node]):
        self.loop_var = loop_var.strip()
        self.list_expr = list_expr.strip()
        self.body = body
    def render(self, ctx: Context) -> str:
        items = resolve_var(ctx, self.list_expr)
        if not isinstance(items, list):
            return ""
        out: List[str] = []
        for i, item in enumerate(items):
            inner = Context({
                self.loop_var: item,
                "loop": {"index": i, "first": i == 0, "last": i == len(items) - 1}
            }, ctx)
            for node in self.body:
                out.append(node.render(inner))
        return "".join(out)

class IfNode(Node):
    def __init__(self, cond_expr: str, true_body: List[Node], false_body: List[Node]):
        self.cond_expr = cond_expr.strip()
        self.true_body = true_body
        self.false_body = false_body
    def render(self, ctx: Context) -> str:
        val = resolve_var(ctx, self.cond_expr)
        body = self.true_body if val else self.false_body
        return "".join(n.render(ctx) for n in body)

def parse_template(template_str: str) -> List[Node]:
    token_re = re.compile(r"(?s)({{.*?}}|{%.*?%})")
    parts = token_re.split(template_str)

    root: List[Node] = []
    stack: List[Dict[str, Any]] = []

    def current_list() -> List[Node]:
        if not stack:
            return root
        top = stack[-1]
        if top['type'] == 'for':
            return top['nodes']
        if top['type'] == 'if':
            return top['true_nodes'] if top['current'] == 'true' else top['false_nodes']
        if top['type'] == 'comment':
            return []
        return root

    for part in parts:
        if not part:
            continue

        if part.startswith('{{') and part.endswith('}}'):
            if stack and stack[-1]['type'] == 'comment':
                continue
            expr = part[2:-2]
            current_list().append(VarNode(expr))
            continue

        if part.startswith('{%') and part.endswith('%}'):
            content = part[2:-2].strip()
            if not content:
                continue
            tokens = content.split()
            tag = tokens[0]

            if tag == 'comment':
                stack.append({'type': 'comment'})
                continue
            if stack and stack[-1]['type'] == 'comment':
                if tag == 'endcomment':
                    stack.pop()
                continue

            if tag == 'for':
                # {% for x in list %}
                if len(tokens) != 4 or tokens[2] != 'in':
                    raise TemplateError(f"Malformed for: {content}")
                stack.append({'type': 'for', 'nodes': [], 'loop_var': tokens[1], 'list_expr': tokens[3]})
                continue

            if tag == 'endfor':
                if not stack or stack[-1]['type'] != 'for':
                    raise TemplateError('Unmatched endfor')
                blk = stack.pop()
                current_list().append(ForNode(blk['loop_var'], blk['list_expr'], blk['nodes']))
                continue

            if tag == 'if':
                # {% if cond %}
                if len(tokens) != 2:
                    raise TemplateError(f"Malformed if: {content}")
                stack.append({'type': 'if', 'cond_expr': tokens[1], 'true_nodes': [], 'false_nodes': [], 'current': 'true'})
                continue

            if tag == 'else':
                if not stack or stack[-1]['type'] != 'if':
                    raise TemplateError('Unmatched else')
                stack[-1]['current'] = 'false'
                continue

            if tag == 'endif':
                if not stack or stack[-1]['type'] != 'if':
                    raise TemplateError('Unmatched endif')
                blk = stack.pop()
                current_list().append(IfNode(blk['cond_expr'], blk['true_nodes'], blk['false_nodes']))
                continue

            raise TemplateError(f"Unknown tag: {tag}")

        # plain text
        if stack and stack[-1]['type'] == 'comment':
            continue
        current_list().append(TextNode(part))

    if stack:
        raise TemplateError(f"Unclosed block: {stack[-1]['type']}")

    return root

def render_template(template_str: str, context_data: Dict[str, Any]) -> str:
    nodes = parse_template(template_str)
    ctx = Context(context_data)
    return ''.join(n.render(ctx) for n in nodes)

# --- Helpers ---

def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

TypeRef = Union[str, Dict[str, Any]]

def is_void_type(tr: TypeRef) -> bool:
    if isinstance(tr, str):
        return tr == 'void'
    if isinstance(tr, dict) and tr.get('kind') == 'builtin':
        return tr.get('builtin') == 'void'
    return False

def builtin_name(tr: TypeRef) -> Optional[str]:
    if isinstance(tr, str):
        return tr
    if isinstance(tr, dict) and tr.get('kind') == 'builtin':
        return tr.get('builtin')
    return None

def named_name(tr: TypeRef) -> Optional[str]:
    if isinstance(tr, dict) and tr.get('kind') == 'named':
        return tr.get('name')
    if isinstance(tr, str):
        # legacy named types are indistinguishable; treat non-builtin as named
        if tr not in {'string','int','float','bool','bytes','any','void'}:
            return tr
    return None

def c_type(tr: TypeRef) -> str:
    # Handle string types directly
    if isinstance(tr, str):
        # Map primitive types
        type_map = {
            'string': 'const char*',
            'int': 'int64_t',
            'float': 'double',
            'bool': 'bool',
            'bytes': 'const uint8_t*',
            'byte[]': 'const uint8_t*',  # Fixed: handle byte[] type
            'any': 'void*',
            'void': 'void',
            # Fixed-width integer types
            'i8': 'int8_t',
            'i16': 'int16_t',
            'i32': 'int32_t',
            'i64': 'int64_t',
            'u8': 'uint8_t',
            'u16': 'uint16_t',
            'u32': 'uint32_t',
            'u64': 'uint64_t',
            # Floating point types
            'f32': 'float',
            'f64': 'double',
        }
        
        if tr in type_map:
            return type_map[tr]
        
        # Not a builtin, treat as named type
        return f'struct {tr}'
    
    # Handle dict-based type refs
    b = builtin_name(tr)
    if b == 'string':
        return 'const char*'
    if b == 'int':
        return 'int64_t'
    if b == 'float':
        return 'double'
    if b == 'bool':
        return 'bool'
    if b == 'bytes':
        return 'const uint8_t*'
    if b == 'any':
        return 'void*'
    if b == 'void':
        return 'void'

    # structured typerefs (optional/list/map) => coarse placeholder
    if isinstance(tr, dict):
        k = tr.get('kind')
        if k == 'optional':
            inner = tr.get('inner')
            inner_c = c_type(inner) if inner is not None else 'void*'
            if inner_c == 'void':
                return 'void*'
            return f'{inner_c}*'
        if k in {'list','map'}:
            return 'void*'

    nn = named_name(tr)
    if nn:
        return f'struct {nn}'

    return 'void*'

def c_default_return(tr: TypeRef) -> str:
    # Handle string types directly
    if isinstance(tr, str):
        # Integer types
        if tr in ('i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64', 'int'):
            return '0'
        # Float types
        if tr in ('f32', 'f64', 'float'):
            return '0.0'
        # Bool type
        if tr == 'bool':
            return 'false'
        # Pointer types
        if tr in ('string', 'bytes', 'any'):
            return 'NULL'
        # Void type
        if tr == 'void':
            return ''
        # Named types
        return f'(struct {tr}){{0}}'
    
    # Handle dict-based type refs
    b = builtin_name(tr)
    if b == 'int':
        return '0'
    if b == 'float':
        return '0.0'
    if b == 'bool':
        return 'false'
    if b in {'string','bytes','any'}:
        return 'NULL'
    nn = named_name(tr)
    if nn:
        return f'(struct {nn}){{0}}'
    return '0'

def rs_type(tr: TypeRef) -> str:
    # Handle string types directly
    if isinstance(tr, str):
        type_map = {
            'string': 'String',
            'int': 'i64',
            'float': 'f64',
            'bool': 'bool',
            'bytes': 'Vec<u8>',
            'any': 'String',
            'void': '()',
            # Fixed-width integer types
            'i8': 'i8',
            'i16': 'i16',
            'i32': 'i32',
            'i64': 'i64',
            'u8': 'u8',
            'u16': 'u16',
            'u32': 'u32',
            'u64': 'u64',
            # Floating point types
            'f32': 'f32',
            'f64': 'f64',
        }
        
        if tr in type_map:
            return type_map[tr]
        
        # Not a builtin, treat as named type
        return tr
    
    # Handle dict-based type refs
    b = builtin_name(tr)
    if b == 'string':
        return 'String'
    if b == 'int':
        return 'i64'
    if b == 'float':
        return 'f64'
    if b == 'bool':
        return 'bool'
    if b == 'bytes':
        return 'Vec<u8>'
    if b == 'any':
        return 'String'
    if b == 'void':
        return '()'

    if isinstance(tr, dict):
        k = tr.get('kind')
        if k == 'optional':
            inner = tr.get('inner')
            return f'Option<{rs_type(inner)}>' if inner is not None else 'Option<()>'
        if k == 'list':
            items = tr.get('items')
            return f'Vec<{rs_type(items)}>' if items is not None else 'Vec<()>'
        if k == 'map':
            key = tr.get('key')
            val = tr.get('value')
            kt = rs_type(key) if key is not None else '()'
            vt = rs_type(val) if val is not None else '()'
            return f'std::collections::BTreeMap<{kt}, {vt}>'

    nn = named_name(tr)
    if nn:
        return nn

    return '()'

def wat_type(tr: TypeRef) -> Optional[str]:
    # Return None for void/no-result
    if is_void_type(tr):
        return None
    b = builtin_name(tr)
    if b == 'int':
        return 'i64'
    if b == 'float':
        return 'f64'
    if b == 'bool':
        return 'i32'
    # string/bytes/any/named/compound => pointer/handle
    return 'i32'

def wat_default_instr(wt: Optional[str]) -> str:
    if wt is None:
        return ''
    if wt == 'i64':
        return '(i64.const 0)'
    if wt == 'f64':
        return '(f64.const 0)'
    return '(i32.const 0)'


def infer_c_type_from_value(value: str) -> str:
    """
    Infer C type from a value string.
    Simple heuristic-based type inference.
    """
    value = str(value).strip()
    
    # Boolean values
    if value in ('true', 'false'):
        return 'bool'
    
    # Floating point
    if '.' in value:
        return 'double'
    
    # Negative integer
    if value.startswith('-') and value[1:].isdigit():
        return 'int32_t'
    
    # Positive integer
    if value.isdigit():
        num = int(value)
        if num <= 255:
            return 'uint8_t'  # Small positive integers
        else:
            return 'int32_t'
    
    # Default to int32_t for complex expressions
    return 'int32_t'

def translate_steps_to_c(steps: List[Dict[str, Any]], ret_type: TypeRef, indent: int = 1) -> str:
    """
    Translate IR steps to C code.
    
    Supported step operations:
    - assign: Variable assignment (target = value)
    - return: Return statement
    - call: Function call
    - nop: No operation (comment only)
    - if: If/else statement (condition, then_block, optional else_block)
    - while: While loop (condition, body)
    - for: For loop (init, condition, increment, body)
    - break: Break statement (v0.9.0)
    - continue: Continue statement (v0.9.0)
    - switch: Switch/case statement (expr, cases, optional default) (v0.9.0)
    
    Args:
        steps: List of step dictionaries from IR
        ret_type: Return type of the function
        indent: Indentation level (default 1 for function body)
        
    Returns:
        String containing C code for the function body
    """
    if not steps:
        # Empty function body
        c_ret = c_type(ret_type)
        indent_str = '  ' * indent
        if c_ret == 'void':
            return f'{indent_str}/* Empty function body */\n{indent_str}return;'
        else:
            return f'{indent_str}/* Empty function body */\n{indent_str}return {c_default_return(ret_type)};'
    
    lines = []
    local_vars = {}  # Track declared variables with their types
    indent_str = '  ' * indent
    
    for step in steps:
        if not isinstance(step, dict):
            continue
            
        op = step.get('op')
        
        if op == 'assign':
            target = step.get('target', '')
            value = step.get('value', '0')
            
            # Track local variable with type inference
            if target and target not in local_vars:
                var_type = infer_c_type_from_value(value)
                local_vars[target] = var_type
                lines.append(f'{indent_str}{var_type} {target} = {value};')
            else:
                lines.append(f'{indent_str}{target} = {value};')
                
        elif op == 'return':
            value = step.get('value', '')
            c_ret = c_type(ret_type)
            
            if value:
                # Handle struct return types
                if c_ret.startswith('struct '):
                    # Check if value looks like a compound literal (TypeName{...})
                    if '{' in str(value) and '}' in str(value):
                        # Extract the brace-enclosed part
                        # Pattern: TypeName{fields}
                        import re
                        match = re.match(r'(\w+)\{(.+)\}', str(value))
                        if match:
                            # Just use the brace part with proper C99 syntax
                            fields = match.group(2)
                            lines.append(f'{indent_str}return ({c_ret}){{{fields}}};')
                        else:
                            # Fallback: wrap as-is
                            lines.append(f'{indent_str}return ({c_ret}){value};')
                    else:
                        lines.append(f'{indent_str}return {value};')
                else:
                    lines.append(f'{indent_str}return {value};')
            else:
                lines.append(f'{indent_str}return;')
                
        elif op == 'call':
            # Get the function call expression from value field
            # Format: "function_name(arg1, arg2, ...)"
            call_expr = step.get('value', 'unknown()')
            target = step.get('target')  # Optional assignment target
            
            if target:
                # Call with assignment
                if target not in local_vars:
                    # Default to int32_t for function return values
                    local_vars[target] = 'int32_t'
                    lines.append(f'{indent_str}int32_t {target} = {call_expr};')
                else:
                    lines.append(f'{indent_str}{target} = {call_expr};')
            else:
                # Call without assignment
                lines.append(f'{indent_str}{call_expr};')
                
        elif op == 'nop':
            # No operation - just a comment
            lines.append(f'{indent_str}/* nop */')
            
        elif op == 'if':
            # If/else statement
            condition = step.get('condition', 'true')
            then_block = step.get('then_block', [])
            else_block = step.get('else_block', [])
            
            lines.append(f'{indent_str}if ({condition}) {{')
            # Recursively translate then_block with increased indentation
            then_body = translate_steps_to_c(then_block, ret_type, indent + 1)
            lines.append(then_body)
            
            if else_block:
                lines.append(f'{indent_str}}} else {{')
                # Recursively translate else_block with increased indentation
                else_body = translate_steps_to_c(else_block, ret_type, indent + 1)
                lines.append(else_body)
            
            lines.append(f'{indent_str}}}')
            
        elif op == 'while':
            # While loop
            condition = step.get('condition', 'true')
            body = step.get('body', [])
            
            lines.append(f'{indent_str}while ({condition}) {{')
            # Recursively translate body with increased indentation
            loop_body = translate_steps_to_c(body, ret_type, indent + 1)
            lines.append(loop_body)
            lines.append(f'{indent_str}}}')
            
        elif op == 'for':
            # For loop
            init = step.get('init', '')
            condition = step.get('condition', 'true')
            increment = step.get('increment', '')
            body = step.get('body', [])
            
            lines.append(f'{indent_str}for ({init}; {condition}; {increment}) {{')
            # Recursively translate body with increased indentation
            loop_body = translate_steps_to_c(body, ret_type, indent + 1)
            lines.append(loop_body)
            lines.append(f'{indent_str}}}')
            
        elif op == 'break':
            # v0.9.0: Break statement
            lines.append(f'{indent_str}break;')
            
        elif op == 'continue':
            # v0.9.0: Continue statement
            lines.append(f'{indent_str}continue;')
            
        elif op == 'switch':
            # v0.9.0: Switch/case statement
            expr = step.get('expr', '0')
            cases = step.get('cases', [])
            default = step.get('default', [])
            
            lines.append(f'{indent_str}switch ({expr}) {{')
            
            # Generate case labels
            for case in cases:
                case_value = case.get('value', 0)
                case_body = case.get('body', [])
                
                lines.append(f'{indent_str}  case {case_value}:')
                # Recursively translate case body with increased indentation
                case_code = translate_steps_to_c(case_body, ret_type, indent + 2)
                lines.append(case_code)
            
            # Generate default case if present
            if default:
                lines.append(f'{indent_str}  default:')
                default_code = translate_steps_to_c(default, ret_type, indent + 2)
                lines.append(default_code)
            
            lines.append(f'{indent_str}}}')
            
        elif op == 'try':
            # v0.8.7: try/catch/finally exception handling
            # Note: C doesn't have native exception handling; use setjmp/longjmp pattern
            try_block = step.get('try_block', [])
            catch_blocks = step.get('catch_blocks', [])
            finally_block = step.get('finally_block', [])
            
            lines.append(f'{indent_str}/* BEGIN TRY-CATCH BLOCK */')
            lines.append(f'{indent_str}{{')
            lines.append(f'{indent_str}  jmp_buf __stunir_exception_buf;')
            lines.append(f'{indent_str}  int __stunir_exception_code = 0;')
            lines.append(f'{indent_str}  if ((__stunir_exception_code = setjmp(__stunir_exception_buf)) == 0) {{')
            
            # Try block
            lines.append(f'{indent_str}    /* TRY */')
            try_body = translate_steps_to_c(try_block, ret_type, indent + 2)
            lines.append(try_body)
            
            lines.append(f'{indent_str}  }} else {{')
            lines.append(f'{indent_str}    /* CATCH */')
            
            # Catch blocks
            if catch_blocks:
                for i, catch in enumerate(catch_blocks):
                    exc_type = catch.get('exception_type', '*')
                    exc_var = catch.get('exception_var', 'e')
                    catch_body = catch.get('body', [])
                    
                    if exc_type == '*':
                        # Catch all
                        lines.append(f'{indent_str}    /* catch (all) */')
                        lines.append(f'{indent_str}    int {exc_var} = __stunir_exception_code;')
                    else:
                        # Specific exception type (encoded as integer)
                        lines.append(f'{indent_str}    /* catch ({exc_type}) */')
                        lines.append(f'{indent_str}    int {exc_var} = __stunir_exception_code;')
                    
                    catch_code = translate_steps_to_c(catch_body, ret_type, indent + 2)
                    lines.append(catch_code)
            else:
                lines.append(f'{indent_str}    /* No catch handlers */')
            
            lines.append(f'{indent_str}  }}')
            
            # Finally block
            if finally_block:
                lines.append(f'{indent_str}  /* FINALLY */')
                finally_code = translate_steps_to_c(finally_block, ret_type, indent + 1)
                lines.append(finally_code)
            
            lines.append(f'{indent_str}}}')
            lines.append(f'{indent_str}/* END TRY-CATCH BLOCK */')
            
        elif op == 'throw':
            # v0.8.7: throw exception
            exc_type = step.get('exception_type', 'Exception')
            exc_msg = step.get('exception_message', '')
            
            # In C, we use longjmp with an error code
            lines.append(f'{indent_str}/* throw {exc_type}: {exc_msg} */')
            lines.append(f'{indent_str}longjmp(__stunir_exception_buf, 1);')
            
        else:
            # Unknown operation
            lines.append(f'{indent_str}/* UNKNOWN OP: {op} */')
    
    # If no return statement was generated, add a default one
    # Only add default return at the top level (indent == 1)
    if indent == 1 and not any('return' in line for line in lines):
        c_ret = c_type(ret_type)
        if c_ret == 'void':
            lines.append(f'{indent_str}return;')
        else:
            lines.append(f'{indent_str}return {c_default_return(ret_type)};')
    
    return '\n'.join(lines)

def build_render_context(ir: Dict[str, Any], lang: str) -> Dict[str, Any]:
    types_in = ir.get('types') or []
    fns_in = ir.get('functions') or []

    types_render = []
    for t in types_in:
        name = t.get('name') if isinstance(t, dict) else None
        if not name:
            continue
        fields_render = []
        for f in t.get('fields', []):
            if not isinstance(f, dict):
                continue
            fnm = f.get('name')
            fty = f.get('type')
            if lang == 'c':
                fields_render.append({'name': fnm, 'c_type': c_type(fty)})
            elif lang == 'rust':
                fields_render.append({'name': fnm, 'rs_type': rs_type(fty)})
            else:
                fields_render.append({'name': fnm})
        types_render.append({'name': name, 'fields': fields_render})

    functions_render = []
    for fn in fns_in:
        if not isinstance(fn, dict):
            continue
        fn_name = fn.get('name')
        args = fn.get('args') or []
        ret = fn.get('return_type')
        if not fn_name:
            continue

        if lang == 'c':
            c_args_parts = []
            for a in args:
                if not isinstance(a, dict):
                    continue
                an = a.get('name')
                at = a.get('type')
                c_args_parts.append(f"{c_type(at)} {an}")
            c_args = ', '.join(c_args_parts) if c_args_parts else 'void'
            c_ret = c_type(ret)
            c_ret_void = (c_ret == 'void')
            
            # Generate function body from steps
            steps = fn.get('steps', [])
            c_body = translate_steps_to_c(steps, ret)
            
            functions_render.append({
                'name': fn_name,
                'c_args': c_args,
                'c_ret': c_ret,
                'c_ret_void': c_ret_void,
                'c_ret_default': c_default_return(ret),
                'c_body': c_body,  # Add generated body
            })

        elif lang == 'rust':
            rs_args_parts = []
            for a in args:
                if not isinstance(a, dict):
                    continue
                an = a.get('name')
                at = a.get('type')
                rs_args_parts.append(f"{an}: {rs_type(at)}")
            rs_args = ', '.join(rs_args_parts)
            functions_render.append({
                'name': fn_name,
                'rs_args': rs_args,
                'rs_ret': rs_type(ret),
            })

        elif lang == 'wasm':
            params = []
            for a in args:
                if not isinstance(a, dict):
                    continue
                an = a.get('name')
                at = a.get('type')
                wt = wat_type(at) or 'i32'
                params.append(f" (param ${an} {wt})")
            rt = wat_type(ret)
            wat_params = ''.join(params)
            wat_results = f" (result {rt})" if rt else ''
            body = wat_default_instr(rt)
            functions_render.append({
                'name': fn_name,
                'wat_params': wat_params,
                'wat_results': wat_results,
                'wat_body': body,
            })

        elif lang == 'python':
            py_args = ', '.join(a.get('name', '') for a in args if isinstance(a, dict))
            functions_render.append({
                'name': fn_name,
                'py_args': py_args,
                'args': args,
            })
        
        elif lang == 'javascript':
            js_args = ', '.join(a.get('name', '') for a in args if isinstance(a, dict))
            functions_render.append({
                'name': fn_name,
                'js_args': js_args,
                'args': args,
            })
        
        else:
            # asm or unknown: minimal
            functions_render.append({'name': fn_name, 'args': args})

    ctx = {
        'module_name': ir.get('module_name', 'output'),
        'docstring': ir.get('docstring', ''),
        'types_render': types_render,
        'functions_render': functions_render,
        'tool': {'name': TOOL_ID, 'version': VERSION, 'lang': lang}
    }
    return ctx

def main() -> None:
    p = argparse.ArgumentParser(description='STUNIR IR to Code Emitter')
    p.add_argument('--ir', required=True, help='Path to IR JSON file')
    p.add_argument('--lang', required=True, help='Target language (python/rust/javascript/c/asm/wasm)')
    p.add_argument('--templates', required=True, help='Path to template pack directory')
    p.add_argument('--out', required=True, help='Output directory')
    p.add_argument('--emit-receipt', action='store_true', help='Generate stunir.emit.v1.json')

    args = p.parse_args()

    try:
        ir_data = load_json(args.ir)
    except Exception as e:
        logger.error(f'Error loading IR: {e}')
        sys.exit(4)

    tpl_path = os.path.join(args.templates, 'module.template')
    if not os.path.exists(tpl_path):
        logger.error(f'Missing module.template in {args.templates}')
        sys.exit(3)

    with open(tpl_path, 'r', encoding='utf-8') as f:
        tpl_content = f.read()

    # Prepare context
    context = build_render_context(ir_data, args.lang)

    try:
        rendered_code = render_template(tpl_content, context)
    except TemplateError as e:
        logger.error(f'Template rendering failed: {e}')
        sys.exit(5)

    os.makedirs(args.out, exist_ok=True)

    ext_map = {
        'python': '.py',
        'rust': '.rs',
        'javascript': '.js',
        'c': '.c',
        'asm': '.s',
        'wasm': '.wat',
    }
    ext = ext_map.get(args.lang, '.txt')

    out_filename = str(ir_data.get('module_name', 'output')) + ext
    out_path = os.path.join(args.out, out_filename)

    with open(out_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(rendered_code)

    logger.info(f'Generated: {out_path}')

    if args.emit_receipt:
        receipt = {
            'tool': TOOL_ID,
            'version': VERSION,
            'inputs': {
                'ir_sha256': sha256_file(args.ir),
                'template_sha256': sha256_file(tpl_path),
            },
            'outputs': {
                out_filename: sha256_file(out_path)
            }
        }
        receipt_path = os.path.join(args.out, 'stunir.emit.v1.json')
        with open(receipt_path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(receipt, f, indent=2, sort_keys=True)
            f.write('\n')
        logger.info(f'Receipt: {receipt_path}')

if __name__ == '__main__':
    main()