#!/usr/bin/env python3
"""
STUNIR IR to Code Generator - Unified CLI with Semantic IR Integration

Integrates with semantic IR emitters from tools/semantic_ir/emitters/
Supports both general-purpose languages and specialized emitters.

PRIMARY IMPLEMENTATION: Ada SPARK
    Location: tools/spark/bin/stunir_ir_to_code_main
    Build:    cd tools/spark && gprbuild -P stunir_tools.gpr

This Python version provides:
1. Unified CLI for all emitters (general + specialized)
2. General-purpose language generation (C, Rust, Python, etc.)
3. Specialized emitter support (GPU, WASM, Assembly, etc.)
4. Fallback when Ada SPARK tools are not available

Usage:
    # General-purpose languages
    python tools/ir_to_code.py --ir input.json --lang c --out output/
    python tools/ir_to_code.py --ir input.json --lang rust --out output/
    
    # Specialized emitters
    python tools/ir_to_code.py --ir input.json --emitter gpu --out output/
    python tools/ir_to_code.py --ir input.json --emitter wasm --out output/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add tools directory to path for semantic_ir imports
sys.path.insert(0, str(Path(__file__).parent))

# Import semantic IR types
try:
    from semantic_ir.emitters.types import (
        IRModule, IRFunction, IRParameter, IRStatement, IRType, IRTypeField,
        IRDataType, IRStatementType, GeneratedFile
    )
    from semantic_ir.emitters.base_emitter import (
        BaseEmitter, EmitterConfig, EmitterResult, EmitterStatus
    )
    SEMANTIC_IR_AVAILABLE = True
except ImportError as e:
    SEMANTIC_IR_AVAILABLE = False
    print(f"Warning: semantic_ir not available: {e}", file=sys.stderr)


# Registry of specialized emitters
EMITTER_REGISTRY: Dict[str, Any] = {}


def register_specialized_emitters():
    """Register all available specialized emitters."""
    global EMITTER_REGISTRY
    
    if not SEMANTIC_IR_AVAILABLE:
        return
    
    # Core emitters
    try:
        from semantic_ir.emitters.core.assembly import AssemblyEmitter, AssemblyEmitterConfig
        EMITTER_REGISTRY["assembly"] = (AssemblyEmitter, AssemblyEmitterConfig)
        EMITTER_REGISTRY["asm"] = (AssemblyEmitter, AssemblyEmitterConfig)
    except ImportError:
        pass
    
    try:
        from semantic_ir.emitters.core.embedded import EmbeddedEmitter, EmbeddedEmitterConfig
        EMITTER_REGISTRY["embedded"] = (EmbeddedEmitter, EmbeddedEmitterConfig)
    except ImportError:
        pass
    
    try:
        from semantic_ir.emitters.core.gpu import GPUEmitter, GPUEmitterConfig
        EMITTER_REGISTRY["gpu"] = (GPUEmitter, GPUEmitterConfig)
    except ImportError:
        pass
    
    try:
        from semantic_ir.emitters.core.wasm import WebAssemblyEmitter, WebAssemblyEmitterConfig
        EMITTER_REGISTRY["wasm"] = (WebAssemblyEmitter, WebAssemblyEmitterConfig)
        EMITTER_REGISTRY["webassembly"] = (WebAssemblyEmitter, WebAssemblyEmitterConfig)
    except ImportError:
        pass
    
    try:
        from semantic_ir.emitters.core.polyglot import PolyglotEmitter, PolyglotEmitterConfig
        EMITTER_REGISTRY["polyglot"] = (PolyglotEmitter, PolyglotEmitterConfig)
    except ImportError:
        pass
    
    # Language family emitters
    try:
        from semantic_ir.emitters.language_families.lisp import LispEmitter, LispEmitterConfig
        EMITTER_REGISTRY["lisp"] = (LispEmitter, LispEmitterConfig)
    except ImportError:
        pass
    
    try:
        from semantic_ir.emitters.language_families.prolog import PrologEmitter, PrologEmitterConfig
        EMITTER_REGISTRY["prolog"] = (PrologEmitter, PrologEmitterConfig)
    except ImportError:
        pass
    
    # Specialized emitters
    specialized_modules = [
        ("asm_ir", "semantic_ir.emitters.specialized.asm_ir", "AssemblyIREmitter", "AssemblyIREmitterConfig"),
        ("bytecode", "semantic_ir.emitters.specialized.bytecode", "BytecodeEmitter", "BytecodeEmitterConfig"),
        ("constraints", "semantic_ir.emitters.specialized.constraints", "ConstraintEmitter", "ConstraintEmitterConfig"),
        ("fpga", "semantic_ir.emitters.specialized.fpga", "FPGAEmitter", "FPGAEmitterConfig"),
        ("expert", "semantic_ir.emitters.specialized.expert", "ExpertSystemEmitter", "ExpertSystemEmitterConfig"),
        ("functional", "semantic_ir.emitters.specialized.functional", "FunctionalEmitter", "FunctionalEmitterConfig"),
        ("grammar", "semantic_ir.emitters.specialized.grammar", "GrammarEmitter", "GrammarEmitterConfig"),
        ("mobile", "semantic_ir.emitters.specialized.mobile", "MobileEmitter", "MobileEmitterConfig"),
        ("oop", "semantic_ir.emitters.specialized.oop", "OOPEmitter", "OOPEmitterConfig"),
        ("planning", "semantic_ir.emitters.specialized.planning", "PlanningEmitter", "PlanningEmitterConfig"),
        ("parser", "semantic_ir.emitters.specialized.parser", "ParserEmitter", "ParserEmitterConfig"),
        ("systems", "semantic_ir.emitters.specialized.systems", "SystemsEmitter", "SystemsEmitterConfig"),
        ("scientific", "semantic_ir.emitters.specialized.scientific", "ScientificEmitter", "ScientificEmitterConfig"),
        ("lexer", "semantic_ir.emitters.specialized.lexer", "LexerEmitter", "LexerEmitterConfig"),
        ("business", "semantic_ir.emitters.specialized.business", "BusinessEmitter", "BusinessEmitterConfig"),
        ("beam", "semantic_ir.emitters.specialized.beam", "BEAMEmitter", "BEAMEmitterConfig"),
        ("asp", "semantic_ir.emitters.specialized.asp", "ASPEmitter", "ASPEmitterConfig"),
    ]

    for name, module_path, class_name, config_name in specialized_modules:
        try:
            module = __import__(module_path, fromlist=[class_name, config_name])
            emitter_class = getattr(module, class_name)
            config_class = getattr(module, config_name)
            EMITTER_REGISTRY[name] = (emitter_class, config_class)
        except ImportError:
            pass


def _normalize_op(op: str) -> str:
    """Normalize operation name to standard form."""
    if op == "noop":
        return "nop"
    if op == "generic_call":
        return "call"
    return op


def _steps_to_statements(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert flat steps to nested statements with proper control flow structure."""
    total = len(steps)
    processed = set()

    def mark_range(start: int, count: int) -> None:
        if start <= 0 or count <= 0:
            return
        for idx in range(start, start + count):
            if 1 <= idx <= total:
                processed.add(idx)

    def build_block(start: int, count: int) -> List[Dict[str, Any]]:
        if start <= 0 or count <= 0:
            return []
        end = min(start + count - 1, total)
        out: List[Dict[str, Any]] = []
        i = start
        while i <= end:
            if i in processed:
                i += 1
                continue
            step = steps[i - 1]
            op = _normalize_op(step.get("op", ""))
            stmt: Dict[str, Any] = {"type": op}
            if op == "if":
                stmt["condition"] = step.get("condition", "")
                then_start = int(step.get("block_start", 0) or 0)
                then_count = int(step.get("block_count", 0) or 0)
                else_start = int(step.get("else_start", 0) or 0)
                else_count = int(step.get("else_count", 0) or 0)
                if then_start and then_count:
                    mark_range(then_start, then_count)
                    stmt["then"] = build_block(then_start, then_count)
                if else_start and else_count:
                    mark_range(else_start, else_count)
                    stmt["else"] = build_block(else_start, else_count)
            elif op == "while":
                stmt["condition"] = step.get("condition", "")
                body_start = int(step.get("block_start", 0) or 0)
                body_count = int(step.get("block_count", 0) or 0)
                if body_start and body_count:
                    mark_range(body_start, body_count)
                    stmt["body"] = build_block(body_start, body_count)
            elif op == "for":
                stmt["init"] = step.get("init", "")
                stmt["condition"] = step.get("condition", "")
                stmt["increment"] = step.get("increment", "")
                body_start = int(step.get("block_start", 0) or 0)
                body_count = int(step.get("block_count", 0) or 0)
                if body_start and body_count:
                    mark_range(body_start, body_count)
                    stmt["body"] = build_block(body_start, body_count)
            elif op == "switch":
                stmt["expr"] = step.get("expr", step.get("value", ""))
                cases = []
                for case in step.get("cases", []) or []:
                    case_start = int(case.get("block_start", 0) or 0)
                    case_count = int(case.get("block_count", 0) or 0)
                    case_body = []
                    if case_start and case_count:
                        mark_range(case_start, case_count)
                        case_body = build_block(case_start, case_count)
                    cases.append({"value": case.get("value"), "body": case_body})
                if cases:
                    stmt["cases"] = cases
                default_start = int(step.get("default_start", 0) or 0)
                default_count = int(step.get("default_count", 0) or 0)
                if default_start and default_count:
                    mark_range(default_start, default_count)
                    stmt["default"] = build_block(default_start, default_count)
            else:
                if "target" in step:
                    stmt["target"] = step.get("target")
                if "value" in step:
                    stmt["value"] = step.get("value")
                if "cast_type" in step:
                    stmt["cast_type"] = step.get("cast_type")
            out.append(stmt)
            i += 1
        return out

    return build_block(1, total)


def json_to_ir_module(json_data: Dict[str, Any]) -> 'IRModule':
    """Convert JSON IR to IRModule object."""
    # Extract module info
    module_name = json_data.get("module_name") or json_data.get("ir_module") or "module"
    ir_version = json_data.get("ir_version", "1.0")
    docstring = json_data.get("docstring")

    # Convert types
    types = []
    for type_data in json_data.get("types", []):
        fields = []
        for field_data in type_data.get("fields", []):
            fields.append(IRTypeField(
                name=field_data.get("name", ""),
                field_type=field_data.get("type", ""),
                optional=field_data.get("optional", False)
            ))
        types.append(IRType(
            name=type_data.get("name", ""),
            fields=fields,
            docstring=type_data.get("docstring")
        ))

    # Convert functions
    functions = []
    funcs_data = json_data.get("ir_functions", json_data.get("functions", []))
    for func_data in funcs_data:
        # Convert parameters
        params = []
        args_data = func_data.get("args", func_data.get("parameters", []))
        for arg_data in args_data:
            if isinstance(arg_data, dict):
                # Support both "type" and "param_type" keys
                type_str = arg_data.get("param_type", arg_data.get("type", "void"))
                param_type = _parse_type(type_str)
                params.append(IRParameter(
                    name=arg_data.get("name", ""),
                    param_type=param_type
                ))

        # Convert statements
        statements = []
        stmts_data = func_data.get("statements", func_data.get("body", []))
        if not stmts_data and func_data.get("steps"):
            stmts_data = _steps_to_statements(func_data.get("steps", []))
        for stmt_data in stmts_data:
            if isinstance(stmt_data, dict):
                # IR uses "type" field, not "kind"
                stmt_type_str = stmt_data.get("type", stmt_data.get("kind", "nop"))
                # Map IR statement types to IRStatementType enum values
                stmt_type_map = {
                    "while": "loop",
                    "for": "loop",
                    "if": "if",
                    "assign": "assign",
                    "return": "return",
                    "var_decl": "var_decl",
                    "call": "call",
                    "nop": "nop",
                    "add": "add",
                    "sub": "sub",
                    "mul": "mul",
                    "div": "div",
                    "break": "break",
                    "continue": "continue",
                    "block": "block",
                }
                mapped_type = stmt_type_map.get(stmt_type_str, stmt_type_str)
                statements.append(IRStatement(
                    stmt_type=IRStatementType(mapped_type),
                    data_type=_parse_type(stmt_data.get("data_type", stmt_data.get("var_type", "void"))),
                    target=stmt_data.get("target"),
                    value=stmt_data.get("value"),
                    left_op=stmt_data.get("left"),
                    right_op=stmt_data.get("right")
                ))
        
        functions.append(IRFunction(
            name=func_data.get("name", ""),
            return_type=_parse_type(func_data.get("return_type", "void")),
            parameters=params,
            statements=statements,
            docstring=func_data.get("docstring")
        ))
    
    return IRModule(
        ir_version=ir_version,
        module_name=module_name,
        types=types,
        functions=functions,
        docstring=docstring
    )


def _parse_type(type_str: str) -> 'IRDataType':
    """Parse type string to IRDataType."""
    type_map = {
        "void": IRDataType.VOID,
        "bool": IRDataType.BOOL,
        "i8": IRDataType.I8,
        "i16": IRDataType.I16,
        "i32": IRDataType.I32,
        "i64": IRDataType.I64,
        "u8": IRDataType.U8,
        "u16": IRDataType.U16,
        "u32": IRDataType.U32,
        "u64": IRDataType.U64,
        "f32": IRDataType.F32,
        "f64": IRDataType.F64,
        "char": IRDataType.CHAR,
        "string": IRDataType.STRING,
        "pointer": IRDataType.POINTER,
        "array": IRDataType.ARRAY,
        "struct": IRDataType.STRUCT,
    }
    return type_map.get(type_str.lower(), IRDataType.VOID)


# General-purpose language type mappings
IR_TYPE_MAPS = {
    "c": {
        "void": "void", "bool": "bool",
        "i8": "int8_t", "i16": "int16_t", "i32": "int32_t", "i64": "int64_t",
        "u8": "uint8_t", "u16": "uint16_t", "u32": "uint32_t", "u64": "uint64_t",
        "f32": "float", "f64": "double",
        "char": "char", "string": "const char*",
        "pointer": "void*", "array": "array", "struct": "struct",
    },
    "rust": {
        "void": "()", "bool": "bool",
        "i8": "i8", "i16": "i16", "i32": "i32", "i64": "i64",
        "u8": "u8", "u16": "u16", "u32": "u32", "u64": "u64",
        "f32": "f32", "f64": "f64",
        "char": "char", "string": "String",
        "pointer": "*mut c_void", "array": "Vec", "struct": "struct",
    },
    "python": {
        "void": "None", "bool": "bool",
        "i8": "int", "i16": "int", "i32": "int", "i64": "int",
        "u8": "int", "u16": "int", "u32": "int", "u64": "int",
        "f32": "float", "f64": "float",
        "char": "str", "string": "str",
        "pointer": "Any", "array": "List", "struct": "Dict",
    },
    "javascript": {
        "void": "void", "bool": "boolean",
        "i8": "number", "i16": "number", "i32": "number", "i64": "bigint",
        "u8": "number", "u16": "number", "u32": "number", "u64": "bigint",
        "f32": "number", "f64": "number",
        "char": "string", "string": "string",
        "pointer": "any", "array": "Array", "struct": "object",
    },
    "zig": {
        "void": "void", "bool": "bool",
        "i8": "i8", "i16": "i16", "i32": "i32", "i64": "i64",
        "u8": "u8", "u16": "u16", "u32": "u32", "u64": "u64",
        "f32": "f32", "f64": "f64",
        "char": "u8", "string": "[]const u8",
        "pointer": "*anyopaque", "array": "[]", "struct": "struct",
    },
    "go": {
        "void": "", "bool": "bool",
        "i8": "int8", "i16": "int16", "i32": "int32", "i64": "int64",
        "u8": "uint8", "u16": "uint16", "u32": "uint32", "u64": "uint64",
        "f32": "float32", "f64": "float64",
        "char": "rune", "string": "string",
        "pointer": "unsafe.Pointer", "array": "[]", "struct": "struct",
    },
    "ada": {
        "void": "", "bool": "Boolean",
        "i8": "Integer_8", "i16": "Integer_16", "i32": "Integer_32", "i64": "Integer_64",
        "u8": "Unsigned_8", "u16": "Unsigned_16", "u32": "Unsigned_32", "u64": "Unsigned_64",
        "f32": "Float", "f64": "Long_Float",
        "char": "Character", "string": "String",
        "pointer": "System.Address", "array": "Array", "struct": "record",
    },
}


def map_type(type_str: str, lang: str) -> str:
    """Map IR type to target language type."""
    type_map = IR_TYPE_MAPS.get(lang, IR_TYPE_MAPS["c"])
    # Handle both enum values (e.g., "I32") and string representations (e.g., "i32")
    type_key = type_str.lower()
    return type_map.get(type_key, type_key)


def default_return(type_str: str, lang: str) -> str:
    """Get default return value for a type in a language."""
    defaults = {
        "c": {"bool": "false", "i8": "0", "i16": "0", "i32": "0", "i64": "0",
              "u8": "0", "u16": "0", "u32": "0", "u64": "0",
              "f32": "0.0f", "f64": "0.0", "char": "'\\0'", "string": "NULL",
              "pointer": "NULL"},
        "rust": {"bool": "false", "i8": "0", "i16": "0", "i32": "0", "i64": "0",
                 "u8": "0", "u16": "0", "u32": "0", "u64": "0",
                 "f32": "0.0", "f64": "0.0", "char": "'\\0'", "string": "String::new()",
                 "pointer": "std::ptr::null_mut()"},
        "python": {"bool": "False", "i8": "0", "i16": "0", "i32": "0", "i64": "0",
                   "u8": "0", "u16": "0", "u32": "0", "u64": "0",
                   "f32": "0.0", "f64": "0.0", "char": "'\\0'", "string": '""',
                   "pointer": "None"},
        "javascript": {"bool": "false", "i8": "0", "i16": "0", "i32": "0", "i64": "0n",
                       "u8": "0", "u16": "0", "u32": "0", "u64": "0n",
                       "f32": "0.0", "f64": "0.0", "char": "'\\0'", "string": '""',
                       "pointer": "null"},
        "zig": {"bool": "false", "i8": "0", "i16": "0", "i32": "0", "i64": "0",
                "u8": "0", "u16": "0", "u32": "0", "u64": "0",
                "f32": "0.0", "f64": "0.0", "char": "0", "string": "&[_]u8{}",
                "pointer": "undefined"},
        "go": {"bool": "false", "i8": "0", "i16": "0", "i32": "0", "i64": "0",
               "u8": "0", "u16": "0", "u32": "0", "u64": "0",
               "f32": "0.0", "f64": "0.0", "char": "0", "string": '""',
               "pointer": "nil"},
        "ada": {"bool": "False", "i8": "0", "i16": "0", "i32": "0", "i64": "0",
                "u8": "0", "u16": "0", "u32": "0", "u64": "0",
                "f32": "0.0", "f64": "0.0", "char": "Character'Val(0)", "string": '""',
                "pointer": "System.Null_Address"},
    }
    return defaults.get(lang, {}).get(type_str.lower(), "0")


def emit_statement(stmt: Dict[str, Any], lang: str, indent: int = 0) -> List[str]:
    """Emit a single IR statement to target language code.

    Args:
        stmt: IR statement dictionary
        lang: Target language
        indent: Indentation level

    Returns:
        List of code lines
    """
    lines = []
    indent_str = "    " * indent

    stmt_type = stmt.get("type", "")

    if stmt_type == "var_decl":
        var_name = stmt.get("var_name", "")
        var_type = stmt.get("var_type", "")
        init = stmt.get("init", None)

        mapped_type = map_type(var_type, lang)

        if lang == "c":
            if init is not None:
                lines.append(f"{indent_str}{mapped_type} {var_name} = {init};")
            else:
                lines.append(f"{indent_str}{mapped_type} {var_name};")
        elif lang == "rust":
            if init is not None:
                lines.append(f"{indent_str}let {var_name}: {mapped_type} = {init};")
            else:
                lines.append(f"{indent_str}let {var_name}: {mapped_type};")
        elif lang == "python":
            if init is not None:
                lines.append(f"{indent_str}{var_name} = {init}")
            else:
                lines.append(f"{indent_str}{var_name} = None")
        elif lang == "ada":
            if init is not None:
                lines.append(f"{indent_str}{var_name} : {mapped_type} := {init};")
            else:
                lines.append(f"{indent_str}{var_name} : {mapped_type};")

    elif stmt_type == "assign":
        target = stmt.get("target", "")
        value = stmt.get("value", "")

        if lang == "python":
            lines.append(f"{indent_str}{target} = {value}")
        else:
            lines.append(f"{indent_str}{target} = {value};")

    elif stmt_type == "return":
        value = stmt.get("value", None)
        if value is not None:
            if lang == "python":
                lines.append(f"{indent_str}return {value}")
            else:
                lines.append(f"{indent_str}return {value};")
        else:
            if lang == "python":
                lines.append(f"{indent_str}return")
            else:
                lines.append(f"{indent_str}return;")

    elif stmt_type == "if":
        condition = stmt.get("condition", "")
        then_body = stmt.get("then", [])
        else_body = stmt.get("else", [])

        if lang == "python":
            lines.append(f"{indent_str}if {condition}:")
            if then_body:
                for s in then_body:
                    lines.extend(emit_statement(s, lang, indent + 1))
            else:
                lines.append(f"{indent_str}    pass")
            if else_body:
                lines.append(f"{indent_str}else:")
                for s in else_body:
                    lines.extend(emit_statement(s, lang, indent + 1))
        else:
            lines.append(f"{indent_str}if ({condition}) {{")
            for s in then_body:
                lines.extend(emit_statement(s, lang, indent + 1))
            if else_body:
                lines.append(f"{indent_str}}} else {{")
                for s in else_body:
                    lines.extend(emit_statement(s, lang, indent + 1))
            lines.append(f"{indent_str}}}")

    elif stmt_type == "while" or stmt_type == "loop":
        condition = stmt.get("condition", "")
        body = stmt.get("body", [])

        if lang == "python":
            lines.append(f"{indent_str}while {condition}:")
            if body:
                for s in body:
                    lines.extend(emit_statement(s, lang, indent + 1))
            else:
                lines.append(f"{indent_str}    pass")
        else:
            lines.append(f"{indent_str}while ({condition}) {{")
            for s in body:
                lines.extend(emit_statement(s, lang, indent + 1))
            lines.append(f"{indent_str}}}")

    return lines

    elif stmt_type == "if":
        condition = stmt.get("condition", "")
        then_body = stmt.get("then", [])
        else_body = stmt.get("else", [])

        if lang == "python":
            lines.append(f"{indent_str}if {condition}:")
            if then_body:
                for s in then_body:
                    lines.extend(emit_statement(s, lang, indent + 1))
            else:
                lines.append(f"{indent_str}    pass")
            if else_body:
                lines.append(f"{indent_str}else:")
                for s in else_body:
                    lines.extend(emit_statement(s, lang, indent + 1))
        else:
            lines.append(f"{indent_str}if ({condition}) {{")
            for s in then_body:
                lines.extend(emit_statement(s, lang, indent + 1))
            if else_body:
                lines.append(f"{indent_str}}} else {{")
                for s in else_body:
                    lines.extend(emit_statement(s, lang, indent + 1))
            lines.append(f"{indent_str}}}")

    elif stmt_type == "while":
        condition = stmt.get("condition", "")
        body = stmt.get("body", [])

        if lang == "python":
            lines.append(f"{indent_str}while {condition}:")
            if body:
                for s in body:
                    lines.extend(emit_statement(s, lang, indent + 1))
            else:
                lines.append(f"{indent_str}    pass")
        else:
            lines.append(f"{indent_str}while ({condition}) {{")
            for s in body:
                lines.extend(emit_statement(s, lang, indent + 1))
            lines.append(f"{indent_str}}}")

    return lines


def emit_function_body(body: List[Dict[str, Any]], lang: str) -> List[str]:
    """Emit function body from IR statements.

    Args:
        body: List of IR statement dictionaries
        lang: Target language

    Returns:
        List of code lines
    """
    lines = []
    for stmt in body:
        lines.extend(emit_statement(stmt, lang, indent=1))
    return lines


def emit_general_language(ir_module: 'IRModule', lang: str) -> str:
    """Generate code for a general-purpose language from IRModule."""
    lines = []

    if lang == "python":
        lines.append(f"# STUNIR Generated Code")
        lines.append(f"# Target Language: {lang}")
        lines.append(f"# Module: {ir_module.module_name}")
        lines.append("")
        lines.append("from typing import *")
        lines.append("")
    else:
        lines = [
            f"// STUNIR Generated Code",
            f"// Target Language: {lang}",
            f"// Module: {ir_module.module_name}",
            "",
        ]

    if lang == "rust":
        lines.append("// Generated Rust code")
        lines.append("")
    elif lang == "go":
        lines.append(f"package {ir_module.module_name.lower().replace('-', '_')}")
        lines.append("")
    elif lang == "ada":
        lines.append(f"package {ir_module.module_name.title().replace('_', '')} is")
        lines.append("")

    for func in ir_module.functions:
        # Generate function signature
        ret_type = map_type(func.return_type.value, lang)

        # Build parameter list
        arg_parts = []
        for param in func.parameters:
            param_type = map_type(param.param_type.value, lang)
            if lang == "go" and param_type:
                arg_parts.append(f"{param.name} {param_type}")
            elif lang == "ada" and param_type:
                arg_parts.append(f"{param.name} : {param_type}")
            elif lang == "python":
                arg_parts.append(f"{param.name}: {param_type}")
            elif lang == "rust":
                arg_parts.append(f"{param.name}: {param_type}")
            elif lang == "javascript":
                arg_parts.append(param.name)
            elif lang == "zig":
                arg_parts.append(f"{param.name}: {param_type}")
            else:
                arg_parts.append(f"{param_type} {param.name}")

        args_str = ", ".join(arg_parts)

        # Add docstring
        if func.docstring:
            if lang == "python":
                # Python docstring goes inside the function
                pass
            elif lang in ("c", "cpp", "zig", "go"):
                lines.append(f"// {func.docstring}")
            elif lang == "rust":
                lines.append(f"/// {func.docstring}")
            elif lang == "javascript":
                lines.append(f"/** {func.docstring} */")
            elif lang == "ada":
                lines.append(f"   -- {func.docstring}")

        # Generate function signature
        if lang == "python":
            lines.append(f"def {func.name}({args_str}) -> {ret_type}:")
            if func.docstring:
                lines.append(f'    """{func.docstring}"""')
        elif lang == "javascript":
            lines.append(f"function {func.name}({args_str}) {{")
        elif lang == "rust":
            lines.append(f"pub fn {func.name}({args_str}) -> {ret_type} {{")
        elif lang == "zig":
            lines.append(f"pub fn {func.name}({args_str}) {ret_type} {{")
        elif lang == "go":
            if ret_type:
                lines.append(f"func {func.name}({args_str}) {ret_type} {{")
            else:
                lines.append(f"func {func.name}({args_str}) {{")
        elif lang == "ada":
            if ret_type:
                lines.append(f"   function {func.name}({'; '.join(arg_parts)}) return {ret_type};")
            else:
                lines.append(f"   procedure {func.name}({'; '.join(arg_parts)});")
            continue
        else:  # c
            lines.append(f"{ret_type} {func.name}({args_str}) {{")

        # Generate body from IR statements
        body_dicts = []
        if hasattr(func, 'statements') and func.statements:
            # Convert IRStatement objects to dicts for emit_function_body
            for stmt in func.statements:
                stmt_dict = {"type": stmt.stmt_type.value if hasattr(stmt.stmt_type, 'value') else str(stmt.stmt_type)}
                if stmt.target:
                    stmt_dict["target"] = stmt.target
                if stmt.value:
                    stmt_dict["value"] = stmt.value
                if stmt.left_op:
                    stmt_dict["left"] = stmt.left_op
                if stmt.right_op:
                    stmt_dict["right"] = stmt.right_op
                body_dicts.append(stmt_dict)

        if body_dicts:
            body_lines = emit_function_body(body_dicts, lang)
            lines.extend(body_lines)
        else:
            # Fallback to TODO if no body
            if lang == "python":
                lines.append("    # TODO: Implement")
            else:
                lines.append("    // TODO: Implement")

        if lang != "ada":
            # Only add default return if no explicit return in body
            has_return = any(s.stmt_type == IRStatementType.RETURN for s in func.statements) if hasattr(func, 'statements') and func.statements else False
            if not has_return and func.return_type != IRDataType.VOID and ret_type not in ("", "void", "None", "()"):
                default = default_return(func.return_type.value, lang)
                if lang == "python":
                    lines.append(f"    return {default}")
                else:
                    lines.append(f"    return {default};")

            if lang == "python":
                # Python doesn't use braces
                lines.append("")
            elif lang in ("javascript", "rust", "zig", "go"):
                lines.append("}")
                lines.append("")
            else:
                lines.append("}")
                lines.append("")

    if lang == "ada":
        lines.append(f"end {ir_module.module_name.title().replace('_', '')};")

    return "\n".join(lines).rstrip() + "\n"


def list_available_targets() -> Tuple[List[str], List[str]]:
    """Return (general_languages, specialized_emitters)."""
    general = ["c", "rust", "python", "javascript", "js", "zig", "go", "ada"]
    specialized = sorted(EMITTER_REGISTRY.keys())
    return general, specialized


def main() -> int:
    """CLI entry point."""
    register_specialized_emitters()

    general_langs, specialized = list_available_targets()

    parser = argparse.ArgumentParser(
        description="STUNIR IR to Code Generator - Unified CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available Targets:
  General Languages: {', '.join(general_langs)}
  Specialized Emitters: {', '.join(specialized) if specialized else 'None loaded'}

Examples:
  # Generate C code
  python tools/ir_to_code.py --ir input.json --lang c --out output/

  # Generate Rust code
  python tools/ir_to_code.py --ir input.json --lang rust --out output/

  # Use specialized GPU emitter
  python tools/ir_to_code.py --ir input.json --emitter gpu --out output/
        """
    )

    parser.add_argument("--ir", help="Path to IR JSON file")
    parser.add_argument("--out", help="Output directory")
    parser.add_argument("--lang", help=f"Target language ({', '.join(general_langs)})")
    parser.add_argument("--emitter", help=f"Specialized emitter ({', '.join(specialized) if specialized else 'none'})")
    parser.add_argument("--list", action="store_true", help="List available targets and exit")

    args = parser.parse_args()

    if args.list:
        print("Available Targets:")
        print(f"  General Languages: {', '.join(general_langs)}")
        if specialized:
            print(f"  Specialized Emitters: {', '.join(specialized)}")
        return 0

    if not args.ir or not args.out:
        print("Error: --ir and --out are required (unless using --list)", file=sys.stderr)
        parser.print_help()
        return 1

    if not args.lang and not args.emitter:
        print("Error: Must specify either --lang or --emitter", file=sys.stderr)
        parser.print_help()
        return 1

    if args.lang and args.emitter:
        print("Error: Cannot specify both --lang and --emitter", file=sys.stderr)
        return 1

    # Load IR
    try:
        with open(args.ir, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: IR file not found: {args.ir}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in IR file: {e}", file=sys.stderr)
        return 1

    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.lang:
        # Use general-purpose language emitter
        lang = args.lang.lower()
        if lang not in general_langs:
            print(f"Error: Language '{lang}' not supported. Use: {', '.join(general_langs)}",
                  file=sys.stderr)
            return 1

        if not SEMANTIC_IR_AVAILABLE:
            print("Error: semantic_ir module required for code generation", file=sys.stderr)
            return 1

        # Convert JSON to IRModule and generate code
        ir_module = json_to_ir_module(json_data)
        code = emit_general_language(ir_module, lang)

        # Determine file extension
        extensions = {
            "c": ".c", "rust": ".rs", "python": ".py",
            "javascript": ".js", "js": ".js", "zig": ".zig",
            "go": ".go", "ada": ".adb"
        }
        ext = extensions.get(lang, ".txt")
        output_path = output_dir / f"{ir_module.module_name}{ext}"

        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(code)

        print(f"Generated: {output_path}")

    else:
        # Use specialized emitter
        if not SEMANTIC_IR_AVAILABLE:
            print("Error: semantic_ir module required for specialized emitters", file=sys.stderr)
            return 1

        emitter_name = args.emitter.lower()
        if emitter_name not in EMITTER_REGISTRY:
            print(f"Error: Emitter '{emitter_name}' not available. Use --list to see options.",
                  file=sys.stderr)
            return 1

        emitter_class, config_class = EMITTER_REGISTRY[emitter_name]

        # Convert JSON to IRModule
        ir_module = json_to_ir_module(json_data)

        config = config_class(
            output_dir=str(output_dir),
            module_name=ir_module.module_name
        )
        emitter = emitter_class(config)
        result = emitter.emit(ir_module)

        if result.status == EmitterStatus.SUCCESS:
            print(f"Generated {result.files_count} files:")
            for gen_file in result.files:
                print(f"  - {gen_file.path} ({gen_file.size} bytes)")
        else:
            print(f"Error: {result.error_message}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())