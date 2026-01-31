"""STUNIR Code Generator Utilities - Python Reference Implementation

Utility functions for code generation across all emitters.
Based on Ada SPARK code generation utilities.
"""

import re
from typing import List, Dict, Optional, Tuple
from .types import IRDataType, Architecture


class CodeGenerator:
    """Utility class for common code generation operations."""

    @staticmethod
    def sanitize_identifier(name: str) -> str:
        """Sanitize identifier to be safe for code generation.
        
        Args:
            name: Raw identifier name
            
        Returns:
            Sanitized identifier safe for use in generated code
        """
        # Remove invalid characters
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        
        # Ensure starts with letter or underscore
        if sanitized and sanitized[0].isdigit():
            sanitized = '_' + sanitized
        
        return sanitized or "unnamed"

    @staticmethod
    def wrap_line(line: str, max_length: int = 100, indent: str = "") -> List[str]:
        """Wrap long line to maximum length.
        
        Args:
            line: Line to wrap
            max_length: Maximum line length
            indent: Indentation for continuation lines
            
        Returns:
            List of wrapped lines
        """
        if len(line) <= max_length:
            return [line]
        
        lines = []
        current = ""
        
        for word in line.split():
            if current and len(current) + len(word) + 1 > max_length:
                lines.append(current)
                current = indent + word
            else:
                current = current + " " + word if current else word
        
        if current:
            lines.append(current)
        
        return lines

    @staticmethod
    def escape_string(s: str, style: str = "c") -> str:
        """Escape string for code generation.
        
        Args:
            s: String to escape
            style: Escape style ("c", "python", "rust", etc.)
            
        Returns:
            Escaped string
        """
        if style == "c":
            return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
        elif style == "python":
            return s.replace("\\", "\\\\").replace('"', '\\"')
        elif style == "rust":
            return s.replace("\\", "\\\\").replace('"', '\\"')
        else:
            return s

    @staticmethod
    def format_comment(text: str, style: str = "c", width: int = 80) -> List[str]:
        """Format multi-line comment.
        
        Args:
            text: Comment text
            style: Comment style ("c", "cpp", "python", "ada", "rust")
            width: Maximum comment width
            
        Returns:
            List of formatted comment lines
        """
        lines = []
        
        if style == "c":
            lines.append("/*")
            for line in text.split("\n"):
                lines.append(f" * {line}")
            lines.append(" */")
        elif style == "cpp":
            for line in text.split("\n"):
                lines.append(f"// {line}")
        elif style == "python":
            lines.append('"""')
            lines.extend(text.split("\n"))
            lines.append('"""')
        elif style == "ada":
            for line in text.split("\n"):
                lines.append(f"-- {line}")
        elif style == "rust":
            for line in text.split("\n"):
                lines.append(f"/// {line}")
        
        return lines

    @staticmethod
    def generate_include_guard(filename: str) -> Tuple[str, str]:
        """Generate C/C++ include guard macros.
        
        Args:
            filename: Header filename
            
        Returns:
            Tuple of (start_guard, end_guard)
        """
        guard = re.sub(r'[^A-Z0-9_]', '_', filename.upper())
        guard = f"STUNIR_{guard}_"
        
        start = f"#ifndef {guard}\n#define {guard}"
        end = f"#endif /* {guard} */"
        
        return start, end

    @staticmethod
    def map_type_to_language(
        ir_type: IRDataType,
        language: str
    ) -> str:
        """Map IR type to language-specific type.
        
        Args:
            ir_type: IR data type
            language: Target language ("c", "python", "rust", "haskell")
            
        Returns:
            Language-specific type name
        """
        if language == "c":
            return {
                IRDataType.VOID: "void",
                IRDataType.BOOL: "bool",
                IRDataType.I8: "int8_t",
                IRDataType.I16: "int16_t",
                IRDataType.I32: "int32_t",
                IRDataType.I64: "int64_t",
                IRDataType.U8: "uint8_t",
                IRDataType.U16: "uint16_t",
                IRDataType.U32: "uint32_t",
                IRDataType.U64: "uint64_t",
                IRDataType.F32: "float",
                IRDataType.F64: "double",
                IRDataType.CHAR: "char",
                IRDataType.STRING: "char*",
                IRDataType.POINTER: "void*",
            }.get(ir_type, "void")
        elif language == "python":
            return {
                IRDataType.BOOL: "bool",
                IRDataType.I32: "int",
                IRDataType.I64: "int",
                IRDataType.F32: "float",
                IRDataType.F64: "float",
                IRDataType.STRING: "str",
            }.get(ir_type, "Any")
        elif language == "rust":
            return {
                IRDataType.VOID: "()",
                IRDataType.BOOL: "bool",
                IRDataType.I8: "i8",
                IRDataType.I16: "i16",
                IRDataType.I32: "i32",
                IRDataType.I64: "i64",
                IRDataType.U8: "u8",
                IRDataType.U16: "u16",
                IRDataType.U32: "u32",
                IRDataType.U64: "u64",
                IRDataType.F32: "f32",
                IRDataType.F64: "f64",
                IRDataType.CHAR: "char",
                IRDataType.STRING: "String",
            }.get(ir_type, "()")
        elif language == "haskell":
            return {
                IRDataType.VOID: "()",
                IRDataType.BOOL: "Bool",
                IRDataType.I32: "Int32",
                IRDataType.I64: "Int64",
                IRDataType.F32: "Float",
                IRDataType.F64: "Double",
                IRDataType.STRING: "String",
            }.get(ir_type, "()")
        
        return "unknown"

    @staticmethod
    def generate_function_signature(
        name: str,
        params: List[Tuple[str, str]],
        return_type: str,
        language: str
    ) -> str:
        """Generate function signature for target language.
        
        Args:
            name: Function name
            params: List of (param_name, param_type) tuples
            return_type: Return type string
            language: Target language
            
        Returns:
            Function signature string
        """
        if language == "c":
            param_str = ", ".join(f"{t} {n}" for n, t in params)
            return f"{return_type} {name}({param_str or 'void'})"
        elif language == "python":
            param_str = ", ".join(f"{n}: {t}" for n, t in params)
            return f"def {name}({param_str}) -> {return_type}:"
        elif language == "rust":
            param_str = ", ".join(f"{n}: {t}" for n, t in params)
            return f"fn {name}({param_str}) -> {return_type}"
        elif language == "haskell":
            type_sig = " -> ".join([t for _, t in params] + [return_type])
            return f"{name} :: {type_sig}"
        
        return f"{name}(...)"  # Fallback
