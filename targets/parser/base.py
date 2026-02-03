#!/usr/bin/env python3
"""Base class for parser emitters.

Provides common functionality for all parser code emitters.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

# Import from parser module
try:
    from ir.parser.parse_table import ParseTable, LL1Table
    from ir.parser.ast_node import ASTSchema, ASTNodeSpec
    from ir.parser.parser_generator import ParserGeneratorResult
    from ir.grammar.grammar_ir import Grammar
except ImportError:
    ParseTable = Any
    LL1Table = Any
    ASTSchema = Any
    ParserGeneratorResult = Any
    Grammar = Any


@dataclass
class ParserEmitterResult:
    """Result of parser emission.
    
    Attributes:
        code: Generated parser code
        ast_code: Generated AST node definitions
        manifest: Build manifest dictionary
        warnings: List of warnings
        auxiliary_files: Dict of additional file name to content
    """
    code: str
    ast_code: str
    manifest: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    auxiliary_files: Dict[str, str] = field(default_factory=dict)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def add_auxiliary_file(self, filename: str, content: str) -> None:
        """Add an auxiliary file."""
        self.auxiliary_files[filename] = content
    
    def total_size(self) -> int:
        """Get total size of all generated code."""
        size = len(self.code) + len(self.ast_code)
        for content in self.auxiliary_files.values():
            size += len(content)
        return size


class ParserEmitterBase(ABC):
    """Abstract base class for parser emitters.
    
    Subclasses implement target language-specific emission.
    
    Attributes:
        LANGUAGE: Target language name (e.g., "python", "rust")
        FILE_EXTENSION: Default file extension
    """
    
    LANGUAGE: str = "base"
    FILE_EXTENSION: str = ".txt"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the emitter.
        
        Args:
            config: Optional configuration dictionary
                - include_comments: bool, whether to include comments (default: True)
                - error_recovery: bool, whether to include error recovery (default: False)
                - optimize: bool, whether to optimize generated code (default: False)
        """
        self.config = config or {}
        self._warnings: List[str] = []
    
    @abstractmethod
    def emit(self, result: ParserGeneratorResult, 
             grammar: Grammar) -> ParserEmitterResult:
        """Emit parser code for target language.
        
        Args:
            result: Parser generator result with tables
            grammar: Source grammar
        
        Returns:
            ParserEmitterResult with generated code
        """
        pass
    
    @abstractmethod
    def emit_parse_table(self, table: Union[ParseTable, LL1Table]) -> str:
        """Emit parse table as target language data structure.
        
        Args:
            table: Parse table to emit
        
        Returns:
            Code string representing the table
        """
        pass
    
    @abstractmethod
    def emit_ast_nodes(self, schema: ASTSchema) -> str:
        """Emit AST node definitions.
        
        Args:
            schema: AST schema with node specifications
        
        Returns:
            Code string with AST class/struct definitions
        """
        pass
    
    def _add_warning(self, message: str) -> None:
        """Add a warning message."""
        self._warnings.append(message)
    
    def _get_warnings(self) -> List[str]:
        """Get and clear accumulated warnings."""
        warnings = self._warnings.copy()
        self._warnings.clear()
        return warnings
    
    def _include_comments(self) -> bool:
        """Check if comments should be included."""
        return self.config.get('include_comments', True)
    
    def _include_error_recovery(self) -> bool:
        """Check if error recovery should be included."""
        return self.config.get('error_recovery', False)
    
    def _generate_manifest(self, code: str, ast_code: str, 
                          grammar: Grammar,
                          auxiliary_files: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Generate deterministic manifest for the emitted parser.
        
        Args:
            code: Generated parser code
            ast_code: Generated AST code
            grammar: Source grammar
            auxiliary_files: Optional dict of auxiliary files
        
        Returns:
            Manifest dictionary
        """
        def compute_sha256(data: bytes) -> str:
            return hashlib.sha256(data).hexdigest()
        
        def canonical_json(obj: Any) -> str:
            return json.dumps(obj, sort_keys=True, separators=(',', ':'))
        
        total_size = len(code) + len(ast_code)
        if auxiliary_files:
            for content in auxiliary_files.values():
                total_size += len(content)
        
        manifest = {
            "schema": f"stunir.parser.{self.LANGUAGE}.v1",
            "generator": f"stunir.parser.{self.LANGUAGE}_emitter",
            "epoch": int(time.time()),
            "grammar_name": grammar.name if hasattr(grammar, 'name') else "unknown",
            "language": self.LANGUAGE,
            "parser_code_hash": compute_sha256(code.encode('utf-8')),
            "parser_code_size": len(code),
            "ast_code_hash": compute_sha256(ast_code.encode('utf-8')),
            "ast_code_size": len(ast_code),
            "total_size": total_size,
        }
        
        if auxiliary_files:
            manifest["auxiliary_file_count"] = len(auxiliary_files)
        
        # Compute manifest hash
        manifest_for_hash = {k: v for k, v in manifest.items() if k != "manifest_hash"}
        manifest["manifest_hash"] = compute_sha256(
            canonical_json(manifest_for_hash).encode('utf-8')
        )
        
        return manifest
    
    def _emit_header(self, grammar: Grammar) -> str:
        """Generate header comment for the output.
        
        Args:
            grammar: Source grammar
        
        Returns:
            Header comment string
        """
        if not self._include_comments():
            return ""
        
        import datetime
        
        comment = self._get_comment_style()
        lines = [
            f"{comment['start']} Parser for: {grammar.name if hasattr(grammar, 'name') else 'unknown'}",
            f"{comment['line']} Language: {self.LANGUAGE}",
            f"{comment['line']} Generated: {datetime.datetime.now().isoformat()}",
            f"{comment['line']} Generator: STUNIR Parser Emitter",
            f"{comment['end']}",
            "",
        ]
        return "\n".join(lines)
    
    def _get_comment_style(self) -> Dict[str, str]:
        """Get comment style for this language.
        
        Returns:
            Dict with 'start', 'line', 'end' keys
        """
        # Default: C-style block comments
        return {
            'start': '/*',
            'line': ' *',
            'end': ' */',
        }
    
    def _format_symbol_name(self, name: str) -> str:
        """Format a symbol name for use in generated code.
        
        Args:
            name: Original symbol name
        
        Returns:
            Formatted name safe for use in code
        """
        # Replace non-alphanumeric characters
        result = []
        for c in name:
            if c.isalnum():
                result.append(c)
            elif c in '+-*/<>=!&|':
                # Operator symbols
                op_names = {
                    '+': 'PLUS', '-': 'MINUS', '*': 'STAR', '/': 'SLASH',
                    '<': 'LT', '>': 'GT', '=': 'EQ', '!': 'BANG',
                    '&': 'AND', '|': 'OR',
                }
                result.append('_' + op_names.get(c, 'OP') + '_')
            else:
                result.append('_')
        
        return ''.join(result)
    
    def _map_type(self, ir_type: str) -> str:
        """Map an IR type to target language type.
        
        Override in subclasses for language-specific mapping.
        
        Args:
            ir_type: IR type name
        
        Returns:
            Target language type name
        """
        return ir_type
