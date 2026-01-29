"""
Tool Generator for STUNIR Bootstrap.

Generates parser and lexer code from STUNIR specifications using:
- Parser Generator (Phase 6B) for parsing tables
- Lexer Generator (Phase 6C) for DFA
- Parser Emitters for Python parser code
- Lexer Emitters for Python lexer code

The generated tools are saved to:
- bootstrap/stunir_parser.py (generated parser)
- bootstrap/stunir_lexer_gen.py (generated lexer)
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import grammar and lexer builders
from .stunir_grammar import STUNIRGrammarBuilder
from .stunir_lexer import STUNIRLexerBuilder

# Import from Phase 6 modules
from ir.grammar.grammar_ir import Grammar
from ir.lexer.token_spec import LexerSpec
from ir.lexer.lexer_generator import LexerGenerator
from ir.parser.parser_generator import ParserGenerator

# Import emitters
from targets.parser.python_parser import PythonParserEmitter
from targets.lexer.python_lexer import PythonLexerEmitter


@dataclass
class GenerationResult:
    """
    Result of tool generation.
    
    Attributes:
        success: True if generation succeeded
        parser_code: Generated parser code
        lexer_code: Generated lexer code
        parser_path: Path to saved parser file
        lexer_path: Path to saved lexer file
        manifest: Build manifest with hashes
        errors: List of error messages
        warnings: List of warning messages
    """
    success: bool
    parser_code: str = ""
    lexer_code: str = ""
    parser_path: Optional[Path] = None
    lexer_path: Optional[Path] = None
    manifest: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class ToolGenerator:
    """
    Generator for STUNIR parser and lexer tools.
    
    Uses the Grammar IR (Phase 6A), Parser Generator (Phase 6B), and
    Lexer Generator (Phase 6C) to create working parser and lexer
    implementations for the STUNIR language.
    
    Usage:
        generator = ToolGenerator()
        result = generator.generate_all()
        
        # Or step by step:
        parser_code = generator.generate_parser()
        lexer_code = generator.generate_lexer()
        generator.save_all()
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the tool generator.
        
        Args:
            output_dir: Directory to save generated files (default: bootstrap/)
        """
        if output_dir is None:
            # Default to bootstrap directory
            output_dir = Path(__file__).parent
        self.output_dir = Path(output_dir)
        
        # Build specifications
        self.grammar: Optional[Grammar] = None
        self.lexer_spec: Optional[LexerSpec] = None
        
        # Generated code
        self.parser_code: Optional[str] = None
        self.lexer_code: Optional[str] = None
        
        # Generation metadata
        self._errors: List[str] = []
        self._warnings: List[str] = []
    
    def build_specs(self) -> bool:
        """
        Build grammar and lexer specifications.
        
        Returns:
            True if specifications built successfully
        """
        try:
            # Build grammar
            grammar_builder = STUNIRGrammarBuilder()
            self.grammar = grammar_builder.build()
            
            # Validate grammar
            from ir.grammar.validation import GrammarValidator
            validator = GrammarValidator(self.grammar)
            result = validator.validate()
            
            if not result.valid:
                for error in result.errors:
                    self._errors.append(f"Grammar: {error}")
                return False
            
            for warning in result.warnings:
                self._warnings.append(f"Grammar: {warning}")
            
            # Build lexer spec
            lexer_builder = STUNIRLexerBuilder()
            self.lexer_spec = lexer_builder.build()
            
            # Validate lexer spec
            lexer_errors = self.lexer_spec.validate()
            if lexer_errors:
                for error in lexer_errors:
                    self._errors.append(f"Lexer: {error}")
                return False
            
            return True
            
        except Exception as e:
            self._errors.append(f"Failed to build specs: {e}")
            return False
    
    def generate_parser(self) -> Optional[str]:
        """
        Generate STUNIR parser code.
        
        Returns:
            Generated parser code or None on error
        """
        if self.grammar is None:
            self._errors.append("Grammar not built. Call build_specs() first.")
            return None
        
        try:
            # Use parser generator
            parser_gen = ParserGenerator(self.grammar)
            gen_result = parser_gen.generate()
            
            if gen_result.has_conflicts():
                # Report conflicts as warnings (may still work)
                for conflict in gen_result.conflicts[:10]:  # First 10
                    self._warnings.append(f"Parser conflict: {conflict}")
            
            # Generate Python code
            emitter = PythonParserEmitter({
                'include_comments': True,
                'class_name': 'STUNIRParser',
            })
            
            emit_result = emitter.emit(gen_result, self.grammar)
            
            # Add header
            header = '''"""
STUNIR Parser - Generated by Bootstrap Tool Generator.

DO NOT EDIT - This file is auto-generated from stunir_grammar.py

This parser recognizes the STUNIR language as defined in the grammar
specification. It produces an AST that can be processed by the
bootstrap compiler.
"""

'''
            self.parser_code = header + emit_result.code
            
            for warning in emit_result.warnings:
                self._warnings.append(f"Parser emitter: {warning}")
            
            return self.parser_code
            
        except Exception as e:
            self._errors.append(f"Failed to generate parser: {e}")
            return None
    
    def generate_lexer(self) -> Optional[str]:
        """
        Generate STUNIR lexer code.
        
        Returns:
            Generated lexer code or None on error
        """
        if self.lexer_spec is None:
            self._errors.append("Lexer spec not built. Call build_specs() first.")
            return None
        
        try:
            # Use lexer generator
            lexer_gen = LexerGenerator(self.lexer_spec)
            dfa = lexer_gen.generate()
            
            # Generate Python code
            emitter = PythonLexerEmitter({
                'include_comments': True,
                'class_name': 'STUNIRLexer',
            })
            
            emit_result = lexer_gen.emit(emitter)
            
            # Add header and keyword handling
            header = '''"""
STUNIR Lexer - Generated by Bootstrap Tool Generator.

DO NOT EDIT - This file is auto-generated from stunir_lexer.py

This lexer tokenizes STUNIR source code according to the token
specifications. Keywords are handled via post-processing of
IDENTIFIER tokens.
"""

from bootstrap.stunir_lexer import STUNIR_KEYWORDS

'''
            
            # Add keyword handling to the lexer
            keyword_handler = '''

    def _handle_keywords(self, token):
        """Convert IDENTIFIER tokens to keywords when applicable."""
        if token.type == 'IDENTIFIER' and token.value in STUNIR_KEYWORDS:
            token.type = STUNIR_KEYWORDS[token.value]
        return token
    
    def next_token(self):
        """Get next token with keyword handling."""
        token = self._raw_next_token()
        if token:
            token = self._handle_keywords(token)
        return token

'''
            
            self.lexer_code = header + emit_result.code + keyword_handler
            
            for warning in emit_result.warnings:
                self._warnings.append(f"Lexer emitter: {warning}")
            
            return self.lexer_code
            
        except Exception as e:
            self._errors.append(f"Failed to generate lexer: {e}")
            return None
    
    def save_parser(self, path: Optional[Path] = None) -> Optional[Path]:
        """
        Save generated parser to file.
        
        Args:
            path: Output path (default: bootstrap/stunir_parser.py)
            
        Returns:
            Path to saved file or None on error
        """
        if self.parser_code is None:
            self._errors.append("Parser not generated. Call generate_parser() first.")
            return None
        
        if path is None:
            path = self.output_dir / 'stunir_parser.py'
        
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.parser_code)
            return path
        except Exception as e:
            self._errors.append(f"Failed to save parser: {e}")
            return None
    
    def save_lexer(self, path: Optional[Path] = None) -> Optional[Path]:
        """
        Save generated lexer to file.
        
        Args:
            path: Output path (default: bootstrap/stunir_lexer_gen.py)
            
        Returns:
            Path to saved file or None on error
        """
        if self.lexer_code is None:
            self._errors.append("Lexer not generated. Call generate_lexer() first.")
            return None
        
        if path is None:
            path = self.output_dir / 'stunir_lexer_gen.py'
        
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.lexer_code)
            return path
        except Exception as e:
            self._errors.append(f"Failed to save lexer: {e}")
            return None
    
    def generate_manifest(self) -> Dict[str, Any]:
        """
        Generate build manifest for generated tools.
        
        Returns:
            Manifest dictionary with hashes and metadata
        """
        manifest = {
            'schema': 'stunir.bootstrap.manifest.v1',
            'generated_at': int(time.time()),
            'generator': 'ToolGenerator',
            'components': {}
        }
        
        if self.parser_code:
            parser_hash = hashlib.sha256(self.parser_code.encode()).hexdigest()
            manifest['components']['parser'] = {
                'filename': 'stunir_parser.py',
                'sha256': parser_hash,
                'size': len(self.parser_code),
                'lines': self.parser_code.count('\n') + 1,
            }
        
        if self.lexer_code:
            lexer_hash = hashlib.sha256(self.lexer_code.encode()).hexdigest()
            manifest['components']['lexer'] = {
                'filename': 'stunir_lexer_gen.py',
                'sha256': lexer_hash,
                'size': len(self.lexer_code),
                'lines': self.lexer_code.count('\n') + 1,
            }
        
        manifest['warnings'] = self._warnings
        manifest['errors'] = self._errors
        
        return manifest
    
    def generate_all(self, save: bool = True) -> GenerationResult:
        """
        Generate all tools (parser and lexer).
        
        Args:
            save: Whether to save generated files
            
        Returns:
            GenerationResult with all outputs and metadata
        """
        result = GenerationResult(success=False)
        
        # Build specifications
        if not self.build_specs():
            result.errors = self._errors.copy()
            return result
        
        # Generate parser
        parser_code = self.generate_parser()
        if parser_code is None:
            result.errors = self._errors.copy()
            return result
        result.parser_code = parser_code
        
        # Generate lexer
        lexer_code = self.generate_lexer()
        if lexer_code is None:
            result.errors = self._errors.copy()
            return result
        result.lexer_code = lexer_code
        
        # Save if requested
        if save:
            parser_path = self.save_parser()
            if parser_path:
                result.parser_path = parser_path
            
            lexer_path = self.save_lexer()
            if lexer_path:
                result.lexer_path = lexer_path
        
        # Generate manifest
        result.manifest = self.generate_manifest()
        result.warnings = self._warnings.copy()
        result.success = True
        
        return result


def generate_stunir_tools(output_dir: Optional[Path] = None) -> GenerationResult:
    """
    Generate STUNIR parser and lexer tools.
    
    Convenience function for tool generation.
    
    Args:
        output_dir: Directory to save files (default: bootstrap/)
        
    Returns:
        GenerationResult with all outputs
    """
    generator = ToolGenerator(output_dir)
    return generator.generate_all()


if __name__ == '__main__':
    # Run tool generation
    import sys
    
    print("Generating STUNIR tools...")
    result = generate_stunir_tools()
    
    if result.success:
        print(f"✅ Parser generated: {result.parser_path}")
        print(f"✅ Lexer generated: {result.lexer_path}")
        print(f"   Parser lines: {result.manifest['components']['parser']['lines']}")
        print(f"   Lexer lines: {result.manifest['components']['lexer']['lines']}")
        
        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings[:10]:
                print(f"  ⚠ {warning}")
        
        sys.exit(0)
    else:
        print("❌ Generation failed:")
        for error in result.errors:
            print(f"  ✗ {error}")
        sys.exit(1)
