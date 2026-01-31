"""Main Semantic IR Parser.

This module provides the main SpecParser class that orchestrates all
parsing stages: specification loading, AST building, semantic analysis,
and IR generation.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from .types import ParseError, ParserOptions, ErrorType
from .ast_builder import ASTBuilder, AST
from .semantic_analyzer import SemanticAnalyzer, AnnotatedAST
from .ir_generator import IRGenerator, SemanticIR


class ValidationResult:
    """Result of IR validation."""
    
    def __init__(self, is_valid: bool, errors: List[ParseError]):
        self.is_valid = is_valid
        self.errors = errors


class SpecParser:
    """Main parser for specifications to Semantic IR.
    
    This parser coordinates all parsing stages:
    1. Specification file loading and validation
    2. AST construction from specification
    3. Semantic analysis and type inference
    4. Semantic IR generation
    """

    def __init__(self, options: ParserOptions):
        """Initialize parser with options.
        
        Args:
            options: Parser configuration options
        """
        self.options = options
        self.errors: List[ParseError] = []
        
        # Initialize stage components
        self.ast_builder = ASTBuilder(options.category)
        self.semantic_analyzer = SemanticAnalyzer(options.category)
        self.ir_generator = IRGenerator(options.category)

    def parse_file(self, path: str) -> Optional[SemanticIR]:
        """Parse specification file to Semantic IR.
        
        Args:
            path: Path to specification file (JSON or YAML)
            
        Returns:
            Semantic IR object, or None if parsing failed
        """
        # Stage 1: Load and validate specification
        spec = self._load_specification(path)
        if spec is None:
            return None
        
        return self.parse_dict(spec, filename=path)

    def parse_string(self, content: str, format: str = "json") -> Optional[SemanticIR]:
        """Parse specification string to Semantic IR.
        
        Args:
            content: Specification content as string
            format: Format of content ("json" or "yaml")
            
        Returns:
            Semantic IR object, or None if parsing failed
        """
        # Parse content based on format
        try:
            if format == "json":
                spec = json.loads(content)
            elif format == "yaml":
                spec = yaml.safe_load(content)
            else:
                self._add_error(
                    ErrorType.SYNTAX,
                    f"Unknown format: {format}",
                    "<input>", 0, 0
                )
                return None
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            self._add_error(
                ErrorType.SYNTAX,
                f"Failed to parse {format}: {e}",
                "<input>", 0, 0
            )
            return None
        
        return self.parse_dict(spec, filename="<input>")

    def parse_dict(self, spec: Dict, filename: str = "<input>") -> Optional[SemanticIR]:
        """Parse specification dictionary to Semantic IR.
        
        Args:
            spec: Specification as dictionary
            filename: Source filename for error reporting
            
        Returns:
            Semantic IR object, or None if parsing failed
        """
        # Update AST builder filename
        self.ast_builder.filename = filename
        
        # Validate category
        spec_category = spec.get("metadata", {}).get("category", "")
        if spec_category != self.options.category:
            self._add_error(
                ErrorType.CATEGORY,
                f"Category mismatch: expected '{self.options.category}', got '{spec_category}'",
                filename, 0, 0
            )
            if len(self.errors) >= self.options.max_errors:
                return None
        
        # Stage 2: Build AST
        ast = self.ast_builder.build_ast(spec)
        self.errors.extend(self.ast_builder.errors)
        
        if len(self.errors) >= self.options.max_errors:
            return None
        
        # Stage 3: Semantic analysis
        if self.options.enable_type_inference:
            annotated_ast = self.semantic_analyzer.analyze(ast)
            self.errors.extend(self.semantic_analyzer.errors)
            
            if len(self.errors) >= self.options.max_errors:
                return None
        else:
            # Skip semantic analysis
            annotated_ast = AnnotatedAST(ast=ast)
        
        # Stage 4: Generate IR
        ir = self.ir_generator.generate_ir(annotated_ast)
        
        # Set generation timestamp
        ir.metadata.generated_at = datetime.utcnow().isoformat() + "Z"
        
        return ir

    def validate_ir(self, ir: SemanticIR) -> ValidationResult:
        """Validate generated IR.
        
        Args:
            ir: Semantic IR to validate
            
        Returns:
            Validation result with any errors found
        """
        errors = []
        
        # Check schema version
        if not ir.schema:
            errors.append(ParseError(
                location=self._make_location("<ir>", 0, 0),
                error_type=ErrorType.SEMANTIC,
                message="Missing schema field in IR",
            ))
        
        # Check metadata
        if not ir.metadata.category:
            errors.append(ParseError(
                location=self._make_location("<ir>", 0, 0),
                error_type=ErrorType.SEMANTIC,
                message="Missing category in metadata",
            ))
        
        # Validate functions
        for func in ir.functions:
            if not func.name:
                errors.append(ParseError(
                    location=self._make_location("<ir>", 0, 0),
                    error_type=ErrorType.SEMANTIC,
                    message="Function missing name",
                ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )

    def get_errors(self) -> List[ParseError]:
        """Get all parsing errors.
        
        Returns:
            List of all errors encountered during parsing
        """
        return self.errors

    def _load_specification(self, path: str) -> Optional[Dict]:
        """Load specification from file.
        
        Args:
            path: Path to specification file
            
        Returns:
            Specification as dictionary, or None if loading failed
        """
        file_path = Path(path)
        
        # Check if file exists
        if not file_path.exists():
            self._add_error(
                ErrorType.SYNTAX,
                f"File not found: {path}",
                path, 0, 0
            )
            return None
        
        # Load based on file extension
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix in ['.json']:
                    spec = json.load(f)
                elif file_path.suffix in ['.yaml', '.yml']:
                    spec = yaml.safe_load(f)
                else:
                    self._add_error(
                        ErrorType.SYNTAX,
                        f"Unknown file format: {file_path.suffix}",
                        path, 0, 0
                    )
                    return None
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            self._add_error(
                ErrorType.SYNTAX,
                f"Failed to parse file: {e}",
                path, 0, 0
            )
            return None
        except IOError as e:
            self._add_error(
                ErrorType.SYNTAX,
                f"Failed to read file: {e}",
                path, 0, 0
            )
            return None
        
        return spec

    def _add_error(self, error_type: ErrorType, message: str, 
                   filename: str, line: int, column: int):
        """Add parsing error."""
        from .types import SourceLocation
        
        error = ParseError(
            location=SourceLocation(filename, line, column),
            error_type=error_type,
            message=message,
        )
        self.errors.append(error)

    def _make_location(self, filename: str, line: int, column: int):
        """Create source location."""
        from .types import SourceLocation
        return SourceLocation(filename, line, column)
