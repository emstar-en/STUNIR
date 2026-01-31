"""STUNIR Semantic IR Parser Package

This package provides parsers for transforming high-level specifications
into semantically-rich Intermediate Reference (IR).
"""

__version__ = "1.0.0"

from .parser import SpecParser, ParseError, ParserOptions
from .ast_builder import ASTBuilder, AST, ASTNode
from .semantic_analyzer import SemanticAnalyzer, AnnotatedAST
from .ir_generator import IRGenerator, SemanticIR

__all__ = [
    "SpecParser",
    "ParseError",
    "ParserOptions",
    "ASTBuilder",
    "AST",
    "ASTNode",
    "SemanticAnalyzer",
    "AnnotatedAST",
    "IRGenerator",
    "SemanticIR",
]
