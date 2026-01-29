"""STUNIR Semantic Analysis Module.

Provides comprehensive semantic analysis including variable tracking,
expression analysis, dead code detection, and constant evaluation.
"""

from .analyzer import (
    AnalysisErrorKind, WarningSeverity, SemanticIssue,
    VariableInfo, FunctionInfo, SymbolTable, SemanticAnalyzer
)

from .expression import (
    OperatorPrecedence, OperatorAssociativity, OperatorInfo, OPERATORS,
    Expression, LiteralExpr, VariableExpr, BinaryExpr, UnaryExpr,
    CallExpr, IndexExpr, MemberExpr, TernaryExpr, CastExpr,
    ExpressionParser, ConstantFolder, CommonSubexpressionEliminator,
    ExpressionEmitter
)

from .checker import (
    CheckKind, CheckResult,
    DeadCodeDetector, UnreachableCodeDetector, ConstantExpressionEvaluator,
    SemanticChecker
)

__all__ = [
    # Analyzer
    'AnalysisErrorKind', 'WarningSeverity', 'SemanticIssue',
    'VariableInfo', 'FunctionInfo', 'SymbolTable', 'SemanticAnalyzer',
    # Expression
    'OperatorPrecedence', 'OperatorAssociativity', 'OperatorInfo', 'OPERATORS',
    'Expression', 'LiteralExpr', 'VariableExpr', 'BinaryExpr', 'UnaryExpr',
    'CallExpr', 'IndexExpr', 'MemberExpr', 'TernaryExpr', 'CastExpr',
    'ExpressionParser', 'ConstantFolder', 'CommonSubexpressionEliminator',
    'ExpressionEmitter',
    # Checker
    'CheckKind', 'CheckResult',
    'DeadCodeDetector', 'UnreachableCodeDetector', 'ConstantExpressionEvaluator',
    'SemanticChecker'
]
