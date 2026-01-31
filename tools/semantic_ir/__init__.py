"""STUNIR Semantic IR Python Implementation.

DO-178C Level A Compliant
Pydantic-based data structures with runtime validation.
"""

__version__ = "1.0.0"
__author__ = "STUNIR Team"

from .ir_types import (
    IRPrimitiveType,
    IRNodeKind,
    BinaryOperator,
    UnaryOperator,
    StorageClass,
    VisibilityKind,
    MutabilityKind,
    InlineHint,
    SourceLocation,
    TargetCategory,
    SafetyLevel,
)

from .nodes import (
    IRNodeBase,
    TypeKind,
    TypeReference,
    PrimitiveTypeRef,
    TypeRef,
)

from .expressions import (
    ExpressionNode,
    IntegerLiteral,
    FloatLiteral,
    StringLiteral,
    BoolLiteral,
    VarRef,
    BinaryExpr,
    UnaryExpr,
    FunctionCall,
    MemberExpr,
    ArrayAccess,
    CastExpr,
    TernaryExpr,
)

from .statements import (
    StatementNode,
    BlockStmt,
    ExprStmt,
    IfStmt,
    WhileStmt,
    ForStmt,
    ReturnStmt,
    BreakStmt,
    ContinueStmt,
    VarDeclStmt,
    AssignStmt,
)

from .declarations import (
    DeclarationNode,
    FunctionDecl,
    TypeDecl,
    ConstDecl,
    VarDecl,
)

from .modules import (
    ImportStmt,
    ModuleMetadata,
    IRModule,
)

from .validation import (
    ValidationStatus,
    ValidationResult,
    validate_node_id,
    validate_hash,
    validate_module,
)

__all__ = [
    # Types
    "IRPrimitiveType",
    "IRNodeKind",
    "BinaryOperator",
    "UnaryOperator",
    "StorageClass",
    "VisibilityKind",
    "MutabilityKind",
    "InlineHint",
    "SourceLocation",
    "TargetCategory",
    "SafetyLevel",
    # Nodes
    "IRNodeBase",
    "TypeKind",
    "TypeReference",
    "PrimitiveTypeRef",
    "TypeRef",
    # Expressions
    "ExpressionNode",
    "IntegerLiteral",
    "FloatLiteral",
    "StringLiteral",
    "BoolLiteral",
    "VarRef",
    "BinaryExpr",
    "UnaryExpr",
    "FunctionCall",
    "MemberExpr",
    "ArrayAccess",
    "CastExpr",
    "TernaryExpr",
    # Statements
    "StatementNode",
    "BlockStmt",
    "ExprStmt",
    "IfStmt",
    "WhileStmt",
    "ForStmt",
    "ReturnStmt",
    "BreakStmt",
    "ContinueStmt",
    "VarDeclStmt",
    "AssignStmt",
    # Declarations
    "DeclarationNode",
    "FunctionDecl",
    "TypeDecl",
    "ConstDecl",
    "VarDecl",
    # Modules
    "ImportStmt",
    "ModuleMetadata",
    "IRModule",
    # Validation
    "ValidationStatus",
    "ValidationResult",
    "validate_node_id",
    "validate_hash",
    "validate_module",
]
