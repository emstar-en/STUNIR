"""IR Generator for Semantic IR parser."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import json
import hashlib

from .types import Type, Function, TypeDef, Constant, Parameter, Statement, Expression
from .semantic_analyzer import AnnotatedAST


@dataclass
class IRType:
    """Type in Semantic IR."""
    name: str
    kind: str  # primitive, struct, union, enum, pointer, array
    size: Optional[int] = None
    alignment: Optional[int] = None
    fields: Dict[str, 'IRType'] = field(default_factory=dict)
    element_type: Optional['IRType'] = None


@dataclass
class IRParameter:
    """Function parameter in Semantic IR."""
    name: str
    type: IRType


@dataclass
class IRStatement:
    """Statement in Semantic IR."""
    kind: str
    operands: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IRFunction:
    """Function in Semantic IR."""
    name: str
    parameters: List[IRParameter]
    return_type: IRType
    body: List[IRStatement]
    attributes: Dict[str, Any] = field(default_factory=dict)
    complexity: int = 0


@dataclass
class IRMetadata:
    """Metadata for Semantic IR."""
    version: str
    category: str
    source_hash: str
    generated_at: str
    complexity_metrics: Dict[str, int] = field(default_factory=dict)
    optimization_hints: List[str] = field(default_factory=list)


@dataclass
class SemanticIR:
    """Semantic Intermediate Reference - STUNIR v1 Format."""
    schema: str = "stunir_ir_v1"
    ir_version: str = "v1"
    module_name: str = "unknown"
    docstring: str = ""
    types: List[IRType] = field(default_factory=list)
    functions: List[IRFunction] = field(default_factory=list)
    generated_at: str = ""
    
    # Legacy metadata for backward compatibility
    _metadata: Optional[IRMetadata] = field(default=None, repr=False)
    _category: str = field(default="unknown", repr=False)
    
    @property
    def metadata(self) -> IRMetadata:
        """Get legacy metadata (for backward compatibility)."""
        if self._metadata is None:
            self._metadata = IRMetadata(
                version=self.ir_version,
                category=self._category,
                source_hash="",
                generated_at=self.generated_at,
            )
        return self._metadata
    
    @metadata.setter
    def metadata(self, value: IRMetadata):
        """Set legacy metadata (for backward compatibility)."""
        self._metadata = value
        self.ir_version = value.version
        self.generated_at = value.generated_at
        self._category = value.category

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary in STUNIR v1 format."""
        # Convert to STUNIR v1 format (matching SPARK output)
        result = {
            "schema": self.schema,
            "ir_version": self.ir_version,
            "module_name": self.module_name,
            "docstring": self.docstring,
            "types": self.types,
            "functions": [],
            "generated_at": self.generated_at,
        }
        
        # Convert functions to dict format
        for func in self.functions:
            func_dict = {
                "name": func.name,
                "args": [
                    {"name": p.name, "type": p.type.name}
                    for p in func.parameters
                ],
                "return_type": func.return_type.name,
                "steps": [
                    {"kind": stmt.kind, "data": str(stmt.operands)}
                    for stmt in func.body
                ],
            }
            result["functions"].append(func_dict)
        
        return result

    def to_json(self, pretty: bool = True) -> str:
        """Convert to JSON string."""
        data = self.to_dict()
        if pretty:
            return json.dumps(data, indent=2, sort_keys=True)
        return json.dumps(data, sort_keys=True)


class IRGenerator:
    """Generates Semantic IR from annotated AST."""

    def __init__(self, category: str):
        self.category = category

    def generate_ir(self, ast: AnnotatedAST) -> SemanticIR:
        """Generate Semantic IR from annotated AST."""
        ir = SemanticIR()
        
        # Set STUNIR v1 fields
        ir.schema = "stunir_ir_v1"
        ir.ir_version = "v1"
        ir.module_name = getattr(ast.ast, 'module_name', self.category)
        ir.docstring = getattr(ast.ast, 'docstring', f"{self.category} module")
        ir._category = self.category  # Set internal category for backward compatibility
        
        # Generate types
        for typedef in ast.ast.types:
            ir_type = self.generate_type_ir(typedef.type)
            ir.types.append(ir_type)
        
        # Generate functions
        for func in ast.ast.functions:
            ir_func = self.generate_function_ir(func)
            ir.functions.append(ir_func)
        
        return ir

    def generate_function_ir(self, func: Function) -> IRFunction:
        """Generate IR for a function."""
        # Generate parameters
        ir_params = []
        for param in func.parameters:
            ir_type = self._convert_type(param.type)
            ir_params.append(IRParameter(name=param.name, type=ir_type))
        
        # Generate return type
        ir_return_type = self._convert_type(func.return_type)
        
        # Generate body
        ir_body = []
        for stmt in func.body:
            ir_stmt = self._convert_statement(stmt)
            ir_body.append(ir_stmt)
        
        # Compute function complexity
        complexity = self._compute_complexity(func)
        
        # Set attributes
        attributes = {}
        if func.is_inline:
            attributes["inline"] = True
        if func.is_static:
            attributes["static"] = True
        
        return IRFunction(
            name=func.name,
            parameters=ir_params,
            return_type=ir_return_type,
            body=ir_body,
            attributes=attributes,
            complexity=complexity,
        )

    def generate_type_ir(self, type_obj: Type) -> IRType:
        """Generate IR for a type."""
        return self._convert_type(type_obj)

    def compute_metadata(self, ir: SemanticIR) -> Dict[str, Any]:
        """Compute metadata for IR."""
        metadata = {
            "complexity": {},
            "hints": [],
        }
        
        # Compute total complexity
        total_complexity = sum(func.complexity for func in ir.functions)
        metadata["complexity"]["total"] = total_complexity
        metadata["complexity"]["functions"] = len(ir.functions)
        metadata["complexity"]["types"] = len(ir.types)
        
        # Generate optimization hints
        for func in ir.functions:
            if func.complexity > 20:
                metadata["hints"].append(
                    f"Function '{func.name}' has high complexity ({func.complexity}), consider refactoring"
                )
        
        return metadata

    def _convert_type(self, type_obj: Type) -> IRType:
        """Convert AST type to IR type."""
        kind = "primitive" if type_obj.is_primitive else "struct"
        if type_obj.is_pointer:
            kind = "pointer"
        elif type_obj.is_array:
            kind = "array"
        
        ir_type = IRType(
            name=type_obj.name,
            kind=kind,
        )
        
        # Set element type for pointers/arrays
        if type_obj.element_type:
            ir_type.element_type = self._convert_type(type_obj.element_type)
        
        # Set fields for structs
        for field_name, field_type in type_obj.fields.items():
            ir_type.fields[field_name] = self._convert_type(field_type)
        
        return ir_type

    def _convert_statement(self, stmt: Statement) -> IRStatement:
        """Convert AST statement to IR statement."""
        operands = []
        
        # Convert expressions to operands
        for expr in stmt.expressions:
            operands.append(self._convert_expression(expr))
        
        return IRStatement(
            kind=stmt.kind,
            operands=operands,
        )

    def _convert_expression(self, expr: Expression) -> Dict[str, Any]:
        """Convert AST expression to IR operand."""
        return {
            "kind": expr.kind,
            "value": expr.value,
        }

    def _compute_complexity(self, func: Function) -> int:
        """Compute cyclomatic complexity of function."""
        # Simplified complexity calculation
        complexity = 1  # Base complexity
        
        for stmt in func.body:
            if stmt.kind in ["if", "while", "for"]:
                complexity += 1
        
        return complexity

    def _compute_ir_hash(self, ir: SemanticIR) -> str:
        """Compute SHA-256 hash of IR."""
        # Serialize IR to canonical JSON
        json_str = ir.to_json(pretty=False)
        
        # Compute SHA-256 hash
        hash_obj = hashlib.sha256(json_str.encode('utf-8'))
        return hash_obj.hexdigest()
