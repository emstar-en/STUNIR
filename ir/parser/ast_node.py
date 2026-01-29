#!/usr/bin/env python3
"""AST node type definitions for generated parsers.

This module provides:
- ASTNodeSpec: Specification for an AST node type
- ASTSchema: Complete AST schema for a grammar
- Utilities for generating AST schemas from grammars
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

# Import from grammar module - will be available after Phase 6A
try:
    from ir.grammar.production import ProductionRule
except ImportError:
    ProductionRule = Any  # Type stub for development


@dataclass
class ASTNodeSpec:
    """Specification for an AST node type.
    
    Represents a single AST node type that can be generated from
    a grammar production or non-terminal.
    
    Attributes:
        name: Node type name (e.g., "BinaryExpr")
        fields: List of (field_name, field_type) tuples
        base_class: Optional base class name
        production: Associated production rule (if any)
        is_abstract: Whether this is an abstract base class
    
    Example:
        >>> node = ASTNodeSpec("BinaryExpr")
        >>> node.add_field("left", "Expr")
        >>> node.add_field("operator", "Token")
        >>> node.add_field("right", "Expr")
    """
    name: str
    fields: List[Tuple[str, str]] = field(default_factory=list)
    base_class: Optional[str] = None
    production: Optional[ProductionRule] = None
    is_abstract: bool = False
    
    def add_field(self, name: str, field_type: str) -> None:
        """Add a field to the node.
        
        Args:
            name: Field name
            field_type: Field type string
        """
        self.fields.append((name, field_type))
    
    def get_field(self, name: str) -> Optional[Tuple[str, str]]:
        """Get a field by name.
        
        Args:
            name: Field name to find
        
        Returns:
            Tuple of (name, type) or None if not found
        """
        for fname, ftype in self.fields:
            if fname == name:
                return (fname, ftype)
        return None
    
    def field_count(self) -> int:
        """Get number of fields."""
        return len(self.fields)
    
    def __str__(self) -> str:
        fields_str = ", ".join(f"{n}: {t}" for n, t in self.fields)
        base_str = f" extends {self.base_class}" if self.base_class else ""
        return f"{self.name}{base_str} {{ {fields_str} }}"


@dataclass
class ASTSchema:
    """Complete AST schema for a grammar.
    
    Contains all AST node specifications needed to represent
    parse trees for a grammar.
    
    Attributes:
        nodes: List of AST node specifications
        base_node_name: Name of the base AST node class
        token_type_name: Name of the token type
        metadata: Additional schema metadata
    
    Example:
        >>> schema = ASTSchema()
        >>> schema.add_node(ASTNodeSpec("ExprNode", is_abstract=True))
        >>> schema.add_node(ASTNodeSpec("BinaryExprNode", base_class="ExprNode"))
    """
    nodes: List[ASTNodeSpec] = field(default_factory=list)
    base_node_name: str = "ASTNode"
    token_type_name: str = "Token"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_node(self, node: ASTNodeSpec) -> None:
        """Add a node specification to the schema.
        
        Args:
            node: The node specification to add
        """
        self.nodes.append(node)
    
    def get_node(self, name: str) -> Optional[ASTNodeSpec]:
        """Get a node specification by name.
        
        Args:
            name: Node name to find
        
        Returns:
            ASTNodeSpec or None if not found
        """
        for node in self.nodes:
            if node.name == name:
                return node
        return None
    
    def get_nodes_by_base(self, base_class: str) -> List[ASTNodeSpec]:
        """Get all nodes with a specific base class.
        
        Args:
            base_class: Base class name to filter by
        
        Returns:
            List of matching node specifications
        """
        return [n for n in self.nodes if n.base_class == base_class]
    
    def get_abstract_nodes(self) -> List[ASTNodeSpec]:
        """Get all abstract node specifications."""
        return [n for n in self.nodes if n.is_abstract]
    
    def get_concrete_nodes(self) -> List[ASTNodeSpec]:
        """Get all concrete (non-abstract) node specifications."""
        return [n for n in self.nodes if not n.is_abstract]
    
    def node_count(self) -> int:
        """Get total number of nodes."""
        return len(self.nodes)
    
    def validate(self) -> List[str]:
        """Validate the schema for consistency.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        node_names = {n.name for n in self.nodes}
        
        for node in self.nodes:
            # Check base class exists
            if node.base_class and node.base_class not in node_names:
                if node.base_class != self.base_node_name:
                    errors.append(f"Node '{node.name}' has unknown base class '{node.base_class}'")
            
            # Check for duplicate field names
            field_names = [f[0] for f in node.fields]
            if len(field_names) != len(set(field_names)):
                errors.append(f"Node '{node.name}' has duplicate field names")
        
        return errors
    
    def __str__(self) -> str:
        lines = [f"ASTSchema (base: {self.base_node_name})"]
        for node in self.nodes:
            prefix = "[abstract] " if node.is_abstract else ""
            lines.append(f"  {prefix}{node}")
        return "\n".join(lines)


def to_pascal_case(name: str) -> str:
    """Convert a name to PascalCase.
    
    Args:
        name: Input name (may contain - or _)
    
    Returns:
        PascalCase version of the name
    """
    parts = name.replace('-', '_').split('_')
    return ''.join(word.capitalize() for word in parts)


def to_snake_case(name: str) -> str:
    """Convert a name to snake_case.
    
    Args:
        name: Input name (may be PascalCase or contain -)
    
    Returns:
        snake_case version of the name
    """
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append('_')
        result.append(char.lower())
    return ''.join(result).replace('-', '_')


def generate_ast_schema(grammar: Any, naming_convention: str = "pascal") -> ASTSchema:
    """Generate AST node specifications from a grammar.
    
    Creates an ASTNodeSpec for each non-terminal, with fields
    based on production body elements.
    
    Args:
        grammar: The grammar (Grammar object from ir.grammar)
        naming_convention: "pascal" for PascalCase, "snake" for snake_case
    
    Returns:
        ASTSchema with node specifications
    """
    schema = ASTSchema()
    
    name_func = to_pascal_case if naming_convention == "pascal" else to_snake_case
    
    for nonterminal in sorted(grammar.nonterminals, key=lambda s: s.name):
        productions = grammar.get_productions(nonterminal)
        
        if len(productions) == 0:
            continue
        
        if len(productions) == 1:
            # Single production: create node with fields from body
            prod = productions[0]
            node_name = name_func(nonterminal.name) + "Node"
            node = ASTNodeSpec(
                name=node_name,
                production=prod
            )
            
            # Add fields for each body element
            for i, sym in enumerate(prod.body):
                if sym.is_epsilon():
                    continue
                
                if sym.is_nonterminal():
                    field_type = name_func(sym.name) + "Node"
                else:
                    field_type = schema.token_type_name
                
                # Use symbol name as field name if available
                field_name = to_snake_case(sym.name) if sym.name else f"child{i}"
                node.add_field(field_name, field_type)
            
            schema.add_node(node)
        
        else:
            # Multiple productions: create abstract base and variants
            base_name = name_func(nonterminal.name) + "Node"
            base_node = ASTNodeSpec(
                name=base_name,
                is_abstract=True
            )
            schema.add_node(base_node)
            
            for i, prod in enumerate(productions):
                # Use production label if available, otherwise use Alt{i}
                label = prod.label if prod.label else f"Alt{i}"
                variant_name = name_func(nonterminal.name) + name_func(label) + "Node"
                
                variant = ASTNodeSpec(
                    name=variant_name,
                    base_class=base_name,
                    production=prod
                )
                
                # Add fields for each body element
                for j, sym in enumerate(prod.body):
                    if sym.is_epsilon():
                        continue
                    
                    if sym.is_nonterminal():
                        field_type = name_func(sym.name) + "Node"
                    else:
                        field_type = schema.token_type_name
                    
                    field_name = to_snake_case(sym.name) if sym.name else f"child{j}"
                    variant.add_field(field_name, field_type)
                
                schema.add_node(variant)
    
    return schema
