#!/usr/bin/env python3
"""STUNIR Symbolic IR Extensions.

This module provides symbolic programming constructs for STUNIR IR,
including S-expression representation, macro support, quote/unquote
operators, and first-class function support needed by Lisp-family emitters.

Part of Phase 5A: Core Lisp Implementation.
"""

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable


class SymbolicExprKind(Enum):
    """Symbolic expression kinds."""
    QUOTE = "quote"
    QUASIQUOTE = "quasiquote"
    UNQUOTE = "unquote"
    UNQUOTE_SPLICING = "unquote_splicing"
    LAMBDA = "lambda"
    SYMBOL = "symbol"
    LIST = "list"
    CONS = "cons"


class SymbolicStmtKind(Enum):
    """Symbolic statement kinds."""
    DEFMACRO = "defmacro"
    MULTIPLE_VALUE = "multiple_value"


# All symbolic IR kinds
SYMBOLIC_KINDS = {
    'quote', 'quasiquote', 'unquote', 'unquote_splicing',
    'lambda', 'symbol', 'list', 'cons',
    'defmacro', 'multiple_value', 'template_list'
}


@dataclass
class Symbol:
    """Symbolic name reference.
    
    Represents a Lisp symbol with optional package qualification.
    """
    name: str
    package: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        result = {'kind': 'symbol', 'name': self.name}
        if self.package:
            result['package'] = self.package
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Symbol':
        """Create Symbol from IR dictionary."""
        return cls(name=data['name'], package=data.get('package'))
    
    def __str__(self) -> str:
        if self.package:
            return f"{self.package}:{self.name}"
        return self.name


@dataclass
class Atom:
    """An atomic S-expression value.
    
    Atoms are self-evaluating values like numbers, strings, and booleans.
    """
    value: Union[int, float, str, bool, None]
    
    def to_dict(self) -> Any:
        """Convert to IR representation (just the value)."""
        return self.value
    
    def __str__(self) -> str:
        if self.value is None:
            return "nil"
        if isinstance(self.value, bool):
            return "t" if self.value else "nil"
        if isinstance(self.value, str):
            return f'"{self.value}"'
        return str(self.value)


@dataclass
class SList:
    """A list S-expression.
    
    Represents a proper Lisp list of elements.
    """
    elements: List[Any] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        return {
            'kind': 'list',
            'elements': [_to_dict(el) for el in self.elements]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SList':
        """Create SList from IR dictionary."""
        return cls(elements=[_from_dict(el) for el in data.get('elements', [])])
    
    def __len__(self) -> int:
        return len(self.elements)
    
    def __getitem__(self, index: int) -> Any:
        return self.elements[index]
    
    def __str__(self) -> str:
        return '(' + ' '.join(str(el) for el in self.elements) + ')'


@dataclass
class Cons:
    """A cons cell (pair).
    
    The fundamental building block of Lisp lists.
    """
    car: Any
    cdr: Any
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        return {
            'kind': 'cons',
            'car': _to_dict(self.car),
            'cdr': _to_dict(self.cdr)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Cons':
        """Create Cons from IR dictionary."""
        return cls(car=_from_dict(data['car']), cdr=_from_dict(data['cdr']))
    
    def __str__(self) -> str:
        return f'({self.car} . {self.cdr})'


@dataclass
class Quote:
    """A quoted datum.
    
    Prevents evaluation of its contents.
    """
    value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        return {'kind': 'quote', 'value': _to_dict(self.value)}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Quote':
        """Create Quote from IR dictionary."""
        return cls(value=_from_dict(data['value']))
    
    def __str__(self) -> str:
        return f"'{self.value}"


@dataclass
class Quasiquote:
    """A quasiquoted template.
    
    Allows selective evaluation within a quoted template.
    """
    value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        return {'kind': 'quasiquote', 'value': _to_dict(self.value)}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Quasiquote':
        """Create Quasiquote from IR dictionary."""
        return cls(value=_from_dict(data['value']))
    
    def __str__(self) -> str:
        return f"`{self.value}"


@dataclass
class Unquote:
    """An unquoted expression within quasiquote.
    
    Forces evaluation within a quasiquoted template.
    """
    value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        return {'kind': 'unquote', 'value': _to_dict(self.value)}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Unquote':
        """Create Unquote from IR dictionary."""
        return cls(value=_from_dict(data['value']))
    
    def __str__(self) -> str:
        return f",{self.value}"


@dataclass
class UnquoteSplicing:
    """A splice-unquoted expression.
    
    Evaluates and splices a list into the surrounding template.
    """
    value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        return {'kind': 'unquote_splicing', 'value': _to_dict(self.value)}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnquoteSplicing':
        """Create UnquoteSplicing from IR dictionary."""
        return cls(value=_from_dict(data['value']))
    
    def __str__(self) -> str:
        return f",@{self.value}"


@dataclass
class Lambda:
    """An anonymous function (lambda).
    
    Represents a first-class function value.
    """
    params: List[Dict[str, Any]] = field(default_factory=list)
    body: List[Dict[str, Any]] = field(default_factory=list)
    rest_param: Optional[str] = None
    docstring: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        result = {'kind': 'lambda', 'params': self.params, 'body': self.body}
        if self.rest_param:
            result['rest_param'] = self.rest_param
        if self.docstring:
            result['docstring'] = self.docstring
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Lambda':
        """Create Lambda from IR dictionary."""
        return cls(
            params=data.get('params', []),
            body=data.get('body', []),
            rest_param=data.get('rest_param'),
            docstring=data.get('docstring')
        )


@dataclass
class Macro:
    """A macro definition.
    
    Macros are compile-time code transformers.
    """
    name: str
    params: List[Dict[str, Any]] = field(default_factory=list)
    body: List[Dict[str, Any]] = field(default_factory=list)
    rest_param: Optional[str] = None
    docstring: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to IR dictionary representation."""
        result = {
            'kind': 'defmacro',
            'name': self.name,
            'params': self.params,
            'body': self.body
        }
        if self.rest_param:
            result['rest_param'] = self.rest_param
        if self.docstring:
            result['docstring'] = self.docstring
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Macro':
        """Create Macro from IR dictionary."""
        return cls(
            name=data['name'],
            params=data.get('params', []),
            body=data.get('body', []),
            rest_param=data.get('rest_param'),
            docstring=data.get('docstring')
        )


def _to_dict(obj: Any) -> Any:
    """Convert symbolic object to IR dictionary."""
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, list):
        return [_to_dict(el) for el in obj]
    if isinstance(obj, dict):
        return obj
    return obj


def _from_dict(data: Any) -> Any:
    """Convert IR dictionary to symbolic object."""
    if not isinstance(data, dict):
        return data
    
    kind = data.get('kind')
    if kind == 'symbol':
        return Symbol.from_dict(data)
    if kind == 'list':
        return SList.from_dict(data)
    if kind == 'cons':
        return Cons.from_dict(data)
    if kind == 'quote':
        return Quote.from_dict(data)
    if kind == 'quasiquote':
        return Quasiquote.from_dict(data)
    if kind == 'unquote':
        return Unquote.from_dict(data)
    if kind == 'unquote_splicing':
        return UnquoteSplicing.from_dict(data)
    if kind == 'lambda':
        return Lambda.from_dict(data)
    if kind == 'defmacro':
        return Macro.from_dict(data)
    
    return data


class SymbolicIRExtension:
    """Processor for symbolic IR extensions.
    
    Validates, analyzes, and transforms IR with symbolic features
    for use by Lisp-family code emitters.
    """
    
    def __init__(self, schema_path: Optional[str] = None):
        """Initialize the symbolic IR extension processor.
        
        Args:
            schema_path: Optional path to the symbolic_ir.json schema.
        """
        self.schema = self._load_schema(schema_path)
    
    def _load_schema(self, schema_path: Optional[str] = None) -> Dict[str, Any]:
        """Load the symbolic IR JSON schema."""
        if schema_path is None:
            # Find schema relative to this file
            base_dir = Path(__file__).parent.parent.parent
            schema_path = base_dir / 'schemas' / 'symbolic_ir.json'
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return minimal schema if file not found
            return {'definitions': {}}
    
    def validate(self, ir: Dict[str, Any]) -> bool:
        """Validate IR against symbolic schema extension.
        
        Args:
            ir: IR dictionary with potential symbolic features.
            
        Returns:
            True if valid.
            
        Raises:
            ValueError: If IR is invalid.
        """
        # Basic validation - check for required fields
        if not isinstance(ir, dict):
            raise ValueError("IR must be a dictionary")
        
        # Check module name if present
        if 'module' in ir and not isinstance(ir['module'], str):
            raise ValueError("Module name must be a string")
        
        # Validate symbolic constructs
        errors = self._validate_recursive(ir, path=[])
        if errors:
            raise ValueError(f"Symbolic IR validation errors: {'; '.join(errors)}")
        
        return True
    
    def _validate_recursive(self, data: Any, path: List[str]) -> List[str]:
        """Recursively validate symbolic constructs."""
        errors = []
        
        if isinstance(data, dict):
            kind = data.get('kind')
            
            # Validate symbol
            if kind == 'symbol':
                if 'name' not in data:
                    errors.append(f"Symbol at {'.'.join(path)} missing 'name'")
                elif not isinstance(data['name'], str):
                    errors.append(f"Symbol name at {'.'.join(path)} must be string")
            
            # Validate quote
            elif kind == 'quote':
                if 'value' not in data:
                    errors.append(f"Quote at {'.'.join(path)} missing 'value'")
            
            # Validate lambda
            elif kind == 'lambda':
                if 'params' not in data:
                    errors.append(f"Lambda at {'.'.join(path)} missing 'params'")
                if 'body' not in data:
                    errors.append(f"Lambda at {'.'.join(path)} missing 'body'")
            
            # Validate defmacro
            elif kind == 'defmacro':
                if 'name' not in data:
                    errors.append(f"Defmacro at {'.'.join(path)} missing 'name'")
                if 'params' not in data:
                    errors.append(f"Defmacro at {'.'.join(path)} missing 'params'")
                if 'body' not in data:
                    errors.append(f"Defmacro at {'.'.join(path)} missing 'body'")
            
            # Validate list
            elif kind == 'list':
                if 'elements' not in data:
                    errors.append(f"List at {'.'.join(path)} missing 'elements'")
                elif not isinstance(data['elements'], list):
                    errors.append(f"List elements at {'.'.join(path)} must be array")
            
            # Validate cons
            elif kind == 'cons':
                if 'car' not in data:
                    errors.append(f"Cons at {'.'.join(path)} missing 'car'")
                if 'cdr' not in data:
                    errors.append(f"Cons at {'.'.join(path)} missing 'cdr'")
            
            # Recurse into all values
            for key, value in data.items():
                errors.extend(self._validate_recursive(value, path + [key]))
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                errors.extend(self._validate_recursive(item, path + [str(i)]))
        
        return errors
    
    def has_symbolic_features(self, ir: Dict[str, Any]) -> bool:
        """Check if IR contains any symbolic constructs.
        
        Args:
            ir: IR dictionary to check.
            
        Returns:
            True if IR uses any symbolic features.
        """
        return self._contains_kinds(ir, SYMBOLIC_KINDS)
    
    def _contains_kinds(self, data: Any, kinds: set) -> bool:
        """Recursively check if data contains any of the specified kinds."""
        if isinstance(data, dict):
            if data.get('kind') in kinds:
                return True
            for value in data.values():
                if self._contains_kinds(value, kinds):
                    return True
        elif isinstance(data, list):
            for item in data:
                if self._contains_kinds(item, kinds):
                    return True
        return False
    
    def extract_macros(self, ir: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract macro definitions from IR.
        
        Args:
            ir: IR dictionary containing potential macro definitions.
            
        Returns:
            List of macro definition dictionaries with name, params, body.
        """
        macros = []
        
        # Check top-level macros field
        if 'macros' in ir:
            macros.extend(ir['macros'])
        
        # Check definitions array
        for item in ir.get('definitions', []):
            if isinstance(item, dict) and item.get('kind') == 'defmacro':
                macros.append(item)
        
        # Check functions for inline macro definitions
        for func in ir.get('functions', []):
            self._extract_macros_from_body(func.get('body', []), macros)
        
        return macros
    
    def _extract_macros_from_body(self, body: List[Any], macros: List[Dict[str, Any]]):
        """Extract macro definitions from function body."""
        for stmt in body:
            if isinstance(stmt, dict):
                if stmt.get('kind') == 'defmacro':
                    macros.append(stmt)
                # Recurse into nested bodies
                if 'body' in stmt:
                    self._extract_macros_from_body(stmt['body'], macros)
    
    def extract_lambdas(self, ir: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract lambda expressions from IR.
        
        Args:
            ir: IR dictionary.
            
        Returns:
            List of lambda expression dictionaries.
        """
        lambdas = []
        self._extract_kind_recursive(ir, 'lambda', lambdas)
        return lambdas
    
    def extract_quotes(self, ir: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract quote expressions from IR.
        
        Args:
            ir: IR dictionary.
            
        Returns:
            List of quote expression dictionaries.
        """
        quotes = []
        self._extract_kind_recursive(ir, 'quote', quotes)
        self._extract_kind_recursive(ir, 'quasiquote', quotes)
        return quotes
    
    def _extract_kind_recursive(self, data: Any, kind: str, results: List):
        """Recursively extract items of a specific kind."""
        if isinstance(data, dict):
            if data.get('kind') == kind:
                results.append(data)
            for value in data.values():
                self._extract_kind_recursive(value, kind, results)
        elif isinstance(data, list):
            for item in data:
                self._extract_kind_recursive(item, kind, results)
    
    def transform_quotes(self, expr: Dict[str, Any]) -> Dict[str, Any]:
        """Transform quoted expressions for emitters.
        
        Normalizes quote representation for consistent emission.
        
        Args:
            expr: Expression dictionary containing quotes.
            
        Returns:
            Transformed expression dictionary.
        """
        if not isinstance(expr, dict):
            return expr
        
        kind = expr.get('kind')
        
        if kind == 'quote':
            # Ensure value is properly normalized
            return {
                'kind': 'quote',
                'value': self._normalize_quoted_datum(expr.get('value'))
            }
        
        if kind == 'quasiquote':
            return {
                'kind': 'quasiquote',
                'value': self._transform_quasiquote_template(expr.get('value'))
            }
        
        # Transform recursively for other kinds
        result = {}
        for key, value in expr.items():
            if isinstance(value, dict):
                result[key] = self.transform_quotes(value)
            elif isinstance(value, list):
                result[key] = [self.transform_quotes(v) if isinstance(v, dict) else v 
                               for v in value]
            else:
                result[key] = value
        
        return result
    
    def _normalize_quoted_datum(self, datum: Any) -> Any:
        """Normalize a quoted datum."""
        if isinstance(datum, dict):
            kind = datum.get('kind')
            if kind == 'list':
                return {
                    'kind': 'list',
                    'elements': [self._normalize_quoted_datum(el) 
                                for el in datum.get('elements', [])]
                }
            if kind == 'symbol':
                return datum
        return datum
    
    def _transform_quasiquote_template(self, template: Any) -> Any:
        """Transform a quasiquote template."""
        if isinstance(template, dict):
            kind = template.get('kind')
            if kind in ('unquote', 'unquote_splicing'):
                return template
            if kind == 'list':
                return {
                    'kind': 'template_list',
                    'elements': [self._transform_quasiquote_template(el)
                                for el in template.get('elements', [])]
                }
        return template
    
    def to_sexpression_string(self, data: Any) -> str:
        """Convert IR data to S-expression string representation.
        
        Args:
            data: IR data (dict, list, or primitive).
            
        Returns:
            S-expression string.
        """
        if isinstance(data, dict):
            kind = data.get('kind')
            
            if kind == 'symbol':
                pkg = data.get('package')
                name = data.get('name', '')
                return f"{pkg}:{name}" if pkg else name
            
            if kind == 'quote':
                return f"'{self.to_sexpression_string(data.get('value'))}"
            
            if kind == 'quasiquote':
                return f"`{self.to_sexpression_string(data.get('value'))}"
            
            if kind == 'unquote':
                return f",{self.to_sexpression_string(data.get('value'))}"
            
            if kind == 'unquote_splicing':
                return f",@{self.to_sexpression_string(data.get('value'))}"
            
            if kind == 'list':
                elements = data.get('elements', [])
                inner = ' '.join(self.to_sexpression_string(el) for el in elements)
                return f"({inner})"
            
            if kind == 'cons':
                car = self.to_sexpression_string(data.get('car'))
                cdr = self.to_sexpression_string(data.get('cdr'))
                return f"({car} . {cdr})"
            
            if kind == 'lambda':
                params = ' '.join(p.get('name', '_') for p in data.get('params', []))
                return f"(lambda ({params}) ...)"
            
            # Generic dict - treat as association list
            return str(data)
        
        if isinstance(data, list):
            inner = ' '.join(self.to_sexpression_string(el) for el in data)
            return f"({inner})"
        
        if isinstance(data, str):
            return f'"{data}"'
        
        if isinstance(data, bool):
            return 't' if data else 'nil'
        
        if data is None:
            return 'nil'
        
        return str(data)


# Convenience function for building S-expressions
def sexpr(*args) -> SList:
    """Build an S-expression list.
    
    Args:
        *args: Elements of the list.
        
    Returns:
        SList containing the elements.
    """
    elements = []
    for arg in args:
        if isinstance(arg, str):
            elements.append(Symbol(arg))
        elif isinstance(arg, (int, float, bool, type(None))):
            elements.append(Atom(arg))
        else:
            elements.append(arg)
    return SList(elements)


def sym(name: str, package: Optional[str] = None) -> Symbol:
    """Create a symbol."""
    return Symbol(name, package)


def quote(value: Any) -> Quote:
    """Create a quoted value."""
    return Quote(value)


def quasiquote(value: Any) -> Quasiquote:
    """Create a quasiquoted template."""
    return Quasiquote(value)


def unquote(value: Any) -> Unquote:
    """Create an unquoted expression."""
    return Unquote(value)


def unquote_splicing(value: Any) -> UnquoteSplicing:
    """Create a splice-unquoted expression."""
    return UnquoteSplicing(value)


# Export all public symbols
__all__ = [
    'SymbolicExprKind', 'SymbolicStmtKind', 'SYMBOLIC_KINDS',
    'Symbol', 'Atom', 'SList', 'Cons', 'Quote', 'Quasiquote',
    'Unquote', 'UnquoteSplicing', 'Lambda', 'Macro',
    'SymbolicIRExtension',
    'sexpr', 'sym', 'quote', 'quasiquote', 'unquote', 'unquote_splicing'
]
