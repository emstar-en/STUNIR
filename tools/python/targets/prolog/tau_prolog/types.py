#!/usr/bin/env python3
"""Type mapping for Tau Prolog.

Maps STUNIR IR types to Tau Prolog types and provides
JavaScript interoperability type conversions.

Part of Phase 5D-4: Extended Prolog Targets (Tau Prolog).
"""

from typing import Dict, Any, Optional, List, Set


# Type mapping from IR types to Tau Prolog/JS types
TAU_PROLOG_TYPES: Dict[str, str] = {
    # Numeric types
    'i8': 'integer',
    'i16': 'integer',
    'i32': 'integer',
    'i64': 'integer',
    'u8': 'integer',
    'u16': 'integer',
    'u32': 'integer',
    'u64': 'integer',
    'f32': 'float',
    'f64': 'float',
    'number': 'number',
    
    # Boolean
    'bool': 'boolean',
    'boolean': 'boolean',
    
    # String types
    'string': 'atom',
    'str': 'atom',
    'char': 'atom',
    
    # Collections
    'list': 'list',
    'array': 'list',
    
    # JavaScript types
    'js_object': 'term',
    'js_function': 'term',
    'js_array': 'list',
    'js_promise': 'term',
    
    # Special
    'void': 'true',
    'any': 'any',
    'term': 'term',
}


# JavaScript to Prolog type mapping
JS_TO_PROLOG_TYPES: Dict[str, str] = {
    'number': 'number',
    'string': 'atom',
    'boolean': 'atom',  # true/false atoms
    'object': 'term',
    'array': 'list',
    'null': 'null',
    'undefined': 'undefined',
}


# Tau Prolog standard libraries
TAU_LIBRARIES: List[str] = [
    'lists',      # List operations
    'apply',      # Meta-call predicates
    'assoc',      # Association lists
    'charsio',    # Character I/O
    'dom',        # DOM manipulation (browser only)
    'format',     # String formatting
    'js',         # JavaScript interop
    'os',         # OS operations (Node.js)
    'random',     # Random number generation
    'statistics', # Runtime statistics
    'strings',    # String operations
]


# DOM predicates (browser-only) - name: arity
DOM_PREDICATES: Dict[str, int] = {
    'get_by_id': 2,           # get_by_id(+Id, -Element)
    'get_by_class': 2,        # get_by_class(+Class, -Elements)
    'get_by_tag': 2,          # get_by_tag(+Tag, -Elements)
    'create': 2,              # create(+Tag, -Element)
    'get_attr': 3,            # get_attr(+Element, +Attr, -Value)
    'set_attr': 3,            # set_attr(+Element, +Attr, +Value)
    'get_html': 2,            # get_html(+Element, -Html)
    'set_html': 2,            # set_html(+Element, +Html)
    'add_class': 2,           # add_class(+Element, +Class)
    'remove_class': 2,        # remove_class(+Element, +Class)
    'parent_of': 2,           # parent_of(+Element, -Parent)
    'append_child': 2,        # append_child(+Parent, +Child)
    'remove_child': 2,        # remove_child(+Parent, +Child)
    'bind': 4,                # bind(+Element, +Event, +Handler, -Id)
    'unbind': 2,              # unbind(+Element, +Id)
    'style': 3,               # style(+Element, +Property, -Value)
    'set_style': 3,           # set_style(+Element, +Property, +Value)
    'alert': 1,               # alert(+Message)
    'confirm': 2,             # confirm(+Message, -Result)
    'document': 1,            # document(-Document)
    'body': 1,                # body(-Body)
    'head': 1,                # head(-Head)
}


# JavaScript interop predicates - name: arity
JS_PREDICATES: Dict[str, int] = {
    'apply': 4,               # apply(+Obj, +Method, +Args, -Result)
    'prop': 3,                # prop(+Obj, +Property, -Value)
    'set_prop': 3,            # set_prop(+Obj, +Property, +Value)
    'json_prolog': 2,         # json_prolog(+JSON, -Term)
    'prolog_json': 2,         # prolog_json(+Term, -JSON)
    'global': 2,              # global(+Name, -Object)
    'new': 3,                 # new(+Constructor, +Args, -Instance)
    'is_object': 1,           # is_object(+Term)
    'is_array': 1,            # is_array(+Term)
}


# Standard list predicates (from lists library)
LISTS_PREDICATES: Dict[str, int] = {
    'append': 3,
    'member': 2,
    'length': 2,
    'nth0': 3,
    'nth1': 3,
    'last': 2,
    'reverse': 2,
    'flatten': 2,
    'msort': 2,
    'sort': 2,
    'permutation': 2,
    'maplist': 2,
    'maplist': 3,
    'include': 3,
    'exclude': 3,
    'foldl': 4,
    'foldl': 5,
    'sumlist': 2,
    'max_list': 2,
    'min_list': 2,
}


# Mode declarations for predicate arguments
MODE_DECLARATIONS: Dict[str, str] = {
    'input': '+',
    'output': '-',
    'bidirectional': '?',
    'ground': '@',
}


class TauPrologTypeMapper:
    """Maps STUNIR types to Tau Prolog types with JS interop support.
    
    Provides type conversion and library detection for
    generating proper Tau Prolog code.
    """
    
    def __init__(self, enable_js_interop: bool = True):
        """Initialize type mapper.
        
        Args:
            enable_js_interop: Whether to enable JavaScript interop
        """
        self.enable_js_interop = enable_js_interop
        self.type_map = dict(TAU_PROLOG_TYPES)
    
    def map_type(self, ir_type: str) -> str:
        """Map IR type to Tau Prolog type.
        
        Args:
            ir_type: STUNIR IR type name
            
        Returns:
            Tau Prolog type name
        """
        # Handle compound types like list(i32)
        if '(' in ir_type:
            base = ir_type.split('(')[0]
            return self.type_map.get(base, 'term')
        
        return self.type_map.get(ir_type, 'term')
    
    def is_dom_predicate(self, name: str) -> bool:
        """Check if predicate is a DOM operation.
        
        Args:
            name: Predicate name
            
        Returns:
            True if this is a DOM predicate
        """
        return name in DOM_PREDICATES
    
    def is_js_predicate(self, name: str) -> bool:
        """Check if predicate is a JS interop operation.
        
        Args:
            name: Predicate name
            
        Returns:
            True if this is a JS interop predicate
        """
        return name in JS_PREDICATES
    
    def is_lists_predicate(self, name: str) -> bool:
        """Check if predicate requires lists library.
        
        Args:
            name: Predicate name
            
        Returns:
            True if this predicate is from lists library
        """
        return name in LISTS_PREDICATES
    
    def get_required_libraries(self, predicates: Set[str]) -> List[str]:
        """Determine required libraries based on predicates used.
        
        Args:
            predicates: Set of predicate names used in the program
            
        Returns:
            List of required library names
        """
        libs: Set[str] = set()
        
        for pred in predicates:
            if self.is_dom_predicate(pred):
                libs.add('dom')
            if self.is_js_predicate(pred):
                libs.add('js')
            if self.is_lists_predicate(pred):
                libs.add('lists')
        
        return sorted(libs)
    
    def infer_mode(self, param: Dict[str, Any]) -> str:
        """Infer argument mode from parameter info.
        
        Args:
            param: Parameter dictionary with optional 'mode' field
            
        Returns:
            Mode character (+, -, ?, etc.)
        """
        mode = param.get('mode', 'bidirectional')
        return MODE_DECLARATIONS.get(mode, '?')
    
    def format_type_declaration(self, name: str, params: list) -> str:
        """Format a type declaration for a predicate.
        
        Args:
            name: Predicate name
            params: List of parameter dictionaries
            
        Returns:
            Type declaration comment string
        """
        modes = []
        for param in params:
            mode_char = self.infer_mode(param)
            type_name = self.map_type(param.get('type', 'any'))
            modes.append(f"{mode_char}{type_name}")
        
        return f"%% {name}({', '.join(modes)})"
    
    def is_numeric(self, ir_type: str) -> bool:
        """Check if type is numeric.
        
        Args:
            ir_type: IR type name
            
        Returns:
            True if numeric type
        """
        return ir_type in ('i8', 'i16', 'i32', 'i64', 
                          'u8', 'u16', 'u32', 'u64',
                          'f32', 'f64', 'number')
    
    def is_string(self, ir_type: str) -> bool:
        """Check if type is string-like.
        
        Args:
            ir_type: IR type name
            
        Returns:
            True if string-like type
        """
        return ir_type in ('string', 'str', 'char', 'atom')


__all__ = [
    'TauPrologTypeMapper',
    'TAU_PROLOG_TYPES',
    'JS_TO_PROLOG_TYPES',
    'TAU_LIBRARIES',
    'DOM_PREDICATES',
    'JS_PREDICATES',
    'LISTS_PREDICATES',
    'MODE_DECLARATIONS',
]
