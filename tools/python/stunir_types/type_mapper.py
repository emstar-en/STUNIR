#!/usr/bin/env python3
"""STUNIR Type Mapper Module.

Provides cross-language type mapping for Python, Rust, Haskell, C89/C99,
and Assembly targets. Handles type equivalence, coercion, and conversion.

This module is part of the STUNIR code generation enhancement suite.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
from enum import Enum, auto

from .type_system import (
    STUNIRType, TypeKind, Mutability, Ownership,
    VoidType, UnitType, BoolType, IntType, FloatType, CharType, StringType,
    PointerType, ReferenceType, ArrayType, SliceType,
    StructType, StructField, UnionType,
    EnumType, EnumVariant, TaggedUnionType, TaggedVariant,
    FunctionType, ClosureType,
    GenericType, TypeVar, OpaqueType, RecursiveType,
    OptionalType, ResultType, TupleType,
    TypeRegistry
)


class TargetLanguage(Enum):
    """Supported target languages."""
    PYTHON = auto()
    RUST = auto()
    HASKELL = auto()
    C89 = auto()
    C99 = auto()
    ASM_X86 = auto()
    ASM_ARM = auto()
    GO = auto()
    JAVA = auto()


@dataclass
class MappedType:
    """Result of type mapping to a target language."""
    code: str  # The type as written in target language
    imports: List[str] = None  # Required imports
    declarations: List[str] = None  # Required type declarations
    size_bytes: Optional[int] = None  # Size in bytes if known
    alignment: Optional[int] = None  # Alignment in bytes if known
    needs_allocation: bool = False  # True if heap allocation needed
    notes: List[str] = None  # Additional notes for codegen
    
    def __post_init__(self):
        self.imports = self.imports or []
        self.declarations = self.declarations or []
        self.notes = self.notes or []


class TypeMapper:
    """Maps STUNIR types to target language types."""
    
    def __init__(self, target: TargetLanguage):
        self.target = target
        self._mappers: Dict[TypeKind, Callable] = {
            TypeKind.VOID: self._map_void,
            TypeKind.UNIT: self._map_unit,
            TypeKind.BOOL: self._map_bool,
            TypeKind.INT: self._map_int,
            TypeKind.FLOAT: self._map_float,
            TypeKind.CHAR: self._map_char,
            TypeKind.STRING: self._map_string,
            TypeKind.POINTER: self._map_pointer,
            TypeKind.REFERENCE: self._map_reference,
            TypeKind.ARRAY: self._map_array,
            TypeKind.SLICE: self._map_slice,
            TypeKind.STRUCT: self._map_struct,
            TypeKind.UNION: self._map_union,
            TypeKind.ENUM: self._map_enum,
            TypeKind.TAGGED_UNION: self._map_tagged_union,
            TypeKind.FUNCTION: self._map_function,
            TypeKind.CLOSURE: self._map_closure,
            TypeKind.GENERIC: self._map_generic,
            TypeKind.TYPE_VAR: self._map_type_var,
            TypeKind.OPAQUE: self._map_opaque,
            TypeKind.RECURSIVE: self._map_recursive,
            TypeKind.OPTIONAL: self._map_optional,
            TypeKind.RESULT: self._map_result,
            TypeKind.TUPLE: self._map_tuple,
        }
    
    def map_type(self, typ: STUNIRType) -> MappedType:
        """Map a STUNIR type to target language type."""
        mapper = self._mappers.get(typ.kind)
        if mapper:
            return mapper(typ)
        return MappedType(code='/* unknown type */')
    
    # Void type mapping
    def _map_void(self, typ: VoidType) -> MappedType:
        mappings = {
            TargetLanguage.PYTHON: 'None',
            TargetLanguage.RUST: '()',
            TargetLanguage.HASKELL: '()',
            TargetLanguage.C89: 'void',
            TargetLanguage.C99: 'void',
            TargetLanguage.ASM_X86: '; void',
            TargetLanguage.ASM_ARM: '; void',
            TargetLanguage.GO: '',
            TargetLanguage.JAVA: 'void',
        }
        return MappedType(code=mappings.get(self.target, 'void'), size_bytes=0)
    
    def _map_unit(self, typ: UnitType) -> MappedType:
        mappings = {
            TargetLanguage.PYTHON: 'None',
            TargetLanguage.RUST: '()',
            TargetLanguage.HASKELL: '()',
            TargetLanguage.C89: 'void',
            TargetLanguage.C99: 'void',
            TargetLanguage.ASM_X86: '; unit',
            TargetLanguage.ASM_ARM: '; unit',
            TargetLanguage.GO: 'struct{}',
            TargetLanguage.JAVA: 'Void',
        }
        return MappedType(code=mappings.get(self.target, 'void'), size_bytes=0)
    
    def _map_bool(self, typ: BoolType) -> MappedType:
        mappings = {
            TargetLanguage.PYTHON: 'bool',
            TargetLanguage.RUST: 'bool',
            TargetLanguage.HASKELL: 'Bool',
            TargetLanguage.C89: 'int',  # C89 has no bool
            TargetLanguage.C99: '_Bool',
            TargetLanguage.ASM_X86: 'BYTE',
            TargetLanguage.ASM_ARM: '.byte',
            TargetLanguage.GO: 'bool',
            TargetLanguage.JAVA: 'boolean',
        }
        imports = []
        if self.target == TargetLanguage.C99:
            imports = ['<stdbool.h>']
        return MappedType(code=mappings.get(self.target, 'bool'), 
                         size_bytes=1, imports=imports)
    
    def _map_int(self, typ: IntType) -> MappedType:
        bits = typ.bits
        signed = typ.signed
        
        if self.target == TargetLanguage.PYTHON:
            return MappedType(code='int', imports=['from typing import int'])
        
        elif self.target == TargetLanguage.RUST:
            prefix = 'i' if signed else 'u'
            return MappedType(code=f'{prefix}{bits}', size_bytes=bits // 8)
        
        elif self.target == TargetLanguage.HASKELL:
            haskell_types = {
                (8, True): 'Int8', (8, False): 'Word8',
                (16, True): 'Int16', (16, False): 'Word16',
                (32, True): 'Int32', (32, False): 'Word32',
                (64, True): 'Int64', (64, False): 'Word64',
            }
            code = haskell_types.get((bits, signed), 'Int')
            imports = ['import Data.Int', 'import Data.Word'] if bits != 32 else []
            return MappedType(code=code, imports=imports, size_bytes=bits // 8)
        
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            c_types = {
                (8, True): 'int8_t', (8, False): 'uint8_t',
                (16, True): 'int16_t', (16, False): 'uint16_t',
                (32, True): 'int32_t', (32, False): 'uint32_t',
                (64, True): 'int64_t', (64, False): 'uint64_t',
            }
            code = c_types.get((bits, signed))
            if code is None:
                # Fallback for non-standard sizes
                if bits <= 8:
                    code = 'signed char' if signed else 'unsigned char'
                elif bits <= 16:
                    code = 'short' if signed else 'unsigned short'
                elif bits <= 32:
                    code = 'int' if signed else 'unsigned int'
                else:
                    code = 'long long' if signed else 'unsigned long long'
            imports = ['<stdint.h>'] if 'int' in str(code) and '_t' in str(code) else []
            return MappedType(code=code, imports=imports, size_bytes=bits // 8)
        
        elif self.target in (TargetLanguage.ASM_X86, TargetLanguage.ASM_ARM):
            asm_types = {8: 'BYTE', 16: 'WORD', 32: 'DWORD', 64: 'QWORD'}
            code = asm_types.get(bits, 'DWORD')
            return MappedType(code=code, size_bytes=bits // 8)
        
        elif self.target == TargetLanguage.GO:
            prefix = 'int' if signed else 'uint'
            return MappedType(code=f'{prefix}{bits}', size_bytes=bits // 8)
        
        elif self.target == TargetLanguage.JAVA:
            java_types = {
                (8, True): 'byte', (16, True): 'short',
                (32, True): 'int', (64, True): 'long',
            }
            code = java_types.get((bits, signed), 'int')
            return MappedType(code=code, size_bytes=bits // 8)
        
        return MappedType(code='int', size_bytes=4)
    
    def _map_float(self, typ: FloatType) -> MappedType:
        bits = typ.bits
        
        mappings = {
            TargetLanguage.PYTHON: 'float',
            TargetLanguage.RUST: f'f{bits}',
            TargetLanguage.HASKELL: 'Float' if bits == 32 else 'Double',
            TargetLanguage.C89: 'float' if bits == 32 else 'double',
            TargetLanguage.C99: 'float' if bits == 32 else 'double',
            TargetLanguage.ASM_X86: 'REAL4' if bits == 32 else 'REAL8',
            TargetLanguage.ASM_ARM: '.float' if bits == 32 else '.double',
            TargetLanguage.GO: 'float32' if bits == 32 else 'float64',
            TargetLanguage.JAVA: 'float' if bits == 32 else 'double',
        }
        return MappedType(code=mappings.get(self.target, 'double'), 
                         size_bytes=bits // 8)
    
    def _map_char(self, typ: CharType) -> MappedType:
        mappings = {
            TargetLanguage.PYTHON: 'str',
            TargetLanguage.RUST: 'char',
            TargetLanguage.HASKELL: 'Char',
            TargetLanguage.C89: 'char',
            TargetLanguage.C99: 'char',
            TargetLanguage.ASM_X86: 'BYTE',
            TargetLanguage.ASM_ARM: '.byte',
            TargetLanguage.GO: 'rune',
            TargetLanguage.JAVA: 'char',
        }
        size = 4 if typ.unicode else 1  # Unicode char is 4 bytes in Rust
        return MappedType(code=mappings.get(self.target, 'char'), size_bytes=size)
    
    def _map_string(self, typ: StringType) -> MappedType:
        if self.target == TargetLanguage.PYTHON:
            return MappedType(code='str')
        elif self.target == TargetLanguage.RUST:
            return MappedType(code='String' if typ.owned else '&str',
                            needs_allocation=typ.owned)
        elif self.target == TargetLanguage.HASKELL:
            return MappedType(code='String' if typ.owned else 'Text',
                            imports=['import Data.Text'] if not typ.owned else [])
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            return MappedType(code='char*', needs_allocation=typ.owned)
        elif self.target in (TargetLanguage.ASM_X86, TargetLanguage.ASM_ARM):
            return MappedType(code='; string ptr', size_bytes=8)
        elif self.target == TargetLanguage.GO:
            return MappedType(code='string')
        elif self.target == TargetLanguage.JAVA:
            return MappedType(code='String')
        return MappedType(code='string')
    
    def _map_pointer(self, typ: PointerType) -> MappedType:
        pointee = self.map_type(typ.pointee)
        
        if self.target == TargetLanguage.PYTHON:
            # Python uses ctypes for pointers
            return MappedType(
                code=f'ctypes.POINTER({pointee.code})',
                imports=['import ctypes'],
                needs_allocation=True
            )
        elif self.target == TargetLanguage.RUST:
            mut = 'mut' if typ.mutability == Mutability.MUTABLE else 'const'
            return MappedType(code=f'*{mut} {pointee.code}', size_bytes=8)
        elif self.target == TargetLanguage.HASKELL:
            return MappedType(
                code=f'Ptr {pointee.code}',
                imports=['import Foreign.Ptr'],
                size_bytes=8
            )
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            return MappedType(code=f'{pointee.code}*', size_bytes=8)
        elif self.target in (TargetLanguage.ASM_X86, TargetLanguage.ASM_ARM):
            return MappedType(code='QWORD', size_bytes=8)
        elif self.target == TargetLanguage.GO:
            return MappedType(code=f'*{pointee.code}', size_bytes=8)
        elif self.target == TargetLanguage.JAVA:
            # Java doesn't have direct pointers
            return MappedType(code=f'{pointee.code}', 
                            notes=['Java: wrapped in reference type'])
        return MappedType(code='void*', size_bytes=8)
    
    def _map_reference(self, typ: ReferenceType) -> MappedType:
        referent = self.map_type(typ.referent)
        
        if self.target == TargetLanguage.PYTHON:
            return MappedType(code=referent.code, 
                            notes=['Python: all objects are references'])
        elif self.target == TargetLanguage.RUST:
            lifetime = f'{typ.lifetime} ' if typ.lifetime else ''
            mut = 'mut ' if typ.mutability == Mutability.MUTABLE else ''
            return MappedType(code=f'&{lifetime}{mut}{referent.code}', size_bytes=8)
        elif self.target == TargetLanguage.HASKELL:
            return MappedType(
                code=f'IORef {referent.code}',
                imports=['import Data.IORef']
            )
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            # C uses pointers for references
            return MappedType(code=f'{referent.code}*', size_bytes=8)
        return MappedType(code=f'&{referent.code}', size_bytes=8)
    
    def _map_array(self, typ: ArrayType) -> MappedType:
        element = self.map_type(typ.element)
        
        if self.target == TargetLanguage.PYTHON:
            if typ.size is not None:
                return MappedType(
                    code=f'List[{element.code}]',
                    imports=['from typing import List'],
                    notes=[f'Fixed size: {typ.size}']
                )
            return MappedType(code=f'List[{element.code}]',
                            imports=['from typing import List'])
        elif self.target == TargetLanguage.RUST:
            if typ.size is not None:
                return MappedType(
                    code=f'[{element.code}; {typ.size}]',
                    size_bytes=element.size_bytes * typ.size if element.size_bytes else None
                )
            return MappedType(code=f'Vec<{element.code}>', needs_allocation=True)
        elif self.target == TargetLanguage.HASKELL:
            return MappedType(
                code=f'Array Int {element.code}',
                imports=['import Data.Array']
            )
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            if typ.size is not None:
                return MappedType(code=f'{element.code}[{typ.size}]')
            return MappedType(code=f'{element.code}*')  # VLA or dynamic
        elif self.target in (TargetLanguage.ASM_X86, TargetLanguage.ASM_ARM):
            if typ.size is not None:
                return MappedType(
                    code=f'TIMES {typ.size} {element.code}',
                    size_bytes=element.size_bytes * typ.size if element.size_bytes else None
                )
            return MappedType(code='; dynamic array')
        elif self.target == TargetLanguage.GO:
            if typ.size is not None:
                return MappedType(code=f'[{typ.size}]{element.code}')
            return MappedType(code=f'[]{element.code}')
        elif self.target == TargetLanguage.JAVA:
            return MappedType(code=f'{element.code}[]')
        return MappedType(code=f'{element.code}[]')
    
    def _map_slice(self, typ: SliceType) -> MappedType:
        element = self.map_type(typ.element)
        
        if self.target == TargetLanguage.PYTHON:
            return MappedType(code=f'List[{element.code}]',
                            imports=['from typing import List'])
        elif self.target == TargetLanguage.RUST:
            lifetime = f'{typ.lifetime} ' if typ.lifetime else ''
            mut = 'mut ' if typ.mutability == Mutability.MUTABLE else ''
            return MappedType(code=f'&{lifetime}{mut}[{element.code}]', size_bytes=16)
        elif self.target == TargetLanguage.HASKELL:
            return MappedType(code=f'[{element.code}]')
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            # C doesn't have slices, use struct with ptr + len
            return MappedType(
                code=f'struct {{ {element.code} *ptr; size_t len; }}',
                imports=['<stddef.h>'],
                size_bytes=16
            )
        elif self.target == TargetLanguage.GO:
            return MappedType(code=f'[]{element.code}')
        return MappedType(code=f'[]{element.code}')
    
    def _map_struct(self, typ: StructType) -> MappedType:
        if self.target == TargetLanguage.PYTHON:
            fields = ', '.join(f"'{f.name}': {self.map_type(f.type).code}" 
                              for f in typ.fields)
            code = f"TypedDict('{typ.name}', {{{fields}}})"
            return MappedType(
                code=typ.name,
                imports=['from typing import TypedDict'],
                declarations=[code]
            )
        elif self.target == TargetLanguage.RUST:
            fields = '\n    '.join(
                f'{f.name}: {self.map_type(f.type).code},'
                for f in typ.fields
            )
            decl = f'struct {typ.name} {{\n    {fields}\n}}'
            return MappedType(code=typ.name, declarations=[decl])
        elif self.target == TargetLanguage.HASKELL:
            fields = ', '.join(
                f'{f.name} :: {self.map_type(f.type).code}'
                for f in typ.fields
            )
            decl = f'data {typ.name} = {typ.name} {{ {fields} }}'
            return MappedType(code=typ.name, declarations=[decl])
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            packed = '__attribute__((packed)) ' if typ.packed else ''
            fields = '\n    '.join(
                f'{self.map_type(f.type).code} {f.name};'
                for f in typ.fields
            )
            decl = f'{packed}struct {typ.name} {{\n    {fields}\n}};'
            return MappedType(code=f'struct {typ.name}', declarations=[decl])
        elif self.target in (TargetLanguage.ASM_X86, TargetLanguage.ASM_ARM):
            lines = [f'{typ.name}:']
            for f in typ.fields:
                mapped = self.map_type(f.type)
                lines.append(f'    .{f.name} {mapped.code}')
            decl = '\n'.join(lines)
            return MappedType(code=typ.name, declarations=[decl])
        elif self.target == TargetLanguage.GO:
            fields = '\n    '.join(
                f'{f.name} {self.map_type(f.type).code}'
                for f in typ.fields
            )
            decl = f'type {typ.name} struct {{\n    {fields}\n}}'
            return MappedType(code=typ.name, declarations=[decl])
        elif self.target == TargetLanguage.JAVA:
            fields = '\n    '.join(
                f'{self.map_type(f.type).code} {f.name};'
                for f in typ.fields
            )
            decl = f'class {typ.name} {{\n    {fields}\n}}'
            return MappedType(code=typ.name, declarations=[decl])
        return MappedType(code=typ.name)
    
    def _map_union(self, typ: UnionType) -> MappedType:
        if self.target == TargetLanguage.PYTHON:
            variants = ', '.join(self.map_type(v.type).code for v in typ.variants)
            return MappedType(
                code=f'Union[{variants}]',
                imports=['from typing import Union']
            )
        elif self.target == TargetLanguage.RUST:
            # Rust doesn't have C-style unions in safe code
            return MappedType(
                code=typ.name,
                notes=['Rust: Consider using enum instead of union']
            )
        elif self.target == TargetLanguage.HASKELL:
            # Haskell uses algebraic data types
            return MappedType(code=typ.name)
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            fields = '\n    '.join(
                f'{self.map_type(v.type).code} {v.name};'
                for v in typ.variants
            )
            decl = f'union {typ.name} {{\n    {fields}\n}};'
            return MappedType(code=f'union {typ.name}', declarations=[decl])
        return MappedType(code=typ.name)
    
    def _map_enum(self, typ: EnumType) -> MappedType:
        if self.target == TargetLanguage.PYTHON:
            variants = ', '.join(
                f'{v.name} = {v.value if v.value is not None else i}'
                for i, v in enumerate(typ.variants)
            )
            decl = f'class {typ.name}(Enum):\n    {variants}'
            return MappedType(
                code=typ.name,
                imports=['from enum import Enum'],
                declarations=[decl]
            )
        elif self.target == TargetLanguage.RUST:
            variants = ',\n    '.join(
                f'{v.name} = {v.value}' if v.value is not None else v.name
                for v in typ.variants
            )
            decl = f'enum {typ.name} {{\n    {variants}\n}}'
            return MappedType(code=typ.name, declarations=[decl])
        elif self.target == TargetLanguage.HASKELL:
            variants = ' | '.join(v.name for v in typ.variants)
            decl = f'data {typ.name} = {variants} deriving (Eq, Show)'
            return MappedType(code=typ.name, declarations=[decl])
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            variants = ',\n    '.join(
                f'{v.name} = {v.value}' if v.value is not None else v.name
                for v in typ.variants
            )
            decl = f'enum {typ.name} {{\n    {variants}\n}};'
            return MappedType(code=f'enum {typ.name}', declarations=[decl])
        return MappedType(code=typ.name)
    
    def _map_tagged_union(self, typ: TaggedUnionType) -> MappedType:
        if self.target == TargetLanguage.PYTHON:
            # Use Union with dataclasses
            variants = ', '.join(
                f'{typ.name}_{v.name}'
                for v in typ.variants
            )
            imports = ['from typing import Union', 'from dataclasses import dataclass']
            decls = []
            for v in typ.variants:
                if v.fields:
                    fields = ', '.join(
                        f'{i}: {self.map_type(f).code}'
                        for i, f in enumerate(v.fields)
                    )
                    decls.append(f'@dataclass\nclass {typ.name}_{v.name}:\n    {fields}')
                else:
                    decls.append(f'@dataclass\nclass {typ.name}_{v.name}: pass')
            decls.append(f'{typ.name} = Union[{variants}]')
            return MappedType(code=typ.name, imports=imports, declarations=decls)
        
        elif self.target == TargetLanguage.RUST:
            variants = []
            for v in typ.variants:
                if v.fields:
                    fields_str = ', '.join(self.map_type(f).code for f in v.fields)
                    variants.append(f'{v.name}({fields_str})')
                elif v.named_fields:
                    fields_str = ', '.join(
                        f'{k}: {self.map_type(t).code}'
                        for k, t in v.named_fields.items()
                    )
                    variants.append(f'{v.name} {{ {fields_str} }}')
                else:
                    variants.append(v.name)
            variants_str = ',\n    '.join(variants)
            decl = f'enum {typ.name} {{\n    {variants_str}\n}}'
            return MappedType(code=typ.name, declarations=[decl])
        
        elif self.target == TargetLanguage.HASKELL:
            variants = []
            for v in typ.variants:
                if v.fields:
                    fields_str = ' '.join(self.map_type(f).code for f in v.fields)
                    variants.append(f'{v.name} {fields_str}')
                else:
                    variants.append(v.name)
            variants_str = ' | '.join(variants)
            decl = f'data {typ.name} = {variants_str}'
            return MappedType(code=typ.name, declarations=[decl])
        
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            # C requires tagged union pattern
            tag_enum = f'enum {typ.name}_Tag {{ ' + ', '.join(
                f'{typ.name}_{v.name}'
                for v in typ.variants
            ) + ' };'
            
            union_fields = []
            for v in typ.variants:
                if v.fields:
                    struct_name = f'{typ.name}_{v.name}_Data'
                    fields = '; '.join(
                        f'{self.map_type(f).code} f{i}'
                        for i, f in enumerate(v.fields)
                    )
                    union_fields.append(f'struct {{ {fields}; }} {v.name.lower()};')
            
            decl = f'''{tag_enum}
struct {typ.name} {{
    enum {typ.name}_Tag tag;
    union {{
        {chr(10).join('        ' + f for f in union_fields)}
    }} data;
}};'''
            return MappedType(code=f'struct {typ.name}', declarations=[decl])
        
        return MappedType(code=typ.name)
    
    def _map_function(self, typ: FunctionType) -> MappedType:
        params = [self.map_type(p) for p in typ.params]
        returns = self.map_type(typ.returns)
        
        if self.target == TargetLanguage.PYTHON:
            params_str = ', '.join(p.code for p in params)
            return MappedType(
                code=f'Callable[[{params_str}], {returns.code}]',
                imports=['from typing import Callable']
            )
        elif self.target == TargetLanguage.RUST:
            params_str = ', '.join(p.code for p in params)
            return MappedType(code=f'fn({params_str}) -> {returns.code}')
        elif self.target == TargetLanguage.HASKELL:
            params_str = ' -> '.join(p.code for p in params)
            if params:
                return MappedType(code=f'{params_str} -> {returns.code}')
            return MappedType(code=f'() -> {returns.code}')
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            params_str = ', '.join(p.code for p in params)
            if typ.variadic:
                params_str += ', ...'
            return MappedType(code=f'{returns.code} (*)({params_str})')
        return MappedType(code='fn')
    
    def _map_closure(self, typ: ClosureType) -> MappedType:
        params = [self.map_type(p) for p in typ.params]
        returns = self.map_type(typ.returns)
        
        if self.target == TargetLanguage.PYTHON:
            params_str = ', '.join(p.code for p in params)
            return MappedType(
                code=f'Callable[[{params_str}], {returns.code}]',
                imports=['from typing import Callable'],
                notes=['Python: closures are first-class']
            )
        elif self.target == TargetLanguage.RUST:
            params_str = ', '.join(p.code for p in params)
            return MappedType(
                code=f'impl Fn({params_str}) -> {returns.code}',
                notes=['Rust: Use Box<dyn Fn> for owned closures']
            )
        elif self.target == TargetLanguage.HASKELL:
            return MappedType(
                code=f'{" -> ".join(p.code for p in params)} -> {returns.code}',
                notes=['Haskell: all functions can capture environment']
            )
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            # C closures require manual capture struct
            return MappedType(
                code='void*',
                notes=['C: Closures require manual capture handling']
            )
        return MappedType(code='closure')
    
    def _map_generic(self, typ: GenericType) -> MappedType:
        args = [self.map_type(a) for a in typ.args]
        args_str = ', '.join(a.code for a in args)
        
        if self.target == TargetLanguage.PYTHON:
            return MappedType(code=f'{typ.base}[{args_str}]')
        elif self.target == TargetLanguage.RUST:
            return MappedType(code=f'{typ.base}<{args_str}>')
        elif self.target == TargetLanguage.HASKELL:
            return MappedType(code=f'{typ.base} {" ".join(a.code for a in args)}')
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            # C doesn't have generics, use void* or macros
            return MappedType(
                code=typ.base,
                notes=[f'C: Generics not supported, args: {args_str}']
            )
        elif self.target == TargetLanguage.GO:
            return MappedType(code=f'{typ.base}[{args_str}]')
        elif self.target == TargetLanguage.JAVA:
            return MappedType(code=f'{typ.base}<{args_str}>')
        return MappedType(code=f'{typ.base}<{args_str}>')
    
    def _map_type_var(self, typ: TypeVar) -> MappedType:
        if self.target == TargetLanguage.PYTHON:
            constraints = ', '.join(typ.constraints) if typ.constraints else ''
            return MappedType(
                code=typ.name,
                imports=['from typing import TypeVar'],
                declarations=[f'{typ.name} = TypeVar("{typ.name}", {constraints})']
            )
        elif self.target == TargetLanguage.RUST:
            constraints = ' + '.join(typ.constraints) if typ.constraints else ''
            if constraints:
                return MappedType(code=f'{typ.name}: {constraints}')
            return MappedType(code=typ.name)
        elif self.target == TargetLanguage.HASKELL:
            return MappedType(code=typ.name.lower())
        return MappedType(code=typ.name)
    
    def _map_opaque(self, typ: OpaqueType) -> MappedType:
        if self.target == TargetLanguage.PYTHON:
            return MappedType(code='Any', imports=['from typing import Any'])
        elif self.target == TargetLanguage.RUST:
            return MappedType(code=typ.name)
        elif self.target == TargetLanguage.HASKELL:
            return MappedType(code=typ.name)
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            return MappedType(code=f'struct {typ.name}')
        return MappedType(code=typ.name)
    
    def _map_recursive(self, typ: RecursiveType) -> MappedType:
        if self.target == TargetLanguage.PYTHON:
            return MappedType(code=f"'{typ.name}'")  # Forward reference
        elif self.target == TargetLanguage.RUST:
            return MappedType(
                code=f'Box<{typ.name}>',
                notes=['Rust: Recursive types need Box for heap allocation']
            )
        elif self.target == TargetLanguage.HASKELL:
            return MappedType(code=typ.name)
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            return MappedType(code=f'struct {typ.name}*')
        return MappedType(code=typ.name)
    
    def _map_optional(self, typ: OptionalType) -> MappedType:
        inner = self.map_type(typ.inner)
        
        if self.target == TargetLanguage.PYTHON:
            return MappedType(
                code=f'Optional[{inner.code}]',
                imports=['from typing import Optional']
            )
        elif self.target == TargetLanguage.RUST:
            return MappedType(code=f'Option<{inner.code}>')
        elif self.target == TargetLanguage.HASKELL:
            return MappedType(code=f'Maybe {inner.code}')
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            return MappedType(
                code=f'{inner.code}*',
                notes=['C: Optional represented as nullable pointer']
            )
        elif self.target == TargetLanguage.GO:
            return MappedType(code=f'*{inner.code}')
        elif self.target == TargetLanguage.JAVA:
            return MappedType(
                code=f'Optional<{inner.code}>',
                imports=['java.util.Optional']
            )
        return MappedType(code=f'Optional<{inner.code}>')
    
    def _map_result(self, typ: ResultType) -> MappedType:
        ok = self.map_type(typ.ok_type)
        err = self.map_type(typ.err_type)
        
        if self.target == TargetLanguage.PYTHON:
            return MappedType(
                code=f'Union[{ok.code}, {err.code}]',
                imports=['from typing import Union'],
                notes=['Python: Consider using Result type from returns library']
            )
        elif self.target == TargetLanguage.RUST:
            return MappedType(code=f'Result<{ok.code}, {err.code}>')
        elif self.target == TargetLanguage.HASKELL:
            return MappedType(code=f'Either {err.code} {ok.code}')
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            return MappedType(
                code='int',
                notes=[f'C: Result as error code, success: {ok.code}, error: {err.code}']
            )
        elif self.target == TargetLanguage.GO:
            return MappedType(code=f'({ok.code}, error)')
        return MappedType(code=f'Result<{ok.code}, {err.code}>')
    
    def _map_tuple(self, typ: TupleType) -> MappedType:
        elements = [self.map_type(e) for e in typ.elements]
        
        if self.target == TargetLanguage.PYTHON:
            elems_str = ', '.join(e.code for e in elements)
            return MappedType(
                code=f'Tuple[{elems_str}]',
                imports=['from typing import Tuple']
            )
        elif self.target == TargetLanguage.RUST:
            elems_str = ', '.join(e.code for e in elements)
            return MappedType(code=f'({elems_str})')
        elif self.target == TargetLanguage.HASKELL:
            elems_str = ', '.join(e.code for e in elements)
            return MappedType(code=f'({elems_str})')
        elif self.target in (TargetLanguage.C89, TargetLanguage.C99):
            # C doesn't have tuples, use anonymous struct
            fields = '; '.join(
                f'{e.code} f{i}'
                for i, e in enumerate(elements)
            )
            return MappedType(code=f'struct {{ {fields}; }}')
        elif self.target == TargetLanguage.GO:
            # Go doesn't have tuples, use struct
            return MappedType(
                code='struct{}',
                notes=['Go: Use multiple return values instead of tuple']
            )
        return MappedType(code=f'({", ".join(e.code for e in elements)})')


def create_type_mapper(target: str) -> TypeMapper:
    """Factory function to create a TypeMapper for a target language."""
    target_map = {
        'python': TargetLanguage.PYTHON,
        'rust': TargetLanguage.RUST,
        'haskell': TargetLanguage.HASKELL,
        'c89': TargetLanguage.C89,
        'c99': TargetLanguage.C99,
        'c': TargetLanguage.C99,
        'asm': TargetLanguage.ASM_X86,
        'x86': TargetLanguage.ASM_X86,
        'arm': TargetLanguage.ASM_ARM,
        'go': TargetLanguage.GO,
        'java': TargetLanguage.JAVA,
    }
    target_lang = target_map.get(target.lower(), TargetLanguage.C99)
    return TypeMapper(target_lang)


__all__ = [
    'TargetLanguage', 'MappedType', 'TypeMapper', 'create_type_mapper'
]
