#!/usr/bin/env python3
"""STUNIR Systems IR - Memory management constructs.

This module defines IR nodes for memory management operations
including pointers/access types, allocation, deallocation,
and address operations.

Usage:
    from ir.systems.memory import AccessType, Allocator, Deallocate
    
    # Create an access type (pointer)
    int_ptr = AccessType(target_type=TypeRef(name='Integer'), not_null=True)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
from ir.systems.systems_ir import SystemsNode, TypeRef, Expr


# =============================================================================
# Access Types (Pointers)
# =============================================================================

@dataclass
class AccessType(SystemsNode):
    """Access (pointer) type.
    
    Ada: access types with optional not null
    D: pointers with @safe annotations
    
    Examples:
        Ada: type Int_Ptr is access Integer;
             type Not_Null_Ptr is not null access Integer;
             type Const_Ptr is access constant Integer;
        D: int*, ref int
    """
    target_type: TypeRef = None
    not_null: bool = False  # Ada: not null access
    constant: bool = False  # Access to constant
    all_access: bool = False  # Ada: access all (including aliased)
    pool: Optional[str] = None  # Custom storage pool
    kind: str = 'access_type'


@dataclass
class AccessTypeDecl(SystemsNode):
    """Access type declaration.
    
    Ada: type Name is [not null] access [all|constant] Type;
    """
    name: str = ''
    target_type: TypeRef = None
    not_null: bool = False
    constant: bool = False
    all_access: bool = False
    pool: Optional[str] = None
    kind: str = 'access_type_decl'


# =============================================================================
# Memory Operations
# =============================================================================

@dataclass
class Allocator(SystemsNode):
    """Memory allocation.
    
    Ada: new T, new T'(initializer), new T'(Field => Value)
    D: new T() or malloc-based allocation
    
    Examples:
        Ada: Ptr := new Integer;
             Ptr := new Integer'(42);
             Rec_Ptr := new Record_Type'(X => 1, Y => 2);
        D: auto ptr = new int(42);
    """
    type_ref: TypeRef = None
    initializer: Optional[Expr] = None  # Qualified expression
    pool: Optional[str] = None  # Ada storage pool
    subpool: Optional[Expr] = None  # Ada subpool handle
    kind: str = 'allocator'


@dataclass
class Deallocate(SystemsNode):
    """Memory deallocation.
    
    Ada: Requires instantiation of Unchecked_Deallocation
    D: destroy() or GC
    
    Examples:
        Ada:
            procedure Free is new Unchecked_Deallocation(Integer, Int_Ptr);
            Free(Ptr);
        D:
            destroy(obj);
    """
    target: Expr = None
    deallocation_proc: Optional[str] = None  # Ada: name of Free procedure
    kind: str = 'deallocate'


@dataclass
class AddressOf(SystemsNode):
    """Address-of operation.
    
    Ada: X'Address
    D: &x
    
    Returns the memory address of an object.
    """
    target: Expr = None
    kind: str = 'address_of'


@dataclass
class Dereference(SystemsNode):
    """Pointer dereference operation.
    
    Ada: Ptr.all
    D: *ptr
    """
    target: Expr = None
    kind: str = 'dereference'


# =============================================================================
# Storage Pools (Ada)
# =============================================================================

@dataclass
class StoragePoolDecl(SystemsNode):
    """Storage pool type declaration (Ada).
    
    Defines a custom memory pool for allocation.
    
    Ada: type Pool_Type is new Root_Storage_Pool with ...;
    """
    name: str = ''
    parent_pool: Optional[str] = None  # Base pool type
    size: Optional[Expr] = None  # Pool size
    kind: str = 'storage_pool_decl'


@dataclass
class UsePool(SystemsNode):
    """Specify storage pool for access type.
    
    Ada: for Access_Type'Storage_Pool use Pool_Object;
    """
    access_type: str = ''
    pool: str = ''  # Pool object name
    kind: str = 'use_pool'


# =============================================================================
# Unchecked Operations (Ada)
# =============================================================================

@dataclass
class UncheckedConversion(SystemsNode):
    """Unchecked type conversion.
    
    Ada: function To_Int is new Unchecked_Conversion(Float, Integer);
    D: cast(T)value
    """
    source_type: TypeRef = None
    target_type: TypeRef = None
    name: Optional[str] = None  # Function name
    kind: str = 'unchecked_conversion'


@dataclass
class UncheckedDeallocation(SystemsNode):
    """Unchecked deallocation instantiation.
    
    Ada: procedure Free is new Unchecked_Deallocation(T, T_Ptr);
    """
    object_type: TypeRef = None
    access_type: TypeRef = None
    name: str = ''  # Procedure name (e.g., 'Free')
    kind: str = 'unchecked_deallocation'


# =============================================================================
# Memory Safety (D)
# =============================================================================

@dataclass
class ScopedRef(SystemsNode):
    """Scoped reference (D @safe).
    
    D: scope ref T
    
    A reference that cannot escape its scope.
    """
    target_type: TypeRef = None
    is_return: bool = False  # return ref
    kind: str = 'scoped_ref'


@dataclass
class SliceExpr(SystemsNode):
    """Array/string slice expression.
    
    Ada: Arr(First .. Last)
    D: arr[first .. last]
    """
    target: Expr = None
    start: Expr = None
    end: Expr = None
    kind: str = 'slice_expr'
