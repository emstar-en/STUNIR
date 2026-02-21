#!/usr/bin/env python3
"""STUNIR Systems IR - Formal verification constructs (SPARK).

This module defines IR nodes for formal verification features,
primarily supporting Ada SPARK annotations for proving
absence of runtime errors and functional correctness.

Usage:
    from ir.systems.verification import Contract, GlobalSpec, LoopInvariant
    
    # Create a precondition contract
    pre = Contract(
        condition=BinaryOp('>', VarExpr(name='X'), Literal(value=0)),
        message="X must be positive"
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union
from ir.systems.systems_ir import SystemsNode, Expr, Statement, Declaration, TypeRef


# =============================================================================
# Contracts
# =============================================================================

@dataclass
class Contract(SystemsNode):
    """Contract expression (Pre/Post/Invariant).
    
    Ada SPARK: Pre => Condition, Post => Condition
    D: in { assert(condition); }, out { assert(condition); }
    
    Examples:
        Ada: Pre => X > 0
             Post => Result'Result = X + Y
        D: in (x > 0)
           out (result; result == x + y)
    """
    condition: Expr = None
    message: Optional[str] = None  # Optional message for failed contract
    kind: str = 'contract'


@dataclass
class ContractCase(SystemsNode):
    """Contract case (SPARK Contract_Cases).
    
    Ada SPARK: Contract_Cases => (Guard1 => Post1, Guard2 => Post2, ...)
    
    Each case specifies a precondition (guard) and corresponding
    postcondition (consequence) that must hold when the guard is true.
    
    Examples:
        Ada: Contract_Cases =>
               (X >= 0 => Abs_Value'Result = X,
                X < 0  => Abs_Value'Result = -X)
    """
    guard: Expr = None
    consequence: Expr = None
    kind: str = 'contract_case'


@dataclass
class TypeInvariant(SystemsNode):
    """Type invariant declaration.
    
    Ada: Type_Invariant => Condition
    D: invariant { assert(condition); }
    
    Specifies a condition that must hold for all objects of a type.
    """
    condition: Expr = None
    type_name: Optional[str] = None
    kind: str = 'type_invariant'


@dataclass
class SubtypePredicate(SystemsNode):
    """Subtype predicate (SPARK).
    
    Ada: Static_Predicate => Condition
         Dynamic_Predicate => Condition
    
    Constrains the values a subtype can have.
    """
    condition: Expr = None
    is_static: bool = False  # Static vs dynamic predicate
    kind: str = 'subtype_predicate'


# =============================================================================
# SPARK Flow Analysis Annotations
# =============================================================================

@dataclass
class GlobalSpec(SystemsNode):
    """SPARK Global specification.
    
    Specifies which global variables a subprogram reads from,
    writes to, or both.
    
    Ada SPARK: Global => (Input => (X, Y), Output => Z, In_Out => W)
               Global => null  (no global effects)
    
    Categories:
        - Input: Read-only globals
        - Output: Write-only globals  
        - In_Out: Read-write globals
        - Proof_In: Used only in proofs
    """
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    in_outs: List[str] = field(default_factory=list)
    proof_ins: List[str] = field(default_factory=list)
    is_null: bool = False  # Global => null
    kind: str = 'global_spec'


@dataclass
class DependsSpec(SystemsNode):
    """SPARK Depends specification.
    
    Specifies the data dependencies: which outputs depend on which inputs.
    
    Ada SPARK: Depends => (X => Y, Z => (A, B), W => W)
    
    The left side is an output, the right side lists its inputs.
    Self-dependency (X => X) indicates the variable is both read and written.
    """
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    kind: str = 'depends_spec'


@dataclass
class InitializesSpec(SystemsNode):
    """SPARK Initializes specification (package level).
    
    Specifies which package state is initialized during elaboration.
    
    Ada SPARK: Initializes => (State1, State2 => External_Input)
    """
    initialized: List[str] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    kind: str = 'initializes_spec'


@dataclass
class AbstractState(SystemsNode):
    """SPARK Abstract_State declaration.
    
    Declares abstract state for a package to hide implementation details.
    
    Ada SPARK: Abstract_State => (State1, State2 with External)
    """
    name: str = ''
    is_external: bool = False
    is_synchronous: bool = False
    constituents: List[str] = field(default_factory=list)  # Refined_State
    kind: str = 'abstract_state'


@dataclass
class RefinedState(SystemsNode):
    """SPARK Refined_State specification.
    
    Maps abstract state to concrete variables in the package body.
    
    Ada SPARK: Refined_State => (State => (Var1, Var2))
    """
    refinements: Dict[str, List[str]] = field(default_factory=dict)
    kind: str = 'refined_state'


# =============================================================================
# Loop Annotations
# =============================================================================

@dataclass
class LoopInvariant(SystemsNode):
    """Loop invariant annotation.
    
    A condition that must be true at the start of each loop iteration.
    
    Ada SPARK: pragma Loop_Invariant(Condition);
    
    Examples:
        Ada: pragma Loop_Invariant(Sum = I * (I - 1) / 2);
             pragma Loop_Invariant(for all J in 1 .. I - 1 => A(J) = 0);
    """
    condition: Expr = None
    kind: str = 'loop_invariant'


@dataclass
class LoopVariant(SystemsNode):
    """Loop variant annotation (termination proof).
    
    Specifies expressions that decrease (or increase) with each iteration,
    proving the loop will terminate.
    
    Ada SPARK: pragma Loop_Variant(Decreases => N - I, Increases => J);
    
    Each expression must progress in the specified direction.
    """
    expressions: List['VariantExpr'] = field(default_factory=list)
    kind: str = 'loop_variant'


@dataclass
class VariantExpr(SystemsNode):
    """Single expression in a loop variant."""
    expr: Expr = None
    direction: str = 'decreases'  # 'decreases' or 'increases'
    kind: str = 'variant_expr'


# =============================================================================
# Ghost Code
# =============================================================================

@dataclass
class GhostCode(SystemsNode):
    """Ghost code for verification only.
    
    Ghost code exists only for specification and proof purposes.
    It is not included in the executable.
    
    Ada SPARK: X : Integer with Ghost;
               function Is_Sorted return Boolean with Ghost;
    
    Examples:
        Ada: Ghost_Sum : Integer := 0 with Ghost;
             procedure Update_Ghost with Ghost;
    """
    content: Union[Statement, Expr, Declaration] = None
    kind: str = 'ghost_code'


@dataclass
class GhostVariable(SystemsNode):
    """Ghost variable declaration.
    
    A variable used only for specification and proofs.
    """
    name: str = ''
    type_ref: TypeRef = None
    initializer: Optional[Expr] = None
    kind: str = 'ghost_variable'


@dataclass
class GhostFunction(SystemsNode):
    """Ghost function declaration.
    
    A function used only in contracts and proofs.
    """
    name: str = ''
    parameters: List['Parameter'] = field(default_factory=list)
    return_type: TypeRef = None
    expression: Optional[Expr] = None  # Expression function body
    kind: str = 'ghost_function'


# =============================================================================
# Assertions and Assumptions
# =============================================================================

@dataclass
class AssertPragma(SystemsNode):
    """Assert pragma for proofs.
    
    Ada: pragma Assert(Condition [, Message]);
    D: assert(condition, message);
    
    A condition that must be proven true at this point.
    """
    condition: Expr = None
    message: Optional[str] = None
    kind: str = 'assert_pragma'


@dataclass
class AssumePragma(SystemsNode):
    """Assume pragma for proofs.
    
    Ada SPARK: pragma Assume(Condition);
    
    A condition assumed to be true without proof.
    Used to bridge gaps in proof or for external guarantees.
    """
    condition: Expr = None
    kind: str = 'assume_pragma'


@dataclass
class CheckPragma(SystemsNode):
    """Check pragma (Ada).
    
    Ada: pragma Check(Kind, Condition [, Message]);
    
    General assertion mechanism with named check kinds.
    """
    check_kind: str = ''  # 'Assertion', 'Pre', 'Post', etc.
    condition: Expr = None
    message: Optional[str] = None
    kind: str = 'check_pragma'


# =============================================================================
# SPARK Mode Pragma
# =============================================================================

@dataclass
class SparkMode(SystemsNode):
    """SPARK_Mode pragma.
    
    Ada SPARK: pragma SPARK_Mode [(On|Off)];
    
    Enables or disables SPARK analysis for a region of code.
    """
    enabled: bool = True
    kind: str = 'spark_mode'


# =============================================================================
# Proof-Only Constructs
# =============================================================================

@dataclass
class QuantifiedExpr(SystemsNode):
    """Quantified expression for contracts.
    
    Ada: (for all I in Range => Condition)
         (for some I in Range => Condition)
    
    Examples:
        Ada: (for all I in A'Range => A(I) >= 0)
             (for some I in 1 .. N => A(I) = Target)
    """
    quantifier: str = 'all'  # 'all' or 'some'
    variable: str = ''
    range_expr: Expr = None
    condition: Expr = None
    kind: str = 'quantified_expr'


@dataclass
class OldExpr(SystemsNode):
    """Old attribute expression (postcondition reference to pre-state).
    
    Ada: X'Old
    D: __old(x)
    
    References the value of an expression at subprogram entry.
    """
    expr: Expr = None
    kind: str = 'old_expr'


@dataclass
class ResultExpr(SystemsNode):
    """Result attribute expression.
    
    Ada: Function_Name'Result
    D: __result
    
    References the return value in a postcondition.
    """
    function_name: Optional[str] = None  # Ada requires function name
    kind: str = 'result_expr'


@dataclass
class LoopEntryExpr(SystemsNode):
    """Loop_Entry attribute expression (SPARK).
    
    Ada: X'Loop_Entry [[(Loop_Name)]]
    
    References the value at the start of a loop iteration.
    """
    expr: Expr = None
    loop_name: Optional[str] = None
    kind: str = 'loop_entry_expr'


# =============================================================================
# D Contract Constructs
# =============================================================================

@dataclass
class DContractIn(SystemsNode):
    """D in contract (precondition block).
    
    D: int divide(int a, int b)
       in (b != 0)
       { ... }
    """
    assertions: List[Expr] = field(default_factory=list)
    kind: str = 'd_contract_in'


@dataclass
class DContractOut(SystemsNode):
    """D out contract (postcondition block).
    
    D: int abs(int x)
       out (result; result >= 0)
       { ... }
    
    The result parameter name is specified for referencing the return value.
    """
    result_name: Optional[str] = None  # Name for result reference
    assertions: List[Expr] = field(default_factory=list)
    kind: str = 'd_contract_out'


@dataclass
class DInvariant(SystemsNode):
    """D struct/class invariant.
    
    D: struct Counter {
          int value;
          invariant { assert(value >= 0); }
       }
    """
    assertions: List[Expr] = field(default_factory=list)
    kind: str = 'd_invariant'


# Forward reference
Parameter = 'Parameter'
