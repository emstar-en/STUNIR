"""Answer Set Programming IR package.

This package provides an intermediate representation for Answer Set
Programming (ASP), supporting various rule types, aggregates, and
optimization statements.

Part of Phase 7D: Answer Set Programming

Example usage:
    from ir.asp import (
        ASPProgram, Atom, Term, Literal,
        pos, neg, atom, term, var
    )
    
    # Create a graph coloring program
    program = ASPProgram("graph_coloring")
    
    # Add facts
    program.add_fact(atom("node", "a"))
    program.add_fact(atom("node", "b"))
    program.add_fact(atom("edge", "a", "b"))
    program.add_fact(atom("col", "red"))
    program.add_fact(atom("col", "green"))
    
    # Choice rule: each node gets exactly one color
    from ir.asp import ChoiceElement
    program.add_choice_rule(
        elements=[ChoiceElement(
            atom("color", var("X"), var("C")),
            [pos(atom("col", var("C")))]
        )],
        body=[pos(atom("node", var("X")))],
        lower=1, upper=1
    )
    
    # Constraint: adjacent nodes cannot have same color
    program.add_constraint([
        pos(atom("edge", var("X"), var("Y"))),
        pos(atom("color", var("X"), var("C"))),
        pos(atom("color", var("Y"), var("C")))
    ])
    
    # Show output
    program.add_show("color", 2)
    
    print(program)
"""

# Core types
from .asp_ir import (
    RuleType,
    AggregateFunction,
    ComparisonOp,
    NegationType,
    ASPDialect,
    DEFAULT_PRIORITY,
    DEFAULT_WEIGHT,
    validate_identifier,
    is_variable,
    is_constant,
)

# Atom and Literal types
from .atom import (
    Term,
    Atom,
    Literal,
    term,
    var,
    const,
    atom,
    pos,
    neg,
    classical_neg,
)

# Aggregate types
from .aggregate import (
    AggregateElement,
    Guard,
    Aggregate,
    Comparison,
    count,
    sum_agg,
    min_agg,
    max_agg,
    agg_element,
    compare,
)

# Rule types
from .rule import (
    BodyElement,
    HeadElement,
    ChoiceElement,
    Rule,
    normal_rule,
    fact,
    constraint,
    choice_rule,
    disjunctive_rule,
    weak_constraint,
)

# Program types
from .program import (
    ShowStatement,
    OptimizeStatement,
    ConstantDef,
    ASPProgram,
    program,
)

__all__ = [
    # Enums and constants
    "RuleType",
    "AggregateFunction",
    "ComparisonOp",
    "NegationType",
    "ASPDialect",
    "DEFAULT_PRIORITY",
    "DEFAULT_WEIGHT",
    # Functions
    "validate_identifier",
    "is_variable",
    "is_constant",
    # Atom/Literal
    "Term",
    "Atom",
    "Literal",
    "term",
    "var",
    "const",
    "atom",
    "pos",
    "neg",
    "classical_neg",
    # Aggregates
    "AggregateElement",
    "Guard",
    "Aggregate",
    "Comparison",
    "count",
    "sum_agg",
    "min_agg",
    "max_agg",
    "agg_element",
    "compare",
    # Rules
    "BodyElement",
    "HeadElement",
    "ChoiceElement",
    "Rule",
    "normal_rule",
    "fact",
    "constraint",
    "choice_rule",
    "disjunctive_rule",
    "weak_constraint",
    # Program
    "ShowStatement",
    "OptimizeStatement",
    "ConstantDef",
    "ASPProgram",
    "program",
]
