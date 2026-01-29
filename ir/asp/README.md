# ASP IR (Answer Set Programming Intermediate Representation)

Part of Phase 7D: Answer Set Programming

## Overview

The ASP IR provides a structured representation for Answer Set Programs,
supporting various rule types, aggregates, and optimization statements.
It enables STUNIR to represent, validate, and transform ASP programs before
emitting them to different ASP solver formats.

## Features

### Rule Types

- **Normal Rules**: `head :- body.`
- **Choice Rules**: `{head} :- body.` or `L {head} U :- body.`
- **Constraint Rules**: `:- body.`
- **Disjunctive Rules**: `head1 | head2 :- body.`
- **Weak Constraints**: `:~ body. [weight@priority]`

### Aggregates

- `#count { X : condition(X) }`
- `#sum { W,X : weight(X,W) }`
- `#min { V,X : value(X,V) }`
- `#max { V,X : value(X,V) }`

### Negation

- **Default Negation (NAF)**: `not p(X)` - closed world assumption
- **Classical Negation**: `-p(X)` - strong/explicit negation

### Optimization

- `#minimize { W@P,X : cost(X,W) }`
- `#maximize { V@P,X : value(X,V) }`

## Module Structure

```
ir/asp/
├── __init__.py      # Package exports
├── asp_ir.py        # Core enums and types
├── atom.py          # Term, Atom, Literal classes
├── aggregate.py     # Aggregate definitions
├── rule.py          # Rule definitions
├── program.py       # ASPProgram class
└── README.md        # This file
```

## Usage

### Creating a Simple Program

```python
from ir.asp import (
    ASPProgram, program, atom, var, pos, neg,
    ChoiceElement, agg_element
)

# Create a graph coloring program
p = program("graph_coloring")

# Add domain facts
p.add_fact(atom("node", "1"))
p.add_fact(atom("node", "2"))
p.add_fact(atom("node", "3"))
p.add_fact(atom("edge", "1", "2"))
p.add_fact(atom("edge", "2", "3"))
p.add_fact(atom("edge", "1", "3"))
p.add_fact(atom("col", "red"))
p.add_fact(atom("col", "green"))
p.add_fact(atom("col", "blue"))

# Choice rule: each node gets exactly one color
p.add_choice_rule(
    elements=[ChoiceElement(
        atom("color", var("X"), var("C")),
        [pos(atom("col", var("C")))]
    )],
    body=[pos(atom("node", var("X")))],
    lower=1, upper=1
)

# Constraint: adjacent nodes cannot have the same color
p.add_constraint([
    pos(atom("edge", var("X"), var("Y"))),
    pos(atom("color", var("X"), var("C"))),
    pos(atom("color", var("Y"), var("C")))
])

# Show only color assignments
p.add_show("color", 2)

print(p)
```

### Creating Rules with Aggregates

```python
from ir.asp import (
    ASPProgram, atom, var, pos,
    count, sum_agg, agg_element, ComparisonOp, Term
)

p = ASPProgram("scheduling")

# Constraint: at least 2 workers assigned to each task
# :- task(T), #count { W : assigned(W,T) } < 2.
agg = count()
agg.add_element(
    [var("W")],
    [pos(atom("assigned", var("W"), var("T")))]
)
agg.set_right_guard(ComparisonOp.LT, Term("2"))

# Add as body element with aggregate
from ir.asp import BodyElement, Rule, RuleType
rule = Rule(
    rule_type=RuleType.CONSTRAINT,
    body=[
        BodyElement(literal=pos(atom("task", var("T")))),
        BodyElement(aggregate=agg)
    ]
)
p.add_rule(rule)
```

### Optimization Problems

```python
from ir.asp import ASPProgram, atom, var, pos, agg_element

p = ASPProgram("knapsack")

# Items with weights and values
p.add_fact(atom("item", "a"))
p.add_fact(atom("weight", "a", "3"))
p.add_fact(atom("value", "a", "5"))

# Choice: select items
from ir.asp import ChoiceElement
p.add_choice_rule(
    elements=[ChoiceElement(atom("selected", var("I")))],
    body=[pos(atom("item", var("I")))]
)

# Maximize value
p.add_maximize([
    agg_element(
        [var("V"), var("I")],
        [pos(atom("selected", var("I"))),
         pos(atom("value", var("I"), var("V")))]
    )
])

p.add_show("selected", 1)
```

## API Reference

### Core Classes

- `Term`: Constants, variables, or function terms
- `Atom`: Predicate applications (e.g., `edge(X, Y)`)
- `Literal`: Atoms with optional negation
- `Aggregate`: Aggregate expressions (#count, #sum, etc.)
- `Rule`: ASP rules of various types
- `ASPProgram`: Complete ASP program

### Factory Functions

- `term(name, *args)`: Create a term
- `var(name)`: Create a variable
- `const(name)`: Create a constant
- `atom(predicate, *terms)`: Create an atom
- `pos(atom)`: Create a positive literal
- `neg(atom)`: Create a default-negated literal
- `classical_neg(atom)`: Create a classically-negated literal
- `count()`, `sum_agg()`, `min_agg()`, `max_agg()`: Create aggregates
- `normal_rule()`, `fact()`, `constraint()`: Create rules
- `choice_rule()`, `disjunctive_rule()`, `weak_constraint()`: More rule types

### Enums

- `RuleType`: NORMAL, CHOICE, CONSTRAINT, DISJUNCTIVE, WEAK
- `AggregateFunction`: COUNT, SUM, MIN, MAX, SUM_PLUS
- `ComparisonOp`: EQ, NE, LT, LE, GT, GE
- `NegationType`: NONE, DEFAULT, CLASSICAL

## See Also

- [targets/asp/README.md](../../targets/asp/README.md) - Emitter documentation
- [ASP-Core-2 Standard](https://www.mat.unical.it/aspcomp2013/ASPStandardization) - Official ASP standard
- [Clingo User Guide](https://potassco.org/clingo/) - Clingo documentation
