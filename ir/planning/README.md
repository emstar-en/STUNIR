# Planning IR

Planning Intermediate Representation for STUNIR - supporting automated planning domains and problems in PDDL format.

## Overview

The Planning IR provides a complete representation for:
- **Domains**: Type hierarchies, predicates, functions, and action schemas
- **Problems**: Objects, initial state, goals, and metrics
- **Formulas**: Logical formulas for preconditions, effects, and goals
- **Actions**: Action schemas with parameters, preconditions, and effects

## Module Structure

```
ir/planning/
├── __init__.py          # Package exports
├── planning_ir.py       # Core enums and exceptions
├── predicate.py         # Types, predicates, atoms
├── action.py            # Formulas, effects, actions
├── domain.py            # Domain definitions
├── problem.py           # Problem definitions
└── README.md            # This file
```

## Usage

### Creating a Domain (Blocks World)

```python
from ir.planning import (
    Domain, Action, Predicate, Parameter, Formula, Effect,
    PDDLRequirement, EffectType
)

# Create domain
domain = Domain(
    name="blocks-world",
    requirements=[PDDLRequirement.STRIPS, PDDLRequirement.TYPING]
)

# Add types
domain.add_type("block")

# Add predicates
domain.add_predicate(Predicate("on", [
    Parameter("?x", "block"),
    Parameter("?y", "block")
]))
domain.add_predicate(Predicate("ontable", [Parameter("?x", "block")]))
domain.add_predicate(Predicate("clear", [Parameter("?x", "block")]))
domain.add_predicate(Predicate("holding", [Parameter("?x", "block")]))
domain.add_predicate(Predicate("arm-empty", []))

# Add action
pick_up = Action(
    name="pick-up",
    parameters=[Parameter("?x", "block")],
    precondition=Formula.make_and(
        Formula.make_atom("clear", "?x"),
        Formula.make_atom("ontable", "?x"),
        Formula.make_atom("arm-empty")
    ),
    effect=Effect.make_compound(
        Effect.make_positive("holding", "?x"),
        Effect.make_negative("ontable", "?x"),
        Effect.make_negative("clear", "?x"),
        Effect.make_negative("arm-empty")
    )
)
domain.add_action(pick_up)
```

### Creating a Problem

```python
from ir.planning import Problem, InitialState, Formula

problem = Problem(
    name="blocks-4-0",
    domain_name="blocks-world"
)

# Add objects
problem.add_objects(["b1", "b2", "b3", "b4"], "block")

# Set initial state
problem.init.add_fact("clear", "b1")
problem.init.add_fact("on", "b1", "b2")
problem.init.add_fact("on", "b2", "b3")
problem.init.add_fact("ontable", "b3")
problem.init.add_fact("ontable", "b4")
problem.init.add_fact("clear", "b4")
problem.init.add_fact("arm-empty")

# Set goal
problem.set_goal(Formula.make_and(
    Formula.make_atom("on", "b4", "b3"),
    Formula.make_atom("on", "b3", "b2"),
    Formula.make_atom("on", "b2", "b1")
))
```

## PDDL Requirements

Supported requirements:
- `:strips` - Basic STRIPS planning
- `:typing` - Typed objects and parameters
- `:negative-preconditions` - Negated atoms in preconditions
- `:disjunctive-preconditions` - OR in preconditions
- `:equality` - `=` predicate
- `:existential-preconditions` - `exists` quantifier
- `:universal-preconditions` - `forall` quantifier
- `:conditional-effects` - `when` effects
- `:numeric-fluents` - Numeric functions
- `:action-costs` - Action costs
- `:adl` - Action Description Language
- And more...

## Formula Construction

The `Formula` class provides factory methods:

```python
# Atomic formula
atom = Formula.make_atom("at", "?obj", "?loc")

# Conjunction
conj = Formula.make_and(atom1, atom2, atom3)

# Disjunction
disj = Formula.make_or(atom1, atom2)

# Negation
neg = Formula.make_not(atom)

# Implication
impl = Formula.make_imply(antecedent, consequent)

# Quantified formulas
exists = Formula.make_exists([Parameter("?x", "type")], body)
forall = Formula.make_forall([Parameter("?x", "type")], body)
```

## Effect Construction

The `Effect` class provides factory methods:

```python
# Positive effect (add fact)
pos = Effect.make_positive("holding", "?x")

# Negative effect (delete fact)
neg = Effect.make_negative("clear", "?x")

# Compound effect
comp = Effect.make_compound(pos, neg)

# Conditional effect
cond = Effect.make_conditional(condition, effect)

# Numeric effects
inc = Effect.make_increase("total-cost", [], 1)
dec = Effect.make_decrease("fuel", ["?v"], 10)
```

## Validation

Both Domain and Problem support validation:

```python
# Validate domain
errors = domain.validate()
if errors:
    print("Domain validation errors:", errors)

# Validate problem against domain
errors = problem.validate(domain)
if errors:
    print("Problem validation errors:", errors)
```

## Integration with PDDL Emitter

```python
from targets.planning import PDDLEmitter

emitter = PDDLEmitter()

# Emit domain only
result = emitter.emit_domain(domain)
print(result.domain_code)

# Emit problem only
result = emitter.emit_problem(problem)
print(result.problem_code)

# Emit both
result = emitter.emit(domain, problem)
result.write_domain("domain.pddl")
result.write_problem("problem.pddl")
```
