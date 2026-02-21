# Constraint Programming IR

This package provides intermediate representation (IR) classes for constraint satisfaction and optimization problems (CSP/COP).

## Overview

Constraint Programming is a paradigm for solving combinatorial problems by declaring constraints that must be satisfied. This IR supports:

- **Decision Variables**: Integer, float, boolean, and set variables
- **Domains**: Range domains, explicit sets, boolean domains
- **Constraints**: Arithmetic, logical, and global constraints
- **Objectives**: Minimize, maximize, or satisfy
- **Search Strategies**: Various variable and value selection heuristics

## Quick Start

```python
from ir.constraints import (
    ConstraintModel, Domain, VariableType, Objective,
    VariableRef, alldifferent
)

# Create an N-Queens model
n = 8
model = ConstraintModel("nqueens")

# Add queen position variables (one per row)
for i in range(1, n + 1):
    model.add_int_variable(f"q{i}", 1, n)

# All queens in different columns
vars_list = [VariableRef(f"q{i}") for i in range(1, n + 1)]
model.add_constraint(alldifferent(vars_list))

# Find a solution
model.set_objective(Objective.satisfy())

print(model)
```

## Module Structure

```
ir/constraints/
├── __init__.py          # Package exports
├── constraint_ir.py     # Core types and enums
├── variable.py          # Variable and array definitions
├── domain.py            # Domain definitions
├── constraint.py        # Constraint definitions
├── objective.py         # Objective and model classes
└── README.md            # This file
```

## Core Classes

### Variables

```python
from ir.constraints import Variable, ArrayVariable, IndexSet, Domain, VariableType

# Integer variable with range 1..10
var = Variable("x", VariableType.INT, Domain.int_range(1, 10))

# Boolean variable
bool_var = Variable("flag", VariableType.BOOL, Domain.bool_domain())

# Array variable
arr = ArrayVariable(
    "queens",
    VariableType.INT,
    IndexSet([(1, 8)]),
    Domain.int_range(1, 8)
)
```

### Domains

```python
from ir.constraints import Domain, DomainType

# Integer range domain
int_dom = Domain.int_range(1, 100)

# Float range domain
float_dom = Domain.float_range(0.0, 1.0)

# Boolean domain
bool_dom = Domain.bool_domain()

# Explicit set domain
set_dom = Domain.set_domain({1, 3, 5, 7, 9})

# Unbounded domain
unbounded_dom = Domain.unbounded()
```

### Expressions

```python
from ir.constraints import (
    VariableRef, Literal, BinaryOp, UnaryOp,
    ArrayAccess, FunctionCall
)

# Variable reference
x = VariableRef("x")

# Literal value
five = Literal(5)

# Binary operation: x + 5
add = BinaryOp("+", x, five)

# Unary operation: -x
neg = UnaryOp("-", x)

# Array access: arr[i]
acc = ArrayAccess("arr", [VariableRef("i")])

# Function call: abs(x)
abs_x = FunctionCall("abs", [x])
```

### Constraints

```python
from ir.constraints import (
    eq, ne, lt, le, gt, ge,           # Relational
    conjunction, disjunction, negation, implies,  # Logical
    alldifferent,                      # Global
    VariableRef, Literal
)

# Relational constraints
c1 = eq(VariableRef("x"), Literal(5))      # x = 5
c2 = ne(VariableRef("x"), VariableRef("y")) # x != y
c3 = lt(VariableRef("x"), Literal(10))     # x < 10
c4 = ge(VariableRef("y"), Literal(0))      # y >= 0

# Logical constraints
c5 = conjunction([c1, c2])   # c1 AND c2
c6 = disjunction([c3, c4])   # c3 OR c4
c7 = negation(c1)            # NOT c1
c8 = implies(c1, c2)         # c1 -> c2

# Global constraints
vars = [VariableRef("x"), VariableRef("y"), VariableRef("z")]
c9 = alldifferent(vars)      # all different values
```

### Objectives

```python
from ir.constraints import Objective, VariableRef

# Find any solution
sat = Objective.satisfy()

# Minimize a variable
min_obj = Objective.minimize(VariableRef("cost"))

# Maximize a variable
max_obj = Objective.maximize(VariableRef("profit"))
```

### Constraint Model

```python
from ir.constraints import ConstraintModel, Objective

# Create model
model = ConstraintModel("example")

# Add variables
model.add_int_variable("x", 1, 10)
model.add_int_variable("y", 1, 10)
model.add_bool_variable("b")
model.add_int_array("arr", 5, 0, 100)

# Add parameters (constants)
model.add_parameter("n", 10)

# Add constraints
from ir.constraints import eq, VariableRef, Literal
model.add_constraint(eq(VariableRef("x"), Literal(5)))

# Set objective
model.minimize(VariableRef("y"))

# Validate model
errors = model.validate()
if errors:
    print("Validation errors:", errors)
```

## Constraint Types

### Arithmetic Constraints

| Function | Description | Example |
|----------|-------------|---------|
| `eq(a, b)` | Equality | `x = 5` |
| `ne(a, b)` | Not equal | `x != y` |
| `lt(a, b)` | Less than | `x < 10` |
| `le(a, b)` | Less or equal | `x <= 10` |
| `gt(a, b)` | Greater than | `x > 0` |
| `ge(a, b)` | Greater or equal | `x >= 0` |

### Logical Constraints

| Function | Description | Example |
|----------|-------------|---------|
| `conjunction(cs)` | AND | `c1 /\ c2` |
| `disjunction(cs)` | OR | `c1 \/ c2` |
| `negation(c)` | NOT | `not(c1)` |
| `implies(c1, c2)` | Implication | `c1 -> c2` |

### Global Constraints

| Type | Description |
|------|-------------|
| `ALLDIFFERENT` | All variables have different values |
| `CUMULATIVE` | Resource scheduling constraint |
| `ELEMENT` | Array element access |
| `TABLE` | Allowed combinations table |
| `CIRCUIT` | Hamiltonian circuit |
| `COUNT` | Counting constraint |
| `BIN_PACKING` | Bin packing constraint |
| `GLOBAL_CARDINALITY` | Value occurrence counting |

## Search Strategies

```python
from ir.constraints import SearchStrategy, ValueChoice

# Variable selection strategies
SearchStrategy.INPUT_ORDER       # Use input order
SearchStrategy.FIRST_FAIL        # Smallest domain first
SearchStrategy.ANTI_FIRST_FAIL   # Largest domain first
SearchStrategy.SMALLEST          # Smallest value first
SearchStrategy.LARGEST           # Largest value first
SearchStrategy.MOST_CONSTRAINED  # Most constrained variable

# Value choice heuristics
ValueChoice.INDOMAIN_MIN     # Try smallest value first
ValueChoice.INDOMAIN_MAX     # Try largest value first
ValueChoice.INDOMAIN_MEDIAN  # Try median value first
ValueChoice.INDOMAIN_RANDOM  # Random value
ValueChoice.INDOMAIN_SPLIT   # Binary split
```

## Example: Sudoku

```python
from ir.constraints import ConstraintModel, VariableRef, alldifferent, Objective

def create_sudoku_model():
    model = ConstraintModel("sudoku")
    
    # 9x9 grid of variables with values 1-9
    for i in range(9):
        for j in range(9):
            model.add_int_variable(f"cell_{i}_{j}", 1, 9)
    
    # Row constraints: all different in each row
    for i in range(9):
        row_vars = [VariableRef(f"cell_{i}_{j}") for j in range(9)]
        model.add_constraint(alldifferent(row_vars))
    
    # Column constraints: all different in each column
    for j in range(9):
        col_vars = [VariableRef(f"cell_{i}_{j}") for i in range(9)]
        model.add_constraint(alldifferent(col_vars))
    
    # 3x3 box constraints
    for box_row in range(3):
        for box_col in range(3):
            box_vars = []
            for i in range(3):
                for j in range(3):
                    r = box_row * 3 + i
                    c = box_col * 3 + j
                    box_vars.append(VariableRef(f"cell_{r}_{c}"))
            model.add_constraint(alldifferent(box_vars))
    
    model.set_objective(Objective.satisfy())
    return model
```

## Integration with Emitters

The Constraint IR is designed to work with the constraint emitters:

```python
from ir.constraints import ConstraintModel
from targets.constraints import MiniZincEmitter, CHREmitter

# Create your model
model = ConstraintModel("example")
# ... add variables, constraints, objective

# Emit to MiniZinc
mzn_emitter = MiniZincEmitter()
mzn_result = mzn_emitter.emit(model)
print(mzn_result.code)  # MiniZinc code

# Emit to CHR
chr_emitter = CHREmitter()
chr_result = chr_emitter.emit(model)
print(chr_result.code)  # Prolog CHR code
```

## See Also

- `targets/constraints/README.md` - Constraint emitters documentation
- `tests/ir/test_constraint_ir.py` - Comprehensive test cases
