# Rule-Based IR for Expert Systems

**Phase:** 7A (Expert Systems Foundation)  
**Status:** Complete

## Overview

The Rule-Based IR provides a comprehensive intermediate representation for expert systems supporting forward chaining inference with CLIPS and Jess emitters. This component enables STUNIR to represent, validate, and execute rule-based knowledge bases.

## Architecture

```
ir/rules/
├── __init__.py           # Package exports
├── rule_ir.py            # Core enums and data structures
├── pattern.py            # Pattern matching implementation
├── fact.py               # Fact and FactTemplate classes
├── rule.py               # Rule, Condition, Action classes
├── working_memory.py     # Working memory management
├── forward_chaining.py   # Forward chaining inference engine
└── README.md             # This documentation
```

## Components

### Core Types (rule_ir.py)

- **PatternType**: Literal, Variable, Wildcard, Multifield, Constraint
- **ConditionType**: Pattern, Test, AND, OR, NOT, EXISTS, FORALL
- **ActionType**: Assert, Retract, Modify, Bind, Call, Printout, Halt
- **ConflictResolutionStrategy**: Salience, Recency, Specificity, LEX, MEA, Random

### Facts (fact.py)

```python
from ir.rules import Fact, FactTemplate

# Create a template
template = FactTemplate(
    name="person",
    slots=[("name", "string"), ("age", "i32")],
    default_values={"age": 0}
)

# Create a template fact
fact = Fact(template_name="person", slots={"name": "John", "age": 30})

# Create an ordered fact
ordered_fact = Fact(values=("animal", "dog", "fido"))
```

### Patterns (pattern.py)

```python
from ir.rules import LiteralPattern, VariablePattern, WildcardPattern, PatternMatcher

# Match literal value
lit = LiteralPattern("hello")

# Bind to variable
var = VariablePattern("x")

# Match anything
wild = WildcardPattern()

# Pattern matcher
matcher = PatternMatcher()
result = matcher.match_pattern(pattern, fact, bindings)
```

### Rules (rule.py)

```python
from ir.rules import Rule, RuleBase, PatternCondition, PrintoutAction

# Create a rule
rule = Rule(
    name="greet",
    conditions=[
        PatternCondition(
            template_name="person",
            patterns=[("name", VariablePattern("n"))]
        )
    ],
    actions=[
        PrintoutAction(items=["Hello ", "?n", "crlf"])
    ],
    salience=10
)

# Create a rule base
rulebase = RuleBase(
    name="my-kb",
    templates=[template],
    rules=[rule],
    initial_facts=[fact]
)
```

### Working Memory (working_memory.py)

```python
from ir.rules import WorkingMemory, Fact

wm = WorkingMemory()

# Assert facts
fact_id = wm.assert_fact(Fact(values=("test",)))

# Retract facts
wm.retract_fact(fact_id)

# Modify facts
wm.modify_fact(fact_id, {"age": 31})

# Query facts
for fact in wm.get_facts_by_template("person"):
    print(fact)

# Listen for changes
wm.add_listener(lambda event, fact: print(f"{event}: {fact}"))
```

### Forward Chaining Engine (forward_chaining.py)

```python
from ir.rules import ForwardChainingEngine, RuleBase

# Create engine from rule base
engine = ForwardChainingEngine(rulebase)

# Reset to initial state
engine.reset()

# Add facts
engine.working_memory.assert_fact(Fact(values=("data",)))

# Run inference
fired_count = engine.run(max_iterations=1000)

# Step through inference
while engine.step():
    print(f"Fired: {engine.get_fired_rules()[-1]}")

# Get statistics
stats = engine.get_statistics()
print(f"Facts: {stats['facts_count']}, Fired: {stats['fired_count']}")
```

## Rule Structure

Rules follow the IF-THEN structure:

```
IF (conditions match facts in working memory)
THEN (execute actions)
```

### Condition Types

- **PatternCondition**: Match facts against patterns
- **TestCondition**: Evaluate predicate expressions
- **CompositeCondition**: AND, OR, NOT combinations

### Action Types

- **AssertAction**: Add new facts
- **RetractAction**: Remove facts
- **ModifyAction**: Change fact slots
- **BindAction**: Bind variables to values
- **CallAction**: Call functions
- **PrintoutAction**: Output text
- **HaltAction**: Stop execution

## Algorithm

The forward chaining engine implements basic Rete algorithm concepts:

1. **Match**: Find all rule instantiations that match current facts
2. **Conflict Resolution**: Order activations by salience
3. **Execute**: Fire the highest priority activation
4. **Update**: Propagate fact changes to agenda

## Example: Animal Identification

```python
from ir.rules import *

# Define the rule base
rb = RuleBase(
    name="animal-id",
    rules=[
        Rule(
            name="identify-mammal",
            conditions=[
                PatternCondition(None, [], [
                    LiteralPattern("has"),
                    VariablePattern("x"),
                    LiteralPattern("hair")
                ])
            ],
            actions=[
                AssertAction(None, {}, ["is", "?x", "mammal"])
            ]
        ),
        Rule(
            name="identify-dog",
            conditions=[
                PatternCondition(None, [], [
                    LiteralPattern("is"),
                    VariablePattern("x"),
                    LiteralPattern("mammal")
                ]),
                PatternCondition(None, [], [
                    LiteralPattern("says"),
                    VariablePattern("x"),
                    LiteralPattern("bark")
                ])
            ],
            actions=[
                AssertAction(None, {}, ["is", "?x", "dog"])
            ]
        )
    ]
)

# Run inference
engine = ForwardChainingEngine(rb)
engine.reset()
engine.working_memory.assert_fact(Fact(values=("has", "fido", "hair")))
engine.working_memory.assert_fact(Fact(values=("says", "fido", "bark")))
engine.run()

# Result: (is fido mammal), (is fido dog)
```

## Integration with CLIPS/Jess Emitters

Rule bases can be emitted to expert system languages:

```python
from targets.expert_systems import CLIPSEmitter, JessEmitter

# Emit to CLIPS
clips = CLIPSEmitter()
result = clips.emit(rulebase)
print(result.code)

# Emit to Jess (with Java integration)
jess = JessEmitter(java_integration=True)
jess.add_java_import("java.util.ArrayList")
result = jess.emit(rulebase)
```

## Testing

Run tests with:

```bash
python -m pytest tests/ir/test_rule_ir.py tests/ir/test_forward_chaining.py -v
```

## Related Components

- `targets/expert_systems/` - CLIPS and Jess emitters
- `manifests/` - Manifest generation for expert systems
