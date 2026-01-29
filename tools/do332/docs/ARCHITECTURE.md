# DO-332 OOP Verification Architecture

## Component Overview

```
┌──────────────────────────────────────────────────┐
│          do332_main.adb (Entry Point)        │
└───────────────────────┬──────────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
┌────────┴───────┐  ┌───┴───────┐  ┌─┴────────────┐
│   oop_types   │  │oop_analysis│  │test_generator │
└────────────────┘  └──────┬─────┘  └────────────────┘
                       │
    ┌────────────────┼────────────────┐
    │                │                │
┌───┴─────────┐ ┌───┴─────────┐ ┌───┴─────────┐
│ inheritance │ │ polymorphism│ │  dispatch   │
│  _analyzer  │ │  _verifier  │ │  _analyzer  │
└─────┬───────┘ └─────┬───────┘ └─────┬───────┘
      │             │             │
┌─────┴──────┐  ┌──┴────────┐ ┌──┴─────────┐
│ inheritance│  │substitut- │ │  vtable    │
│  _metrics  │  │ ability    │ │  _builder  │
└────────────┘  └───────────┘ └────────────┘

┌──────────────┐
│   coupling   │
│   _analyzer  │
└─────┬────────┘
      │
┌─────┴───────┐
│   coupling   │
│   _metrics   │
└──────────────┘
```

## Package Descriptions

### oop_types
Core type definitions for OOP constructs:
- Class_Info, Method_Info, Field_Info
- Inheritance_Link, Dependency
- VTable_Entry
- DAL requirements

### oop_analysis
Main analysis framework:
- Class hierarchy container
- Analysis configuration
- Result types
- Orchestration functions

### inheritance_analyzer
DO-332 OO.1 implementation:
- Depth calculation
- Diamond detection
- Circular detection
- Override verification
- Ancestor tracking

### polymorphism_verifier
DO-332 OO.2 implementation:
- Virtual method scanning
- Type counting
- LSP verification
- Covariance/contravariance

### dispatch_analyzer
DO-332 OO.3 implementation:
- Target resolution
- Site analysis
- Devirtualization
- Bounded dispatch proof

### coupling_analyzer
DO-332 OO.4 implementation:
- Dependency graph
- CBO, RFC, LCOM metrics
- Circular dependency detection
- Threshold checking

### test_generator
Test case generation:
- Per-objective tests
- Coverage point tracking
- Template-based generation

## Data Flow

```
IR Input → Parse → Build Hierarchy → Analyze → Generate Reports
                         │
                         ├─→ Inheritance Analysis
                         ├─→ Polymorphism Verification
                         ├─→ Dispatch Analysis
                         ├─→ Coupling Analysis
                         └─→ Test Generation
```

## SPARK Contracts

All analysis functions include SPARK contracts:
- Preconditions validate inputs
- Postconditions guarantee properties
- Type invariants ensure consistency
