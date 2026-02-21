# STUNIR DO-333 Formal Methods Support

DO-333 Formal Methods Supplement compliance tools for STUNIR.

## Overview

This module provides formal verification capabilities supporting DO-333 objectives:

| Objective | Description | Status |
|-----------|-------------|--------|
| FM.1 | Formal Specification | ✅ Implemented |
| FM.2 | Formal Verification (Proofs) | ✅ Implemented |
| FM.3 | Proof Coverage | ✅ Implemented |
| FM.4 | Verification Condition Management | ✅ Implemented |
| FM.5 | Formal Methods Integration | ✅ Implemented |
| FM.6 | Certification Evidence | ✅ Implemented |

## Features

- **Formal Specification Framework**: Pre/postconditions, invariants, ghost code
- **Proof Obligation Manager**: PO tracking, prioritization, coverage metrics
- **Verification Condition Tracker**: VC status, complexity analysis
- **SPARK Integration**: GNATprove wrapper with proof mode configuration
- **Evidence Generator**: Reports, compliance matrix, justification templates

## Requirements

- GNAT Ada compiler (2020 or later)
- SPARK Pro with GNATprove
- Set `STUNIR_ENABLE_COMPLIANCE=1` to enable

## Building

```bash
# Enable compliance tools
export STUNIR_ENABLE_COMPLIANCE=1

# Build
cd tools/do333
make build

# Run SPARK proofs
make prove

# Run tests
make test
```

## Usage

### Command Line

```bash
# Show help
./bin/do333_analyzer --help

# Run demonstration
./bin/do333_analyzer demo

# Analyze a project
./bin/do333_analyzer analyze <project.gpr>

# Generate reports
./bin/do333_analyzer report text
./bin/do333_analyzer report json
./bin/do333_analyzer report html
```

### Shell Script

```bash
# Via main verification script
../scripts/do333_verify.sh demo
../scripts/do333_verify.sh prove
../scripts/do333_verify.sh analyze myproject.gpr
```

### Python Wrapper

```python
from tools.do333.do333_wrapper import demo, analyze, generate_report

# Run demo
demo()

# Analyze project
analyze("myproject.gpr")

# Generate JSON report
generate_report("json")
```

## Architecture

```
tools/do333/
├── src/                          # Ada SPARK source
│   ├── formal_spec.ads/adb       # Formal specification types
│   ├── spec_parser.ads/adb       # Specification parser
│   ├── proof_obligation.ads/adb  # PO types
│   ├── po_manager.ads/adb        # PO collection manager
│   ├── verification_condition.ads/adb  # VC types
│   ├── vc_tracker.ads/adb        # VC collection tracker
│   ├── spark_integration.ads/adb # GNATprove integration
│   ├── gnatprove_wrapper.ads/adb # GNATprove wrapper
│   ├── evidence_generator.ads/adb # Report generation
│   ├── report_formatter.ads/adb  # Formatting utilities
│   └── do333_main.adb            # Main program
├── tests/                        # Ada SPARK tests
├── examples/                     # Example verified code
├── docs/                         # Documentation
├── do333.gpr                     # GNAT project file
├── Makefile                      # Build automation
├── do333_wrapper.py              # Python wrapper
└── README.md                     # This file
```

## SPARK Verification

All packages have `SPARK_Mode (On)` and are verified:

- No runtime exceptions
- All contracts proved
- Flow analysis clean

## Integration

### With DO-331 (SysML 2.0)

Formal specifications can reference DO-331 model elements.

### With DO-332 (OOP)

Formal verification supports Liskov Substitution Principle checks.

## Tool Qualification

- Classification: TQL-5 (development tool)
- Self-verified using SPARK proofs
- Qualification data available

## License

Apache-2.0

## Copyright

Copyright (C) 2026 STUNIR Project
