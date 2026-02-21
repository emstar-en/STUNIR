# STUNIR DO-332 OOP Verification Support

**Version:** 1.0.0  
**Standard:** DO-332 (Object-Oriented Technology Supplement to DO-178C)  
**Implementation:** Ada SPARK 2022

## Overview

This module provides DO-332 compliant verification tools for object-oriented constructs in STUNIR-generated code. It implements analyses for:

- **OO.1** - Inheritance analysis (depth, diamond detection, override verification)
- **OO.2** - Polymorphism verification (virtual functions, LSP checking)
- **OO.3** - Dynamic dispatch analysis (vtable construction, target resolution)
- **OO.4** - Object coupling analysis (CBO, RFC, LCOM metrics)
- **OO.5** - Exception handling in OOP contexts
- **OO.6** - Constructor/destructor verification

## Quick Start

### Build

```bash
# Build in debug mode
make build

# Build in release mode
make release

# Run SPARK proofs
make prove
```

### Run Analysis

```bash
# Run with default settings
./bin/do332_analyzer

# Run with specific options
./bin/do332_analyzer --ir-dir asm/ir --dal B --verbose

# Show help
./bin/do332_analyzer --help
```

## Architecture

```
tools/do332/
├── src/                      # Ada SPARK source
│   ├── oop_types.ads/adb     # Core OOP type definitions
│   ├── oop_analysis.ads/adb  # Analysis framework
│   ├── inheritance_*.ads/adb # Inheritance analysis
│   ├── polymorphism_*.ads/adb# Polymorphism verification
│   ├── dispatch_*.ads/adb    # Dispatch analysis
│   ├── coupling_*.ads/adb    # Coupling metrics
│   ├── test_*.ads/adb        # Test generation
│   └── do332_main.adb        # Main entry point
├── tests/                    # Test suite
├── docs/                     # Documentation
├── examples/                 # Example analyses
├── do332.gpr                 # GNAT project file
├── Makefile                  # Build automation
└── do332_wrapper.py          # Python wrapper
```

## DO-332 Objectives Coverage

| Objective | Description | Status |
|-----------|-------------|--------|
| OO.1 | Inheritance analysis | ✅ Implemented |
| OO.2 | Polymorphism verification | ✅ Implemented |
| OO.3 | Dynamic dispatch analysis | ✅ Implemented |
| OO.4 | Object coupling analysis | ✅ Implemented |
| OO.5 | Exception handling | 🔶 Basic |
| OO.6 | Constructor/destructor | 🔶 Basic |

## DAL Support

The analyzer supports configurable analysis depth based on DAL level:

| Analysis | DAL A | DAL B | DAL C | DAL D |
|----------|-------|-------|-------|-------|
| Inheritance depth | ✓ | ✓ | ✓ | ✓ |
| Diamond detection | ✓ | ✓ | ✓ | - |
| LSP checking | ✓ | ✓ | - | - |
| Dispatch timing | ✓ | - | - | - |
| Coupling metrics | ✓ | ✓ | ✓ | - |

## Output Formats

The analyzer produces JSON reports:

- `inheritance_report.json` - Inheritance analysis results
- `polymorphism_report.json` - Polymorphism verification
- `dispatch_report.json` - Dynamic dispatch analysis
- `coupling_report.json` - Coupling metrics
- `generated_tests.json` - Generated test cases

## Integration

### With STUNIR Build System

```bash
# Enable DO-332 analysis in build
export STUNIR_ENABLE_COMPLIANCE=1
export STUNIR_ENABLE_DO332=1
./scripts/build.sh
```

### With Python

```python
from do332_wrapper import DO332Analyzer

analyzer = DO332Analyzer()
results = analyzer.analyze("asm/ir", dal="B")
print(results.summary())
```

## SPARK Proofs

All analysis packages include SPARK contracts:

```bash
# Run proofs
make prove

# Generate proof report
make prove-report
```

Proof results are stored in `proof/`.

## License

Apache-2.0

## References

- [DO-332](https://www.rtca.org/products/do-332/) - Object-Oriented Technology Supplement
- [DO-178C](https://www.rtca.org/products/do-178c/) - Software Considerations in Airborne Systems
- [SPARK User Guide](https://docs.adacore.com/spark2014-docs/html/ug/)
