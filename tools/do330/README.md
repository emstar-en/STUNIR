# STUNIR DO-330 Tool Qualification Framework

**Version:** 1.0.0  
**Status:** Production Ready  
**Standard:** DO-330 (Software Tool Qualification Considerations)

---

## Overview

The STUNIR DO-330 Tool Qualification Framework provides a simplified, template-based approach for generating DO-330 certification packages. It is implemented in Ada SPARK for memory safety and formal verification support.

## Features

- **Template System**: Pre-defined templates for TOR, TQP, TAS, and verification procedures
- **Data Collector**: Collects qualification data from DO-331/332/333 implementations
- **Package Generator**: Generates complete certification packages
- **SPARK Implementation**: Memory-safe, formally verifiable code
- **Integration**: Unified reporting from all DO-178C supplements

## Quick Start

```bash
# Build the framework
make build

# Run tests
make test

# Generate a certification package
./bin/do330_generator --tool=verify_build --tql=4 --output=./pkg
```

## Directory Structure

```
tools/do330/
├── src/                    # Ada SPARK source code
│   ├── templates.ads       # Template types
│   ├── templates.adb       # Template implementation
│   ├── template_engine.ads # Template engine spec
│   ├── template_engine.adb # Template engine impl
│   ├── data_collector.ads  # Data collector spec
│   ├── data_collector.adb  # Data collector impl
│   ├── package_generator.ads # Generator spec
│   ├── package_generator.adb # Generator impl
│   └── do330_main.adb      # Main entry point
├── tests/                  # Test suite
│   ├── test_templates.adb  # Template tests
│   ├── test_collector.adb  # Collector tests
│   └── test_generator.adb  # Generator tests
├── templates/              # Document templates
│   ├── TOR_template.txt    # Tool Operational Requirements
│   ├── TQP_template.txt    # Tool Qualification Plan
│   ├── TAS_template.txt    # Tool Accomplishment Summary
│   └── verification_template.txt
├── docs/                   # Documentation
│   ├── USER_GUIDE.md       # User guide
│   └── DO330_COMPLIANCE.md # Compliance guide
├── examples/               # Example packages
├── do330.gpr               # GNAT project file
└── Makefile                # Build automation
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--tool=<name>` | Tool name to qualify (required) |
| `--version=<ver>` | Tool version (default: 1.0.0) |
| `--tql=<1-5>` | TQL level (1=most rigorous) |
| `--dal=<A-E>` | DAL level (A=catastrophic) |
| `--output=<dir>` | Output directory |
| `--validate` | Validate package only |
| `--help` | Show help message |

## TQL Levels

| TQL | Description | Typical Use |
|-----|-------------|-------------|
| TQL-1 | Most rigorous | Code generators (DAL A) |
| TQL-2 | High rigor | Code generators (DAL B) |
| TQL-3 | Moderate | Code generators (DAL C) |
| TQL-4 | Lower rigor | Verification tools |
| TQL-5 | None needed | Output verified externally |

## Integration

Collects data from:
- `tools/do331/` - DO-331 Model-Based Development
- `tools/do332/` - DO-332 Object-Oriented Technology
- `tools/do333/` - DO-333 Formal Methods

## Building

### Requirements

- GNAT 2024 or later
- GNATprove (optional, for SPARK proofs)
- Make

### Targets

```bash
make build    # Build debug version
make release  # Build release version
make test     # Run all tests
make prove    # Run SPARK proofs
make clean    # Clean build artifacts
```

## Documentation

- [User Guide](docs/USER_GUIDE.md)
- [DO-330 Compliance Guide](docs/DO330_COMPLIANCE.md)

## License

Copyright (C) 2026 STUNIR Project  
SPDX-License-Identifier: Apache-2.0
