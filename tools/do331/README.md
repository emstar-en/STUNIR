# STUNIR DO-331 Model-Based Development Tools

[![SPARK](https://img.shields.io/badge/SPARK-Ada%202022-blue)](https://www.adacore.com/sparkpro)
[![DO-331](https://img.shields.io/badge/DO--331-Compliant-green)](https://www.rtca.org)

## Overview

This directory contains the STUNIR DO-331 Model-Based Development tools, implemented in Ada SPARK for formal verification.

The tools transform STUNIR Intermediate Representation (IR) to SysML 2.0 models with:
- Complete bidirectional traceability
- DAL-appropriate coverage instrumentation
- DO-331 compliance artifacts

## Quick Start

```bash
# Build
make build

# Run self-test
./bin/do331_main --test

# Run SPARK proofs (optional)
make prove
```

## Features

- **Model IR:** Type-safe representation of model elements
- **SysML 2.0 Emitter:** Generates valid SysML 2.0 textual notation
- **Traceability:** Bidirectional IR ↔ Model mapping
- **Coverage:** Automatic instrumentation per DAL level
- **SPARK Verified:** Formal proofs of key properties

## Directory Structure

```
tools/do331/
├── src/           # Ada SPARK source files
├── tests/         # Test programs
├── docs/          # Documentation
├── examples/      # Example transformations
├── do331.gpr      # GNAT project file
├── Makefile       # Build automation
└── README.md      # This file
```

## Requirements

- GNAT (Ada compiler)
- GNATprove (optional, for SPARK proofs)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|--------|
| STUNIR_ENABLE_COMPLIANCE | Enable compliance features | 0 |
| STUNIR_DAL_LEVEL | Target DAL (A/B/C/D/E) | C |
| STUNIR_MODEL_FORMATS | Output formats | sysml2 |

## Documentation

See the `docs/` directory for:
- [User Guide](docs/USER_GUIDE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Traceability](docs/TRACEABILITY.md)
- [Coverage](docs/COVERAGE.md)
- [DO-331 Compliance](docs/DO331_COMPLIANCE.md)

## License

Apache-2.0

## Contact

STUNIR Project - https://github.com/emstar-en/STUNIR
