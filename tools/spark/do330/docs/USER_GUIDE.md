# DO-330 Tool Qualification Framework - User Guide

**Version:** 1.0.0  
**Date:** 2026-01-29  
**Standard:** DO-330 (Software Tool Qualification Considerations)

---

## 1. Overview

The STUNIR DO-330 Tool Qualification Framework provides a simplified, template-based approach for generating DO-330 certification packages. It integrates with DO-331/332/333 implementations to collect qualification data and generate required documentation.

### Key Features

- **Template-Based Generation**: Pre-defined templates for TOR, TQP, TAS, and other DO-330 artifacts
- **Data Collection**: Automatic collection from DO-331/332/333 outputs
- **Ada SPARK Implementation**: Memory-safe, formally verifiable code
- **Unified Reporting**: Combined qualification data from all supplements

---

## 2. Installation

### Prerequisites

- GNAT 2024 or later (Ada compiler)
- GNATprove (for SPARK proofs, optional)
- Make (build automation)

### Building

```bash
cd tools/do330
make build
```

This creates the `do330_generator` executable in `bin/`.

### Running Tests

```bash
make test
```

### Running SPARK Proofs

```bash
make prove
```

---

## 3. Quick Start

### Generate a certification package

```bash
./bin/do330_generator \
    --tool=verify_build \
    --tql=4 \
    --output=./certification_package
```

### Using the shell wrapper

```bash
./scripts/do330_generate.sh \
    --tool=verify_build \
    --tql=4 \
    --output=./pkg
```

---

## 4. Command Line Options

| Option | Description | Default |
|--------|-------------|----------|
| `--tool=<name>` | Tool name to qualify (required) | - |
| `--version=<ver>` | Tool version | 1.0.0 |
| `--tql=<1-5>` | TQL level | 5 |
| `--dal=<A-E>` | DAL level | E |
| `--output=<dir>` | Output directory | ./certification_package |
| `--template=<dir>` | Template directory | ./templates |
| `--include-do331` | Include DO-331 data | Yes |
| `--include-do332` | Include DO-332 data | Yes |
| `--include-do333` | Include DO-333 data | Yes |
| `--validate` | Validate package only | No |
| `--help` | Show help | - |

---

## 5. TQL Levels

| TQL | Description | Requirements |
|-----|-------------|-------------|
| TQL-1 | Most rigorous | All DO-330 data items, 100% coverage |
| TQL-2 | High rigor | Most data items, 90% coverage |
| TQL-3 | Moderate rigor | Standard data items, 80% coverage |
| TQL-4 | Lower rigor | TQP, TOR, tests, TAS |
| TQL-5 | No qualification | None (output verified externally) |

---

## 6. Generated Artifacts

### Core Documents

- **TOR.md** - Tool Operational Requirements
- **TQP.md** - Tool Qualification Plan
- **TAS.md** - Tool Accomplishment Summary

### Traceability

- **tor_to_test.json** - TOR to test case mapping
- **do330_objectives.json** - DO-330 objectives mapping

### Configuration

- **config_index.json** - Configuration items index

### Integration

- **do331_summary.json** - DO-331 model-based integration
- **do332_summary.json** - DO-332 OOP integration
- **do333_summary.json** - DO-333 formal methods integration

---

## 7. Template Customization

### Template Variables

Templates use `{{VARIABLE}}` syntax:

```
{{TOOL_NAME}}           - Tool identifier
{{TOOL_VERSION}}        - Version string
{{TQL_LEVEL}}           - TQL-1 through TQL-5
{{DAL_LEVEL}}           - DAL A through E
{{QUALIFICATION_DATE}}  - ISO date
{{AUTHOR}}              - Document author
```

### Custom Templates

1. Copy template from `templates/` directory
2. Modify as needed
3. Use `--template=<dir>` to specify custom location

---

## 8. Integration with DO-331/332/333

The framework automatically collects data from:

- `tools/do331/output/` - Model coverage, traceability
- `tools/do332/output/` - OOP metrics, class analysis
- `tools/do333/output/` - Proof results, VCs

### Data Collection Configuration

```json
{
  "integration": {
    "do331_path": "tools/do331/output/",
    "do332_path": "tools/do332/output/",
    "do333_path": "tools/do333/output/"
  }
}
```

---

## 9. Examples

### Example 1: TQL-4 Verification Tool

```bash
./bin/do330_generator \
    --tool=verify_build \
    --version=1.2.0 \
    --tql=4 \
    --dal=A \
    --output=./pkg/verify_build
```

### Example 2: TQL-5 Development Tool

```bash
./bin/do330_generator \
    --tool=ir_emitter \
    --tql=5 \
    --output=./pkg/ir_emitter
```

### Example 3: Validate Existing Package

```bash
./bin/do330_generator \
    --validate \
    ./pkg/verify_build
```

---

## 10. Troubleshooting

### Build Errors

1. Ensure GNAT 2024+ is installed
2. Run `make clean` before rebuilding
3. Check for missing dependencies

### Template Processing Errors

1. Verify template file exists
2. Check for missing required variables
3. Ensure output directory is writable

### Data Collection Errors

1. Verify DO-331/332/333 output directories exist
2. Check for valid JSON format in source files
3. Run tools with `--verbose` for diagnostics

---

## 11. Support

- **Documentation:** `tools/do330/docs/`
- **Templates:** `tools/do330/templates/`
- **Examples:** `tools/do330/examples/`

---

**Copyright (C) 2026 STUNIR Project**  
**SPDX-License-Identifier: Apache-2.0**
