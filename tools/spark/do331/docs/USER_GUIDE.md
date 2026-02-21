# DO-331 Model-Based Development Tools - User Guide

**Version:** 1.0  
**STUNIR Project**  
**Standard:** DO-331 (Model-Based Development Supplement to DO-178C)

---

## Overview

The STUNIR DO-331 tools provide IR-to-Model transformation capability that generates DO-331 compliant model artifacts from STUNIR Intermediate Representation. Users then use their own qualified tools to generate code from these models.

## Quick Start

### 1. Enable Compliance Mode

```bash
export STUNIR_ENABLE_COMPLIANCE=1
export STUNIR_DAL_LEVEL=B  # A, B, C, D, or E
```

### 2. Run Transformation

```bash
# Using the shell script
scripts/do331_transform.sh --ir-dir asm/ir --output models/do331

# Or directly with the binary
bin/do331_main --transform input.json output.sysml
```

### 3. Review Output

The transformation generates:
- `*.sysml` - SysML 2.0 model files
- `trace_matrix.json` - Traceability data
- `coverage_report.json` - Coverage analysis

## DAL Level Configuration

| DAL | Coverage Requirements |
|-----|----------------------|
| A | MC/DC + Decision + Statement + State + Transition |
| B | Decision + Statement + State + Transition |
| C | Statement + State + Transition |
| D | Minimal |
| E | None |

## Output Formats

### SysML 2.0 Textual Notation

The primary output format. Example:

```sysml
package FlightController {
    action def CheckAltitude {
        in altitude : Real;
        out result : Boolean;
        
        // DO-331 Coverage Point: CP_ENTRY (entry)
        first start;
        
        then if altitude > 35000.0 [
            // DO-331 Coverage Point: CP_DEC_1_T (decision_true)
            then assign result := true;
        ] else [
            then assign result := false;
        ]
        
        then done;
    }
}
```

## Traceability

The trace matrix provides bidirectional traceability:

- **Forward:** IR element → Model element
- **Backward:** Model element → IR element
- **Transformation rule** applied
- **DO-331 objective** addressed

## Coverage Analysis

Coverage points are automatically instrumented based on DAL level:

- Entry/Exit points
- Decision points (true/false branches)
- State coverage
- Transition coverage
- MC/DC (DAL A only)

## Integration

The tools integrate with the STUNIR build pipeline when `STUNIR_ENABLE_COMPLIANCE=1`.

No changes to existing workflows are required for users who don't need compliance features.
