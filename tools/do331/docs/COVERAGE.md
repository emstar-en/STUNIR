# DO-331 Coverage Guide

## Overview

Model coverage analysis per DO-331 Table MB-A.5.

## Coverage Types

| Type | Description | Required For |
|------|-------------|-------------|
| State_Coverage | Each state visited | DAL A, B, C |
| Transition_Coverage | Each transition executed | DAL A, B, C |
| Decision_Coverage | Each decision true/false | DAL A, B |
| Condition_Coverage | Each condition true/false | DAL A |
| MCDC_Coverage | MC/DC analysis | DAL A only |
| Entry_Coverage | Function entry points | All DALs |
| Exit_Coverage | Function exit points | All DALs |

## DAL Requirements (DO-331 Table MB-A.5)

### DAL A
- Modified Condition/Decision Coverage (MC/DC)
- Decision Coverage
- Statement Coverage
- State Coverage
- Transition Coverage

### DAL B
- Decision Coverage
- Statement Coverage
- State Coverage
- Transition Coverage

### DAL C
- Statement Coverage
- State Coverage
- Transition Coverage

### DAL D/E
- Minimal requirements

## Coverage Point Format

```sysml
action def ProcessInput {
    // DO-331 Coverage Point: CP_ENTRY_1 (entry)
    first start;
    
    then if input > threshold [
        // DO-331 Coverage Point: CP_DEC_1_T (decision_true)
        then assign result := true;
    ] else [
        // DO-331 Coverage Point: CP_DEC_1_F (decision_false)
        then assign result := false;
    ]
    
    // DO-331 Coverage Point: CP_EXIT_1 (exit)
    then done;
}
```

## Coverage Report

```json
{
  "schema": "stunir.coverage.do331.v1",
  "statistics": {
    "total_points": 42,
    "instrumented": 42,
    "covered": 0
  },
  "dal_coverage": {
    "DAL_A": {
      "required": 42,
      "achieved": 0,
      "percent": 0,
      "meets_objective": false
    }
  }
}
```
