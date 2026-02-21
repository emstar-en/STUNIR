# DO-331 Compliance Documentation

## Standard Reference

**DO-331:** Model-Based Development and Verification Supplement to DO-178C and DO-278A

## Addressed Objectives

### MB.1 - High-Level Requirements

| Objective | How Addressed |
|-----------|---------------|
| MB.1.1 | Traceability matrix links model to requirements |
| MB.1.2 | Bidirectional traceability supported |
| MB.1.3 | Test case mapping via trace framework |

### MB.2 - Model Development

| Objective | How Addressed |
|-----------|---------------|
| MB.2.1 | IR-to-Model transformation preserves semantics |
| MB.2.2 | SysML 2.0 standard format |
| MB.2.3 | Deterministic transformation |

### MB.3 - Detail Design

| Objective | How Addressed |
|-----------|---------------|
| MB.3.1 | Action definitions from functions |
| MB.3.2 | State machines from behavioral specs |
| MB.3.3 | Complete structural mapping |

### MB.4 - Model Verification

| Objective | How Addressed |
|-----------|---------------|
| MB.4.1 | Review checklist generation |
| MB.4.2 | Static analysis support |
| MB.4.3 | Test vector generation framework |
| MB.4.4 | Coverage point tracking |

### MB.5 - Model Coverage

| Objective | How Addressed |
|-----------|---------------|
| MB.5.1 | Automatic coverage instrumentation |
| MB.5.2 | DAL-appropriate coverage |
| MB.5.3 | Coverage reporting |

### MB.6 - Configuration Management

| Objective | How Addressed |
|-----------|---------------|
| MB.6.1 | Hash-based artifact identification |
| MB.6.2 | Traceability to source IR |
| MB.6.3 | Deterministic output |

## Tool Qualification

### Classification

**Tool Type:** Development Tool (Criteria 3 - Output Verified)  
**TQL Level:** TQL-5 (when output is verified by qualified downstream tools)

### Rationale

1. STUNIR generates models, not executable code
2. Users verify model output with their own tools
3. Users generate code using DO-330 qualified code generators
4. All transformation rules are documented
5. Deterministic output enables reproducibility

### Qualification Evidence

- Tool Operational Requirements (TOR)
- Tool Qualification Plan (TQP)
- Test cases for all transformation rules
- SPARK proof artifacts

## Limitations

1. **Code Generation:** STUNIR does NOT generate executable code from models. Users must use their own qualified tools.

2. **Model Simulation:** STUNIR does NOT simulate models. Users must use qualified simulation tools.

3. **Requirements Management:** STUNIR traces to requirements but does not manage requirements.
