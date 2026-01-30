# DO-330 Compliance Guide

**Version:** 1.0.0  
**Date:** 2026-01-29  
**Standard:** DO-330 (Software Tool Qualification Considerations)

---

## 1. Introduction

### 1.1 Purpose

This document describes how the STUNIR DO-330 Tool Qualification Framework supports compliance with DO-330 objectives for tool qualification.

### 1.2 Scope

This guide covers:
- DO-330 objectives mapping
- Tool classification methodology
- TQL determination criteria
- Qualification data requirements
- Integration with DO-178C supplements

---

## 2. DO-330 Overview

### 2.1 What is DO-330?

DO-330 provides guidance for qualifying software development tools used in DO-178C/DO-278A compliant projects. It defines:

- Tool classification criteria
- Tool Qualification Levels (TQL-1 through TQL-5)
- Required qualification data items
- Verification objectives

### 2.2 Tool Classification Criteria

| Criteria | Description | Impact |
|----------|-------------|--------|
| Criteria 1 | Tool output is part of airborne software and could insert errors | Highest qualification needed |
| Criteria 2 | Tool automates verification and could fail to detect errors | Moderate qualification needed |
| Criteria 3 | Tool output is verified by other means | Lowest/no qualification needed |

---

## 3. STUNIR Tool Classification

### 3.1 Classification Matrix

| Tool | Criteria | Rationale |
|------|----------|----------|
| `ir_to_model` | 3 | Model output verified by code generator |
| `spec_to_ir` | 3 | IR verified by downstream tools |
| `dcbor` encoder | 3 | Encoding verified by manifest |
| `canonical_json` | 3 | Output verified by hash comparison |
| `verify_build` | 2 | Automates verification |
| `verify_*_manifest` | 2 | Automates verification |
| `ir_to_code` | 1 or 3 | Depends on output verification |
| `gen_receipt` | 3 | Receipts independently verified |
| `stunir-native` | 2 | Automates verification |

### 3.2 TQL Determination

**For Criteria 1 Tools:**

| Software DAL | Required TQL |
|--------------|-------------|
| DAL A | TQL-1 |
| DAL B | TQL-2 |
| DAL C | TQL-3 |
| DAL D | TQL-4 |
| DAL E | TQL-5 |

**For Criteria 2 Tools:**

| Software DAL | Required TQL |
|--------------|-------------|
| DAL A | TQL-4 |
| DAL B | TQL-5 |
| DAL C | TQL-5 |
| DAL D | TQL-5 |
| DAL E | TQL-5 |

**For Criteria 3 Tools:**

All DAL levels = TQL-5 (no qualification required)

---

## 4. DO-330 Objectives Mapping

### 4.1 Tool Qualification Objectives

| Objective | Description | Implementation |
|-----------|-------------|----------------|
| T-0 | Tool Qualification Plan | TQP_template.txt |
| T-1 | Tool Operational Requirements | TOR_template.txt |
| T-2 | Tool Qualification | Data Collector |
| T-3 | Tool Accomplishment Summary | TAS_template.txt |
| T-4 | Tool Configuration Management | Config Index |
| T-5 | Tool Quality Assurance | QA Records |

### 4.2 Objective Applicability by TQL

| Objective | TQL-1 | TQL-2 | TQL-3 | TQL-4 | TQL-5 |
|-----------|-------|-------|-------|-------|-------|
| T-0 | ✓ | ✓ | ✓ | ✓ | - |
| T-1 | ✓ | ✓ | ✓ | ✓ | - |
| T-2 | ✓ | ✓ | ✓ | ✓ | - |
| T-3 | ✓ | ✓ | ✓ | ✓ | - |
| T-4 | ✓ | ✓ | ✓ | ✓ | - |
| T-5 | ✓ | ✓ | ✓ | - | - |

---

## 5. Qualification Data Requirements

### 5.1 Data Items by TQL

| Data Item | TQL-1 | TQL-2 | TQL-3 | TQL-4 | TQL-5 |
|-----------|-------|-------|-------|-------|-------|
| Tool Qualification Plan (TQP) | ✓ | ✓ | ✓ | ✓ | - |
| Tool Operational Requirements (TOR) | ✓ | ✓ | ✓ | ✓ | - |
| Tool Requirements Document (TRD) | ✓ | ✓ | ✓ | - | - |
| Tool Design Description | ✓ | ✓ | - | - | - |
| Tool Source Code | ✓ | - | - | - | - |
| Test Cases | ✓ | ✓ | ✓ | ✓ | - |
| Test Procedures | ✓ | ✓ | ✓ | ✓ | - |
| Test Results | ✓ | ✓ | ✓ | ✓ | - |
| Tool Accomplishment Summary (TAS) | ✓ | ✓ | ✓ | ✓ | - |
| Problem Reports | ✓ | ✓ | ✓ | ✓ | - |
| Configuration Index | ✓ | ✓ | ✓ | ✓ | - |

### 5.2 Framework Support

The STUNIR DO-330 Framework generates:

- ✅ TQP (Tool Qualification Plan)
- ✅ TOR (Tool Operational Requirements)
- ✅ TAS (Tool Accomplishment Summary)
- ✅ Verification Cases and Procedures
- ✅ Configuration Index
- ✅ Traceability Matrices

---

## 6. Integration with DO-178C Supplements

### 6.1 DO-331 (Model-Based Development)

The framework collects from DO-331 implementation:
- Model coverage metrics
- Traceability links (model ↔ code)
- SysML/XMI export status

### 6.2 DO-332 (Object-Oriented Technology)

The framework collects from DO-332 implementation:
- Classes analyzed
- Inheritance verification status
- Polymorphism verification status
- Coupling metrics

### 6.3 DO-333 (Formal Methods)

The framework collects from DO-333 implementation:
- Total verification conditions (VCs)
- Proven VCs
- Proof coverage percentage
- Prover information

---

## 7. Verification Requirements

### 7.1 Coverage Requirements by TQL

| TQL | Requirements Coverage | Structural Coverage |
|-----|----------------------|--------------------|
| TQL-1 | 100% TOR verified | MC/DC + 100% Statement |
| TQL-2 | 100% TOR verified | Decision + 100% Statement |
| TQL-3 | 100% TOR verified | 100% Statement |
| TQL-4 | 100% TOR verified | N/A |
| TQL-5 | N/A | N/A |

### 7.2 SPARK Proof Benefits

Using SPARK proofs provides additional assurance:
- Memory safety (AoRTE)
- Functional correctness
- Data flow analysis
- Information flow analysis

---

## 8. Certification Package Structure

```
certification_package/
├── index.json                    # Package manifest
├── TOR.md                        # Tool Operational Requirements
├── TQP.md                        # Tool Qualification Plan
├── TAS.md                        # Tool Accomplishment Summary
├── verification/
│   ├── test_cases.json
│   └── test_results.json
├── traceability/
│   ├── tor_to_test.json
│   └── do330_objectives.json
├── configuration/
│   └── config_index.json
├── integration/
│   ├── do331_summary.json
│   ├── do332_summary.json
│   └── do333_summary.json
└── qa_records/
    └── audit_trail.json
```

---

## 9. DER Submission Guidance

### 9.1 Preparing for DER Review

1. **Complete all required documentation** per TQL level
2. **Ensure traceability** from TOR to tests to results
3. **Resolve all problem reports** or document deviations
4. **Generate final TAS** with compliance status
5. **Create configuration baseline** with all artifacts

### 9.2 Common DER Questions

- How was the tool classification determined?
- What is the verification strategy for tool output?
- How is configuration management implemented?
- What is the impact of open problem reports?

---

## 10. References

- **DO-330**: Software Tool Qualification Considerations
- **DO-178C**: Software Considerations in Airborne Systems
- **DO-278A**: Software Considerations in CNS/ATM Systems
- **DO-331**: Model-Based Development and Verification
- **DO-332**: Object-Oriented Technology and Related Techniques
- **DO-333**: Formal Methods Supplement

---

**Copyright (C) 2026 STUNIR Project**  
**SPDX-License-Identifier: Apache-2.0**
