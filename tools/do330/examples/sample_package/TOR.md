# Tool Operational Requirements (TOR)

**Document ID:** TOR-verify_build
**Version:** 1.0.0
**TQL Level:** TQL-4
**DAL Level:** DAL-A
**Date:** 2026-01-29
**Author:** STUNIR System
**Schema:** stunir.do330.v1

---

## 1. Tool Identification

- **Tool Name:** verify_build
- **Tool Version:** 1.0.0
- **Classification:** Criteria 2 (Verification Automation)
- **TQL Target:** TQL-4

## 2. Functional Requirements

### TOR-FUNC-001: Deterministic Output
- **Description:** The tool shall produce byte-identical outputs for identical inputs.
- **Verification Method:** Test
- **Derived From:** DO-178C Objective A-1

### TOR-FUNC-002: Valid Output Format
- **Description:** The tool shall generate output conforming to defined schemas.
- **Verification Method:** Test
- **Derived From:** User Requirement

### TOR-FUNC-003: Error Reporting
- **Description:** The tool shall report all errors with clear diagnostic messages.
- **Verification Method:** Test
- **Derived From:** User Requirement

### TOR-FUNC-004: Hash Verification
- **Description:** The tool shall verify SHA256 hashes of build artifacts.
- **Verification Method:** Test
- **Derived From:** STUNIR Spec

### TOR-FUNC-005: Manifest Validation
- **Description:** The tool shall validate manifest completeness and consistency.
- **Verification Method:** Test
- **Derived From:** STUNIR Spec

## 3. Environmental Requirements

### TOR-ENV-001: Operating System
- **Requirement:** Tool shall operate on Linux x86_64.
- **Verification:** Installation test

### TOR-ENV-002: Dependencies
- **Requirement:** Tool requires Python 3.9+ or GNAT 2024+.
- **Verification:** Dependency check script

## 4. Interface Requirements

### TOR-IF-001: Input Format
- **Requirement:** Tool accepts manifest JSON per STUNIR schema.
- **Verification:** Schema validation test

### TOR-IF-002: Output Format
- **Requirement:** Tool produces verification report JSON.
- **Verification:** Output validation test

### TOR-IF-003: Exit Codes
- **Requirement:** Tool returns 0 on success, non-zero on failure.
- **Verification:** Exit code test

## 5. Constraints

### TOR-CON-001: Determinism
- **Requirement:** Same inputs produce byte-identical outputs.
- **Verification:** Determinism test suite

### TOR-CON-002: No False Positives
- **Requirement:** Tool shall not report errors when none exist.
- **Verification:** Valid input test suite

### TOR-CON-003: No False Negatives
- **Requirement:** Tool shall detect all specified error conditions.
- **Verification:** Error detection test suite

## 6. Traceability

| TOR ID | Verification | DO-330 Objective | Status |
|--------|--------------|------------------|--------|
| TOR-FUNC-001 | TC-001 | T-1 | Verified |
| TOR-FUNC-002 | TC-002 | T-1 | Verified |
| TOR-FUNC-003 | TC-003 | T-1 | Verified |
| TOR-FUNC-004 | TC-004 | T-1 | Verified |
| TOR-FUNC-005 | TC-005 | T-1 | Verified |
| TOR-ENV-001 | TC-010 | T-1 | Verified |
| TOR-ENV-002 | TC-011 | T-1 | Verified |
| TOR-IF-001 | TC-020 | T-1 | Verified |
| TOR-IF-002 | TC-021 | T-1 | Verified |
| TOR-CON-001 | TC-030 | T-1 | Verified |
| TOR-CON-002 | TC-031 | T-1 | Verified |
| TOR-CON-003 | TC-032 | T-1 | Verified |

---

**Document History:**

| Version | Date | Author | Changes |
|---------|------|--------|----------|
| 1.0 | 2026-01-29 | STUNIR System | Initial release |
