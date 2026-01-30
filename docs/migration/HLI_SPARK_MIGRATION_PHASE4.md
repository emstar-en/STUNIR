# HLI: SPARK Migration Phase 4 - Tool Integration

## Document Information
- **Generated:** 2026-01-29
- **Phase:** 4 - Tool Integration
- **Target:** Ada SPARK 2014+
- **Source Repository:** /home/ubuntu/stunir_repo/
- **Framework Path:** /home/ubuntu/stunir_implementation_framework/

---

## 1. Executive Summary

Phase 4 SPARK Migration focuses on tool integration utilities for DO-178C/DO-330 compliance. This phase migrates 6 critical Python/Shell tool integration scripts to Ada SPARK with formal verification.

### Component Overview

| # | Python/Shell Source | SPARK Component | Lines | Proofs | Priority |
|---|---------------------|-----------------|-------|--------|----------|
| 1 | `gen_receipt.py`, `record_receipt.py` | `core/report_generator/` | ~700 | ~50 | HIGH |
| 2 | `do331_wrapper.py`, `do331_transform.sh` | `core/do331_integration/` | ~600 | ~45 | HIGH |
| 3 | `do332_wrapper.py` | `core/do332_integration/` | ~500 | ~40 | HIGH |
| 4 | `do333_wrapper.py`, `do333_verify.sh` | `core/do333_integration/` | ~550 | ~45 | HIGH |
| 5 | `discover_toolchain.py` | `core/tool_interface/` | ~450 | ~35 | MEDIUM |
| 6 | `do330_generate.sh` | `core/compliance_package/` | ~500 | ~40 | MEDIUM |

**Total Estimated:** ~3,300 Ada lines, ~255 SPARK proofs

---

## 2. Analysis of Python/Shell Components

### 2.1 Report Generator (`tools/gen_receipt.py`)

**Purpose:** Generate deterministic receipts and reports in multiple formats.

**Key Operations:**
- Canonical JSON generation (RFC 8785 compliant)
- SHA-256 hash computation
- Receipt core ID calculation
- Multi-format output (Text, JSON, HTML, XML)

**Safety Requirements:**
- Deterministic output (same inputs → same outputs)
- No buffer overflows
- Bounded string operations
- Valid JSON/XML generation

### 2.2 DO-331 Integration (`tools/do331/do331_wrapper.py`)

**Purpose:** SysML 2.0 model transformation and coverage analysis.

**Key Operations:**
- IR to SysML 2.0 transformation
- Coverage data collection
- Traceability link generation
- Model validation

**Safety Requirements:**
- Valid model structure
- Complete traceability
- No data loss during transformation
- Bounded memory usage

### 2.3 DO-332 Integration (`tools/do332/do332_wrapper.py`)

**Purpose:** OOP verification for inheritance and polymorphism.

**Key Operations:**
- Class hierarchy analysis
- Inheritance depth verification
- Polymorphism safety checking
- Coupling metrics calculation

**Safety Requirements:**
- Sound OOP analysis
- No false negatives
- Bounded analysis time
- Valid metric ranges

### 2.4 DO-333 Integration (`tools/do333/do333_wrapper.py`)

**Purpose:** Formal verification integration with GNATprove.

**Key Operations:**
- Proof obligation management
- Verification condition tracking
- Evidence generation
- Proof status reporting

**Safety Requirements:**
- Complete VC tracking
- No lost proof obligations
- Valid proof status
- Deterministic evidence

### 2.5 External Tool Interface (`tools/discover_toolchain.py`)

**Purpose:** Safe external tool execution and discovery.

**Key Operations:**
- Tool discovery (path search)
- Version detection
- Safe command execution
- Output parsing

**Safety Requirements:**
- No command injection
- Bounded execution time
- Safe output parsing
- Error propagation

### 2.6 Compliance Package (`scripts/do330_generate.sh`)

**Purpose:** DO-330 certification package generation.

**Key Operations:**
- TOR/TQP/TAS generation
- Artifact collection
- Traceability matrix
- Configuration index

**Safety Requirements:**
- Complete package generation
- Valid document structure
- Deterministic output
- Audit trail

---

## 3. Architecture Design

### 3.1 Component Structure

```
core/
├── report_generator/
│   ├── report_types.ads          -- Report type definitions
│   ├── report_types.adb
│   ├── report_formatter.ads      -- Multi-format formatting
│   ├── report_formatter.adb
│   ├── json_emitter.ads          -- JSON output
│   ├── json_emitter.adb
│   ├── html_emitter.ads          -- HTML output
│   └── html_emitter.adb
├── do331_integration/
│   ├── do331_types.ads           -- DO-331 data types
│   ├── do331_types.adb
│   ├── do331_interface.ads       -- Integration interface
│   └── do331_interface.adb
├── do332_integration/
│   ├── do332_types.ads           -- DO-332 data types
│   ├── do332_types.adb
│   ├── do332_interface.ads       -- Integration interface
│   └── do332_interface.adb
├── do333_integration/
│   ├── do333_types.ads           -- DO-333 data types
│   ├── do333_types.adb
│   ├── do333_interface.ads       -- Integration interface
│   └── do333_interface.adb
├── tool_interface/
│   ├── external_tool.ads         -- External tool types
│   ├── external_tool.adb
│   ├── command_runner.ads        -- Safe command execution
│   └── command_runner.adb
└── compliance_package/
    ├── package_types.ads         -- Package types
    ├── package_types.adb
    ├── artifact_collector.ads    -- Artifact collection
    ├── artifact_collector.adb
    ├── trace_matrix.ads          -- Traceability matrix
    └── trace_matrix.adb
```

### 3.2 Integration with Existing Components

The Phase 4 components integrate with:
- Phase 1: Type system, IR validator, semantic checker
- Phase 2: Build orchestrator, receipt manager, epoch manager
- Existing DO-330/331/332/333 Ada implementations

---

## 4. Data Structure Mappings

### 4.1 Report Types (Python → Ada)

**Python:**
```python
receipt = {
    "schema": "stunir.receipt.build.v1",
    "target": str(target_path),
    "status": "success",
    "sha256": sha256_file(target_path)
}
```

**Ada SPARK:**
```ada
type Report_Format is (Text_Format, JSON_Format, HTML_Format, XML_Format);

type Report_Entry is record
   Key      : Entry_Key_String;
   Key_Len  : Key_Length_Type;
   Value    : Entry_Value_String;
   Value_Len: Value_Length_Type;
end record;

type Report_Data is record
   Schema      : Schema_String;
   Schema_Len  : Schema_Length_Type;
   Entries     : Entry_Array;
   Entry_Count : Entry_Count_Type;
   Format      : Report_Format;
   Is_Valid    : Boolean;
end record;
```

### 4.2 DO-33x Integration Types

**Ada SPARK:**
```ada
--  DO-331 Integration
type DO331_Result is record
   Models_Generated    : Natural;
   Coverage_Percentage : Percentage_Type;
   Traceability_Links  : Natural;
   Success             : Boolean;
end record;

--  DO-332 Integration
type DO332_Result is record
   Classes_Analyzed    : Natural;
   Inheritance_Valid   : Boolean;
   Polymorphism_Valid  : Boolean;
   Max_Depth           : Natural;
   Success             : Boolean;
end record;

--  DO-333 Integration
type DO333_Result is record
   Total_VCs     : Natural;
   Proven_VCs    : Natural;
   Unproven_VCs  : Natural;
   Proof_Rate    : Percentage_Type;
   Success       : Boolean;
end record;
```

### 4.3 External Tool Interface

**Ada SPARK:**
```ada
type Tool_Entry is record
   Name        : Tool_Name_String;
   Name_Len    : Name_Length_Type;
   Path        : Tool_Path_String;
   Path_Len    : Path_Length_Type;
   Version     : Version_String;
   Version_Len : Version_Length_Type;
   Available   : Boolean;
   Verified    : Boolean;
end record;

type Command_Result is record
   Exit_Code   : Integer;
   Output      : Output_Buffer;
   Output_Len  : Output_Length_Type;
   Timed_Out   : Boolean;
   Success     : Boolean;
end record;
```

---

## 5. Formal Specifications

### 5.1 Report Generator Contracts

```ada
--  Report generation must be deterministic
procedure Generate_Report
  (Data   : Report_Data;
   Format : Report_Format;
   Output : out Output_Buffer;
   Length : out Output_Length_Type;
   Status : out Generate_Status)
with
  Pre  => Data.Is_Valid and Data.Entry_Count > 0,
  Post => (if Status = Success then Length > 0 and 
           Output(1..Length)'Initialized);

--  JSON output must be canonical
procedure Emit_JSON
  (Data   : Report_Data;
   Output : out JSON_Buffer;
   Length : out JSON_Length_Type;
   Status : out Emit_Status)
with
  Pre  => Data.Is_Valid,
  Post => (if Status = Success then 
           Is_Canonical_JSON(Output(1..Length)));
```

### 5.2 Tool Interface Contracts

```ada
--  External command execution must be bounded
procedure Execute_Command
  (Command    : Command_String;
   Args       : Argument_Array;
   Timeout_MS : Positive;
   Result     : out Command_Result)
with
  Pre  => Command'Length > 0 and 
          Command'Length <= Max_Command_Length and
          Timeout_MS <= Max_Timeout_MS,
  Post => Result.Exit_Code in Valid_Exit_Range;

--  Tool discovery must find valid paths
procedure Discover_Tool
  (Name   : Tool_Name_String;
   Entry  : out Tool_Entry;
   Status : out Discover_Status)
with
  Pre  => Name'Length > 0 and Name'Length <= Max_Name_Length,
  Post => (if Status = Success then 
           Entry.Available and Entry.Path_Len > 0);
```

### 5.3 DO-33x Integration Contracts

```ada
--  DO-331 model transformation
procedure Transform_To_SysML
  (IR_Data    : IR_Data_Type;
   DAL_Level  : DAL_Type;
   Result     : out DO331_Result;
   Status     : out Transform_Status)
with
  Pre  => IR_Data.Is_Valid and DAL_Level in DAL_A .. DAL_E,
  Post => (if Status = Success then 
           Result.Success and Result.Models_Generated > 0);

--  DO-332 OOP analysis
procedure Analyze_OOP
  (IR_Data    : IR_Data_Type;
   DAL_Level  : DAL_Type;
   Result     : out DO332_Result;
   Status     : out Analysis_Status)
with
  Pre  => IR_Data.Is_Valid,
  Post => (if Status = Success then Result.Success);

--  DO-333 formal verification
procedure Verify_Formally
  (Source_Dir : Path_String;
   DAL_Level  : DAL_Type;
   Result     : out DO333_Result;
   Status     : out Verify_Status)
with
  Pre  => Source_Dir'Length > 0,
  Post => (if Status = Success then 
           Result.Success and 
           Result.Total_VCs = Result.Proven_VCs + Result.Unproven_VCs);
```

---

## 6. Implementation Guidance

### 6.1 Report Generator Implementation

1. **report_types.ads/adb**: Define bounded string types and report structures
2. **report_formatter.ads/adb**: Multi-format dispatch and formatting
3. **json_emitter.ads/adb**: RFC 8785 canonical JSON generation
4. **html_emitter.ads/adb**: Basic HTML report generation

Key considerations:
- Use fixed-size buffers for SPARK compatibility
- Implement deterministic key sorting for JSON
- Use preconditions to enforce valid input ranges

### 6.2 DO-33x Integration Implementation

1. Define integration types matching existing DO-33x Ada components
2. Create interface packages that bridge to existing implementations
3. Add SPARK contracts for all public operations
4. Implement error handling with bounded error arrays

### 6.3 External Tool Interface Implementation

1. **command_runner.ads/adb**: Safe subprocess execution
2. **external_tool.ads/adb**: Tool discovery and verification
3. Use Ada.Directories for path operations
4. Implement timeout handling for bounded execution

### 6.4 Compliance Package Implementation

1. Integrate with existing DO-330 package_generator
2. Add artifact collection with SHA-256 verification
3. Generate traceability matrices
4. Create configuration index

---

## 7. Testing Strategy

### 7.1 Unit Tests

- Test each component in isolation
- Verify contract satisfaction
- Test boundary conditions
- Verify deterministic output

### 7.2 Integration Tests

- Test DO-331/332/333 integration together
- Verify report generation with real data
- Test tool discovery on target platforms
- Validate compliance package completeness

### 7.3 Equivalence Tests

- Compare output with Python versions
- Verify identical JSON canonicalization
- Validate identical hash computation
- Check report format equivalence

### 7.4 SPARK Proof Strategy

1. Run GNATprove with --level=2 minimum
2. Target all preconditions/postconditions proven
3. Verify no runtime exceptions
4. Document any manual reviews required

---

## 8. Estimated Verification Conditions

| Component | VCs Estimated |
|-----------|---------------|
| Report Generator | ~50 |
| DO-331 Integration | ~45 |
| DO-332 Integration | ~40 |
| DO-333 Integration | ~45 |
| Tool Interface | ~35 |
| Compliance Package | ~40 |
| **Total** | **~255 VCs** |

---

## 9. Dependencies

### 9.1 Internal Dependencies

- Phase 1: STUNIR_Types, STUNIR_Strings, STUNIR_Hashes
- Phase 2: Receipt_Types, Epoch_Types, Build_Config
- Existing: DO-330/331/332/333 Ada packages

### 9.2 External Dependencies

- GNAT Runtime Library
- Ada.Directories
- Ada.Text_IO
- Ada.Strings.Fixed

---

## 10. Deliverables

1. **Ada SPARK Packages:**
   - 14 specification files (.ads)
   - 14 implementation files (.adb)
   - ~3,300 lines of Ada code

2. **Test Suite:**
   - 60+ new test procedures
   - Integration test driver
   - Equivalence test suite

3. **Documentation:**
   - API documentation (inline comments)
   - Migration guide
   - Integration guide

4. **Build System:**
   - Updated GNAT project files
   - Updated Makefiles
   - Shell script wrappers

---

## 11. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Complex external tool integration | Medium | High | Use bounded buffers, timeouts |
| JSON canonicalization edge cases | Low | Medium | Extensive testing |
| DO-33x interface changes | Low | Medium | Abstract interface layer |
| SPARK proof complexity | Medium | Medium | Incremental proof strategy |

---

## 12. Schedule Estimate

- **Week 1:** Report Generator + Tool Interface
- **Week 2:** DO-331/332/333 Integration
- **Week 3:** Compliance Package + Testing
- **Week 4:** Integration + Documentation + Proofs

**Total:** ~3-4 weeks (accelerated timeline)
