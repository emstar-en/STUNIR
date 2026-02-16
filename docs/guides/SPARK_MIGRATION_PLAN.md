# STUNIR SPARK Migration Plan

## Executive Summary

This plan outlines the complete migration of all Python-based STUNIR components to Ada SPARK. The goal is to achieve DO-333 compliance and formal verification for all critical pipeline components.

## Migration Priority Matrix

### Phase 1: Core Pipeline (CRITICAL)
These are the essential components that form the STUNIR pipeline backbone.

| Component | Python File | SPARK Package | Complexity | Priority | Primary Outputs |
|-----------|-------------|---------------|------------|----------|-----------------|
| Spec Assembler | `bridge_spec_assemble.py` | `Spec_Assembler` | Medium | P0 | `spec.json` |
| IR Converter | `bridge_spec_to_ir.py` | `IR_Converter` | Medium | P0 | `ir.json` |
| Code Emitter | `bridge_ir_to_code.py` | `Code_Emitter` | High | P0 | target source files |
| Pipeline Runner | `stunir_pipeline.py` | `Pipeline_Driver` | Medium | P0 | end-to-end outputs |

### Phase 2: Extraction & Parsing (HIGH)
Components that parse source code and extract function signatures.

| Component | Python File | SPARK Package | Complexity | Priority | Primary Outputs |
|-----------|-------------|---------------|------------|----------|-----------------|
| C/C++ Parser | `extract_bc_functions.py` | `C_Parser` | High | P1 | extraction model |
| Signature Extractor | `extract_signatures.py` | `Signature_Extractor` | Medium | P1 | signatures list |
| Extraction Creator | `create_extraction.py` | `Extraction_Creator` | Low | P1 | `extraction.json` |

### Phase 3: Validation & Verification (HIGH)
Components that validate intermediate representations.

| Component | Python File | SPARK Package | Complexity | Priority | Primary Outputs |
|-----------|-------------|---------------|------------|----------|-----------------|
| IR Validator | `validate_ir.py` | `IR_Validator` | Medium | P1 | validation report |
| Spec Validator | `validate_spec.py` | `Spec_Validator` | Medium | P1 | validation report |
| IR Checker | `check_ir.py` | `IR_Checker` | Low | P1 | check summary |
| IR Error Reporter | `get_ir_errors.py` | `IR_Error_Reporter` | Low | P1 | error list |

### Phase 4: Analysis & Comparison (MEDIUM)
Components for analyzing and comparing outputs.

| Component | Python File | SPARK Package | Complexity | Priority | Primary Outputs |
|-----------|-------------|---------------|------------|----------|-----------------|
| File Comparator | `compare_files.py` | `File_Comparator` | Low | P2 | diff summary |
| Analysis Parser | `parse_analysis.py` | `Analysis_Parser` | Medium | P2 | analysis model |
| Function Counter | `analyze_function_counts.py` | `Function_Counter` | Low | P2 | counts report |
| Comprehensive Analyzer | `comprehensive_analysis.py` | `Comprehensive_Analyzer` | High | P2 | full analysis |

### Phase 5: Testing & Quality Assurance (MEDIUM)
Testing infrastructure and validation tools.

| Component | Python File | SPARK Package | Complexity | Priority | Primary Outputs |
|-----------|-------------|---------------|------------|----------|-----------------|
| Test Runner | `test_stunir_pipeline.py` | `Test_Runner` | Medium | P2 | test results |
| Golden Master Generator | `generate_golden_master.py` | `Golden_Master_Generator` | Medium | P2 | golden baselines |
| Simple Golden Creator | `create_simple_golden.py` | `Simple_Golden_Creator` | Low | P2 | baseline samples |
| Final Comparator | `final_comparison.py` | `Final_Comparator` | Medium | P2 | final report |
| Release Validator | `validate_release.py` | `Release_Validator` | Medium | P2 | release gate |

### Phase 6: Utilities & Infrastructure (LOW)
Supporting tools and utilities.

| Component | Python File | SPARK Package | Complexity | Priority | Primary Outputs |
|-----------|-------------|---------------|------------|----------|-----------------|
| Chunker | `stunir_chunker.py` | `Chunker` | Medium | P3 | chunked inputs |
| Executor | `stunir_executor.py` | `Executor` | High | P3 | execution logs |
| Index Updater | `update_indexes.py` | `Index_Updater` | Low | P3 | updated index |
| IR Converter (Legacy) | `convert_to_ir.py` | `Legacy_IR_Converter` | Medium | P3 | legacy IR |
| IR Fix Verifier | `verify_ir_fixes.py` | `IR_Fix_Verifier` | Low | P3 | fix report |

## Technical Architecture

### Directory Structure
```
tools/spark/
├── src/
│   ├── core/                    # Phase 1: Core Pipeline
│   │   ├── spec_assembler.ads/adb
│   │   ├── ir_converter.ads/adb
│   │   ├── code_emitter.ads/adb
│   │   └── pipeline_driver.ads/adb
│   ├── parsing/                 # Phase 2: Extraction & Parsing
│   │   ├── c_parser.ads/adb
│   │   ├── signature_extractor.ads/adb
│   │   └── extraction_creator.ads/adb
│   ├── validation/              # Phase 3: Validation
│   │   ├── ir_validator.ads/adb
│   │   ├── spec_validator.ads/adb
│   │   ├── ir_checker.ads/adb
│   │   └── ir_error_reporter.ads/adb
│   ├── analysis/                # Phase 4: Analysis
│   │   ├── file_comparator.ads/adb
│   │   ├── analysis_parser.ads/adb
│   │   ├── function_counter.ads/adb
│   │   └── comprehensive_analyzer.ads/adb
│   ├── testing/                 # Phase 5: Testing
│   │   ├── test_runner.ads/adb
│   │   ├── golden_master_generator.ads/adb
│   │   ├── simple_golden_creator.ads/adb
│   │   ├── final_comparator.ads/adb
│   │   └── release_validator.ads/adb
│   └── utils/                   # Phase 6: Utilities
│       ├── chunker.ads/adb
│       ├── executor.ads/adb
│       ├── index_updater.ads/adb
│       └── ir_fix_verifier.ads/adb
├── tests/                       # SPARK test suite
├── obj/                         # Build artifacts
└── Makefile                     # Build configuration
```

### Common Types Package
All components will share a common types package:

```ada
pragma SPARK_Mode (On);

with Ada.Strings.Bounded;

package STUNIR_Types is

   package JSON_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max => 1_000_000);
   subtype JSON_String is JSON_Strings.Bounded_String;

   package Identifier_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max => 256);
   subtype Identifier_String is Identifier_Strings.Bounded_String;

   package Type_Name_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max => 128);
   subtype Type_Name_String is Type_Name_Strings.Bounded_String;

   type Status_Code is (
      Success,
      Error_File_Not_Found,
      Error_Invalid_JSON,
      Error_Invalid_Schema,
      Error_Buffer_Overflow,
      Error_Unsupported_Type,
      Error_Parse_Error
   );

   type Parameter is record
      Name : Identifier_String;
      Param_Type : Type_Name_String;
   end record;

   type Parameter_Array is array (Positive range <>) of Parameter;

   type Function_Signature is record
      Name : Identifier_String;
      Return_Type : Type_Name_String;
      Params : Parameter_Array (1 .. 32);
      Param_Count : Natural range 0 .. 32;
   end record;

end STUNIR_Types;
```

## Implementation Strategy

### 1. JSON Handling in SPARK
Since SPARK has strict limitations on dynamic memory, we'll use:
- Bounded strings for all text data
- Fixed-size arrays with maximum limits
- Streaming JSON parser (SAX-style) instead of DOM
- Custom JSON lexer/parser with formal contracts

### 2. File I/O
- Use Ada.Text_IO with SPARK_Mode => Off wrappers
- All file operations return status codes
- Pre/post conditions verify file state

### 3. Error Handling
- No exceptions in SPARK code
- All functions return Status_Code
- Error messages written to separate log

### 4. Build System
```makefile
all: core parsing validation analysis testing utils

core:
    gnatmake -P core.gpr -Xmode=prove
    gnatprove -P core.gpr --level=4

parsing:
    gnatmake -P parsing.gpr -Xmode=prove
    gnatprove -P parsing.gpr --level=3
```

### 5. Interfaces and Compatibility
- JSON schemas remain source of truth for IO
- SPARK binaries mimic CLI flags of Python tools
- Backward compatibility with older `extraction.json` formats
- Deterministic ordering of JSON keys for reproducible outputs

### 6. Proof Strategy
- P0 components: contracts on all public subprograms
- P1 components: contracts on parsing and validation boundaries
- P2/P3 components: flow analysis and bounded data proofs
- Use global contracts to model shared state explicitly

## Verification Levels

Each component will be verified to:

- **Gold Level (P0 components)**: Full functional correctness proofs
- **Silver Level (P1 components)**: Flow analysis + partial proofs
- **Bronze Level (P2/P3 components)**: SPARK_Mode On, basic flow analysis

## Milestones

1. Phase 1 specification and implementation of core pipeline
2. Phase 1 proof completion at required levels
3. Phase 1 integration replacing Python binaries in CI
4. Phase 2 extraction and parsing implementation
5. Phase 3 validation and verification implementation
6. Phase 4 analysis tooling implementation
7. Phase 5 testing suite implementation
8. Phase 6 utility tooling implementation
9. Final integration with all Python fallbacks removed where approved

## Success Criteria

1. All P0 components compile with GNATprove at level 4
2. All Python tests pass against SPARK binaries
3. Zero runtime errors in formal verification
4. Performance within 2x of Python equivalents
5. Full DO-333 compliance documentation

## Risk Mitigation

- **Complexity Risk**: Start with simpler components to build expertise
- **Performance Risk**: Benchmark early and optimize critical paths
- **Integration Risk**: Maintain Python versions until SPARK is fully validated
- **Scope Risk**: Prioritize P0/P1 tooling to enable early adoption

## Quality Assurance Strategy

### 1. Verification vs Validation
- **Verification (SPARK)**: Mathematical proof that code matches contracts (Pre/Post).
- **Validation (Testing)**: Dynamic testing ensuring software meets user needs/schema requirements.

### 2. Testing Tiers
| Tier | Scope | Implementation | Frequency |
|------|-------|----------------|-----------|
| Unit (L1) | Single Functions | AUnit/GTest | Per Commit |
| Integration (L2) | Component Boundaries | Python Test Harness | Per Merge Request |
| System (L3) | End-to-End Pipeline | `stunir_pipeline_hardened.py` comparison | Nightly |
| Certification (L4) | Traceability & Coverage | GNATcoverage + GNATtest | Pre-Release |

### 3. Tool Qualification
For DO-333 credit, the verification tools themselves must be trusted:
- **GNATprove**: Qualified as a TQL-5 tool (verification tool).
- **Test Runner**: Qualified as a TQL-5 tool if used for formal test credit.

### 4. Continuous Integration
All SPARK components must require:
- Clean proof run (`gnatprove --level=2` minimum)
- 100% statement coverage on unit tests
- Passing regression suite against Python reference implementation

## Safety Case Construction
The SPARK migration supports the following safety arguments:
1. **Absence of Run-Time Errors (AoRTE)**: Proven by GNATprove.
2. **Data Flow Integrity**: Proven by SPARK flow analysis (Initialization, strict mode).
3. **Requirement Traceability**: Tests and contracts linked to JSON Schema requirements.

## Documentation Standards
- **Specs (.ads)**: Must contain formal contracts (`Pre`, `Post`, `Global`) and high-level comments.
- **Bodies (.adb)**: Must contain loop invariants and assertions to aid provers.
- **Architecture**: All design decisions documented in `design/` (to be created).

## Governance and Review

- All P0/P1 changes require proof artifacts review
- Each component must include test vectors aligned with schema versions
- External interfaces must be documented in code contracts

## Integration Plan

1. **Drop-in Replacement**: SPARK binaries maintain identical CLI flags and JSON IO.
2. **Dual-Run Mode**: CI executes both Python and SPARK binaries; diffs must be zero.
3. **Fallback Mechanism**: If SPARK binary fails, use Python and record failure.
4. **Cutover Criteria**: SPARK binary must pass proof level for its phase and test suite.

## Deprecation Policy

- Python components are deprecated only after SPARK equivalents pass all tests and proof requirements.
- Deprecated Python scripts remain for one release cycle before removal.
- Removal requires updating any orchestration or CI scripts.

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Proof Complexity | Schedule & cost | Prioritize simpler components and reuse contracts |
| JSON Schema Drift | Integration failures | Enforce schema versioning in contracts |
| Performance Regression | Pipeline slowdown | Benchmark after each phase |
| Toolchain Changes | Build breakage | Lock GNATprove versions in CI |

## Next Steps

1. Create `tools/spark/` directory structure
2. Implement `STUNIR_Types` common package
3. Begin Phase 1 with Spec Assembler
4. Set up continuous integration for SPARK builds
5. Create Python-to-SPARK test harness
