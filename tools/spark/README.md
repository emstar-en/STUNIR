# STUNIR SPARK Components

This directory contains Ada SPARK implementations of STUNIR pipeline components, replacing the Python reference implementations with formally verified alternatives while preserving existing JSON interfaces.

## Overview

The SPARK migration provides:
- **Formal Verification**: GNATprove-backed proofs for critical components
- **DO-333 Alignment**: Evidence for high-assurance certification workflows
- **Determinism**: Predictable outputs with bounded data structures
- **Safety**: Defensive handling of malformed inputs and bounded resource use

## Scope

Included:
- Core pipeline stages (spec assembly, IR conversion, code emission, orchestration)
- Parsing, validation, analysis, testing, and utility tooling

Not yet included:
- Remaining Python components pending migration (see `SPARK_MIGRATION_PLAN.md`)

## Inputs and Outputs

| Stage | Input | Output |
|-------|-------|--------|
| Spec Assembler | `extraction.json` | `spec.json` |
| IR Converter | `spec.json` | `ir.json` |
| Code Emitter | `ir.json` | target source files |
| Pipeline Driver | `extraction.json` | `spec.json`, `ir.json`, target source files |

## Target Languages

- C, C++, Rust, Go, Python
- JavaScript, Java, C#, Swift, Kotlin

## Directory Structure

```
tools/spark/
├── src/
│   ├── stunir_types.ads          # Common types (bounded strings, etc.)
│   ├── stunir_json_parser.ads    # Streaming JSON parser
│   ├── core/                      # Phase 1: Core Pipeline
│   │   ├── spec_assembler.ads     # extraction.json → spec.json
│   │   ├── ir_converter.ads       # spec.json → ir.json
│   │   ├── code_emitter.ads       # ir.json → target code
│   │   └── pipeline_driver.ads    # Pipeline orchestrator
│   ├── parsing/                   # Phase 2: Extraction & Parsing
│   ├── validation/                # Phase 3: Validation
│   ├── analysis/                  # Phase 4: Analysis
│   ├── testing/                   # Phase 5: Testing
│   └── utils/                     # Phase 6: Utilities
├── tests/                         # SPARK test suite
├── obj/                           # Build artifacts
├── Makefile                       # Build configuration
└── core.gpr                       # GNAT project file
```

## Components

### Phase 1: Core Pipeline (P0 - Critical)

| SPARK Package | Replaces Python | Status |
|--------------|-----------------|--------|
| `Spec_Assembler` | `bridge_spec_assemble.py` | Specification Complete |
| `IR_Converter` | `bridge_spec_to_ir.py` | Specification Complete |
| `Code_Emitter` | `bridge_ir_to_code.py` | Specification Complete |
| `Pipeline_Driver` | `stunir_pipeline.py` | Specification Complete |

### Phase 2: Extraction & Parsing (P1 - High)

| SPARK Package | Replaces Python | Status |
|--------------|-----------------|--------|
| `C_Parser` | `extract_bc_functions.py` | Pending |
| `Signature_Extractor` | `extract_signatures.py` | Pending |
| `Extraction_Creator` | `create_extraction.py` | Pending |

### Phase 3: Validation (P1 - High)

| SPARK Package | Replaces Python | Status |
|--------------|-----------------|--------|
| `IR_Validator` | `validate_ir.py` | Pending |
| `Spec_Validator` | `validate_spec.py` | Pending |
| `IR_Checker` | `check_ir.py` | Pending |

### Phase 4-6: Analysis, Testing, Utilities (P2/P3)

See `SPARK_MIGRATION_PLAN.md` for complete list.

## Building

### Prerequisites
- GNAT Pro or GNAT Community with SPARK support
- GNATprove for formal verification

### Build All Components
```bash
cd tools/spark
make all
```

### Build Modes
```bash
make core MODE=prove
make core MODE=debug
make core MODE=release
```

### Run Formal Verification
```bash
make prove
```

### Prove a Single Phase
```bash
make core-prove
make parsing-prove
make validation-prove
```

### Build Specific Phase
```bash
make core          # Phase 1
make parsing       # Phase 2
make validation    # Phase 3
```

### Clean
```bash
make clean
make distclean
```

### Run Tests
```bash
make test
```

## Usage

Once built, the SPARK binaries replace Python scripts:

```bash
# Instead of:
python bridge_spec_assemble.py -i extraction.json -o spec.json

# Use:
./bin/spec_assembler -i extraction.json -o spec.json

# Instead of:
python stunir_pipeline.py --input extraction.json --output ./out

# Use:
./bin/pipeline_driver -i extraction.json -o ./out
```

### Pipeline Flags

```bash
./bin/pipeline_driver \
  -i extraction.json \
  -o ./out \
  --targets c,cpp,rust,go,python,js,java,csharp,swift,kotlin \
  --emit-ir \
  --emit-spec
```

### Single-Stage Commands

```bash
./bin/spec_assembler -i extraction.json -o spec.json
./bin/ir_converter -i spec.json -o ir.json
./bin/code_emitter -i ir.json -o ./out --targets c,cpp
```

## Interoperability

- Inputs/outputs are JSON-compatible with existing Python tooling
- When SPARK binaries are missing, Python scripts remain the fallback
- All file names and extensions are preserved per target language

## Verification Levels

| Component Level | GNATprove Level | Description |
|----------------|-----------------|-------------|
| Gold (P0) | 4 | Full functional correctness proofs |
| Silver (P1) | 3 | Flow analysis + partial proofs |
| Bronze (P2/P3) | 1-2 | Basic flow analysis |

### Verification Artifacts

- `obj/` contains `.ali` and proof artifacts per unit
- GNATprove logs are generated per run and should be archived with build outputs
- Proof results are required for P0/P1 components before release

## Migration Status

- [x] Analysis of Python components
- [x] Directory structure created
- [x] Common types package (`STUNIR_Types`)
- [x] JSON parser specification
- [x] Phase 1 specifications complete
- [ ] Phase 1 implementations
- [ ] Phase 2-6 specifications
- [ ] Phase 2-6 implementations
- [ ] Build system complete
- [ ] Test suite
- [ ] Integration with remaining Python tools

## Python Components Status

### Replaced by SPARK
- `bridge_spec_assemble.py` → `Spec_Assembler`
- `bridge_spec_to_ir.py` → `IR_Converter`
- `bridge_ir_to_code.py` → `Code_Emitter`
- `stunir_pipeline.py` → `Pipeline_Driver`

### Pending Migration
All other Python files listed in `SPARK_MIGRATION_PLAN.md`

## Developer Workflow

### 1. Adding a New Feature
1. **Design**: Update `stunir_types.ads` if data structures change.
2. **Specify**: Write package spec `.ads` with formal contracts (`Pre`, `Post`).
3. **Verify Spec**: Run `gnatprove` on spec to check contract consistency.
4. **Implement**: Write package body `.adb`.
5. **Prove**: Run `gnatprove` iteratively until AoRTE is proven.
6. **Test**: Add unit tests in `tests/`.

### 2. Common Proof Issues
- **Loop Invariants**: Required for all loops. Must capture everything that changes.
- **flow errors**: Often mean uninitialized variables or missing `Global` contracts.
- **overflow check**: Use saturating arithmetic or preconditions to constrain inputs.

## CI/CD Integration

The build system is designed for easy CI integration:

```yaml
# Example GitLab CI / GitHub Actions step
spark-build:
  image: adacore/gnatpro:25.0
  script:
    - cd tools/spark
    - make all MODE=release
    - make test

spark-proof:
  image: adacore/gnatpro:25.0
  script:
    - cd tools/spark
    - make prove PROOF_LEVEL=2
  artifacts:
    paths:
      - tools/spark/gnatprove/
```

## Troubleshooting

| Error | Cause | Resolution |
|-------|-------|------------|
| `medium: overflow check might fail` | Arithmetic without constraints | Add `Pre` condition or use bounded types |
| `medium: range check might fail` | Array indexing | Assert index is within `'Range` |
| `high: initialisation of "X" failed` | Variable read before write | Initialize variable at declaration or ensure all paths write to it |
| `file "X.ads" not found` | Missing dependencies | Check `core.gpr` source dirs |

## Python Components Status

### Replaced by SPARK
- `bridge_spec_assemble.py` → `Spec_Assembler`
- `bridge_spec_to_ir.py` → `IR_Converter`
- `bridge_ir_to_code.py` → `Code_Emitter`
- `stunir_pipeline.py` → `Pipeline_Driver`

### Pending Migration
All other Python files listed in `SPARK_MIGRATION_PLAN.md`

## Contributing

When implementing SPARK packages:
1. Always use `pragma SPARK_Mode (On)`
2. Use bounded strings from `STUNIR_Types`
3. Return `Status_Code` instead of raising exceptions
4. Add pre/post conditions for verification
5. Avoid dynamic allocation in P0/P1 components
6. Run `gnatprove` before committing
7. Capture proof logs with build artifacts

## Artifacts

- `bin/` contains compiled SPARK binaries
- `obj/` contains `.o`, `.ali`, and proof artifacts
- Proof logs should be archived per build

## Configuration

- `MODE=prove|debug|release` controls build switches
- `PROOF_LEVEL=0..4` controls GNATprove effort
- `--targets` accepts a comma-separated list of target languages

## License

Apache-2.0 - See SPDX-License-Identifier in source files
