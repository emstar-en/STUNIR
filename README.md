# STUNIR

**STUNIR = Standardization Theorem + Unique Normals + Intermediate Reference**

STUNIR is a deterministic code generation harness that transforms human-authored specifications into:
- A canonical Intermediate Reference (IR)
- Runtime outputs (hosted and/or compiled)
- Receipt bundles for verification

---

## 50 Programming Languages Supported

### Hosted Runtimes
Python, JavaScript/Node.js, TypeScript, Ruby, PHP, Java, Kotlin, Scala, C#/.NET, F#, Go, Rust, Swift, Dart, Julia, R, Perl, Lua, Elixir, Clojure

### Compiled Targets
C, C++, Assembly (x86_64, ARM64, RISC-V), WebAssembly, LLVM IR, Zig, Nim, Crystal, D, Fortran

### Formal Methods & Logic
SPARK Ada, Coq, Agda, Lean, Idris, Prolog, Datalog, ASP, MiniZinc, SMT-LIB, TLA+, Alloy, Z3

### Domain-Specific
VHDL, Verilog, SystemVerilog, SQL, GraphQL, Protocol Buffers, Terraform HCL, Bash/Shell

---

## Quick Start

### Step 1: Get STUNIR
Download or clone the repository:
```bash
git clone https://github.com/emstar-en/STUNIR.git
```

### Step 2: Use with AI
- Upload the repository to your AI assistant (ChatGPT, Claude, etc.)
- Or share the repository URL if the AI has internet access

### Step 3: Describe Your Project
Tell the AI what you want to build. The system will generate code across supported languages.

---

## Core Components

| Component | Description |
|-----------|-------------|
| **Common** | Shared types and utilities |
| **Config_Manager** | Configuration handling |
| **Epoch_Manager** | Epoch/version tracking |
| **Type_System** | Type definitions and checking |
| **Symbol_Table** | Symbol management |
| **Semantic_Checker** | Semantic analysis |
| **IR_Validator** | Intermediate representation validation |
| **IR_Transform** | IR transformations |
| **Dependency_Resolver** | Dependency graph resolution |
| **Toolchain_Discovery** | Toolchain detection |
| **Build_System** | Build orchestration |
| **Receipt_Manager** | Receipt generation and handling |
| **Report_Generator** | Report generation |
| **Tool_Interface** | External tool integration |

---

## Testing

```bash
# Run Python tests
pytest tests/ -v

# Run Rust tests
cd tools/native/rust/stunir-native && cargo test

# Run SPARK tests (in core/)
cd core && make test

# Run integration tests
pytest tests/integration/ -v
```

---

## Core Principles

- **Determinism-first**: Outputs are reproducible given the same inputs
- **Canonical meaning**: IR has a unique normal form for hashing and equivalence
- **Small verifiers**: Verification is simpler than generation
- **Models are untrusted**: Models propose; tools commit

---

## Documentation

| Category | Links |
|----------|-------|
| **Getting Started** | [ENTRYPOINT.md](ENTRYPOINT.md) |
| **Technical** | [verification.md](docs/verification.md) • [shell_native.md](docs/shell_native.md) |
| **Development** | [Testing Strategy](docs/TESTING_STRATEGY.md) • [Native Tools](tools/native/README.md) |

---

## License

See [LICENSE](LICENSE) for details.
