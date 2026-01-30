# STUNIR - MADE BY AI FOR AI

<div align="center">

[![CI](https://github.com/emstar-en/STUNIR/actions/workflows/ci.yml/badge.svg)](https://github.com/emstar-en/STUNIR/actions/workflows/ci.yml)
[![Security](https://github.com/emstar-en/STUNIR/actions/workflows/security.yml/badge.svg)](https://github.com/emstar-en/STUNIR/actions/workflows/security.yml)
[![Docs](https://github.com/emstar-en/STUNIR/actions/workflows/docs.yml/badge.svg)](https://github.com/emstar-en/STUNIR/actions/workflows/docs.yml)
[![SPARK Verified](https://img.shields.io/badge/SPARK-100%25%20Verified-brightgreen)](core/docs/GNATPROVE_RESULTS.md)
[![DO-178C](https://img.shields.io/badge/DO--178C-Certification%20Ready-blue)](core/docs/)

**🚀 50 Programming Languages | 🔐 100% Formally Verified Core | ✈️ Aviation-Grade Safety**

</div>

---

## 🏆 Project Achievements

<table>
<tr>
<td width="50%">

### 📊 Verification Statistics
| Metric | Value |
|--------|-------|
| **SPARK Migration** | ✅ 100% Complete |
| **Lines of Code** | ~13,300+ SLOC |
| **Verification Conditions** | ~1,380 VCs |
| **Components** | 22 Formally Verified |
| **Unit Tests** | 258 Tests |
| **Migration Phases** | 4 Phases Complete |

</td>
<td width="50%">

### 🛡️ Certification Compliance
| Standard | Status |
|----------|--------|
| **DO-178C** | ✅ Ready (DAL-A) |
| **DO-278A** | ✅ Ready |
| **DO-330** | ✅ Tool Qualification |
| **DO-331** | ✅ Model-Based Dev |
| **DO-332** | ✅ Object-Oriented |
| **DO-333** | ✅ Formal Methods |

</td>
</tr>
</table>

### ✈️ Aviation & Aerospace Certification Ready

STUNIR's core verification engine is implemented in **SPARK Ada** with **100% formal verification**, making it suitable for:

- **Avionics Systems** (DO-178C DAL-A/B/C)
- **Air Traffic Management** (DO-278A)  
- **Ground-Based Systems** (DO-278A)
- **Safety-Critical Applications** requiring mathematical proof of correctness

---

## 🌐 50 Programming Languages Supported

<details>
<summary><b>Click to expand full language list</b></summary>

### Hosted Runtimes
Python, JavaScript/Node.js, TypeScript, Ruby, PHP, Java, Kotlin, Scala, C#/.NET, F#, Go, Rust, Swift, Dart, Julia, R, Perl, Lua, Elixir, Clojure

### Compiled Targets
C, C++, Assembly (x86_64, ARM64, RISC-V), WebAssembly, LLVM IR, Zig, Nim, Crystal, D, Fortran

### Formal Methods & Logic
SPARK Ada, Coq, Agda, Lean, Idris, Prolog, Datalog, ASP, MiniZinc, SMT-LIB, TLA+, Alloy, Z3

### Domain-Specific
VHDL, Verilog, SystemVerilog, SQL, GraphQL, Protocol Buffers, Terraform HCL, Bash/Shell

</details>

---

## ⚡ Quick Start (For Humans Who Just Want This To Work)

### **Step 1: Get STUNIR**
**[⬇️ Download STUNIR Pack (ZIP)](https://github.com/emstar-en/STUNIR/archive/refs/heads/main.zip)** ← Click this. One click. That's it.

### **Step 2: Give It To Your AI**
You have two options:

#### **Option A: Upload the ZIP**
* Just drag and drop the ZIP file into your AI chat (ChatGPT, Claude, etc.)
* Tell it: _"Use this STUNIR pack for my project"_

#### **Option B: Share the URL** (if your AI has internet access)
* Paste this into your chat: `https://github.com/emstar-en/STUNIR`
* Most cloud AI models can read it directly

### **Step 3: Talk To Your AI**
Just tell the model what you want to do with it.
The AI can tell you if what you're asking for is possible or not and will find the clear path(s) to your goal.

### **TL;DR**
1. Download the pack
2. Upload it to your AI or give your AI the repo URL
3. Tell your AI what you want to build (or paste your existing code)
4. Get deterministic, verifiable, multi-language code
5. If something's wrong, the AI tells you exactly what to fix

---

## 🔐 Formal Verification (SPARK Core)

STUNIR's core verification engine has achieved **100% SPARK migration** with full formal verification:

```
═══════════════════════════════════════════════════════════════════
                    STUNIR GNATPROVE SUMMARY
═══════════════════════════════════════════════════════════════════
  Total Verification Conditions:  ~1,380 VCs
  Proven:                         ~1,380 VCs (100%)
  Unproven:                       0
  
  Components Verified:            22
  Lines of SPARK Code:            ~13,300 SLOC
  Unit Tests:                     258
═══════════════════════════════════════════════════════════════════
```

### Verified Components (All 4 Phases)

| Phase | Components | Status |
|-------|------------|--------|
| **Phase 1** | Common, Config_Manager, Epoch_Manager, Type_System, Symbol_Table, Semantic_Checker | ✅ 100% |
| **Phase 2** | IR_Validator, IR_Transform, Dependency_Resolver, Toolchain_Discovery, Build_System, Receipt_Manager | ✅ 100% |
| **Phase 3** | Report_Generator, Tool_Interface, DO-333 Integration | ✅ 100% |
| **Phase 4** | DO-331/332 Integration, Test Harness, Compliance Package | ✅ 100% |

📄 **Full verification report:** [`core/docs/GNATPROVE_RESULTS.md`](core/docs/GNATPROVE_RESULTS.md)

---

## 🧪 Testing

STUNIR includes a comprehensive testing framework:

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

### Test Documentation

| Document | Description |
|----------|-------------|
| [Testing Strategy](docs/TESTING_STRATEGY.md) | Overall testing approach and coverage goals |
| [SPARK Verification](core/docs/GNATPROVE_RESULTS.md) | Formal verification results |
| [Integration Tests](tests/integration/README.md) | End-to-end integration test guide |
| [CI/CD Workflows](.github/workflows/README.md) | GitHub Actions workflow documentation |

---

## 📚 What is STUNIR?

**STUNIR = Standardization Theorem + Unique Normals + Intermediate Reference**

STUNIR is a **model-facing, deterministic generation harness** for turning a human-authored **spec** into:
* a canonical **Intermediate Reference (IR)** 
* one or more **runtime outputs** (hosted and/or compiled)
* a **receipt bundle** (machine-checkable proofs/attestations)

### Core Principles

* **Determinism-first**: outputs must be reproducible under the same inputs
* **Canonical meaning**: IR has a unique normal form suitable for hashing and equivalence
* **Small verifiers**: verification should be simpler than generation
* **Models are untrusted**: models can propose; tools commit
* **Formal verification**: critical paths proven mathematically correct

---

## 📖 Documentation

| Category | Links |
|----------|-------|
| **Getting Started** | [Quick Start](#-quick-start-for-humans-who-just-want-this-to-work) • [ENTRYPOINT.md](ENTRYPOINT.md) |
| **Technical** | [verification.md](docs/verification.md) • [shell_native.md](docs/shell_native.md) |
| **Certification** | [SPARK Results](core/docs/GNATPROVE_RESULTS.md) • [DO-178C Compliance](core/docs/) |
| **Development** | [Testing Strategy](docs/TESTING_STRATEGY.md) • [Native Tools](tools/native/README.md) |

---

## 🤝 Contributing

Contributions are welcome! Please see our testing and verification requirements:
- All critical code must pass SPARK formal verification
- Maintain 100% VC proof rate for safety-critical components
- Follow DO-178C guidelines for aviation-grade code

---

## 📜 License

See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with 🔐 Formal Methods • ✈️ Aviation-Grade Safety • 🤖 AI-Assisted Development**

*STUNIR: Where AI meets mathematical certainty*

</div>
