# STUNIR - MADE BY AI FOR AI

##  Quick Start (For Humans Who Just Want This To Work)
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
It will lead to more questions you have to answer and you can ask questions about those questions.
The AI can tell you if what you're asking for is possible or not and will find the clear path(s) to your goal.
Can you understand code/mark-up?  If not, lean on the AI.
### WARNING: This isn't magic, you still have to put effort into it.
You just don't need to know any specific programming languages.
### CAUTION: You DO have to be open to learning programming and systems archetecture, though.
The AI can teach you there.  These wretched bundles of math are good for something healthy.
### **What Can You Do With This?**
If you've ever:
*  **Asked AI to generate a program** and got something that _almost_ works
*  **Needed to translate code** between languages (Python → Rust, JavaScript → Go, etc.)
*  **Had working code in one language** and wanted it in another (just paste your code/snippets as input!)
*  **Knew exactly what implementation you wanted** but only knew how to write it in one language
*  **Wanted the same code in multiple languages** without copy-pasting and praying
*  **Crashed out debugging boilerplate** that should've been automated
*  **Needed verifiable, reproducible builds** (so you can prove your code does what it says)
*  **Wanted AI-generated code that doesn't randomly change** every time you regenerate it
*  **Needed to ship the same logic** as a web app, CLI tool, and WASM module
*  **Wanted receipts/proofs** that your build pipeline is legit
...then STUNIR is for you.
### **What STUNIR Actually Does**
Think of it as a **safety harness for AI code generation**:
1. You (or your AI) provide input:
* A **spec** (what you want the program to do), OR
* **Existing code/snippets** (in any language you know)
1. STUNIR turns that into a **canonical reference** (the "true meaning" of your program)
2. STUNIR generates **actual code** in whatever languages you need (Python, Rust, JavaScript, Assembly, WASM, etc.)
3. STUNIR gives you **receipts** (cryptographic proof that everything was built correctly and reproducibly)
**The key difference:** Your AI can _suggest_ code, but STUNIR's deterministic tools are what actually _produce_ it. This means:
* No more "it worked yesterday but not today"
* No more "I swear I didn't change anything"
* No more guessing if AI hallucinated part of your build
**And if your inputs are no good?** You're using AI, so the model tells you exactly what you need to change in your spec/code. It's the future now.
### **Who Is This For?**
* **Vibe-coders** who want their vibes to compile reliably
* **People who know the implementation** but only in one language
* **Code translators** who are tired of manual porting
* **Hackathon heroes** who need multi-language output _fast_
* **People who got a link at a convention** and were told "just use this"
* **Anyone who's tired of AI code that's 90% perfect and 10% chaos**
* **Developers who need audit trails** for generated code
* **Teams shipping AI-generated code to ALPHA prototypes** (early testing phase)
### **TL;DR**
1. Download the pack
2. Upload it to your AI or give your AI the repo URL
3. Tell your AI what you want to build (or paste your existing code)
4. Get deterministic, verifiable, multi-language code
5. If something's wrong, the AI tells you exactly what to fix
6. Stop crashing out over boilerplate
---
**Below this point:** Technical deep-dive for people who want to understand how the sausage is made. If you just want to _use_ STUNIR, you're already done. Go forth and "code" responsibly.
---

## Ada SPARK is the Default Implementation Language

**STUNIR uses Ada SPARK as its PRIMARY implementation language for all tools.**

**Current Status**: Ada SPARK is the ONLY fully functional pipeline. Python and other language implementations are under active development.

### Tool Options: Choose Based on Your Needs

| Priority | Implementation | Status | Timeline |
|----------|---------------|--------|----------|
| **1. PRIMARY** | **Ada SPARK** | ✅ **Complete and functional** | Available now |
| 2. Alternative | Python | 🔨 **Under development** (incomplete) | Future release |
| 3. Native | Rust/Haskell | ⏳ Planned | Future |

### Precompiled SPARK Binaries (Recommended for ALPHA Testing)

STUNIR provides **precompiled Ada SPARK binaries** that eliminate the need for GNAT compiler installation:

```bash
# Spec to IR conversion
precompiled/linux-x86_64/spark/bin/stunir_spec_to_ir_main

# IR to code emission
precompiled/linux-x86_64/spark/bin/stunir_ir_to_code_main

# Embedded target emitter
precompiled/linux-x86_64/spark/bin/embedded_emitter_main
```

**Benefits of Precompiled Binaries:**
- ✅ No GNAT compiler required
- ✅ Instant usage (no build time)
- ✅ Platform-specific optimization
- ✅ Verified at build time

**Supported Platforms:**
- Linux x86_64: `precompiled/linux-x86_64/spark/bin/`
- macOS (coming soon)

### Building Ada SPARK from Source (Optional)

If you need to rebuild from source or target unsupported platforms:

```bash
cd tools/spark/
gprbuild -P stunir_tools.gpr
```

**Requirements:**
- GNAT compiler with SPARK support (FSF GNAT 12+ or GNAT Community Edition)
- See [`tools/spark/README.md`](tools/spark/README.md) for detailed build instructions

### Python Tools (Under Development)

**STATUS: Python implementations (`tools/spec_to_ir.py`, `tools/ir_to_code.py`) are INCOMPLETE and under active development.**

**Current State:**
- ⚠️ Partial functionality implemented
- ⚠️ Not fully aligned with Ada SPARK pipeline
- ⚠️ Missing receipt generation and verification
- ⚠️ Not production-ready

**Intended Purpose (when complete):**
- Full production pipeline alternative to Ada SPARK
- Easier to modify and extend
- Suitable for rapid prototyping and development
- Lower barrier to entry (no GNAT toolchain required)

**For production use NOW, use Ada SPARK binaries** (`tools/spark/bin/stunir_spec_to_ir_main`, `stunir_ir_to_code_main`)

**Why Ada SPARK is production-ready:**
- ✅ Deterministic execution with formal verification
- ✅ Complete spec→IR→code pipeline functional
- ✅ Hash-stable canonical output
- ✅ No runtime variations between executions
- ✅ DO-178C compliance support

**Python pipeline completion is planned** and many users are waiting for it. It will provide full production capability when development is complete.

---

## Getting Started with SPARK Tools (For Developers)

If you're a developer who wants to directly use STUNIR's Ada SPARK tools:

### Using Precompiled SPARK Binaries (Recommended)

**No installation required!** Just run the precompiled binaries:

```bash
# 1. Convert spec to IR
precompiled/linux-x86_64/spark/bin/stunir_spec_to_ir_main \
  --spec-dir spec/ \
  --output asm/spec_ir.txt

# 2. Generate code from IR
precompiled/linux-x86_64/spark/bin/stunir_ir_to_code_main \
  --ir-file asm/spec_ir.txt \
  --target c99 \
  --output build/generated/

# 3. Generate embedded code (ARM/AVR/MIPS)
precompiled/linux-x86_64/spark/bin/embedded_emitter_main \
  --ir-file asm/spec_ir.txt \
  --arch arm \
  --output build/embedded/
```

### Using the Build Script (Automated)

The easiest way to use STUNIR is through the automated build script:

```bash
# Run the full STUNIR pipeline (auto-detects SPARK)
./scripts/build.sh

# Verify the build
./scripts/verify.sh
```

The build script automatically:
- ✅ Detects precompiled SPARK binaries
- ✅ Falls back to built SPARK tools if precompiled unavailable
- ✅ Generates deterministic receipts
- ✅ Produces verification reports

### Building SPARK Tools from Source (Optional)

Only needed if:
- You're on an unsupported platform (non-Linux x86_64)
- You need to modify the SPARK source code
- You want to run SPARK formal verification

```bash
# Install GNAT with SPARK support (if not already installed)
# On Ubuntu/Debian:
# sudo apt-get install gnat-12 gprbuild

# Build the SPARK tools
cd tools/spark/
gprbuild -P stunir_tools.gpr

# Optional: Run SPARK formal verification
gnatprove -P stunir_tools.gpr --level=2
```

### Python Tools (Alternative Pipeline)

Python files (`tools/spec_to_ir.py`, `tools/ir_to_code.py`) provide a **fully functional alternative** to Ada SPARK:

**When to use Python:**
- ✅ Rapid prototyping and development workflows
- ✅ Easier to read, understand, and modify
- ✅ When GNAT/SPARK toolchain is unavailable
- ✅ CI/CD pipelines where speed matters more than formal verification
- ✅ Learning and experimentation
- ✅ Production use cases where formal verification is not required

**When to prefer Ada SPARK:**
- Safety-critical applications requiring DO-178C compliance
- Production systems requiring formal verification
- Reproducible builds with maximum determinism

**Both pipelines produce identical IR output** - choose based on your verification and workflow needs.

## What is STUNIR?
**STUNIR = Deterministic Multi-Language Code Generation**
STUNIR is a **model-facing, deterministic generation harness** for turning a human-authored **spec** into:
* a canonical **Intermediate Reference (IR)** (a canonical meaning object / referent; it is also an "intermediate representation" in the usual compiler sense)
* one or more **runtime outputs** (hosted and/or compiled)
* a **receipt bundle** (machine-checkable proofs/attestations) that binds the entire pipeline
STUNIR is designed for workflows where:
* Humans author _specs_ (and occasionally review proofs).
* Models propose and orchestrate.
* **Only deterministic tooling is trusted** to produce IR, artifacts, and receipts.
STUNIR is not meant to be a human-operated CLI product. Humans typically do **not** interact with a STUNIR pack directly.
## Where to start

### Understanding STUNIR's Ada SPARK Architecture

If you want to understand "how STUNIR works", start with:
* **`tools/spark/`** — Ada SPARK primary implementation (DO-178C Level A)
  * `tools/spark/src/stunir_spec_to_ir.adb` — spec to IR conversion logic
  * `tools/spark/src/stunir_ir_to_code.adb` — IR to code emission logic
  * `tools/spark/stunir_tools.gpr` — GNAT project file for building
* **`precompiled/linux-x86_64/spark/bin/`** — Precompiled SPARK binaries (recommended)
* **`scripts/build.sh`** — Polyglot build dispatcher (auto-detects and prioritizes SPARK)
* **`scripts/verify.sh`** — Receipt verification (uses SPARK verifier when available)
* **`docs/verification.md`** — Local vs DSSE verification model
* **`tools/native/README.md`** — Native verification stages (Rust/Haskell; SPARK-compatible)

### Pack/Container Interchange Format

If you are looking for the **pack/container** interchange format (used when shipping bundles between environments), start with:
* `ENTRYPOINT.md`
* `docs/root/stunir_pack_spec_v0.md`
## How STUNIR Works: Deterministic Pipeline Architecture
STUNIR exists because "model-written code" becomes practical only when the **model is not the authority**. Models can propose plans and edits; deterministic tooling must be the sole producer of commitments.

### Deterministic Pipeline
STUNIR implements a **standardized, deterministic sequence of steps** from inputs → IR → outputs, ensuring the process is replayable and verifiable.
This repo encodes that sequence explicitly (see `scripts/build.sh`).

### Hash Uniqueness (Foundation Layer)

STUNIR uses **hash uniqueness** as the determinism/equivalence foundation:
* Canonicalization produces stable byte-level representations
* Commitments/hashes bind artifacts to those representations
* Verification stays cheap (re-hash + compare)

Concretely, this repo normalizes JSON into canonical bytes (sorted keys, stable separators), so "same structure" ⇒ "same bytes" ⇒ "same sha256" (see `tools/spark/src/stunir_spec_to_ir.adb` for the primary SPARK implementation).

**Note**: Hash-based equivalence ensures syntactic identity. For semantic equivalence, see Semantic IR below.

### Semantic IR (Implemented)

STUNIR now implements **Semantic IR** (AST-based intermediate representation) that:
* Preserves semantic meaning, not just syntactic structure
* Allows different syntactic forms to normalize to the same semantic IR
* Enables true semantic equivalence checking
* Supports sophisticated code transformations

The Semantic IR system is available in `tools/rust/semantic_ir/` and provides:
* Canonical AST representation with semantic-preserving transformations
* Language-agnostic semantic analysis
* Cross-language equivalence verification

See `docs/SEMANTIC_IR_SPECIFICATION.md` for detailed design documentation.

### Future: Church-Rosser Confluence (CRIR Project)
Full Church–Rosser confluence properties will be implemented in the upcoming **CRIR (Church-Rosser Intermediate Reference)** utility—a separate project that will provide formal confluence guarantees across transformation pipelines.

STUNIR focuses on deterministic code generation; CRIR will focus on provable confluence.

### IR — Intermediate Reference
IR is treated as the canonical _referent_ that everything else is checked against:
* spec → IR (canonicalization / compilation)
* IR → outputs (generation)
* outputs → IR (verification / reconstruction / equivalence checks, where applicable)

"Reference" is intentional: the IR is what receipts bind and what verifiers check, regardless of how humans authored the spec.
## Origin / motivation (why this exists)
This harness came from a practical problem:
* A program design was already known, but hand-coding it was too slow/inefficient.
* Writing the same output in multiple languages multiplies that cost.

So the workflow became:
1. Use NLP prompts to get a model to propose a **spec**.
2. Deterministically compile the spec into a canonical IR.
3. Generate one or more language outputs from the IR.
4. Cross-check via hash-based receipts so determinism and auditability are built in.

The current implementation uses hash-based determinism as an efficient foundation with Semantic IR for semantic equivalence. The roadmap includes formal confluence properties (via the separate CRIR project).
## What STUNIR is (and is not)
### STUNIR is
* A repeatable way to compile **spec → IR → artifacts** under strict determinism controls.
* A receipt system that makes outputs auditable and replayable.
* A structure for treating models as **untrusted planners** and the toolchain as the **sole producer** of commitments.
### STUNIR is not
* A deployment platform.
* A "run these commands" developer UX.
* A promise that every supported language is equally turnkey as a runtime _today_.
STUNIR's contract is: **if an artifact exists, it is receipt-bound**.
## Core idea: shackle models to a deterministic toolchain
A model may:
* choose targets
* propose edits
* propose plans
…but it must not be the authority for:
* what the IR "means"
* what artifacts "are"
* whether a build "counts"
Instead, the deterministic toolchain produces:
* canonical IR commitments
* artifacts
* receipts
Verification is also deterministic. Models can present the raw proof, but the proof is generated by a deterministic process.
## Packs and attestation artifacts (distribution vs workspace)
STUNIR frequently needs to move results across machines (CI → auditor laptop, orchestrator → consumer, etc.). For that purpose, STUNIR defines a **pack**: a deterministic container + commitment that can be verified offline.
Terminology used in this repo:
* **Attestation artifacts**: the umbrella category for emitted evidence objects (step receipts, root attestation, envelopes, provenance/SBOM objects, etc.).
* **Receipt**: a step-scoped evidence object emitted by the harness.
* **Root attestation**: the bootstrap "shopping list" that inventories pack contents by digest.
### Pack bootstrap: root attestation
Pack v0 is rooted at:
* `root_attestation.dcbor` (preferred; canonical dCBOR)
To support highly constrained environments, pack v0 also defines an equivalent minimal-toolchain encoding:
* `root_attestation.txt` (line-oriented; designed for shell/PowerShell parsing)
If both are present, they must be equivalent and consumers should treat `root_attestation.dcbor` as authoritative.
See:
* `ENTRYPOINT.md`
* `docs/root/stunir_pack_spec_v0.md`
* `docs/root/stunir_pack_root_attestation_v0.md`
* `docs/root/stunir_pack_root_attestation_text_v0.md`
### Inclusion vs materialization
STUNIR distinguishes:
* **Included**: bytes are stored under `objects/sha256/` and referenced by digest in the root attestation.
* **Materialized**: bytes are written to user-chosen paths in a workspace.
Paths are UX; digests define identity.
(See `docs/root/stunir_pack_materialization_v0.md`.)
## Verification (small checkers, multiple toolchain levels)
This repo supports two verification layers (see `docs/verification.md`):
1. **Local verification** (default) — verify the receipts/manifests produced by `scripts/build.sh` in your working tree.
2. **DSSE verification** — verify a DSSE v1 envelope containing an in-toto Statement payload.
STUNIR also aims to make integrity verification possible in constrained environments. Practically, that means **verification profiles** (see `docs/verification_profiles.md`):
### Profile 1: Full verification (Python)
* Uses the repo's Python tools.
* Performs canonicalization checks, receipt integrity checks, IR manifest checks, etc.
### Profile 2: Portable verifier binary (no Python)
* Implemented as `stunir-native` in **two equivalent** native toolchains (Rust + Haskell). See `tools/native/README.md`.
* Includes:
  * IR validation (standardization gate): `stunir-native validate ...`
  * Pack / receipt verification (Profile-3-style pack verifier): `stunir-native verify pack ...`
* For pack verification, it aims to match the failure tags and behavior contract in `contracts/stunir_profile3_contract.json` (stable tag string + exit 1).
### Profile 3: Shell-Native (Minimal)
*   **Status:** Fully Implemented (Bootstrapping & Verification).
*   **Runtime:** POSIX Shell (Bash) + Coreutils. No Python required.
*   **Capabilities:**
    *   Toolchain Discovery & Locking (`scripts/lib/manifest.sh`)
    *   Receipt Generation (`scripts/lib/receipt.sh`)
    *   JSON Generation (`scripts/lib/json.sh`)
*   **Documentation:** See [docs/shell_native.md](docs/shell_native.md).

## Ada SPARK Core (Formal Methods Implementation)

The `src/ada/core/` directory contains an Ada SPARK 2014 implementation of STUNIR's core components. This implementation provides determinism guarantees through formal methods and static verification.

### Purpose

Ada SPARK enables compile-time proof of absence of runtime errors (no buffer overflows, no uninitialized reads, no integer overflows) and supports formal verification of functional correctness properties. This is useful for environments requiring high assurance or where determinism must be machine-checkable.

### Implementation Phases

The SPARK implementation is organized into four phases:

#### Phase 1: Core Utilities
Foundation types and operations used throughout the codebase:
- `STUNIR.Common` — shared type definitions
- `STUNIR.Config_Manager` — configuration loading and validation
- `STUNIR.Epoch_Manager` — epoch/timestamp handling for determinism
- `STUNIR.Type_System` — type definitions and constraints
- `STUNIR.Symbol_Table` — symbol storage and lookup

#### Phase 2: Build System
IR processing and build orchestration:
- `STUNIR.IR_Validator` — validates IR structure against schema
- `STUNIR.IR_Transform` — applies transformations to IR
- `STUNIR.Semantic_Checker` — semantic analysis passes
- `STUNIR.Dependency_Resolver` — resolves and orders dependencies
- `STUNIR.Build_System` — coordinates build pipeline stages

#### Phase 3: Test Infrastructure
Testing and verification support:
- `STUNIR.Test_Framework` — test harness and assertions
- `STUNIR.Property_Tests` — property-based test generators
- Unit tests for each component

#### Phase 4: Tool Integration
External tool coordination:
- `STUNIR.Toolchain_Discovery` — locates and fingerprints external tools
- `STUNIR.Receipt_Manager` — generates and validates receipts
- `STUNIR.Report_Generator` — produces verification reports
- `STUNIR.Tool_Interface` — interfaces with external compilers/provers

### Building and Testing

```bash
cd src/ada/core/
make          # Build all SPARK components
make test     # Run unit tests
make prove    # Run GNATprove (requires SPARK toolchain)
```

### Directory Structure

```
src/ada/core/
├── src/           # SPARK source files (.ads, .adb)
├── tests/         # Test sources
├── obj/           # Build artifacts
├── docs/          # SPARK-specific documentation
└── Makefile       # Build configuration
```

## Mechanics (how this repo works)
This section is intentionally concrete. If you want to understand "how STUNIR works", start with `scripts/build.sh` and the Ada SPARK implementations in `tools/spark/`.

### SPARK-First Architecture

STUNIR's deterministic pipeline is built around **Ada SPARK** as the primary implementation:

1. **Precompiled SPARK Binaries** (recommended): `precompiled/linux-x86_64/spark/bin/`
   - `stunir_spec_to_ir_main` — Spec to IR conversion
   - `stunir_ir_to_code_main` — IR to code emission
   - `embedded_emitter_main` — Embedded target code generation
   
2. **Source SPARK Tools** (if building from source): `tools/spark/bin/`
   - Same binaries, built from Ada SPARK source via GNAT

3. **Python Implementation** (fully functional alternative): `tools/*.py`
   - Complete, production-ready Python implementations
   - Use when SPARK toolchain is unavailable or for rapid development
   - Produces identical IR output to SPARK tools

### Polyglot Build System
The entry point `scripts/build.sh` now implements a **Polyglot Dispatcher** with SPARK priority:
1.  **Detects Runtime:** Checks for Precompiled SPARK -> Built SPARK -> Native Binary -> Python (reference) -> Shell.
2.  **Selects Profile:** Automatically picks the best available profile (SPARK Primary, Native, or Shell-Native).
3.  **Locks Toolchain:** Generates `local_toolchain.lock.json` to pin absolute paths and hashes of all tools.

### Determinism baseline
`scripts/build.sh` sets determinism-oriented defaults:
* `LC_ALL=C`, `LANG=C`, `TZ=UTC`
* `PYTHONHASHSEED=0`
* `umask 022`
### Epoch selection
Epoch selection uses `scripts/build.sh` logic (with Ada SPARK integration where available) to choose a single `selected_epoch` in priority order:
1. `STUNIR_BUILD_EPOCH`
2. `SOURCE_DATE_EPOCH`
3. `DERIVED_SPEC_DIGEST_V1` (deterministic; derived from the `spec/` tree digest)
4. `GIT_COMMIT_EPOCH` (derived from `git log -1 --format=%ct` when available)
5. else `ZERO` (0), unless explicitly allowing current time
The choice is written to `build/epoch.json`.
In strict mode, `scripts/build.sh` can forbid non-deterministic epochs; by default the pipeline should not require the user to manually provide an epoch.

### IR emission (current)
This repo currently emits IR using **Ada SPARK** as the primary implementation:

1. **Primary SPARK Implementation**: 
   - Tool: `precompiled/linux-x86_64/spark/bin/stunir_spec_to_ir_main` (or `tools/spark/bin/stunir_spec_to_ir_main`)
   - Output: `asm/spec_ir.txt` — deterministic manifest-style summary of spec JSON files (file + sha256 + optional id/name)
   - Source: `tools/spark/src/stunir_spec_to_ir.adb` (DO-178C Level A verified)

2. **Normalized IR files** (Ada SPARK):
   - Each `spec/**.json` is normalized into deterministic CBOR bytes (dCBOR-style map ordering)
   - The encoder uses canonical map key ordering and a configurable float policy
   - A manifest `receipts/ir_manifest.json` is written containing sha256 for each IR file (and epoch metadata when available)
   - Optional bundle output: `asm/ir_bundle.bin` with `receipts/ir_bundle_manifest.json`

3. **Python Implementation** (alternative):
   - Files: `tools/spec_to_ir.py`, `tools/spec_to_ir_files.py`
   - Complete, production-ready implementation
   - Use when SPARK toolchain is unavailable or for rapid development
   - Produces identical IR output to SPARK tools
#### dCBOR float policy (Ada SPARK implementation)

The **Ada SPARK implementation** handles float encoding with formal verification guarantees:
- Primary implementation: `tools/spark/src/stunir_spec_to_ir.adb`
- Float policy is built into the SPARK type system with SPARK contracts
- Ensures deterministic encoding with compile-time verification

**Python implementation** (`tools/dcbor.py`) provides the float encoding policy:
Policies:
* `forbid_floats` — reject any float values.
* `float64_fixed` — encode floats always as IEEE-754 float64 (deterministic, not "shortest form").
* `dcbor_shortest` — dCBOR-style numeric reduction and shortest-width float encoding.

Configuration:
* Environment: `STUNIR_CBOR_FLOAT_POLICY` (default: `float64_fixed`)
* CLI: `tools/spec_to_ir_files.py --float-policy` (overrides env)

Note: In `dcbor_shortest`, numeric reduction means values like `1.0` encode as the integer `1`, and both `0.0` and `-0.0` encode as integer `0` (consistent with the dCBOR draft rules for negative zero and integer reduction).

#### Canonical JSON note (numbers)
The **Ada SPARK** implementation provides deterministic JSON number handling with formal verification.

The **native** canonicalizers used by `stunir-native` currently **reject non-integer JSON numbers** (see `tools/native/README.md`). If you need Profile-2 validation/verification for an IR that uses floats, you must either keep the IR integer-only or extend the native stages to a full JSON canonicalization policy that includes floats (e.g., RFC 8785-style number formatting).
### Provenance commitment
After IR emission, provenance digests are computed using the Ada SPARK toolchain (or shell fallback):
* `spec_digest` = sha256 over sorted `(relpath + bytes)` traversal of `spec/`
* `asm_digest` = sha256 over sorted `(relpath + bytes)` traversal of `asm/`

It writes:
* `build/provenance.json`
* `build/provenance.h` (tiny header used by runtime tooling)

### Receipt emission
Receipt generation uses the Ada SPARK toolchain when available, with machine-checkable receipts that bind:
* target path + sha256
* build epoch + epoch manifest
* input files and directory digests
* tool identity (path + sha256) and argv
* `receipt_core_id_sha256` (excludes platform noise for stable receipts)

### Optional native tool build
If a C compiler exists, `scripts/build.sh` builds `tools/prov_emit.c` into `bin/prov_emit` and records a receipt.
If the toolchain is missing, a receipt still records the skip/requirement status.

### Verification (small checker)
`scripts/verify.sh` runs verification using **Ada SPARK verifier** when available:
* Primary: Ada SPARK verification tools (precompiled or built from source)
* Fallback: Native verification tools (`stunir-native`)
* Alternative: `python3 -B tools/verify_build.py --repo . --strict`

**SPARK Verifier** (primary) checks:
* `build/provenance.json` matches recomputed digests of `spec/` and `asm/`
* `build/provenance.h` is reproducible
* `receipts/spec_ir.json` sha256 matches `asm/spec_ir.txt`
* `receipts/ir_manifest.json` matches the `asm/ir/**.dcbor` set and sha256s (exact set in `--strict` mode)
* `receipts/prov_emit.json` sha256 matches `bin/prov_emit` when status is `BINARY_EMITTED`
* receipt epochs match `build/epoch.json`
## Pipeline (conceptual)
Inputs:
* **Spec** (human-authored)
* **STUNIR Pack** (this repository + its deterministic tooling)
* **Build Epoch** (explicit time pin / determinism control)
* **Target Selection** (what to generate)
Outputs:
* **IR commitment** (canonical, hash-addressed meaning)
* **Artifacts** (hosted runtimes, compiled runtimes, raw language modules)
* **Receipts** (proof bundle binding inputs → tools → outputs)
## Output taxonomy (stable even as language support grows)
Instead of maintaining a brittle "list of languages," STUNIR describes outputs by **execution mode** and **role**.
### Execution modes
#### 1) Hosted runtimes (mass-market)
A **hosted runtime** output is a complete runnable project whose execution engine is an existing runtime/VM already present in the environment.
If the host runtime is installed, this output functions like an "assembled runtime."
Examples:
* Python
* Node.js
* JVM
* .NET
* Ruby
* PHP
Receipt expectations (hosted):
* host runtime fingerprint (name/version/platform)
* dependency closure digest (lockfile/resolved set)
* entrypoint + generated sources digests
* determinism controls (epoch, env guards)
#### 2) Compiled runtimes (direct-to-environment)
A **compiled runtime** output produces a low-level artifact intended for execution without a language VM (or via a minimal platform runtime like a WASM engine).
Examples:
* Assembly (default) → native artifacts via assembler/linker
* WASM → portable modules (optionally WASI)
Receipt expectations (compiled):
* toolchain identity (assembler/linker/compiler) + flags
* target ABI/triple + platform assumptions
* produced artifact hashes
#### 3) Raw outputs (language artifacts)
A **raw output** is a language-facing artifact intended for interoperability, embedding, verification, or downstream compilation.
Raw outputs may be:
* general-purpose source (e.g., C variants, Rust, Go, Haskell, Erlang/Elixir, Lisp)
* logic/constraint modules (e.g., Prolog, Datalog, ASP, MiniZinc)
* proof/safety gate inputs (e.g., SMT-LIB)
Receipt expectations (solver/prover-backed raw modules):
* solver identity + flags + determinism settings (threads/seed)
* input module hash + data hash
* output certificate/model/trace hash (when available)
### Roles (how a target is used)
A single language can appear in multiple roles depending on how STUNIR uses it.
* **Host Runtime Role** (Python/Node/JVM/.NET/Ruby/PHP)
* the language runtime/VM is the execution engine
* **Backend Compiler Role** (C/Rust/Go/Haskell/Erlang/Elixir/…)
* the language toolchain produces a distributable runtime (native/WASM/BEAM/etc.)
* **Logic/Constraint Role** (Datalog/ASP/MiniZinc/Prolog)
* the artifact expresses rules/constraints/search, executed by an embedded engine or a pinned external solver
* **Proof/Safety Gate Role** (SMT)
* the artifact proves/checks properties; outputs are pass/fail plus checkable side artifacts when available
## Promotion workflow (staging → compiled distribution)
STUNIR supports a partner-friendly two-stage deployment model:
1. **Stage using hosted runtimes** (fast iteration)
* generate a hosted runtime project (e.g., Python/Node)
* partners integrate and test live on their preferred hosting platform
1. **Promote to compiled runtimes** (broad distribution)
* promote from the same IR commitment
* generate compiled deliverables (Assembly/WASM) for target machines
* ship artifacts together with receipts for verification and replay
### Live modification policy (important)
If a partner modifies hosted runtime code during staging, treat that change as either:
* a **receipt-tracked input** (overlay/patch/spec update that becomes part of spec/IR inputs), or
* a **local experiment** (not expected to match promoted compiled outputs until reconciled)
STUNIR does not attempt to "own" what happens beyond the point where deterministic verification completes.
## Receipts: what they are supposed to prove
A receipt bundle is meant to make the generation pipeline **checkable by a small deterministic verifier**.
At minimum, receipts should bind:
* the **exact inputs** (spec + pack + target selection + epoch)
* the **exact tools** used (versions, digests, flags)
* the **exact outputs** (artifact hashes)
Stronger receipt bundles may additionally include:
* canonicalization traces
* solver/prover side artifacts (models, unsat cores, certificates)
* deterministic replay logs
A verifier should be able to recompute commitments and confirm that the receipt bundle matches, without trusting a model.
## Integration contract (for orchestrators)
STUNIR is intended to be called by an orchestrator (often a model under constraints) that:
* provides a spec
* selects targets
* ensures determinism inputs (epoch / environment constraints)
* triggers deterministic generation and deterministic verification
The orchestrator is responsible for:
* ensuring the STUNIR pack itself is pinned by digest
* ensuring tool invocations are policy-compliant (e.g., single-threaded, pinned versions)
STUNIR is responsible for:
* canonical IR generation
* artifact generation
* receipt emission
* deterministic verification outputs
## Repository conventions
This repository may evolve, but STUNIR packs typically include:
* `spec/` — example or reference specs (optional)
* `tools/` — deterministic compilers/normalizers/receipt emitters
* `receipts/` — receipt schema, examples, and emitted receipt bundles
* `build/` — build outputs and intermediate commitments
* `asm/` / `wasm/` — compiled runtime outputs (when applicable)
* `schemas/` — JSON Schema / format definitions for targets, receipts, and IR
* `core/` — Ada SPARK 2014 implementation (formal methods)
## Design principles
* **Determinism-first**: outputs must be reproducible under the same inputs.
* **Canonical meaning**: IR has a unique normal form suitable for hashing and equivalence.
* **Small verifiers**: verification should be simpler than generation.
* **Models are untrusted**: models can propose; tools commit.
* **Extensible targets**: add languages by declaring mode + role, not by rewriting the worldview.
## Target descriptor (optional)
To avoid hardcoding language lists, targets can be described with a small descriptor object.
At minimum:
* `mode`: `hosted | compiled | raw`
* `role`: `host_runtime | backend_compiler | logic_constraint | proof_safety`
(See `schemas/` in this repo if present.)
## Editing policy (do not delete meaning)
This README is part of the harness contract. When updating it:
* do not silently delete naming/rationale/mechanics sections
* prefer additive edits or explicit deprecations
* if content is superseded, mark it as such rather than removing it
