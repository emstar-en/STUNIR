### Python target implementation plan (STUNIR)

#### 0) Scope and intent
This is a concrete implementation plan for adding/strengthening **Python raw-code emission** in STUNIR.

Primary purpose: make STUNIR easier to build by enabling sample-input testing using a hosted runtime target.

Non-goals (v0):
- Fully specifying IR→Python semantics.
- Implementing a complete sample corpus.

#### 1) Targets
Define two Python targets:

- Target `python` (baseline)
  - Meaning: **Python source emission only**.
  - No Python runtime required to emit the artifacts.

- Target `python_cpython` (variant)
  - Meaning: **CPython-backed execution variant**.
  - Requires the `python_runtime` contract acceptance.

Design rule:
- `python` MUST be deterministic byte-for-byte given the same IR.
- `python_cpython` MAY execute sample inputs (and capture runtime outputs) but emission remains deterministic.

#### 2) Artifact layout (exact paths)
Baseline (`python`) outputs:
- Root: `asm/python/portable/`
- Files (v0):
  - `asm/python/portable/runtime.py`
  - `asm/python/portable/program.py`
  - `asm/python/portable/README.md` (optional, non-normative)
- Manifest: `receipts/python_portable_manifest.json`
- Receipt: `receipts/python_portable.json`

Variant (`python_cpython`) outputs:
- Root: `asm/python/cpython/`
- Files (v0):
  - `asm/python/cpython/runtime.py`
  - `asm/python/cpython/program.py`
  - `asm/python/cpython/README.md` (optional, non-normative)
- Manifest: `receipts/python_cpython_manifest.json`
- Receipt: `receipts/python_cpython.json`

#### 3) Receipt bindings (input closure)
Python receipts MUST bind at least:
- `build/epoch.json`
- `build/provenance.json`
- `receipts/ir_manifest.json` and `receipts/ir_bundle_manifest.json`
- `asm/ir/` (or enumerated IR files actually read)
- the Python backend tool sources used to emit code

Variant-only:
- `receipts/requirements.json`
- `receipts/deps/python_runtime.json`

#### 4) Determinism tests
Backend determinism (mandatory for both targets at emission layer):
- Run codegen twice into two temp roots.
- Compare file set equality and per-file sha256 equality.
- Compare manifest bytes equality.

Contract determinism (for `python_runtime`):
- Use the existing Python contract and a fixed test vector that writes a deterministic output file.

#### 5) Verifier integration
Local verifier should optionally verify:
- `receipts/python_*_manifest.json` digests vs on-disk files.
- `receipts/python_*.json` receipt bindings.

DSSE mode:
- Prefer a single aggregate `receipts/output_manifest.json` that references per-target manifests.

#### 6) Implementation checklist
1) Update target requirements (patch provided in `spec/patches/`).
2) Add backend tool:
   - `tools/ir_to_python.py` with `--variant portable|cpython`.
3) Extend `scripts/build.sh` to emit Python targets and write receipts/manifests.
4) Extend `tools/verify_build.py` to verify Python manifests if present.

#### 7) Acceptance criteria (v0)
- `STUNIR_OUTPUT_TARGETS=python` emits deterministic Python source + receipts, and local verify passes.
- `STUNIR_OUTPUT_TARGETS=python_cpython` gates on accepted `python_runtime` and binds any runtime outputs deterministically.
