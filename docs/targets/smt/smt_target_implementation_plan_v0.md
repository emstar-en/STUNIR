### SMT-LIB (SMT2) target implementation plan (STUNIR)

#### 0) Scope and intent
This is a concrete plan for adding/strengthening **SMT-LIB v2 (SMT2) raw constraint emission** in STUNIR.

Primary purpose: validate IR lowering and semantics using solver-backed sample-input checks.

Non-goals (v0):
- Full semantic coverage of the IR.
- Proof generation and cross-solver equivalence.

#### 1) Targets
Define two SMT targets:

- Target `smt` (baseline)
  - Meaning: **SMT2 source emission only**.
  - No solver required to emit artifacts.

- Target `smt_z3` (variant)
  - Meaning: **Z3-backed execution variant**.
  - Requires the `z3_solver` contract acceptance.

Design rule:
- Emitted `*.smt2` MUST be deterministic byte-for-byte.

#### 2) Artifact layout (exact paths)
Baseline (`smt`) outputs:
- Root: `asm/smt/portable/`
- Files (v0):
  - `asm/smt/portable/problem.smt2`
  - `asm/smt/portable/README.md` (optional, non-normative)
- Manifest: `receipts/smt_portable_manifest.json`
- Receipt: `receipts/smt_portable.json`

Variant (`smt_z3`) outputs:
- Root: `asm/smt/z3/`
- Files (v0):
  - `asm/smt/z3/problem.smt2`
  - `asm/smt/z3/solver_stdout.txt` (captured stdout)
- Manifest: `receipts/smt_z3_manifest.json`
- Receipt: `receipts/smt_z3.json`

#### 3) Receipt bindings (input closure)
SMT receipts MUST bind at least:
- `build/epoch.json`
- `build/provenance.json`
- `receipts/ir_manifest.json` and `receipts/ir_bundle_manifest.json`
- `asm/ir/` (or enumerated IR files actually read)
- the SMT backend tool sources used to emit constraints

Variant-only:
- `receipts/requirements.json`
- `receipts/deps/z3_solver.json`

#### 4) Determinism tests
Backend determinism (mandatory):
- Run SMT emission twice; compare file set + sha256 + manifest bytes.

Contract determinism (for `z3_solver`):
- Use the existing Z3 contract and hash solver stdout on a fixed vector.

#### 5) Verifier integration
Local verifier should optionally verify:
- `receipts/smt_*_manifest.json` digests vs on-disk files.
- `receipts/smt_*.json` receipt bindings.

DSSE mode:
- Prefer a single aggregate `receipts/output_manifest.json` that references per-target manifests.

#### 6) Implementation checklist
1) Update target requirements (patch provided in `spec/patches/`).
2) Add backend tool:
   - `tools/ir_to_smt2.py` with `--variant portable|z3`.
3) Extend `scripts/build.sh` to emit SMT targets and write receipts/manifests.
4) Extend `tools/verify_build.py` to verify SMT manifests if present.

#### 7) Acceptance criteria (v0)
- `STUNIR_OUTPUT_TARGETS=smt` emits deterministic SMT2 source + receipts, and local verify passes.
- `STUNIR_OUTPUT_TARGETS=smt_z3` gates on accepted `z3_solver` and binds solver stdout deterministically.
