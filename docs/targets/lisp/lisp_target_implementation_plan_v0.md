### Lisp target implementation plan (STUNIR)

#### 0) Scope and non-goals
This document is a **concrete implementation plan** for adding Lisp output to STUNIR as **two targets**:

1) **Portable Common Lisp baseline** (family-level, standard CL only)
2) **SBCL-first variant** (implementation-specific where appropriate; SBCL required)

This plan is written to avoid repo-crawl loops by anchoring all references to the repo index:
- `AI_START_HERE.md` (navigation index)
- `ENTRYPOINT.md` (pack entrypoint + navigation)

Non-goals (for this bundle):
- Implementing the full Lisp backend codegen now.
- Defining the complete IR→language lowering semantics.

The intent is to specify **exact targets**, **artifact layout**, **receipt bindings**, and **determinism tests** so implementation can proceed mechanically.

#### 1) Target names (authoritative)
STUNIR currently uses `STUNIR_OUTPUT_TARGETS` (comma-separated) for dependency resolution/probing.
We will add/define the following output targets:

- Target `lisp`  
  Meaning: **Portable Common Lisp baseline**.

- Target `lisp_sbcl`  
  Meaning: **SBCL-first variant**.

Design rule:
- The baseline target (`lisp`) MUST be **runnable (in principle)** on any conforming Common Lisp, but the build does not require any Lisp runtime.
- The SBCL variant (`lisp_sbcl`) MAY use SBCL extensions and MAY optionally produce SBCL-facing materialized artifacts.

#### 2) Dependency contracts & requirements mapping
Toolchain contracts live in `contracts/*.json` and are mapped by `contracts/target_requirements.json`.

##### 2.1 Required contracts per target
- `lisp`:
  - `required_contracts`: `[]`
  - `optional_contracts`: `[]` (note: the harness currently probes *required only*)

- `lisp_sbcl`:
  - `required_contracts`: `["sbcl"]`
  - `optional_contracts`: `[]`

Rationale:
- The portable baseline is **source emission only**; it should not depend on any external runtime.
- The SBCL variant explicitly chooses a concrete implementation and therefore requires the `sbcl` contract.

##### 2.2 Aliases
Keep existing aliases that map Common Lisp names to the baseline:
- `cl`, `commonlisp` → `lisp`

Add new aliases for explicitly requesting the SBCL variant:
- `sbcl_lisp`, `lisp_sbcl`, `cl_sbcl` → `lisp_sbcl`

(Exact alias keys are supplied in the patch file in this bundle.)

#### 3) Artifact layout (exact paths)
STUNIR already emits canonical IR into `asm/` and uses `receipts/` for manifests/receipts.
We will follow the same pattern as the IR file manifest (`receipts/ir_manifest.json` listing bytes under `asm/ir`).

##### 3.1 Baseline target (`lisp`): portable Common Lisp
Output directory (materialized, deterministic):
- `asm/lisp/portable/`

Minimum files (v0):
- `asm/lisp/portable/package.lisp`  (packages, nicknames, exported symbols)
- `asm/lisp/portable/runtime.lisp`  (small STUNIR runtime helpers, pure CL)
- `asm/lisp/portable/program.lisp`  (the generated program entry)
- `asm/lisp/portable/README.md`     (non-normative usage notes; may be omitted)

Canonical output manifest (file list + digests):
- `receipts/lisp_portable_manifest.json`

Build receipt (binds tool+argv+inputs to manifest digest):
- `receipts/lisp_portable.json` with:
  - `target`: `receipts/lisp_portable_manifest.json`
  - `status`: `CODEGEN_EMITTED_LISP_PORTABLE` (or `CODEGEN_SKIPPED` when not requested)

##### 3.2 SBCL target (`lisp_sbcl`): implementation-specific variant
Output directory (materialized, deterministic at the *source* level):
- `asm/lisp/sbcl/`

Minimum files (v0):
- `asm/lisp/sbcl/package.lisp`
- `asm/lisp/sbcl/runtime.lisp`  (may include `sb-ext:` features guarded by `#+sbcl`)
- `asm/lisp/sbcl/program.lisp`
- `asm/lisp/sbcl/README.md` (non-normative; may be omitted)

Optional additional artifacts (v1+), explicitly scoped:
- `asm/lisp/sbcl/materialized/` for SBCL-produced artifacts (e.g., FASL/core images) **only when explicitly enabled**.
  - These are implementation- and platform-sensitive.
  - If included, they MUST be listed in the corresponding manifest.

Canonical output manifest:
- `receipts/lisp_sbcl_manifest.json`

Build receipt:
- `receipts/lisp_sbcl.json` with:
  - `target`: `receipts/lisp_sbcl_manifest.json`
  - `status`: `CODEGEN_EMITTED_LISP_SBCL` (or `TOOLCHAIN_REQUIRED_MISSING` / `CODEGEN_SKIPPED`)

#### 4) Receipt bindings (what must be included as inputs)
Receipts are the binding layer; they must make it possible to re-check determinism and tool identity.

##### 4.1 Baseline (`lisp`) receipt input closure
`receipts/lisp_portable.json` MUST include (as `inputs` and/or `input-dirs`) at least:
- `build/epoch.json` (via `--epoch-json`, already captured in the receipt)
- `build/provenance.json`
- `receipts/ir_manifest.json` and `receipts/ir_bundle_manifest.json`
- `asm/ir/` as an input-dir (or the specific IR files the Lisp backend reads)
- the Lisp codegen tool source(s), e.g.:
  - `tools/ir_to_lisp_portable.py` (proposed)
  - shared libraries used by codegen (e.g., `tools/dcbor.py` if it reads dCBOR)

Rule:
- The portable target MUST NOT depend on any dependency acceptance receipt.

##### 4.2 SBCL (`lisp_sbcl`) receipt input closure
`receipts/lisp_sbcl.json` MUST include everything above PLUS:
- `receipts/requirements.json` (so the target selection and required contracts are bound)
- `receipts/deps/sbcl.json` (dependency acceptance receipt)

This creates an explicit binding:
- **Codegen output digest** is bound to **SBCL acceptance evidence**.

#### 5) Determinism tests
Determinism exists in two layers:
- (A) Toolchain contract determinism (dependency probing)
- (B) Backend/codegen determinism (STUNIR’s own emission)

##### 5.1 Toolchain contract determinism (SBCL)
The repo already has:
- `contracts/sbcl.json` with a deterministic script test (`test_vectors/lisp/hello_sbcl.lisp`).

Strengthening plan (recommended):
- Add a second SBCL contract test that compiles a fixed input file to a deterministic output file and checks byte equality across repeats.
- Use a test vector that avoids embedding host paths/time.

Notes:
- If SBCL’s FASL format embeds non-stabilized metadata, prefer a determinism invariant on a **runtime-produced output file** as the primary test (which is already present), and keep compile determinism as a best-effort capability check.

##### 5.2 Backend/codegen determinism (portable)
For `lisp` codegen, determinism is byte-level:
- Given the same `asm/ir` + epoch manifest, the emitted `asm/lisp/portable/*` MUST be byte-identical.

Test method:
- Run the Lisp codegen twice into two separate temp output roots.
- Compare:
  - file set equality
  - per-file sha256 equality
  - manifest bytes equality

Evidence capture (v0):
- Store only the final output + its manifest, and rely on the verifier to re-run codegen in DSSE mode.
Evidence capture (v1+):
- Extend the manifest schema to include `repeat` digests (like dependency probes).

##### 5.3 Backend/codegen determinism (SBCL variant)
For `lisp_sbcl` v0, determinism is required at the source-emission layer:
- `asm/lisp/sbcl/*` MUST be byte-identical.

Optional (v1+), only if materialized SBCL artifacts are enabled:
- Require determinism of the materialized outputs within a fixed environment.
- If not feasible, require deterministic *runtime outputs* rather than compiled artifact bytes.

#### 6) Verifier integrations
Local verifier (`tools/verify_build.py`) currently checks:
- `receipts/spec_ir.json`, `receipts/prov_emit.json`
- `receipts/ir_manifest.json`, `receipts/ir_bundle_manifest.json`

Additions (non-breaking):
- If `receipts/lisp_portable_manifest.json` exists, verify it is canonical JSON and that every file listed matches its digest.
- If `receipts/lisp_sbcl_manifest.json` exists, verify similarly.
- If `receipts/lisp_portable.json` / `receipts/lisp_sbcl.json` exist, verify they are canonical receipts and their `target` digest binds.

DSSE verifier mode already supports:
- rebuilding IR via `predicate.ir.rebuildCommand`
- rebuilding outputs via `predicate.codegen.rebuildCommand`
- verifying a single output manifest via `predicate.codegen.outputs.manifest`

Plan alignment:
- When producing a DSSE statement for Lisp outputs, set `predicate.codegen.outputs.manifest.uri` to an **aggregate output manifest** URI (see next section).

#### 7) Aggregate output manifest (recommended shape)
To fit DSSE mode (single manifest), create an aggregate manifest:
- `receipts/output_manifest.json`

Containing:
- entries for each emitted target manifest:
  - `receipts/lisp_portable_manifest.json`
  - `receipts/lisp_sbcl_manifest.json`
- and optionally direct entries for all leaf artifact files.

Then, the DSSE statement’s subject[0] can be:
- `name`: `receipts/output_manifest.json`
- `digest`: `{ "sha256": "..." }`

#### 8) Implementation checklist (mechanical)
1) Update `contracts/target_requirements.json`:
   - redefine `lisp` as portable (no required contracts)
   - add `lisp_sbcl` requiring `sbcl`
   - add new aliases

2) Add Lisp codegen tools (names are suggestions):
   - `tools/ir_to_lisp_portable.py`
   - `tools/ir_to_lisp_sbcl.py` (or a single tool with `--variant portable|sbcl`)

3) Extend `scripts/build.sh`:
   - parse `STUNIR_OUTPUT_TARGETS`
   - when it includes `lisp`, run portable codegen into `asm/lisp/portable/`
   - write `receipts/lisp_portable_manifest.json`
   - record `receipts/lisp_portable.json`
   - when it includes `lisp_sbcl`, ensure deps (already gated) and run SBCL variant codegen

4) Extend `tools/verify_build.py` local mode to verify Lisp manifests/receipts if present.

5) (Optional) Strengthen `contracts/sbcl.json` with an additional deterministic capability test.

#### 9) Acceptance criteria
The implementation is “done” for v0 when:
- `STUNIR_OUTPUT_TARGETS=lisp` produces:
  - `asm/lisp/portable/` deterministic bytes
  - `receipts/lisp_portable_manifest.json` canonical JSON, correct digests
  - `receipts/lisp_portable.json` canonical receipt, correct bindings
  - local verify passes (`scripts/verify.sh`)

- `STUNIR_OUTPUT_TARGETS=lisp_sbcl` produces:
  - `asm/lisp/sbcl/` deterministic bytes
  - `receipts/lisp_sbcl_manifest.json` canonical JSON, correct digests
  - `receipts/lisp_sbcl.json` canonical receipt, correct bindings
  - `receipts/deps/sbcl.json` accepted (from `scripts/ensure_deps.sh`)
  - local verify passes (with new optional Lisp checks)

#### 10) Files in this bundle
- `lisp_target_implementation_plan_v0.md` (this document)
- `rfc6902_target_requirements_lisp_split.json` (patch operations)
- `rfc6902_machine_plan_add_lisp.json` (optional patch operations)
- `lisp_targets_matrix.json` (machine-readable mapping: targets → outputs/receipts/contracts)
- `lisp_plan_bundle_manifest.json` (sha256 of bundle contents)
