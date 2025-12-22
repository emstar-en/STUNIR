# STUNIR

STUNIR is a **deterministic, receipt-emitting generation harness** for building programs from a **JSON specification pack** via a canonical **Intermediate Representation (IR)**.

It is designed for workflows where you want to *generate*, *re-generate*, and *verify* artifacts (IR, assembly/binary outputs, manifests) with **machine-checkable provenance**—so you can answer:

- *What inputs produced this output?*
- *Which tools ran, with which versions/arguments?*
- *Was the build time/epoch pinned?*
- *Do the hashes of the inputs/outputs match the recorded receipts?*

In short: **STUNIR turns “generation” into a reproducible build** with a paper trail.

## What STUNIR *is* (and isn’t)

### STUNIR is

- A **pack format** (directory layout + JSON files) for describing a build as data.
- A **spec → IR → artifacts** pipeline where IR is **canonicalized** (unique normal form) so equality checks are cheap.
- A **receipt/provenance system** that records deterministic commitments (hashes) to:
  - the spec inputs
  - the IR outputs
  - the emitted artifacts
  - the toolchain and build epoch

### STUNIR is *not*

- A monolithic compiler for a specific language.
- A full assembler or C toolchain (you can *optionally* require one, but STUNIR itself is the harness around tools).
- A magical “LLM writes code” system—STUNIR assumes the planner can be untrusted; **the deterministic toolchain and receipts are the source of truth**.

## Why this exists

Modern “generate my program” workflows often fail the moment you ask for auditability:

- outputs change run-to-run
- inputs are implicit (environment, time, hidden defaults)
- there is no *receipt* you can independently re-check

STUNIR’s stance is:

1. **Generation must be replayable.**
2. **Meaning should be captured in canonical IR.**
3. **Every step should emit checkable receipts.**

That makes it practical to “shackle” non-deterministic planning (including ML/LLM proposals) to a deterministic, verifiable production pipeline.

## Repository / pack layout (high level)

You’ll typically see directories like:

- `spec/` — JSON specification inputs (the authoritative intent)
- `asm/` — assembly-related artifacts and/or low-level IR
- `receipts/` — machine-checkable build receipts (hash commitments, parameters, epochs)
- `tools/` — deterministic tooling (spec→IR, provenance hashing, receipt emitters)
- `scripts/` — orchestration scripts (build / verify entrypoints)
- `build/` — build outputs such as manifests, epoch pinning, provenance snapshots

The exact contents will evolve, but the invariant is:

> **Specs and tools in → canonical IR + artifacts + receipts out**

## Quickstart

### Build

From the repo root:

```bash
# Strongly recommended: pin the epoch for determinism
export STUNIR_BUILD_EPOCH=1730000000
export STUNIR_REQUIRE_DETERMINISTIC_EPOCH=1

# Run the build
bash scripts/build.sh
```

### Verify (recommended workflow)

A typical verification loop is:

1. Recompute provenance hashes for `spec/` and `asm/`
2. Recompute canonical IR normalization
3. Compare against `receipts/*.json`

If you don’t have a dedicated `scripts/verify.sh` yet, add one that recomputes and checks:

- `build/epoch.json` / pinned epoch
- `build/provenance.json` digests
- IR normalization outputs
- receipt file integrity

(Keeping verification logic in a *small checker* is the point.)

## Determinism controls

STUNIR’s determinism comes from:

- **Pinned build epoch** (no “current time” leaks)
- **Canonical JSON normalization** (stable byte representation)
- **Stable directory hashing** for provenance (inputs committed by digest)

Common environment toggles you may see/use:

- `STUNIR_BUILD_EPOCH` — the epoch value used for receipts/build metadata
- `STUNIR_REQUIRE_DETERMINISTIC_EPOCH=1` — fail the build if the epoch is not explicitly pinned
- `STUNIR_REQUIRE_C_TOOLCHAIN=1` — optionally require an external toolchain (when relevant)

If you are aiming for *strong* reproducibility, treat these as mandatory in CI.

## Receipts: what you should expect to find

Receipts are JSON sidecars intended to be easy to parse and easy to verify.
While exact fields may differ, good receipts generally include:

- input digests (e.g., spec directory hash)
- output digests (IR/artifact hashes)
- tool identifiers + versions
- arguments / configuration
- pinned epoch / build parameters

Think of a receipt as a **replay + integrity contract**.

## Extending STUNIR

STUNIR is most powerful when you plug in real backends:

- a real assembler / linker
- a verified lowering pass
- an interpreter or emulator to validate behavior

Even without those, you can still get value today by tightening:

- canonical IR definition
- bidirectional checks (spec ⇄ IR, IR ⇄ output) *up to canonical forms*
- receipt verification tooling

## Security / trust model (recommended)

- Treat any planner (including ML/LLMs) as **untrusted**.
- Trust only:
  - deterministic tools you can hash/pin
  - receipts you can independently re-check
  - small verifiers/checkers

If a step can’t “show its work” (produce a checkable receipt/certificate), it shouldn’t be allowed to affect the final output.

## Appendix: previous README (kept for reference)

The content below is preserved from the prior `README.md` found in the pack, in case there are notes you still want to keep.

# STUNIR Pack (alpha, reproducible timestamp ready)

This pack bakes in reproducible build timestamp handling. Build scripts prefer the
following environment variables when determining the canonical build epoch:

1. STUNIR_BUILD_EPOCH (seconds since Unix epoch)
2. SOURCE_DATE_EPOCH (seconds since Unix epoch; standard for reproducible builds)
3. GIT_COMMIT_EPOCH (last commit time if `git` is available in the repo that invokes the build)
4. Fallback to current UTC time (preserves baseline behavior if you do nothing)

Nothing in this pack requires Docker/Nix; those remain optional. When you do use them,
this same epoch preference applies inside the container or Nix shell.

Quick start
- Local: `scripts/build.sh`
- Override time: `STUNIR_BUILD_EPOCH=1730000000 scripts/build.sh`
- Standard reproducible: `export SOURCE_DATE_EPOCH=$(git log -1 --format=%ct || echo 0); scripts/build.sh`

Outputs
- receipts/ contain JSON receipts with `status`, `sha256`, and `build_epoch`.
- build/provenance.h and build/provenance.json are generated in a deterministic way from the epoch and inputs.
- If a C compiler is available, `bin/prov_emit` is built to print embedded provenance at runtime.



## Transparency & exceptions
- All receipts include an `epoch` block with `selected_epoch`, `source`, and raw inputs.
- Set `STUNIR_EPOCH_EXCEPTION_REASON` when a platform forces non-unified epoch behavior; this is recorded in the receipt.
- The selected epoch is propagated to provenance artifacts and compiled targets where feasible.


## IR generation
- The build generates a deterministic IR summary at asm/spec_ir.txt from the contents of spec/ using tools/spec_to_ir.py.
- A receipt (receipts/spec_ir.json) records its sha256 and the exact epoch manifest.

## Enforcing toolchains
- Set `STUNIR_REQUIRE_C_TOOLCHAIN=1` to make the build fail if the C compiler is unavailable (receipt status TOOLCHAIN_REQUIRED_MISSING).
- Otherwise, the build records SKIPPED_TOOLCHAIN and continues.

## Docker toolchain
- If you lack a native compiler, run: `scripts/build_docker.sh`
- This compiles bin/prov_emit inside the gcc:13 container and records receipts on the host.
