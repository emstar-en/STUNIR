# STUNIR Dead-End Handling Contract (DEHC) v1

STUNIR is determinism-first: models can propose, but deterministic tooling must be the sole producer of commitments.

This document defines a general rule for "dead ends" encountered during deterministic builds, canonicalization, packing, verification, or toolchain probing.

## Core rule
**If there exists a deterministic, attestable way to proceed, STUNIR should allow it.**

Corollaries:
- **No silent meaning changes.** If proceeding requires changing semantics, it must be explicit and attestable.
- **No silent policy weakening.** If proceeding requires relaxing a policy, that relaxation must be explicit and attestable.
- **Prefer deterministic tool-driven selection over model-driven search.**

## Definitions
- **Dead end:** A build step cannot complete under the requested policy/constraints (e.g., tool missing; policy forbids an input feature; representation edge).
- **Resolution path:** A well-defined alternative that permits progress.
- **Attestable:** The choice of resolution is recorded in receipts/manifests so that a verifier can confirm what happened.
- **Semantic class:** Whether the resolution preserves meaning or changes it.

## Resolution ladder (preferred order)
When a dead end occurs, STUNIR should choose (or require the orchestrator to choose) a resolution from this ladder.

### R1 — Same meaning, different representation (preferred)
Proceed by selecting an alternative representation that preserves the intended meaning.

Examples:
- Encoding policy change that preserves the same JSON object model (e.g., `dcbor_shortest` → `float64_fixed`).
- Switching a canonicalization format that is proven equivalent (e.g., two encodings of the same root attestation).

### R2 — Deterministic auto-selection (tool-driven)
Proceed by selecting from a set of allowed representations using a deterministic rule derived solely from:
- the inputs (bytes)
- the declared build policy
- the tool version / tool digest

The decision rule itself must be stable and verifiable.

### R3 — Explicit semantic downgrade (allowed, but must be labeled)
Proceed by switching to a less-structured meaning.

Example:
- "Treat this file as opaque bytes" instead of "parse JSON and canonicalize".

This is **allowed only** when it is:
- explicitly requested (flag/policy)
- clearly labeled in the manifest/receipt
- bound by digest so verifiers can reproduce it

### R4 — Refuse (hard fail)
If no deterministic, attestable resolution exists within policy, the step must fail.

## What must be attested
Whenever a resolution is used (R1–R3), emit a **Decision Record** (machine-readable) and bind it in receipts.

A Decision Record MUST capture:
- **schema/version** of the Decision Record
- **step** (what operation was being performed)
- **requested** mode/policy
- **selected** mode/policy
- **resolution_class** (`R1 | R2 | R3`)
- **reason_code** (stable enum)
- **inputs_digest** (what the decision was based on)
- **tool_identity** (path + sha256 digest, and optionally version output)
- **constraints** (e.g., verification profile, determinism toggles)

A Decision Record SHOULD capture:
- **rejection evidence** for the losing option(s) (e.g., error code/message hash)
- **deterministic decision rule id** (for R2)

## Stable reason codes
Reason codes should be stable, short, and machine-checkable.

Suggested core codes:
- `POLICY_FORBIDS_FEATURE`
- `ENCODING_RANGE_LIMIT`
- `TOOLCHAIN_MISSING`
- `TOOLCHAIN_NONDETERMINISTIC`
- `PROFILE_CONSTRAINT`
- `INPUT_PARSE_FAILED`
- `EQUIVALENCE_UNAVAILABLE`

## Verifier expectations
A verifier SHOULD be able to:
- recompute digests of inputs used in the decision
- check that the selected mode/policy matches the emitted artifacts
- confirm that the decision is consistent with the declared build policy

If a verifier cannot validate the equivalence claim (e.g., R1 equivalence not proven in that profile), it must treat the decision as:
- either **not acceptable** under that verification profile, or
- acceptable only if an explicit policy allows it

## Guidance for tool authors
### Strict-by-default with explicit escape hatches
Tool behavior should follow this pattern:

1) **Reject on policy violations** (default).
2) **Provide explicit, attestable alternatives** (flags/policies) for R1–R3.
3) **Avoid catch-all fallbacks** that turn errors into different semantics.

### Example: IR emission for JSON specs
Recommended behavior for a JSON→IR normalizer:
- If JSON decoding fails: allow explicit "opaque bytes" mode (R3) OR fail.
- If JSON decoding succeeds but canonicalization/encoding fails: fail by default, or allow a deterministic R1/R2 alternative.

### Example: float policy in deterministic CBOR
- `forbid_floats`: hard fail if floats are present.
- `dcbor_shortest`: if numeric reduction would require an integer outside CBOR int/uint range, do **not** fail; encode as float (R1) and attest the reason (`ENCODING_RANGE_LIMIT`) if you want this surfaced.
- `float64_fixed`: deterministic baseline.

## Orchestrator guidance (models as planners)
Orchestrators may attempt multiple plans, but STUNIR prefers that fallback behavior be:
- tool-defined (deterministic), or
- explicitly user-policy-defined

If orchestrators do perform retries, they SHOULD:
- record failure receipts (optional but recommended)
- ensure the final successful build is fully receipt-bound

## Relationship to packs
When decisions affect pack contents (e.g., which root attestation encoding is included), the Decision Record MUST be included in the pack inventory so that offline verifiers can explain and validate the path taken.

## Compatibility and evolution
- This contract is additive: new reason codes and optional fields may be introduced.
- Breaking changes require a schema version bump.
