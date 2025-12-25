### Tier B (declarative) frontends plan (STUNIR)

#### 0) Goal
Add *real* (semantic) source→IR frontends **first** for declarative languages:
- Prolog
- Datalog
- ASP
- MiniZinc

This is Tier B: parsing + normalization + semantics-lift into canonical STUNIR IR.

#### 1) Why declarative first
- Clearer mapping to canonical meaning objects (rules/relations/constraints).
- Less entanglement with platform UB, headers, macro systems.
- Easier to define deterministic normalization (ordering, alpha-renaming, etc.).

#### 2) Compatibility policy
Each frontend MUST define:
- accepted dialect/version baseline
- rejected constructs (explicit)
- deterministic normalization rules

Frontends should begin as **strict subsets** with clear error messages.

#### 3) Output: canonical IR bundle
Tier B frontends MUST produce the **byte-exact canonical IR bundle**.
This implies:
- a fully-specified canonical serialization for produced IR
- no dependence on runtime timestamps, filesystem paths, locale, etc.

#### 4) Proposed phased delivery

##### Phase B1: Surface syntax + canonical normalized AST (still meaningful)
- Parse input into an AST.
- Normalize deterministically:
  - stable ordering of declarations
  - alpha-renaming of variables
  - normalized literals
- Emit IR that faithfully captures normalized AST.

This enables:
- reproducible ingest
- subsequent semantics passes without re-parsing

##### Phase B2: Semantics-lift into STUNIR “meaning” IR
Define and implement the mapping into STUNIR meaning forms:
- Prolog: Horn clauses + resolution model subset
- Datalog: stratified negation subset, fixed-point semantics
- ASP: stable model semantics subset
- MiniZinc: constraint model subset

#### 5) Receipts
Each Tier B frontend run must produce a receipt binding:
- input program bytes digest
- frontend tool identity + version + dialect config
- produced `receipts/ir_bundle_manifest.json` digest
- normalization parameters

Receipts should explicitly record:
- language (`prolog|datalog|asp|minizinc`)
- dialect baseline
- subset gates enforced

#### 6) Coexistence with Tier A
Tier A remains valid for these languages and should be the default path when:
- the source file is just a transport for an already-chosen IR bundle

Tier B is used when:
- the source program itself is the authoritative meaning input

#### 7) Non-goals
- Full-language support.
- Solver/compiler execution as part of determinism.
- Cross-language equivalence proofs (until semantics are nailed down).
