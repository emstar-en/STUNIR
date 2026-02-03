### Polyglot target implementation plan (STUNIR)

#### 0) Scope and non-goals
This bundle adds **planning artifacts** (not implementations) for the following new STUNIR output target families:

- **C variants**: `c89`, `c99`, `c11`, `c17`, `c23` (with alias `c` → `c11`)
- **Rust**: `rust` (edition 2021)
- **Go**: `go`
- **Haskell**: `haskell`
- **BEAM**: `erlang`, `elixir`
- **Logic / constraints**:
  - Prolog: `prolog` (portable/ISO-leaning)
  - Datalog: `datalog` (Soufflé-flavored baseline)
  - ASP: `asp` (clingo-flavored baseline)
  - MiniZinc: `minizinc`

Non-goals for this bundle:
- Implementing IR→language lowering semantics.
- Adding compiler/runtime dependency contracts (e.g., `gcc`, `rustc`, `go`, `ghc`, `swipl`, `souffle`, `clingo`, `minizinc`) as required gates.
- Producing compiled artifacts or solver outputs.

Design rule (v0): all targets here are **source emission** targets. They should be runnable/compilable in principle, but **the build must not require toolchains**.

#### 1) Target names (authoritative)
Target keys (for `STUNIR_OUTPUT_TARGETS`, comma-separated):

- C family:
  - `c89`, `c99`, `c11`, `c17`, `c23`
  - Alias: `c` → `c11`

- `rust` (Rust 2021 edition)
- `go`
- `haskell`
- `erlang`
- `elixir`
- `prolog`
- `datalog`
- `asp`
- `minizinc`

Rationale: these names are stable, short, and map directly to output roots under `asm/`.

#### 2) Artifact layout (exact paths)
Each target emits deterministic source artifacts under `asm/…` and binds them with:

- per-target manifest: `receipts/<target>_manifest.json`
- per-target receipt: `receipts/<target>.json`

Suggested minimum file sets (v0):

- `c89` / `c99` / `c11` / `c17` / `c23`:
  - `asm/c/<variant>/stunir_runtime.h`
  - `asm/c/<variant>/stunir_runtime.c`
  - `asm/c/<variant>/main.c`
  - `asm/c/<variant>/README.md`

- `rust`:
  - `asm/rust/edition2021/Cargo.toml`
  - `asm/rust/edition2021/src/runtime.rs`
  - `asm/rust/edition2021/src/main.rs`
  - `asm/rust/edition2021/README.md`

- `go`:
  - `asm/go/module/go.mod`
  - `asm/go/module/runtime.go`
  - `asm/go/module/main.go`
  - `asm/go/module/README.md`

- `haskell`:
  - `asm/haskell/cabal/stunir.cabal`
  - `asm/haskell/cabal/src/STUNIR/Runtime.hs`
  - `asm/haskell/cabal/app/Main.hs`
  - `asm/haskell/cabal/README.md`

- `erlang`:
  - `asm/erlang/rebar3/rebar.config`
  - `asm/erlang/rebar3/src/stunir_runtime.erl`
  - `asm/erlang/rebar3/src/stunir_program.erl`
  - `asm/erlang/rebar3/README.md`

- `elixir`:
  - `asm/elixir/mix/mix.exs`
  - `asm/elixir/mix/lib/stunir_runtime.ex`
  - `asm/elixir/mix/lib/stunir_program.ex`
  - `asm/elixir/mix/README.md`

- `prolog`:
  - `asm/prolog/iso/runtime.pl`
  - `asm/prolog/iso/program.pl`
  - `asm/prolog/iso/README.md`

- `datalog`:
  - `asm/datalog/souffle/program.dl`
  - `asm/datalog/souffle/README.md`

- `asp`:
  - `asm/asp/clingo/program.lp`
  - `asm/asp/clingo/README.md`

- `minizinc`:
  - `asm/minizinc/model/program.mzn`
  - `asm/minizinc/model/README.md`

#### 3) Determinism policy
- The IR bundle remains the byte-exact anchor.
- These targets are required to be **byte-deterministic source emission** given the same canonical IR + epoch/provenance inputs.
- Runtime/compiled/solver outputs are explicitly out of scope for v0.

#### 4) Contracts policy (v0)
All targets in this plan are defined with:

- `required_contracts`: `[]`
- `optional_contracts`: `[]`

Future work (separate bundles): add contracts and define variant targets that require them (e.g., `prolog_swi`, `datalog_souffle`, `asp_clingo`, `minizinc_gecode`, `c_clang`, `rust_rustc`).

#### 5) Files in this bundle
- `spec/targets/polyglot/polyglot_target_implementation_plan_v0.md`
- `spec/targets/polyglot/polyglot_targets_matrix.json`
- `spec/targets/polyglot/patches/rfc6902_target_requirements_polyglot_add.json`
- `spec/targets/polyglot/patches/rfc6902_machine_plan_add_polyglot.json`
