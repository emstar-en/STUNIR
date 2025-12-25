### Polyglot target implementation plan (STUNIR) — v1 (with non-Python environment files)

#### 0) Scope
This bundle adds **planning artifacts** plus **language-environment skeleton files** needed so the emitted artifacts for non-Python targets can be **built/run in principle** without any Python at runtime.

Targets covered:
- C variants: `c89`, `c99`, `c11`, `c17`, `c23` (alias `c` → `c11`)
- `rust` (edition 2021)
- `go`
- `haskell`
- `erlang`, `elixir`
- `prolog`, `datalog`, `asp`, `minizinc`

Non-goals:
- Implementing full IR→language semantics.
- Adding required toolchain contracts (compilers/solvers) for these targets.
- Producing compiled artifacts or solver outputs.

Design rule (v0 semantics): all targets here are **source emission** targets (no required contracts).

#### 1) Artifact layout (emitted outputs)
Each target emits deterministic source artifacts under `asm/…` and binds them with:
- per-target manifest: `receipts/<target>_manifest.json`
- per-target receipt: `receipts/<target>.json`

Minimum file sets (v0):

##### 1.1 C variants
Root: `asm/c/<variant>/` where `<variant>` ∈ {`c89`,`c99`,`c11`,`c17`,`c23`}
- `stunir_runtime.h`
- `stunir_runtime.c`
- `main.c`
- `README.md`
- `Makefile` (optional but recommended)

##### 1.2 Rust (2021)
Root: `asm/rust/edition2021/`
- `Cargo.toml`
- `src/runtime.rs`
- `src/main.rs`
- `README.md`

##### 1.3 Go
Root: `asm/go/module/`
- `go.mod`
- `runtime.go`
- `main.go`
- `README.md`

##### 1.4 Haskell
Root: `asm/haskell/cabal/`
- `stunir.cabal`
- `src/STUNIR/Runtime.hs`
- `app/Main.hs`
- `README.md`

##### 1.5 Erlang (rebar3)
Root: `asm/erlang/rebar3/`
- `rebar.config`
- `src/stunir_program.app.src`
- `src/stunir_runtime.erl`
- `src/stunir_program.erl`
- `README.md`

##### 1.6 Elixir (mix)
Root: `asm/elixir/mix/`
- `mix.exs`
- `lib/stunir_runtime.ex`
- `lib/stunir_program.ex`
- `README.md`

##### 1.7 Prolog
Root: `asm/prolog/iso/`
- `runtime.pl`
- `program.pl`
- `README.md`

##### 1.8 Datalog (Soufflé-flavored baseline)
Root: `asm/datalog/souffle/`
- `program.dl`
- `README.md`

##### 1.9 ASP (clingo-flavored baseline)
Root: `asm/asp/clingo/`
- `program.lp`
- `README.md`

##### 1.10 MiniZinc
Root: `asm/minizinc/model/`
- `program.mzn`
- `README.md`

#### 2) Skeletons included in this bundle
This bundle includes **static skeleton templates** under:
- `spec/targets/polyglot/skeletons/...`

These are:
- minimal, self-contained project/config files
- designed to compile/run with the target ecosystem’s usual tooling
- intentionally contain **stub program bodies**, so codegen can later replace only the relevant sections deterministically.

Skeletons are not emitted outputs; they are **reference templates** that the eventual codegen tools can follow verbatim.

#### 3) Contracts policy (v0)
All targets in this plan are defined with:
- `required_contracts`: `[]`
- `optional_contracts`: `[]`

Future work (separate bundles): add contracts and define toolchain-required variants.

#### 4) Files in this bundle
- planning:
  - `spec/targets/polyglot/polyglot_target_implementation_plan_v1.md`
  - `spec/targets/polyglot/polyglot_targets_matrix_v1.json`
  - `spec/targets/polyglot/patches/rfc6902_target_requirements_polyglot_add.json`
  - `spec/targets/polyglot/patches/rfc6902_machine_plan_add_polyglot.json`
- skeleton templates:
  - `spec/targets/polyglot/skeletons/**` (see directory tree)
