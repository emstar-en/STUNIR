# Bundle C: Solver/5GL targets + stdout/stderr output artifacts

This bundle does two things:

1) Extends `tools/probe_dependency.py` so dependency probes can treat **stdout/stderr as output artifacts**.

- In any contract test, set `outputs` to include `@stdout` and/or `@stderr`.
- These pseudo-outputs are hashed exactly like file outputs (sha256 of the stream bytes).

Also adds invariant types usable in contract tests:

- `stdout_nonempty`, `stdout_empty`
- `stderr_nonempty`, `stderr_empty`

2) Adds three solver-oriented contracts + test vectors:

- `z3_solver` (SMT)
- `clingo` (ASP)
- `minizinc` (constraint modeling)

Targets:

- `smt` → Z3
- `asp` → clingo
- `minizinc` → MiniZinc

Note: These contracts intentionally avoid downloads/installs and use tiny, no-deps test vectors.
