# Bundle B: Hosted runtime breadth (Lua/Ruby/PHP/R/Julia)

This bundle adds hosted-runtime contracts + deterministic test vectors for:

- Lua (`lua_runtime`)
- Ruby (`ruby_runtime`)
- PHP (`php_runtime`)
- R (`r_runtime` via `Rscript`)
- Julia (`julia_runtime`)

All probes are designed to be safe:

- no network access
- no third-party dependencies
- deterministic output artifacts (file digests compared across repeated runs)

If you later want package-manager coverage (e.g., `bundler`, `composer`, `renv`, `Pkg`), add them as separate optional contracts so policy can require them per target.
