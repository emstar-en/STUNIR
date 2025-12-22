
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
Verification

* After building, run: `scripts/verify.sh`
* Strict mode defaults to on (`STUNIR_STRICT=1`) and will:
  - require a deterministic epoch (no CURRENT_TIME)
  - disable noisy platform fields in receipts by default (`STUNIR_INCLUDE_PLATFORM=0`)
  - run verification automatically (`STUNIR_VERIFY_AFTER_BUILD=1`)

Opt out for legacy behavior:

* `STUNIR_STRICT=0 scripts/build.sh`

