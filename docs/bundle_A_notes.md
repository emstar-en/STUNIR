# Bundle A: Go / Rust / Java compiled-target contracts

This bundle adds compiled-target contracts and test vectors for:

- Go (`go_toolchain`)
- Rust (`cargo`)
- Java (`javac`)

The contracts are designed to be safe for probing:

- no third-party dependencies
- no network access required
- outputs are concrete files whose sha256 can be compared across repeated runs

If you want a stronger Rust determinism story across machines/paths, the next step is adding path remapping (e.g., `--remap-path-prefix`) via a crate-local config approach.
