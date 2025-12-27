# Target language scaffolding
This directory collects **per-output-language planning artifacts** used to implement STUNIR output targets.

- Narrative implementation plans live under `docs/targets/`.
- Machine-readable target matrices live under `spec/targets/`.
- RFC6902 patch proposals live under `spec/patches/`.

Cross-cutting emission interface:
- `docs/code_emission.md` (IR → code emission contract)

These files are intended to be kept in-repo as part of the implementation record.
They are not build outputs.
