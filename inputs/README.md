# STUNIR Inputs Directory

Place your raw source code files here (e.g., `.py`, `.rs`, `.js`, `.c`).

During the build process, STUNIR will:
1. Scan this directory.
2. Wrap each file into a deterministic JSON Spec (`stunir.blob`).
3. Save the specs to `spec/imported/`.

This allows you to use existing code as an input to the STUNIR pipeline without manually writing JSON specs.
