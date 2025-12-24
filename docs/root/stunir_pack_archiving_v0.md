### STUNIR Pack v0 — Deterministic Archiving Guidance
This document defines recommendations for producing stable archive bytes for a STUNIR pack directory.

#### 1. General requirements
When producing an archive from a pack directory:
- Archive entries MUST be sorted lexicographically by path.
- Paths MUST be relative (no leading `/`).
- Paths MUST NOT contain `..` segments.
- File bytes MUST be identical to the directory representation.

#### 2. Zip guidance
Zip is widely supported but has multiple metadata footguns.

Recommendations:
- Set every entry’s timestamp to the pack `epoch` (or to 0 if unspecified).
- Normalize permissions (e.g., `0644` for files, `0755` for dirs) unless permissions are explicitly part of the artifact bytes.
- Exclude extra fields that capture host metadata.
- Ensure deterministic compression (or use `STORE` / no compression).

If you must use a command-line zipper, prefer options that strip extra attributes (tool-dependent).

#### 3. Tar guidance (often easier)
For deterministic tar:
- Use a fixed sort order.
- Set `mtime` to the pack `epoch` for every entry.
- Set `uid=0`, `gid=0`, `uname=""`, `gname=""`.
- Avoid PAX headers that leak timestamps.
- Prefer uncompressed tar for maximal determinism; if compressing, ensure compressor settings are fixed.

#### 4. Canonical archive recommendation
If you need a single normative “canonical archive format” for byte-identical distribution, pick ONE of:
- uncompressed tar with strict metadata normalization, or
- zip with STORE and strict metadata normalization.

Then define the canonical archive’s digest as an optional additional artifact in the root attestation (do not replace root-attestation-based verification).
