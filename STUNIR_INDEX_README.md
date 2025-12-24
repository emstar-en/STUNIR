# STUNIR Index Bundle

This bundle provides deterministic, model-agnostic indexing of the STUNIR repository to help humans and AI agents navigate without looping.

## Files

- `STUNIR_REPO_INDEX.json`: per-file metadata for the whole repo (including `test_vectors/`).
- `STUNIR_CHUNK_INDEX.jsonl`: chunked text for markdown docs (excludes `test_vectors/`, `build/`, `receipts/` by default to avoid noise).

## Design notes

- `test_vectors/` are included in the repo index so tooling can locate them deterministically, but they are not chunked for semantic search by default.
- Git blob SHAs in the index come from the repository tree (sha1). If you need sha256 for files, compute it locally during checkout/materialization.

## Typical uses

- Build a vector index externally using `STUNIR_CHUNK_INDEX.jsonl` (stable chunk IDs + sha256).
- Validate documentation cross-references by using `outbound_repo_links` fields in `STUNIR_REPO_INDEX.json`.
- Quickly answer: “where is X defined?” by scanning `category` + `path`.

Generated against branch: `main`.
Generated UTC: `2025-12-24T04:38:03Z`.
