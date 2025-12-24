# build/

This directory is a **build output location** for intermediate commitments (e.g., `epoch.json`, `provenance.json`).

Default policy: build outputs are generated per run and are not committed.
If a preserved copy is desired for audit/review, snapshot along with receipts using `scripts/snapshot_receipts.sh`.
