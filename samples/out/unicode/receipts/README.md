# receipts/

This directory is a **build output location**.

- Default policy: receipts are **generated per run** by deterministic tooling and are **not committed**.
- Optional policy: if a user requests a preserved, reviewable copy, snapshot the receipts into
  `fixtures/receipts/<tag>/` and commit that snapshot.

See:
- `docs/receipt_storage_policy.md`
- `scripts/snapshot_receipts.sh`
