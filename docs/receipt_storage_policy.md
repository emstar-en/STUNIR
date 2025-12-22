# Receipt storage policy (STUNIR)

STUNIR emits receipts as **build artifacts**. By default, receipts are **not committed**; they are generated, verified, and then treated as per-run outputs (e.g., attached to an artifact/release, or retained locally).

If a user requests a preserved copy, the orchestrator should offer two commit modes:

- **Snapshot**: copy the current run’s `receipts/` (and typically `build/`) into `fixtures/receipts/<tag>/` so humans can review/diff/audit what happened.
- **Golden fixture**: like snapshot, but additionally treated as a regression baseline that CI may compare against under a controlled environment.

This policy does **not** alter STUNIR’s internal “no versions” conventions. STUNIR *outputs* (artifacts/receipts) are versioned according to the **spec’s** own versioning rules.

Implementation helper:
- Use `scripts/snapshot_receipts.sh` to create snapshots.
