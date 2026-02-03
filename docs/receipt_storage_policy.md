# Receipt storage policy (STUNIR)

STUNIR emits receipts as **build artifacts**.

By default, receipts are **not committed**; they are generated, verified, and then treated as per-run outputs (e.g., attached to an artifact/release, or retained locally).

If a user requests a preserved copy, the orchestrator should offer two commit modes:

- **Snapshot**: copy the current run’s `receipts/` (and typically `build/`) into `fixtures/receipts/<TAG>/` so humans can review/diff/audit what happened.
- **Golden fixture**: like snapshot, but additionally treated as a regression baseline that CI may compare against under a controlled environment.

This policy does **not** alter STUNIR’s internal “no versions” conventions. STUNIR outputs (artifacts/receipts) are versioned according to the **spec’s** own versioning rules.

Implementation helper:

- Use `scripts/snapshot_receipts.sh` to create snapshots.

## Verifying a snapshot later

A snapshot is only meaningful when verified against the **same repo state** (same commit) and outputs rebuilt with the **same epoch**.

A practical workflow:

```bash
# 1) Checkout the commit that produced the snapshot.

# 2) Restore the snapshot epoch
cp fixtures/receipts/<TAG>/build/epoch.json build/epoch.json

# 3) Rebuild outputs using that epoch (but don't auto-verify using current receipts/)
STUNIR_PRESERVE_EPOCH=1 STUNIR_VERIFY_AFTER_BUILD=0 ./scripts/build.sh

# 4) Verify the snapshot receipts against the rebuilt outputs
./scripts/verify.sh --root fixtures/receipts/<TAG>
```
