#!/usr/bin/env python3
"""STUNIR verifier.

This tool supports two verification modes:

1) Local mode ("small checker" for scripts/build.sh outputs)
   - verifies build/ + receipts/ + asm/ (+ bin/ when present)
   - checks deterministic JSON serialization (canonical bytes)
   - checks receipt core-id and sha256 bindings
   - can verify snapshot fixtures created by scripts/snapshot_receipts.sh

2) DSSE mode
   - verifies DSSE v1 signature + canonical payload bytes
   - verifies input-closure manifest
   - rebuilds IR + outputs per receipt-declared commands and checks digests

Design goals:
- Deterministic: compares bytes/digests, not semantics.
- Transparent: explicit policy fields, strict checks, clear failures.
- Small: avoids complex frameworks.

DSSE payload canonicalization is "stunir-json-c14n-v1" as defined by canonical_json_bytes().
Local JSON canonicalization is the same encoder, accepting an optional trailing newline.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HEX_RE = re.compile(r"^[0-9a-f]+$")

ALG_ALIASES = {
    "sha-256": "sha256",
    "sha_256": "sha256",
    "sha256": "sha256",
    "sha-512": "sha512",
    "sha_512": "sha512",
    "sha512": "sha512",
    "blake2b-256": "blake2b-256",
    "blake2b256": "blake2b-256",
}

REQUIRED_DIGEST_ALGS_DEFAULT = ["sha256"]


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)


def read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except Exception as e:
        die(f"cannot read {path}: {e}")


def parse_json_bytes(b: bytes, where: str) -> Any:
    try:
        return json.loads(b.decode("utf-8"))
    except Exception as e:
        die(f"invalid JSON in {where}: {e}")


# ---------------- Canonical JSON: stunir-json-c14n-v1 ----------------
# Intentionally strict and small:
# - UTF-8
# - no floats anywhere (integers only)
# - object keys sorted lexicographically by Unicode codepoint
# - no whitespace
# - JSON string escaping as produced by Python json.dumps(..., ensure_ascii=False)
# NOTE: This is not RFC8785; it's fully specified by this function.

def _check_no_floats(x: Any, where: str = "$") -> None:
    if isinstance(x, float):
        die(f"floats forbidden for canonicalization (at {where})")

    if isinstance(x, dict):
        for k, v in x.items():
            if not isinstance(k, str):
                die(f"non-string key forbidden (at {where})")
            _check_no_floats(v, where + "." + k)
        return

    if isinstance(x, list):
        for i, v in enumerate(x):
            _check_no_floats(v, where + f"[{i}]")
        return

    if isinstance(x, (str, int, bool)) or x is None:
        return

    die(f"unsupported JSON type {type(x)} at {where}")


def canonical_json_bytes(obj: Any) -> bytes:
    _check_no_floats(obj)
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return s.encode("utf-8")


def file_is_canonical_json(path: Path) -> None:
    b = read_bytes(path)
    obj = parse_json_bytes(b, where=str(path))
    c = canonical_json_bytes(obj)
    if b == c:
        return
    if b == c + b"\n":
        return
    die(f"{path} is not canonical JSON (stunir-json-c14n-v1, optional trailing newline)")


# ---------------- DSSE v1 ----------------

def pae(payload_type: str, payload: bytes) -> bytes:
    pt = payload_type.encode("utf-8")
    return (
        b"DSSEv1 "
        + str(len(pt)).encode()
        + b" "
        + pt
        + b" "
        + str(len(payload)).encode()
        + b" "
        + payload
    )


@dataclass
class TrustedKey:
    keyid: str
    public_key_pem: bytes


def verify_signature_cryptography(public_key_pem: bytes, sig_b64: str, msg: bytes) -> None:
    # Optional dependency; preferred if available.
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa

    sig = base64.b64decode(sig_b64)
    pub = serialization.load_pem_public_key(public_key_pem)

    if isinstance(pub, ed25519.Ed25519PublicKey):
        pub.verify(sig, msg)
        return

    if isinstance(pub, ec.EllipticCurvePublicKey):
        pub.verify(sig, msg, ec.ECDSA(hashes.SHA256()))
        return

    if isinstance(pub, rsa.RSAPublicKey):
        pub.verify(
            sig,
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return

    raise ValueError(f"unsupported public key type: {type(pub)}")


def verify_signature_openssl(public_key_pem_path: Path, sig_b64: str, msg: bytes, tmp_dir: Path) -> None:
    # Fallback via openssl if cryptography isn't available.
    msg_path = tmp_dir / "_dsse_msg.bin"
    sig_path = tmp_dir / "_dsse_sig.bin"
    msg_path.write_bytes(msg)
    sig_path.write_bytes(base64.b64decode(sig_b64))

    cmd = [
        "openssl",
        "pkeyutl",
        "-verify",
        "-pubin",
        "-inkey",
        str(public_key_pem_path),
        "-sigfile",
        str(sig_path),
        "-in",
        str(msg_path),
    ]

    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise ValueError(f"openssl verify failed: {(r.stderr.strip() or r.stdout.strip())}")


def load_trusted_keys(trust_args: List[str]) -> Dict[str, TrustedKey]:
    keys: Dict[str, TrustedKey] = {}
    for t in trust_args:
        if "=" not in t:
            die(f"--trust-key must be KEYID=PATH, got: {t}")
        keyid, p = t.split("=", 1)
        pem = Path(p).read_bytes()
        keys[keyid] = TrustedKey(keyid=keyid, public_key_pem=pem)
    return keys


def verify_dsse_envelope(
    envelope: Dict[str, Any],
    trusted_keys: Dict[str, TrustedKey],
    tmp_dir: Path,
    require_canonical_payload: bool = True,
) -> Tuple[bytes, Dict[str, Any]]:
    payload_type = envelope.get("payloadType")
    payload_b64 = envelope.get("payload")
    sigs = envelope.get("signatures")

    if not isinstance(payload_type, str) or not isinstance(payload_b64, str) or not isinstance(sigs, list) or not sigs:
        die("invalid DSSE envelope: require payloadType(str), payload(b64 str), signatures[](non-empty)")

    payload_bytes = base64.b64decode(payload_b64)
    msg = pae(payload_type, payload_bytes)

    verified = False
    last_err: Optional[Exception] = None

    for s in sigs:
        if not isinstance(s, dict):
            continue
        keyid = s.get("keyid")
        sig_b64 = s.get("sig")
        if not isinstance(keyid, str) or not isinstance(sig_b64, str):
            continue
        if keyid not in trusted_keys:
            continue

        pub_pem = trusted_keys[keyid].public_key_pem
        try:
            try:
                verify_signature_cryptography(pub_pem, sig_b64, msg)
            except ModuleNotFoundError:
                tmp_pub = tmp_dir / f"_pub_{keyid}.pem"
                tmp_pub.write_bytes(pub_pem)
                verify_signature_openssl(tmp_pub, sig_b64, msg, tmp_dir)
            verified = True
            break
        except Exception as e:
            last_err = e

    if not verified:
        die(f"no valid DSSE signature found for trusted keyids; last error: {last_err}")

    stmt = parse_json_bytes(payload_bytes, where="DSSE payload")
    if require_canonical_payload:
        c14n = canonical_json_bytes(stmt)
        if c14n != payload_bytes:
            die("payload JSON is not in canonical form (stunir-json-c14n-v1): bytes differ")

    if not isinstance(stmt, dict):
        die("DSSE payload must be a JSON object")

    return payload_bytes, stmt


# ---------------- Digests (DSSE mode) ----------------

def normalize_alg(name: str) -> str:
    n = name.strip().lower()
    n = re.sub(r"[^a-z0-9\-]", "", n)
    return ALG_ALIASES.get(n, n)


def digest_file(alg: str, path: Path) -> str:
    a = normalize_alg(alg)
    if a in ("sha256", "sha512"):
        h = hashlib.new(a)
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    if a == "blake2b-256":
        h = hashlib.blake2b(digest_size=32)
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    raise ValueError(f"unsupported digest algorithm: {alg}")


def verify_digest_map(path: Path, expected: Dict[str, str], required_algs: List[str]) -> None:
    if not isinstance(expected, dict) or not expected:
        die(f"missing/invalid digest map for {path}")

    expected_norm = {normalize_alg(k): v for k, v in expected.items()}

    for ra in required_algs:
        if ra not in expected_norm:
            die(f"required digest alg {ra} missing for {path}")

    for alg_norm, exp in expected_norm.items():
        if not isinstance(exp, str):
            die(f"digest for {path} alg {alg_norm} must be string")
        exp_l = exp.lower()
        if not HEX_RE.match(exp_l):
            die(f"digest for {path} alg {alg_norm} must be hex")

        got = digest_file(alg_norm, path)
        if got != exp_l:
            die(f"digest mismatch for {path} ({alg_norm}): expected {exp_l}, got {got}")


# ---------------- Paths / closure (DSSE mode) ----------------

def safe_relpath(p: str) -> Path:
    if not isinstance(p, str) or not p:
        die("path/uri must be non-empty string")
    pp = Path(p)
    if pp.is_absolute():
        die(f"absolute paths forbidden: {p}")

    norm = Path(os.path.normpath(p))
    # normalize to forward slashes for traversal check
    norm_s = str(norm).replace("\\", "/")
    if norm_s == ".." or norm_s.startswith("../") or "/../" in norm_s:
        die(f"path traversal forbidden: {p}")

    return norm


def find_manifest_material(predicate: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    spec_closure = predicate.get("specClosure") or {}
    manifest = (spec_closure.get("manifest") or {})
    uri = manifest.get("uri")
    if not isinstance(uri, str):
        die("predicate.specClosure.manifest.uri required")

    materials = predicate.get("materials")
    if not isinstance(materials, list) or not materials:
        die("predicate.materials[] required (must include the input manifest)")

    matches = []
    for m in materials:
        if not isinstance(m, dict):
            continue
        if m.get("uri") != uri:
            continue
        d = m.get("digest")
        if not isinstance(d, dict) or not d:
            die("material digest must be a non-empty object")
        matches.append((uri, d))

    if len(matches) != 1:
        die(f"expected exactly 1 materials entry matching manifest uri {uri!r}, found {len(matches)}")

    return matches[0]


def verify_input_manifest(repo: Path, manifest_uri: str, manifest_digest: Dict[str, str], required_algs: List[str]) -> None:
    manifest_path = repo / safe_relpath(manifest_uri)
    if not manifest_path.exists():
        die(f"missing input manifest: {manifest_uri}")

    verify_digest_map(manifest_path, manifest_digest, required_algs)

    manifest_obj = parse_json_bytes(read_bytes(manifest_path), where=f"manifest {manifest_uri}")
    files = manifest_obj.get("files")
    if not isinstance(files, list):
        die(f"manifest {manifest_uri} must contain files[]")

    for ent in files:
        if not isinstance(ent, dict):
            die(f"manifest entry must be object: {ent}")
        p = ent.get("path")
        dig = ent.get("digest")
        if not isinstance(p, str) or not isinstance(dig, dict):
            die(f"manifest entry must contain path(str) and digest(object): {ent}")

        rel = safe_relpath(p)
        full = repo / rel
        if not full.exists():
            die(f"missing input file: {p}")
        if full.is_symlink():
            die(f"symlinks forbidden (input file is symlink): {p}")

        verify_digest_map(full, dig, required_algs)


# ---------------- Rebuild + compare (DSSE mode) ----------------

def run_cmd(cmd: Any, cwd: Path, env: Dict[str, str]) -> None:
    if not isinstance(cmd, list) or not cmd or not all(isinstance(x, str) and x for x in cmd):
        die(f"rebuild command must be a non-empty string list, got: {cmd}")

    r = subprocess.run(cmd, cwd=str(cwd), env=env)
    if r.returncode != 0:
        die(f"command failed ({r.returncode}): {' '.join(cmd)}")


def setup_env(build_policy: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TZ", "UTC")
    env.setdefault("LC_ALL", "C")
    env.setdefault("LANG", "C")
    env.setdefault("PYTHONHASHSEED", "0")

    sde = build_policy.get("sourceDateEpoch")
    if sde is not None:
        if not isinstance(sde, int):
            die("buildPolicy.sourceDateEpoch must be integer")
        env["SOURCE_DATE_EPOCH"] = str(sde)

    return env


def verify_statement_shape(stmt: Dict[str, Any]) -> None:
    if stmt.get("_type") != "https://in-toto.io/Statement/v1":
        die("payload._type must be https://in-toto.io/Statement/v1")
    if stmt.get("predicateType") != "urn:stunir:receipt:v1":
        die("payload.predicateType must be urn:stunir:receipt:v1")
    if not isinstance(stmt.get("subject"), list) or not stmt["subject"]:
        die("payload.subject[] must be non-empty")
    if not isinstance(stmt.get("predicate"), dict):
        die("payload.predicate must be object")


def get_required_algs(arg: Optional[str]) -> List[str]:
    if not arg:
        return REQUIRED_DIGEST_ALGS_DEFAULT[:]

    out: List[str] = []
    for a in arg.split(","):
        a = normalize_alg(a)
        if a:
            out.append(a)

    if not out:
        die("--required-algs produced empty list")

    return out


# ---------------- Local-mode verification ----------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_dir_legacy(root: Path) -> str:
    """Match legacy helper behavior used by some tools: sorted(root.rglob('*'))"""
    if not root.exists():
        return "0" * 64
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode("utf-8"))
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
    return h.hexdigest()


def sha256_of_dir_posix(root: Path) -> str:
    """More explicit traversal ordering: sort by relative posix path."""
    if not root.exists():
        return "0" * 64
    h = hashlib.sha256()
    files: List[Path] = [p for p in root.rglob("*") if p.is_file()]
    files.sort(key=lambda p: p.relative_to(root).as_posix())
    for p in files:
        h.update(p.relative_to(root).as_posix().encode("utf-8"))
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def map_local_path(repo: Path, *, receipts_dir: Path, build_dir: Path, asm_dir: Path, bin_dir: Path, rel: str) -> Path:
    p = safe_relpath(rel)
    s = p.as_posix()

    for prefix, base in (
        ("receipts/", receipts_dir),
        ("build/", build_dir),
        ("asm/", asm_dir),
        ("ir/", asm_dir / "ir"),
        ("bin/", bin_dir),
    ):
        if s == prefix[:-1] or s.startswith(prefix):
            rest = s[len(prefix) :]
            return (base / rest).resolve()

    return (repo / p).resolve()


def verify_tool_identity(tool_obj: Dict[str, Any], strict: bool) -> None:
    if not isinstance(tool_obj, dict):
        die("receipt.tool must be object")
    p = tool_obj.get("path")
    exp = tool_obj.get("sha256")
    if p is None and exp is None:
        return
    if not isinstance(p, str):
        die("receipt.tool.path must be string when tool is present")

    if exp is None:
        if strict:
            die(f"receipt.tool.sha256 missing for tool path {p!r} (strict)")
        return

    if not isinstance(exp, str) or not HEX_RE.match(exp.lower()):
        die("receipt.tool.sha256 must be hex string")

    path = Path(p)
    if not path.exists() and "/" not in p and "\\" not in p:
        resolved = shutil.which(p)
        if resolved:
            path = Path(resolved)

    got = sha256_file(path)
    if got is None:
        die(f"tool not found or not a file for hashing: {p}")

    if got != exp.lower():
        die(f"tool sha256 mismatch for {p}: expected {exp.lower()}, got {got}")


def compute_receipt_core_id(receipt: Dict[str, Any]) -> str:
    core = {
        "schema": receipt.get("schema"),
        "target": receipt.get("target"),
        "status": receipt.get("status"),
        "build_epoch": receipt.get("build_epoch"),
        "sha256": receipt.get("sha256"),
        "epoch": receipt.get("epoch"),
        "inputs": receipt.get("inputs"),
        "tool": receipt.get("tool"),
        "argv": receipt.get("argv"),
    }
    return sha256_bytes(canonical_json_bytes(core))


def verify_build_receipt(
    receipt_path: Path,
    repo: Path,
    *,
    receipts_dir: Path,
    build_dir: Path,
    asm_dir: Path,
    bin_dir: Path,
    epoch_json: Dict[str, Any],
    strict: bool,
) -> Dict[str, Any]:
    if not receipt_path.exists():
        die(f"missing receipt: {receipt_path}")

    file_is_canonical_json(receipt_path)
    receipt = parse_json_bytes(read_bytes(receipt_path), where=str(receipt_path))
    if not isinstance(receipt, dict):
        die(f"receipt {receipt_path} must be an object")

    if receipt.get("schema") != "stunir.receipt.build.v1":
        die(f"unexpected receipt schema in {receipt_path}: {receipt.get('schema')!r}")

    # Epoch consistency
    try:
        sel = int(epoch_json.get("selected_epoch"))
    except Exception:
        sel = None

    if sel is not None:
        if receipt.get("build_epoch") != sel:
            die(f"{receipt_path}: build_epoch mismatch vs build/epoch.json")
        ep = receipt.get("epoch")
        if not isinstance(ep, dict):
            die(f"{receipt_path}: epoch must be object")
        if ep.get("selected_epoch") != sel:
            die(f"{receipt_path}: epoch.selected_epoch mismatch vs build/epoch.json")
        if ep.get("source") != epoch_json.get("source"):
            die(f"{receipt_path}: epoch.source mismatch vs build/epoch.json")

    # Receipt core id
    exp_core = receipt.get("receipt_core_id_sha256")
    if not isinstance(exp_core, str) or not HEX_RE.match(exp_core.lower()):
        die(f"{receipt_path}: missing/invalid receipt_core_id_sha256")
    got_core = compute_receipt_core_id(receipt)
    if got_core != exp_core.lower():
        die(f"{receipt_path}: receipt_core_id_sha256 mismatch: expected {exp_core.lower()}, got {got_core}")

    # Tool identity (optional)
    if receipt.get("tool") is not None:
        verify_tool_identity(receipt["tool"], strict=strict)

    # Verify inputs closure (files + dirs)
    inputs = receipt.get("inputs")
    if not isinstance(inputs, list):
        die(f"{receipt_path}: inputs must be list")

    for ent in inputs:
        if not isinstance(ent, dict):
            die(f"{receipt_path}: input entry must be object")
        p = ent.get("path")
        kind = ent.get("kind")
        exp = ent.get("sha256")
        if not isinstance(p, str) or kind not in ("file", "dir"):
            die(f"{receipt_path}: bad input entry: {ent}")

        full = map_local_path(repo, receipts_dir=receipts_dir, build_dir=build_dir, asm_dir=asm_dir, bin_dir=bin_dir, rel=p)

        if kind == "file":
            got = sha256_file(full)
            if exp is None:
                if strict and got is not None:
                    die(f"{receipt_path}: input file {p} exists but receipt sha256 is null")
                continue
            if not isinstance(exp, str) or not HEX_RE.match(exp.lower()):
                die(f"{receipt_path}: input file sha256 must be hex: {p}")
            if got is None:
                die(f"{receipt_path}: missing input file: {p} (resolved {full})")
            if got != exp.lower():
                die(f"{receipt_path}: input file digest mismatch for {p}: expected {exp.lower()}, got {got}")

        else:  # dir
            if exp is None:
                if strict and full.exists():
                    die(f"{receipt_path}: input dir {p} exists but receipt sha256 is null")
                continue
            if not isinstance(exp, str) or not HEX_RE.match(exp.lower()):
                die(f"{receipt_path}: input dir sha256 must be hex: {p}")

            # Accept either traversal ordering (legacy or explicit)
            got1 = sha256_of_dir_legacy(full)
            got2 = sha256_of_dir_posix(full)
            if exp.lower() not in (got1, got2):
                die(
                    f"{receipt_path}: input dir digest mismatch for {p}: expected {exp.lower()}, got {got1} (legacy) / {got2} (posix)"
                )

    # Verify target sha256 binding if present
    tgt = receipt.get("target")
    if not isinstance(tgt, str):
        die(f"{receipt_path}: target must be string")

    target_full = map_local_path(repo, receipts_dir=receipts_dir, build_dir=build_dir, asm_dir=asm_dir, bin_dir=bin_dir, rel=tgt)
    exp_tgt_sha = receipt.get("sha256")
    got_tgt_sha = sha256_file(target_full)

    if exp_tgt_sha is None:
        # OK when the build skipped target creation.
        return receipt

    if not isinstance(exp_tgt_sha, str) or not HEX_RE.match(exp_tgt_sha.lower()):
        die(f"{receipt_path}: target sha256 must be hex or null")

    if got_tgt_sha is None:
        die(f"{receipt_path}: target missing: {tgt} (resolved {target_full})")

    if got_tgt_sha != exp_tgt_sha.lower():
        die(f"{receipt_path}: target sha256 mismatch for {tgt}: expected {exp_tgt_sha.lower()}, got {got_tgt_sha}")

    return receipt


def verify_ir_manifest(manifest_path: Path, repo: Path, *, asm_dir: Path, strict: bool) -> None:
    if not manifest_path.exists():
        die(f"missing ir manifest: {manifest_path}")

    file_is_canonical_json(manifest_path)
    man = parse_json_bytes(read_bytes(manifest_path), where=str(manifest_path))
    if not isinstance(man, dict):
        die("ir manifest must be object")
    if man.get("schema") != "stunir.ir_manifest.v2":
        die(f"unexpected ir manifest schema: {man.get('schema')!r}")

    files = man.get("files")
    if not isinstance(files, list):
        die("ir manifest files must be list")

    seen: set[str] = set()
    for rec in files:
        if not isinstance(rec, dict):
            die("ir manifest entry must be object")
        rel = rec.get("file")
        sha = rec.get("sha256")
        if not isinstance(rel, str) or not isinstance(sha, str):
            die("ir manifest entry must contain file(str) and sha256(str)")
        if rel in seen:
            die(f"duplicate ir manifest entry: {rel}")
        seen.add(rel)

        p = (asm_dir / "ir" / safe_relpath(rel)).resolve()
        if not p.exists():
            die(f"missing IR file listed in manifest: {rel} (resolved {p})")
        got = sha256_file(p)
        if got != sha.lower():
            die(f"IR file sha256 mismatch for {rel}: expected {sha.lower()}, got {got}")

    if strict:
        actual = [p for p in (asm_dir / "ir").rglob("*.dcbor") if p.is_file()]
        actual_set = {p.relative_to(asm_dir / "ir").as_posix() for p in actual}
        if actual_set != seen:
            missing = sorted(actual_set - seen)
            extra = sorted(seen - actual_set)
            die(f"ir manifest set mismatch (strict): missing={missing} extra={extra}")


def verify_ir_bundle_manifest(bundle_manifest_path: Path, repo: Path, *, asm_dir: Path) -> None:
    if not bundle_manifest_path.exists():
        die(f"missing ir bundle manifest: {bundle_manifest_path}")

    file_is_canonical_json(bundle_manifest_path)
    bm = parse_json_bytes(read_bytes(bundle_manifest_path), where=str(bundle_manifest_path))
    if not isinstance(bm, dict):
        die("ir bundle manifest must be object")
    if bm.get("schema") != "stunir.ir_bundle_manifest.v1":
        die(f"unexpected ir bundle manifest schema: {bm.get('schema')!r}")

    bundle_rel = bm.get("bundle")
    if not isinstance(bundle_rel, str):
        die("bundle manifest missing bundle path")

    bundle_path = map_local_path(repo, receipts_dir=repo / "receipts", build_dir=repo / "build", asm_dir=asm_dir, bin_dir=repo / "bin", rel=bundle_rel)
    data = read_bytes(bundle_path)
    got_bundle_sha = sha256_bytes(data)
    exp_bundle_sha = bm.get("bundle_sha256")
    if not isinstance(exp_bundle_sha, str) or got_bundle_sha != exp_bundle_sha.lower():
        die(f"bundle sha256 mismatch: expected {str(exp_bundle_sha).lower()}, got {got_bundle_sha}")

    entries = bm.get("entries")
    if not isinstance(entries, list):
        die("bundle manifest entries must be list")

    for ent in entries:
        if not isinstance(ent, dict):
            die("bundle entry must be object")
        rel = ent.get("file")
        # Entries are expected to be relative to asm/ir; tolerate accidental 'ir/' prefix
        if rel.startswith('ir/'):
            rel = rel[len('ir/'):]
        off = ent.get("offset")
        ln = ent.get("length")
        sha = ent.get("sha256")
        if not isinstance(rel, str) or not isinstance(off, int) or not isinstance(ln, int) or not isinstance(sha, str):
            die("bundle entry missing required fields")

        seg = data[off : off + ln]
        seg_sha = sha256_bytes(seg)
        if seg_sha != sha.lower():
            die(f"bundle entry sha mismatch for {rel}: expected {sha.lower()}, got {seg_sha}")

        # Optional consistency: segment should match emitted per-file bytes
        file_path = (asm_dir / 'ir' / safe_relpath(rel))
        if file_path.exists():
            fb = read_bytes(file_path)
            if fb != seg:
                die(f"bundle bytes differ from file bytes for {rel}")


def verify_provenance(build_dir: Path, repo: Path, *, spec_dir: Path, asm_dir: Path, tmp_dir: Path, strict: bool) -> None:
    prov_json = build_dir / "provenance.json"
    prov_h = build_dir / "provenance.h"

    if not prov_json.exists():
        die(f"missing provenance: {prov_json}")

    file_is_canonical_json(prov_json)

    # Recompute by re-running the provenance tool (keeps the verifier small and matches exact header format).
    # This is a pragmatic check, not a formal trust boundary.
    gen = repo / "tools" / "gen_provenance.py"
    if not gen.exists():
        die(f"missing tool: {gen}")

    prov_obj = parse_json_bytes(read_bytes(prov_json), where=str(prov_json))
    if not isinstance(prov_obj, dict):
        die("provenance.json must be object")

    build_epoch = prov_obj.get("build_epoch")
    epoch_source = prov_obj.get("epoch_source", "UNKNOWN")
    if not isinstance(build_epoch, int):
        die("provenance.json build_epoch must be int")

    out_json = tmp_dir / "_prov_rebuild.json"
    out_h = tmp_dir / "_prov_rebuild.h"

    cmd = [
        sys.executable,
        str(gen),
        "--epoch",
        str(build_epoch),
        "--epoch-source",
        str(epoch_source),
        "--spec-root",
        spec_dir.relative_to(repo).as_posix(),
        "--asm-root",
        asm_dir.relative_to(repo).as_posix(),
        "--out-header",
        out_h.relative_to(repo).as_posix(),
        "--out-json",
        out_json.relative_to(repo).as_posix(),
    ]

    # Ensure tmp paths exist (relative_to(repo) requires they be under repo)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_h.parent.mkdir(parents=True, exist_ok=True)

    r = subprocess.run(cmd, cwd=str(repo))
    if r.returncode != 0:
        die("failed to rebuild provenance via tools/gen_provenance.py")

    # Compare provenance.json bytes
    if read_bytes(out_json) != read_bytes(prov_json):
        die("build/provenance.json is not reproducible")

    # provenance.h is expected but allow missing in non-strict mode
    if prov_h.exists():
        if read_bytes(out_h) != read_bytes(prov_h):
            die("build/provenance.h is not reproducible")
    else:
        if strict:
            die("missing build/provenance.h (strict)")


def verify_local(
    repo: Path,
    *,
    receipts_dir: Path,
    build_dir: Path,
    asm_dir: Path,
    bin_dir: Path,
    tmp_dir: Path,
    strict: bool,
) -> None:
    # Epoch manifest used by receipts
    epoch_path = build_dir / "epoch.json"
    if not epoch_path.exists():
        die(f"missing epoch manifest: {epoch_path}")

    file_is_canonical_json(epoch_path)
    epoch_json = parse_json_bytes(read_bytes(epoch_path), where=str(epoch_path))
    if not isinstance(epoch_json, dict):
        die("epoch.json must be object")

    # Provenance
    spec_dir = repo / "spec"
    verify_provenance(build_dir, repo, spec_dir=spec_dir, asm_dir=asm_dir, tmp_dir=tmp_dir, strict=strict)

    # Core receipts
    spec_ir_receipt = receipts_dir / "spec_ir.json"
    prov_emit_receipt = receipts_dir / "prov_emit.json"

    if strict:
        for p in [spec_ir_receipt, prov_emit_receipt]:
            if not p.exists():
                die(f"missing required receipt (strict): {p}")

    if spec_ir_receipt.exists():
        verify_build_receipt(
            spec_ir_receipt,
            repo,
            receipts_dir=receipts_dir,
            build_dir=build_dir,
            asm_dir=asm_dir,
            bin_dir=bin_dir,
            epoch_json=epoch_json,
            strict=strict,
        )

    if prov_emit_receipt.exists():
        verify_build_receipt(
            prov_emit_receipt,
            repo,
            receipts_dir=receipts_dir,
            build_dir=build_dir,
            asm_dir=asm_dir,
            bin_dir=bin_dir,
            epoch_json=epoch_json,
            strict=strict,
        )

    # IR manifests
    ir_manifest = receipts_dir / "ir_manifest.json"
    ir_bundle_manifest = receipts_dir / "ir_bundle_manifest.json"

    if strict:
        for p in [ir_manifest, ir_bundle_manifest]:
            if not p.exists():
                die(f"missing required manifest (strict): {p}")

    if ir_manifest.exists():
        verify_ir_manifest(ir_manifest, repo, asm_dir=asm_dir, strict=strict)

    if ir_bundle_manifest.exists():
        verify_ir_bundle_manifest(ir_bundle_manifest, repo, asm_dir=asm_dir)

    print(
        "OK: local verification passed (canonical JSON; epochs consistent; receipts + manifests match files)"
    )


# ---------------- Main ----------------

def main() -> int:
    ap = argparse.ArgumentParser()

    # Mode selection
    ap.add_argument("--local", action="store_true", help="verify local build outputs/receipts")

    # DSSE mode args (kept compatible)
    ap.add_argument("--envelope", default=None, help="DSSE envelope JSON file")
    ap.add_argument("--trust-key", action="append", default=[], help="KEYID=PATH_TO_PEM (repeatable)")
    ap.add_argument("--no-require-canonical-payload", action="store_true")
    ap.add_argument("--required-algs", default=None, help="comma-separated required digest algs (default: sha256)")

    # Shared
    ap.add_argument("--repo", default=".")
    ap.add_argument("--tmp-dir", default="_verify_build")

    # Local mode paths
    ap.add_argument("--root", default=None, help="fixture root containing receipts/ and build/ (optional bin/)")
    ap.add_argument("--receipts-dir", default=None)
    ap.add_argument("--build-dir", default=None)
    ap.add_argument("--asm-dir", default=None)
    ap.add_argument("--bin-dir", default=None)
    ap.add_argument("--strict", action="store_true")

    args, unknown = ap.parse_known_args()
    if unknown:
        # Keep behavior explicit: unknown args are almost always a user mistake.
        die(f"unknown args: {unknown}")

    repo = Path(args.repo).resolve()
    tmp = Path(args.tmp_dir)
    if not tmp.is_absolute():
        tmp = (repo / tmp).resolve()
    tmp.mkdir(parents=True, exist_ok=True)

    required_algs = get_required_algs(args.required_algs)

    if args.local:
        # Determine local-mode directories
        if args.root is not None:
            root = Path(args.root)
            if not root.is_absolute():
                root = (repo / root).resolve()
            receipts_dir = root / "receipts"
            build_dir = root / "build"
            bin_dir = (root / "bin") if (root / "bin").exists() else (repo / "bin")
        else:
            receipts_dir = (repo / (args.receipts_dir or "receipts")).resolve()
            build_dir = (repo / (args.build_dir or "build")).resolve()
            bin_dir = (repo / (args.bin_dir or "bin")).resolve()

        asm_dir = (repo / (args.asm_dir or "asm")).resolve()

        verify_local(
            repo,
            receipts_dir=receipts_dir,
            build_dir=build_dir,
            asm_dir=asm_dir,
            bin_dir=bin_dir,
            tmp_dir=tmp,
            strict=bool(args.strict),
        )
        return 0

    # DSSE mode
    if not args.envelope:
        die("--envelope is required unless --local is set")
    if not args.trust_key:
        die("at least one --trust-key KEYID=PUBKEY_PEM is required")

    trusted = load_trusted_keys(args.trust_key)

    envelope_path = Path(args.envelope)
    if not envelope_path.is_absolute():
        envelope_path = (Path.cwd() / envelope_path).resolve()

    envelope = parse_json_bytes(read_bytes(envelope_path), where=str(envelope_path))
    if not isinstance(envelope, dict):
        die("DSSE envelope must be object")

    _, stmt = verify_dsse_envelope(
        envelope,
        trusted_keys=trusted,
        tmp_dir=tmp,
        require_canonical_payload=(not args.no_require_canonical_payload),
    )

    verify_statement_shape(stmt)
    predicate = stmt["predicate"]

    # 2) Closure via input manifest
    manifest_uri, manifest_digest = find_manifest_material(predicate)
    verify_input_manifest(repo, manifest_uri, manifest_digest, required_algs)

    # Policy
    build_policy = predicate.get("buildPolicy") or {}
    if not isinstance(build_policy, dict):
        die("predicate.buildPolicy must be object")

    env = setup_env(build_policy)

    # 3) Rebuild IR
    ir = predicate.get("ir")
    if not isinstance(ir, dict):
        die("predicate.ir must be object")

    ir_cmd = ir.get("rebuildCommand")
    ir_out = ir.get("output")
    if not isinstance(ir_out, dict):
        die("predicate.ir.output must be object")

    ir_uri = ir_out.get("uri")
    ir_digest = ir_out.get("digest")

    run_cmd(ir_cmd, cwd=repo, env=env)

    if not isinstance(ir_uri, str) or not isinstance(ir_digest, dict):
        die("predicate.ir.output requires uri(str) and digest(object)")

    ir_path = repo / safe_relpath(ir_uri)
    if not ir_path.exists():
        die(f"IR output missing after rebuild: {ir_uri}")

    if ir_out.get("canonicalization") in ("stunir-json-c14n-v1", "jcs-rfc8785"):
        ir_bytes = read_bytes(ir_path)
        ir_obj = parse_json_bytes(ir_bytes, where=f"IR {ir_uri}")
        if canonical_json_bytes(ir_obj) != ir_bytes:
            die("IR JSON is not canonical (stunir-json-c14n-v1)")

    verify_digest_map(ir_path, ir_digest, required_algs)

    # 4) Rebuild artifacts
    codegen = predicate.get("codegen")
    if not isinstance(codegen, dict):
        die("predicate.codegen must be object")

    gen_cmd = codegen.get("rebuildCommand")
    outputs = codegen.get("outputs")
    if not isinstance(outputs, dict):
        die("predicate.codegen.outputs must be object")

    out_manifest = outputs.get("manifest")
    if not isinstance(out_manifest, dict):
        die("predicate.codegen.outputs.manifest must be object")

    out_uri = out_manifest.get("uri")
    out_digest = out_manifest.get("digest")

    run_cmd(gen_cmd, cwd=repo, env=env)

    if not isinstance(out_uri, str) or not isinstance(out_digest, dict):
        die("predicate.codegen.outputs.manifest requires uri(str) and digest(object)")

    out_path = repo / safe_relpath(out_uri)
    if not out_path.exists():
        die(f"artifact manifest missing after rebuild: {out_uri}")

    verify_digest_map(out_path, out_digest, required_algs)

    # Optional: subject[0] matches artifact manifest
    subj0 = stmt["subject"][0]
    if isinstance(subj0, dict) and subj0.get("name") == out_uri and isinstance(subj0.get("digest"), dict):
        subj_map = {normalize_alg(k): v for k, v in subj0["digest"].items() if isinstance(v, str)}
        for ra in required_algs:
            exp = subj_map.get(ra)
            if exp is None:
                die(f"subject digest missing required alg {ra}")
            got = digest_file(ra, out_path)
            if got != exp.lower():
                die(f"subject digest mismatch ({ra}): expected {exp.lower()}, got {got}")

    print("OK: DSSE signature valid; payload canonical; inputs closed; IR + outputs match digests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
