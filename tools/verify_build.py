#!/usr/bin/env python3
"""STUNIR receipt verifier (DSSE + input-closure manifest + rebuild+compare).

Design goals:
- Deterministic: compares bytes/digests, not semantics.
- Transparent: explicit policy fields, strict checks, clear failures.
- Not SLSA-shackled: uses familiar DSSE + Statement wrapper, but predicate is yours.

Expected envelope: DSSE v1 JSON with fields: payloadType, payload (b64), signatures[].
Expected payload: JSON bytes in *stunir-json-c14n-v1* canonical form (see docs/verification.md).
Expected payload shape: in-toto Statement v1 wrapper with predicateType urn:stunir:receipt:v1.

Closure model (#2): payload.predicate.materials includes the input-manifest file digest,
and payload.predicate.specClosure.manifest.uri points to that same manifest. The manifest enumerates
all input files with multi-alg digests.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
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


def read_json_bytes(path: Path) -> bytes:
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
# NOTE: This is *not* a complete RFC8785 implementation, but is fully specified by this function.

def _check_no_floats(x: Any, where: str = "$") -> None:
    if isinstance(x, float):
        die(f"floats forbidden for canonicalization (at {where})")
    if isinstance(x, dict):
        for k, v in x.items():
            if not isinstance(k, str):
                die(f"non-string key forbidden (at {where})")
            _check_no_floats(v, where + "." + k)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            _check_no_floats(v, where + f"[{i}]")
    elif isinstance(x, (str, int, bool)) or x is None:
        return
    else:
        die(f"unsupported JSON type {type(x)} at {where}")


def canonical_json_bytes(obj: Any) -> bytes:
    _check_no_floats(obj)
    s = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return s.encode("utf-8")


# ---------------- DSSE v1 ----------------

def pae(payload_type: str, payload: bytes) -> bytes:
    pt = payload_type.encode("utf-8")
    return b"DSSEv1 " + str(len(pt)).encode() + b" " + pt + b" " + str(len(payload)).encode() + b" " + payload


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
        pub.verify(sig, msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH), hashes.SHA256())
        return
    raise ValueError(f"unsupported public key type: {type(pub)}")


def verify_signature_openssl(public_key_pem_path: Path, sig_b64: str, msg: bytes, tmp_dir: Path) -> None:
    # Fallback via openssl if cryptography isn't available.
    msg_path = tmp_dir / "_dsse_msg.bin"
    sig_path = tmp_dir / "_dsse_sig.bin"
    msg_path.write_bytes(msg)
    sig_path.write_bytes(base64.b64decode(sig_b64))

    cmd = [
        "openssl", "pkeyutl", "-verify",
        "-pubin", "-inkey", str(public_key_pem_path),
        "-sigfile", str(sig_path),
        "-in", str(msg_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise ValueError(f"openssl verify failed: {r.stderr.strip() or r.stdout.strip()}")


def verify_dsse_envelope(envelope: Dict[str, Any], trusted_keys: Dict[str, TrustedKey], tmp_dir: Path, require_canonical_payload: bool = True) -> Tuple[bytes, Dict[str, Any]]:
    payload_type = envelope.get("payloadType")
    payload_b64 = envelope.get("payload")
    sigs = envelope.get("signatures")

    if not isinstance(payload_type, str) or not isinstance(payload_b64, str) or not isinstance(sigs, list) or not sigs:
        die("invalid DSSE envelope: require payloadType(str), payload(b64 str), signatures[](non-empty)")

    payload_bytes = base64.b64decode(payload_b64)
    msg = pae(payload_type, payload_bytes)

    verified = False
    last_err = None

    for s in sigs:
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
            continue

    if not verified:
        die(f"no valid DSSE signature found for trusted keyids; last error: {last_err}")

    stmt = parse_json_bytes(payload_bytes, where="DSSE payload")

    if require_canonical_payload:
        c14n = canonical_json_bytes(stmt)
        if c14n != payload_bytes:
            die("payload JSON is not in canonical form (stunir-json-c14n-v1): bytes differ")

    return payload_bytes, stmt


# ---------------- Digests ----------------

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


# ---------------- Paths / closure ----------------

def safe_relpath(p: str) -> Path:
    if not isinstance(p, str) or not p:
        die("path/uri must be non-empty string")
    pp = Path(p)
    if pp.is_absolute():
        die(f"absolute paths forbidden: {p}")
    norm = Path(os.path.normpath(p))
    if str(norm).startswith("..") or "/.." in str(norm).replace("\", "/"):
        die(f"path traversal forbidden: {p}")
    return norm


def load_trusted_keys(trust_args: List[str]) -> Dict[str, TrustedKey]:
    keys: Dict[str, TrustedKey] = {}
    for t in trust_args:
        if "=" not in t:
            die(f"--trust-key must be KEYID=PATH, got: {t}")
        keyid, p = t.split("=", 1)
        pem = Path(p).read_bytes()
        keys[keyid] = TrustedKey(keyid=keyid, public_key_pem=pem)
    return keys


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

    manifest_obj = parse_json_bytes(manifest_path.read_bytes(), where=f"manifest {manifest_uri}")
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


# ---------------- Rebuild + compare ----------------

def run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
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
    if not isinstance(stmt, dict):
        die("payload must be a JSON object")
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
    out = []
    for a in arg.split(","):
        a = normalize_alg(a)
        if a:
            out.append(a)
    if not out:
        die("--required-algs produced empty list")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envelope", required=True, help="DSSE envelope JSON file")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--tmp-dir", default="_verify_build")
    ap.add_argument("--trust-key", action="append", default=[], help="KEYID=PATH_TO_PEM (repeatable)")
    ap.add_argument("--no-require-canonical-payload", action="store_true")
    ap.add_argument("--required-algs", default=None, help="comma-separated required digest algs (default: sha256)")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    tmp = Path(args.tmp_dir).resolve()
    tmp.mkdir(parents=True, exist_ok=True)

    required_algs = get_required_algs(args.required_algs)

    if not args.trust_key:
        die("at least one --trust-key KEYID=PUBKEY_PEM is required")
    trusted = load_trusted_keys(args.trust_key)

    envelope_bytes = read_json_bytes(Path(args.envelope))
    envelope = parse_json_bytes(envelope_bytes, where=args.envelope)

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

    ir_path = repo / safe_relpath(ir_uri)
    if not ir_path.exists():
        die(f"IR output missing after rebuild: {ir_uri}")

    if ir_out.get("canonicalization") in ("stunir-json-c14n-v1", "jcs-rfc8785"):
        ir_bytes = ir_path.read_bytes()
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
