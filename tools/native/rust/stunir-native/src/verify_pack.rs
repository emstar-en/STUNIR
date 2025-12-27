use crate::errors::StunirError;
use crate::hash::sha256_hex;
use crate::path_policy::{check_relpath_safe, is_scope_excluded};
use base64::Engine;
use ed25519_dalek::{Signature, VerifyingKey};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const ROOT_ATTESTATION_VERSION: &str = "stunir.pack.root_attestation_text.v0";

#[derive(Debug, Clone)]
struct AttestationRecord {
    rec_type: String,
    kv: BTreeMap<String, String>,
}

pub fn cmd_verify_pack(
    root: &str,
    objects_dir_rel: &str,
    pack_manifest_rel: &str,
    root_attestation_rel: &str,
    check_completeness: bool,
    ed25519_pubkey_b64: Option<&str>,
) -> Result<(), StunirError> {
    let rootp = PathBuf::from(root);
    let objects_dir = rootp.join(objects_dir_rel);
    let pack_manifest = rootp.join(pack_manifest_rel);
    let root_attestation = rootp.join(root_attestation_rel);

    if !objects_dir.exists() {
        return Err(StunirError::VerifyFailed("missing_objects_dir"));
    }

    let ra_bytes = std::fs::read(&root_attestation)
        .map_err(|_| StunirError::VerifyFailed("missing_root_attestation"))?;

    maybe_verify_ed25519_signature(&rootp, &ra_bytes, ed25519_pubkey_b64)?;

    let ra_text = String::from_utf8_lossy(&ra_bytes);
    let mut ra_lines = Vec::new();
    for raw in ra_text.lines() {
        let line = raw.strip_suffix('\r').unwrap_or(raw).to_string();
        if !line.trim().is_empty() {
            ra_lines.push(line);
        }
    }

    if ra_lines.is_empty() || ra_lines[0] != ROOT_ATTESTATION_VERSION {
        return Err(StunirError::VerifyFailed("bad_version_line"));
    }

    let mut records = Vec::new();
    for line in ra_lines.iter().skip(1) {
        let rec = parse_record_line(line)?;
        if rec.rec_type != "artifact" && rec.rec_type != "ir" {
            return Err(StunirError::VerifyFailed("unknown_record_type"));
        }
        records.push(rec);
    }

    let ir_count = records.iter().filter(|r| r.rec_type == "ir").count();
    if ir_count != 1 {
        return Err(StunirError::VerifyFailed("ir_count_not_one"));
    }

    let manifest_rec = records.iter().find(|r| {
        r.rec_type == "artifact"
            && r.kv.get("kind").map(|s| s.as_str()) == Some("manifest")
            && r.kv.get("logical_path").map(|s| s.as_str()) == Some("pack_manifest.tsv")
    });

    let manifest_rec = manifest_rec.ok_or_else(|| StunirError::VerifyFailed("missing_manifest_binding"))?;

    let manifest_digest = manifest_rec
        .kv
        .get("digest")
        .ok_or_else(|| StunirError::VerifyFailed("missing_manifest_binding"))?;

    let (algo, hex) = parse_digest(manifest_digest)?;
    if algo != "sha256" {
        return Err(StunirError::VerifyFailed("bad_digest"));
    }

    let pm_bytes = std::fs::read(&pack_manifest)
        .map_err(|_| StunirError::VerifyFailed("missing_pack_manifest"))?;
    if pm_bytes.iter().any(|b| *b == b'\r') {
        return Err(StunirError::VerifyFailed("crlf_detected_in_pack_manifest"));
    }

    let pm_hash = sha256_hex(&pm_bytes);
    if pm_hash != hex {
        return Err(StunirError::VerifyFailed("pack_manifest_hash_mismatch"));
    }

    verify_object_store_blob(&objects_dir, &hex, &pm_bytes)
        .map_err(|e| e)?;

    let pm_text = std::str::from_utf8(&pm_bytes)
        .map_err(|_| StunirError::VerifyFailed("bad_digest"))?;

    let mut entries: Vec<(String, String)> = Vec::new();
    for line in pm_text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let (h, p) = parse_manifest_line(line).map_err(|_| StunirError::VerifyFailed("bad_digest"))?;
        if is_scope_excluded(&p) {
            return Err(StunirError::VerifyFailed("manifest_scope_violation"));
        }
        check_relpath_safe(&p)?;
        entries.push((h, p));
    }

    for w in entries.windows(2) {
        let a = &w[0].1;
        let b = &w[1].1;
        if a.as_bytes() > b.as_bytes() {
            return Err(StunirError::VerifyFailed("manifest_not_sorted"));
        }
        if a == b {
            return Err(StunirError::VerifyFailed("unsafe_filename"));
        }
    }

    let mut manifest_paths = BTreeSet::new();
    for (h, p) in &entries {
        manifest_paths.insert(p.clone());

        let fp = rootp.join(p);
        let meta = std::fs::metadata(&fp)
            .map_err(|_| StunirError::VerifyFailed("manifest_file_missing"))?;
        if !meta.is_file() {
            return Err(StunirError::VerifyFailed("missing_object"));
        }
        let bytes = std::fs::read(&fp).map_err(|_| StunirError::VerifyFailed("manifest_file_missing"))?;
        let digest = sha256_hex(&bytes);
        if &digest != h {
            return Err(StunirError::VerifyFailed("manifest_file_hash_mismatch"));
        }

        verify_object_store_blob(&objects_dir, h, &bytes)
            .map_err(|e| e)?;
    }

    for r in &records {
        if let Some(d) = r.kv.get("digest") {
            let (algo, hex) = parse_digest(d)?;
            if algo != "sha256" {
                return Err(StunirError::VerifyFailed("bad_digest"));
            }
            let obj_path = objects_dir.join(&hex);
            let obj_bytes = std::fs::read(&obj_path)
                .map_err(|_| StunirError::VerifyFailed("missing_object"))?;
            let obj_hash = sha256_hex(&obj_bytes);
            if obj_hash != hex {
                return Err(StunirError::VerifyFailed("object_hash_mismatch"));
            }
        }
    }

    if check_completeness {
        let mut actual = BTreeSet::new();
        for entry in WalkDir::new(&rootp).follow_links(false) {
            let entry = entry.map_err(|_| StunirError::VerifyFailed("bad_digest"))?;
            if !entry.file_type().is_file() {
                continue;
            }
            let rel = entry.path().strip_prefix(&rootp)
                .map_err(|_| StunirError::VerifyFailed("bad_digest"))?;
            let rel_s = rel.to_string_lossy().replace('\\', "/");
            if rel_s == pack_manifest_rel || rel_s.starts_with(objects_dir_rel) {
                continue;
            }
            actual.insert(rel_s);
        }

        if actual != manifest_paths {
            return Err(StunirError::VerifyFailed("manifest_incomplete_or_extra_files"));
        }
    }

    println!("OK verify.pack");
    Ok(())
}

fn parse_record_line(line: &str) -> Result<AttestationRecord, StunirError> {
    let mut parts = line.split_whitespace();
    let rec_type = parts
        .next()
        .ok_or_else(|| StunirError::VerifyFailed("unknown_record_type"))?
        .to_string();

    let mut kv = BTreeMap::new();
    for t in parts {
        if let Some((k, v)) = t.split_once('=') {
            kv.insert(k.to_string(), v.to_string());
        }
    }

    Ok(AttestationRecord { rec_type, kv })
}

fn is_lower_hex64(s: &str) -> bool {
    s.len() == 64 && s.chars().all(|c| matches!(c, '0'..='9' | 'a'..='f'))
}

fn parse_digest(d: &str) -> Result<(String, String), StunirError> {
    let (algo, hex) = d
        .split_once(':')
        .ok_or_else(|| StunirError::VerifyFailed("bad_digest"))?;

    if !is_lower_hex64(hex) {
        return Err(StunirError::VerifyFailed("bad_digest"));
    }

    Ok((algo.to_string(), hex.to_string()))
}

fn parse_manifest_line(line: &str) -> Result<(String, String), ()> {
    let mut it = line.splitn(2, ' ');
    let hex = it.next().ok_or(())?.to_string();
    let path = it.next().ok_or(())?.to_string();
    if !is_lower_hex64(&hex) {
        return Err(());
    }
    if path.is_empty() {
        return Err(());
    }
    if line.contains('\t') {
        return Err(());
    }
    Ok((hex, path))
}

fn verify_object_store_blob(objects_dir: &Path, hex: &str, expected_bytes: &[u8]) -> Result<(), StunirError> {
    let p = objects_dir.join(hex);
    let bytes = std::fs::read(&p).map_err(|_| StunirError::VerifyFailed("missing_object"))?;
    let digest = sha256_hex(&bytes);
    if digest != hex {
        return Err(StunirError::VerifyFailed("object_hash_mismatch"));
    }
    if bytes != expected_bytes {
        // This should be impossible without a hash collision; treat as hash mismatch.
        return Err(StunirError::VerifyFailed("object_hash_mismatch"));
    }
    Ok(())
}

fn maybe_verify_ed25519_signature(
    root: &Path,
    root_attestation_bytes: &[u8],
    ed25519_pubkey_b64: Option<&str>,
) -> Result<(), StunirError> {
    let sig_path = root.join("root_attestation.txt.sig");
    if !sig_path.exists() {
        return Ok(());
    }

    let Some(pk_b64) = ed25519_pubkey_b64 else {
        return Ok(());
    };

    let sig_b64 = std::fs::read_to_string(&sig_path)
        .map_err(|_| StunirError::VerifyFailed("bad_digest"))?;
    let sig_b64 = sig_b64.trim();

    let pk_bytes = base64::engine::general_purpose::STANDARD
        .decode(pk_b64)
        .map_err(|_| StunirError::VerifyFailed("bad_digest"))?;
    if pk_bytes.len() != 32 {
        return Err(StunirError::VerifyFailed("bad_digest"));
    }

    let sig_bytes = base64::engine::general_purpose::STANDARD
        .decode(sig_b64)
        .map_err(|_| StunirError::VerifyFailed("bad_digest"))?;
    if sig_bytes.len() != 64 {
        return Err(StunirError::VerifyFailed("bad_digest"));
    }

    let vk = VerifyingKey::from_bytes(&pk_bytes.try_into().unwrap())
        .map_err(|_| StunirError::VerifyFailed("bad_digest"))?;
    let sig = Signature::from_bytes(&sig_bytes.try_into().unwrap());

    vk.verify_strict(root_attestation_bytes, &sig)
        .map_err(|_| StunirError::VerifyFailed("bad_digest"))?;

    Ok(())
}
