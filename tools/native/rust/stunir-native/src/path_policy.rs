use crate::errors::StunirError;

pub fn check_relpath_safe(p: &str) -> Result<(), StunirError> {
    if p.is_empty() {
        return Err(StunirError::VerifyFailed("unsafe_filename"));
    }
    if p.starts_with('/') {
        return Err(StunirError::VerifyFailed("unsafe_filename"));
    }
    if p.starts_with("./") {
        return Err(StunirError::VerifyFailed("unsafe_filename"));
    }
    if p.contains('\\') {
        return Err(StunirError::VerifyFailed("unsafe_filename"));
    }
    if p.chars().any(|c| c.is_whitespace()) {
        return Err(StunirError::VerifyFailed("unsafe_filename"));
    }

    for ch in p.chars() {
        let ok = ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-' || ch == '/';
        if !ok {
            return Err(StunirError::VerifyFailed("unsafe_filename"));
        }
    }

    for seg in p.split('/') {
        if seg.is_empty() {
            return Err(StunirError::VerifyFailed("unsafe_filename"));
        }
        if seg == "." || seg == ".." {
            return Err(StunirError::VerifyFailed("unsafe_filename"));
        }
        if seg.starts_with('-') {
            return Err(StunirError::VerifyFailed("unsafe_filename"));
        }
    }

    Ok(())
}

pub fn is_scope_excluded(p: &str) -> bool {
    p == "pack_manifest.tsv" || p.starts_with("objects/sha256/")
}
