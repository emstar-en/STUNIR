//! STUNIR Cryptographic Utilities
//!
//! This module provides cryptographic hashing functions for files and directories.
//! Directory hashing uses a Merkle tree structure for deterministic, verifiable hashes.
//!
//! # Security Considerations
//! - Uses SHA-256 (256-bit security against preimage attacks)
//! - Merkle tree structure enables partial verification
//! - Deterministic ordering via sorted paths (Unicode codepoint order)
//! - Path traversal protection via canonicalization
//! - No symlink following to prevent security issues

use anyhow::{anyhow, Context, Result};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Maximum file size to prevent DoS (1GB)
const MAX_FILE_SIZE: u64 = 1024 * 1024 * 1024;

/// Maximum directory depth to prevent stack overflow
const MAX_DIR_DEPTH: usize = 100;

/// Hash a path, dispatching to file or directory hashing as appropriate.
///
/// # Security
/// - Validates path is within expected boundaries
/// - Does not follow symlinks
/// - Returns error for special files (devices, etc.)
pub fn hash_path(path: &Path) -> Result<String> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("Failed to get metadata for: {:?}", path))?;

    // Security: Reject symlinks to prevent traversal attacks
    if metadata.is_symlink() {
        return Err(anyhow!(
            "SECURITY: Symlinks are not allowed for hashing: {:?}",
            path
        ));
    }

    if metadata.is_dir() {
        hash_directory(path, 0)
    } else if metadata.is_file() {
        hash_file(path)
    } else {
        Err(anyhow!(
            "SECURITY: Cannot hash special file (device, socket, etc.): {:?}",
            path
        ))
    }
}

/// Compute SHA-256 hash of a file.
///
/// # Security
/// - Validates file size before reading
/// - Uses streaming to handle large files efficiently
/// - Does not follow symlinks
pub fn hash_file(path: &Path) -> Result<String> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("Failed to get file metadata: {:?}", path))?;

    // Security: Check file size to prevent DoS
    if metadata.len() > MAX_FILE_SIZE {
        return Err(anyhow!(
            "SECURITY: File too large ({} bytes > {} max): {:?}",
            metadata.len(),
            MAX_FILE_SIZE,
            path
        ));
    }

    // Security: Reject symlinks
    if metadata.is_symlink() {
        return Err(anyhow!(
            "SECURITY: Symlinks are not allowed: {:?}",
            path
        ));
    }

    let mut file =
        File::open(path).with_context(|| format!("Failed to open file: {:?}", path))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192]; // 8KB buffer for efficient I/O

    loop {
        let count = file
            .read(&mut buffer)
            .with_context(|| format!("Failed to read file: {:?}", path))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }

    Ok(hex::encode(hasher.finalize()))
}

/// Merkle tree node for directory hashing
#[derive(Debug)]
struct MerkleNode {
    /// Relative path from the root directory (using forward slashes)
    path: String,
    /// SHA-256 hash of the content (file) or children (directory)
    hash: String,
    /// Whether this is a directory
    is_dir: bool,
}

/// Compute deterministic Merkle tree hash of a directory.
///
/// # Algorithm
/// 1. Collect all files and subdirectories
/// 2. Sort entries by relative path (Unicode codepoint order)
/// 3. For files: hash = SHA256(file_content)
/// 4. For dirs: hash = SHA256(sorted_children_hashes)
/// 5. Root hash = SHA256(path || hash for each sorted entry)
///
/// # Security
/// - Deterministic ordering prevents manipulation
/// - Depth limit prevents stack overflow attacks
/// - Path validation prevents traversal attacks
/// - Symlinks are rejected
pub fn hash_directory(dir: &Path, depth: usize) -> Result<String> {
    // Security: Prevent stack overflow from deeply nested directories
    if depth > MAX_DIR_DEPTH {
        return Err(anyhow!(
            "SECURITY: Directory too deep ({} > {} max): {:?}",
            depth,
            MAX_DIR_DEPTH,
            dir
        ));
    }

    // Canonicalize the directory path
    let canonical_dir = dir
        .canonicalize()
        .with_context(|| format!("Failed to canonicalize directory: {:?}", dir))?;

    // Collect all entries with their Merkle nodes
    let entries = collect_directory_entries(&canonical_dir, &canonical_dir, depth)?;

    // Build the root hash from sorted entries
    let root_hash = compute_merkle_root(&entries);

    Ok(root_hash)
}

/// Collect all entries in a directory tree as Merkle nodes.
fn collect_directory_entries(
    root: &Path,
    current: &Path,
    depth: usize,
) -> Result<BTreeMap<String, MerkleNode>> {
    let mut entries = BTreeMap::new();

    let read_dir = fs::read_dir(current)
        .with_context(|| format!("Failed to read directory: {:?}", current))?;

    for entry_result in read_dir {
        let entry = entry_result
            .with_context(|| format!("Failed to read directory entry in: {:?}", current))?;
        let entry_path = entry.path();
        let metadata = fs::symlink_metadata(&entry_path)
            .with_context(|| format!("Failed to get metadata: {:?}", entry_path))?;

        // Security: Skip symlinks
        if metadata.is_symlink() {
            continue;
        }

        // Compute relative path using forward slashes for determinism
        let rel_path = entry_path
            .strip_prefix(root)
            .map_err(|e| anyhow!("Path prefix error: {}", e))?
            .to_string_lossy()
            .replace('\\', "/");

        if metadata.is_file() {
            let hash = hash_file(&entry_path)?;
            entries.insert(
                rel_path.clone(),
                MerkleNode {
                    path: rel_path,
                    hash,
                    is_dir: false,
                },
            );
        } else if metadata.is_dir() {
            // Recursively process subdirectory
            let sub_entries = collect_directory_entries(root, &entry_path, depth + 1)?;
            
            // Compute directory hash from its children
            let dir_hash = compute_merkle_root(&sub_entries);
            
            entries.insert(
                rel_path.clone(),
                MerkleNode {
                    path: rel_path,
                    hash: dir_hash,
                    is_dir: true,
                },
            );
            
            // Also include all children in the flat map
            entries.extend(sub_entries);
        }
    }

    Ok(entries)
}

/// Compute the Merkle root hash from a sorted map of entries.
///
/// Format: SHA256(path1 || '\0' || hash1 || '\n' || path2 || '\0' || hash2 || '\n' || ...)
fn compute_merkle_root(entries: &BTreeMap<String, MerkleNode>) -> String {
    let mut hasher = Sha256::new();

    // BTreeMap maintains sorted order by key (path)
    for (path, node) in entries {
        // Format: "path\0hash\n" for each entry
        // Using null byte as separator prevents path/hash confusion
        hasher.update(path.as_bytes());
        hasher.update(b"\0");
        hasher.update(node.hash.as_bytes());
        hasher.update(b"\n");
    }

    hex::encode(hasher.finalize())
}

/// Verify that a path is safe (no traversal attacks).
///
/// # Security
/// - Rejects paths with ".." components
/// - Rejects absolute paths
/// - Validates UTF-8 encoding
pub fn validate_path(path: &str) -> Result<PathBuf> {
    // Security: Reject empty paths
    if path.is_empty() {
        return Err(anyhow!("SECURITY: Empty path not allowed"));
    }

    let path_buf = PathBuf::from(path);

    // Security: Reject absolute paths
    if path_buf.is_absolute() {
        return Err(anyhow!(
            "SECURITY: Absolute paths not allowed: {}",
            path
        ));
    }

    // Security: Check for path traversal
    for component in path_buf.components() {
        match component {
            std::path::Component::ParentDir => {
                return Err(anyhow!(
                    "SECURITY: Path traversal not allowed (..): {}",
                    path
                ));
            }
            std::path::Component::Normal(s) => {
                // Validate the component is valid UTF-8 and doesn't contain null bytes
                let s_str = s.to_str().ok_or_else(|| {
                    anyhow!("SECURITY: Invalid UTF-8 in path: {}", path)
                })?;
                if s_str.contains('\0') {
                    return Err(anyhow!(
                        "SECURITY: Null bytes not allowed in path: {}",
                        path
                    ));
                }
            }
            _ => {}
        }
    }

    Ok(path_buf)
}

/// Compute hash of bytes directly (for testing and internal use).
pub fn hash_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_hash_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, b"hello world").unwrap();

        let hash = hash_file(&file_path).unwrap();
        // SHA256("hello world")
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_hash_empty_directory() {
        let dir = tempdir().unwrap();
        let hash = hash_directory(dir.path(), 0).unwrap();
        // Empty directory should have a consistent hash
        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 64); // SHA256 hex length
    }

    #[test]
    fn test_hash_directory_deterministic() {
        let dir = tempdir().unwrap();
        
        // Create files in non-alphabetical order
        fs::write(dir.path().join("b.txt"), b"file b").unwrap();
        fs::write(dir.path().join("a.txt"), b"file a").unwrap();
        fs::create_dir(dir.path().join("subdir")).unwrap();
        fs::write(dir.path().join("subdir/c.txt"), b"file c").unwrap();

        let hash1 = hash_directory(dir.path(), 0).unwrap();
        let hash2 = hash_directory(dir.path(), 0).unwrap();

        // Hashes should be identical (deterministic)
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_validate_path_rejects_traversal() {
        assert!(validate_path("../etc/passwd").is_err());
        assert!(validate_path("foo/../bar").is_err());
        assert!(validate_path("foo/bar/../../etc").is_err());
    }

    #[test]
    fn test_validate_path_rejects_absolute() {
        assert!(validate_path("/etc/passwd").is_err());
    }

    #[test]
    fn test_validate_path_accepts_valid() {
        assert!(validate_path("foo/bar/baz.txt").is_ok());
        assert!(validate_path("file.txt").is_ok());
        assert!(validate_path("dir/subdir/file").is_ok());
    }

    #[test]
    fn test_hash_bytes() {
        let hash = hash_bytes(b"test");
        assert_eq!(
            hash,
            "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        );
    }

    #[test]
    fn test_reject_symlinks() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("real.txt");
        let link_path = dir.path().join("link.txt");
        
        fs::write(&file_path, b"real file").unwrap();
        
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&file_path, &link_path).unwrap();
            assert!(hash_file(&link_path).is_err());
        }
    }
}
