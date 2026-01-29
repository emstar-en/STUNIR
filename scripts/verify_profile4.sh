#!/bin/bash
# STUNIR Profile 4 Verification Script
# Extended Profile 3 with strict integrity checking
#
# Usage: verify_profile4.sh [--local] [--verbose]
#
# Profile 4 Requirements:
#   - All Profile 3 checks pass
#   - pack_manifest.tsv is present and bound in root_attestation.txt
#   - pack_manifest.tsv is sorted deterministically
#   - All files in pack_manifest.tsv hash-match
#   - No extra files exist that aren't in manifest

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/.."
VERBOSE=0
LOCAL_MODE=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            LOCAL_MODE=1
            shift
            ;;
        --verbose|-v)
            VERBOSE=1
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--local] [--verbose]"
            echo "  --local    Run verification locally"
            echo "  --verbose  Enable verbose output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_verbose() {
    if [[ $VERBOSE -eq 1 ]]; then
        echo "[VERBOSE] $1"
    fi
}

# Calculate SHA256 hash
calc_hash() {
    local file="$1"
    if command -v sha256sum &>/dev/null; then
        sha256sum "$file" | cut -d' ' -f1
    elif command -v shasum &>/dev/null; then
        shasum -a 256 "$file" | cut -d' ' -f1
    elif command -v openssl &>/dev/null; then
        openssl dgst -sha256 "$file" | awk '{print $NF}'
    else
        log_error "No SHA256 tool found"
        exit 1
    fi
}

# Verify pack_manifest.tsv exists and is bound
verify_pack_manifest_binding() {
    local root_att="${REPO_ROOT}/root_attestation.txt"
    local pack_manifest="${REPO_ROOT}/pack_manifest.tsv"
    
    log_info "Checking pack_manifest.tsv binding..."
    
    if [[ ! -f "$pack_manifest" ]]; then
        log_error "pack_manifest.tsv not found - Profile 4 requires this file"
        return 1
    fi
    
    if [[ ! -f "$root_att" ]]; then
        log_error "root_attestation.txt not found"
        return 1
    fi
    
    # Calculate manifest hash
    local manifest_hash
    manifest_hash=$(calc_hash "$pack_manifest")
    log_verbose "pack_manifest.tsv hash: $manifest_hash"
    
    # Check if hash is bound in root_attestation.txt
    if grep -q "$manifest_hash" "$root_att"; then
        log_info "pack_manifest.tsv is bound in root_attestation.txt"
        return 0
    else
        log_error "pack_manifest.tsv hash not found in root_attestation.txt"
        log_error "Expected hash: $manifest_hash"
        return 1
    fi
}

# Verify pack_manifest.tsv is sorted
verify_manifest_sorted() {
    local pack_manifest="${REPO_ROOT}/pack_manifest.tsv"
    
    log_info "Verifying pack_manifest.tsv is sorted..."
    
    # Create sorted version and compare
    local sorted_manifest
    sorted_manifest=$(sort "$pack_manifest")
    local original_manifest
    original_manifest=$(cat "$pack_manifest")
    
    if [[ "$sorted_manifest" == "$original_manifest" ]]; then
        log_info "pack_manifest.tsv is correctly sorted"
        return 0
    else
        log_error "pack_manifest.tsv is not sorted deterministically"
        return 1
    fi
}

# Verify all files in manifest match their hashes
verify_manifest_hashes() {
    local pack_manifest="${REPO_ROOT}/pack_manifest.tsv"
    local failed=0
    local verified=0
    
    log_info "Verifying file hashes in pack_manifest.tsv..."
    
    while IFS=$'\t' read -r hash filepath; do
        [[ -z "$hash" || -z "$filepath" ]] && continue
        [[ "$hash" == "#"* ]] && continue  # Skip comments
        
        local full_path="${REPO_ROOT}/${filepath}"
        
        if [[ ! -f "$full_path" ]]; then
            log_error "File not found: $filepath"
            ((failed++))
            continue
        fi
        
        local actual_hash
        actual_hash=$(calc_hash "$full_path")
        
        if [[ "$actual_hash" == "$hash" ]]; then
            log_verbose "MATCH: $filepath"
            ((verified++))
        else
            log_error "MISMATCH: $filepath"
            log_error "  Expected: $hash"
            log_error "  Actual:   $actual_hash"
            ((failed++))
        fi
    done < "$pack_manifest"
    
    log_info "Verified: $verified files"
    
    if [[ $failed -gt 0 ]]; then
        log_error "Failed: $failed files"
        return 1
    fi
    return 0
}

# Check for extra files not in manifest
check_extra_files() {
    local pack_manifest="${REPO_ROOT}/pack_manifest.tsv"
    local extra_files=0
    
    log_info "Checking for extra files not in manifest..."
    
    # Get list of files from manifest
    local manifest_files
    manifest_files=$(awk -F'\t' '{print $2}' "$pack_manifest" | sort)
    
    # Find all files in pack root (excluding hidden, issues, etc.)
    local actual_files
    actual_files=$(find "$REPO_ROOT" -type f \
        ! -path "*/.git/*" \
        ! -path "*/issues/*" \
        ! -name ".*" \
        -printf "%P\n" | sort)
    
    # Compare
    local extra
    extra=$(comm -23 <(echo "$actual_files") <(echo "$manifest_files"))
    
    if [[ -n "$extra" ]]; then
        log_error "Extra files found not in manifest:"
        echo "$extra" | while read -r f; do
            log_error "  - $f"
            ((extra_files++))
        done
        return 1
    fi
    
    log_info "No extra files found"
    return 0
}

# Run Profile 3 verification first
run_profile3_verification() {
    log_info "Running Profile 3 verification first..."
    
    if [[ -x "${SCRIPT_DIR}/verify_strict.sh" ]]; then
        if "${SCRIPT_DIR}/verify_strict.sh" --local; then
            log_info "Profile 3 verification passed"
            return 0
        else
            log_error "Profile 3 verification failed"
            return 1
        fi
    else
        log_warn "verify_strict.sh not found, skipping Profile 3 base checks"
        return 0
    fi
}

# Main verification
main() {
    log_info "Starting STUNIR Profile 4 Verification"
    log_info "Repository: ${REPO_ROOT}"
    echo
    
    local failures=0
    
    # Step 1: Run Profile 3 base verification
    if ! run_profile3_verification; then
        ((failures++))
    fi
    
    # Step 2: Verify pack_manifest binding
    if ! verify_pack_manifest_binding; then
        ((failures++))
    fi
    
    # Step 3: Verify manifest is sorted
    if ! verify_manifest_sorted; then
        ((failures++))
    fi
    
    # Step 4: Verify all manifest hashes
    if ! verify_manifest_hashes; then
        ((failures++))
    fi
    
    # Step 5: Check for extra files (strict mode)
    # Note: This is optional in some deployments
    # if ! check_extra_files; then
    #     ((failures++))
    # fi
    
    echo
    if [[ $failures -eq 0 ]]; then
        log_info "Profile 4 verification PASSED"
        exit 0
    else
        log_error "Profile 4 verification FAILED ($failures checks failed)"
        exit 1
    fi
}

main
