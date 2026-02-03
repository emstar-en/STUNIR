# Instructions for Manually Adding Semantic IR Validation Workflow

## Background
The Semantic IR validation workflow file (`semantic_ir_validation.yml`) was removed from the git commits due to GitHub App permission limitations (workflows permission not available). This document provides step-by-step instructions for adding the workflow file manually via the GitHub web interface.

## Workflow File Location
- **Repository**: emstar-en/STUNIR
- **Branch**: devsite
- **Target Path**: `.github/workflows/semantic_ir_validation.yml`
- **Source File**: `WORKFLOW_FILE_FOR_MANUAL_ADDITION.yml` (saved in repository root)
- **Backup Location**: `/home/ubuntu/stunir_workflow_backup/semantic_ir_validation.yml`

## Step-by-Step Instructions

### Option 1: Using GitHub Web Interface (Recommended)

1. **Navigate to the Repository**
   - Go to: https://github.com/emstar-en/STUNIR
   - Switch to the `devsite` branch

2. **Navigate to Workflows Directory**
   - Click on the `.github` folder
   - Click on the `workflows` folder
   - You should see existing workflows like `release.yml` and `verify.yml`

3. **Create New Workflow File**
   - Click the "Add file" button
   - Select "Create new file"
   - Name the file: `semantic_ir_validation.yml`

4. **Copy Workflow Content**
   - Open the file `WORKFLOW_FILE_FOR_MANUAL_ADDITION.yml` from the repository root
   - Copy the entire content (starting from line 7, after the header comments)
   - Paste into the GitHub editor

5. **Commit the File**
   - Scroll down to the commit section
   - Commit message: `Add Semantic IR validation workflow`
   - Commit description (optional): `Manual addition due to GitHub App workflows permission limitation`
   - Select "Commit directly to the devsite branch"
   - Click "Commit new file"

### Option 2: Using GitHub CLI (if available)

```bash
# Ensure you're in the stunir_repo directory
cd /home/ubuntu/stunir_repo

# Add the workflow file back
git checkout devsite
cp WORKFLOW_FILE_FOR_MANUAL_ADDITION.yml .github/workflows/semantic_ir_validation.yml

# Remove the header comments (first 6 lines)
sed -i '1,6d' .github/workflows/semantic_ir_validation.yml

# Commit and push
git add .github/workflows/semantic_ir_validation.yml
git commit -m "Add Semantic IR validation workflow (manual addition)"
git push origin devsite
```

### Option 3: Using GitHub API

If you have a personal access token with `repo` and `workflow` permissions:

```bash
# Create the workflow file via GitHub API
# Replace YOUR_TOKEN with your personal access token

REPO_OWNER="emstar-en"
REPO_NAME="STUNIR"
BRANCH="devsite"
FILE_PATH=".github/workflows/semantic_ir_validation.yml"

# Read the file content (skip header comments)
CONTENT=$(tail -n +7 /home/ubuntu/stunir_repo/WORKFLOW_FILE_FOR_MANUAL_ADDITION.yml | base64 -w 0)

# Create the file via API
curl -X PUT \
  -H "Authorization: token YOUR_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/$REPO_OWNER/$REPO_NAME/contents/$FILE_PATH" \
  -d "{\"message\":\"Add Semantic IR validation workflow\",\"content\":\"$CONTENT\",\"branch\":\"$BRANCH\"}"
```

## Verification

After adding the workflow file, verify it was added correctly:

1. **Check Workflow File**
   - Navigate to: https://github.com/emstar-en/STUNIR/blob/devsite/.github/workflows/semantic_ir_validation.yml
   - Verify the content matches the original workflow

2. **Check Actions Tab**
   - Go to: https://github.com/emstar-en/STUNIR/actions
   - You should see the "Semantic IR Validation" workflow listed
   - It may run automatically if there are new commits to `devsite` branch

3. **Test Workflow Trigger**
   - Make a small change to any file in `schemas/semantic_ir/` or `tools/semantic_ir/`
   - Commit and push the change
   - Check the Actions tab to see if the workflow runs

## Workflow File Summary

The Semantic IR validation workflow provides comprehensive validation across multiple languages and platforms:

### Jobs Included:
1. **Python Validation** - Runs pytest and validates example IR files
2. **Rust Validation** - Checks Rust compilation and runs tests
3. **Schema Validation** - Validates JSON schemas
4. **Ada SPARK Validation** - Checks SPARK compilation (if available)
5. **Haskell Validation** - Checks Haskell compilation
6. **Integration Tests** - Tests round-trip serialization
7. **Report Generation** - Generates validation report artifact

### Trigger Conditions:
- Push to branches: `main`, `develop`, `devsite`
- Pull requests to: `main`, `develop`, `devsite`
- Only when files in these paths change:
  - `schemas/semantic_ir/**`
  - `tools/semantic_ir/**`
  - `tools/spark/src/semantic_ir/**`
  - `tools/rust/semantic_ir/**`
  - `tools/haskell/src/STUNIR/SemanticIR/**`
  - `tests/semantic_ir/**`
  - `examples/semantic_ir/**`

## Troubleshooting

### Workflow Not Running
- Check the Actions tab for any errors
- Verify the workflow file syntax is correct
- Ensure the trigger paths match your file changes

### Permission Errors
- GitHub App permissions cannot be modified by users
- Use a personal access token with `repo` and `workflow` permissions for manual workflow management
- Contact repository admin for org-level workflow permissions

### Validation Failures
- Check the workflow run logs in the Actions tab
- Common issues:
  - Missing dependencies (pydantic, jsonschema, pytest)
  - Syntax errors in new Semantic IR files
  - Schema validation failures

## Additional Resources

- **Workflow File**: `WORKFLOW_FILE_FOR_MANUAL_ADDITION.yml` (in repository root)
- **Backup**: `/home/ubuntu/stunir_workflow_backup/semantic_ir_validation.yml`
- **GitHub Actions Documentation**: https://docs.github.com/en/actions
- **Semantic IR Documentation**: See `docs/SEMANTIC_IR_SCHEMA_GUIDE.md`

## Support

If you encounter any issues:
1. Check the workflow run logs in GitHub Actions
2. Verify the workflow file syntax using a YAML validator
3. Review the Phase 1 and Phase 2 completion reports for context
4. Check the STUNIR documentation for Semantic IR implementation details

---

**Last Updated**: January 31, 2026
**Repository**: https://github.com/emstar-en/STUNIR
**Branch**: devsite
