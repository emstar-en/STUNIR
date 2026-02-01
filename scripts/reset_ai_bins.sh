#!/bin/bash
# Resets the AI Memory Bins to their default template state.
# Use this when starting a completely new task.

BIN_DIR="meta/ai_bins"
mkdir -p "$BIN_DIR"

echo "Resetting AI Bins in $BIN_DIR..."

# Reset A_TASK.md
cat > "$BIN_DIR/A_TASK.md" <<EOF
# Bin A: Current Task Focus

## 🎯 Objective
[WAITING FOR INPUT]

## 📋 Checklist
- [ ] Step 1: ...

## 🛑 Blockers
- None
EOF

# Reset B_CONTEXT.md
cat > "$BIN_DIR/B_CONTEXT.md" <<EOF
# Bin B: Active Context (Working Set)

## 🔥 Hot Files
- scripts/build.sh

## 🔑 Key Variables / Constants
- STUNIR_BUILD_EPOCH

## 🧠 Knowledge Cache
EOF

# Reset C_DRAFT.md
cat > "$BIN_DIR/C_DRAFT.md" <<EOF
# Bin C: Code Draft Workbench

## 📝 Snippet Scratchpad
\`\`\`bash
# Draft Area
\`\`\`
EOF

# Reset D_ERRORS.md
cat > "$BIN_DIR/D_ERRORS.md" <<EOF
# Bin D: Error Analysis & Debugging

## 💥 Recent Failure

## 🕵️ Analysis

## 🛠️ Proposed Fix
EOF

echo "Bins reset. AI memory cleared."
