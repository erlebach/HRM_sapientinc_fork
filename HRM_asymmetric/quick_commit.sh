#!/bin/bash

# Quick commit script - commits modified files with intelligent message

# Check for changes
if [ -z "$(git diff --name-only)" ]; then
    echo "No changes to commit"
    exit 0
fi

# Stage modified files
git add -u

# Generate message based on changes
if git diff --cached --name-only | grep -q "asymmetric_hrm.py"; then
    if git diff --cached asymmetric_hrm.py | grep -q "H_to_L_proj\|H_module"; then
        MSG="fix: correct HRM architecture - move projections and H processing outside T_cycles loop"
    elif git diff --cached asymmetric_hrm.py | grep -q "def\|class"; then
        MSG="feat: add new features and improve code structure"
    else
        MSG="refactor: improve code quality and structure"
    fi
elif git diff --cached --name-only | grep -q "JOURNAL.md\|SNAPSHOT.md"; then
    MSG="docs: update project documentation and tracking"
else
    MSG="fix: general improvements and updates"
fi

# Add file details
FILES=$(git diff --cached --name-only | tr '\n' ' ')
MSG="$MSG

Files: $FILES"

# Show message and commit
echo "Commit message:"
echo "$MSG"
echo ""
read -p "Proceed? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git commit -m "$MSG"
    echo "✅ Committed successfully!"
else
    echo "❌ Commit cancelled"
    git reset HEAD
fi

