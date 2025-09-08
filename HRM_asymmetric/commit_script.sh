#!/bin/bash

# Git Commit Automation Script
# Based on commit.mdc rule

set -e  # Exit on any error

echo "🔍 Checking git status..."

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not a git repository"
    exit 1
fi

# Check for changes (only modified files, not untracked)
if [ -z "$(git diff --name-only)" ] && [ -z "$(git diff --cached --name-only)" ]; then
    echo "ℹ️  No changes to commit"
    exit 0
fi

echo "📝 Changes detected:"
git status --short

echo ""
echo "📦 Staging modified files..."
git add -u  # Only stage modified files, not untracked files

echo ""
echo "🔍 Analyzing changes..."

# Get file changes
MODIFIED_FILES=$(git diff --cached --name-only)
NUM_FILES=$(echo "$MODIFIED_FILES" | wc -l)

echo "Modified files: $NUM_FILES"
echo "$MODIFIED_FILES"

# Analyze the type of changes
if echo "$MODIFIED_FILES" | grep -q "\.py$"; then
    HAS_CODE_CHANGES=true
else
    HAS_CODE_CHANGES=false
fi

if echo "$MODIFIED_FILES" | grep -q "JOURNAL.md\|SNAPSHOT.md"; then
    HAS_DOC_CHANGES=true
else
    HAS_DOC_CHANGES=false
fi

# Generate commit message based on changes
COMMIT_TYPE="fix"
COMMIT_SUMMARY=""
COMMIT_DETAILS=""
COMMIT_IMPACT=""

# Check for specific patterns in changes
if git diff --cached --name-only | grep -q "asymmetric_hrm.py"; then
    if git diff --cached asymmetric_hrm.py | grep -q "H_to_L_proj\|H_module"; then
        COMMIT_TYPE="fix"
        COMMIT_SUMMARY="correct HRM architecture - move projections and H processing outside T_cycles loop"
        COMMIT_DETAILS="- H_to_L projection now computed once per segment (not T_cycles times)
- H_module now processes once per segment (not T_cycles times)
- Aligns with standard HRM algorithm architecture
- Significant computational efficiency improvement"
        COMMIT_IMPACT="- Reduces redundant computations by T_cycles factor
- Corrects total operations from T_cycles × (L_blocks + H_blocks) to T_cycles × L_blocks + H_blocks
- Improves code maintainability and documentation"
    elif git diff --cached asymmetric_hrm.py | grep -q "def\|class"; then
        COMMIT_TYPE="feat"
        COMMIT_SUMMARY="add new features and improve code structure"
        COMMIT_DETAILS="- Enhanced function definitions and class structure
- Improved code organization and readability"
    else
        COMMIT_TYPE="refactor"
        COMMIT_SUMMARY="improve code quality and structure"
        COMMIT_DETAILS="- Enhanced code formatting and organization
- Improved type hints and documentation"
    fi
elif [ "$HAS_DOC_CHANGES" = true ]; then
    COMMIT_TYPE="docs"
    COMMIT_SUMMARY="update documentation and project tracking"
    COMMIT_DETAILS="- Updated JOURNAL.md with development history
- Updated SNAPSHOT.md with current project state"
    COMMIT_IMPACT="- Improved project documentation and tracking
- Better development history maintenance"
fi

# Add file-specific details
COMMIT_FILES=""
for file in $MODIFIED_FILES; do
    case $file in
        "asymmetric_hrm.py")
            COMMIT_FILES="$COMMIT_FILES
- asymmetric_hrm.py: Fixed projection timing and H processing frequency"
            ;;
        "JOURNAL.md")
            COMMIT_FILES="$COMMIT_FILES
- JOURNAL.md: Added detailed architectural fix documentation"
            ;;
        "SNAPSHOT.md")
            COMMIT_FILES="$COMMIT_FILES
- SNAPSHOT.md: Updated current state with fixes"
            ;;
        "example.py")
            COMMIT_FILES="$COMMIT_FILES
- example.py: Minor improvements and type hints"
            ;;
        "train.py")
            COMMIT_FILES="$COMMIT_FILES
- train.py: Training script improvements"
            ;;
        "test_*.py")
            COMMIT_FILES="$COMMIT_FILES
- $file: Test improvements and updates"
            ;;
        *)
            COMMIT_FILES="$COMMIT_FILES
- $file: General improvements"
            ;;
    esac
done

# Construct final commit message
COMMIT_MESSAGE="$COMMIT_TYPE: $COMMIT_SUMMARY

$COMMIT_DETAILS"

if [ -n "$COMMIT_IMPACT" ]; then
    COMMIT_MESSAGE="$COMMIT_MESSAGE

Impact:$COMMIT_IMPACT"
fi

if [ -n "$COMMIT_FILES" ]; then
    COMMIT_MESSAGE="$COMMIT_MESSAGE

Files modified:$COMMIT_FILES"
fi

echo ""
echo "📝 Generated commit message:"
echo "----------------------------------------"
echo "$COMMIT_MESSAGE"
echo "----------------------------------------"
echo ""

# Ask for confirmation
read -p "🤔 Proceed with commit? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Commit cancelled"
    exit 0
fi

echo ""
echo "💾 Committing changes..."

# Execute the commit
if git commit -m "$COMMIT_MESSAGE"; then
    echo "✅ Commit successful!"
    echo ""
    echo "📊 Commit details:"
    git log -1 --oneline
    echo ""
    echo "🔗 To push changes: git push"
else
    echo "❌ Commit failed"
    exit 1
fi
