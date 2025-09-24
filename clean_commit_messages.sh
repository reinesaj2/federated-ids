#!/bin/bash

# This script removes mentions of Claude and Anthropic from git commit messages
# while preserving the code changes

echo "Starting commit message cleanup..."

# Use git filter-branch to rewrite commit messages
git filter-branch --force --msg-filter '
  # Remove Claude-related lines
  sed -e "/Claude/d" \
      -e "/claude/d" \
      -e "/Anthropic/d" \
      -e "/anthropic/d" \
      -e "/🤖 Generated with/d" \
      -e "/Co-Authored-By: Claude/d" \
      -e "/Co-authored-by: Claude/d"
' --tag-name-filter cat -- --all

echo "Commit message cleanup complete!"
echo "Run 'git log --all --grep=\"claude\" --regexp-ignore-case' to verify"
