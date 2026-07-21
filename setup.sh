#!/usr/bin/env bash
# Initialize the backbone submodules (pinned commits) and apply the small
# compatibility patches needed to reproduce the paper's results.
set -e
cd "$(dirname "$0")"

echo "[1/2] fetching submodules (pinned commits) ..."
git submodule update --init --depth 1

echo "[2/2] applying compatibility patches ..."
for p in patches/*.patch; do
  case "$p" in
    patches/igev-*) dir=third_party/IGEV ;;
    *) echo "  (skip: no target for $p)"; continue ;;
  esac
  if git -C "$dir" apply --reverse --check "$PWD/$p" 2>/dev/null; then
    echo "  already applied: $p"
  else
    git -C "$dir" apply "$PWD/$p" && echo "  applied: $p"
  fi
done
echo "done. Submodules ready."
