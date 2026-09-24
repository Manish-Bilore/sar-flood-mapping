#!/usr/bin/env bash
# usage: bash report/publish.sh   — copy rendered reports into site/ and rebuild the landing page
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p site
shopt -s nullglob
for f in report/_site/*.html; do
  cp "$f" site/
  echo "  site/$(basename "$f")  $(du -h "$f" | cut -f1)"
done
python3 report/publish.py
