#!/usr/bin/env bash
# usage: bash render_detect.sh <event>   (run from report/) -> _site/<event>_detectability.html
set -euo pipefail
ev=${1:?event id, e.g. mumbai_2024}
name=$(python3 -c "
import sys; p=sys.argv[1].split('_')
print(' '.join(w.capitalize() if not w.isdigit() else w for w in p))" "$ev")
quarto render detectability.qmd -P event:"$ev" -M title:"SAR flood detectability — $name" --output "${ev}_detectability.html"
echo "-> $(pwd)/_site/${ev}_detectability.html"
