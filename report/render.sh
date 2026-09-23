#!/usr/bin/env bash
# usage: bash render.sh <event_id>   (run from report/) -> _site/<event_id>.html
set -euo pipefail
ev=${1:?event id, e.g. kerala_2018}
name=$(python3 -c "import sys;e=sys.argv[1].split('_');print(' '.join(w.capitalize() for w in e[:-1]),e[-1])" "$ev")
quarto render event.qmd -P event:"$ev" -M title:"SAR flood mapping — $name" --output "$ev.html"
echo "-> $(pwd)/_site/$ev.html"
