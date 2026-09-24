#!/usr/bin/env bash
# usage: bash render.sh <event_id>[_<yyyymmdd>]   (run from report/) -> _site/<id>.html
# For events with more than one co-event date, make_event_products.py writes outputs/<event>_<yyyymmdd>/,
# so pass that same id here.
set -euo pipefail
id=${1:?event id, e.g. kerala_2018 or assam_2022_20220810}
name=$(python3 - "$id" <<'PY'
import sys, datetime
p = sys.argv[1].split("_")
if len(p) > 1 and len(p[-1]) == 8 and p[-1].isdigit():
    d = datetime.datetime.strptime(p.pop(), "%Y%m%d").strftime("%d %b %Y")
else:
    d = ""
name = " ".join(w.capitalize() if not w.isdigit() else w for w in p)
print(f"{name} — {d}" if d else name)
PY
)
quarto render event.qmd -P event:"$id" -M title:"SAR flood mapping — $name" --output "$id.html"
echo "-> $(pwd)/_site/$id.html"
