#!/usr/bin/env bash
# usage: bash report/publish.sh [event ...]   — copy rendered reports into site/ and refresh the index
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p site
for f in report/_site/*.html; do
  [ -e "$f" ] || continue
  cp "$f" site/
  echo "  site/$(basename "$f")  $(du -h "$f" | cut -f1)"
done
python3 - <<'PY'
from pathlib import Path
import datetime, html
site = Path("site"); rows = []
for f in sorted(site.glob("*.html")):
    if f.name == "index.html":
        continue
    name = f.stem.replace("_", " ").title()
    rows.append(f'<li><a href="{f.name}">{html.escape(name)}</a> <span>{f.stat().st_size/1e6:.1f} MB</span></li>')
(site / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>SAR Flood Mapping — Reports</title>
<style>:root{{color-scheme:light dark}}body{{max-width:46rem;margin:4rem auto;padding:0 1rem;
font:16px/1.6 system-ui,sans-serif}}h1{{font-size:1.6rem;margin-bottom:.2rem}}p.sub{{color:#666;margin-top:0}}
ul{{list-style:none;padding:0}}li{{padding:.6rem 0;border-bottom:1px solid #8883}}li span{{color:#888;font-size:.85em}}
a{{text-decoration:none}}a:hover{{text-decoration:underline}}</style></head><body>
<h1>SAR flood mapping without Earth Engine</h1>
<p class="sub">Sentinel-1 event reports — classical baselines, deep learning, coherence, sensitivity.
<a href="https://github.com/Manish-Bilore/sar-flood-mapping">Source on GitHub</a></p>
<ul>{''.join(rows)}</ul>
<p class="sub">Updated {datetime.date.today().isoformat()}</p></body></html>""")
print("  site/index.html ->", len(rows), "report(s)")
PY
