"""publish.py — copy rendered reports into site/ and build the landing page.

Content and ordering come from site/manifest.yml; headline numbers are read from outputs/*/summary.json so the
page cannot drift from the results. Events without a report yet appear as in-progress or planned, which is how
the set stays legible while it is still being filled in.

  bash report/publish.sh          # copies report/_site/*.html into site/, then runs this
  python report/publish.py        # rebuild index.html only
"""
from __future__ import annotations
import html
import json
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
STATUS = {"published": ("Published", "ok"), "in-progress": ("In progress", "wip"),
          "planned": ("Planned", "todo"), "dropped": ("Dropped", "no")}


def summary(ev: str) -> dict:
    """summary.json for an event, whichever suffix the run produced."""
    for p in sorted((ROOT / "outputs").glob(f"{ev}*/summary.json")):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return {}


def headline(e: dict) -> str:
    """One measured number per event, straight from the pipeline's own output."""
    s = summary(e["id"])
    if not s:
        return "—"
    if e.get("kind") == "detectability" or "verdict" in s:
        v = s.get("verdict", "—")
        return f'<b class="{"no" if v == "not separable" else "ok"}">{html.escape(v)}</b><small>{s.get("n_passes", "?")} passes tested</small>'
    km = s.get("final_km2")
    if km is None:
        return "—"
    v = s.get("final_voters", [])
    n_cls = sum(1 for x in v if not x.startswith("dl_"))
    how = f'majority of {n_cls} classical map{"" if n_cls == 1 else "s"}'
    if any(x.startswith("dl_") for x in v):
        how += " + deep-learning ensemble"
    return f'<b>{km:,.0f} km²</b><small>{html.escape(how)}</small>'


def card(e: dict) -> str:
    label, cls = STATUS.get(e.get("status", "planned"), STATUS["planned"])
    name = html.escape(e["name"])
    title = f'<a href="{e["report"]}">{name}</a>' if e.get("report") and (SITE / e["report"]).exists() else name
    size = ""
    if e.get("report") and (SITE / e["report"]).exists():
        mb = (SITE / e["report"]).stat().st_size / 1e6
        size = f'<span class="size">{mb:.0f} MB</span>' if mb >= 1 else ""
    return f"""    <tr class="{cls}">
      <td class="ev"><span class="dot"></span>{title}{size}
        <small>{html.escape(e.get("setting", ""))}{" · " + html.escape(str(e["date"])) if e.get("date") else ""}</small></td>
      <td class="res">{headline(e)}</td>
      <td class="st"><span class="badge {cls}">{label}</span></td>
      <td class="note">{html.escape(e.get("note", "").strip())}</td>
    </tr>"""


def main() -> None:
    m = yaml.safe_load(open(SITE / "manifest.yml"))
    rows = "\n".join(card(e) for e in m["events"])
    findings = "\n".join(f"      <li>{html.escape(f.strip())}</li>" for f in m.get("findings", []))
    dropped = " · ".join(f'<b>{html.escape(d["name"])}</b> {html.escape(d["why"])}' for d in m.get("dropped", []))
    n_pub = sum(1 for e in m["events"] if e.get("status") == "published")

    (SITE / "index.html").write_text(f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(m["title"])}</title>
<meta name="description" content="{html.escape(m["tagline"].strip())}">
<style>
:root {{
  color-scheme: light dark;
  --fg: #16181d; --dim: #5d6470; --line: #e3e6ec; --bg: #fff; --card: #f7f8fa;
  --ok: #1a7f4b; --wip: #b8770c; --todo: #6b7280; --no: #a02c3a; --link: #1b4ea8;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --fg: #e7e9ee; --dim: #9aa3b2; --line: #2a2f3a; --bg: #111318; --card: #171a20;
           --ok: #4ac585; --wip: #e0a63c; --todo: #8d95a3; --no: #e8798a; --link: #7db0ff; }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.62 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
.wrap {{ max-width: 62rem; margin: 0 auto; padding: 3.5rem 1.25rem 5rem; }}
h1 {{ font-size: 2rem; line-height: 1.2; margin: 0 0 .5rem; letter-spacing: -.02em; }}
.tag {{ color: var(--dim); max-width: 46rem; margin: 0 0 1.25rem; }}
.meta {{ color: var(--dim); font-size: .9rem; margin-bottom: 2.5rem; }}
a {{ color: var(--link); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
h2 {{ font-size: 1.05rem; text-transform: uppercase; letter-spacing: .08em; color: var(--dim);
  margin: 2.75rem 0 .9rem; font-weight: 600; }}
ul.find {{ margin: 0; padding-left: 1.1rem; }}
ul.find li {{ margin-bottom: .55rem; }}
table {{ width: 100%; border-collapse: collapse; }}
th {{ text-align: left; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em;
  color: var(--dim); font-weight: 600; padding: 0 .6rem .5rem 0; border-bottom: 1px solid var(--line); }}
td {{ padding: .95rem .6rem; border-bottom: 1px solid var(--line); vertical-align: top; }}
td small {{ display: block; color: var(--dim); font-size: .82rem; margin-top: .15rem; }}
.ev {{ font-weight: 600; min-width: 13rem; }}
.ev .dot {{ display: inline-block; width: .5rem; height: .5rem; border-radius: 50%;
  margin-right: .5rem; background: var(--todo); vertical-align: middle; }}
tr.ok .dot {{ background: var(--ok); }} tr.wip .dot {{ background: var(--wip); }}
.size {{ color: var(--dim); font-size: .75rem; font-weight: 400; margin-left: .4rem; }}
.res b {{ font-size: 1.02rem; white-space: nowrap; }} .res b.ok {{ color: var(--ok); }} .res b.no {{ color: var(--no); }}
.badge {{ font-size: .72rem; padding: .18rem .5rem; border-radius: 999px; white-space: nowrap;
  border: 1px solid currentColor; }}
.badge.ok {{ color: var(--ok); }} .badge.wip {{ color: var(--wip); }} .badge.todo {{ color: var(--todo); }}
.note {{ color: var(--dim); font-size: .9rem; }}
.foot {{ margin-top: 3rem; padding: 1.1rem 1.2rem; background: var(--card); border-radius: .6rem;
  color: var(--dim); font-size: .88rem; }}
@media (max-width: 720px) {{
  .wrap {{ padding-top: 2.25rem; }} h1 {{ font-size: 1.55rem; }}
  th:nth-child(4), td.note {{ display: none; }}
}}
</style></head>
<body><div class="wrap">

<h1>{html.escape(m["title"])}</h1>
<p class="tag">{html.escape(m["tagline"].strip())}</p>
<p class="meta">{n_pub} of {len(m["events"])} events published ·
  <a href="{m["repo"]}">source and method on GitHub</a> ·
  Manish Bilore · updated {date.today().strftime("%d %b %Y")}</p>

<h2>What came out of it</h2>
<ul class="find">
{findings}
</ul>

<h2>Events</h2>
<table>
  <thead><tr><th>Event</th><th>Flood extent</th><th>Status</th><th>What it tests</th></tr></thead>
  <tbody>
{rows}
  </tbody>
</table>

<div class="foot">
  <b>Not mapped:</b> {dropped}<br><br>
  Every report regenerates from one parameterised Quarto template: raw imagery, speckle filtering, VV against VH,
  InSAR coherence, terrain and water masks, classical baselines, five deep-learning models, inter-method agreement,
  a sensitivity analysis of the ensemble, and the limits of each result. Input data is open throughout —
  Sentinel-1 RTC from the Planetary Computer, SLC bursts through ASF HyP3, Copernicus DEM, JRC Global Surface Water,
  ESA WorldCover.
</div>

</div></body></html>
""")
    print(f"  site/index.html -> {n_pub} published, {len(m['events'])} events listed")


if __name__ == "__main__":
    main()
