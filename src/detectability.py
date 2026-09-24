"""detectability.py — could Sentinel-1 have detected this flood at all?

For events where no acquisition falls near the flood peak (Mumbai 2024: one orbit, 12-day repeat, nearest pass +3 d),
a single map says nothing on its own. This runs the same classical rule over EVERY pass of the season and asks whether
the event date is separable from the rest — i.e. whether the flood signal exceeds the ordinary seasonal and tidal
variation of water extent.

Reads the per-date baselines written by baseline_classical.py (one folder per co-event date) and writes:
  outputs/<event>_detectability/tables/water_timeseries.csv   area per date and method, plus built-up flooded area
  outputs/<event>_detectability/tables/detectability.csv      event date vs the other passes: z-score, rank, verdict,
                                                              and the minimum flood extent this sampling could have detected
  outputs/<event>_detectability/summary.json                  headline numbers for the report

  python -u src/detectability.py mumbai_2024
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, rioxarray, yaml

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/events.yaml"))
EVENTS = {**CFG["rural"], **CFG["urban"]}
METHODS = ["otsu_vv", "split_vv", "cd_vv", "urban_incr"]


def rd(f):
    return rioxarray.open_rasterio(f, masked=True).squeeze("band", drop=True)


def main(a):
    ev = a.event; cfg = EVENTS[ev]
    d = ROOT / f"data/events/{ev}"
    out = ROOT / f"outputs/{ev}_detectability"; tab = out / "tables"; tab.mkdir(parents=True, exist_ok=True)
    ed = str(cfg.get("event_date") or cfg["co"][0]["date"]).replace("-", "")

    bdirs = sorted([p for p in (d / "baseline").iterdir() if p.is_dir()]) if (d / "baseline").exists() else []
    if len(bdirs) < 3:
        sys.exit(f"need per-date baselines in {d / 'baseline'} (found {len(bdirs)}) — run: python -u src/baseline_classical.py {ev}")

    anc = d / "anc"
    wc = rd(anc / "worldcover.tif").values if (anc / "worldcover.tif").exists() else None
    rows = []
    for b in bdirs:
        date = b.name
        co_f = next((d / "rtc").glob(f"co_{date}_*.tif"), None)
        if co_f is None:
            continue
        px_km2 = abs(np.prod(rioxarray.open_rasterio(co_f).rio.resolution())) / 1e6
        st = pd.read_csv(b / "stats.csv") if (b / "stats.csv").exists() else None
        for m in METHODS:
            f = b / f"flood_{m}.tif"
            if not f.exists():
                continue
            msk = rd(f).values == 1
            r = dict(date=pd.to_datetime(date), method=m, area_km2=float(msk.sum() * px_km2),
                     is_event=(date == ed))
            if wc is not None and wc.shape == msk.shape:
                r["builtup_km2"] = float(np.sum(msk & (wc == 50)) * px_km2)
            if st is not None and m in set(st.method):
                r["threshold_db"] = float(st.threshold_db[st.method == m].iloc[0])
            rows.append(r)
    ts = pd.DataFrame(rows).sort_values(["method", "date"])
    if ts.empty:
        sys.exit("no per-date flood maps found")
    ts.to_csv(tab / "water_timeseries.csv", index=False)

    # is the event pass separable from the rest of the season?
    det = []
    for m, g in ts.groupby("method"):
        evr = g[g.is_event]
        oth = g[~g.is_event]
        if evr.empty or len(oth) < 2:
            continue
        x = float(evr.area_km2.iloc[0]); mu, sd = float(oth.area_km2.mean()), float(oth.area_km2.std(ddof=1))
        z = (x - mu) / sd if sd > 0 else np.nan
        rank = int((g.area_km2 > x).sum()) + 1                      # 1 = largest extent of the season
        # minimum detectable event: how much water a flood must ADD before it clears z = 2 against the seasonal spread
        mde = 2 * sd if sd > 0 else np.nan
        det.append(dict(method=m, event_km2=x, others_mean_km2=mu, others_sd_km2=sd, others_max_km2=float(oth.area_km2.max()),
                        z_score=z, rank_of_season=rank, n_passes=len(g),
                        min_detectable_km2=mde, event_excess_km2=x - mu,
                        separable=bool(z >= 2 and rank == 1)))
    det = pd.DataFrame(det)
    det.to_csv(tab / "detectability.csv", index=False)

    sound = det[det.method.isin(["otsu_vv", "split_vv", "cd_vv"])]
    verdict = ("separable" if bool(sound.separable.all()) else
               "partly separable" if bool(sound.separable.any()) else "not separable")
    summ = dict(event=ev, tag=f"{ev}_detectability", event_date=str(pd.to_datetime(ed).date()),
                n_passes=int(ts.date.nunique()), repeat_days=12, path=int(cfg["co"][0]["path"]),
                dates=[str(x.date()) for x in sorted(ts.date.unique())],
                verdict=verdict, methods=det.to_dict("records"),
                bbox=cfg["bbox"], zooms=cfg.get("zooms", {}), zoom_labels=cfg.get("zoom_labels", {}))
    json.dump(summ, open(out / "summary.json", "w"), indent=1, default=str)
    print(det.round(2).to_string(index=False))
    print(f"\n  verdict: the {summ['event_date']} pass is **{verdict}** from the other {summ['n_passes'] - 1} passes")
    print("  ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("event")
    main(ap.parse_args())
