"""acquisition_audit.py — Sentinel-1 acquisition-date audit for all study events.

For each event this finds, per relative orbit (path):
  * co-event scenes inside the flood window and their day offset from the peak
  * a dry-season baseline on the SAME path (needed for change detection)
  * whether a pre-event coherence pair exists: >=2 pre-event SLCs on that path
    within 36 days before the co-event scene (pre-pre coherence + pre-co coherence)
  * whether Planetary Computer RTC exists for that date (intensity route)
Source: ASF search (SLC catalogue = all S1 IW acquisitions) + PC STAC (sentinel-1-rtc).

Usage:  conda activate sarflood && python -u src/acquisition_audit.py
Output: outputs/audit/acquisitions.csv, outputs/audit/summary.csv
"""
from __future__ import annotations
import datetime as dt
from pathlib import Path
import asf_search as asf
import pandas as pd
import pystac_client

OUT = Path("outputs/audit"); OUT.mkdir(parents=True, exist_ok=True)

# bbox = (W, S, E, N); peak = (start, end) of peak inundation; window = search window
# Peak dates from published sources (see docs/03_literature_sota.md); Mumbai is provisional.
EVENTS = {
    "kerala_2018":    dict(bbox=(76.20, 9.05, 76.95, 10.30), peak=("2018-08-15", "2018-08-21"), window=("2018-08-05", "2018-09-05"), dry=("2018-01-01", "2018-04-30"), setting="rural+periurban"),
    "hyderabad_2020": dict(bbox=(78.25, 17.20, 78.70, 17.60), peak=("2020-10-13", "2020-10-18"), window=("2020-10-08", "2020-10-28"), dry=("2020-01-01", "2020-04-30"), setting="urban pluvial"),
    "bengaluru_2022": dict(bbox=(77.45, 12.80, 77.80, 13.15), peak=("2022-08-30", "2022-09-06"), window=("2022-08-25", "2022-09-15"), dry=("2022-01-01", "2022-04-30"), setting="urban pluvial"),
    "delhi_2023":     dict(bbox=(77.10, 28.45, 77.40, 28.90), peak=("2023-07-13", "2023-07-17"), window=("2023-07-08", "2023-07-28"), dry=("2023-01-01", "2023-04-30"), setting="urban riverine"),
    "chennai_2023":   dict(bbox=(80.10, 12.85, 80.35, 13.25), peak=("2023-12-04", "2023-12-08"), window=("2023-12-01", "2023-12-18"), dry=("2023-02-01", "2023-05-31"), setting="urban pluvial/coastal"),
    "mumbai_2024":    dict(bbox=(72.77, 18.88, 73.05, 19.30), peak=("2024-07-08", "2024-07-09"), window=("2024-07-04", "2024-07-20"), dry=("2024-01-01", "2024-04-30"), setting="urban pluvial (provisional date)"),
    "wayanad_2024":   dict(bbox=(76.05, 11.40, 76.25, 11.55), peak=("2024-07-30", "2024-08-02"), window=("2024-07-25", "2024-08-15"), dry=("2024-01-01", "2024-04-30"), setting="landslide (reference only)"),
    "assam_2020":     dict(bbox=(93.00, 26.60, 94.00, 26.90), peak=("2020-07-13", "2020-07-20"), window=("2020-07-05", "2020-07-30"), dry=("2020-01-01", "2020-03-31"), setting="rural riverine"),
    "assam_2022":     dict(bbox=(93.00, 26.60, 94.00, 26.90), peak=("2022-06-15", "2022-06-22"), window=("2022-06-08", "2022-08-15"), dry=("2022-01-01", "2022-03-31"), setting="rural riverine (DeepSARFlood AOI; 10 Aug comparator)"),
    "bihar_2020":     dict(bbox=(85.70, 25.80, 86.60, 26.50), peak=("2020-07-25", "2020-08-02"), window=("2020-07-15", "2020-08-15"), dry=("2020-01-01", "2020-04-30"), setting="rural riverine (Kosi/Darbhanga)"),
}

def wkt(b):
    w, s, e, n = b
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"

def asf_scenes(bbox, start, end):
    res = asf.geo_search(platform=asf.PLATFORM.SENTINEL1, processingLevel=asf.PRODUCT_TYPE.SLC,
                         beamMode=asf.BEAMMODE.IW, intersectsWith=wkt(bbox), start=start, end=end)
    rows = []
    for r in res:
        p = r.properties
        rows.append(dict(scene=p["sceneName"], platform=p["platform"], time=pd.Timestamp(p["startTime"]).tz_localize(None),
                         path=int(p["pathNumber"]), direction=p["flightDirection"], frame=p.get("frameNumber")))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = df.time.dt.normalize()
    return df

pc = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
def pc_rtc_dates(bbox, start, end):
    items = pc.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{start}/{end}").item_collection()
    return {(pd.Timestamp(i.properties["datetime"]).tz_localize(None).normalize(),
             int(i.properties.get("sat:relative_orbit", -1))) for i in items}

acq_rows, summ_rows = [], []
for ev, cfg in EVENTS.items():
    print(f"\n=== {ev}  ({cfg['setting']})")
    p0, p1 = (pd.Timestamp(x) for x in cfg["peak"])
    co = asf_scenes(cfg["bbox"], *cfg["window"])
    if co.empty:
        print("  no S1 IW SLC in window"); summ_rows.append(dict(event=ev, n_co=0)); continue
    # pre-event history for coherence pairs: 40 days before window start
    pre_start = (pd.Timestamp(cfg["window"][0]) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
    hist = asf_scenes(cfg["bbox"], pre_start, cfg["window"][1])
    dry = asf_scenes(cfg["bbox"], *cfg["dry"])
    try:
        rtc = pc_rtc_dates(cfg["bbox"], *cfg["window"])
    except Exception as e:
        print("  PC search failed:", e); rtc = set()

    co = co.drop_duplicates(["date", "path"]).sort_values("time")
    for _, r in co.iterrows():
        d = r.date
        off = 0 if p0 <= d <= p1 else (int((d - p0).days) if d < p0 else int((d - p1).days))
        same = hist[(hist.path == r.path) & (hist.date < d) & (hist.date >= d - pd.Timedelta(days=36))].date.unique()
        n_dry = dry[dry.path == r.path].date.nunique()
        row = dict(event=ev, date=d.date(), platform=r.platform, path=r.path, direction=r.direction,
                   days_from_peak=off, in_peak=off == 0, n_dry_same_path=n_dry,
                   n_pre_same_path_36d=len(same), coherence_pair_ok=len(same) >= 2,
                   pc_rtc=(d, r.path) in rtc or any(x[0] == d for x in rtc))
        acq_rows.append(row)
        print(f"  {row['date']}  {r.platform:12s} path {r.path:3d} {r.direction[:4]}  "
              f"Δpeak {off:+3d} d  dry={n_dry:2d}  pre36d={len(same)}  coh={'Y' if row['coherence_pair_ok'] else 'n'}  rtc={'Y' if row['pc_rtc'] else 'n'}")
    ev_df = pd.DataFrame([a for a in acq_rows if a["event"] == ev])
    best = ev_df.loc[ev_df.days_from_peak.abs().idxmin()]
    summ_rows.append(dict(event=ev, setting=cfg["setting"], n_co=len(ev_df), n_in_peak=int(ev_df.in_peak.sum()),
                          best_date=best.date, best_offset_days=int(best.days_from_peak), best_path=int(best.path),
                          best_dry_baseline=int(best.n_dry_same_path), best_coherence_ok=bool(best.coherence_pair_ok),
                          verdict=("OK" if abs(best.days_from_peak) <= 2 and best.n_dry_same_path > 0
                                   else "MARGINAL" if abs(best.days_from_peak) <= 6 else "MISSED")))

pd.DataFrame(acq_rows).to_csv(OUT / "acquisitions.csv", index=False)
summ = pd.DataFrame(summ_rows); summ.to_csv(OUT / "summary.csv", index=False)
print("\n", summ.to_string(index=False))
print("\nEOS-04 (Bhoonidhi) is not queried here: check MISSED/MARGINAL events manually at bhoonidhi.nrsc.gov.in")
