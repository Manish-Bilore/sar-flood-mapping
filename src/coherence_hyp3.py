"""coherence_hyp3.py — InSAR coherence for an event via ASF HyP3 burst InSAR (processed in the cloud, laptop only downloads).

Pairs (same relative orbit, equal temporal baseline) from config/events.yaml -> <event>.coherence:
  pre = two pre-event dates  -> background coherence (what the ground does without flooding)
  co  = pre-event + co-event -> coherence loss where water appeared (urban flood signal)
Output: data/events/<ev>/coh/pre_coh.tif, co_coh.tif (normalised coherence 0-1, 40 m, looks 10x2).

Needs a free NASA Earthdata login in ~/.netrc:
  machine urs.earthdata.nasa.gov login <user> password <pass>
Cost: HyP3 multi-burst jobs, one per sub-swath and pair (check your monthly credits: python -c "import hyp3_sdk;print(hyp3_sdk.HyP3().check_credits())").

  python -u src/coherence_hyp3.py kerala_2018 --dry-run      # list bursts / jobs only
  python -u src/coherence_hyp3.py kerala_2018                # submit, wait, download, mosaic
"""
from __future__ import annotations
import argparse, datetime as dt, sys, zipfile
from pathlib import Path
import asf_search as asf, numpy as np, pandas as pd, rioxarray, yaml
from rioxarray.merge import merge_arrays

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/events.yaml"))
EVENTS = {**CFG["rural"], **CFG["urban"]}


COLS = ["scene", "full_id", "swath", "idx"]


def path_dates(bbox, path, start, end, pol="VV"):
    """Every S1 burst acquisition date on this relative orbit in a window (used to repair missing pair dates)."""
    w, s, e, n = bbox
    res = asf.geo_search(dataset=asf.DATASET.SLC_BURST, intersectsWith=f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))",
                         start=str(start), end=str(end), relativeOrbit=int(path), polarization=pol)
    return sorted({pd.Timestamp(r.properties["startTime"]).tz_localize(None).normalize() for r in res})


def bursts(bbox, date, path, pol="VV"):
    w, s, e, n = bbox
    d0 = pd.Timestamp(date); d1 = d0 + pd.Timedelta(days=1)
    res = asf.geo_search(dataset=asf.DATASET.SLC_BURST, intersectsWith=f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))",
                         start=d0.strftime("%Y-%m-%d"), end=d1.strftime("%Y-%m-%d"), relativeOrbit=int(path),
                         polarization=pol)
    rows = [dict(scene=r.properties["sceneName"], full_id=r.properties["burst"]["fullBurstID"],
                 swath=r.properties["burst"]["subswath"], idx=r.properties["burst"]["burstIndex"]) for r in res]
    return (pd.DataFrame(rows, columns=COLS) if not rows else pd.DataFrame(rows)).drop_duplicates("full_id")


def plan(ev, cfg, pol="VV"):
    c = cfg["coherence"]; jobs = []
    for pair in [k for k in ("pre", "co", "dry") if k in c or f"{k}_window" in c]:
        if pair in c:
            d_ref, d_sec = [str(x) for x in c[pair]]
            a, b = bursts(c["bbox"], d_ref, c["path"], pol), bursts(c["bbox"], d_sec, c["path"], pol)
            if a.empty or b.empty:                      # a configured date with no acquisition -> show what exists
                miss = [d for d, df in ((d_ref, a), (d_sec, b)) if df.empty]
                w0 = (pd.Timestamp(d_ref) - pd.Timedelta(days=40)).date()
                w1 = (pd.Timestamp(d_sec) + pd.Timedelta(days=40)).date()
                have = path_dates(c["bbox"], c["path"], w0, w1, pol)
                print(f"  !! {pair}: no bursts on {', '.join(miss)} (path {c['path']}). Acquisitions nearby: "
                      f"{', '.join(d.strftime('%Y-%m-%d') for d in have) or 'none'}")
                print(f"     fix config/events.yaml, or replace '{pair}:' with '{pair}_window: [start, end]' to pick automatically")
                continue
        else:                                            # <pair>_window: choose the tightest consecutive pair available
            w0, w1 = [str(x) for x in c[f"{pair}_window"]]
            have = path_dates(c["bbox"], c["path"], w0, w1, pol)
            if len(have) < 2:
                print(f"  !! {pair}_window {w0}..{w1}: fewer than two acquisitions on path {c['path']} — skipped")
                continue
            gaps = [(have[i + 1] - have[i], have[i], have[i + 1]) for i in range(len(have) - 1)]
            _, r0, r1 = min(gaps, key=lambda g: g[0])
            d_ref, d_sec = r0.strftime("%Y-%m-%d"), r1.strftime("%Y-%m-%d")
            print(f"  {pair}: auto-selected {d_ref} -> {d_sec} ({(r1 - r0).days} d) from {len(have)} acquisitions in the window")
            a, b = bursts(c["bbox"], d_ref, c["path"], pol), bursts(c["bbox"], d_sec, c["path"], pol)
        m = a.merge(b, on=["full_id", "swath"], suffixes=("_ref", "_sec")).sort_values(["swath", "idx_ref"])
        print(f"  {pair}: {d_ref} -> {d_sec}: {len(a)} / {len(b)} bursts, {len(m)} matched")
        for sw, g in m.groupby("swath"):
            g = g.reset_index(drop=True)
            runs, cur = [], [0]                          # split into contiguous runs of <= 15 bursts
            for i in range(1, len(g)):
                if g.idx_ref[i] == g.idx_ref[i - 1] + 1 and len(cur) < 15:
                    cur.append(i)
                else:
                    runs.append(cur); cur = [i]
            runs.append(cur)
            for k, r in enumerate(runs):
                jobs.append(dict(pair=pair, name=f"{ev}_{pair}_{sw}_{k}" + ("" if pol == "VV" else f"_{pol.lower()}"), ref=list(g.scene_ref[r]), sec=list(g.scene_sec[r])))
    return jobs


def main(a):
    cfg = EVENTS[a.event]
    if "coherence" not in cfg:
        sys.exit(f"no 'coherence' block for {a.event} in config/events.yaml")
    pol = a.pol.upper()
    if pol not in ("VV", "HH"):
        sys.exit(f"HyP3 burst InSAR processes VV and HH only (the API rejects {pol}). For UrbanSARFloods-style stacks, "
                 f"train on the 'vvcoh_int' band subset instead — see src/train_usf.py.")
    jobs = plan(a.event, cfg, pol)
    for j in jobs:
        print(f"  job {j['name']}: {len(j['ref'])} bursts")
    if a.dry_run:
        return
    import hyp3_sdk as sdk
    hyp3 = sdk.HyP3()
    print("  credits remaining:", hyp3.check_credits())
    out = ROOT / f"data/events/{a.event}/coh"; raw = out / "raw"; raw.mkdir(parents=True, exist_ok=True)
    batch = sdk.Batch()
    for j in jobs:
        existing = hyp3.find_jobs(name=j["name"])
        if len(existing):
            batch += existing; print(f"  reuse {j['name']}"); continue
        if len(j["ref"]) == 1:
            batch += hyp3.submit_insar_isce_burst_job(j["ref"][0], j["sec"][0], name=j["name"], looks=a.looks)
        else:
            batch += hyp3.submit_insar_isce_multi_burst_job(j["ref"], j["sec"], name=j["name"], looks=a.looks)
    print(f"  {len(batch)} jobs submitted/reused; waiting (typically 10-40 min)...")
    batch = hyp3.watch(batch)
    zips = sorted(raw.glob("*.zip"))
    if len(zips) < len(batch):                        # reuse anything already downloaded
        zips = batch.download_files(raw)
    for f in zips:
        with zipfile.ZipFile(f) as z:
            for n in z.namelist():
                if n.endswith("_corr.tif"):
                    z.extract(n, raw)
    ref = rioxarray.open_rasterio(sorted((ROOT / f"data/events/{a.event}/rtc").glob("co_*.tif"))[0], chunks={})
    for pair in [k for k in ("pre", "co", "dry") if k in cfg["coherence"] or f"{k}_window" in cfg["coherence"]]:
        sel = []                                     # product file names don't carry the job name -> map via the batch
        for jb in batch:
            if jb.name.startswith(f"{a.event}_{pair}_") and jb.name.endswith("_vh") == (pol == "VH"):
                stem = Path(jb.files[0]["filename"]).stem
                sel += list(raw.rglob(f"{stem}*_corr.tif"))
        if not sel:
            print(f"  !! no coherence files for {pair}"); continue
        arrs = [rioxarray.open_rasterio(t, masked=True).rio.reproject(ref.rio.crs) for t in sel]
        mos = merge_arrays(arrs, method="max").squeeze("band", drop=True)
        mos.attrs.pop("_FillValue", None); mos.encoding.pop("_FillValue", None)   # HyP3 sets both -> xarray refuses
        mos = mos.astype("float32").rio.write_nodata(np.nan, encoded=False)
        fn = f"{pair}_coh.tif" if pol == "VV" else f"{pair}_coh_{pol.lower()}.tif"
        mos.rio.to_raster(out / fn, driver="COG", compress="DEFLATE")
        print(f"  wrote {fn} from {len(sel)} products")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("event"); ap.add_argument("--looks", default="10x2", choices=["20x4", "10x2", "5x1"])
    ap.add_argument("--pol", default="VV", choices=["VV", "VH", "vv", "vh"],
                    help="burst polarization; VH coherence is what the UrbanSARFloods 'all'/'coh' models expect")
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args())
