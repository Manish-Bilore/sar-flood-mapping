"""baseline_classical.py — unsupervised flood baselines per event (no training data needed).

Methods (all on Lee-filtered LINEAR gamma0, thresholds in dB):
  otsu_vv     co VV < Otsu(VV)                                      single-image, global
  otsu_vh     co VH < Otsu(VH)
  split_vv    co VV < split-based threshold (bimodal tiles, Ashman D >= 2)
  cd_vv       log-ratio 10log10(co/dry) VV < -3 dB  AND  co VV < split_vv threshold
  gee2024     re-implementation of the 2024 GEE logic for comparison:
              Refined Lee, mosaics before/after as in GEE, ratio of dB images > 1.1, GSW seas>=5, slope<5 (dB-division bug kept on purpose)
  urban_incr  built-up (WorldCover 50) AND log-ratio VV > +3 dB   -> candidate double-bounce flood (reported separately)
Post-processing for water classes: exclude permanent water (GSW seasonality >= --perm-months),
HAND > 15 m, slope > 5 deg; remove objects < 8 px (as in the 2024 script).

Outputs data/events/<ev>/baseline/: flood_<method>.tif (uint8) + stats.csv (km² by method x WorldCover class)
Usage: python -u src/baseline_classical.py kerala_2018 [--perm-months 10] [--co 20180821]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, rioxarray, yaml

sys.path.insert(0, str(Path(__file__).parent))
from sarlib import (lee_filter, refined_lee, to_db, otsu_threshold, split_based_threshold,  # noqa: E402
                    exclusion_mask, clean)

ROOT = Path(__file__).resolve().parents[1]
_CFG = yaml.safe_load(open(ROOT / "config/events.yaml"))
EVENTS = {**_CFG["rural"], **_CFG["urban"]}
WC = {10: "tree", 20: "shrub", 30: "grass", 40: "cropland", 50: "built-up", 60: "bare",
      80: "perm. water", 90: "herb. wetland", 95: "mangrove", 70: "snow", 100: "moss"}

def read(f):
    return rioxarray.open_rasterio(f, masked=True)

def main(ev, perm_months, co_tag):
    d = ROOT / f"data/events/{ev}"
    co_f = sorted((d / "rtc").glob(f"co_{co_tag or ''}*.tif"))[0]
    path = co_f.stem.split("_p")[-1]
    dry_f, pre_f = d / f"rtc/dry_p{path}.tif", sorted((d / "rtc").glob(f"pre_*_p{path}.tif"))
    n_co = len(list((d / "rtc").glob("co_*.tif")))
    out = d / "baseline" if n_co == 1 else d / "baseline" / co_f.stem.split("_")[1]   # one folder per co-event date
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== {ev}: co={co_f.name} dry={dry_f.name} pre={[p.name for p in pre_f]}")

    co = read(co_f); ref = co.isel(band=0)
    vv, vh = lee_filter(co.values[0]), lee_filter(co.values[1])
    vv_db, vh_db = to_db(vv), to_db(vh)
    valid = np.isfinite(vv_db) & np.isfinite(vh_db)

    anc = d / "anc"
    hand = read(anc / "hand.tif").values[0] if (anc / "hand.tif").exists() else None
    slope = read(anc / "slope.tif").values[0] if (anc / "slope.tif").exists() else None
    seas = read(anc / "gsw_seas.tif").values[0] if (anc / "gsw_seas.tif").exists() else None
    wc = read(anc / "worldcover.tif").values[0] if (anc / "worldcover.tif").exists() else None
    ex = exclusion_mask(hand, slope)
    hand_ex = (np.nan_to_num(hand, nan=0) > 15.0) if hand is not None else None
    slope_ex = (np.nan_to_num(slope, nan=0) > 5.0) if slope is not None else None
    perm = (np.nan_to_num(seas, nan=0) >= perm_months) if seas is not None else np.zeros_like(valid)
    tidal = np.zeros_like(valid)                                # coastal events: water that comes and goes with the tide
    itd = EVENTS[ev].get("intertidal")
    if itd and (anc / "gsw_occ.tif").exists():
        occ = read(anc / "gsw_occ.tif").values[0]
        tidal = (np.nan_to_num(occ, nan=0) >= itd[0]) & (np.nan_to_num(occ, nan=0) <= itd[1]) & ~perm
        print(f"  intertidal band (GSW occurrence {itd[0]}-{itd[1]} %): {tidal.sum() * 1e-4 * 100:.1f} km² "
              f"— excluded from flood and reported separately")
    if seas is not None and wc is not None:
        perm |= np.isnan(seas) & (wc == 80)                      # open sea: GSW nodata, WorldCover water
    if ex is None:
        print("  !! no HAND/slope — run fetch_ancillary.py first; continuing without terrain mask")
        ex = np.zeros_like(valid)

    thr = {"otsu_vv": otsu_threshold(vv_db), "otsu_vh": otsu_threshold(vh_db)}
    thr["split_vv"], nsel, ntot = split_based_threshold(vv_db)
    print(f"  thresholds dB: otsu_vv {thr['otsu_vv']:.2f}  otsu_vh {thr['otsu_vh']:.2f}  "
          f"split_vv {thr['split_vv']:.2f} ({nsel}/{ntot} bimodal tiles)")

    # Plausibility guard. Open water at C band sits near -15 to -22 dB (VV) / -22 to -28 dB (VH). A global threshold far
    # above that means the histogram valley the method found separates something else — in a dense city, built-up from
    # vegetation. Such a map is not a flood map, so it is rejected here rather than passed downstream.
    lim = EVENTS[ev].get("max_water_db", {"otsu_vv": -10.0, "split_vv": -10.0, "otsu_vh": -15.0})
    if isinstance(lim, (int, float)):
        lim = {"otsu_vv": float(lim), "split_vv": float(lim), "otsu_vh": float(lim) - 5}
    rejected = {m: thr[m] for m in ("otsu_vv", "otsu_vh", "split_vv") if thr[m] > lim.get(m, -10.0)}
    for m, v in rejected.items():
        print(f"  !! {m}: threshold {v:.2f} dB is above the plausible water range ({lim.get(m):.0f} dB) — "
              f"map rejected (the histogram split is not water/land here)")
    if "split_vv" not in rejected and ntot and nsel / ntot < 0.05:
        print(f"  !! split_vv: only {nsel}/{ntot} tiles were bimodal ({100 * nsel / ntot:.1f} %) — threshold is weakly supported")

    water = {"otsu_vv": vv_db < thr["otsu_vv"], "otsu_vh": vh_db < thr["otsu_vh"], "split_vv": vv_db < thr["split_vv"]}
    water = {k: v for k, v in water.items() if k not in rejected}
    for m in rejected:                                  # a previous run may have written this map before the guard existed
        stale = out / f"flood_{m}.tif"
        if stale.exists():
            stale.unlink(); print(f"     removed stale {stale.name} from an earlier run")
    res = dict(water)

    if dry_f.exists():
        dry = read(dry_f); dvv = lee_filter(dry.values[0])
        lr = vv_db - to_db(dvv)                                  # log-ratio in dB
        res["cd_vv"] = (lr < -3.0) & (vv_db < thr["split_vv"])
        if wc is not None:
            res["urban_incr"] = (wc == 50) & (lr > 3.0)
        rioxarray_lr = ref.copy(data=lr.astype("float32"))
        rioxarray_lr.rio.to_raster(out / "logratio_vv_db.tif", driver="COG", compress="DEFLATE")

    gb, ga = d / "rtc/gee2024_before.tif", d / "rtc/gee2024_after.tif"
    if gb.exists() and ga.exists():                               # faithful 2024-GEE replica (see fetch_gee2024_inputs.py)
        b_db, a_db = to_db(refined_lee(read(gb).values[0])), to_db(refined_lee(read(ga).values[0]))   # same filter as GEE
        with np.errstate(divide="ignore", invalid="ignore"):
            res["gee2024"] = (a_db / b_db) > 1.1                  # dB/dB ratio, final threshold 1.1 as in the script
    else:                                # the replica needs the exact 2024 mosaics; approximating it would not be a replica
        print("  (no gee2024 mosaics for this event -> the 2024-rule replica is skipped)")

    px_km2 = abs(np.prod(co.rio.resolution())) / 1e6
    dmask = None
    if (anc / "districts.gpkg").exists():
        import geopandas as gpd
        from rasterio.features import geometry_mask
        g = gpd.read_file(anc / "districts.gpkg").to_crs(co.rio.crs)
        dmask = ~geometry_mask(g.geometry, out_shape=vv.shape, transform=co.rio.transform())
        print(f"  districts {list(g['shapeName'])}: {dmask.sum() * px_km2:.1f} km² (GEE GAUL AOI was 5767.41 km²)")
    rows = []
    for m, mask in res.items():
        mask = mask & valid
        raw = mask.copy()
        if m == "gee2024":                                        # script: GSW seasonality>=5, slope<5 deg, no HAND
            if seas is not None:
                mask = mask & ~(np.nan_to_num(seas, nan=0) >= 5)
            if slope is not None:
                mask = mask & ~(np.nan_to_num(slope, nan=0) >= 5)
        elif m != "urban_incr":
            mask = mask & ~perm & ~tidal & ~ex
        mask = clean(mask, 8)
        ref.copy(data=mask.astype("uint8")).rio.write_nodata(255).rio.to_raster(
            out / f"flood_{m}.tif", driver="COG", compress="DEFLATE")
        row = dict(event=ev, method=m, threshold_db=thr.get(m, np.nan), area_km2=mask.sum() * px_km2)
        if dmask is not None:
            row["area_km2_districts"] = (mask & dmask).sum() * px_km2
        # what each mask removed from the raw detection (sequential attribution: perm -> HAND -> slope)
        row["raw_km2"] = raw.sum() * px_km2
        if m not in ("gee2024", "urban_incr"):
            r = raw & ~perm; row["rm_perm_km2"] = (raw & perm).sum() * px_km2
            row["rm_tidal_km2"] = (r & tidal).sum() * px_km2; r = r & ~tidal
            if hand_ex is not None:
                row["rm_hand_km2"] = (r & hand_ex).sum() * px_km2; r = r & ~hand_ex
            if slope_ex is not None:
                row["rm_slope_km2"] = (r & slope_ex).sum() * px_km2
        if wc is not None:
            for k, name in WC.items():
                row[f"km2_{name}"] = np.sum(mask & (wc == k)) * px_km2
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "stats.csv", index=False)
    if rejected:                                   # so the report can state which methods failed here, and why
        import json
        json.dump({m: dict(threshold_db=float(v), limit_db=float(lim.get(m, -10.0)),
                           reason="global threshold above the plausible open-water range — histogram split is not water/land")
                   for m, v in rejected.items()} | ({"split_vv_support": dict(bimodal_tiles=int(nsel), tiles=int(ntot))}
                                                    if ntot else {}),
                  open(out / "rejected_methods.json", "w"), indent=1)
    print(df[["method", "threshold_db", "area_km2"] + [c for c in df if c in ("area_km2_districts", "km2_built-up", "km2_cropland", "km2_tree")]]
          .round(2).to_string(index=False))
    print("\n  mask attribution (km², sequential perm -> HAND>15 m -> slope>5°):")
    print(df[["method"] + [c for c in df if c in ("raw_km2", "rm_perm_km2", "rm_hand_km2", "rm_slope_km2", "area_km2")]]
          .round(1).to_string(index=False))
    print(f"  AOI valid area: {valid.sum() * px_km2:.0f} km²; permanent water (seas>={perm_months}): {(perm & valid).sum() * px_km2:.1f} km²;"
          f" terrain-excluded: {(ex & valid).sum() * px_km2:.0f} km²")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("event"); ap.add_argument("--perm-months", type=int, default=10)
    ap.add_argument("--co", default=None, help="co-event date tag, e.g. 20180821")
    a = ap.parse_args()
    cos = sorted((ROOT / f"data/events/{a.event}/rtc").glob("co_*.tif"))
    tags = [a.co] if a.co else [f.stem.split("_")[1] for f in cos]      # no --co -> every co-event date
    if not tags:
        sys.exit(f"no co-event scene in data/events/{a.event}/rtc")
    for t in tags:
        main(a.event, a.perm_months, t)
