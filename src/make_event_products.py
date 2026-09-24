"""make_event_products.py — everything the event report needs, computed once in Python, read by R/Quarto.

Writes outputs/<event>/ :
  display/*.tif      50 m display rasters (linear averaged before dB; masks -> flooded fraction per cell)
  zoom/<name>/*.tif  10 m detail windows (raw / Lee / Refined Lee / flood maps / DL prob / coherence)
  tables/*.csv       histograms, speckle statistics, VV-VH separability, method agreement, district areas,
                     land-cover areas, reference metrics (if a reference map exists), acquisitions
  summary.json       dates, orbits, thresholds, coverage, file inventory
Run after baseline_classical.py (and optionally predict_event.py / coherence_hyp3.py).
  python -u src/make_event_products.py kerala_2018 [--factor 5]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, rioxarray, xarray as xr, yaml
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).parent))
from sarlib import lee_filter, refined_lee, to_db, otsu_threshold, metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/events.yaml"))
EVENTS = {**CFG["rural"], **CFG["urban"]}
WC = {10: "tree", 20: "shrub", 30: "grass", 40: "cropland", 50: "built-up", 60: "bare", 80: "water",
      90: "herb. wetland", 95: "mangrove", 70: "snow", 100: "moss"}


def pick_co(d, want=None):
    """Select the co-event scene and a suffix that keeps multi-date events apart.
    Returns (path, suffix) where suffix is '' for single-date events and '_<yyyymmdd>' when the event has several."""
    fs = sorted((d / "rtc").glob("co_*.tif"))
    if not fs:
        sys.exit(f"no co-event scene in {d / 'rtc'}")
    if want:
        w = str(want).replace("-", "")
        sel = [f for f in fs if w in f.stem]
        if not sel:
            sys.exit(f"co date {want} not found; available: {[f.stem.split('_')[1] for f in fs]}")
        fs_sel = sel[0]
    else:
        fs_sel = fs[0]
        if len(fs) > 1:
            print(f"  note: {len(fs)} co-event dates ({[f.stem.split('_')[1] for f in fs]}); using {fs_sel.stem.split('_')[1]} — pass --co to choose")
    return fs_sel, (f"_{fs_sel.stem.split('_')[1]}" if len(fs) > 1 else "")


def rd(f, band=None):
    da = rioxarray.open_rasterio(f, masked=True)
    return da.isel(band=band) if band is not None else da.squeeze("band", drop=True)


def save(da, f, dtype="float32"):
    f.parent.mkdir(parents=True, exist_ok=True)
    da = da.astype(dtype)
    if "band" in da.dims and da.sizes["band"] == 1:
        da = da.squeeze("band", drop=True)
    da.rio.to_raster(f, driver="COG", compress="DEFLATE")


def coarsen_lin_db(da, k):
    return 10 * np.log10(da.coarsen(x=k, y=k, boundary="trim").mean().clip(min=1e-6))


def coarsen_mean(da, k):
    return da.coarsen(x=k, y=k, boundary="trim").mean()


def window(da, lon, lat, km):
    t = Transformer.from_crs(4326, da.rio.crs, always_xy=True)
    x, y = t.transform(lon, lat); h = km * 500
    return da.sel(x=slice(x - h, x + h), y=slice(y + h, y - h))


def main(ev, k, co_date=None):
    cfg = EVENTS[ev]; d = ROOT / f"data/events/{ev}"
    co_f, sfx = pick_co(d, co_date)
    out = ROOT / f"outputs/{ev}{sfx}"
    disp, tab = out / "display", out / "tables"; tab.mkdir(parents=True, exist_ok=True)
    path = co_f.stem.split("_p")[-1]
    bdir = d / "baseline" if (d / "baseline/stats.csv").exists() else d / "baseline" / co_f.stem.split("_")[1]
    summ = {"event": ev, "tag": ev + sfx, "co_file": co_f.name, "path": int(path), "bbox": cfg["bbox"], "factor": k}
    print(f"=== {ev}: {co_f.name}, baselines in {bdir.relative_to(ROOT)}")

    # ---------------- raw SAR, full resolution ----------------
    co = rioxarray.open_rasterio(co_f, masked=True)
    vv_lin, vh_lin = co.isel(band=0), co.isel(band=1)
    ref = vv_lin
    dry = rioxarray.open_rasterio(d / f"rtc/dry_p{path}.tif", masked=True)
    pre_fs = sorted((d / "rtc").glob(f"pre_*_p{path}.tif"))
    px_km2 = abs(np.prod(co.rio.resolution())) / 1e6

    # ---------------- 50 m display rasters ----------------
    print("  display rasters")
    layers = {"co_vv_db": coarsen_lin_db(vv_lin, k), "co_vh_db": coarsen_lin_db(vh_lin, k),
              "dry_vv_db": coarsen_lin_db(dry.isel(band=0), k), "dry_vh_db": coarsen_lin_db(dry.isel(band=1), k)}
    if pre_fs:
        p = rioxarray.open_rasterio(pre_fs[0], masked=True)
        layers["pre_vv_db"] = coarsen_lin_db(p.isel(band=0), k); summ["pre_file"] = pre_fs[0].name
    for nm in ("gee2024_before", "gee2024_after"):
        if (d / f"rtc/{nm}.tif").exists():
            layers[f"{nm}_vv_db"] = coarsen_lin_db(rd(d / f"rtc/{nm}.tif"), k)
    if (bdir / "logratio_vv_db.tif").exists():
        layers["logratio_vv_db"] = coarsen_mean(rd(bdir / "logratio_vv_db.tif"), k)
    for f in sorted(bdir.glob("flood_*.tif")):
        layers[f.stem + "_frac"] = coarsen_mean(rd(f).where(lambda a: a < 255), k)
    for f in sorted((d / f"dl{sfx}").glob("*_prob.tif")) if (d / f"dl{sfx}").exists() else []:
        layers["dl_" + f.stem] = coarsen_mean(rd(f).astype("float32") / 100, k)
    for f in sorted((d / f"dl{sfx}").glob("*_flood.tif")) if (d / f"dl{sfx}").exists() else []:
        layers["dl_" + f.stem + "_frac"] = coarsen_mean(rd(f).where(lambda a: a < 255), k)
    for f in sorted((d / f"usf{sfx}").glob("*_p_urban.tif")) if (d / f"usf{sfx}").exists() else []:
        layers["usf_" + f.stem] = coarsen_mean(rd(f).where(lambda a: a < 255).astype("float32") / 100, k)
    for f in sorted((d / f"usf{sfx}").glob("*_class.tif")) if (d / f"usf{sfx}").exists() else []:
        c = rd(f).where(lambda a: a < 255)
        layers["usf_" + f.stem + "_urbanfrac"] = coarsen_mean((c == 2).astype("float32"), k)
        layers["usf_" + f.stem + "_openfrac"] = coarsen_mean((c == 1).astype("float32"), k)
    anc = d / "anc"
    for nm in ("hand", "slope", "dem", "gsw_seas", "gsw_occ"):
        if (anc / f"{nm}.tif").exists():
            layers[nm] = coarsen_mean(rd(anc / f"{nm}.tif").astype("float32"), k)
    if (anc / "worldcover.tif").exists():
        layers["worldcover"] = rd(anc / "worldcover.tif").isel(x=slice(k // 2, None, k), y=slice(k // 2, None, k))
    for f in sorted((d / "coh").glob("*_coh.tif")) if (d / "coh").exists() else []:
        c = rd(f)
        layers["coh_" + f.stem] = c.rio.reproject_match(layers["co_vv_db"], resampling=5)   # 5 = average
    for nm, da in layers.items():
        save(da, disp / f"{nm}.tif", "uint8" if nm == "worldcover" else "float32")
    summ["display_layers"] = sorted(layers)

    # ---------------- full-resolution arrays for statistics ----------------
    print("  statistics (full resolution)")
    vv_f, vh_f = lee_filter(vv_lin.values), lee_filter(vh_lin.values)
    vv_db, vh_db = to_db(vv_f), to_db(vh_f)
    valid = np.isfinite(vv_db) & np.isfinite(vh_db)
    seas = rd(anc / "gsw_seas.tif").values if (anc / "gsw_seas.tif").exists() else np.zeros(vv_db.shape)
    wc = rd(anc / "worldcover.tif").values if (anc / "worldcover.tif").exists() else np.zeros(vv_db.shape)
    hand = rd(anc / "hand.tif").values if (anc / "hand.tif").exists() else np.zeros(vv_db.shape)
    masks = {f.stem.replace("flood_", ""): (rd(f).values == 1) for f in sorted(bdir.glob("flood_*.tif"))}
    if not masks:
        sys.exit(f"no flood_*.tif in {bdir} — run: python -u src/baseline_classical.py {ev}"
                 + (f" --co {co_f.stem.split('_')[1]}" if sfx else ""))
    if (d / f"dl{sfx}").exists():
        masks.update({"dl_" + f.stem.replace("_flood", ""): (rd(f).values == 1) for f in sorted((d / f"dl{sfx}").glob("*_flood.tif"))})
    excl = list(cfg.get("exclude_methods") or [])   # stay in the per-method tables, out of consensus + agreement
    summ["excluded_methods"] = excl
    rj = bdir / "rejected_methods.json"
    summ["rejected_methods"] = json.load(open(rj)) if rj.exists() else {}
    for m in [k for k in summ["rejected_methods"] if k != "split_vv_support"]:
        masks.pop(m, None)
    summ["builtup_km2"] = float(np.sum((wc == 50) & valid) * px_km2)
    inland_water = (np.nan_to_num(seas) >= 10) & valid
    flood_ref = masks.get("cd_vv", np.zeros_like(valid))
    sound = [m for n, m in masks.items() if n in ("otsu_vv", "otsu_vh", "split_vv", "cd_vv") and n not in excl] or [flood_ref]
    land = valid & ~inland_water & ~np.isnan(seas) & ~np.logical_or.reduce(sound) & (np.nan_to_num(hand) < 15)
    rng = np.random.default_rng(0)

    def sample(a, m, n=200_000):
        v = a[m & np.isfinite(a)]
        return v if v.size <= n else rng.choice(v, n, replace=False)

    # histograms + separability of VV / VH / VV-VH per class
    bins = np.arange(-35, 10.25, 0.25); rows, sep = [], []
    feats = {"VV": vv_db, "VH": vh_db, "VV-VH": vv_db - vh_db}
    classes = {"permanent water": inland_water, "flood (change detection)": flood_ref & valid, "non-flooded land": land}
    for fn, arr in feats.items():
        smp = {cn: sample(arr, m) for cn, m in classes.items()}
        for cn, v in smp.items():
            h, _ = np.histogram(v, bins=bins, density=True)
            rows += [dict(feature=fn, cls=cn, db=b, density=x) for b, x in zip(bins[:-1] + 0.125, h)]
        a_, b_ = smp["flood (change detection)"], smp["non-flooded land"]
        if a_.size > 100 and b_.size > 100:
            m1, m2, s1, s2 = a_.mean(), b_.mean(), a_.std(), b_.std()
            bd = 0.25 * (m1 - m2) ** 2 / (s1 ** 2 + s2 ** 2) + 0.5 * np.log((s1 ** 2 + s2 ** 2) / (2 * s1 * s2))
            thr = otsu_threshold(np.concatenate([a_, b_]))
            allv = np.concatenate([a_, b_]); truth = np.r_[np.ones(a_.size, bool), np.zeros(b_.size, bool)]
            mt = max((metrics(allv < thr, truth), metrics(allv > thr, truth)), key=lambda m: m["f1"])   # either direction
            sep.append(dict(feature=fn, flood_mean=m1, flood_sd=s1, land_mean=m2, land_sd=s2, bhattacharyya=bd,
                            otsu_db=thr, f1_at_otsu=mt["f1"], iou_at_otsu=mt["iou"]))
    pd.DataFrame(rows).to_csv(tab / "hist_classes.csv", index=False)
    pd.DataFrame(sep).to_csv(tab / "vv_vh_separability.csv", index=False)

    # speckle statistics: ENL + mean preservation on homogeneous land windows (chosen on the dry-season median)
    print("  speckle statistics")
    raw = vv_lin.values; w = 31; H, W = raw.shape; cand = []
    dry_vv = dry.isel(band=0).values
    for r in range(0, H - w, w * 4):
        for c in range(0, W - w, w * 4):
            if land[r:r + w, c:c + w].mean() < 0.95:
                continue
            blk = dry_vv[r:r + w, c:c + w]
            cand.append((np.nanstd(blk) / np.nanmean(blk), r, c))
    cand = sorted(cand)[:25]
    filt = {"raw": raw, "Lee 5x5": vv_f}
    spk = []
    for name, img in filt.items():
        for cv, r, c in cand:
            blk = img[r:r + w, c:c + w]
            spk.append(dict(filter=name, enl=np.nanmean(blk) ** 2 / np.nanvar(blk), mean_db=10 * np.log10(np.nanmean(blk))))
    for cv, r, c in cand:                                        # Refined Lee on a padded window
        r0, c0 = max(r - 16, 0), max(c - 16, 0)
        blk = refined_lee(raw[r0:r + w + 16, c0:c + w + 16])[r - r0:r - r0 + w, c - c0:c - c0 + w]
        spk.append(dict(filter="Refined Lee", enl=np.nanmean(blk) ** 2 / np.nanvar(blk), mean_db=10 * np.log10(np.nanmean(blk))))
    pd.DataFrame(spk).to_csv(tab / "speckle_enl.csv", index=False)

    # final extent = majority vote of the physically sound methods (legacy GEE rule and urban candidate excluded)
    # classical voters + ONE deep-learning vote (majority of models that pass the benchmark gate), so that
    # several weak / half-trained networks cannot outvote three physically consistent classical maps
    DL_GATE = 0.60                                   # Sen1Floods11 test IoU required for a model to vote
    dl_ok, dl_out = [], {}
    for n in [n for n in masks if n.startswith("dl_") and n not in excl]:
        mj = ROOT / "runs/s1f11" / n[3:] / "metrics.json"
        iou = json.load(open(mj))["test"]["iou"] if mj.exists() else None
        if iou is None:
            dl_out[n] = "no metrics.json (training unfinished)"
        elif iou < DL_GATE:
            dl_out[n] = f"test IoU {iou:.3f} < {DL_GATE}"
        else:
            dl_ok.append(n)
    if dl_ok:
        masks["dl_ensemble"] = np.sum([masks[n] for n in dl_ok], axis=0) * 2 >= len(dl_ok)
    summ["dl_voting_models"], summ["dl_excluded"], summ["dl_gate_test_iou"] = dl_ok, dl_out, DL_GATE
    voters = [n for n in ("cd_vv", "otsu_vv", "split_vv") if n in masks] + (["dl_ensemble"] if dl_ok else [])
    if voters:
        votes = np.sum([masks[n] for n in voters], axis=0)
        masks["final_majority"] = (votes * 2 > len(voters)) & valid
        vv_lin.copy(data=masks["final_majority"].astype("uint8")).rio.write_nodata(255).rio.to_raster(
            out / "final_flood_10m.tif", driver="COG", compress="DEFLATE")
        save(coarsen_mean(vv_lin.copy(data=votes.astype("float32") / len(voters)), k), disp / "consensus_frac.tif")
        save(coarsen_mean(vv_lin.copy(data=masks["final_majority"].astype("float32")), k), disp / "final_frac.tif")
        summ["final_voters"] = voters
        summ["final_km2"] = float(masks["final_majority"].sum() * px_km2)

    # coherence by land cover / flood state (only if HyP3 products exist)
    cohf = {f.stem.replace("_coh", ""): f for f in sorted((d / "coh").glob("*_coh.tif"))} if (d / "coh").exists() else {}
    if {"pre", "co"} <= set(cohf):
        C = {k: rd(v).rio.reproject_match(vv_lin, resampling=5).values for k, v in cohf.items()}
        cpre, cco = C["pre"], C["co"]
        cdry = C.get("dry")
        fin = masks.get("final_majority", np.zeros_like(valid))
        rows = []
        WCN = {10: "tree", 20: "shrub", 30: "grass", 40: "cropland", 50: "built-up", 60: "bare", 80: "permanent water", 90: "herb. wetland"}
        groups = {f"{v} (all)": (wc == kk) for kk, v in WCN.items()}
        groups.update({"built-up, mapped flood": (wc == 50) & fin, "built-up, not flooded": (wc == 50) & ~fin,
                       "cropland, mapped flood": (wc == 40) & fin, "cropland, not flooded": (wc == 40) & ~fin,
                       "all mapped flood": fin, "all non-flood land": land})
        fin_ok = np.isfinite(cpre) & np.isfinite(cco) & (np.isfinite(cdry) if cdry is not None else True)
        for nm, g in groups.items():
            m = g & valid & fin_ok
            if m.sum() < 500:
                continue
            r = dict(group=nm, coh_pre=float(np.median(cpre[m])), coh_co=float(np.median(cco[m])),
                     delta=float(np.median(cco[m] - cpre[m])), n_px=int(m.sum()))
            if cdry is not None:
                r.update(coh_dry=float(np.median(cdry[m])), delta_vs_dry=float(np.median(cco[m] - cdry[m])))
            rows.append(r)
        if rows:
            pd.DataFrame(rows).to_csv(tab / "coherence_by_class.csv", index=False)

    # method agreement (pairwise IoU) + areas by land cover and district
    print("  agreement, land cover, districts")
    names = [n for n in masks if n != "urban_incr" and n not in excl]
    agr = [dict(a=a, b=b, iou=metrics(masks[a] & valid, masks[b] & valid)["iou"]) for a in names for b in names]
    pd.DataFrame(agr).to_csv(tab / "method_agreement.csv", index=False)
    lc = [dict(method=m, landcover=WC.get(int(c), str(int(c))), km2=np.sum(msk & (wc == c)) * px_km2)
          for m, msk in masks.items() for c in np.unique(wc[np.isfinite(wc)]) if c in WC]
    pd.DataFrame(lc).to_csv(tab / "area_by_landcover.csv", index=False)
    if (anc / "districts.gpkg").exists():
        import geopandas as gpd
        from rasterio.features import rasterize
        g = gpd.read_file(anc / "districts.gpkg").to_crs(co.rio.crs).reset_index(drop=True)
        lab = rasterize([(geom, i + 1) for i, geom in enumerate(g.geometry)], out_shape=valid.shape,
                        transform=co.rio.transform(), fill=0, dtype="uint8")
        dist = [dict(method=m, district=g.shapeName[i], km2=np.sum(msk & (lab == i + 1)) * px_km2,
                     district_km2=np.sum(lab == i + 1) * px_km2) for m, msk in masks.items() for i in range(len(g))]
        pd.DataFrame(dist).to_csv(tab / "area_by_district.csv", index=False)
        g.to_crs(4326).to_file(out / "districts.geojson")

    # optional reference map (e.g. GFM, UNOSAT, hand-digitised) -> accuracy per method
    reff = [f for f in sorted((d / "reference").glob("*.tif")) if not f.name.endswith("_geo.tif")] \
        if (d / "reference").exists() else []
    if reff:
        rf = rd(reff[0]).rio.reproject_match(ref, resampling=0).values
        rv = np.isfinite(rf) & valid
        # native 10 m comparison + a comparison aggregated to the reference's own effective cell size, because a
        # generalised print product cannot resolve 10 m detail and raw IoU therefore penalises us for being finer
        agg = max(int(round(a_ref_m / abs(ref.rio.resolution()[0]))), 1) if (a_ref_m := float(cfg.get("reference_cell_m", 90))) else 1
        rows = []
        for m, msk in masks.items():
            r = dict(method=m, reference=reff[0].name, scale="native", **metrics(msk, rf == 1, rv))
            inter = float((msk & (rf == 1) & rv).sum())
            r.update(recall_of_reference=inter / max(float(((rf == 1) & rv).sum()), 1),
                     outside_reference=float((msk & (rf != 1) & rv).sum()) / max(float((msk & rv).sum()), 1))
            rows.append(r)
            if agg > 1:                                          # majority within each reference-sized cell
                c = dict(x=agg, y=agg)
                ma = xr.DataArray(msk & rv, dims=("y", "x")).coarsen(**c, boundary="trim").mean().values > 0.5
                ra = xr.DataArray((rf == 1) & rv, dims=("y", "x")).coarsen(**c, boundary="trim").mean().values > 0.5
                va = xr.DataArray(rv, dims=("y", "x")).coarsen(**c, boundary="trim").mean().values > 0.5
                r2 = dict(method=m, reference=reff[0].name, scale=f"{int(a_ref_m)} m", **metrics(ma, ra, va))
                r2.update(recall_of_reference=float((ma & ra & va).sum()) / max(float((ra & va).sum()), 1),
                          outside_reference=float((ma & ~ra & va).sum()) / max(float((ma & va).sum()), 1))
                rows.append(r2)
        pd.DataFrame(rows).to_csv(tab / "accuracy_vs_reference.csv", index=False)
        summ["reference"] = dict(file=reff[0].name, cell_m=a_ref_m,
                                 meta=json.load(open(d / "reference/nrsc_meta.json")) if (d / "reference/nrsc_meta.json").exists() else {})

    # ---------------- zoom windows at 10 m ----------------
    print("  zoom windows")
    lin_layers = {"co_vv": vv_lin, "co_vh": vh_lin, "dry_vv": dry.isel(band=0)}
    if pre_fs:
        lin_layers["pre_vv"] = rioxarray.open_rasterio(pre_fs[0], masked=True).isel(band=0)
    from scipy import ndimage as ndi
    edge_rows = []
    fin_da = rd(out / "final_flood_10m.tif") if (out / "final_flood_10m.tif").exists() else None
    zooms_cfg = cfg.get("zooms") or {}
    if not zooms_cfg:                       # no hand-picked windows -> centre them on the largest flood clusters
        from scipy import ndimage as ndi2
        src_mask = masks.get("final_majority", masks.get("cd_vv"))
        if src_mask is not None and src_mask.any():
            small = src_mask[::5, ::5]
            lab, n = ndi2.label(small)
            if n:
                sizes = np.bincount(lab.ravel())[1:]
                order = np.argsort(sizes)[::-1][:3]
                tr_back = Transformer.from_crs(vv_lin.rio.crs, 4326, always_xy=True)
                for rank, ci in enumerate(order, 1):
                    cy, cx = ndi2.center_of_mass(lab == ci + 1)
                    xs = float(vv_lin.x.values[min(int(cx * 5), vv_lin.sizes["x"] - 1)])
                    ys = float(vv_lin.y.values[min(int(cy * 5), vv_lin.sizes["y"] - 1)])
                    lon, lat = tr_back.transform(xs, ys)
                    zooms_cfg[f"flood_cluster_{rank}"] = [round(lon, 4), round(lat, 4), 6]
                summ["zooms_auto"] = True
                print("  zoom windows derived from the largest flood clusters:", list(zooms_cfg))
    summ["zooms"] = zooms_cfg
    for zn, (lon, lat, km) in zooms_cfg.items():
        zd = out / "zoom" / zn
        for nm, da in lin_layers.items():
            wdw = window(da, lon, lat, km)
            save(10 * np.log10(wdw.clip(min=1e-6)), zd / f"{nm}_raw_db.tif")
            if nm == "co_vv":
                arr = wdw.values
                flt = {"raw": arr, "Lee 5x5": lee_filter(arr), "Refined Lee": refined_lee(arr)}
                save(wdw.copy(data=to_db(flt["Lee 5x5"])), zd / "co_vv_lee_db.tif")
                save(wdw.copy(data=to_db(flt["Refined Lee"])), zd / "co_vv_refinedlee_db.tif")
                # edge preservation: dB contrast across water/land boundaries of the final map, 2-px bands each side.
                # Raw contrast is unbiased (speckle averages out over the band); smoothing blurs the step -> lower contrast.
                if fin_da is not None:
                    fm = window(fin_da, lon, lat, km).values == 1
                    if fm.shape == arr.shape and fm.sum() > 500:
                        wat = fm & ~ndi.binary_erosion(fm, iterations=2)
                        lnd = ndi.binary_dilation(fm, iterations=2) & ~fm
                        for fn, a in flt.items():
                            db = to_db(a)
                            edge_rows.append(dict(zoom=zn, filter=fn, land_side_db=np.nanmean(db[lnd]),
                                                  water_side_db=np.nanmean(db[wat]), contrast_db=np.nanmean(db[lnd]) - np.nanmean(db[wat]),
                                                  n_edge_px=int(wat.sum() + lnd.sum())))
        extra = [out / "final_flood_10m.tif"] if (out / "final_flood_10m.tif").exists() else []
        for f in sorted(bdir.glob("flood_*.tif")) + (sorted((d / f"dl{sfx}").glob("*_prob.tif")) if (d / f"dl{sfx}").exists() else []) + extra:
            save(window(rd(f), lon, lat, km), zd / f.name, "uint8")
        for f in (sorted((d / "coh").glob("*_coh.tif")) if (d / "coh").exists() else []):
            save(window(rd(f), lon, lat, km), zd / f.name)
        if (anc / "worldcover.tif").exists():
            save(window(rd(anc / "worldcover.tif"), lon, lat, km), zd / "worldcover.tif", "uint8")

    if edge_rows:
        e = pd.DataFrame(edge_rows)
        e["retention"] = e.contrast_db / e.groupby("zoom").contrast_db.transform(lambda x: x[e.loc[x.index, "filter"] == "raw"].iloc[0])
        e.to_csv(tab / "speckle_edges.csv", index=False)

    # ---------------- acquisitions + summary ----------------
    acq = ROOT / "outputs/audit/acquisitions.csv"
    if acq.exists():
        a = pd.read_csv(acq); a[a.event == ev].to_csv(tab / "acquisitions.csv", index=False)
    st = bdir / "stats.csv"
    if st.exists():
        pd.read_csv(st).to_csv(tab / "baseline_stats.csv", index=False)
    if (d / f"dl{sfx}/stats.csv").exists():
        pd.read_csv(d / f"dl{sfx}/stats.csv").to_csv(tab / "dl_stats.csv", index=False)
    if (d / f"usf{sfx}/stats.csv").exists():
        pd.read_csv(d / f"usf{sfx}/stats.csv").to_csv(tab / "usf_event.csv", index=False)
    runs = []
    for mj in sorted((ROOT / "runs/s1f11").glob("*/metrics.json")):
        m = json.load(open(mj))
        for split in ("valid", "test", "bolivia"):
            runs.append(dict(model=m["model"], params_M=m["params_M"], split=split, **{k: m[split][k] for k in
                        ("iou", "f1", "precision", "recall", "kappa")}))
    if runs:
        pd.DataFrame(runs).to_csv(tab / "s1f11_metrics.csv", index=False)
    hist = [pd.read_csv(h).assign(model=h.parent.name) for h in sorted((ROOT / "runs/s1f11").glob("*/history.csv"))]
    if hist:
        pd.concat(hist).to_csv(tab / "s1f11_history.csv", index=False)
    usf = []
    for mj in sorted((ROOT / "runs/usf").glob("*/metrics.json")):
        m = json.load(open(mj))["valid_full"]
        for c in ("open_flood", "urban_flood"):
            usf.append(dict(run=mj.parent.name, cls=c, **m[c]))
    if usf:
        pd.DataFrame(usf).to_csv(tab / "usf_metrics.csv", index=False)
    for extra in ("band_by_class.csv",):
        f = ROOT / "data/benchmarks/usf" / extra
        if f.exists():
            pd.read_csv(f).to_csv(tab / f"usf_{extra}", index=False)
    summ.update(valid_km2=float(valid.sum() * px_km2), published=cfg.get("published", []),
                zoom_labels=cfg.get("zoom_labels", {}), setting=cfg.get("setting", ""), co_dates=[str(c["date"]) for c in cfg["co"]])
    json.dump(summ, open(out / "summary.json", "w"), indent=1, default=str)
    print(f"  done -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("event"); ap.add_argument("--factor", type=int, default=5)
    ap.add_argument("--co", help="co-event date (YYYY-MM-DD) for events with more than one")
    a = ap.parse_args(); main(a.event, a.factor, a.co)
