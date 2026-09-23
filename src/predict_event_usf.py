"""predict_event_usf.py — apply the UrbanSARFloods 3-class models (non-flood / open flood / urban flood) to an event.

Why: every intensity rule in this project detects flooding as a *drop* in backscatter, so flooded streets — where
double bounce off buildings makes the co-event image *brighter* — are invisible to them by construction. The USF
models are the only detector here with an explicit urban-flood class.

Band stack (USF order, see train_usf.BANDS):
  0 coh pre VH | 1 coh pre VV | 2 coh co VH | 3 coh co VV | 4 int pre VH | 5 int pre VV | 6 int co VH | 7 int co VV
Intensity comes from the RTC scenes (dry-season median as the pre image unless a usable pre scene exists), coherence
from the HyP3 products. We only order VV coherence, so 'int' and 'co_int' runs work out of the box; 'all' and 'coh'
runs need VH coherence and are skipped unless --allow-missing-coh is given (missing bands are then zero-filled,
i.e. set to the training mean — report that as a caveat, it is not a neutral choice).

Outputs data/events/<event>/usf/:
  <run>_class.tif     0 non-flood, 1 open flood, 2 urban flood (uint8, 255 = nodata)
  <run>_p_urban.tif   urban-flood probability in % (uint8)
  stats.csv           km² per class, and the urban-flood share of built-up land

  python -u src/predict_event_usf.py kerala_2018 [--cpu] [--runs unet_r34_int]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, rioxarray, torch, yaml

sys.path.insert(0, str(Path(__file__).parent))
from models import build  # noqa: E402
from train_usf import BANDS, norm_stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/events.yaml"))
EVENTS = {**CFG["rural"], **CFG["urban"]}
CLASSES = {0: "non_flood", 1: "open_flood", 2: "urban_flood"}


def rd(f, band=0):
    return rioxarray.open_rasterio(f, masked=True).isel(band=band)


def to_db(lin):
    return 10 * np.log10(np.clip(lin, 1e-6, None))


@torch.no_grad()
def predict(model, x, dev, tile=512, ov=64):
    C, H, W = x.shape
    out = np.zeros((3, H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    ramp = np.ones(tile, np.float32); ramp[:ov] = np.linspace(0, 1, ov); ramp[-ov:] = np.linspace(1, 0, ov)
    win = ramp[:, None] * ramp[None, :]
    step = tile - ov
    for r in range(0, max(H - ov, 1), step):
        for c in range(0, max(W - ov, 1), step):
            r0, c0 = min(r, max(H - tile, 0)), min(c, max(W - tile, 0))
            pad = np.zeros((C, tile, tile), np.float32)
            blk = x[:, r0:r0 + tile, c0:c0 + tile]
            pad[:, :blk.shape[1], :blk.shape[2]] = blk
            p = torch.softmax(model(torch.from_numpy(pad)[None].to(dev)).float(), 1)[0].cpu().numpy()
            h, w = blk.shape[1], blk.shape[2]
            out[:, r0:r0 + h, c0:c0 + w] += p[:, :h, :w] * win[:h, :w]
            wsum[r0:r0 + h, c0:c0 + w] += win[:h, :w]
    return out / np.maximum(wsum, 1e-6)


def main(a):
    ev = a.event
    d = ROOT / f"data/events/{ev}"; out = d / "usf"; out.mkdir(parents=True, exist_ok=True)
    co_f = sorted((d / "rtc").glob("co_*.tif"))[0]; path = co_f.stem.split("_p")[-1]
    co = rioxarray.open_rasterio(co_f, masked=True)
    ref = co.isel(band=0)
    px_km2 = abs(np.prod(ref.rio.resolution())) / 1e6

    dry_f = d / f"rtc/dry_p{path}.tif"
    if not dry_f.exists():
        sys.exit(f"missing {dry_f} — the pre-event intensity bands come from the dry-season median")
    dry = rioxarray.open_rasterio(dry_f, masked=True)

    band = {}
    band[4], band[5] = to_db(dry.isel(band=1).values), to_db(dry.isel(band=0).values)      # pre VH, pre VV
    band[6], band[7] = to_db(co.isel(band=1).values), to_db(co.isel(band=0).values)        # co  VH, co  VV
    cohd = d / "coh"
    for k, nm in ((1, "pre"), (3, "co")):
        f = cohd / f"{nm}_coh.tif"
        if f.exists():
            band[k] = rd(f).rio.reproject_match(ref, resampling=5).values                  # VV coherence
    have = set(band)
    print("  bands available:", sorted(have), "(missing VH coherence: 0, 2)")

    runs = sorted((ROOT / "runs/usf").glob("*/best.pt"))
    if a.runs:
        runs = [r for r in runs if r.parent.name in a.runs.split(",")]
    if not runs:
        sys.exit("no trained USF runs in runs/usf/ — train one with src/train_usf.py --bands int")

    mean, std = norm_stats()
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    valid = np.isfinite(band[7]) & np.isfinite(band[6])
    wc_f = d / "anc/worldcover.tif"
    built = (rd(wc_f).rio.reproject_match(ref, resampling=0).values == 50) if wc_f.exists() else None
    rows = []
    for rp in runs:
        tag = rp.parent.name                                   # <model>_<bands>
        bands_key = tag.rsplit("_", 1)[-1] if tag.rsplit("_", 1)[-1] in BANDS else "all"
        idx = BANDS[bands_key]
        miss = [b for b in idx if b not in have]
        if miss and not a.allow_missing_coh:
            print(f"  {tag}: needs bands {miss} (VH coherence) — skipped"); continue
        x = np.stack([np.nan_to_num((band.get(b, np.full(ref.shape, np.nan)) - mean[b]) / std[b],
                                    nan=0.0, posinf=0.0, neginf=0.0) for b in idx]).astype(np.float32)
        model, _, _ = build(tag.rsplit("_", 1)[0], in_ch=len(idx), pretrained=False, classes=3)
        model.load_state_dict(torch.load(rp, map_location=dev)); model.to(dev).eval()
        print(f"  {tag} on {dev} ({len(idx)} bands{', zero-filled: ' + str(miss) if miss else ''})", flush=True)
        p = predict(model, x, dev, a.tile, a.overlap)
        cls = p.argmax(0).astype("uint8"); cls[~valid] = 255
        ref.copy(data=cls).rio.write_nodata(255).rio.to_raster(out / f"{tag}_class.tif", driver="COG", compress="DEFLATE")
        pu = np.where(valid, np.round(p[2] * 100), 255).astype("uint8")
        ref.copy(data=pu).rio.write_nodata(255).rio.to_raster(out / f"{tag}_p_urban.tif", driver="COG", compress="DEFLATE")
        row = dict(run=tag, bands=bands_key, zero_filled=",".join(map(str, miss)),
                   **{f"km2_{v}": float(((cls == k) & valid).sum() * px_km2) for k, v in CLASSES.items()})
        if built is not None:
            row["urban_flood_pct_of_builtup"] = float(100 * ((cls == 2) & built & valid).sum() / max((built & valid).sum(), 1))
        rows.append(row); print("   ", {k: round(v, 1) for k, v in row.items() if k.startswith("km2_")})
        del model
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    if rows:
        pd.DataFrame(rows).to_csv(out / "stats.csv", index=False)
        print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("  ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("event"); ap.add_argument("--runs", help="comma-separated run names under runs/usf/")
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--allow-missing-coh", action="store_true", dest="allow_missing_coh",
                    help="run models that need VH coherence anyway, zero-filling those bands (= training mean)")
    main(ap.parse_args())
