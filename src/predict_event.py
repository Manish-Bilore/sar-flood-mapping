"""predict_event.py — apply every trained Sen1Floods11 model to an event's co-event RTC scene.

For each runs/s1f11/<model>/best.pt:
  data/events/<ev>/dl/<model>_prob.tif    water probability 0-100 (uint8, 255 = nodata)
  data/events/<ev>/dl/<model>_flood.tif   flood = water & not permanent water & HAND<=15 m & slope<=5 deg, cleaned (<8 px)
  data/events/<ev>/dl/stats.csv           km² per model (+ districts, land cover)
Input identical to training (s1data.linear_to_input: VV, VH, VV-VH dB, no speckle filter — Sen1Floods11 chips are unfiltered).
Domain note: training chips are sigma0 (GEE GRD), events are gamma0 RTC -> report, don't hide.

  python -u src/predict_event.py kerala_2018 [--models unet_r18,unet_r34] [--cpu] [--tile 512 --overlap 64]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, rioxarray, torch

sys.path.insert(0, str(Path(__file__).parent))
from models import build                         # noqa: E402
from s1data import linear_to_input               # noqa: E402
from sarlib import exclusion_mask, clean         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WC = {10: "tree", 20: "shrub", 30: "grass", 40: "cropland", 50: "built-up", 60: "bare", 80: "water", 90: "herb. wetland", 95: "mangrove"}


@torch.no_grad()
def predict(model, x, dev, tile, ov, amp):
    """Sliding-window inference with linear blending weights. x: (3,H,W) float32 -> prob (H,W)."""
    C, H, W = x.shape; st = tile - ov
    ramp = np.minimum(np.arange(tile) + 1, np.arange(tile)[::-1] + 1).astype(np.float32)
    wgt = np.minimum.outer(np.minimum(ramp, ov), np.minimum(ramp, ov)); wgt /= wgt.max()
    acc = np.zeros((H, W), np.float32); den = np.zeros((H, W), np.float32)
    rows = list(range(0, max(H - tile, 0) + 1, st)) + ([H - tile] if H > tile and (H - tile) % st else [])
    cols = list(range(0, max(W - tile, 0) + 1, st)) + ([W - tile] if W > tile and (W - tile) % st else [])
    n = len(rows) * len(cols)
    for i, r in enumerate(rows):
        for c in cols:
            blk = x[:, r:r + tile, c:c + tile]
            if not blk.any():
                continue
            h, w = blk.shape[1:]
            pad = np.zeros((C, tile, tile), np.float32); pad[:, :h, :w] = blk
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
                p = torch.sigmoid(model(torch.from_numpy(pad)[None].to(dev)).float())[0, 0].cpu().numpy()
            acc[r:r + h, c:c + w] += p[:h, :w] * wgt[:h, :w]; den[r:r + h, c:c + w] += wgt[:h, :w]
        if i % max(len(rows) // 10, 1) == 0:
            print(f"    row {i + 1}/{len(rows)} ({n} tiles)", flush=True)
    return np.where(den > 0, acc / np.maximum(den, 1e-6), np.nan)


def main(a):
    d = ROOT / f"data/events/{a.event}"
    from make_event_products import pick_co
    co_f, sfx = pick_co(d, a.co)
    out = d / f"dl{sfx}"; out.mkdir(exist_ok=True)
    co = rioxarray.open_rasterio(co_f, masked=True); ref = co.isel(band=0)
    x, valid = linear_to_input(co.values[0], co.values[1])
    anc = d / "anc"
    rdv = lambda n: rioxarray.open_rasterio(anc / f"{n}.tif", masked=True).values[0] if (anc / f"{n}.tif").exists() else None
    hand, slope, seas, wc = rdv("hand"), rdv("slope"), rdv("gsw_seas"), rdv("worldcover")
    ex = exclusion_mask(hand, slope); ex = np.zeros_like(valid) if ex is None else ex
    perm = (np.nan_to_num(seas, nan=0) >= 10) if seas is not None else np.zeros_like(valid)
    if seas is not None and wc is not None:
        perm |= np.isnan(seas) & (wc == 80)
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    px_km2 = abs(np.prod(co.rio.resolution())) / 1e6
    runs = sorted((ROOT / "runs/s1f11").glob("*/best.pt"))
    if a.models:
        runs = [r for r in runs if r.parent.name in a.models.split(",")]
    rows = []
    for rp in runs:
        name = rp.parent.name
        if (out / f"{name}_prob.tif").exists() and not a.overwrite:
            print(f"  {name}: exists, skipping (use --overwrite)"); prob = None
        else:
            print(f"  {name} on {dev}", flush=True)
            model, _, amp = build(name, pretrained=False)
            try:                                              # the 2 GB GPU is often busy with training -> fall back
                model.load_state_dict(torch.load(rp, map_location=dev)); model.to(dev).eval()
                prob = predict(model, x, dev, a.tile, a.overlap, amp)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if dev.type != "cuda" or "out of memory" not in str(e).lower():
                    raise
                print(f"    GPU out of memory ({e.__class__.__name__}) -> running {name} on CPU", flush=True)
                torch.cuda.empty_cache(); dev = torch.device("cpu"); amp = False
                model, _, _ = build(name, pretrained=False)
                model.load_state_dict(torch.load(rp, map_location=dev)); model.to(dev).eval()
                prob = predict(model, x, dev, a.tile, a.overlap, amp)
            prob[~valid] = np.nan
            ref.copy(data=np.where(np.isfinite(prob), np.round(prob * 100), 255).astype("uint8")).rio.write_nodata(255) \
               .rio.to_raster(out / f"{name}_prob.tif", driver="COG", compress="DEFLATE")
            del model; torch.cuda.empty_cache() if dev.type == "cuda" else None
        if prob is None:
            p = rioxarray.open_rasterio(out / f"{name}_prob.tif").values[0]
            prob = np.where(p == 255, np.nan, p / 100.0)
        water = np.nan_to_num(prob) > 0.5
        flood = clean(water & valid & ~perm & ~ex, 8)
        ref.copy(data=flood.astype("uint8")).rio.write_nodata(255).rio.to_raster(out / f"{name}_flood.tif", driver="COG", compress="DEFLATE")
        row = dict(model=name, water_km2=(water & valid).sum() * px_km2, flood_km2=flood.sum() * px_km2)
        if wc is not None:
            row.update({f"km2_{v}": np.sum(flood & (wc == k)) * px_km2 for k, v in WC.items()})
        rows.append(row); print(f"    water {row['water_km2']:.1f} km², flood {row['flood_km2']:.1f} km²")
    pd.DataFrame(rows).to_csv(out / "stats.csv", index=False)
    print(pd.DataFrame(rows)[["model", "water_km2", "flood_km2"]].round(1).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("event"); ap.add_argument("--models", default=None); ap.add_argument("--co", default=None)
    ap.add_argument("--tile", type=int, default=512); ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
