"""fetch_event_rtc.py — AOI-clipped Sentinel-1 RTC (linear gamma0, VV+VH, 10 m) per event from Planetary Computer.

Per event writes to data/events/<event>/rtc/:
  dry_p<path>.tif        same-path dry-season MEDIAN (<= 4 scenes, linear)   -> change-detection baseline
  pre_<date>_p<path>.tif last same-path acquisition before co-event          -> urban pre-event
  co_<date>_p<path>.tif  co-event scene(s) listed in config/events.yaml
  manifest.json          STAC item ids used (provenance for the Quarto doc)

sentinel-1-rtc needs a Planetary Computer subscription key (env PC_SDK_SUBSCRIPTION_KEY).
The script tests access first and stops with a clear message if the key is missing.

Usage: python -u src/fetch_event_rtc.py [event ...]      (default: all events)
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
import pystac_client, planetary_computer as pc, odc.stac, rioxarray  # noqa: F401
from pyproj import CRS

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/events.yaml"))
EVENTS = {**CFG["rural"], **CFG["urban"]}
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

def access_test():
    it = next(cat.search(collections=["sentinel-1-rtc"], bbox=[76.3, 9.4, 76.4, 9.5], datetime="2018-08-21", max_items=1).items())
    import rasterio
    try:
        with rasterio.open(it.assets["vv"].href) as src:
            src.read(1, window=((0, 8), (0, 8)))
        print("PC sentinel-1-rtc access OK")
    except Exception as e:
        sys.exit(f"\nPC sentinel-1-rtc NOT readable ({type(e).__name__}). Set PC_SDK_SUBSCRIPTION_KEY "
                 f"(key present: {bool(os.getenv('PC_SDK_SUBSCRIPTION_KEY'))}) or tell Claude to switch to the CDSE openEO backend.")

def utm(bbox):
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return CRS.from_dict({"proj": "utm", "zone": int((lon + 180) // 6) + 1, "south": lat < 0}).to_epsg()

def search(bbox, start, end, path):
    items = cat.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{start}/{end}").item_collection()
    return [i for i in items if int(i.properties.get("sat:relative_orbit", -1)) == int(path)]

def load(items, bbox, epsg):
    ds = odc.stac.load(items, bands=["vv", "vh"], bbox=bbox, crs=f"EPSG:{epsg}", resolution=10,
                       groupby="solar_day", chunks={"x": 2048, "y": 2048}, resampling="bilinear", dtype="float32")
    return ds.where(ds > 0)          # RTC nodata/zeros -> NaN; values stay LINEAR gamma0

def write(da, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    da.to_array("band").rio.to_raster(path, driver="COG", compress="DEFLATE", predictor=3)
    print(f"    wrote {path.relative_to(ROOT)}  {path.stat().st_size/2**20:.0f} MB")

def run(ev, cfg):
    print(f"\n=== {ev}")
    bbox, epsg = cfg["bbox"], utm(cfg["bbox"])
    out, man = ROOT / f"data/events/{ev}/rtc", {"epsg": epsg, "bbox": bbox}
    for c in cfg["co"]:
        d, p = pd.Timestamp(c["date"]), c["path"]
        co = search(bbox, d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d"), p)
        if not co:
            print(f"  !! no RTC for {d.date()} path {p}"); continue
        f = out / f"co_{d:%Y%m%d}_p{p}.tif"
        if not f.exists(): write(load(co, bbox, epsg).isel(time=0), f)
        pre = search(bbox, (d - pd.Timedelta(days=40)).strftime("%Y-%m-%d"), (d - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), p)
        pre = sorted(pre, key=lambda i: i.datetime)[-1:] if pre else []
        if pre:
            f = out / f"pre_{pre[0].datetime:%Y%m%d}_p{p}.tif"
            if not f.exists(): write(load(pre, bbox, epsg).isel(time=0), f)
        dry = sorted(search(bbox, *map(str, cfg["dry"]), p), key=lambda i: i.datetime)
        dry = [dry[k] for k in np.linspace(0, len(dry) - 1, min(4, len(dry))).round().astype(int)] if dry else []
        f = out / f"dry_p{p}.tif"
        if dry and not f.exists():
            write(load(dry, bbox, epsg).median("time", skipna=True), f)   # median in LINEAR units
        man[f"p{p}_{d:%Y%m%d}"] = {"co": [i.id for i in co], "pre": [i.id for i in pre], "dry": [i.id for i in dry]}
    out.mkdir(parents=True, exist_ok=True)
    json.dump(man, open(out / "manifest.json", "w"), indent=1)

if __name__ == "__main__":
    access_test()
    for ev in (sys.argv[1:] or EVENTS):
        run(ev, EVENTS[ev])
