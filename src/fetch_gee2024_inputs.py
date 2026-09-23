"""fetch_gee2024_inputs.py — rebuild the exact inputs of the 2024 GEE Kerala script, for a faithful replica.

GEE logic reproduced:
  S1 IW, VV, DESCENDING (any relative orbit), 10 m
  before = mosaic(2018-07-15 .. 2018-08-10)   after = mosaic(2018-08-10 .. 2018-08-23)
  ee.ImageCollection.mosaic() -> most recent image on top, gaps filled by older images
Writes data/events/kerala_2018/rtc/gee2024_before.tif, gee2024_after.tif (VV, linear gamma0) + manifest.

Note: GEE's COPERNICUS/S1_GRD is sigma0 (ellipsoid-corrected, no terrain flattening); PC RTC is gamma0-terrain.
Values differ by the terrain/incidence normalisation; the replica keeps every other step identical.
"""
import json, sys
import numpy as np
from pathlib import Path
import rioxarray, odc.stac, planetary_computer as pc, pystac_client
import odc.geo.xr  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
EV = "kerala_2018"
WINDOWS = {"before": ("2018-07-15", "2018-08-09T23:59:59Z"), "after": ("2018-08-10", "2018-08-22T23:59:59Z")}
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

ref = sorted((ROOT / f"data/events/{EV}/rtc").glob("co_*.tif"))[0]
gbox = rioxarray.open_rasterio(ref, chunks={}).odc.geobox
bbox = list(gbox.geographic_extent.boundingbox)
man = {}
for name, (t0, t1) in WINDOWS.items():
    f = ROOT / f"data/events/{EV}/rtc/gee2024_{name}.tif"
    items = [i for i in cat.search(collections=["sentinel-1-rtc"], bbox=bbox, datetime=f"{t0}/{t1}").item_collection()
             if i.properties.get("sat:orbit_state", "").lower() == "descending"]
    items = sorted(items, key=lambda i: i.datetime)
    man[name] = [(i.id, str(i.datetime.date()), i.properties.get("sat:relative_orbit")) for i in items]
    print(f"{name}: {len(items)} DESC items", sorted({(d, p) for _, d, p in man[name]}))
    if f.exists():
        continue
    ds = odc.stac.load(items, bands=["vv"], geobox=gbox, groupby="solar_day", chunks={"x": 2048, "y": 2048},
                       resampling="bilinear", dtype="float32")
    vv = ds["vv"]
    # ee mosaic(): latest valid pixel on top. Plain numpy loop (xarray ffill needs 'bottleneck').
    mos = None
    for t in range(vv.sizes["time"]):                 # time is ascending -> later dates overwrite
        a = vv.isel(time=t).values
        a = np.where(a > 0, a, np.nan).astype("float32")
        mos = a if mos is None else np.where(np.isfinite(a), a, mos)
    mosaic = vv.isel(time=0, drop=True).copy(data=mos)
    cov = float(np.isfinite(mos).mean()) * 100
    mosaic.rio.to_raster(f, driver="COG", compress="DEFLATE", predictor=3)
    print(f"  wrote {f.name}  coverage {cov:.1f} %")
json.dump(man, open(ROOT / f"data/events/{EV}/rtc/gee2024_manifest.json", "w"), indent=1)
