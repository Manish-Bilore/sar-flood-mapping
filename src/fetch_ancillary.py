"""fetch_ancillary.py — unified ancillary layers on the event's RTC grid (10 m, UTM).

Writes data/events/<event>/anc/:
  dem.tif         Copernicus GLO-30 (PC cop-dem-glo-30), bilinear to 10 m
  slope.tif       degrees, computed on the 30 m DEM then resampled
  hand.tif        Height Above Nearest Drainage, pysheds on the 30 m DEM (stream = acc > 1 km²)
  gsw_occ.tif     JRC GSW occurrence (%)          PC jrc-gsw
  gsw_seas.tif    JRC GSW seasonality (months)    PC jrc-gsw
  worldcover.tif  ESA WorldCover 2021 class       PC esa-worldcover
All open collections on Planetary Computer (no subscription key needed).

Usage: python -u src/fetch_ancillary.py [event ...]
"""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
import numpy as np, rioxarray, xarray as xr, yaml
import odc.geo.xr  # noqa: F401  (.odc accessor)
import odc.stac, planetary_computer as pc, pystac_client
from odc.geo.geobox import GeoBox

sys.path.insert(0, str(Path(__file__).parent))
from sarlib import slope_deg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/events.yaml"))
EVENTS = {**CFG["rural"], **CFG["urban"]}
cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=pc.sign_inplace)

def ref_geobox(ev):
    f = sorted((ROOT / f"data/events/{ev}/rtc").glob("co_*.tif"))[0]
    return rioxarray.open_rasterio(f, chunks={}).odc.geobox

def load(coll, bands, bbox, geobox, resampling, dtype, query=None):
    items = cat.search(collections=[coll], bbox=bbox, query=query or {}).item_collection()
    if not items:
        raise RuntimeError(f"no {coll} items for {bbox}")
    # static products: all tiles share one datetime -> solar_day grouping mosaics them (first valid pixel wins)
    ds = odc.stac.load(items, bands=bands, geobox=geobox, resampling=resampling, dtype=dtype, groupby="solar_day")
    return ds.isel(time=0, drop=True)

def save(da, f, nodata=None):
    if nodata is not None:
        da = da.rio.write_nodata(nodata)
    da.rio.to_raster(f, driver="COG", compress="DEFLATE")
    print(f"    {f.name:15s} {f.stat().st_size / 2**20:6.1f} MB")

def hand_30m(dem30: xr.DataArray) -> xr.DataArray:
    if not hasattr(np, "in1d"):          # pysheds still calls np.in1d, removed in NumPy 2.4
        np.in1d = np.isin
    from pysheds.grid import Grid
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dem.tif"
        dem30.rio.write_nodata(-9999.0).rio.to_raster(p)
        g = Grid.from_raster(str(p)); d = g.read_raster(str(p))
        f = g.resolve_flats(g.fill_depressions(g.fill_pits(d)))
        fdir = g.flowdir(f)
        acc = g.accumulation(fdir)
        px_km2 = abs(dem30.rio.resolution()[0] * dem30.rio.resolution()[1]) / 1e6
        hand = g.compute_hand(fdir, d, acc > (1.0 / px_km2))
    return dem30.copy(data=np.asarray(hand, dtype="float32"))

def districts(ev, names):
    """Clip polygons for the named districts (geoBoundaries gbOpen IND ADM2, simplified) -> anc/districts.gpkg"""
    import json, urllib.request, geopandas as gpd
    f = ROOT / "data/boundaries/IND_ADM2_simplified.geojson"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        meta = json.load(urllib.request.urlopen("https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/"))
        urllib.request.urlretrieve(meta["simplifiedGeometryGeoJSON"], f)
    g = gpd.read_file(f)
    if isinstance(names, str) and names == "auto" or names == ["auto"]:
        from shapely.geometry import box
        w, s_, e, n = EVENTS[ev]["bbox"]
        sel = g[g.intersects(box(w, s_, e, n))]
        print(f"    districts: auto -> {sorted(sel['shapeName'])}")
        sel.to_file(ROOT / f"data/events/{ev}/anc/districts.gpkg")
        return
    sel = g[g["shapeName"].str.lower().isin([n.lower() for n in names])]
    missing = set(n.lower() for n in names) - set(sel["shapeName"].str.lower())
    if missing:
        print(f"    !! district names not found: {missing}; candidates:",
              sorted(g[g["shapeName"].str.lower().str.contains("|".join(missing))]["shapeName"]))
    sel.to_file(ROOT / f"data/events/{ev}/anc/districts.gpkg")
    area = sel.to_crs(sel.estimate_utm_crs()).area.sum() / 1e6
    print(f"    districts.gpkg  {list(sel['shapeName'])}  {area:.1f} km²")

def run(ev):
    cfg = EVENTS[ev]; bbox = cfg["bbox"]
    out = ROOT / f"data/events/{ev}/anc"; out.mkdir(parents=True, exist_ok=True)
    gb10 = ref_geobox(ev)
    gb30 = GeoBox.from_bbox(gb10.boundingbox, crs=gb10.crs, resolution=30)
    print(f"\n=== {ev}  grid {gb10.shape} @10 m, {gb30.shape} @30 m")

    if not (out / "hand.tif").exists():
        dem30 = load("cop-dem-glo-30", ["data"], bbox, gb30, "bilinear", "float32")["data"]
        dem30 = dem30.rio.write_crs(gb30.crs)
        slope30 = dem30.copy(data=slope_deg(dem30.values, 30.0))
        hand30 = hand_30m(dem30)
        for name, da in [("dem", dem30), ("slope", slope30), ("hand", hand30)]:
            da10 = da.odc.reproject(gb10, resampling="bilinear")
            save(da10.astype("float32"), out / f"{name}.tif")

    if not (out / "gsw_seas.tif").exists():
        g = load("jrc-gsw", ["occurrence", "seasonality"], bbox, gb10, "nearest", "uint8")
        save(g["occurrence"], out / "gsw_occ.tif", nodata=255)
        save(g["seasonality"], out / "gsw_seas.tif", nodata=255)

    if cfg.get("districts") and not (out / "districts.gpkg").exists():
        districts(ev, cfg["districts"])

    if not (out / "worldcover.tif").exists():
        wc = load("esa-worldcover", ["map"], bbox, gb10, "nearest", "uint8",
                  query={"esa_worldcover:product_version": {"eq": "2.0.0"}})["map"]
        save(wc, out / "worldcover.tif", nodata=0)

if __name__ == "__main__":
    for ev in (sys.argv[1:] or EVENTS):
        run(ev)
