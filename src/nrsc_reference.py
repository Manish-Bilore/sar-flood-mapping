"""nrsc_reference.py — turn an NRSC/NDEM flood map PDF into a georeferenced reference raster.

The NDEM sheets carry a lat/lon graticule whose tick labels are real text, so the georeferencing needs no manual
clicking: pdftotext -bbox gives each label's position, those become GCPs, and a first-order least-squares fit
turns them into the sheet's affine transform (the sheets are north-up in lat/lon, so no resampling is needed). The inundation
class is then picked out by its legend swatch colour (sampled next to the words "Flood Inundation" in the legend).

Outputs (data/events/<event>/reference/):
  nrsc_map_geo.tif      georeferenced RGB sheet (for the overlay figure)
  nrsc_flood.tif        1 = flood inundation, 0 = other, on the event's own 10 m grid
  nrsc_meta.json        sensor / date / GCP residuals / legend colour / source URL

  python -u src/nrsc_reference.py kerala_2018 \\
      --url https://ndem.nrsc.gov.in/documents/Disaster_Document/2018/KL/klflood50dsc21082018_1830hrs/klflood50dsc21082018_1830hrs_map.pdf

Caveats carried into the report: the sheet is a 50 m product generalised for print (≈90 m effective at 300 dpi),
roads/settlement symbols overprint the polygons, and "Normal river/water bodies" is a separate class that is NOT
merged into the flood class here.
"""
from __future__ import annotations
import argparse, json, re, shutil, subprocess, sys
from pathlib import Path
import warnings
import numpy as np, pandas as pd, rioxarray, yaml
from xml.etree import ElementTree as ET
from affine import Affine

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/events.yaml"))
EVENTS = {**CFG["rural"], **CFG["urban"]}
DEG = re.compile(r"^(\d{1,3})\s*°\s*(\d{1,2})?'?\s*(\d{1,2})?\"?\s*([NSEW])$")


def need(*tools):
    miss = [t for t in tools if shutil.which(t) is None]
    if miss:
        sys.exit(f"missing command(s): {', '.join(miss)}  (conda install -c conda-forge poppler)")


def affine_from_gcps(gcps):
    """Least-squares first-order fit (pixel, line) -> (lon, lat). Returns the affine and the RMS residual in metres."""
    P = np.array([[px, py, 1.0] for px, py, _, _ in gcps])
    L = np.array([[lon, lat] for _, _, lon, lat in gcps])
    coef, *_ = np.linalg.lstsq(P, L, rcond=None)
    pred = P @ coef
    res = pred - L
    rms_m = float(np.sqrt((res ** 2).sum(1).mean()) * 111_320)
    (a, d), (b, e), (c, f) = coef                       # lon = a*px + b*py + c ; lat = d*px + e*py + f
    return Affine(a, b, c, d, e, f), rms_m


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        sys.exit(f"{cmd[0]} failed:\n{p.stderr[-2000:]}")
    return p.stdout


def dms(txt):
    """'76°30'0"E' -> 76.5 (signed)."""
    m = DEG.match(txt.replace("’", "'").replace("″", '"').replace("''", '"').strip())
    if not m:
        return None
    d, mi, se, h = m.groups()
    v = int(d) + int(mi or 0) / 60 + int(se or 0) / 3600
    return dict(N=v, E=v, S=-v, W=-v)[h]


def text_boxes(pdf):
    """[(text, x_pt, y_pt, page_w, page_h)] from pdftotext -bbox."""
    xml = run(["pdftotext", "-bbox", "-f", "1", "-l", "1", str(pdf), "-"])
    root = ET.fromstring(xml)
    ns = {"x": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
    page = root.find(".//x:page", ns) if ns else root.find(".//page")
    pw, ph = float(page.get("width")), float(page.get("height"))
    out = []
    for w in (page.iter("{%s}word" % ns["x"]) if ns else page.iter("word")):
        t = (w.text or "").strip()
        if t:
            out.append((t, (float(w.get("xMin")) + float(w.get("xMax"))) / 2,
                        (float(w.get("yMin")) + float(w.get("yMax"))) / 2, pw, ph))
    return out


def graticule_gcps(boxes, dpi, pw, ph, margin=0.12):
    """Tick labels in the sheet margins -> GCPs (pixel/line -> lon/lat). Words may be split, so join neighbours."""
    joined = []
    for t, x, y, *_ in boxes:
        if joined and abs(joined[-1][1] - x) < 28 and abs(joined[-1][2] - y) < 4:
            joined[-1] = (joined[-1][0] + t, (joined[-1][1] + x) / 2, joined[-1][2])
        else:
            joined.append((t, x, y))
    lons, lats = [], []
    for t, x, y in joined:
        v = dms(t)
        if v is None:
            continue
        sx, sy = dpi / 72.0, dpi / 72.0
        if t[-1] in "EW" and (y < ph * margin or y > ph * (1 - margin)):         # top / bottom margin -> longitude
            lons.append((x * sx, v))
        elif t[-1] in "NS" and (x < pw * margin or x > pw * (1 - margin)):       # left / right margin -> latitude
            lats.append((y * sy, v))
    # one GCP per (lon tick, lat tick) pair, de-duplicated by value
    lons = {round(v, 6): p for p, v in lons}            # value -> pixel column
    lats = {round(v, 6): p for p, v in lats}            # value -> pixel row
    return ([(px, py, lon, lat) for lon, px in sorted(lons.items()) for lat, py in sorted(lats.items())],
            len(lons), len(lats))


def main(a):
    warnings.filterwarnings("ignore", message=".*no geotransform.*")
    need("pdftotext", "pdftoppm")
    ev = a.event
    out = ROOT / f"data/events/{ev}/reference"; out.mkdir(parents=True, exist_ok=True)
    pdf = out / "nrsc_map.pdf"
    if a.pdf:
        shutil.copy(a.pdf, pdf)
    elif not pdf.exists():
        import urllib.request
        print("  downloading", a.url); urllib.request.urlretrieve(a.url, pdf)

    boxes = text_boxes(pdf)
    pw, ph = boxes[0][3], boxes[0][4]
    png = out / "nrsc_map.png"
    if not png.exists() or a.overwrite:
        run(["pdftoppm", "-r", str(a.dpi), "-png", "-f", "1", "-l", "1", "-singlefile", str(pdf), str(png.with_suffix(""))])

    gcps, nlon, nlat = graticule_gcps(boxes, a.dpi, pw, ph)
    print(f"  graticule: {nlon} longitude ticks x {nlat} latitude ticks -> {len(gcps)} GCPs")
    if len(gcps) < 4:
        sys.exit("could not read enough graticule labels — check `pdftotext -bbox` output, or georeference by hand")

    # first-order fit; the sheet is north-up in lat/lon, so the affine can be applied directly (no resampling)
    aff, rms_m = affine_from_gcps(gcps)
    print(f"  affine fit: RMS residual {rms_m:.0f} m over {len(gcps)} GCPs")
    if rms_m > a.max_rms_m:
        sys.exit(f"georeferencing residual {rms_m:.0f} m exceeds --max-rms-m {a.max_rms_m} — check the tick labels")
    rgb = rioxarray.open_rasterio(png)
    rgb = rgb.rio.write_crs("EPSG:4326").rio.write_transform(aff)
    rgb = rgb.assign_coords(x=aff.c + aff.a * (np.arange(rgb.sizes["x"]) + 0.5) + aff.b * 0,
                            y=aff.f + aff.e * (np.arange(rgb.sizes["y"]) + 0.5) + aff.d * 0)
    geo = out / "nrsc_map_geo.tif"
    rgb.astype("uint8").rio.to_raster(geo, driver="COG", compress="DEFLATE")

    # legend swatch colour: the filled rectangle immediately left of the legend line "Flood Inundation"
    col = np.array([int(c) for c in a.colour.split(",")], float) if a.colour else None
    if col is None:
        hits = [b for b in boxes if b[0].lower().startswith("inundat")] or [b for b in boxes if b[0].lower().startswith("flood")]
        if not hits:
            sys.exit("legend text not found — pass --colour R,G,B (read it off the legend in any image viewer)")
        _, hx, hy, *_ = hits[0]
        line = [b for b in boxes if abs(b[2] - hy) < 4 and hx - 150 < b[1] <= hx + 1
                and dms(b[0]) is None]                                            # the label phrase, not a graticule tick
        x0 = min(b[1] for b in line)
        sc = a.dpi / 72.0
        raw = rioxarray.open_rasterio(png).values[:3].astype(float)
        py, px0 = int(hy * sc), int(x0 * sc)
        span = int(a.swatch_search * sc)                                          # search left of the label, in points
        patch = raw[:, max(py - 9, 0):py + 9, max(px0 - span, 0):max(px0 - 6, 1)].reshape(3, -1)
        chroma = patch.max(0) - patch.min(0)
        keep = (chroma > 25) & (patch.max(0) < 250)                              # coloured ink, not paper or grey text
        if keep.sum() < 20:
            sys.exit(f"legend swatch not found within {a.swatch_search} pt left of the label — "
                     f"raise --swatch-search or pass --colour R,G,B")
        col = np.median(patch[:, keep], axis=1)
    print("  flood-inundation legend colour:", col.round().astype(int).tolist())

    v = rgb.values[:3].astype(float)
    dist = np.sqrt(((v - col[:, None, None]) ** 2).sum(0))
    flood = (dist <= a.tol).astype("uint8")
    print(f"  colour match: {flood.mean() * 100:.2f} % of the sheet")

    ref_f = sorted((ROOT / f"data/events/{ev}/rtc").glob("co_*.tif"))[0]
    ref = rioxarray.open_rasterio(ref_f, masked=True).isel(band=0)
    fl = rgb.isel(band=0).copy(data=flood).rio.write_nodata(255)
    fl = fl.rio.reproject_match(ref, resampling=5)                              # 5 = average -> fractional cover
    (fl > 0.5).astype("uint8").rio.write_nodata(255).rio.to_raster(out / "nrsc_flood.tif", driver="COG", compress="DEFLATE")

    json.dump(dict(source=a.url, pdf=str(pdf.name), dpi=a.dpi, n_gcps=len(gcps), lon_ticks=nlon, lat_ticks=nlat,
                   legend_rgb=col.round().astype(int).tolist(), colour_tolerance=a.tol, rms_residual_m=rms_m,
                   note="50 m product generalised for print; 'normal river/water bodies' is a separate class and is not merged"),
              open(out / "nrsc_meta.json", "w"), indent=1)
    print("  ->", out / "nrsc_flood.tif")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("event")
    ap.add_argument("--url", default="https://ndem.nrsc.gov.in/documents/Disaster_Document/2018/KL/"
                                     "klflood50dsc21082018_1830hrs/klflood50dsc21082018_1830hrs_map.pdf")
    ap.add_argument("--pdf", help="use a local copy instead of downloading")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--colour", help="R,G,B of the flood-inundation legend swatch (overrides auto-detection)")
    ap.add_argument("--tol", type=float, default=60.0, help="RGB distance tolerance for the colour match")
    ap.add_argument("--swatch-search", type=float, default=90.0,
                    help="how far left of the legend text to look for the colour swatch, in PDF points")
    ap.add_argument("--max-rms-m", type=float, default=2000.0, dest="max_rms_m")
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
