"""qa_rtc.py — coverage + value sanity check and quicklooks for fetched RTC files.
Prints valid fraction and dB percentiles per file; writes outputs/qa/<event>_<file>.png
(R = VV dB, G = VH dB, B = VV-VH dB), downsampled 10x.
Usage: python -u src/qa_rtc.py [event ...]
"""
import sys
from pathlib import Path
import numpy as np, rioxarray
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/qa"; OUT.mkdir(parents=True, exist_ok=True)

def stretch(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)

evs = sys.argv[1:] or [p.name for p in (ROOT / "data/events").iterdir()]
print(f"{'file':40s} {'valid%':>6s} {'VV p2/p50/p98 dB':>22s} {'VH p2/p50/p98 dB':>22s}")
for ev in evs:
    for f in sorted((ROOT / f"data/events/{ev}/rtc").glob("*.tif")):
        a = rioxarray.open_rasterio(f, masked=True)[:, ::10, ::10].values.astype("float32")
        db = 10 * np.log10(np.clip(a, 1e-6, None)); db[~np.isfinite(a)] = np.nan
        v = np.isfinite(db[0]).mean() * 100
        pv = np.nanpercentile(db[0], [2, 50, 98]); ph = np.nanpercentile(db[1], [2, 50, 98])
        print(f"{ev + '/' + f.name:40s} {v:6.1f} {'/'.join(f'{x:6.1f}' for x in pv):>22s} {'/'.join(f'{x:6.1f}' for x in ph):>22s}")
        rgb = np.dstack([stretch(db[0], -25, 0), stretch(db[1], -30, -5), stretch(db[0] - db[1], 0, 15)])
        rgb[~np.isfinite(rgb)] = 0
        plt.imsave(OUT / f"{ev}_{f.stem}.png", rgb)
print(f"\nquicklooks -> {OUT}")
