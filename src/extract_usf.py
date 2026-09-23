"""extract_usf.py — single streaming pass over the 35 GB UrbanSARFloods tarball; keep a subset, shrink it, delete tar.

Tar content (from the listing): 01_NF 4771 chips (45 GB), 02_FO 1012 (9.5 GB), 03_FU 691 (6.5 GB), 512x512x8 float32.
Kept:
  * all 02_FO (open flood) and 03_FU (urban flood) chips
  * all 01_NF chips listed in Valid_dataset.txt (unbiased validation) + --nf-train-frac of NF training chips
SAR -> float16 .npy (halves size; dB/coherence ranges fit float16), GT -> uint8 .npy. Georeferencing is not needed for training.
Output: data/benchmarks/usf/{SAR,GT}/<folder>__<name>.npy, lists/*.txt, index.csv, band_stats.csv
Step 1 extracts the three text files (tar must be read once to find them), step 2 streams the rest.

  python -u src/extract_usf.py --nf-train-frac 0.35          # ~14 GB output, ~25 min
  (rerunnable: already-written chips are skipped)
"""
from __future__ import annotations
import argparse, hashlib, io, re, subprocess, sys, tarfile
from pathlib import Path
import numpy as np, pandas as pd, rasterio

ROOT = Path(__file__).resolve().parents[1]
TAR = ROOT / "data/benchmarks/urbansarfloods/urban_sar_floods.tar.gz"
OUT = ROOT / "data/benchmarks/usf"


def keep_frac(name, frac):
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac   # deterministic sampling


def read_lists():
    lst = OUT / "lists"; lst.mkdir(parents=True, exist_ok=True)
    need = ["Train_dataset.txt", "Valid_dataset.txt", "data_norm.txt"]
    if not all((lst / n).exists() for n in need):
        print(">> step 1: extracting the text files (one full read of the archive)", flush=True)
        subprocess.run(["tar", "-xzf", str(TAR), "-C", str(lst), "--strip-components=1", "--wildcards",
                        *[f"urban_sar_floods/{n}" for n in need]], check=True)
    for n in need:
        p = lst / n
        if p.is_dir():                                   # listing showed '<name>.txt/' -> may be a directory
            inner = [q for q in p.rglob("*") if q.is_file()]
            print(f"   {n} is a directory with {len(inner)} files: {[q.name for q in inner][:5]}")
    txt = {n: (lst / n) for n in need}
    def lines(p):
        if p.is_dir():
            return [l.strip() for q in p.rglob("*") if q.is_file() for l in q.read_text().splitlines() if l.strip()]
        return [l.strip() for l in p.read_text().splitlines() if l.strip()]
    train, valid = lines(txt["Train_dataset.txt"]), lines(txt["Valid_dataset.txt"])
    print(f"   Train list {len(train)} lines, e.g. {train[:2]}\n   Valid list {len(valid)} lines, e.g. {valid[:2]}")
    print("   data_norm:", lines(txt["data_norm.txt"])[:12])
    return {chip_key(s) for s in train}, {chip_key(s) for s in valid}


def chip_key(s):
    """'../03_FU/GT/20190329_Iran_ID_17_19_GT.tif' and '..._SAR.tif' -> '20190329_Iran_ID_17_19'"""
    return re.sub(r"_(GT|SAR)$", "", Path(s.split()[0]).stem)


def main(a):
    train, valid = read_lists()
    if a.inspect:
        a.max_chips = 6
    (OUT / "SAR").mkdir(parents=True, exist_ok=True); (OUT / "GT").mkdir(parents=True, exist_ok=True)
    n_ren = 0                                               # v1 names kept the _SAR/_GT suffix -> normalise in place
    for f in list((OUT / "SAR").glob("*_SAR.npy")) + list((OUT / "GT").glob("*_GT.npy")):
        f.rename(f.with_name(re.sub(r"_(GT|SAR)\.npy$", ".npy", f.name))); n_ren += 1
    if n_ren:
        print(f"   renamed {n_ren} files from the first run")
    rows, stats, n_seen = [], [], 0
    print(">> step 2: streaming chips", flush=True)
    with tarfile.open(TAR, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".tif"):
                continue
            parts = Path(m.name).parts                      # urban_sar_floods/<folder>/<GT|SAR>/<name>.tif
            folder, kind, stem = parts[1], parts[2], Path(parts[3]).stem
            ck = chip_key(stem)
            split = "valid" if ck in valid else "train" if ck in train else "unlisted"
            keep = folder != "01_NF" or split == "valid" or (split == "train" and keep_frac(ck, a.nf_train_frac))
            n_seen += 1
            if n_seen % 500 == 0:
                print(f"   {n_seen} tif members read, {len(rows)} kept", flush=True)
            dst = OUT / kind / f"{folder}__{ck}.npy"
            if not keep or dst.exists():                    # resumable: existing outputs are skipped
                continue
            with rasterio.MemoryFile(tf.extractfile(m).read()) as mf, mf.open() as ds:
                arr = ds.read()
            if kind == "SAR":
                np.save(dst, arr.astype(np.float16))
                if len(stats) < 200:                        # per-band stats from the first chips
                    stats.append([np.nanpercentile(b, [1, 50, 99]).tolist() + [float(np.isnan(b).mean())] for b in arr])
            else:
                np.save(dst, arr[0].astype(np.uint8))
                if a.inspect:
                    print(f"   GT {folder}/{stem}: values {np.unique(arr, return_counts=True)}")
            rows.append(dict(folder=folder, kind=kind, name=stem, split=split, bands=arr.shape[0]))
            if a.max_chips and len(rows) >= a.max_chips:
                break
    # index built from what is on disk (so reruns are consistent)
    rows = []
    for f in sorted((OUT / "SAR").glob("*.npy")):
        folder, ck = f.stem.split("__", 1)
        rows.append(dict(folder=folder, name=ck, split="valid" if ck in valid else "train" if ck in train else "unlisted",
                         has_gt=(OUT / "GT" / f.name).exists()))
    idx = pd.DataFrame(rows); idx.to_csv(OUT / "index.csv", index=False)
    print(idx.groupby(["folder", "split", "has_gt"]).size().to_string())
    band_check(idx)
    gb = sum(f.stat().st_size for f in OUT.rglob("*.npy")) / 1e9
    print(f">> done. {len(idx)} chips, {gb:.1f} GB. When satisfied: rm {TAR}")


def band_check(idx, n=120):
    """Class-conditional band means on FO/FU chips -> identifies coherence vs intensity and pre vs co-event bands.
    Expected: open flood (1) -> co-event intensity << pre-event; urban flood (2) -> co-event coherence << pre-event."""
    sub = idx[idx.folder.isin(["02_FO", "03_FU"]) & idx.has_gt]
    sel = sub.sample(min(n, len(sub)), random_state=0)
    acc = {c: [] for c in (0, 1, 2)}
    for _, r in sel.iterrows():
        x = np.load(OUT / "SAR" / f"{r.folder}__{r['name']}.npy").astype(np.float32)
        y = np.load(OUT / "GT" / f"{r.folder}__{r['name']}.npy")
        for c in acc:
            m = y == c
            if m.sum() > 50:
                acc[c].append(np.nanmean(x[:, m], axis=1))
    tab = pd.DataFrame({f"class{c}": np.nanmean(v, 0) for c, v in acc.items() if v}, index=[f"b{i+1}" for i in range(8)])
    tab.to_csv(OUT / "band_by_class.csv"); print("\nper-class band means (0 non-flood, 1 open flood, 2 urban flood):")
    print(tab.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nf-train-frac", type=float, default=0.35)
    ap.add_argument("--inspect", action="store_true"); ap.add_argument("--max-chips", type=int, default=0)
    main(ap.parse_args())
