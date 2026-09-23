"""s1data.py — Sentinel-1 input normalisation (shared by training AND event inference) + Sen1Floods11 dataset.

Input channels (3): VV dB, VH dB, VV-VH dB, each clipped and scaled to [0, 1]; NaN -> 0 and masked in the label.
Sen1Floods11 S1Hand = sigma0 dB (GEE GRD, no terrain flattening); event RTC = gamma0 linear -> converted to dB here.
The sigma0/gamma0 offset (~1-2 dB on flat terrain) is a known domain gap; report it, don't hide it.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, rasterio, torch
from torch.utils.data import Dataset

CLIP = {"vv": (-30.0, 0.0), "vh": (-35.0, -5.0), "ratio": (0.0, 20.0)}


def normalise_db(vv_db: np.ndarray, vh_db: np.ndarray):
    """-> (3,H,W) float32 in [0,1], valid mask (H,W)."""
    valid = np.isfinite(vv_db) & np.isfinite(vh_db)
    chans = []
    for arr, (lo, hi) in ((vv_db, CLIP["vv"]), (vh_db, CLIP["vh"]), (vv_db - vh_db, CLIP["ratio"])):
        chans.append(np.clip((np.nan_to_num(arr, nan=lo) - lo) / (hi - lo), 0, 1))
    x = np.stack(chans).astype(np.float32)
    x[:, ~valid] = 0
    return x, valid


def linear_to_input(vv_lin, vh_lin):
    to_db = lambda a: 10 * np.log10(np.where(a > 0, a, np.nan))
    return normalise_db(to_db(vv_lin), to_db(vh_lin))


class Sen1Floods11(Dataset):
    """Hand-labelled split. Label: 1 water, 0 dry, -1 ignore. Loads everything into RAM (~0.5 GB)."""
    def __init__(self, root, split, crop=None, crops_per_chip=1, augment=False, seed=0):
        root = Path(root)
        rows = pd.read_csv(root / f"splits/flood_{split}_data.csv", header=None)
        self.ids, xs, ys = [], [], []
        for s1 in rows[0]:
            cid = s1.replace("_S1Hand.tif", "")
            with rasterio.open(root / "S1Hand" / f"{cid}_S1Hand.tif") as f:
                a = f.read().astype(np.float32)
            with rasterio.open(root / "LabelHand" / f"{cid}_LabelHand.tif") as f:
                lab = f.read(1).astype(np.int16)
            x, valid = normalise_db(a[0], a[1])
            lab[~valid] = -1
            self.ids.append(cid); xs.append(x); ys.append(lab)
        self.x, self.y = np.stack(xs), np.stack(ys)
        self.crop, self.k, self.aug = crop, crops_per_chip, augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.ids) * self.k

    def __getitem__(self, i):
        j = i % len(self.ids)
        x, y = self.x[j], self.y[j]
        if self.crop:
            H, W = y.shape; c = self.crop
            r, s = self.rng.integers(0, H - c + 1), self.rng.integers(0, W - c + 1)
            x, y = x[:, r:r + c, s:s + c], y[r:r + c, s:s + c]
        if self.aug:
            k = self.rng.integers(4)
            x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            if self.rng.random() < .5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)).long()

    def region(self, j):
        return self.ids[j].split("_")[0]
