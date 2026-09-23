"""sarlib.py — core SAR flood-mapping functions (pure numpy; no I/O).

Conventions
  * backscatter arrays are LINEAR gamma0 (float32, NaN = nodata) unless the name ends in _db
  * masks are bool; water/flood = True
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture

EPS = 1e-6

def to_db(x):
    return 10.0 * np.log10(np.clip(x, EPS, None))

def lee_filter(img, size=5):
    """Classic Lee filter in LINEAR units; NaN-aware (NaNs stay NaN)."""
    valid = np.isfinite(img)
    x = np.where(valid, img, 0.0).astype(np.float64)
    w = ndi.uniform_filter(valid.astype(np.float64), size)
    m = ndi.uniform_filter(x, size) / np.maximum(w, EPS)
    m2 = ndi.uniform_filter(x * x, size) / np.maximum(w, EPS)
    var = np.maximum(m2 - m * m, 0)
    # multiplicative-noise variance: estimate from equivalent number of looks (S1 GRD/RTC ~ 4.4 ENL)
    enl = 4.4
    noise_var = (m * m) / enl
    k = np.clip((var - noise_var) / np.maximum(var, EPS), 0, 1)
    out = m + k * (x - m)
    return np.where(valid, out, np.nan).astype(np.float32)

def _sample(a, n=2_000_000, seed=0):
    v = a[np.isfinite(a)]
    if v.size > n:
        v = np.random.default_rng(seed).choice(v, n, replace=False)
    return v

def otsu_threshold(img_db):
    return float(threshold_otsu(_sample(img_db), nbins=512))

def split_based_threshold(img_db, tile=256, min_valid=0.8, ashman_min=2.0, min_prop=0.1,
                          n_per_tile=2000, seed=0):
    """Hierarchical/split-based thresholding (after Martinis 2009, Chini 2017).
    Keep tiles whose 2-component GMM is clearly bimodal (Ashman D >= 2, both modes >= 10 %);
    pool their pixels and take Otsu. Returns (threshold, n_selected_tiles, n_tiles)."""
    rng = np.random.default_rng(seed)
    H, W = img_db.shape
    pooled, sel, tot = [], 0, 0
    for r in range(0, H - tile + 1, tile):
        for c in range(0, W - tile + 1, tile):
            t = img_db[r:r + tile, c:c + tile]
            v = t[np.isfinite(t)]
            if v.size < min_valid * tile * tile:
                continue
            tot += 1
            s = rng.choice(v, min(n_per_tile, v.size), replace=False).reshape(-1, 1)
            g = GaussianMixture(2, random_state=seed).fit(s)
            mu, sd, w = g.means_.ravel(), np.sqrt(g.covariances_.ravel()), g.weights_
            D = np.sqrt(2) * abs(mu[0] - mu[1]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2)
            if D >= ashman_min and w.min() >= min_prop:
                pooled.append(s.ravel()); sel += 1
    if sel == 0:
        return otsu_threshold(img_db), 0, tot
    return float(threshold_otsu(np.concatenate(pooled), nbins=512)), sel, tot

def slope_deg(dem, res):
    gy, gx = np.gradient(dem.astype(np.float64), res)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)

def clean(mask, min_px=8, valid=None):
    """Drop 8-connected objects with < min_px pixels (version-independent of skimage API changes)."""
    mask = mask.astype(bool)
    lab, n = ndi.label(mask, structure=np.ones((3, 3), bool))
    if n:
        sizes = np.bincount(lab.ravel()); keep = sizes >= min_px; keep[0] = False
        mask = keep[lab]
    return mask if valid is None else (mask & valid)

def exclusion_mask(hand=None, slope=None, hand_max=15.0, slope_max=5.0):
    """True where water detection is NOT allowed (high HAND or steep slope)."""
    ex = None
    if hand is not None:
        ex = np.nan_to_num(hand, nan=0) > hand_max
    if slope is not None:
        s = np.nan_to_num(slope, nan=0) > slope_max
        ex = s if ex is None else (ex | s)
    return ex

def metrics(pred, ref, valid=None):
    """Binary metrics; pred/ref bool. Returns dict with TP..kappa, F1, IoU(=CSI), precision, recall, OA."""
    if valid is None:
        valid = np.ones_like(pred, bool)
    p, r = pred[valid].astype(bool), ref[valid].astype(bool)
    tp = np.sum(p & r, dtype=np.int64); fp = np.sum(p & ~r, dtype=np.int64)
    fn = np.sum(~p & r, dtype=np.int64); tn = np.sum(~p & ~r, dtype=np.int64)
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    oa = (tp + tn) / n if n else np.nan
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n) if n else np.nan
    kappa = (oa - pe) / (1 - pe) if n and pe != 1 else np.nan
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), precision=prec, recall=rec,
                f1=f1, iou=iou, oa=oa, kappa=kappa)


# ---------------------------------------------------------------------------------------------
# Refined Lee (Lee 1981 / SNAP S1TBX), line-by-line port of the GEE implementation used in the
# 2024 script, so the GEE replica uses the same speckle filter. Input LINEAR units. Tiled.
# ---------------------------------------------------------------------------------------------
def _wmean_var(x, k):
    w = k / k.sum()
    m = ndi.correlate(x, w, mode="reflect")
    v = ndi.correlate(x * x, w, mode="reflect") - m * m
    n = k.sum()
    return m, np.maximum(v, 0) * n / max(n - 1, 1)          # sample variance (ee.Reducer.variance)


def _refined_lee_block(img):
    k3 = np.ones((3, 3))
    mean3, var3 = _wmean_var(img, k3)
    offs = [(-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 0), (0, 2), (2, -2), (2, 0), (2, 2)]   # neighborhoodToBands order
    sh = lambda a, dy, dx: np.roll(np.roll(a, -dy, 0), -dx, 1)                                # value at pixel offset
    sm = np.stack([sh(mean3, dy, dx) for dy, dx in offs])
    sv = np.stack([sh(var3, dy, dx) for dy, dx in offs])
    grads = np.stack([abs(sm[1] - sm[7]), abs(sm[6] - sm[2]), abs(sm[3] - sm[5]), abs(sm[0] - sm[8])])
    gmask = grads == grads.max(0)
    d = np.stack([(sm[1] - sm[4]) > (sm[4] - sm[7]), (sm[6] - sm[4]) > (sm[4] - sm[2]),
                  (sm[3] - sm[4]) > (sm[4] - sm[5]), (sm[0] - sm[4]) > (sm[4] - sm[8])])
    dirs = np.zeros(img.shape, np.int8)
    for i in range(4):                                      # directions 1-4 and their opposites 5-8
        dirs = np.where(gmask[i] & (dirs == 0), np.where(d[i], i + 1, i + 5), dirs)
    stats = sv / np.maximum(sm * sm, 1e-12)
    sigmaV = np.sort(stats, 0)[:5].mean(0)
    rect = np.zeros((7, 7)); rect[3:, :] = 1                 # rows 3-6 = 1
    diag = np.tril(np.ones((7, 7)))
    dir_mean = np.zeros_like(img); dir_var = np.zeros_like(img)
    kernels = {1: rect, 2: diag}
    for i in range(1, 4):                                   # ee Kernel.rotate(i): clockwise
        kernels[2 * i + 1] = np.rot90(rect, -i); kernels[2 * i + 2] = np.rot90(diag, -i)
    for code, k in kernels.items():
        m, v = _wmean_var(img, k)
        sel = dirs == code
        dir_mean[sel] = m[sel]; dir_var[sel] = v[sel]
    varX = (dir_var - dir_mean * dir_mean * sigmaV) / (sigmaV + 1.0)
    b = varX / np.maximum(dir_var, 1e-12)
    return (dir_mean + b * (img - dir_mean)).astype(np.float32)


def refined_lee(img, tile=1024, halo=8):
    """Refined Lee on a large LINEAR image, processed in overlapping tiles. NaN-aware."""
    valid = np.isfinite(img)
    fill = np.nanmedian(img[valid][:: max(valid.sum() // 100000, 1)]) if valid.any() else 0.0
    x = np.where(valid, img, fill).astype(np.float64)
    out = np.empty(img.shape, np.float32)
    H, W = img.shape
    for r in range(0, H, tile):
        for c in range(0, W, tile):
            r0, c0 = max(r - halo, 0), max(c - halo, 0)
            r1, c1 = min(r + tile + halo, H), min(c + tile + halo, W)
            blk = _refined_lee_block(x[r0:r1, c0:c1])
            out[r:min(r + tile, H), c:min(c + tile, W)] = blk[r - r0:r - r0 + min(tile, H - r), c - c0:c - c0 + min(tile, W - c)]
    out[~valid] = np.nan
    return out
