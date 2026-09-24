"""train_usf.py — urban vs open flood segmentation on UrbanSARFloods; the intensity-vs-coherence experiment.

Band order (verified 20 Sep 2026 from class-conditional means, data/benchmarks/usf/band_by_class.csv):
  b1 coh_pre VH  b2 coh_pre VV  b3 coh_co VH  b4 coh_co VV  b5 int_pre VH  b6 int_pre VV  b7 int_co VH  b8 int_co VV (dB)
  open flood: co-event VV -7.2 dB vs pre;  urban flood: coh VV 0.66 -> 0.33 and co-event VV +1.2 dB (double bounce)
Classes: 0 non-flood, 1 open flood, 2 urban flood (0.03 % of pixels -> weighted CE + Dice + FU-biased sampling).

Input configurations (--bands):
  co_int  co-event VV,VH only (single image, what most operational maps use)
  int     pre + co-event intensity (change detection information)
  all     intensity + coherence (the UrbanSARFloods setting)
  coh     coherence only

  python -u src/train_usf.py --bands int --model unet_r34
  python -u src/train_usf.py --bands all --model unet_r34
  python -u src/train_usf.py --bands vvcoh_int --model unet_r34   # VV coherence + intensity: what HyP3 can supply
Outputs runs/usf/<model>_<bands>/: best.pt, history.csv, metrics.json (per-class IoU/F1/P/R on the full validation list)
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from models import build  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
USF = ROOT / "data/benchmarks/usf"
BANDS = {"co_int": [6, 7], "int": [4, 5, 6, 7], "all": list(range(8)), "coh": [0, 1, 2, 3],
         # ASF HyP3 burst InSAR processes VV and HH only (VH is rejected by the API), so bands 0 and 2 can never be
         # produced for our own events. This subset is what an operational pipeline built on HyP3 can actually assemble:
         # VV coherence (pre, co) + all four intensity bands. Training on it keeps the transfer test honest — matched
         # inputs at train and test time, instead of zero-filling two bands the model was trained to rely on.
         "vvcoh_int": [1, 3, 4, 5, 6, 7]}
IGN = 255


def norm_stats():
    txt = (USF / "lists/data_norm.txt").read_text()
    grab = lambda k: np.array([float(v) for v in txt.split(k)[1].split("[")[1].split("]")[0].split()], np.float32)
    return grab("mean_in"), grab("std_in")


class USFDataset(Dataset):
    def __init__(self, split, bands, crop=256, n_samples=None, fu_bias=0.7, seed=0, subset_nf=None):
        idx = pd.read_csv(USF / "index.csv")
        idx = idx[(idx.split == split) & idx.has_gt].reset_index(drop=True)
        if subset_nf is not None:                                   # smaller NF share for fast per-epoch validation
            nf = idx[idx.folder == "01_NF"].sample(min(subset_nf, (idx.folder == "01_NF").sum()), random_state=seed)
            idx = pd.concat([idx[idx.folder != "01_NF"], nf]).reset_index(drop=True)
        self.idx, self.b, self.crop, self.fu_bias = idx, BANDS[bands], crop, fu_bias
        m, s = norm_stats(); self.mean, self.std = m[self.b, None, None], s[self.b, None, None]
        self.n = n_samples or len(idx)
        self.rng = np.random.default_rng(seed)
        # sampling weights per chip: urban-flood chips x3, open-flood x1.5, non-flood x1
        w = idx.folder.map({"03_FU": 3.0, "02_FO": 1.5, "01_NF": 1.0}).values
        self.p = w / w.sum()

    def __len__(self):
        return self.n

    def load(self, j):
        r = self.idx.iloc[j]
        x = np.load(USF / "SAR" / f"{r.folder}__{r['name']}.npy", mmap_mode="r")[self.b].astype(np.float32)
        y = np.load(USF / "GT" / f"{r.folder}__{r['name']}.npy").astype(np.int64)
        bad = ~np.isfinite(x).all(0)
        x = np.nan_to_num((x - self.mean) / self.std, nan=0.0, posinf=0.0, neginf=0.0)
        y[bad] = IGN
        return x, y

    def __getitem__(self, i):
        if self.crop is None:                                        # evaluation: full chip, deterministic order
            x, y = self.load(i)
        else:
            x, y = self.load(self.rng.choice(len(self.idx), p=self.p))
            c, (H, W) = self.crop, y.shape
            fu = np.argwhere(y == 2)
            if len(fu) and self.rng.random() < self.fu_bias:         # centre crop on an urban-flood pixel
                cy, cx = fu[self.rng.integers(len(fu))]
                r0, c0 = np.clip(cy - c // 2, 0, H - c), np.clip(cx - c // 2, 0, W - c)
            else:
                r0, c0 = self.rng.integers(0, H - c + 1), self.rng.integers(0, W - c + 1)
            x, y = x[:, r0:r0 + c, c0:c0 + c], y[r0:r0 + c, c0:c0 + c]
            k = self.rng.integers(4); x, y = np.rot90(x, k, (1, 2)), np.rot90(y, k)
            if self.rng.random() < .5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def loss_fn(logits, y, w):
    ce = F.cross_entropy(logits, y, weight=w, ignore_index=IGN)
    v = (y != IGN).unsqueeze(1).float()
    p = logits.softmax(1) * v
    t = F.one_hot(y.clamp(max=2), 3).permute(0, 3, 1, 2).float() * v
    inter, den = (p * t).sum((0, 2, 3)), (p + t).sum((0, 2, 3))
    dice = 1 - ((2 * inter + 1) / (den + 1))[1:].mean()             # Dice on the two flood classes
    return ce + dice


def confusion(logits, y):
    v = y != IGN; p = logits.argmax(1)[v]; t = y[v]
    return torch.bincount(t * 3 + p, minlength=9).reshape(3, 3).cpu().numpy()


def scores(cm):
    out = {}
    for c, name in [(1, "open_flood"), (2, "urban_flood")]:
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c, :].sum() - tp
        out[name] = dict(iou=tp / max(tp + fp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1),
                         precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), n_true=int(cm[c].sum()))
    fl = cm[1:, 1:].sum(); fp = cm[0, 1:].sum(); fn = cm[1:, 0].sum()     # any-flood (1 or 2) as binary
    out["any_flood"] = dict(iou=fl / max(fl + fp + fn, 1), f1=2 * fl / max(2 * fl + fp + fn, 1))
    # urban flood pixels predicted as open flood count as "detected water, wrong type"
    out["urban_as_open_frac"] = cm[2, 1] / max(cm[2].sum(), 1)
    out["mIoU_flood"] = (out["open_flood"]["iou"] + out["urban_flood"]["iou"]) / 2
    out["confusion"] = cm.tolist()
    return out


@torch.no_grad()
def evaluate(model, ds, dev, amp):
    model.eval(); cm = np.zeros((3, 3), np.int64)
    for i in range(len(ds)):
        x, y = ds[i]
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
            lg = model(x[None].to(dev)).float()
        cm += confusion(lg, y[None].to(dev))
    return scores(cm)


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    torch.backends.cudnn.benchmark = True; torch.manual_seed(a.seed)
    tag = f"{a.model}_{a.bands}"; out = ROOT / "runs/usf" / tag; out.mkdir(parents=True, exist_ok=True)
    nb = len(BANDS[a.bands])
    model, lr, amp = build(a.model, in_ch=nb, pretrained=not a.no_pretrained, classes=3)
    amp = amp and not a.no_amp; model.to(dev)
    tr = USFDataset("train", a.bands, crop=a.crop, n_samples=a.samples, seed=a.seed)
    va_fast = USFDataset("valid", a.bands, crop=None, subset_nf=a.val_nf)
    dl = DataLoader(tr, batch_size=a.bs, shuffle=False, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0, pin_memory=dev.type == "cuda",
                    worker_init_fn=lambda k: setattr(tr, "rng", np.random.default_rng(a.seed * 100 + k + int(time.time()))))
    w = torch.tensor([1.0, a.w_open, a.w_urban], device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=a.epochs * len(dl), pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=amp and dev.type == "cuda")
    print(f"### {tag}: in_ch {nb}, {len(tr.idx)} train chips, {a.samples} crops/epoch, "
          f"fast-val {len(va_fast)} chips, class weights {w.tolist()}", flush=True)
    hist, best, bad = [], -1, 0
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tl = 0.0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
                loss = loss_fn(model(x).float(), y, w)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step(); tl += loss.item()
        s = evaluate(model, va_fast, dev, amp)
        hist.append(dict(epoch=ep, loss=tl / len(dl), iou_open=s["open_flood"]["iou"], iou_urban=s["urban_flood"]["iou"],
                         miou=s["mIoU_flood"], sec=time.time() - t0))
        flag = ""
        if s["mIoU_flood"] > best:
            best, bad, flag = s["mIoU_flood"], 0, " *"; torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
        print(f"  ep {ep:3d} loss {hist[-1]['loss']:.3f}  IoU open {s['open_flood']['iou']:.3f}  "
              f"urban {s['urban_flood']['iou']:.3f}  (urban->open {s['urban_as_open_frac']:.2f})  {hist[-1]['sec']:.0f}s{flag}", flush=True)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if bad >= a.patience:
            print("  early stop"); break
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    full = evaluate(model, USFDataset("valid", a.bands, crop=None), dev, amp)
    json.dump({"tag": tag, "bands": BANDS[a.bands], "epochs_run": ep, "valid_full": full}, open(out / "metrics.json", "w"),
              indent=1, default=float)
    print(f"  FULL VALID  open IoU {full['open_flood']['iou']:.4f} F1 {full['open_flood']['f1']:.4f} | "
          f"urban IoU {full['urban_flood']['iou']:.4f} F1 {full['urban_flood']['f1']:.4f} R {full['urban_flood']['recall']:.3f} | "
          f"any-flood IoU {full['any_flood']['iou']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default="all", choices=list(BANDS)); ap.add_argument("--model", default="unet_r34")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--samples", type=int, default=3000, help="random crops per epoch")
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--val-nf", type=int, default=250, help="non-flood chips in the per-epoch validation subset")
    ap.add_argument("--w-open", type=float, default=2.0); ap.add_argument("--w-urban", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true"); ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    main(ap.parse_args())
