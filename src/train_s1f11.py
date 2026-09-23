"""train_s1f11.py — train/evaluate one architecture on Sen1Floods11 (hand-labelled, S1 only).

  python -u src/train_s1f11.py --model unet_r34                # defaults: 60 epochs, 256² crops, bs 8
  python -u src/train_s1f11.py --model all --epochs 60         # queue every model (overnight)
Outputs runs/s1f11/<model>/: best.pt, history.csv, metrics.json (valid / test / bolivia, global + per region)
Metric convention = Sen1Floods11 papers: IoU/F1 over all valid pixels of the split (not mean of chip IoUs).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from models import build, SPECS          # noqa: E402
from s1data import Sen1Floods11          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/benchmarks/sen1floods11"


def loss_fn(logits, y):
    m = (y >= 0).float(); t = y.clamp(min=0).float()
    logits = logits.squeeze(1)
    bce = (F.binary_cross_entropy_with_logits(logits, t, reduction="none") * m).sum() / m.sum().clamp(min=1)
    p = torch.sigmoid(logits) * m
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * m).sum() + 1)
    return bce + dice


def confusion(logits, y):
    v = y >= 0; p = (logits.squeeze(1) > 0)[v]; t = (y == 1)[v]
    return np.array([(p & t).sum().item(), (p & ~t).sum().item(), (~p & t).sum().item(), (~p & ~t).sum().item()], np.int64)


def scores(c):
    tp, fp, fn, tn = c.astype(float); n = c.sum()
    oa = (tp + tn) / n; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n ** 2
    return dict(iou=tp / max(tp + fp + fn, 1), f1=2 * tp / max(2 * tp + fp + fn, 1), precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1), oa=oa, kappa=(oa - pe) / (1 - pe), n_px=int(n))


@torch.no_grad()
def evaluate(model, ds, dev, amp):
    model.eval(); tot = np.zeros(4, np.int64); per = {}
    for j in range(len(ds.ids)):
        x = torch.from_numpy(ds.x[j:j + 1]).to(dev); y = torch.from_numpy(ds.y[j:j + 1]).long().to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
            c = confusion(model(x).float(), y)
        tot += c; per.setdefault(ds.region(j), np.zeros(4, np.int64)); per[ds.region(j)] += c
    return scores(tot), {k: scores(v) for k, v in per.items()}


def run(name, a, dev):
    out = ROOT / "runs/s1f11" / name; out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    model, lr, amp = build(name, pretrained=not a.no_pretrained)
    amp = amp and not a.no_amp
    model.to(dev)
    tr = Sen1Floods11(a.data, "train", crop=a.crop, crops_per_chip=a.crops, augment=True, seed=a.seed)
    va = Sen1Floods11(a.data, "valid")
    dl = DataLoader(tr, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                    pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=amp and dev.type == "cuda")
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n### {name}: {n_par:.1f} M params, lr {lr}, amp {amp}, {len(tr)} crops/epoch, {len(dl)} steps/epoch", flush=True)
    hist, best, bad = [], -1, 0
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); run_loss = 0.0
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
                loss = loss_fn(model(x).float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step(); run_loss += loss.item()
        v, _ = evaluate(model, va, dev, amp)
        hist.append(dict(epoch=ep, loss=run_loss / len(dl), val_iou=v["iou"], val_f1=v["f1"], sec=time.time() - t0))
        flag = ""
        if v["iou"] > best:
            best, bad, flag = v["iou"], 0, " *"
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
        print(f"  ep {ep:3d} loss {hist[-1]['loss']:.4f} val IoU {v['iou']:.4f} F1 {v['f1']:.4f} "
              f"({hist[-1]['sec']:.0f}s){flag}", flush=True)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if bad >= a.patience:
            print(f"  early stop (no val gain for {a.patience} epochs)"); break
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    res = {"model": name, "params_M": n_par, "best_val_iou": best, "epochs_run": ep}
    for split in ("valid", "test", "bolivia"):
        g, per = evaluate(model, Sen1Floods11(a.data, split), dev, amp)
        res[split] = g; res[f"{split}_by_region"] = per
        print(f"  {split:8s} IoU {g['iou']:.4f}  F1 {g['f1']:.4f}  P {g['precision']:.3f}  R {g['recall']:.3f}  kappa {g['kappa']:.3f}")
    json.dump(res, open(out / "metrics.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unet_r34", help=f"one of {list(SPECS)} or 'all'")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--bs", type=int, default=8); ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--crops", type=int, default=4, help="random crops per chip per epoch")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true"); ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--data", default=str(DATA)); ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    torch.backends.cudnn.benchmark = True
    names = list(SPECS) if a.model == "all" else a.model.split(",")
    for n in names:
        try:
            run(n, a, dev)
        except torch.OutOfMemoryError:
            print(f"  !! {n}: OOM at bs {a.bs} — rerun with --bs {max(a.bs // 2, 1)}"); torch.cuda.empty_cache()
