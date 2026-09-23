"""sensitivity.py — how much do the ensemble's design choices actually change the flood map?

Three questions, one table each (outputs/<event>/tables/):
  sens_vote.csv       vote rule / DL gate / VH-Otsu inclusion -> area and IoU against the published default
  sens_thresholds.csv HAND, slope and minimum-object-size cut-offs -> area removed by each
                      (the object-size sweep is ADDITIONAL: baseline_classical.py already dropped objects < 8 px)
  sens_redundancy.csv pairwise IoU between the candidate voters (why VH-Otsu is redundant rather than wrong)

Run after baseline_classical.py + predict_event.py (same inputs as make_event_products.py):
  python -u src/sensitivity.py kerala_2018
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, rioxarray, yaml
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).parent))
from sarlib import metrics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/events.yaml"))
EVENTS = {**CFG["rural"], **CFG["urban"]}
CLASSICAL = ["cd_vv", "otsu_vv", "split_vv"]
DEFAULT = dict(gate=0.60, rule="3of4", vh=False, hand=15.0, slope=5.0, mmu=8)


def rd(f):
    return rioxarray.open_rasterio(f, masked=True).squeeze("band", drop=True)


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def clean(m, mmu):
    if mmu <= 0:
        return m
    lab, n = ndi.label(m)
    if not n:
        return m
    keep = np.zeros(n + 1, bool)
    keep[1:] = np.bincount(lab.ravel())[1:] >= mmu
    return keep[lab]


def main(a):
    ev = a.event; cfg = EVENTS[ev]
    d = ROOT / f"data/events/{ev}"; out = ROOT / f"outputs/{ev}"; tab = out / "tables"; tab.mkdir(parents=True, exist_ok=True)
    co_f = sorted((d / "rtc").glob("co_*.tif"))[0]
    bdir = d / "baseline" if (d / "baseline/stats.csv").exists() else d / "baseline" / co_f.stem.split("_")[1]
    anc = d / "anc"
    ref = rioxarray.open_rasterio(co_f, masked=True).isel(band=0)
    px_km2 = abs(np.prod(ref.rio.resolution())) / 1e6
    valid = np.isfinite(ref.values)

    masks = {f.stem.replace("flood_", ""): (rd(f).values == 1) for f in sorted(bdir.glob("flood_*.tif"))}
    dl = {"dl_" + f.stem.replace("_flood", ""): (rd(f).values == 1) for f in sorted((d / "dl").glob("*_flood.tif"))} \
        if (d / "dl").exists() else {}
    gates = {}
    for n in dl:
        mj = ROOT / "runs/s1f11" / n[3:] / "metrics.json"
        gates[n] = json.load(open(mj))["test"]["iou"] if mj.exists() else -1.0

    def ensemble(gate, rule, vh):
        ok = [n for n, v in gates.items() if v >= gate]
        voters = [m for m in CLASSICAL if m in masks]
        if vh and "otsu_vh" in masks:
            voters.append("otsu_vh")
        stack = [masks[v] for v in voters]
        if ok:
            stack.append(np.sum([dl[n] for n in ok], axis=0) * 2 >= len(ok))      # DL casts a single vote
        votes = np.sum(stack, axis=0); n = len(stack)
        need = n / 2 + 0.5 if rule == "majority" else (n if rule == "unanimous" else int(rule[0]))
        return (votes >= need) & valid, voters + (["dl_ensemble"] if ok else []), ok

    base, base_voters, base_ok = ensemble(DEFAULT["gate"], DEFAULT["rule"], DEFAULT["vh"])

    # ---- 1. vote rule / gate / VH inclusion -------------------------------------------------
    rows = []
    variants = [("default (3 of 4: 3 classical + 1 DL vote)", DEFAULT["gate"], "3of4", False),
                ("looser vote (2 of 4)", DEFAULT["gate"], "2of4", False),
                ("unanimous (4 of 4)", DEFAULT["gate"], "unanimous", False),
                ("DL gate 0.55", 0.55, "3of4", False), ("DL gate 0.65", 0.65, "3of4", False),
                ("DL gate 1.01 (classical only, 2 of 3)", 1.01, "majority", False),
                ("classical only, unanimous (3 of 3)", 1.01, "unanimous", False),
                ("+ VH-Otsu as a 5th voter (3 of 5)", DEFAULT["gate"], "3of4", True),
                ("every DL model votes separately (majority of 8)", -1.0, "majority", False)]
    for name, gate, rule, vh in variants:
        if name.startswith("every DL"):
            stack = [masks[v] for v in CLASSICAL if v in masks] + list(dl.values())
            m = (np.sum(stack, axis=0) * 2 > len(stack)) & valid; ok = list(dl)
        else:
            m, _, ok = ensemble(gate, rule, vh)
        rows.append(dict(variant=name, area_km2=float(m.sum() * px_km2), iou_vs_default=iou(m, base),
                         dl_models_voting=len(ok)))
    pd.DataFrame(rows).to_csv(tab / "sens_vote.csv", index=False)

    # ---- 2. post-processing thresholds ------------------------------------------------------
    hand = rd(anc / "hand.tif").values if (anc / "hand.tif").exists() else np.zeros(valid.shape)
    slope = rd(anc / "slope.tif").values if (anc / "slope.tif").exists() else np.zeros(valid.shape)
    seas = rd(anc / "gsw_seas.tif").values if (anc / "gsw_seas.tif").exists() else np.zeros(valid.shape)
    perm = np.nan_to_num(seas, nan=0) >= 10
    raw = np.logical_or.reduce([masks[v] for v in CLASSICAL if v in masks]) & valid & ~perm   # before terrain/MMU
    rows = []
    for nm, vals, f in [("HAND [m]", [5, 10, 15, 25, np.inf], lambda t: np.nan_to_num(hand) <= t),
                        ("slope [deg]", [2, 3, 5, 10, np.inf], lambda t: np.nan_to_num(slope) <= t),
                        ("min object [px], additional", [0, 4, 8, 16, 25, 50], None)]:
        for t in vals:
            m = clean(raw, int(t)) if f is None else (raw & f(t))
            rows.append(dict(control=nm, cutoff=float(t), area_km2=float(m.sum() * px_km2),
                             removed_km2=float((raw.sum() - m.sum()) * px_km2),
                             removed_pct=float(100 * (raw.sum() - m.sum()) / max(raw.sum(), 1))))
    pd.DataFrame(rows).to_csv(tab / "sens_thresholds.csv", index=False)

    # ---- 3. redundancy between candidate voters ---------------------------------------------
    cand = {n: masks[n] for n in ("otsu_vv", "otsu_vh", "split_vv", "cd_vv") if n in masks}
    cand.update({n: dl[n] for n in sorted(dl)})
    rows = [dict(a=x, b=y, iou=iou(cand[x], cand[y])) for i, x in enumerate(cand) for j, y in enumerate(cand) if j > i]
    pd.DataFrame(rows).to_csv(tab / "sens_redundancy.csv", index=False)

    v = pd.read_csv(tab / "sens_vote.csv")
    json.dump({"default_area_km2": float(base.sum() * px_km2), "default_voters": base_voters,
               "dl_gate": DEFAULT["gate"], "dl_models_voting": base_ok,
               "vote_area_min_km2": float(v.area_km2.min()), "vote_area_max_km2": float(v.area_km2.max()),
               "vote_iou_min": float(v.iou_vs_default.min())}, open(tab / "sens_summary.json", "w"), indent=1)
    print(v.round(3).to_string(index=False))
    print("\n->", tab)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("event")
    main(ap.parse_args())
