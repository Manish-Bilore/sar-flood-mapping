# 04 — Compute Assessment and Plan (DarkHorse laptop)

Source: `sysprobe.sh` output, 20 Sep 2026. Numbers marked *est.* are replaced by `gpu_check.py` results.

## 1. Verdict per component

| Component | Measured | Verdict | Action |
|---|---|---|---|
| CPU | i7-8565U, 4C/8T, 15 W TDP, 4.6 GHz boost | OK for classical baselines and SNAP burst coherence. Throttles on long jobs. | Run CPU jobs (SNAP, thresholding) in parallel with GPU training. |
| RAM | 38 GB total, ~24 GB available, 8 GB swap | Sufficient: AOI stacks < 5 GB; SNAP with `-c 16G`. | Close the browser and other apps during SNAP runs. |
| GPU | Quadro P520, **2 GB**, Pascal **sm_61**, driver 535 (CUDA 12.2) | **Unusable as installed.** torch 2.10+cu128 has no sm_61 kernels; `is_available()=True` is misleading, and the first conv would fail. | `setup_env.sh`: torch + cu126, with automatic fallback to 2.7.1+cu118. |
| GPU precision | GP108 FP16 ≈ 1/64 of FP32 rate, no tensor cores | AMP gives **no speed-up**, only memory savings. | Train in FP32 by default. Use AMP only to fit a batch. |
| Disk | 113 GB free (88% used) | Tight. SLC full frames (4–8 GB each) are ruled out. | Data budget ≤ 70 GB. Burst-level SLC only. Stream COGs, don't mirror them. |
| Network | STAC PC/CDSE 200; ASF 400; Bhoonidhi 302; throughput "23 B/s" | ASF 400 = missing query params (the probe's fault). 302 = login redirect. **The 23 B/s figure is a failed test, not a measurement.** | Run `netcheck.sh`, which tests the real data hosts. |
| Python | anaconda base, py3.12, **numpy 2.2 vs modules compiled for numpy 1.x** → pandas, pyarrow, xarray, geopandas all broken | Do not repair base. | Isolated `sarflood` env (conda-forge, py3.11). |
| SAR tools | `snap` found, `gpt` not on PATH; ISCE2, GMTSAR, docker missing | SNAP alone is enough for coherence. **Drop ISCE2/GMTSAR** for the 3-day scope. | `setup_env.sh` locates `gpt` and adds it to PATH. |
| R / Quarto | Both present | OK | Figures in R (terra, sf, ggplot2, patchwork); docs in Quarto. |

## 2a. MEASURED (gpu_check.py, 20 Sep 2026, torch 2.7.1+cu118)

The GPU works. torch 2.7.1+cu118 ships sm_60 kernels, and those run on sm_61. setup_env.sh's "sm_61 MISSING" message was a false alarm from an over-strict check.

| Model (256², bs 8) | FP32 img/s | FP32 peak GB | AMP img/s | Use |
|---|---|---|---|---|
| U-Net ResNet-18 | 19.3 | 1.56 | 17.0 | fast baseline, FP32 |
| U-Net ResNet-34 (ResU-Net) | 14.2 | 1.11 | 12.7 | main CNN, FP32 |
| DeepLabV3+ R34 | 12.2 | 1.21 | 9.8 | FP32 |
| U-Net++ R34 | OOM | — | 5.0 | drop (3× slower than U-Net) |
| SegFormer MiT-B0 | 11.4 | 1.06 | **19.2** | AMP |
| SegFormer MiT-B2 | OOM | — | **7.1** (1.76 GB) | **feasible with AMP** |

- 8-channel input costs nothing extra, so coherence stacks are free.
- AMP slows the CNNs but speeds up SegFormer.
- Figures come from 10 iterations. Sustained speed on a 15 W laptop will be about 10–20% lower from thermal throttling.

**Epoch times** (256² crops):
- Sen1Floods11 hand-labelled set, ~1.8k crops: R34 ≈ 2.1 min, B2 ≈ 4.2 min.
- Sen1Floods11 weak-labelled set, ~17.5k crops: R34 ≈ 21 min.
- UrbanSARFloods: subsample 8k crops per epoch (flood-class balanced), R34 ≈ 10 min.

**Network (measured):** ~1.6 MB/s (~13 Mbit/s) from AWS. **Bandwidth is now the binding constraint**: 70 GB ≈ 12 h. Use parallel connections (`aria2c -x8`, `gsutil -m`), stream AOI windows from COGs, and start downloads first.

## 2. Model feasibility on 2 GB FP32 (pre-measurement estimate, superseded by 2a)

| Model | Params | 256², 2-ch, bs 8 | 256², 8-ch, bs 8 | Role |
|---|---|---|---|---|
| U-Net ResNet-18 | ~14 M | fits | fits | fast baseline |
| U-Net ResNet-34 (**ResU-Net**) | ~24 M | fits | fits | main CNN |
| Attention U-Net (Oktay gates, custom) | ~8–35 M, width-dependent | fits at base width 32 | fits | 2024 recreation |
| U-Net++ ResNet-34 | ~26 M | marginal (bs 4) | marginal | optional |
| DeepLabV3+ ResNet-34 | ~22 M | fits | fits | optional |
| SegFormer MiT-B0 / B1 | 3.7 M / 14 M | fits | fits* | transformer track |
| SegFormer MiT-B2 | 25 M | bs 2–4 | bs 2 | stretch goal |
| Swin-UNet (Swin-T) | ~27 M | bs ≤ 2, slow | likely OOM | **drop**; SegFormer represents the transformer family |

\* Older smp versions reject `in_channels≠3` for MiT encoders. If `gpu_check.py` shows ERR, patch the first patch-embed layer.

Throughput *est.*: U-Net-R34 at 256² runs at ~10–20 img/s. For Sen1Floods11:
- hand-labelled set (446 chips of 512², i.e. 1.8k crops of 256²): ~2–3 min per epoch;
- full weak-label set (~17.6k crops): ~15–30 min per epoch.

Four architectures × ~40 epochs on the hand-labelled set plus two urban models fit into **two overnight runs**.

## 3. Data budget (≤ 70 GB)

| Item | Cap | Note |
|---|---|---|
| Sen1Floods11: S1 chips + hand/weak labels (skip S2 tiles unless used for labels) | ~10 GB | Main rural training and test set |
| UrbanSARFloods: chips only (not the raw SLC zips) | ~15 GB | Urban training set, intensity + coherence |
| Kuro Siwo: subset of S/SE-Asian events | ~10 GB | Cross-dataset test |
| Event AOIs: S1 RTC VV+VH, 3–5 dates × 8 events | ~12 GB | Streamed from PC, AOI-clipped |
| SLC bursts for urban events (3 dates × 2–4 bursts × 2 pol) | ~15 GB | Delete after coherence is exported |
| Ancillary (GLO-30, HAND, WorldCover, GSW, GHSL, S2 masks) | ~5 GB | AOI-clipped |

## 4. Scope decisions forced by the hardware

1. **Transformers**: SegFormer-B0/B1 instead of Swin-UNet.
2. **InSAR**: SNAP `gpt` graph (TOPSAR-Split → Apply-Orbit → Back-Geocoding → Coherence → Deburst → Terrain-Correction) run on burst-extracted SLC from ASF (`asf_search` + `burst2safe`). Coherence is computed for urban events only; rural events use intensity only.
3. **Chip size 256²** everywhere. Inference at 512² with overlap-tiling (inference needs far less memory than training).
4. **No foundation models, no full-frame SLC, no ISCE2.**
5. Optional escape hatch, if you allow it: Kaggle's free T4 (16 GB, ~30 h/week) makes SegFormer-B2 and Swin-UNet feasible. It isn't used unless you say so.

## 5. Three-day schedule

| Block | GPU | CPU / network |
|---|---|---|
| Day 1 AM | — | Env setup; `gpu_check`; `netcheck`; acquisition-date audit for all 8 events (S1 + EOS-04) |
| Day 1 PM | — | Ancillary pipeline; AOI RTC stacks; Kerala classical baselines (Otsu, split-based, log-ratio change detection + HAND/GSW) |
| Day 1 night | Download Sen1Floods11 + UrbanSARFloods | SLC burst download (urban events) |
| Day 2 | Rural models: U-Net-R34, Attention U-Net, SegFormer-B0 (queued overnight) | SNAP coherence for urban events; S2 urban/forest masks |
| Day 3 AM | Urban: U-Net intensity-only vs intensity + coherence | Inference on all events; metrics per stratum (built-up / cropland / tree) |
| Day 3 PM | — | R figures, Quarto site, web viewer (COG/PMTiles on GitHub Pages) |

Risk: three days holds only if the date audit clears at least Kerala, Assam, Bihar, Delhi and one urban pluvial event. If Bengaluru or Chennai lack a near-peak acquisition, they become a "detectability failure" finding rather than a mapped result.
