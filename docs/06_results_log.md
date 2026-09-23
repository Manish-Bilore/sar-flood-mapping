# 06 — Results Log (running)

## Kerala 2018, classical baselines (21 Aug 2018, S1A p165 DESC, PC RTC γ⁰; 3 districts = 6,579 km², geoBoundaries)

| Method | Flood km² (bbox) | Flood km² (3 districts) | Tree | Cropland | Built-up |
|---|---|---|---|---|---|
| Otsu VV (−12.86 dB) | 573 | 517 | 11 | 284 | 1.1 |
| Otsu VH (−17.77 dB) | 608 | 547 | 15 | 301 | 1.7 |
| Split-based VV (−12.10 dB, 380/1728 bimodal tiles) | 596 | 537 | 13 | 293 | 1.3 |
| **Change detection** (log-ratio < −3 dB vs dry-season median, same path) | **440** | **401** | 8 | 251 | 0.2 |
| Urban brightening candidate (built-up, log-ratio > +3 dB) | 48 | 43 | – | – | 48 |
| **2024 GEE logic, replica** (Refined Lee, dB/dB ratio > 1.1) | 1,318 | **984** | **777** | 226 | 89 |

Masks (sequential attribution, km²): permanent water removes 1,879 (Otsu VV) / 423 (CD); HAND > 15 m only 1–15; slope > 5° 33–65.
Terrain exclusion covers 6,646 km² but almost none of it overlaps detections → Copernicus DSM canopy bias on HAND is not material here.

**2024 result check:** the 2024 panel shows "112.92 km²" and "19.47 %" of 5,767.41 km² — inconsistent (19.47 % = 1,122.9 km²).
The replica (984 km², 15 % of the districts) supports ~1,123 km² → the area label most likely dropped a digit.
**The 2024 method is not a flood map:** 79 % of its detections are tree cover. Cause: a ratio of dB values makes the required
backscatter drop proportional to the pixel's own brightness (−7 dB tree needs only −0.7 dB, i.e. speckle-level change).

## Sen1Floods11 (hand-labelled, S1 only, VV/VH/VV−VH), metrics over all valid pixels

| Model | Valid IoU | Test IoU | Test F1 | Bolivia IoU (unseen event) | κ test |
|---|---|---|---|---|---|
| U-Net ResNet-18 (ImageNet init) | 0.651 | **0.681** | 0.810 | 0.680 | 0.784 |
| *DeepSARFlood (ViT/CNN deep ensemble, 2025) — reported SOTA* | | *0.72* | | | |

(remaining architectures queued)

## UrbanSARFloods — band order verified from class-conditional means (120 FO/FU chips)

| Band | Meaning | Non-flood | Open flood | Urban flood |
|---|---|---|---|---|
| b1/b2 | coherence pre-event VH/VV | 0.27 / 0.36 | 0.23 / 0.31 | 0.53 / 0.66 |
| b3/b4 | coherence co-event VH/VV | 0.19 / 0.26 | 0.11 / 0.13 | 0.25 / 0.33 |
| b5/b6 | intensity pre-event VH/VV (dB) | −15.3 / −9.4 | −16.9 / −10.3 | −13.4 / −6.2 |
| b7/b8 | intensity co-event VH/VV (dB) | −14.7 / −8.8 | −22.2 / −17.5 | −13.0 / −5.0 |

Physical reading (matches docs/02):
- **open flood**: specular drop −5.3 dB VH, −7.2 dB VV; coherence collapses to ~0.1.
- **urban flood**: intensity *rises* (+0.5 dB VH, **+1.2 dB VV** — double bounce, co-pol dominated) while VV coherence halves (0.66 → 0.33);
  the non-flood background loses only ~0.1 coherence over the same interval. An intensity-decrease rule can never see this class.
Class codes: 0 non-flood, 1 open flood (1.5 % of pixels), 2 urban flood (0.03 %).
Subset kept: 4,251 chips (NF 1,111 train + 1,439 valid; FO 699 + 312; FU 471 + 219), 19.1 GB float16.
