# 01 — Open SAR, Label and Ancillary Datasets for Flood Mapping

Scope: everything usable for the SAR flood-mapping recreation (Kerala 2018 anchor + Indian urban/rural events), grouped by access tier. Status checked Sept 2026.

Legend — **Access**: `OPEN` = free, register at most · `FOP` = free on accepted proposal · `PRICED` = paid for non-government users.
**STAC?** = queryable via a STAC API today.

---

## 1. SAR imagery archives

### 1.1 Open

| Mission | Band / λ | Pol (land) | Res. (typical) | Archive | Access point(s) | STAC? | Notes for this project |
|---|---|---|---|---|---|---|---|
| **Sentinel-1 A/B/C/D** | C, 5.55 cm | VV+VH (IW) | GRD 10 m px (~20×22 m true), SLC 2.3×14 m | 2014 → | CDSE, ASF DAAC, Microsoft Planetary Computer (PC), Earth Search (AWS), Bhoonidhi | Yes (CDSE, PC, Earth Search) | Backbone. See constellation timeline below — it decides revisit per event. |
| S1 **RTC (γ⁰, 10 m)** derived | C | VV+VH | 10 m | 2014 → | PC `sentinel-1-rtc` | Yes | Analysis-ready COGs → AOI-windowed reads, no full scenes on disk. PC account needed for this collection. |
| S1 **GRD** | C | VV+VH | 10 m | 2014 → | PC `sentinel-1-grd`, CDSE, Earth Search | Yes | Needs own calibration + terrain flattening (sarsen / SNAP). |
| S1 **SLC** (for InSAR coherence) | C | VV+VH | 2.3×14 m | 2014 → | CDSE (STAC/OData), ASF (incl. **burst-level extraction**) | CDSE: yes | Only route to coherence. Not on PC. ASF needs a free NASA Earthdata login — not a permission request. |
| **NISAR L-SAR** | L, 24 cm | dual/quad (mode-dependent) | 3–10 m (GCOV/GSLC) | BETA products earlier in 2026; calibrated PROVISIONAL from **17 Jun 2026** | ASF DAAC (Vertex, `asf_search`), Earthdata | CMR (STAC via CMR-STAC) | **Cannot cover any target event** (all ≤ 2024). Use only to build/demo the pipeline on a monsoon-2026 flood. |
| **NISAR S-SAR** | S, ~10 cm | — | — | 2025 → (samples) | Bhoonidhi (open) | Bhoonidhi API | Limited sample set released so far. |
| **EOS-04 (RISAT-1A)** MRS/CRS ScanSAR | C | dual / hybrid-pol | ~18–36 m (mode-dependent) | Feb 2022 → | Bhoonidhi (open for all; L2B NRB direct download from 1 Feb 2024, earlier dates orderable) | Bhoonidhi API | **Key gap-filler**: S1B was dead from Dec 2021, so 2022–24 events have 12-day S1 revisit. EOS-04 open data covers Bengaluru 2022, Delhi 2023, Chennai 2023, Wayanad 2024. |
| RISAT-1 MRS/CRS | C | dual / hybrid | ~18–50 m | 2012–2016 | Bhoonidhi | API | Historical only. |
| ALOS-2 PALSAR-2 ScanSAR / mosaics | L | HH+HV | 25–100 m | 2014 → | JAXA G-Portal / JAXA EORC | No | L-band HH/HV = flooded-vegetation capability. Open ScanSAR L2.2 and yearly mosaics — **verify event-date coverage per scene before relying on it.** |
| ALOS PALSAR (ALOS-1) | L | HH+HV | 12.5 m RTC | 2006–2011 | ASF | CMR | Historical only. |
| ERS-1/2, Envisat ASAR | C | VV / dual | 25–30 m | 1991–2012 | ESA | No | Historical only. |
| **Capella Open Data** | X, 3 cm | single (HH/VV) | ~0.5–1 m | curated scenes | AWS Open Data | Yes (static) | Sparse; check for Indian scenes. Useful to *show* urban double-bounce at VHR, not for training. |
| **Umbra Open Data** | X | single | 0.25–1 m | curated, incl. some time series | AWS Open Data | Yes (static) | Same role as Capella. |
| ICEYE open datasets | X | VV | ~1–3 m | event samples | ICEYE site / AWS | Partial | Disaster sample sets exist; coverage of target events unlikely. |

### 1.2 Free on proposal (FOP)

| Mission | Band | Route | Turnaround | Relevance |
|---|---|---|---|---|
| TerraSAR-X / TanDEM-X | X | DLR science proposal (TSX-Science portal) | weeks | Urban flood at 1–3 m. Archive over Kerala 2018 unlikely but searchable. |
| COSMO-SkyMed | X | ASI Open Call / ESA TPM | weeks | Same. |
| ICEYE, Capella, others | X | **ESA Third-Party Missions (TPM)** via ESA EO Gateway | weeks–months | One proposal covers several commercial providers. |
| RADARSAT-2 / RCM | C | CSA/MDA science programmes | weeks | Quad-pol C-band. Low priority. |
| SAOCOM-1A/B | L | CONAE catalogue, registration + project approval | variable | L-band full-pol; India coverage sparse. |

**Not accessible to an individual researcher**: International Charter, Sentinel Asia, Copernicus Contributing Missions — agency-only. NovaSAR-1 (S-band) is priced on Bhoonidhi for non-government users.

### 1.3 Sentinel-1 constellation timeline (decides revisit per event)

| Period | Units | Nominal revisit (same orbit) | Events affected |
|---|---|---|---|
| 2016 → Dec 2021 | S1A + S1B | 6 days | Kerala 2018, Hyderabad 2020, Assam/Bihar 2019–20 |
| Dec 2021 → early 2025 | S1A only (S1B failed Dec 2021; S1C launched Dec 2024, data from 2025) | 12 days | **Bengaluru 2022, Delhi 2023, Chennai 2023, Wayanad 2024, Assam 2022** |
| 2025 → Apr 2026 | S1A + S1C | 6 days | — |
| Apr → Jun 2026 | S1A + S1C + S1D | < 6 days | — |
| Jul 2026 → | S1C + S1D (S1A retired 29 Jun 2026) | 6 days | NISAR-era demo |

Consequence: for 2022–24 urban events, a peak-flood S1 acquisition may not exist. **First task per event = acquisition-date audit against flood peak**, before any processing.

---

## 2. Flood label / benchmark datasets

| Dataset | Sensor(s) | Chips / size | Events | Labels | SLC / coherence? | Urban class? | India content | Why use |
|---|---|---|---|---|---|---|---|---|
| **Sen1Floods11** (2020) | S1, S2 | 4,831 × 512² | 11 | 446 hand-labelled, rest weak (Otsu/S2) | No | No | Yes (1 event) | De-facto benchmark; every paper reports IoU on it. DeepSARFlood SOTA IoU 0.72. |
| **Kuro Siwo** (NeurIPS 2024) | S1 GRD + SLC, DEM | 67,490 labelled × 224² (+466k unlabelled) | 43 | Manual, 3 classes (flood / permanent water / no water) | SLC provided | No explicit | Check event list | Best-quality manual labels; pre-pre-post triplets → change-detection models. Baseline F1 > 0.80 flood. |
| **UrbanSARFloods** (CVPRW 2024) | S1 SLC → intensity + coherence (8 bands) | 8,879 × 512² | 18 | Semi-auto: non-flood / open flood / **urban flood** | **Yes** | **Yes** | Check event list | **Only benchmark with an urban-flood class + coherence** → core of the urban track. |
| MMFlood | S1, DEM, OSM hydrography | 1,748 × 2000² | 95 | CEMS rapid-mapping delineations | No | No | Possibly | DEM + hydrography as inputs. |
| S1GFloods | S1 | 5,360 × 256² | 46 | Semi-auto | No | No | ? | Change-detection pairs. |
| ETCI 2021 | S1 | 66,810 × 256² | 5 | ? | No | No | No (Bangladesh nearby) | Legacy competition; weakly documented labels. |
| OmbriaNet | S1, S2 | 1,688 × 256² | 23 | CEMS | No | No | ? | Pre/post pairs. |
| S1S2-Water (DLR, 2023) | S1, S2 | 65 scenes | — | Manual water (not flood) | No | No | ? | Permanent-water pretraining. |
| CAU-Flood | S1, S2 | 18,302 × 256² | 18 | Manual | No | No | ? | Pre/post. |
| UNOSAT FloodAI | S1 | 58,128 × 256² | 15 | Semi-auto | No | No | Some S. Asia | Operational-style labels. |

### Reference *products* (not training labels, but comparators)
| Product | What | Coverage | Use |
|---|---|---|---|
| **Copernicus GFM** (CEMS Global Flood Monitoring) | 3-algorithm ensemble flood extent, likelihood, exclusion mask, 20 m, **every S1 IW scene 2015 →** | Global | **Common reference across all 8 events** where no published map exists. Its exclusion mask flags where SAR mapping is unreliable (urban, shadow, layover) — directly relevant to the urban analysis. Access: GFM portal, API, openEO. |
| CEMS Rapid Mapping | Event-activation vector maps | Activated events only | Check activations for Kerala 2018 / Assam. |
| Global Flood Database (MODIS) | 250 m event maps, 2000–2018 | Global | Coarse sanity check for Kerala 2018. |
| JRC Global Surface Water | Occurrence / seasonality (Landsat) | Global | Permanent-water mask (your old script's `seasonality ≥ 5`). |

---

## 3. Ancillary layers (for the unified ancillary pipeline)

| Layer | Product | Res. | Source (STAC where possible) | Role |
|---|---|---|---|---|
| DEM | Copernicus GLO-30 | 30 m | PC `cop-dem-glo-30`, AWS | Terrain flattening, slope mask. *(existing pipeline)* |
| DEM (bare-earth) | FABDEM | 30 m | Univ. Bristol (non-commercial licence) | Urban HAND without building bias. |
| DEM (national) | CartoDEM v3 | 30 m | Bhoonidhi (open) | Indian cross-check. |
| HAND | GLO-30-derived HAND (ASF, AWS open data) or compute (pysheds / whitebox) | 30 m | AWS | Exclude high-HAND false positives (shadow, dark tarmac). |
| Land cover | ESA WorldCover 2020/2021 | 10 m | PC `esa-worldcover` | Urban / rural / vegetation strata for per-stratum metrics. |
| Built-up | GHSL GHS-BUILT-S / -H | 10–100 m | JRC | Built-up fraction + height → double-bounce likelihood. |
| Buildings | Google Open Buildings 2.5D temporal / Microsoft / Overture / GlobalBuildingAtlas | footprint | cloud buckets | Building orientation vs. S1 azimuth (cardinal effect). |
| Permanent water | JRC GSW | 30 m | PC `jrc-gsw` | Flood = observed water − reference water. |
| Optical | Sentinel-2 L2A | 10 m | PC `sentinel-2-l2a` | Urban/forest masks (Attention U-Net), weak labels via NDWI/MNDWI when cloud-free. |
| Rainfall (context) | IMERG / IMD gridded | 0.1° / 0.25° | NASA GES DISC / IMD | Event timing only. |

---

## 4. Access routes & tooling (corrections to the plan)

- **STAC-only is feasible for intensity** (PC RTC/GRD, CDSE GRD, S2, DEMs, WorldCover, GSW). It is **not** feasible for coherence: SLC is on CDSE and ASF, not on PC. Plan: STAC for everything; CDSE OData/STAC or ASF burst extraction for SLC.
- **ASF**: free Earthdata login only. Burst-level SLC extraction (`asf_search` + `burst2safe`) downloads ~single bursts (hundreds of MB) instead of 4–8 GB frames. This makes laptop-scale InSAR coherence realistic.
- **SNAP / pyroSAR / sarsen**: sensor-generic, globally valid — no India limitation. `sarsen` (pure Python RTC) pairs well with PC GRD.
- **InSAR processors** for coherence: ISCE2 `topsStack` (burst-aware), SNAP `Back-Geocoding → Coherence`, GMTSAR. For coherence only (no unwrapping), SNAP gpt on bursts is the lightest; ISCE2 is the most reproducible.
- OPERA products (RTC-S1, CSLC-S1, DSWx-S1) are **North America only** — not usable here.

## 5. Storage / compute budget (laptop)

| Item | Per unit | Kerala AOI (~5.8k km², 3 dates) |
|---|---|---|
| RTC COG windowed read (VV+VH, float32, 10 m) | ~0.8 GB per 100×100 km per date (uncompressed) | ~1.4 GB (5.8e7 px × 2 pol × 3 dates × 4 B) |
| GRD zip (full frame) | ~1 GB | ~6 GB (2 frames × 3 dates) |
| SLC full frame | 4–8 GB | 25–50 GB |
| SLC bursts (AOI only) | ~0.15–0.3 GB/burst/pol | ~5–10 GB |
| Kuro Siwo / UrbanSARFloods / Sen1Floods11 | ~tens of GB each | — |

Run `sysprobe.sh` (attached) and send the output. The GPU/VRAM line decides which DL models are trainable locally. If it is the ThinkPad P53s with a 2 GB Quadro, SegFormer-B2 / Swin-UNet training is not realistic. The fallback is small encoders (ResNet-18/34, MiT-B0), 256² chips, AMP, and gradient accumulation.
