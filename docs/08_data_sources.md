# 08 — Data sources, access and licences

Everything below is open data. Nothing in this repository redistributes it: the scripts fetch it, and `data/` is
git-ignored.

## SAR

| Dataset | Access | Licence / terms | Used for |
|---|---|---|---|
| Sentinel-1 IW RTC γ⁰ (10 m) | Microsoft Planetary Computer STAC, `sentinel-1-rtc` | Copernicus (free, full, open); PC terms of use | co-event scenes, dry-season medians, all intensity work |
| Sentinel-1 SLC bursts | ASF via `asf_search`, processed by ASF **HyP3** (`insar_isce_burst`) | Copernicus; HyP3 needs a free NASA Earthdata login and spends processing credits | InSAR coherence (10×2 looks ≈ 40 m) |
| Sen1Floods11 | Public Google Cloud bucket / GitHub (Cloud to Street + Google) | CC BY 4.0 | training and testing the 2-class segmentation models |
| UrbanSARFloods | Public release (TUM) | as published by the authors | 3-class (non-flood / open flood / urban flood) models, urban signature reference |

## Ancillary

| Dataset | Access | Licence | Used for |
|---|---|---|---|
| Copernicus DEM GLO-30 | Planetary Computer STAC | ESA / Copernicus DEM licence (free for any use, attribution) | HAND (via `pysheds`), slope |
| JRC Global Surface Water v1.4 | Planetary Computer STAC | © EC JRC, free re-use with attribution | permanent-water mask (seasonality ≥ 10), sea detection |
| ESA WorldCover 2021 v200 | Planetary Computer STAC | CC BY 4.0 | land-cover attribution of detections, built-up definition |
| geoBoundaries gbOpen, India ADM2 | geoboundaries.org API | CC BY 4.0 | district polygons and per-district areas |

## Reference / comparison material

| Item | Access | Terms | Used for |
|---|---|---|---|
| NRSC / NDEM flood inundation sheet, Kerala 21 Aug 2018 (RADARSAT-2 primary, Sentinel-1A secondary, 50 m) | `ndem.nrsc.gov.in` PDF | Government of India / ISRO product, reproduced with attribution for non-commercial research | qualitative and agreement comparison (§9 of the event report) |
| Published Kerala accuracies (PLOS One 2020 OA 94.1 %, Current Science 2021 CSI 81.6 %) | journals | cited, not redistributed | context for achievable accuracy |
| DeepSARFlood (2025) reported test IoU 0.72 | paper | cited | benchmark context for our models |

**Not available for this event:** Copernicus EMS Rapid Mapping has no activation for Kerala 2018 (EMS is triggered by
authorised users and India generally does not request it). Sentinel-2 is unusable at peak monsoon. Whether the Copernicus
GFM archive reaches back to 2018 is unverified. A pixel-level reference therefore remains the main outstanding gap —
UNOSAT products or a hand-digitised sample over a few validation tiles are the realistic routes.

## Credentials the pipeline expects

- **Planetary Computer**: none for the collections used here (anonymous STAC + signed asset URLs).
- **NASA Earthdata** (HyP3 / `asf_search`): `~/.netrc` entry for `urs.earthdata.nasa.gov`, mode 600.
- No API keys are stored in the repository or in `config/events.yaml`.

## Attribution to carry into any figure reused elsewhere

- "Contains modified Copernicus Sentinel data (2018–2024), processed by Microsoft Planetary Computer / ASF HyP3."
- "Copernicus DEM © ESA"; "Global Surface Water © EC JRC"; "ESA WorldCover 2021"; "Boundaries © geoBoundaries (CC BY 4.0)".
- "Flood inundation reference: NRSC/ISRO, NDEM."
