# SAR flood mapping without Earth Engine

Sentinel-1 flood mapping for Indian events, rebuilt from public archives on a single laptop: classical thresholding and
change detection, five segmentation networks trained on Sen1Floods11, InSAR coherence from HyP3, and a parameterised
Quarto report that regenerates end to end for any configured event.

The anchor event is the **Kerala flood of August 2018**, a recreation of an earlier Google Earth Engine workflow by the
same author — including an audit of what that earlier rule got wrong.

| | |
|---|---|
| **Input** | Sentinel-1 IW RTC γ⁰ (Microsoft Planetary Computer STAC), SLC bursts via ASF HyP3 for coherence |
| **Ancillary** | Copernicus GLO-30 (HAND, slope), JRC Global Surface Water, ESA WorldCover 2021, geoBoundaries ADM2 |
| **Benchmarks** | Sen1Floods11 (2-class), UrbanSARFloods (3-class, urban flooding) |
| **Compute** | Ubuntu laptop, Quadro P520 (2 GB) — every result here was produced on it |
| **Stack** | Python (processing, training), R + Quarto (figures, report) |

## Headline results — Kerala, 21 August 2018

| Method | Flood extent |
|---|---|
| Otsu VV (−12.9 dB) | 573 km² |
| Split-based VV | 596 km² |
| Change detection vs dry-season median (−3 dB) | 440 km² |
| Deep learning (5 models, Sen1Floods11-trained) | 150–462 km² |
| **Final extent** (3 classical maps + 1 gated DL vote, ≥3 of 4) | **508 km²** |

Findings that came out of the exercise rather than the literature:

- **Benchmark IoU does not predict transfer.** The five networks sit within 0.663–0.681 test IoU on Sen1Floods11 yet map
  150–462 km² on the same scene. Architecture family, not benchmark rank, decides where they land.
- **The 2024 GEE rule is not a flood map.** `after_dB / before_dB > 1.1` demands a drop proportional to a pixel's own
  brightness, so a −7 dB canopy passes on 0.7 dB of speckle. 79 % of its detections are tree cover (Appendix A of the report).
- **Lee 5×5 beats Refined Lee here on both axes** — ENL 14.9 vs 9.1 *and* edge retention 0.85–0.92 vs 0.83–0.87 — against
  the usual ordering.
- **Coherence adds nothing for this event.** Vegetated and open-water classes sit at the estimator's noise floor (≈0.29) in
  both 12-day pairs; the urban signal survives only as a −0.043 double difference over 0.9 km².
- **Urban flooding is under-detected by construction.** Every intensity rule looks for a backscatter *drop*, while flooded
  streets get *brighter* through double bounce. The final extent is a lower bound over built-up land.

The full report — imagery, speckle statistics, VV/VH separability, coherence, ancillary masks, classical and DL maps,
agreement, sensitivity analysis, NRSC comparison, limitations — is one self-contained HTML file per event.

## Repository layout

```
config/events.yaml     one block per event: bbox, dates, orbit, districts, zoom windows, published comparators
src/                   acquisition audit -> fetching -> baselines -> training -> inference -> products
report/                event.qmd (parameterised), R/helpers.R, architecture figures, render.sh
docs/                  datasets, polarimetry, literature/SOTA, compute plan, event audit, results log
outputs/<event>/       tables + summary.json are committed; rasters are not (regenerate them)
```

Not in git: `data/` (tens of GB, all re-downloadable), `runs/` (model weights), rendered rasters.

## Reproducing an event

```bash
conda env create -f environment.yml && conda activate sarflood

python -u src/acquisition_audit.py                  # what S1 exists for each configured event
python -u src/fetch_event_rtc.py    <event>         # co-event scene + dry-season median (same relative orbit)
python -u src/fetch_ancillary.py    <event>         # HAND, slope, GSW, WorldCover, district polygons
python -u src/baseline_classical.py <event>         # Otsu, split-based, change detection, urban brightening

python -u src/train_s1f11.py --model all            # Sen1Floods11: U-Nets, DeepLabV3+, SegFormers
python -u src/predict_event.py      <event>         # apply them to the event
python -u src/coherence_hyp3.py     <event>         # optional: HyP3 burst InSAR (needs an Earthdata login)
python -u src/predict_event_usf.py  <event>         # optional: UrbanSARFloods 3-class models
python -u src/nrsc_reference.py     <event> --url … # optional: georeference an NRSC/NDEM sheet as a reference

python -u src/sensitivity.py        <event>         # ensemble / threshold sensitivity tables
python -u src/make_event_products.py <event>        # display rasters, zoom windows, every table, summary.json
cd report && bash render.sh         <event>         # -> report/_site/<event>.html
```

Adding an event means adding a block to `config/events.yaml` (bbox, co-event date and relative orbit, dry-season window).
`districts: auto` selects every ADM2 polygon intersecting the bbox, and zoom windows are centred on the largest flood
clusters when none are named.

## Method notes

- All arithmetic in **linear power**; decibels only for display and thresholds.
- Change detection compares against a **dry-season median of the same relative orbit**, never a single pre-event scene
  (for Kerala the 9 Aug scene is already flooded and covers part of the frame).
- Permanent water from JRC GSW seasonality ≥ 10 months; terrain exclusion HAND > 15 m and slope > 5°; objects under 8 px
  dropped. The sensitivity section quantifies what each of those choices costs.
- The final extent is a majority of three classical maps plus **one** vote for deep learning (the majority of networks
  passing a Sen1Floods11 test-IoU gate), so a handful of weak networks cannot outvote physically consistent baselines.
- Reported agreement against the NRSC rapid-assessment sheet is **agreement, not accuracy** — that product is generalised
  for print at 50 m. A pixel-level reference is still outstanding.

## Data sources and licences

See `docs/08_data_sources.md`. Everything used here is open data; the NRSC/NDEM sheet is reproduced with attribution as a
qualitative reference. Model weights and derived rasters are not redistributed.

## Author

Manish Bilore — [GitHub](https://github.com/Manish-Bilore) · [LinkedIn](https://www.linkedin.com/in/manish-b-44212084/)

Code is MIT licensed (`LICENSE`). The input datasets keep their own licences.
