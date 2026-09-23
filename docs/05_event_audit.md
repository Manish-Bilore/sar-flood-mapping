# 05 — Sentinel-1 Acquisition Audit and Final Event Set

Run: `src/acquisition_audit.py`, 20 Sep 2026. Source: ASF SLC catalogue (all S1 IW passes) + PC `sentinel-1-rtc`. Δpeak = days outside the published peak window.

| Event | Best co-event scene | Δpeak | Dry baseline (same path) | Coherence pre-pair | Verdict |
|---|---|---|---|---|---|
| Kerala 2018 | 2018-08-21 S1A p165 DESC | 0 | 10 | yes | **keep, anchor** |
| Assam 2020 | 2020-07-15 S1A p143 ASC (11 scenes in window) | 0 | 8 | yes | keep |
| Assam 2022 | 2022-06-16 p41 ASC; comparator 2022-08-10 p143 | 0 / +49 | 7 | yes | keep (DeepSARFlood AOI) |
| Bihar 2020 | 2020-07-26 S1A p121 DESC | 0 | 12 | yes | keep |
| Bengaluru 2022 | 2022-09-05 S1A p165 DESC (~06:00 IST, hours after the 4 Sep storm) | 0 | 9 | yes | **keep** — better than expected |
| Delhi 2023 | 2023-07-16 p27 ASC; series 12/16/19/24 Jul | 0 | 10 | yes | keep, with a time series |
| Hyderabad 2020 | 2020-10-21 p165 DESC (clean pre: 10-09) | +3 (+7 after main peak) | 12 | yes | keep as **residual-inundation / detectability** case |
| Mumbai 2024 | 2024-07-11 p34 DESC | +2 | 10 | yes | provisional; pluvial water drains within hours → date needs checking |
| Chennai 2023 | **none, 1–18 Dec 2023** | — | — | — | **drop** (EOS-04 only; qualitative at most) |
| Wayanad 2024 | 2024-08-01 p165 | 0 | 9 | yes | **drop** (landslide, not inundation) |

## Findings to carry into the write-up

1. **The 2024 GEE baseline was contaminated.** The old script's "before" window (15 Jul–10 Aug 2018) includes 9 Aug, which is already flood onset (the PLOS One 2020 study maps flooding on 9 Aug). Its "change" therefore under-states the flood. The new baseline is a same-path dry-season median (Jan–Apr 2018, p165).
2. **Mixed-orbit mosaics:** Kerala has DESC p165 and ASC p173 on 21 Aug. The old `.mosaic()` over date windows could mix geometries. The new change pairs use one relative orbit only.
3. **Revisit loss is real but event-specific.** Chennai 2023 has no S1 image at all. Bengaluru, Delhi and Mumbai still got a pass in the window despite S1A flying alone.
4. **Acquisition time matters for pluvial floods.** DESC ≈ 00:30 UTC (06:00 IST), ASC ≈ 13:00 UTC (18:30 IST). Bengaluru's DESC pass came a few hours after the rain, which is the best case for urban pluvial detection.
5. Every kept event has a dry baseline and a coherence pre-pair on the co-event path, so the urban coherence track is feasible for all four urban events.

Config: `config/events.yaml`.
