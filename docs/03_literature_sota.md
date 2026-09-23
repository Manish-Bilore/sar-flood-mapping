# 03 — Literature and State of the Art: SAR Flood Mapping (Urban and Rural)

Purpose: orientation, not a formal review. Venue and year are given so each item is findable. Items marked *(preprint)* are arXiv, not yet peer-reviewed.

---

## 1. Method families at a glance

```
Unsupervised / rule-based ──► Change detection ──► ML/DL on intensity ──► DL on intensity + coherence (urban)
   (thresholding)              (pre vs post)       (CNN → transformer)      + uncertainty, ensembles
```

| Family | Core idea | Rural performance | Urban performance | Status |
|---|---|---|---|---|
| Global thresholding (Otsu, Kittler–Illingworth) | Water is the dark mode of a bimodal histogram | Good when the scene is bimodal | Poor | Baseline everywhere; common in Indian case studies (GEE + Otsu) |
| Split-based / hierarchical tiling thresholds | Find tiles that *are* bimodal; derive local thresholds | Good, automatic | Poor | Operational core (Martinis 2009 NHESS; Chini 2017 TGRS) |
| Change detection (log-ratio, NCI, z-score vs dry-season stack) | Flood = anomalous drop vs baseline | Very good; removes permanent water and dark soils | Poor if it is drop-only | Standard; your old script is a crude version (DeVries 2020 RSE: S1 time-series z-score) |
| Bayesian / fuzzy / probabilistic | Water probability from class-conditional PDFs + priors (HAND, land cover) | Good + uncertainty | Moderate | Giustarini 2016 TGRS; one of the GFM algorithms |
| **Operational ensemble** | Three algorithms, majority vote, likelihood + exclusion mask | Good | **Explicitly excludes** most urban areas | Copernicus **GFM** (2021 →, archive 2015 →; RSE 2025 overview) |
| Classical ML (RF, SVM, CART) | Pixel features (VV, VH, ratio, texture, DEM) | Good with labels | Moderate | Common in regional studies (e.g. Delhi 2023 CART) |
| **CNN segmentation** (U-Net, Attention U-Net, U-Net++, DeepLabv3+) | Learn spatial context | F1 ~0.7–0.85 on benchmarks | Weak without coherence | Mature baseline (Bonafilia 2020 Sen1Floods11; Bereczky 2022 JSTARS: CNNs vs DLR rule-based chain) |
| **Transformers / hybrids** (SegFormer, Swin-UNet, ViT-CNN hybrids) | Long-range context; better cross-event generalisation | Current top scores | Limited evidence | DeepSARFlood (2025) IoU 0.72 Sen1Floods11 SOTA; DAM-Net (2024 ISPRS JPRS) change-detection ViT |
| **Urban: intensity + coherence** | Double-bounce increase + coherence loss | — | **Only approach that works reliably** | Chini 2019 (Houston coherence); Li 2019 ISPRS (TerraSAR-X, self-learning CNN); Zhao 2022 urban-aware U-Net; UrbanSARFloods 2024; Soudagar 2025 GMM |
| Foundation models (Prithvi, TerraMind, DOFA, SSL4EO) | Pretrain → fine-tune | Strong | Unclear | **Out of scope** (separate project) |

## 2. Rural / open-area state of the art

- **Solved-ish problem at 10–20 m.** Change detection + HAND/slope masking + permanent-water removal gives OA > 90% in most published Indian studies. The errors that remain are structural, not algorithmic:
  - wind-roughened water;
  - dry smooth surfaces (sand bars, harvested paddies) that look like water;
  - **flooded vegetation**: at C-band it is dim or absent; L-band HH or ALOS-2 is needed (Plank 2017; Tsyganskaya 2018 review).
- **DL benchmarks**:
  - Sen1Floods11 is the common yardstick. DeepSARFlood (Sharma & Saharia, IIT Delhi, *Sci. Remote Sens.* 2025) reaches IoU 0.72 with ViT/CNN-ViT deep ensembles trained on weak optical labels, with uncertainty estimates. Its demonstrations include **Assam 2020 and the Brahmaputra 2022** floods. It is the closest Indian SOTA comparator, with open code and weights.
  - Kuro Siwo (NeurIPS 2024): manual labels, 43 events; BlackBench baselines exceed F1 0.80 for flood.
- **Generalisation is the open issue.**
  - Portales-Julia et al. (*RSE* 2025) trained across Kuro Siwo, WorldFloods and S1S2-Water. Cross-dataset scores drop markedly, and **label protocol differences** explain much of the gap.
  - Implication: report in-distribution **and** cross-event scores, and never compare an F1 across different label sets as if equal.
- **Scale**: Misra et al. (*Nat. Commun.* 2025) mapped global floods from ten years of S1 with a DL model. The GFM archive does the same with rule-based methods. Both are usable reference layers.
- **Metrics hygiene**:
  - Many Indian studies report **Overall Accuracy**, which is inflated by the dominant dry class. It is not comparable to F1/IoU.
  - Use F1, IoU, precision, recall, and kappa, per stratum.

## 3. Urban state of the art

- **Why it is hard**: double-bounce (brighter, not darker), shadow and layover, 10–20 m pixels vs 6–12 m streets, flash floods shorter than the revisit. Operational services, GFM included, largely mask urban areas out.
- **What works**:
  1. **Coherence loss**. Pre-event pair high, co-event pair low. First shown at scale with S1 for Houston/Harvey (Chini 2019). Coherence reduces urban under-detection in every study that tested it.
  2. **Double-bounce increase**, conditioned on building orientation and a DSM. Mason and colleagues: TerraSAR-X work, then S1 + WorldDEM + World Settlement Footprint (*JARS* 2021).
  3. **DL on intensity + coherence stacks**:
     - Li 2019 (TerraSAR-X, Joso, Japan): active self-learning CNN.
     - Zhao 2022: urban-aware U-Net using a SAR-derived probabilistic urban mask.
     - **UrbanSARFloods** (Zhao, Xiong, Zhu, CVPRW 2024): first benchmark with an *urban flood* class. Class imbalance and few urban samples remain the main failure. ImageNet pretraining helps little for 8-band SAR input.
  4. **Unsupervised GMM on intensity + coherence**: Soudagar et al. (ISPRS Annals 2025), 2023 Greece floods.
- **Open problems**:
  - no Indian urban benchmark;
  - C-band revisit vs flash-flood duration;
  - ground truth in cities is almost never published as maps;
  - effect of building density and orientation on detectability is under-quantified.

  **These are the gaps your project can target.**
- **Recent preprints (2026)**:
  - S1+S2 cross-sensor learning with generative despeckling (arXiv 2606.30511);
  - explainability of CNN vs transformer flood segmentation (arXiv 2606.16302);
  - VV/VH cross-pol fusion (arXiv 2605.02153).

## 4. Indian event literature and comparability

| Event | Type | S1 revisit at event | Published SAR work (examples) | Reported metric | Comparable? |
|---|---|---|---|---|---|
| **Kerala Aug 2018** (anchor) | Riverine + backwater, rural + peri-urban | 6 d (A+B) | Otsu on S1 in GEE (*PLOS One* 2020): OA 94.3% (9 Aug), 94.1% (21 Aug). Rule-based S1 time series + ALOS-2 for Alappuzha/Kottayam (*Current Science* 2021): OA 90.6%, CSI 81.6%. Vishnu et al. (*Geomatics NHR* 2019): S1 21 Aug vs S2 MNDWI baseline; Kuttanad/Kole depths. | OA, CSI | **Yes — best case.** Same districts; CSI is directly comparable to IoU. |
| **Hyderabad Oct 2020** | Pluvial urban flash flood | 6 d | Mainly **hydrodynamic** (HEC-RAS: Rangari, Bhatt, Umamahesh, *Current Science* 2021; a Springer chapter reports ~658 km² modelled extent). Little SAR-validated mapping found. | Modelled extent | Weak — model vs observation; check S1 dates vs 13–14 Oct peak. |
| **Bengaluru Sept 2022** | Pluvial urban (ORR, Bellandur, Mahadevapura) | **12 d (A only)** | No peer-reviewed SAR flood map found. Technical and insurance reports only. | — | **At risk** — acquisition may miss the flood; EOS-04 open data is the fallback. |
| **Delhi Jul 2023** (Yamuna) | Riverine into urban floodplain | 12 d | S1 dual-pol NCI change detection (ArcIndia News 2025, not peer-reviewed): onset 12 Jul, peak 16 Jul, recession 24 Jul. CART on S1 (AGU 2023 abstract). Open-source mapper on PC RTC (GitHub `yamuna-flood-mapper`). | Extent/timing, no accuracy | Partial — compare extent and timing. |
| **Chennai Dec 2023** (Michaung) | Pluvial + coastal urban | 12 d | Chennai 2015 SAR studies exist (NCI; underestimation in dense urban). For 2023, multi-factor comparisons (*Water* 2024). A Thoothukudi study notes **incomplete S1 coverage in Dec 2023**. | — | **At risk** — audit dates first; EOS-04 fallback (Bhoonidhi imaged Michaung). |
| **Wayanad Jul 2024** | **Landslide / debris flow**, not inundation | 12 d | Landslide literature, not flood mapping | — | **Recommend dropping or reframing** — the target class is not floodwater. |
| **Assam (2020, 2022)** | Riverine rural, Brahmaputra | 6 d (2020), 12 d (2022) | Otsu VV in GEE 2018–20 (Mudi 2022): ≥87% vs S2. Change detection 2015–20 (Vekaria, *JESS* 2022). **DeepSARFlood** Assam 2020 + Brahmaputra 10 Aug 2022 vs S2 labels. | OA; F1/IoU (DeepSARFlood) | **Yes** — DeepSARFlood is a direct DL comparator. |
| **Bihar / Kosi (2019, 2020)** | Riverine rural | 6 d | GEE threshold, Bihar 2020 paddy impact: OA 93.81%, κ 0.70. Kosi SAR depth estimation (Parida, *Geocarto* 2021). Kosi AHP risk 2015–20 (2025). | OA, κ | Yes (κ comparable). |

**Suggested urban replacement for Wayanad**: Mumbai. There is a multi-year S1 VV/VH study, 2018–25 (MDPI journal, 2026: "Spatio-Temporal Analysis of Urban Floods in Mumbai, India, Using Sentinel-1 SAR Data"), with ward-level extents. Its finding was that VV detects roughly twice the VH extent. You also already hold Mumbai building footprints and heights for the double-bounce/orientation analysis.

## 5. What the SOTA implies for the recreation

| Track | Baseline to beat | Target |
|---|---|---|
| Rural (Kerala, Assam, Bihar) | Otsu / change detection (published OA 87–94%); DeepSARFlood (IoU 0.72 on Sen1Floods11) | U-Net / Attention U-Net / ResNet-U-Net / SegFormer / Swin-UNet on Sen1Floods11 + Kuro Siwo; report IoU on Sen1Floods11 test **and** on the Kerala districts |
| Urban (Hyderabad, Delhi, Chennai, Bengaluru, Mumbai) | GFM (masks urban); intensity-only CNN | Intensity-only vs intensity + coherence (UrbanSARFloods-trained), per built-up stratum. The headline finding is the urban recall gain from coherence. |
| Your 2024 numbers | 82% accuracy, F1 0.823 (in-house labels) | Re-report on public test splits so it becomes comparable |

## 6. Key references (findable by title)

- Martinis, Twele, Voigt (2009) *NHESS* — split-based automatic thresholding (TerraSAR-X).
- Twele et al. (2016) *IJRS* — fully automated Sentinel-1 flood processing chain (HAND).
- Giustarini et al. (2016) *IEEE TGRS* — probabilistic flood mapping from SAR.
- Chini et al. (2017) *IEEE TGRS* — hierarchical split-based thresholding.
- Plank et al. (2017) — ALOS-2 + S1 flooded vegetation. Tsyganskaya et al. (2018) — SAR flooded-vegetation review.
- Chini et al. (2019) *Remote Sensing* — S1 InSAR coherence for urban floods, Houston/Harvey.
- Li et al. (2019) *ISPRS JPRS* — urban flood, active self-learning CNN, TerraSAR-X intensity + coherence.
- DeVries et al. (2020) *RSE* — S1 + Landsat rapid flood monitoring in GEE.
- Bonafilia et al. (2020) *CVPRW* — Sen1Floods11.
- Mason, Dance, Cloke (2021) *J. Appl. Remote Sens.* — urban floodwater with S1 + WorldDEM.
- Zhao et al. (2022) — urban-aware U-Net, multitemporal S1 intensity + coherence.
- Bentivoglio et al. (2022) *HESS* — review of DL for flood mapping.
- Bereczky et al. (2022) *IEEE JSTARS* — CNNs vs operational rule-based S1 chain.
- Zhao, Xiong, Zhu (2024) *CVPRW* — UrbanSARFloods; plus their review of urban SAR flood mapping (linked from the dataset repo).
- Bountos et al. (2024) *NeurIPS D&B* — Kuro Siwo / BlackBench.
- Saleh et al. (2024) *ISPRS JPRS* — DAM-Net, ViT change detection.
- Sharma & Saharia (2025) *Sci. Remote Sens.* 11:100203 — DeepSARFlood (code: github.com/hydrosenselab/DeepSARFlood).
- Portales-Julia, Mateo-García, Gómez-Chova (2025) *RSE* — cross-modality and cross-dataset flood models.
- Misra et al. (2025) *Nat. Commun.* 16:5762 — global floods from 10 years of S1.
- GFM team (2025) *RSE* — Sentinel-1 Global Flood Monitoring: challenges and directions.
- Soudagar et al. (2025) *ISPRS Annals* X-G — urban flood, intensity + coherence GMM.
- Oktay et al. (2018) — Attention U-Net (architecture reference).
