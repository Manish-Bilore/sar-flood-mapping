# 02 — Polarimetry Brief: What VV / VH / HH / HV Mean for Flood Mapping

Depth: enough to reason about signals and failure modes in urban vs rural floods. No decomposition maths beyond what changes a design decision.

---

## 1. Polarisation notation

A SAR antenna transmits and receives linearly polarised waves. The notation is **transmit-then-receive**:

| Channel | Transmit | Receive | Type | Typical sensor |
|---|---|---|---|---|
| **VV** | V | V | like / co-pol | Sentinel-1 IW (land default) |
| **VH** | V | H | cross-pol | Sentinel-1 IW |
| **HH** | H | H | co-pol | ALOS-2, NISAR L, TerraSAR-X, RISAT |
| **HV** | H | V | cross-pol | ALOS-2, NISAR L |

- **Dual-pol**: one transmit polarisation, two receive polarisations. Sentinel-1 is VV+VH over land; HH+HV mostly over polar areas.
- **Quad / full-pol**: HH, HV, VH, VV, with phase between channels. This gives the full scattering matrix. Examples: EOS-04 FRS-2 and some NISAR modes.
- **Hybrid / compact-pol**: circular transmit, linear receive (RISAT, EOS-04 hybrid mode). It approximates part of the quad-pol information at a wider swath.

**Units**: σ⁰ (sigma-nought) and γ⁰ (gamma-nought, terrain-flattened) are *linear power ratios*. dB = 10·log₁₀(linear).
- Average, filter, and form ratios **in linear space**.
- Convert to dB **only for display and thresholding**.
- A "VV/VH ratio" computed on dB values (as in the old script) is meaningless. Use VV_dB − VH_dB, which equals 10·log₁₀(VV/VH) in linear.

## 2. Three scattering mechanisms (everything follows from these)

| Mechanism | Geometry | Channel response | Flood relevance |
|---|---|---|---|
| **Specular / surface** | Smooth plane reflects energy away from the sensor | Very low co-pol and cross-pol | **Open floodwater looks dark.** Typical S1 values: VV ≈ −18 to −25 dB, VH ≈ −24 to −30 dB. |
| **Volume** | Random multiple scattering in a canopy or crop | Cross-pol (VH/HV) relatively strong | Dry vegetation is bright in VH. When floodwater sits beneath vegetation, the volume term changes. |
| **Double-bounce (dihedral)** | Horizontal surface + vertical wall or trunk form a corner reflector | Strong **co-pol**, especially HH; weak cross-pol | **Flooding under vegetation or between buildings makes pixels brighter.** Water is a better mirror than soil or asphalt. |

Core consequence: **flooding can decrease backscatter (open areas) or increase it (vegetation, urban).** A "darker-after" rule alone is structurally blind to half the problem.

## 3. What each channel is good and bad at

| Channel | Strengths | Weaknesses |
|---|---|---|
| **VV** | Highest water/land contrast in open areas. Best single band (Copernicus GFM uses VV). | Wind or rain roughening of water raises VV → missed water. Sensitive to wet soil. |
| **VH** | Less sensitive to wind on water. Separates water from vegetation well. | Lower SNR; near the noise floor (~−25 to −28 dB) over water. Weak over urban areas. |
| **HH** (L/X/C) | Strongest double-bounce, so the best channel for **flooded vegetation**, especially at L-band. | Not available on S1 over India. |
| **HV** | Volume and biomass indicator. | Low SNR over water. |
| **VV − VH (dB)** | Dimensionless and robust to incidence angle. High over double-bounce and bare surfaces, lower over vegetation. | Noisy where VH hits the noise floor. |

## 4. Frequency (wavelength) matters more than polarisation for vegetation

| Band | λ | Penetrates | Implication |
|---|---|---|---|
| X | 3 cm | Almost nothing | Sees urban geometry at 1 m, but blind to water under canopy. |
| **C (S1, EOS-04)** | 5.6 cm | Sparse crops, early-stage rice, thin canopy | Misses water under dense forest or plantation (Kerala uplands, Assam forests). |
| S (NISAR-S, NovaSAR) | ~10 cm | Moderate canopy | Intermediate. |
| **L (NISAR, ALOS-2)** | 24 cm | Forest canopy to trunks | HH double-bounce reveals flooded forest and plantations. Also more stable **coherence**. |

## 5. InSAR coherence — the non-polarimetric signal that matters for cities

Coherence **|γ|** ∈ [0, 1] measures how similar the complex phase-and-amplitude pattern is between two dates. It needs **SLC** data, not GRD.

| Surface | Pre-event coherence | Co-event coherence (one image during the flood) |
|---|---|---|
| Buildings, roads (stable) | High (0.6–0.9) | **Drops sharply** when water enters the resolution cell |
| Open water | ≈ 0 | ≈ 0 |
| Vegetation | Low (decorrelates in days at C-band) | Low → little information |
| Bare soil (dry) | Moderate | Drops with wetting or ploughing (a false-positive source) |

Urban flood signature = **pre-event coherence high AND co-event coherence low AND intensity unchanged or increased.** GRD intensity alone cannot detect this.

## 6. Geometry artefacts (urban + terrain)

| Artefact | Cause | Effect | Mitigation |
|---|---|---|---|
| **Radar shadow** | Tall object or steep slope hides the ground from the beam | Dark → **false flood** | Shadow/layover mask from DEM + building heights; HAND / slope masks |
| **Layover** | Top of a building is closer to the radar than its base | Bright smear | Mask from DSM + orbit geometry |
| **Cardinal / orientation effect** | Streets and façades parallel to the flight direction give strong dihedral returns | Double-bounce depends on street orientation relative to azimuth | Use building orientation (footprints) as a model feature; compare ascending vs descending |
| **Incidence-angle dependence** | S1 IW spans 29–46° across the swath | Same surface is darker at far range | Use RTC γ⁰; compare same-orbit pre/post only (never mix orbits in a change pair) |
| **Speckle** | Coherent interference | Salt-and-pepper noise | Refined Lee / multitemporal filter in **linear** space, or let the CNN learn it |

## 7. Implications: rural vs urban

| Setting | Dominant flood signal (C-band S1) | Best inputs | Main error sources |
|---|---|---|---|
| **Rural – open fields, bare, early crops** (Bihar, Assam floodplain, Kuttanad paddies) | Strong **decrease** in VV and VH | Post VV/VH, pre-post log-ratio, HAND, GSW | Wind-roughened water (VV); smooth dry sand bars and harvested fields look like water; shadow in hills |
| **Rural – flooded vegetation** (tall crops, plantations, forest) | **Increase** in co-pol (double-bounce), often small at C-band | VV increase + VV−VH change; L-band HH (NISAR/ALOS-2) if available | C-band cannot see under dense canopy → systematic under-detection |
| **Urban – wide open water** (low-lying colonies, parks, lake overflow) | Decrease (like rural) | Same as rural | Shadow false positives next to tall buildings |
| **Urban – streets between buildings** (Chennai, Hyderabad, Bengaluru ORR) | **Increase** in VV (double-bounce) and/or **coherence loss** | Pre-event and co-event coherence, VV change, building density/orientation | 10–20 m pixels are larger than street widths; revisit misses short flash floods; shadow and layover |
| **Dense old core** | Often no detectable change | Coherence only | Sub-pixel water; partial occlusion |

## 8. Design decisions this implies for the project

1. **Two-sided change detection**: model both backscatter decrease *and* increase, per stratum. Do not use a single "after < before" threshold.
2. **Work in linear γ⁰**; compute ratios and differences correctly; build change pairs from the **same relative orbit only**.
3. **Urban track requires SLC coherence** (pre-event + co-event pair). Budget it via burst extraction.
4. **Masks** (HAND, slope, shadow/layover, GSW permanent water) are inputs or post-filters, not afterthoughts. Report how much area each mask removes.
5. **Report metrics per stratum** (WorldCover built-up vs cropland vs tree cover). Aggregate F1 hides the urban failure mode.
6. The comment on findings should state explicitly how much **urban under-detection** remains with intensity-only models versus intensity + coherence. That comparison is the scientific contribution.
