# Step 1 — Sampling & Anchor Design

The only human-authored physical inputs of the whole pipeline (dashed boxes in
Fig. 1 of the paper). Everything downstream is automatic.

## What is specified here

- **12 vowel centers + 8 consonant-class rules** — `settings/anchors.yaml`
  (loaded by `modules/anchors.py`; AMAX=0.9, vowel budget 2.0 / consonant 2.3).
  Vowels are Gaussian centers in 11-D activation space; consonants are
  activation *bands* on their defining muscles (place-of-articulation based).
- **Section mixture** — `settings/sampling.yaml` (loaded by
  `modules/sampling_design.py`): enumerated
  REST / SINGLE / PAIR / TRIPLE plus filled
  anchor 0.35 / neighbor 0.20 / effort 0.15 / spacefill 0.30.
- **`settings/centers.csv`** — the QA-facing anchor target table (vowels incl. /æ/,
  consonant classes, functional targets) with per-row provenance
  (`lit` = direct literature pattern, `interp` = interpolation between
  literature endpoints, `anat` = anatomically self-evident intrinsic action).
  Used by Step 5 as prescriptive/target-directed anchors.

## References behind the `refs` tags

B09 Buchaillard, Perrier & Payan (2009) JASA 126(4) · H17 Harandi et al. (2017)
JASA 141(4) · TH07 Takano & Honda (2007) Speech Comm. 49 · H96 Honda (1996)
J. Phonetics 24 · S12r Stavness et al. (2012) JASA 131(5) · X22 tongue
fiber-strain MRI (PMC8744002) · ST25 Strycharczuk et al. (2025) Language and
Speech · C24 The Compartmental Tongue (2024) JSLHR · F06 Flemming, "The
Phonetics of Schwa Vowels".

The literature provides *which muscles dominate and in which direction*; the
normalized 11-D values are model-specific assignments reflecting those patterns,
not copied numbers (validated downstream via geometry/PCA/F1-F2 checks).

## Check

```bash
python scripts/step1_check_design.py    # budgets, bands, centers.csv consistency, section ratios
```
