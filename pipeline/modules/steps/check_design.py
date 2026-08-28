#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 1 — sanity-check the human-authored specification.

Verifies:
  - every anchor center / rule respects AMAX and the activation budgets
  - centers.csv is consistent with modules.anchors for the 12 shared vowels
  - prints the full anchor table (the paper's "human-specified once" inputs)

Usage: python -m modules.steps.check_design
"""
import os

import numpy as np
import pandas as pd

from ..anchors import AMAX, BUDGET, CBUDGET, CRULES, VCENTERS
from ..config import PIPELINE_ROOT
from ..muscles import MUS
from ..sampling_design import RATIOS

SETTINGS = os.path.join(PIPELINE_ROOT, "settings")


def main():
    ok = True
    print("== vowel centers (12) ==")
    for key, ((ipa, sig, refs), vec) in VCENTERS.items():
        v = np.asarray(vec, float)
        assert len(v) == len(MUS)
        if v.max() > AMAX + 1e-9:
            print(f"  !! {key}: max {v.max()} > AMAX {AMAX}")
            ok = False
        if v.sum() > BUDGET + 1e-9:
            print(f"  !! {key}: sum {v.sum():.2f} > BUDGET {BUDGET}")
            ok = False
        print(f"  {key:10s} /{ipa}/ sigma={sig} sum={v.sum():.2f}  refs={refs}")

    print("\n== consonant rules (8) ==")
    for key, ((ipa, refs), rule) in CRULES.items():
        for m, (lo, hi) in rule.items():
            if not (0 <= lo <= hi <= AMAX + 1e-9):
                print(f"  !! {key}: {m} band ({lo},{hi}) outside [0,{AMAX}]")
                ok = False
        mid = sum((lo + hi) / 2 for lo, hi in rule.values())
        print(f"  {key:12s} /{ipa}/ defining={list(rule)} band-mid sum={mid:.2f} (cap {CBUDGET})  refs={refs}")

    print("\n== centers.csv (QA anchor targets) ==")
    cen = pd.read_csv(os.path.join(SETTINGS, "centers.csv"))
    print(f"  {len(cen)} rows: " + ", ".join(f"{c}={n}" for c, n in cen.category.value_counts().items()))
    byipa = {VCENTERS[k][0][0]: np.asarray(VCENTERS[k][1], float) for k in VCENTERS}
    for _, r in cen.iterrows():
        if r["category"] == "vowel" and r["ipa"] in byipa:
            v = np.array([float(r[m]) for m in MUS])
            if not np.allclose(v, byipa[r["ipa"]], atol=1e-9):
                print(f"  !! centers.csv /{r['ipa']}/ differs from modules.anchors")
                ok = False

    assert abs(sum(RATIOS.values()) - 1.0) < 1e-9, "section ratios must sum to 1"
    print("\nsection ratios:", RATIOS)
    print("OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
