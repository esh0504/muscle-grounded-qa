# -*- coding: utf-8 -*-
"""Step 1 anchor specification — loader + helpers.

The DATA (12 vowel centers, 8 consonant rules, activation caps/budgets) is a
user setting in settings/anchors.yaml; this module loads it into the internal
structures (VCENTERS / CRULES, order-preserving) and adds derived helpers.
"""
import os as _os
import sys as _sys

import yaml as _yaml

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from .muscles import MI, D, MUS  # noqa: F401,E402

_YML = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                     "settings", "anchors.yaml")
with open(_YML, encoding="utf-8") as _f:
    _SPEC = _yaml.safe_load(_f)

AMAX = float(_SPEC["limits"]["amax"])
BUDGET = float(_SPEC["limits"]["budget"])
CBUDGET = float(_SPEC["limits"]["cbudget"])

# key -> ((ipa, sigma, refs), 11-D center vector)   — insertion order preserved
VCENTERS = {}
for _k, _v in _SPEC["vowels"].items():
    assert len(_v["vec"]) == D, f"anchors.yaml vowel '{_k}': vec must have {D} entries"
    VCENTERS[_k] = ((str(_v["ipa"]), float(_v["sigma"]), str(_v["refs"])),
                    [float(x) for x in _v["vec"]])

# key -> ((ipa, refs), {muscle: (lo, hi)})
CRULES = {}
for _k, _v in _SPEC["consonants"].items():
    _bands = {}
    for _m, _b in _v["bands"].items():
        assert _m in MI, f"anchors.yaml consonant '{_k}': unknown muscle '{_m}'"
        _bands[_m] = (float(_b[0]), float(_b[1]))
    CRULES[_k] = ((str(_v["ipa"]), str(_v["refs"])), _bands)

PHONEME_KEYS = list(VCENTERS.keys()) + list(CRULES.keys())   # 20 phoneme classes


def center_vec(key):
    """Nominal center vector for any anchor key (consonants: band midpoints)."""
    import numpy as np
    if key in VCENTERS:
        return np.array(VCENTERS[key][1], float)
    (_ipa, _refs), rule = CRULES[key]
    v = np.zeros(D)
    for m, (lo, hi) in rule.items():
        v[MI[m]] = (lo + hi) / 2
    return v
