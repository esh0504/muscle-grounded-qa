# -*- coding: utf-8 -*-
"""Step 2 sampling design — loader for settings/sampling.yaml.

Exposes the constants under their internal names (RATIOS, LEVELS, PAIR_AMP,
NEIGHBOR_STEP, ANCHOR_SELF_FRAC, NACT_W, TRIPLE_N, POOL_TOTAL, SEED).
"""
import os as _os

import yaml as _yaml

_YML = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                     "settings", "sampling.yaml")
with open(_YML, encoding="utf-8") as _f:
    _S = _yaml.safe_load(_f)

POOL_TOTAL = int(_S["total"])
SEED = int(_S["seed"])
REST_FIRST = bool(_S.get("rest_first", True))
RATIOS = {k: float(v) for k, v in _S["ratios"].items()}
TRIPLE_N = int(_S["triple_n"])
LEVELS = [float(x) for x in _S["levels"]]
PAIR_AMP = [float(x) for x in _S["pair_amp"]]
NEIGHBOR_STEP = float(_S["neighbor_step"])
ANCHOR_SELF_FRAC = float(_S["anchor_self_frac"])
NACT_W = {int(k): float(v) for k, v in _S["nact_w"].items()}

assert abs(sum(RATIOS.values()) - 1.0) < 1e-9, "sampling.yaml: ratios must sum to 1"
