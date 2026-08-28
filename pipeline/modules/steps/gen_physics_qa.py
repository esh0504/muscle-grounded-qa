#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5d — deterministic physics QA at scale (Cause / State / Change chain).

One multiturn record per VALID mesh, built from structured facts only
(pool metadata + validity meta + regional displacement + Step-1 anchor targets):
    A2  muscle attribution           D1  motor-equivalence abstention
    C1  volume preservation          B1  single-muscle intervention (single section)
    B2  neighbor counterfactual      prescriptive  target-directed correction

Surface forms come from step4_qa_templates (KO/EN share this generator).
Shard-level resume: existing output files are skipped.

Usage:
    python -m modules.steps.gen_physics_qa --lang ko 0 295157
    python -m modules.steps.gen_physics_qa --lang en 0 295157 --outdir my_qa_dir
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from ..qa_templates import BY_LANG
from ..config import PIPELINE_ROOT, ensure_dir, path as cfg_path
from ..mesh_io import load_meta_all, load_pool_meta, valid_set
from ..muscles import MUS

TSEL_CANON = ["i", "a", "u", "t d s", "k g"]   # representative prescriptive targets


def big_region(r):
    mags = [r[2], r[5], r[8]]
    return ["front", "mid", "back"][int(np.argmax(mags))]


def load_targets():
    cen = pd.read_csv(os.path.join(PIPELINE_ROOT, "settings", "centers.csv"))
    return {r["ipa"]: {"key": r["key"], "vec": {m: float(r[m]) for m in MUS}, "label": r["label_ko"]}
            for _, r in cen.iterrows()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=sorted(BY_LANG), required=True)
    ap.add_argument("start", type=int)
    ap.add_argument("end", type=int)
    ap.add_argument("--outdir", default=None, help="default: {qa_out}/physics_{lang}")
    a = ap.parse_args()

    T = BY_LANG[a.lang]
    outdir = a.outdir or os.path.join(cfg_path("qa_out"), "physics_" + a.lang)
    ensure_dir(outdir)

    def gpt(v):
        return {"from": "gpt", "value": v, "mask_spans": T.physics_mask_spans(v)}

    def hum(v):
        return {"from": "human", "value": v}

    # data
    z = np.load(os.path.join(cfg_path("index_dir"), "region_disp.npz"), allow_pickle=True)
    _D = np.asarray(z["disp"])
    _I = np.asarray(z["idxs"])
    RD = {int(i): _D[k] * 1000.0 for k, i in enumerate(_I)}          # m -> mm
    info = load_pool_meta()
    metas = load_meta_all()
    VALID = valid_set(metas)
    TARGETS = load_targets()
    TSEL = [t for t in TSEL_CANON if t in TARGETS]

    nsh = 0
    nqa = 0
    for sh in range(a.start // 1000, (a.end + 999) // 1000):
        outp = os.path.join(outdir, f"{T.PHYSICS_PREFIX}_{sh:05d}.jsonl")
        if os.path.exists(outp):
            continue
        lo, hi = sh * 1000, sh * 1000 + 1000
        with open(outp, "w", encoding="utf-8") as fo:
            for idx in range(lo, min(hi, a.end)):
                if idx not in VALID or idx not in info.index or idx not in RD:
                    continue
                r = info.loc[idx]
                rd = RD[idx]
                vr = float(metas.loc[idx, "vol_ratio"])
                mv = {m: float(r[m]) for m in MUS}
                active = [m for m in MUS if mv[m] > 0.05]
                ref = {"verts_shard": sh, "row_in_shard": idx % 1000}
                conv = []
                ttypes = []
                # A2 attribution + D1 abstention
                if active:
                    q, ans = T.t_attribution(active)
                    conv += [hum(q), gpt(ans)]
                    q, ans = T.t_identifiability()
                    conv += [hum(q), gpt(ans)]
                    ttypes += ["A2", "D1"]
                # C1 volume preservation
                br = big_region(rd)
                bi = {"front": 0, "mid": 3, "back": 6}[br]
                h, v = T.hv(rd[bi], rd[bi + 1])
                q, ans = T.t_volume(vr, br, h, v)
                conv += [hum(q), gpt(ans)]
                ttypes.append("C1")
                # B1 single-muscle intervention
                if r["section"] == "single" and active:
                    q, ans = T.t_single(active[0], br, h, v, rd)
                    conv += [hum(q), gpt(ans)]
                    ttypes.append("B1")
                # B2 neighbor counterfactual
                if r["section"] == "neighbor" and int(r["base_index"]) in RD:
                    b = RD[int(r["base_index"])]
                    dd = [(rd[i] - b[i]) for i in range(9)]
                    k = int(np.argmax([abs(dd[0]) + abs(dd[1]),
                                       abs(dd[3]) + abs(dd[4]),
                                       abs(dd[6]) + abs(dd[7])]))
                    q, ans = T.t_counterfactual(r["delta"], int(r["base_index"]), dd, k)
                    conv += [hum(q), gpt(ans)]
                    ttypes.append("B2")
                # prescriptive target-directed correction
                if active is not None:
                    tgt = TSEL[idx % len(TSEL)]
                    tv = TARGETS[tgt]["vec"]
                    inc = [m for m in MUS if tv[m] - mv.get(m, 0) > 0.12]
                    dec = [m for m in MUS if mv.get(m, 0) - tv[m] > 0.12]
                    q, ans = T.t_prescriptive(tgt, TARGETS[tgt], inc, dec)
                    conv += [hum(q), gpt(ans)]
                    ttypes.append("prescriptive")
                if len(conv) >= 6:
                    fo.write(json.dumps({"index": idx, "mesh_ref": ref, "scenario": "physics_chain",
                                         "lang": a.lang, "conversations": conv, "turn_types": ttypes,
                                         "verified": True}, ensure_ascii=False) + "\n")
                    nqa += 1
        nsh += 1
    print(f"physics({a.lang}) shards {a.start//1000}~{(a.end+999)//1000-1}: {nsh} files, {nqa} convos")


if __name__ == "__main__":
    main()
