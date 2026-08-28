#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5e — feature-grounded QA families A1 (shape description) + B3 (dose-response).

A1: per VALID+feat_ok mesh, 3 turns (quantitative shape / place interpretation /
    curvature) grounded 100% in the 32 extracted features.
B3: per single-muscle level sweep and per effort-combination sweep, one record
    stating the monotonicity tier (strict / near / partial) with Spearman rho and
    the level->response sequence.

Same record schema as gen_physics_qa. Surface forms from step4_qa_templates.

Usage:
    python -m modules.steps.gen_feature_qa --lang ko A1 0 295157
    python -m modules.steps.gen_feature_qa --lang ko B3
    python -m modules.steps.gen_feature_qa --lang en A1 0 295157 --outdir my_dir
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from ..qa_templates import BY_LANG
from ..config import ensure_dir, path as cfg_path


def spearman(x, y):
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _idx(T):
    d = cfg_path("index_dir")
    return d, pd.read_pickle(os.path.join(d, "full_index.pkl"))


def build_A1(T, start, end, outdir):
    ensure_dir(outdir)
    d, df = _idx(T)
    FEAT = np.load(os.path.join(d, "features_all.npy"))
    OK = np.load(os.path.join(d, "feat_ok.npy"))
    FK = list(np.load(os.path.join(d, "feat_keys.npy")))
    ik = {k: i for i, k in enumerate(FK)}
    df = df.set_index("index")
    VALID = set(df.index[df.label == "VALID"])

    def gv(row, k):
        return float(FEAT[row][ik[k]])

    def gpt(v):
        return {"from": "gpt", "value": v, "mask_spans": T.feature_mask_spans(v)}

    nqa = 0
    nsh = 0
    for sh in range(start // 1000, (end + 999) // 1000):
        outp = os.path.join(outdir, f"{T.FEATURE_A1_PREFIX}_{sh:05d}.jsonl")
        if os.path.exists(outp):
            continue
        lo, hi = sh * 1000, min(sh * 1000 + 1000, end)
        with open(outp, "w", encoding="utf-8") as fo:
            for idx in range(lo, hi):
                if idx not in VALID or not OK[idx]:
                    continue
                f = {k: gv(idx, k) for k in ["cl_t", "cd_min", "peak_xn", "peak_z", "doming",
                                             "tilt", "h_front", "h_mid", "h_back", "arc_len", "curv_peak"]}
                f["hi_reg"] = ["front", "mid", "back"][int(np.argmax([f["h_front"], f["h_mid"], f["h_back"]]))]
                conv = []
                for q, ans in T.t_a1(f):
                    conv += [{"from": "human", "value": q}, gpt(ans)]
                fo.write(json.dumps({"index": idx,
                                     "mesh_ref": {"verts_shard": sh, "row_in_shard": idx % 1000},
                                     "scenario": "shape_desc", "lang": T.LANG, "conversations": conv,
                                     "turn_types": ["A1", "A1_place", "A1_curv"], "verified": True},
                                    ensure_ascii=False) + "\n")
                nqa += 1
        nsh += 1
    print(f"A1({T.LANG}) shards {start//1000}~{(end+999)//1000-1}: {nsh} files, {nqa} convos")


def build_B3(T, outdir):
    ensure_dir(outdir)
    d, df = _idx(T)
    z = np.load(os.path.join(d, "region_disp.npz"), allow_pickle=True)
    D = np.asarray(z["disp"])
    I = np.asarray(z["idxs"])
    # response = mean regional displacement magnitude vs rest, in mm
    RESP = {int(i): float((D[k, 2] + D[k, 5] + D[k, 8]) / 3.0 * 1000.0) for k, i in enumerate(I)}
    VALID = set(df["index"][df.label == "VALID"])
    recs = []

    def gpt(v):
        return {"from": "gpt", "value": v, "mask_spans": T.feature_mask_spans(v)}

    def emit(q_hum, lead, levels, resps, ref):
        rho = spearman(levels, resps)
        mono = all(resps[i] < resps[i + 1] for i in range(len(resps) - 1))
        net_up = resps[-1] > resps[0]
        seq = T.b3_seq(levels, resps)
        tier = "strict" if mono else ("near" if (rho >= 0.9 and net_up) else "partial")
        a = T.a_b3(tier, lead, seq, rho)
        recs.append({"mesh_ref": ref, "scenario": "dose_response", "lang": T.LANG,
                     "conversations": [{"from": "human", "value": q_hum}, gpt(a)],
                     "turn_types": ["B3"], "verified": True,
                     "sweep": {"levels": levels, "resp_mm": [round(r, 3) for r in resps],
                               "rho": round(rho, 3), "monotonic": mono, "tier": tier}})

    # single: one muscle @ 6 levels
    sg = df[df.section == "single"].copy()
    sg["mus"] = sg["detail"].str.extract(r"([A-Z]+)@")
    sg["lvl"] = sg["detail"].str.extract(r"@([0-9.]+)").astype(float)
    for m, g in sg.groupby("mus"):
        g = g[g["index"].isin(VALID)].sort_values("lvl")
        g = g[g["index"].isin(RESP)]
        if len(g) < 3:
            continue
        lv = [float(x) for x in g["lvl"]]
        rp = [RESP[int(i)] for i in g["index"]]
        emit(T.q_b3_single(T.FULL[m], lv[0], lv[-1]), T.FULL[m], lv, rp,
             {"indices": [int(i) for i in g["index"]]})

    # effort: fixed combo @ swept levels
    ef = df[df.section == "effort"].copy()
    ef["combo"] = ef["detail"].str.extract(r"effort:(.+)@")
    ef["lvl"] = ef["detail"].str.extract(r"@([0-9.]+)").astype(float)
    for combo, g in ef.groupby("combo"):
        g = g[g["index"].isin(VALID)].sort_values("lvl")
        g = g[g["index"].isin(RESP)]
        if len(g) == 0 or g["lvl"].nunique() < 3:
            continue
        g = g.drop_duplicates("lvl")
        lv = [float(x) for x in g["lvl"]]
        rp = [RESP[int(i)] for i in g["index"]]
        muses = [c for c in combo.split("+") if c in T.FULL]
        names = "+".join(T.FULL[c] for c in muses) if muses else combo
        emit(T.q_b3_effort(names, lv[0], lv[-1]), names, lv, rp,
             {"indices": [int(i) for i in g["index"]]})

    with open(os.path.join(outdir, T.B3_FILENAME), "w", encoding="utf-8") as fo:
        for r in recs:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    mono = sum(1 for r in recs if r["sweep"]["monotonic"])
    print(f"B3({T.LANG}): {len(recs)} sweeps ({mono} strictly monotonic, {len(recs)-mono} near/partial)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=sorted(BY_LANG), required=True)
    ap.add_argument("family", choices=["A1", "B3"])
    ap.add_argument("start", type=int, nargs="?", default=0)
    ap.add_argument("end", type=int, nargs="?", default=None)
    ap.add_argument("--outdir", default=None, help="default: {qa_out}/features_{lang}")
    a = ap.parse_args()

    T = BY_LANG[a.lang]
    outdir = a.outdir or os.path.join(cfg_path("qa_out"), "features_" + a.lang)
    if a.family == "B3":
        build_B3(T, outdir)
    else:
        if a.end is None:
            raise SystemExit("A1 needs START END (index range)")
        build_A1(T, a.start, a.end, outdir)


if __name__ == "__main__":
    main()
