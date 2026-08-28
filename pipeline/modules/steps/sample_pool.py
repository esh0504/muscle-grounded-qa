#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2 — sample the 11-D muscle-activation pool (section-mixed design).

Reads the Step-1 specification (modules.anchors + modules.sampling_design)
and writes:
    pool.txt        index,GGP..SL           (ArtiSynth Step-3 input)
    pool_meta.csv   index,section,phoneme,detail,base_index,delta,n_active,refs,GGP..SL

The algorithm and RNG call order are identical to the original
gen_pool_sections.py, so (total, seed) reproduce the released pool exactly.

Usage:
    python -m modules.steps.sample_pool                      # total/seed from settings/sampling.yaml
    python -m modules.steps.sample_pool --total 300000 --seed 0
"""
import argparse
import csv
import os
from collections import Counter

import numpy as np

from ..anchors import (AMAX, BUDGET, CBUDGET, CRULES, VCENTERS,
                       PHONEME_KEYS, center_vec)
from ..config import ensure_dir, path as cfg_path
from ..muscles import D, MI, MUS as MO
from ..sampling_design import (ANCHOR_SELF_FRAC, LEVELS, NACT_W,
                               NEIGHBOR_STEP, PAIR_AMP, POOL_TOTAL,
                               RATIOS, REST_FIRST, SEED, TRIPLE_N)


def clip_budget(v, B=BUDGET):
    v = np.clip(v, 0, AMAX)
    s = v.sum()
    if s > B:
        v = v * (B / s)
    return np.round(v, 4)


def csample(rng, rule):
    """Consonant sample: defining-muscle bands + sparse background, CBUDGET cap."""
    v = np.zeros(D)
    for i in range(D):
        if rng.random() < 0.35:
            v[i] = rng.uniform(0, 0.15)
    v[MI["MH"]] = rng.uniform(0, 0.25) if rng.random() < 0.5 else 0.0
    for m, (lo, hi) in rule.items():
        v[MI[m]] = rng.uniform(lo, hi)
    v = np.clip(v, 0, AMAX)
    s = v.sum()
    if s > CBUDGET:
        keep = set(rule) | {"MH"}
        bg = [i for i in range(D) if MO[i] not in keep]
        excess = s - CBUDGET
        bgsum = sum(v[i] for i in bg)
        if bgsum > 0:
            for i in bg:
                v[i] = max(0, v[i] - excess * v[i] / bgsum)
    return np.round(v, 4)


def cmeets(v, rule):
    return all(lo - 1e-6 <= v[MI[m]] <= hi + 1e-6 for m, (lo, hi) in rule.items())


def build_pool(total, seed):
    rng = np.random.default_rng(seed)
    rows = []  # each: dict(sec,phon,detail,refs,vec,base,delta)

    def add(sec, vec, phon="", detail="", refs="", base_bidx=-1, delta=""):
        rows.append(dict(sec=sec, phon=phon, detail=detail, refs=refs,
                         vec=np.round(np.asarray(vec, float), 4),
                         base=base_bidx, delta=delta))
        return len(rows) - 1

    # --- REST ---
    add("rest", np.zeros(D), detail="rest")
    # --- SINGLE ---
    for m in MO:
        for lv in LEVELS:
            v = np.zeros(D)
            v[MI[m]] = lv
            add("single", v, detail="%s@%.2f" % (m, lv))
    # --- PAIR ---
    for i in range(D):
        for j in range(i + 1, D):
            for a in PAIR_AMP:
                for b in PAIR_AMP:
                    v = np.zeros(D)
                    v[MI[MO[i]]] = a
                    v[MI[MO[j]]] = b
                    add("pair", clip_budget(v), detail="%s+%s" % (MO[i], MO[j]))
    # --- TRIPLE ---
    for _ in range(TRIPLE_N):
        idx = rng.choice(D, 3, replace=False)
        v = np.zeros(D)
        for t in idx:
            v[t] = rng.uniform(0.15, 0.9)
        add("triple", clip_budget(v), detail="+".join(MO[t] for t in idx))

    enum_n = len(rows)
    fill = total - enum_n
    n_anchor = int(fill * RATIOS["anchor"])
    n_neigh = int(fill * RATIOS["neighbor"])
    n_eff = int(fill * RATIOS["effort"])
    n_space = fill - n_anchor - n_neigh - n_eff

    # --- ANCHOR (phoneme-balanced: ANCHOR_SELF_FRAC own-center + rest interpolation) ---
    per = max(1, int(n_anchor * ANCHOR_SELF_FRAC / len(PHONEME_KEYS)))
    anchor_idx = []
    for key in VCENTERS:
        (ipa, sig, refs), base = VCENTERS[key]
        base = np.array(base)
        for _ in range(per):
            v = base + rng.normal(0, sig, D) * (base > 0)
            v = np.clip(v, 0, AMAX)
            if rng.random() < 0.35:
                v[MI["MH"]] = max(v[MI["MH"]], rng.uniform(0.1, 0.3))
            anchor_idx.append(add("anchor", clip_budget(v), phon=ipa, detail="anchor:" + key, refs=refs))
    for key in CRULES:
        (ipa, refs), rule = CRULES[key]
        c = 0
        while c < per:
            v = csample(rng, rule)
            if cmeets(v, rule):
                anchor_idx.append(add("anchor", v, phon=ipa, detail="anchor:" + key, refs=refs))
                c += 1
    # interpolation (transitions) — remaining anchor budget
    while len(anchor_idx) < n_anchor:
        k1, k2 = rng.choice(PHONEME_KEYS, 2, replace=False)
        t = rng.choice([0.25, 0.5, 0.75])
        v = (1 - t) * center_vec(k1) + t * center_vec(k2)
        v = clip_budget(v + rng.normal(0, 0.05, D) * (v > 0))
        p1 = VCENTERS[k1][0][0] if k1 in VCENTERS else CRULES[k1][0][0]
        p2 = VCENTERS[k2][0][0] if k2 in VCENTERS else CRULES[k2][0][0]
        anchor_idx.append(add("anchor", v, phon="%s~%s" % (p1, p2),
                              detail="interp:%s~%s@%.2f" % (k1, k2, t)))

    # --- NEIGHBOR (base = random anchor, one muscle ±NEIGHBOR_STEP) ---
    anchor_arr = np.array(anchor_idx)
    for _ in range(n_neigh):
        b = int(anchor_arr[rng.integers(len(anchor_arr))])
        bv = rows[b]["vec"].copy()
        m = int(rng.integers(D))
        sign = rng.choice([-1, 1])
        nv = bv.copy()
        nv[m] = min(AMAX, max(0.0, bv[m] + sign * NEIGHBOR_STEP))
        if abs(nv[m] - bv[m]) < 1e-6:
            nv[m] = min(AMAX, bv[m] + NEIGHBOR_STEP)
        add("neighbor", nv, phon=rows[b]["phon"], detail="d%s%+d" % (MO[m], sign),
            base_bidx=b, delta="%s%+.2f" % (MO[m], sign * NEIGHBOR_STEP))

    # --- EFFORT (fixed direction, amplitude sweep) ---
    for _ in range(n_eff):
        k = rng.integers(1, 4)
        idx = rng.choice(D, k, replace=False)
        direction = np.zeros(D)
        for t in idx:
            direction[t] = rng.uniform(0.5, 1.0)
        s = rng.choice(np.linspace(0.15, 1.0, 6))
        v = clip_budget(direction * s)
        add("effort", v, detail="effort:" + "+".join(MO[t] for t in idx) + "@%.2f" % s)

    # --- SPACEFILL (sparse, n_active ~ NACT_W) ---
    navs = list(NACT_W)
    wts = np.array([NACT_W[k] for k in navs])
    wts = wts / wts.sum()
    for _ in range(n_space):
        na = int(rng.choice(navs, p=wts))
        idx = rng.choice(D, na, replace=False)
        v = np.zeros(D)
        for t in idx:
            v[t] = rng.uniform(0.15, 0.9)
        add("spacefill", clip_budget(v), detail="spacefill")

    # --- dedup (4dp) keeping first; then shuffle with base remap ---
    seen = {}
    keep = []
    for i, r in enumerate(rows):
        key = tuple(r["vec"])
        if key in seen:
            continue
        seen[key] = len(keep)
        keep.append(i)
    rows2 = [rows[i] for i in keep]
    oldpos = {old: new for new, old in enumerate(keep)}      # build idx -> compacted idx
    perm = list(np.random.default_rng(42).permutation(len(rows2)))
    if REST_FIRST:
        # pin the all-zero REST sample at pool index 0 (pilot ranges then always
        # contain the displacement reference); relative order otherwise unchanged
        rest_cp = next(i for i, r in enumerate(rows2) if r["sec"] == "rest")
        perm.remove(rest_cp)
        perm.insert(0, rest_cp)
    final = [rows2[i] for i in perm]
    newpos = {}
    for new, i in enumerate(perm):
        newpos[i] = new                                       # compacted idx -> final idx

    def remap_base(r):
        if r["base"] < 0:
            return -1
        cp = oldpos.get(r["base"], -1)
        return newpos.get(cp, -1) if cp >= 0 else -1

    return rows, final, enum_n, fill, remap_base


def write_pool(final, remap_base, pool_txt, pool_meta):
    with open(pool_txt, "w", newline="") as f:
        f.write("# section-mixed muscle pool. REST/SINGLE/PAIR/TRIPLE + ANCHOR/NEIGHBOR/EFFORT/SPACEFILL\n")
        f.write("# meta(section,phoneme,base_index,delta...) -> pool_meta.csv by index\n")
        f.write("index," + ",".join(MO) + "\n")
        for i, r in enumerate(final):
            f.write("%d," % i + ",".join("%.4f" % x for x in r["vec"]) + "\n")
    with open(pool_meta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "section", "phoneme", "detail", "base_index", "delta", "n_active", "refs"] + MO)
        for i, r in enumerate(final):
            na = int((r["vec"] >= 0.05).sum())
            w.writerow([i, r["sec"], r["phon"], r["detail"], remap_base(r), r["delta"], na, r["refs"]]
                       + ["%.4f" % x for x in r["vec"]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total", type=int, default=POOL_TOTAL, help="default: settings/sampling.yaml")
    ap.add_argument("--seed", type=int, default=SEED, help="default: settings/sampling.yaml")
    ap.add_argument("--pool-txt", default=None, help="default: config pool_txt")
    ap.add_argument("--pool-meta", default=None, help="default: config pool_meta")
    a = ap.parse_args()

    pool_txt = a.pool_txt or cfg_path("pool_txt")
    pool_meta = a.pool_meta or cfg_path("pool_meta")
    ensure_dir(os.path.dirname(pool_txt))
    ensure_dir(os.path.dirname(pool_meta))

    rows, final, enum_n, fill, remap_base = build_pool(a.total, a.seed)
    write_pool(final, remap_base, pool_txt, pool_meta)
    print("enum=%d fill=%d total=%d (after dedup %d)" % (enum_n, fill, len(rows), len(final)))
    print("sections:", dict(Counter(r["sec"] for r in final)))
    print("wrote:", pool_txt)
    print("wrote:", pool_meta)


if __name__ == "__main__":
    main()
