#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5c — 32 palate-normalized features for every mesh (feature QA grounding).

Reads every verts shard, extracts the mid-sagittal dorsum contour with the fixed
vertex list, and computes modules.features (cl_t, cd_min, doming, ...).
Non-finite meshes (FAILED_NUMERICAL) get feat_ok=False.

Output (in CFG index_dir): features_all.npy (N,32) float32,
                           feat_ok.npy (N,) bool, feat_keys.npy
Usage:  python -m modules.steps.extract_features [--workers 2]
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .. import features as F
from ..config import CFG, PIPELINE_ROOT, ensure_dir, path as cfg_path
from ..dorsum import dorsum_contour, load_verts_bin
from ..mesh_io import sim_out

ASSETS = os.path.join(PIPELINE_ROOT, "assets")
FKEYS = list(F.FEATURE_KEYS)
NF = len(FKEYS)
_PALATE = None


def _palate():
    global _PALATE
    if _PALATE is None:
        _PALATE = pd.read_csv(os.path.join(ASSETS, "model_palate.csv"))[["x", "z"]].values
    return _PALATE


def do_shard(path):
    sh = int(os.path.basename(path)[6:11])
    v = load_verts_bin(path, n_surf=int(CFG.get("n_surf_verts", 370)))  # (n,370,3)
    n = len(v)
    fin = np.isfinite(v.reshape(n, -1)).all(1)
    out = np.full((n, NF), np.nan, np.float32)
    ok = np.zeros(n, bool)
    palate = _palate()
    for i in range(n):
        if not fin[i]:
            continue
        try:
            cont = dorsum_contour(v[i], order="x")
            f = F.extract_features(cont, palate)
            row = [f[k] for k in FKEYS]
            if all(np.isfinite(x) for x in row):
                out[i] = row
                ok[i] = True
        except Exception:
            pass
    return (sh, sh * 1000, out, ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    out_dir = ensure_dir(cfg_path("index_dir"))
    files = sorted(glob.glob(os.path.join(sim_out(), "verts", "shard_*.bin")))
    if not files:
        raise SystemExit("no verts shards under %s — run step 3 first" % sim_out())
    # N = last shard base + its row count (shards are 1000 rows except possibly the last)
    last = load_verts_bin(files[-1], n_surf=int(CFG.get("n_surf_verts", 370)))
    N = int(os.path.basename(files[-1])[6:11]) * 1000 + len(last)
    del last

    FEAT = np.full((N, NF), np.nan, np.float32)
    OK = np.zeros(N, bool)
    t = time.time()
    done = 0
    with Pool(a.workers) as p:
        for sh, base, out, ok in p.imap_unordered(do_shard, files):
            n = len(ok)
            FEAT[base:base + n] = out
            OK[base:base + n] = ok
            done += 1
            if done % 40 == 0:
                print(f"  {done}/{len(files)} shards, {time.time()-t:.0f}s", flush=True)
    np.save(os.path.join(out_dir, "features_all.npy"), FEAT)
    np.save(os.path.join(out_dir, "feat_ok.npy"), OK)
    np.save(os.path.join(out_dir, "feat_keys.npy"), np.array(FKEYS))
    print(f"DONE {done} shards in {time.time()-t:.0f}s | feat_ok {OK.sum()}/{N} ({OK.mean()*100:.2f}%)")


if __name__ == "__main__":
    main()
