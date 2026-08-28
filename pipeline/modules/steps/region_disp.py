#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5b — regional displacement field vs rest, for every mesh (vectorized).

The ONLY step-5 stage that needs the verts binaries for the physics QA. The rest
mesh (all-zero activation, CFG rest_index) defines front/mid/back thirds by x;
each mesh gets 9 numbers: (dx, dz, |d|) per region, in metres.

Output: {index_dir}/region_disp.npz  (idxs, disp(n,9), cols)
Usage:  python -m modules.steps.region_disp
"""
import glob
import os
import time

import numpy as np

from ..config import CFG, ensure_dir, path as cfg_path
from ..dorsum import load_verts_bin
from ..mesh_io import rest_verts, sim_out


def main():
    out_dir = ensure_dir(cfg_path("index_dir"))
    rest = rest_verts()                                   # (370,3)
    x = rest[:, 0]
    q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
    masks = {"front": x < q1, "mid": (x >= q1) & (x < q2), "back": x >= q2}

    files = sorted(glob.glob(os.path.join(sim_out(), "verts", "shard_*.bin")))
    n_surf = int(CFG.get("n_surf_verts", 370))
    t = time.time()
    idxs = []
    rows = []
    for f in files:
        sh = int(os.path.basename(f)[6:11])
        V = load_verts_bin(f, n_surf=n_surf).astype(np.float32)   # (n,370,3)
        disp = V - rest                                            # broadcast
        n = len(V)
        out = np.zeros((n, 9), np.float32)                         # front/mid/back x (dx,dz,mag)
        for k, (reg, m) in enumerate(masks.items()):
            d = disp[:, m, :]                                      # (n,|m|,3)
            out[:, k * 3 + 0] = d[:, :, 0].mean(1)
            out[:, k * 3 + 1] = d[:, :, 2].mean(1)
            out[:, k * 3 + 2] = np.linalg.norm(d, axis=2).mean(1)
        idxs.append(sh * 1000 + np.arange(n))
        rows.append(out)
    idxs = np.concatenate(idxs)
    rows = np.vstack(rows)
    np.savez_compressed(
        os.path.join(out_dir, "region_disp.npz"), idxs=idxs, disp=rows,
        cols=np.array(["front_dx", "front_dz", "front_mag", "mid_dx", "mid_dz", "mid_mag",
                       "back_dx", "back_dz", "back_mag"]))
    print(f"region_disp for {len(idxs)} meshes in {time.time()-t:.0f}s -> {out_dir}/region_disp.npz")


if __name__ == "__main__":
    main()
