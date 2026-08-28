# -*- coding: utf-8 -*-
"""Simulation-output readers: shard binaries + per-shard meta CSVs.

Layout produced by step3 (export_static_impl.py), all under CFG["sim_out"]:
    topology.obj / topology_info.txt
    verts/shard_%05d.bin   float32 BIG-ENDIAN (n, n_surf_verts, 3)
    nodes/shard_%05d.bin   float32 BIG-ENDIAN (n, n_fem_nodes, 3)
    meta/shard_%05d.csv    index,label,reason,max_vel,peak_ramp_vel,vol_ratio,
                           n_inverted,min_elem_vol,settle_t,secs
"""
import glob
import os

import numpy as np
import pandas as pd

from .config import CFG, path as cfg_path
from .dorsum import load_verts_bin  # noqa: F401  (re-export)

SHARD = int(CFG.get("shard_size", 1000))
N_SURF = int(CFG.get("n_surf_verts", 370))


def sim_out():
    return cfg_path("sim_out")


def shard_of(index):
    return index // SHARD, index % SHARD


def verts_shard_path(shard, out=None):
    return os.path.join(out or sim_out(), "verts", "shard_%05d.bin" % shard)


def load_shard(shard, out=None):
    """(n, N_SURF, 3) float array for one shard."""
    return load_verts_bin(verts_shard_path(shard, out), n_surf=N_SURF)


def load_mesh(index, out=None):
    sh, row = shard_of(index)
    return load_shard(sh, out)[row]


def load_meta_all(out=None):
    """Concatenate every meta/shard_*.csv -> DataFrame indexed by pool index."""
    files = sorted(glob.glob(os.path.join(out or sim_out(), "meta", "shard_*.csv")))
    if not files:
        raise FileNotFoundError("no meta/shard_*.csv under %s — run step 3 first" % (out or sim_out()))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True).set_index("index")


def load_pool_meta():
    """Step-2 pool metadata (section/phoneme/detail/base_index/delta + activations)."""
    return pd.read_csv(cfg_path("pool_meta")).set_index("index")


def valid_set(metas=None):
    m = load_meta_all() if metas is None else metas
    return set(m.index[m.label == "VALID"])


def rest_index():
    """Pool index of the all-zero REST sample.

    CFG["rest_index"] may be an int or "auto" (default): auto reads pool_meta
    and finds the section == "rest" row — works for any rest_first setting.
    """
    ri = CFG.get("rest_index", "auto")
    if isinstance(ri, int):
        return ri
    meta = load_pool_meta()
    rest = meta.index[meta.section == "rest"]
    if len(rest) == 0:
        raise ValueError("no section=='rest' row in %s" % cfg_path("pool_meta"))
    return int(rest[0])


def rest_verts(out=None):
    """Rest-pose surface vertices (the pool's all-zero activation sample)."""
    return load_mesh(rest_index(), out)
