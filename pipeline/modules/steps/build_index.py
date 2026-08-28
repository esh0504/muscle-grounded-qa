#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5a — join Step-2 pool metadata with Step-3 validity metadata.

Output (in CFG index_dir):
    full_index.pkl   one row per pool index: section/phoneme/detail/base_index/
                     delta/n_active/refs + 11 activations + label/vol_ratio/
                     n_inverted/max_vel + verts_shard/row_in_shard
    meta_all.csv     concatenated per-shard simulation meta (validity screening)

Usage: python -m modules.steps.build_index
"""
import os

from ..config import ensure_dir, path as cfg_path
from ..mesh_io import load_meta_all, load_pool_meta
from ..muscles import MUS


def main():
    out = ensure_dir(cfg_path("index_dir"))
    info = load_pool_meta().reset_index()
    metas = load_meta_all().reset_index()
    print("pool rows", len(info), "| sim meta rows", len(metas))

    metas.to_csv(os.path.join(out, "meta_all.csv"), index=False)
    # Also drop a copy next to the meshes, so downstream consumers (e.g. the
    # MeshLLM baseline) can work from sim_out alone.
    metas.to_csv(os.path.join(cfg_path("sim_out"), "meta_all.csv"), index=False)

    df = info.merge(metas[["index", "label", "vol_ratio", "n_inverted", "max_vel"]],
                    on="index", how="left")
    df["verts_shard"] = df["index"] // 1000
    df["row_in_shard"] = df["index"] % 1000
    print("label counts:\n", df["label"].value_counts(dropna=False))
    print("section counts:\n", df["section"].value_counts(dropna=False))

    df.to_pickle(os.path.join(out, "full_index.pkl"))
    print("saved", os.path.join(out, "full_index.pkl"), df.shape)

    # quick structural peek (neighbor & effort feed B2 / B3)
    nb = df[df.section == "neighbor"].head(8)[["index", "phoneme", "detail", "base_index", "delta", "n_active"] + MUS]
    print("\n-- neighbor sample --\n", nb.to_string())
    ef = df[df.section == "effort"].head(6)[["index", "detail", "base_index", "delta", "n_active"] + MUS]
    print("\n-- effort sample --\n", ef.to_string())


if __name__ == "__main__":
    main()
