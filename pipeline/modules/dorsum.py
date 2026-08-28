# -*- coding: utf-8 -*-
"""Fixed mid-sagittal dorsum vertex indices + contour reader.

All meshes share one topology (370 surface vertices), so the dorsum contour is
read by a FIXED, human-picked vertex index list (3D viewer, tip -> root) applied
identically to every mesh — no automatic contour extraction, no artifacts.
"""
import numpy as np

# Picked once in a 3D viewer (dorsum_picker.html): tongue tip -> dorsum -> root
DORSUM_VERTS = [184, 193, 191, 182, 156, 146, 144, 76, 49, 46]


def dorsum_contour(verts, idx=None, order="anat"):
    """verts:(370,3) -> mid-sagittal (x,z) contour.
    order='anat' keeps the picked order (tip->root, polyline);
    order='x' sorts by x (height-profile use)."""
    idx = DORSUM_VERTS if idx is None else idx
    d = np.asarray(verts)[idx][:, [0, 2]]
    if order == "x":
        d = d[np.argsort(d[:, 0])]
    return d


def load_verts_bin(path, n_surf=370):
    """Read one shard binary (big-endian float32) -> (n, n_surf, 3) float64."""
    return np.fromfile(path, dtype=">f4").reshape(-1, n_surf, 3).astype(float)
