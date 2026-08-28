"""Build / load SpiralNet++ mesh transforms for a fixed template."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

from models.stage1.spiralnet.mesh_sampling import (
    Mesh,
    generate_downsample_matrices,
    scipy_to_torch_sparse,
    setup_deformation_transfer,
)
from models.stage1.spiralnet.official import preprocess_spiral  # 공식 구현 (openmesh)


def load_obj_mesh(path: Path):
    verts, faces = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:]])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def build_spiral_cache(
    template_obj: Path | str,
    *,
    ds_factors=(4, 4),
    seq_lengths=(9, 9),
    dilations=(1, 1),
    out_path: Path | str | None = None,
) -> dict:
    template_obj = Path(template_obj)
    verts, faces = load_obj_mesh(template_obj)
    faces_list, verts_list, down_list = generate_downsample_matrices(
        verts, faces, list(ds_factors)
    )

    n_levels = len(ds_factors)
    if not (len(seq_lengths) == len(dilations) == n_levels):
        raise ValueError("seq_lengths / dilations must match ds_factors length")

    spiral_indices = []
    for i in range(n_levels):
        spiral_indices.append(
            preprocess_spiral(
                faces_list[i],
                seq_length=int(seq_lengths[i]),
                vertices=verts_list[i],
                dilation=int(dilations[i]),
            )
        )

    cache = {
        "template": str(template_obj.resolve()),
        "ds_factors": list(ds_factors),
        "seq_lengths": list(seq_lengths),
        "dilations": list(dilations),
        "faces": faces_list,
        "vertices": verts_list,
        "down_transform": down_list,  # scipy csc
        "spiral_indices": [s.numpy() for s in spiral_indices],
        "n_verts_levels": [int(v.shape[0]) for v in verts_list],
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return cache


def _read_cache(cache_or_path) -> dict:
    if isinstance(cache_or_path, (str, Path)):
        with open(cache_or_path, "rb") as fh:
            return pickle.load(fh)
    return cache_or_path


def add_up_transforms(cache_or_path, out_path: Path | str | None = None) -> dict:
    """기존 캐시에 업샘플링 행렬 U 만 덧붙인다 (디코더용).

    **qslim 을 다시 돌리지 않는다.** 캐시에 이미 있는 계층 메쉬(faces/vertices)에서 U 만
    계산해 넣는다. 새로 데시메이션하면 D 와 나선 인덱스가 달라질 수 있고, 그러면
    "인코더는 같고 목적함수만 다르다"는 AE ablation 의 전제가 깨진다.

      python -c "from models.stage1.spiralnet.preprocess import add_up_transforms; \\
                 add_up_transforms('DATA/mesh/spiral_transform.pkl', \\
                                   'DATA/mesh/spiral_transform_ae.pkl')"
    """
    cache = dict(_read_cache(cache_or_path))
    faces_list, verts_list = cache["faces"], cache["vertices"]

    up_list = []
    for i in range(len(cache["down_transform"])):
        coarse = Mesh(v=verts_list[i + 1], f=faces_list[i + 1])
        fine = Mesh(v=verts_list[i], f=faces_list[i])
        up_list.append(setup_deformation_transfer(coarse, fine).astype("float32"))
    cache["up_transform"] = up_list

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as fh:
            pickle.dump(cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[spiral] up_transform 추가 → {out_path} "
              f"({[u.shape for u in up_list]})")
    return cache


def load_up_tensors(cache_or_path, device=None):
    """캐시의 업샘플링 행렬을 torch sparse 로. `up[i]`: level i+1 → level i."""
    cache = _read_cache(cache_or_path)
    if "up_transform" not in cache:
        raise KeyError(
            "캐시에 up_transform 이 없다 — 디코더를 쓰려면 add_up_transforms() 로 만들어라 "
            "(인코더 전용 캐시에는 U 가 없다)."
        )
    ups = [scipy_to_torch_sparse(u) for u in cache["up_transform"]]
    if device is not None:
        ups = [u.to(device) for u in ups]
    return ups


def load_spiral_tensors(cache_or_path, device=None):
    """Return (spiral_indices, down_transforms) as torch tensors."""
    cache = _read_cache(cache_or_path)

    spirals = [torch.as_tensor(s, dtype=torch.long) for s in cache["spiral_indices"]]
    downs = [scipy_to_torch_sparse(d) for d in cache["down_transform"]]
    if device is not None:
        spirals = [s.to(device) for s in spirals]
        downs = [d.to(device) for d in downs]
    return spirals, downs, cache
