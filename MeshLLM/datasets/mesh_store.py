"""메쉬 IO + mesh grounding 통제 — Stage-1/2/평가가 공유한다.

`MeshStore` 는 topology.obj + verts shard 에서 변위를 꺼내고,
`apply_mesh_control` 은 평가 때 mesh 를 real/shuffle/rest/noise 로 바꿔치기한다.
(원본: eval.py)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from datasets.mesh_dataset import DTYPE, N_SURF, SHARD_SIZE

MESH_ROOT = Path("DATA/mesh")


# --------------------------------------------------------------------------- #
# mesh IO
# --------------------------------------------------------------------------- #
class MeshStore:
    """topology + verts shard 에서 변위를 꺼낸다 (Stage-1/2 와 같은 규약)."""

    def __init__(self, root: Path = MESH_ROOT):
        self.root = Path(root)
        verts, faces = [], []
        for line in (self.root / "topology.obj").read_text().splitlines():
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
        self.rest = np.asarray(verts, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int64)
        self._mm: dict[int, np.ndarray] = {}

    def _shard(self, s: int) -> np.ndarray:
        a = self._mm.get(s)
        if a is None:
            p = self.root / "verts" / f"shard_{s:05d}.bin"
            a = np.asarray(np.memmap(p, dtype=DTYPE, mode="r")).reshape(-1, N_SURF, 3)
            self._mm[s] = a
        return a

    def disp(self, index: int) -> np.ndarray:
        sh, lo = divmod(int(index), SHARD_SIZE)
        return np.asarray(self._shard(sh)[lo], dtype=np.float32) - self.rest

    @staticmethod
    def index_of(rec: dict) -> list[int]:
        ref = rec.get("mesh_ref") or {}
        if "indices" in ref:
            return [int(i) for i in ref["indices"]]
        if "verts_shard" in ref:
            return [int(ref["verts_shard"]) * SHARD_SIZE + int(ref["row_in_shard"])]
        return [int(rec["index"])]


# --------------------------------------------------------------------------- #
# mesh grounding 통제
# --------------------------------------------------------------------------- #
# 모델이 정말 mesh 를 쓰는지 확인하는 eval-time 통제. 학습이 필요 없다.
#   real    — 그대로 (기준)
#   shuffle — 같은 turn_type 안에서 derangement. 질문·문맥은 그대로 두고 mesh 만 틀린 것으로
#             바꾼다. 답이 안 바뀌면 모델은 질문만 보고 있는 것이다.
#   rest    — 변위 0 (= rest 형상). mesh 입력 자체가 의미 있는지.
#   noise   — 항목별 RMS 를 맞춘 가우시안. 통계량만 쓰는지.
MESH_CONTROLS = {"real", "shuffle", "rest", "noise"}


def apply_mesh_control(jobs, mode: str, seed: int = 42):
    """각 job 에 'mesh_indices_eff'(실제로 읽을 mesh)와 'mesh_perturb'를 채운다."""
    for j in jobs:
        j["mesh_indices_eff"] = list(j["mesh_indices"])
        j["mesh_perturb"] = None
    if mode == "real":
        return jobs
    if mode in ("rest", "noise"):
        for j in jobs:
            j["mesh_perturb"] = mode
        return jobs

    # shuffle: turn_type 별로 묶어 자기 자신이 아닌 다른 항목의 mesh 를 받는다.
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for i, j in enumerate(jobs):
        groups.setdefault(str(j.get("turn_type") or ""), []).append(i)
    for tt, idxs in groups.items():
        if len(idxs) < 2:
            # 짝이 없으면 전체에서 아무거나 빌려온다 (통제가 무효화되지 않도록).
            for i in idxs:
                pool = [k for k in range(len(jobs)) if k != i]
                if pool:
                    jobs[i]["mesh_indices_eff"] = list(jobs[rng.choice(pool)]["mesh_indices"])
            continue
        src = idxs[:]
        for _ in range(100):                       # derangement 를 뽑을 때까지
            rng.shuffle(src)
            if all(a != b for a, b in zip(idxs, src)):
                break
        else:                                      # 최후 수단: 한 칸 회전
            src = idxs[1:] + idxs[:1]
        for dst, s in zip(idxs, src):
            jobs[dst]["mesh_indices_eff"] = list(jobs[s]["mesh_indices"])
    return jobs
