"""Qslim-style mesh downsampling (SciPy only, no psbody.mesh).

Adapted from SpiralNet++ / CoMA mesh_sampling.py.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class Mesh:
    v: np.ndarray
    f: np.ndarray

    def __post_init__(self):
        self.v = np.asarray(self.v, dtype=np.float64)
        self.f = np.asarray(self.f, dtype=np.int64)


def row(A):
    return A.reshape((1, -1))


def col(A):
    return A.reshape((-1, 1))


def get_vert_connectivity(mesh_v, mesh_f):
    vpv = sp.csc_matrix((len(mesh_v), len(mesh_v)))
    for i in range(3):
        IS = mesh_f[:, i]
        JS = mesh_f[:, (i + 1) % 3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.ravel()), row(JS.ravel())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T
    return vpv


def get_vertices_per_edge(mesh_v, mesh_f):
    vc = sp.coo_matrix(get_vert_connectivity(mesh_v, mesh_f))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:, 0] < result[:, 1]]
    return result


def vertex_quadrics(mesh: Mesh):
    v_quadrics = np.zeros((len(mesh.v), 4, 4))
    for f_idx in range(len(mesh.f)):
        vert_idxs = mesh.f[f_idx]
        verts = np.hstack((mesh.v[vert_idxs], np.ones((3, 1))))
        _u, _s, vh = np.linalg.svd(verts)
        eq = vh[-1, :].reshape(-1, 1)
        eq = eq / (np.linalg.norm(eq[0:3]))
        for k in range(3):
            v_quadrics[mesh.f[f_idx, k], :, :] += np.outer(eq, eq)
    return v_quadrics


def _get_sparse_transform(faces, num_original_verts):
    verts_left = np.unique(faces.flatten())
    IS = np.arange(len(verts_left))
    JS = verts_left
    data = np.ones(len(JS))
    mp = np.arange(0, np.max(faces.flatten()) + 1)
    mp[JS] = IS
    new_faces = mp[faces.copy().flatten()].reshape((-1, 3))
    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij), shape=(len(verts_left), num_original_verts))
    return new_faces, mtx


def qslim_decimator_transformer(mesh: Mesh, factor=None, n_verts_desired=None):
    if factor is None and n_verts_desired is None:
        raise ValueError("Need either factor or n_verts_desired.")
    if n_verts_desired is None:
        n_verts_desired = math.ceil(len(mesh.v) * factor)

    Qv = vertex_quadrics(mesh)
    vert_adj = get_vertices_per_edge(mesh.v, mesh.f)
    vert_adj = sp.csc_matrix(
        (np.ones(len(vert_adj)), (vert_adj[:, 0], vert_adj[:, 1])),
        shape=(len(mesh.v), len(mesh.v)),
    )
    vert_adj = (vert_adj + vert_adj.T).tocoo()

    def collapse_cost(Qv, r, c, v):
        Qsum = Qv[r, :, :] + Qv[c, :, :]
        p1 = np.vstack((v[r].reshape(-1, 1), [[1.0]]))
        p2 = np.vstack((v[c].reshape(-1, 1), [[1.0]]))
        destroy_c_cost = float(p1.T.dot(Qsum).dot(p1))
        destroy_r_cost = float(p2.T.dot(Qsum).dot(p2))
        return {
            "destroy_c_cost": destroy_c_cost,
            "destroy_r_cost": destroy_r_cost,
            "collapse_cost": min(destroy_c_cost, destroy_r_cost),
            "Qsum": Qsum,
        }

    queue = []
    for k in range(vert_adj.nnz):
        r, c = int(vert_adj.row[k]), int(vert_adj.col[k])
        if r > c:
            continue
        cost = collapse_cost(Qv, r, c, mesh.v)["collapse_cost"]
        heapq.heappush(queue, (cost, (r, c)))

    faces = mesh.f.copy()
    nverts_total = len(mesh.v)
    # `and queue` 는 공식엔 없다. 빈 큐에서 IndexError 대신 정상 종료시키는 가드일 뿐이라
    # 감쇠 결과는 바뀌지 않는다.
    while nverts_total > n_verts_desired and queue:
        e = heapq.heappop(queue)
        r, c = e[1]
        if r == c:
            continue
        cost = collapse_cost(Qv, r, c, mesh.v)
        if cost["collapse_cost"] > e[0]:
            heapq.heappush(queue, (cost["collapse_cost"], e[1]))
            continue

        if cost["destroy_c_cost"] < cost["destroy_r_cost"]:
            to_destroy, to_keep = c, r
        else:
            to_destroy, to_keep = r, c

        np.place(faces, faces == to_destroy, to_keep)
        which1 = [idx for idx in range(len(queue)) if queue[idx][1][0] == to_destroy]
        which2 = [idx for idx in range(len(queue)) if queue[idx][1][1] == to_destroy]
        for k in which1:
            queue[k] = (queue[k][0], (to_keep, queue[k][1][1]))
        for k in which2:
            queue[k] = (queue[k][0], (queue[k][1][0], to_keep))

        Qv[r, :, :] = cost["Qsum"]
        Qv[c, :, :] = cost["Qsum"]

        a = faces[:, 0] == faces[:, 1]
        b = faces[:, 1] == faces[:, 2]
        c_ = faces[:, 2] == faces[:, 0]
        faces = faces[np.logical_not(a | b | c_)].copy()
        nverts_total = len(np.unique(faces.flatten()))

    new_faces, mtx = _get_sparse_transform(faces, len(mesh.v))
    return new_faces, mtx


def _closest_point_barycentric(points, tri_v):
    """각 점에 대해 모든 삼각형 위 최근접점의 바리센트릭 좌표.

    psbody 의 AABB tree (`compute_aabb_tree().nearest`) 를 대신한다. 원본은 C++
    확장이라 이 컨테이너에서 못 쓰는데(boost 필요), 우리 계층 메쉬는 정점 370 → 93 → 24 로
    아주 작아서 전수 계산으로 충분하다 (370 × ~180 삼각형).

    알고리즘은 Ericson, *Real-Time Collision Detection* §5.1.5 의 점-삼각형 최근접점.

    Args:
      points: (P, 3)
      tri_v:  (F, 3, 3) — 삼각형별 정점 좌표

    Returns:
      bary: (P, F, 3) 바리센트릭 좌표 (합 1, 모두 >= 0)
      dist: (P, F)    최근접점까지 거리
    """
    p = np.asarray(points, dtype=np.float64)[:, None, :]        # (P,1,3)
    a = tri_v[None, :, 0, :]                                    # (1,F,3)
    b = tri_v[None, :, 1, :]
    c = tri_v[None, :, 2, :]

    ab, ac = b - a, c - a
    d1 = np.sum(ab * (p - a), axis=-1)
    d2 = np.sum(ac * (p - a), axis=-1)
    d3 = np.sum(ab * (p - b), axis=-1)
    d4 = np.sum(ac * (p - b), axis=-1)
    d5 = np.sum(ab * (p - c), axis=-1)
    d6 = np.sum(ac * (p - c), axis=-1)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    # 기본값: 내부 (denom 이 0 인 퇴화 삼각형은 뒤의 정점/모서리 분기가 덮는다)
    denom = va + vb + vc
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(np.abs(denom) > 0, 1.0 / denom, 0.0)
    v = vb * inv
    w = vc * inv
    u = 1.0 - v - w
    bary = np.stack([u, v, w], axis=-1)

    def _set(mask, uu, vv, ww):
        m = mask[..., None]
        bary[:] = np.where(m, np.stack(np.broadcast_arrays(uu, vv, ww), axis=-1), bary)

    zero = np.zeros_like(d1)
    one = np.ones_like(d1)

    # 모서리 영역
    m_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d1 - d3 != 0, d1 / (d1 - d3), 0.0)
    _set(m_ab, 1.0 - t, t, zero)

    m_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d2 - d6 != 0, d2 / (d2 - d6), 0.0)
    _set(m_ac, 1.0 - t, zero, t)

    m_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    den_bc = (d4 - d3) + (d5 - d6)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(den_bc != 0, (d4 - d3) / den_bc, 0.0)
    _set(m_bc, zero, 1.0 - t, t)

    # 정점 영역 — 모서리보다 우선한다 (Ericson 의 early return 순서)
    _set((d1 <= 0) & (d2 <= 0), one, zero, zero)
    _set((d3 >= 0) & (d4 <= d3), zero, one, zero)
    _set((d6 >= 0) & (d5 <= d6), zero, zero, one)

    q = bary[..., 0:1] * a + bary[..., 1:2] * b + bary[..., 2:3] * c
    dist = np.linalg.norm(p - q, axis=-1)
    return bary, dist


def setup_deformation_transfer(source: Mesh, target: Mesh, tol: float = 1e-9):
    """업샘플링 행렬 U — coarse(source) 특징을 fine(target) 정점으로 보간.

    공식 CoMA / SpiralNet++ `utils/mesh_sampling.py::setup_deformation_transfer` 의
    SciPy 이식본이다. 원본은 최근접 삼각형·부위(part)·최근접점을 psbody 의 AABB tree 로
    얻는데, 그 세 값만 `_closest_point_barycentric` 로 대체했고 **계수 계산 규칙은 원본
    그대로**다.

      part 0      삼각형 내부 → lstsq(A_3x3, 최근접점)  = 바리센트릭 좌표
      part 1..3   모서리 위   → lstsq(A_3x2, 대상 정점) 을 두 끝점에 배분
      part 4..6   정점 위     → 그 정점에 1.0

    (모서리 분기가 최근접점이 아니라 **대상 정점**을 쓰는 것도 원본과 같다.)

    Returns:
      csc_matrix (n_target_verts, n_source_verts)
    """
    src_v, src_f = np.asarray(source.v), np.asarray(source.f)
    tgt_v = np.asarray(target.v)
    n_t = tgt_v.shape[0]

    bary, dist = _closest_point_barycentric(tgt_v, src_v[src_f])
    nearest_faces = np.argmin(dist, axis=1)                     # (P,)
    rows_idx = np.arange(n_t)
    bary_hit = bary[rows_idx, nearest_faces]                    # (P, 3)

    rows = np.zeros(3 * n_t, dtype=np.int64)
    cols = np.zeros(3 * n_t, dtype=np.int64)
    coeffs = np.zeros(3 * n_t, dtype=np.float64)

    # 원본의 part id 규약: 0=내부, 1..3=모서리(v0v1, v1v2, v2v0), 4..6=정점(v0, v1, v2)
    _EDGE_PART = {2: 1, 0: 2, 1: 3}                             # 0 인 바리센트릭 축 → 모서리 id

    for i in range(n_t):
        f_id = int(nearest_faces[i])
        nearest_f = src_f[f_id]
        rows[3 * i:3 * i + 3] = i
        cols[3 * i:3 * i + 3] = nearest_f

        bc = bary_hit[i]
        zeros = np.flatnonzero(bc <= tol)

        if zeros.size == 0:
            n_id = 0
        elif zeros.size == 1:
            n_id = _EDGE_PART[int(zeros[0])]
        else:
            n_id = 4 + int(np.argmax(bc))

        if n_id == 0:
            A = src_v[nearest_f].T                              # (3, 3), 열 = 삼각형 정점
            q = bary_hit[i] @ src_v[nearest_f]                  # 최근접점
            coeffs[3 * i:3 * i + 3] = np.linalg.lstsq(A, q, rcond=None)[0]
        elif 1 <= n_id <= 3:
            e0, e1 = n_id - 1, n_id % 3
            A = np.vstack((src_v[nearest_f[e0]], src_v[nearest_f[e1]])).T   # (3, 2)
            tmp = np.linalg.lstsq(A, tgt_v[i], rcond=None)[0]
            coeffs[3 * i + e0] = tmp[0]
            coeffs[3 * i + e1] = tmp[1]
        else:
            coeffs[3 * i + n_id - 4] = 1.0

    return sp.csc_matrix((coeffs, (rows, cols)), shape=(n_t, src_v.shape[0]))


def generate_transform_matrices(vertices, faces, factors):
    """계층 메쉬 + 다운샘플링(D) **및 업샘플링(U)** 행렬.

    공식 `generate_transform_matrices` 와 같은 구성이다. 인코더만 쓰는 경로는
    `generate_downsample_matrices` 로 충분하고, mesh→mesh AE 처럼 디코더가 필요한
    경로만 이쪽을 쓴다.

    Returns:
      faces_list, verts_list, down_list, up_list
        up_list[i]: level i+1 -> level i  (csc, float32)
    """
    faces_list, verts_list, down_list = generate_downsample_matrices(vertices, faces, factors)
    up_list = []
    for i in range(len(down_list)):
        coarse = Mesh(v=verts_list[i + 1], f=faces_list[i + 1])
        fine = Mesh(v=verts_list[i], f=faces_list[i])
        up_list.append(setup_deformation_transfer(coarse, fine).astype("float32"))
    return faces_list, verts_list, down_list, up_list


def generate_downsample_matrices(vertices, faces, factors):
    """Build hierarchical meshes + downsampling transforms.

    Returns:
      faces_list: F[0]=original, F[1], ...
      verts_list: V[0]=original, ...
      down_list:  D[i] maps level i -> level i+1  (csc, float32)
    """
    factors = [1.0 / float(f) for f in factors]
    mesh = Mesh(v=vertices, f=faces)
    faces_list = [mesh.f.copy()]
    verts_list = [mesh.v.copy()]
    down_list = []
    meshes = [mesh]

    for factor in factors:
        ds_f, ds_D = qslim_decimator_transformer(meshes[-1], factor=factor)
        down_list.append(ds_D.astype("float32"))
        new_v = ds_D.dot(meshes[-1].v)
        new_mesh = Mesh(v=new_v, f=ds_f)
        faces_list.append(new_mesh.f)
        verts_list.append(new_mesh.v)
        meshes.append(new_mesh)

    return faces_list, verts_list, down_list


def scipy_to_torch_sparse(spmat) -> "torch.Tensor":
    import torch

    coo = spmat.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, tuple(coo.shape)).coalesce()
