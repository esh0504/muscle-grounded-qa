"""torch_scatter.scatter_add 대체 (network.py의 Pool 전용).

공식 코드는 `from torch_scatter import scatter_add` 를 쓰지만, torch_scatter의 마지막
릴리스(2.1.2, 2023)는 이 환경의 torch 2.11 + nvcc 부재에서 빌드되지 않는다.
scatter_add 는 torch 기본 `index_add_` 와 정의가 같으므로 그대로 옮겨 적었다.
동일성은 tests 에서 확인한다 (models/stage1/spiralnet/official/VENDORED.md 참고).
"""

from __future__ import annotations

import torch


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size=None):
    if dim < 0:
        dim += src.dim()
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    return out.index_add_(dim, index, src)
