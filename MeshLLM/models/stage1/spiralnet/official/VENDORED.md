# 벤더링 기록 — sw-gong/spiralnet_plus

- 출처: https://github.com/sw-gong/spiralnet_plus
- 커밋: `96df4d8779352d5161b798c00504f276b684c20e` (2020-02-09)
- 가져온 날: 2026-07-31

| 이 저장소 | 원본 경로 | 상태 |
|---|---|---|
| `spiralconv.py` | `conv/spiralconv.py` | **무수정** |
| `network.py` | `reconstruction/network.py` | import 2줄만 변경 (아래) |
| `generate_spiral_seq.py` | `utils/generate_spiral_seq.py` | **무수정** |
| `utils.py` | `utils/utils.py` | **무수정** |
| `scatter.py` | (없음) | 추가 — `torch_scatter.scatter_add` 대체 |

## network.py 에서 바꾼 2줄

```diff
-from torch_scatter import scatter_add
-from conv import SpiralConv
+from .scatter import scatter_add
+from .spiralconv import SpiralConv
```

`SpiralConv` 는 경로만 바뀌었고 코드는 같다. `scatter_add` 를 대체한 이유:

- `torch-scatter` 의 마지막 릴리스는 2.1.2 (2023, torch ≤2.1 대상)이고, 이 환경은 torch 2.11 이다.
- 이 컨테이너에는 nvcc / CUDA_HOME 이 없어 GPU 빌드 자체가 불가능하다 (CPU 전용 빌드는 학습에 못 씀).
- `scatter_add(src, index, dim, dim_size)` 는 `torch.zeros(...).index_add_(dim, index, src)` 와 정의가
  동일하다. `models/stage1/spiralnet/official/scatter.py` 는 그 정의를 그대로 옮긴 것이고,
  공식 `Pool` 결과가 일치하는지는 `python -m models.stage1.spiralnet.test_official_parity` 로 확인한다.

## 함께 쓰는 비-공식 파일

- `../mesh_sampling.py` — 공식 `utils/mesh_sampling.py` 의 SciPy 이식본. 원본은
  `from psbody.mesh import Mesh` 를 요구하는데 psbody-mesh 는 PyPI 에 없다(소스 빌드에 boost 필요).
  qslim 감쇠 함수들(`vertex_quadrics`, `get_vertices_per_edge`, `qslim_decimator_transformer`,
  `_get_sparse_transform`)은 원본과 같은 로직이다.

  업샘플링 행렬 `U`(`setup_deformation_transfer`)도 이식돼 있다 (인코더만 쓰는
  현재 경로에는 필요 없지만, 디코더/업샘플링 용도로 함께 유지한다). 원본이 psbody AABB tree 로 얻는 **(최근접 삼각형, 부위 id, 최근접점)**
  세 값만 `_closest_point_barycentric` (Ericson, *Real-Time Collision Detection* §5.1.5 의
  점-삼각형 최근접점, 전수 계산)로 대체했고, **계수 계산 규칙(part 0/1-3/4-6 별 lstsq)은
  원본 그대로**다. 계층 메쉬가 370 → 93 → 24 로 작아 전수 계산으로 충분하다.

  검증: `U @ coarse_verts` 로 fine 정점을 복원했을 때 상대오차 1.50% / 5.06%,
  삼각형 내부에 떨어진 행(354/370, 78/93)은 row-sum 이 정확히 1 이다. 모서리·정점에
  떨어진 행의 합이 1 에서 살짝 벗어나는 것도 원본과 같다 (원본의 모서리 분기가 최근접점이
  아니라 대상 정점을 두 끝점의 **선형** span 에 최소자승 투영하기 때문).

- `../blocks.py::SpiralDecoder` — 공식 `AE.__init__` 의 decoder 구성과 `AE.decoder` 를
  그대로 옮긴 것이다. 인코더가 `SpiralEncoder` 로 분리돼 있어 짝을 맞춘 것뿐, 로직 변경은 없다.
- `../blocks.py::ClassificationHead` — 공식엔 없는 회귀 헤드 (latent → 근육 활성 11개).

## 의존성

- `openmesh` 1.2.1 — `pip install openmesh --no-build-isolation` 로 py3.11 에서 소스 빌드됨.
  공식 `generate_spiral_seq.py` / `utils.py` 가 그대로 쓴다.
