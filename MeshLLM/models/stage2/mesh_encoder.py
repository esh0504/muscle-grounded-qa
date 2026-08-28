"""Frozen Stage-1 mesh encoder wrapper for Stage-2.

Stage-1 의 SpiralNet++ 인코더를 동결한 채 mesh 토큰 시퀀스를 뽑아 주는 어댑터와,
그 출력의 (정점,채널)별 평균/표준편차를 학습 split 표본에서 재는 유틸이 들어 있다.
표준화가 왜 필요한지는 `compute_feature_stats` / `FrozenMeshEncoder` 주석 참고.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from datasets.mesh_dataset import N_SURF


# --------------------------------------------------------------------------- #
# 2. Frozen Stage-1 mesh encoder wrapper
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_feature_stats(encoder, mesh_root="DATA/mesh", split="train",
                          n: int = 8000, batch: int = 256, seed: int = 42):
    """동결 인코더 출력의 (정점,채널)별 평균/표준편차를 학습 split 표본에서 잰다.

    항목 독립인 DC 성분을 없애기 위한 것이다 — 없으면 서로 다른 mesh 의 prefix 가
    코사인 유사도 0.99+ 로 붕괴해 LLM 이 mesh 를 읽지 못한다 (진단 SUMMARY.md).
    정점 축을 뭉개면 안 된다: DC 성분이 정점마다 달라 평균이 상쇄된다.
    """
    import numpy as np

    from datasets.split_trainvaltest import load_split_indices

    root = Path(mesh_root)
    txt = (root / "topology.obj").read_text(encoding="utf-8", errors="ignore")
    rest = np.array([ln.split()[1:4] for ln in txt.splitlines() if ln.startswith("v ")],
                    dtype=np.float32)[:N_SURF]
    pool = np.array(sorted(load_split_indices(root, split)))
    idx = np.random.default_rng(seed).choice(pool, min(n, len(pool)), replace=False)

    dev = next(encoder.parameters()).device
    acc = None
    for st in range(0, len(idx), batch):
        chunk = idx[st:st + batch]
        arr = np.zeros((len(chunk), N_SURF, 3), np.float32)
        by: dict[int, list] = {}
        for i, g in enumerate(chunk):
            by.setdefault(int(g) // 1000, []).append((i, int(g) % 1000))
        for s, lst in by.items():
            a = np.fromfile(root / f"verts/shard_{s:05d}.bin",
                            dtype=">f4").reshape(-1, N_SURF, 3)
            for i, l in lst:
                arr[i] = a[l]
        v, lat = encoder(torch.from_numpy(arr - rest).to(dev))
        v = v.reshape(v.size(0), -1).double()
        lat = lat.double()
        cur = [v.sum(0), (v * v).sum(0), lat.sum(0), (lat * lat).sum(0), v.size(0)]
        acc = cur if acc is None else [a + b for a, b in zip(acc[:4], cur[:4])] + \
            [acc[4] + cur[4]]

    n_tot = acc[4]
    vm = acc[0] / n_tot
    vs = ((acc[1] / n_tot) - vm ** 2).clamp_min(0).sqrt()
    lm = acc[2] / n_tot
    ls = ((acc[3] / n_tot) - lm ** 2).clamp_min(0).sqrt()
    n_m = encoder.n_mesh_tokens - 1
    return {"v_mean": vm.float().view(n_m, -1), "v_std": vs.float().view(n_m, -1),
            "l_mean": lm.float(), "l_std": ls.float()}


class FrozenMeshEncoder(nn.Module):
    """Wraps the Stage-1 SpiralEncoder; outputs a mesh token sequence.

    per-vertex tokens : (B, N_m, C)  from the last down-sampled level (pre-FC)
    global latent      : (B, latent) from the encoder FC head
    -> returned as (B, N_m + 1, C_out) after per-source linear projections.
    """

    # 배치에서 인코더 입력이 담기는 키.
    input_key = "disp"

    def __init__(self, stage1_ckpt: str | Path | None,
                 model_cfg_path: str | Path = "configs/models/stage1_model.yaml"):
        """stage1_ckpt=None 이면 무작위 초기화 인코더를 동결해 쓴다.

        random-init 동결 인코더는 "근육 사전학습이 기여했다"를 분리하는 ablation row 다
        (doc/experiment.md Phase 3). 구조·동결 방식은 동일하고 가중치만 학습 전 상태다.
        """
        super().__init__()
        from models import find_model_def

        cfg = OmegaConf.load(model_cfg_path)
        ModelClass = find_model_def(cfg.name, cfg.class_name)
        full_model = ModelClass(cfg)

        if stage1_ckpt is None:
            print("[stage2] encoder_init=random — Stage-1 가중치를 불러오지 않는다 (동결은 동일)")
        else:
            ckpt = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
            state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            full_model.load_state_dict(state)

        self.encoder = full_model.encoder  # SpiralEncoder
        self.n_verts_in = full_model.n_verts
        # 공식 SpiralNet++ AE 구조: en_layers = [SpiralEnblock ...] + [Linear(latent)].
        # 마지막 원소가 latent FC 이고 그 앞이 spiral conv/pool 블록이다.
        self.en_blocks = self.encoder.en_layers[:-1]
        self.latent_fc = self.encoder.en_layers[-1]
        self.per_vertex_channels = int(self.encoder.out_channels[-1])
        self.latent_channels = int(self.latent_fc.out_features)
        self.n_mesh_tokens = int(self.encoder.num_vert) + 1  # +1 global token

        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        # 특징 표준화 버퍼. 학습셋에서 미리 잰 채널별 평균/표준편차로 z-score 한다.
        # 왜 필요한가 (진단 SUMMARY.md): 동결 인코더 출력은 **항목 독립인 DC 성분이 지배**해
        # 서로 다른 mesh 사이 코사인 유사도가 0.9865, std/|값| 0.11 밖에 안 된다. 정보는
        # 1.35% 잔차에만 있다. 선형 프로브는 그 방향을 임의로 증폭해 R²=0.93 을 얻지만
        # LLM 의 attention 은 그럴 수 없다 — 실측으로 같은 값을 텍스트로 주면 CE 0.0000,
        # mesh prefix 로 주면 무정보 바닥(0.103)에 머물렀다.
        # z-score 는 DC 성분을 없애고 항목 간 변동을 O(1) 로 올린다. 상수라 batch=1 에서도 쓴다.
        n_m = self.n_mesh_tokens - 1
        self.register_buffer("v_mean", torch.zeros(n_m, self.per_vertex_channels))
        self.register_buffer("v_std", torch.ones(n_m, self.per_vertex_channels))
        self.register_buffer("l_mean", torch.zeros(self.latent_channels))
        self.register_buffer("l_std", torch.ones(self.latent_channels))
        self.feature_norm = False

    def load_feature_stats(self, path: str | Path):
        """통계를 싣고 표준화를 켠다.

        `path == "auto"` 면 학습 split 표본으로 **즉석에서** 잰다. random-init 인코더
        (randinit_en row)는 가중치가 매번 달라 미리 만든 파일이 안 맞으므로 auto 가 안전하다.
        """
        if str(path) == "auto":
            st = compute_feature_stats(self)
            print("[stage2] 인코더 특징 표준화 켜짐 (auto — 학습 split 표본에서 즉석 계산)")
        else:
            st = torch.load(path, map_location="cpu", weights_only=False)
            print(f"[stage2] 인코더 특징 표준화 켜짐 ← {path}")
        self.v_mean.copy_(st["v_mean"].to(self.v_mean.device))
        self.v_std.copy_(st["v_std"].to(self.v_std.device).clamp_min(1e-6))
        self.l_mean.copy_(st["l_mean"].to(self.l_mean.device))
        self.l_std.copy_(st["l_std"].to(self.l_std.device).clamp_min(1e-6))
        self.feature_norm = True

    def train(self, mode: bool = True):  # keep encoder in eval() always
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, disp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, V, 3) 또는 (B, M, V, 3) -> per-vertex, global.

        dose_response 레코드는 mesh 를 여러 개 참조하므로 M 축을 받는다.
        """
        single = disp.dim() == 3
        if single:
            disp = disp.unsqueeze(1)
        b, m = disp.shape[:2]
        x = disp.reshape(b * m, disp.size(2), disp.size(3))
        for i, block in enumerate(self.en_blocks):
            x = block(x, self.encoder.down_transform[i])  # (B*M, N_m, C)
        latent = self.latent_fc(x.reshape(x.size(0), -1))  # (B*M, latent)
        if self.feature_norm:
            x = (x - self.v_mean) / self.v_std
            latent = (latent - self.l_mean) / self.l_std
        x = x.view(b, m, x.size(1), x.size(2))
        latent = latent.view(b, m, latent.size(1))
        return (x[:, 0], latent[:, 0]) if single else (x, latent)
