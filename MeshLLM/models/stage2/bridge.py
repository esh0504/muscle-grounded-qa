"""mesh 토큰 → LLM prefix 다리(bridge).

두 갈래가 들어 있다: Q-Former식 `CrossAttentionFusion`(+`FusionLayer`)과 LLaVA-1.5식
토큰별 선형 사영 `MlpBridge`. 왜 실제로는 MlpBridge 를 쓰는지는 그 독스트링의 실측 근거
참고. `splice_prefix` 는 만들어진 prefix 를 토큰 임베딩에 끼워 넣는 함수다.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from datasets.qa_dataset import IGNORE_INDEX


# --------------------------------------------------------------------------- #
# 3. Cross-Attention Fusion (Q-Former식) + projector
# --------------------------------------------------------------------------- #
class FusionLayer(nn.Module):
    """self-attn(query+text) -> cross-attn(query stream -> mesh) -> FFN."""

    def __init__(self, dim: int, heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(),
            nn.Linear(dim * ffn_mult, dim), nn.Dropout(dropout),
        )

    def forward(self, stream, mesh, stream_pad_mask=None, mesh_pad_mask=None):
        h = self.norm1(stream)
        s, _ = self.self_attn(h, h, h, key_padding_mask=stream_pad_mask)
        stream = stream + s
        h = self.norm2(stream)
        c, _ = self.cross_attn(h, mesh, mesh, key_padding_mask=mesh_pad_mask)
        stream = stream + c
        stream = stream + self.ffn(self.norm3(stream))
        return stream


class CrossAttentionFusion(nn.Module):
    """Fuse frozen mesh tokens with the question, emit LLM prefix tokens.

    learnable query tokens carry the question (via self-attn) and absorb mesh
    features (via cross-attn K/V = mesh). Output query part -> projector -> LLM.
    """

    def __init__(self, mesh_per_vertex_c: int, mesh_latent_c: int,
                 text_dim: int, llm_hidden: int, *, dim: int = 512,
                 n_query: int = 32, layers: int = 4, heads: int = 8,
                 dropout: float = 0.1, prefix_norm: bool = True,
                 prefix_scale_init: float = 0.022):
        super().__init__()
        self.dim = dim
        self.n_query = n_query
        self.query_tokens = nn.Parameter(torch.randn(1, n_query, dim) * 0.02)

        self.mesh_vertex_proj = nn.Linear(mesh_per_vertex_c, dim)
        self.mesh_latent_proj = nn.Linear(mesh_latent_c, dim)
        self.text_proj = nn.Linear(text_dim, dim)
        self.mesh_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)

        self.layers = nn.ModuleList(
            [FusionLayer(dim, heads, dropout=dropout) for _ in range(layers)]
        )
        self.out_norm = nn.LayerNorm(dim)
        self.projector = nn.Sequential(
            nn.Linear(dim, llm_hidden), nn.GELU(), nn.Linear(llm_hidden, llm_hidden)
        )
        # prefix 를 LLM token embedding 과 같은 크기 대역으로 맞춘다.
        # 맞추지 않으면 prefix 가 token embedding 보다 훨씬 커서(실측 8.2배) 초반 LLM 을 교란한다.
        # LayerNorm 출력 RMS≈1 이므로 scale 초기값 = 목표 RMS (Qwen3-8B 는 0.0220).
        # 학습 가능한 스칼라라 필요하면 모델이 스스로 키운다 (강제 정규화가 아니다).
        self.prefix_norm = nn.LayerNorm(llm_hidden) if prefix_norm else None
        self.prefix_scale = nn.Parameter(torch.tensor(float(prefix_scale_init))) \
            if prefix_norm else None

    def forward(self, mesh_vertex, mesh_latent, text_embeds, text_mask=None,
                mesh_valid=None):
        """
        mesh_vertex (B, N_m, C) 또는 (B, M, N_m, C), mesh_latent 도 마찬가지,
        text_embeds (B, L, text_dim), text_mask (B, L) 1=keep,
        mesh_valid (B, M) — 패딩된 mesh 슬롯을 걸러낸다.
        Returns LLM prefix embeds (B, n_query, llm_hidden).
        """
        if mesh_vertex.dim() == 3:              # 단일 mesh → M=1 로 통일
            mesh_vertex = mesh_vertex.unsqueeze(1)
            mesh_latent = mesh_latent.unsqueeze(1)
        b, m, n_m, _ = mesh_vertex.shape

        mv = self.mesh_vertex_proj(mesh_vertex)                  # (B,M,N_m,dim)
        ml = self.mesh_latent_proj(mesh_latent).unsqueeze(2)     # (B,M,1,dim)
        mesh = torch.cat([ml, mv], dim=2).reshape(b, m * (n_m + 1), self.dim)
        mesh = self.mesh_norm(mesh)                              # (B, M*(N_m+1), dim)

        # mesh 쪽 padding: True = 무시. mesh 토큰은 mesh 단위로 (N_m+1)개씩 묶여 있다.
        mesh_pad = None
        if mesh_valid is not None:
            mesh_pad = (~mesh_valid.bool()).repeat_interleave(n_m + 1, dim=1)

        text = self.text_norm(self.text_proj(text_embeds))     # (B, L, dim)
        query = self.query_tokens.expand(b, -1, -1)            # (B, N_q, dim)
        stream = torch.cat([query, text], dim=1)               # (B, N_q+L, dim)

        # key_padding_mask: True = ignore. queries always valid.
        stream_pad = None
        if text_mask is not None:
            q_pad = torch.zeros(b, self.n_query, dtype=torch.bool, device=text.device)
            stream_pad = torch.cat([q_pad, ~text_mask.bool()], dim=1)

        for layer in self.layers:
            stream = layer(stream, mesh, stream_pad_mask=stream_pad,
                           mesh_pad_mask=mesh_pad)

        fused = self.out_norm(stream[:, : self.n_query])       # (B, N_q, dim)
        prefix = self.projector(fused)                         # (B, N_q, llm_hidden)
        if self.prefix_norm is not None:
            prefix = self.prefix_norm(prefix) * self.prefix_scale
        return prefix


class MlpBridge(nn.Module):
    """mesh 토큰을 **토큰 단위로** LLM 임베딩 공간에 사영한다 (LLaVA-1.5 방식).

    Q-Former(CrossAttentionFusion)를 쓰지 않는 이유는 실측 때문이다 (진단 SUMMARY.md):
    항목 간 변별력(std/|값|)이 mesh 토큰 0.1106 → fused 0.0147 → projector 0.0018 로
    무너지고, 서로 다른 mesh 의 prefix 코사인 유사도가 1.0000 이 된다. 원인은 stream 이
    **항목 공통 상수**(query_tokens + 거의 같은 질문)로 시작하고 mesh 가 residual 로만
    얹히는 구조다. 상수가 방향을 지배한다.

    반면 선형 프로브는 mesh 토큰에서 기하량을 R²=0.93 으로 복원한다 — 즉 **선형 사상이면
    충분하다**. 토큰마다 독립적으로 사영하면 항목 간 변별력이 그대로 보존된다.

    prefix 토큰 수 = M × (N_m + 1) (mesh 당 24 + latent 1 = 25). 질문은 쓰지 않는다
    (LLM 이 이미 텍스트로 본다).
    """

    def __init__(self, mesh_per_vertex_c: int, mesh_latent_c: int, llm_hidden: int, *,
                 hidden: int | None = None, prefix_norm: bool = True,
                 prefix_scale_init: float = 0.022, dropout: float = 0.0):
        super().__init__()
        h = hidden or llm_hidden
        self.vertex_proj = nn.Linear(mesh_per_vertex_c, h)
        self.latent_proj = nn.Linear(mesh_latent_c, h)
        self.mlp = nn.Sequential(nn.Linear(h, llm_hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(llm_hidden, llm_hidden))
        self.prefix_norm = nn.LayerNorm(llm_hidden) if prefix_norm else None
        self.prefix_scale = nn.Parameter(torch.tensor(float(prefix_scale_init))) \
            if prefix_norm else None

    def forward(self, mesh_vertex, mesh_latent, text_embeds=None, text_mask=None,
                mesh_valid=None):
        if mesh_vertex.dim() == 3:
            mesh_vertex = mesh_vertex.unsqueeze(1)
            mesh_latent = mesh_latent.unsqueeze(1)
        b, m, n_m, _ = mesh_vertex.shape
        tok = torch.cat([self.latent_proj(mesh_latent).unsqueeze(2),
                         self.vertex_proj(mesh_vertex)], dim=2)     # (B,M,N_m+1,h)
        prefix = self.mlp(tok.reshape(b, m * (n_m + 1), -1))         # (B, M*(N_m+1), H)
        if self.prefix_norm is not None:
            prefix = self.prefix_norm(prefix) * self.prefix_scale
        return prefix

    @staticmethod
    def n_prefix_tokens(n_mesh: int, n_vertex_tokens: int = 24) -> int:
        return n_mesh * (n_vertex_tokens + 1)


def splice_prefix(prefix, tok_embeds, attn, labels=None, weights=None, mesh_at=None):
    """mesh prefix 를 토큰 임베딩에 넣는다. (embeds, attn, labels, weights) 를 돌려준다.

    두 방식:
      inline  — `mesh_at` 이 가리키는 자리(현재 user 턴 본문 맨 앞에 깔린 자리표시자 토큰)의
                임베딩을 prefix 로 **덮어쓴다**. 길이가 안 변하므로 labels/attn/weights 를
                손댈 필요가 없다 (그 자리는 프롬프트 구간이라 이미 IGNORE/0 이다).
      prepend — 대화 전체 앞에 이어붙인다. mesh_at 이 없거나 -1(잘림)일 때의 fallback.

    inline 이 기본이어야 하는 이유는 진단 SUMMARY.md 참조: prepend 는 prefix 가
    <|im_start|>system 보다 앞에 놓여, 기하 정보가 prefix 까지 R²=0.932 로 도달함에도
    instruction-tuned LLM 이 그것을 읽지 않았다.
    """
    b, n_q, _ = prefix.shape
    if mesh_at is not None and bool((mesh_at >= 0).all()):
        embeds = tok_embeds.clone()
        ar = torch.arange(n_q, device=prefix.device)
        pos = mesh_at.to(prefix.device).unsqueeze(1) + ar.unsqueeze(0)      # (B, n_q)
        embeds.scatter_(1, pos.unsqueeze(-1).expand(-1, -1, embeds.size(-1)),
                        prefix.to(embeds.dtype))
        return embeds, attn, labels, weights

    embeds = torch.cat([prefix.to(tok_embeds.dtype), tok_embeds], dim=1)
    pre_attn = torch.ones(b, n_q, dtype=attn.dtype, device=prefix.device)
    attn = torch.cat([pre_attn, attn], dim=1)
    if labels is not None:
        labels = torch.cat([torch.full((b, n_q), IGNORE_INDEX, dtype=labels.dtype,
                                       device=prefix.device), labels], dim=1)
    if weights is not None:
        weights = torch.cat([torch.zeros(b, n_q, dtype=weights.dtype,
                                         device=prefix.device), weights], dim=1)
    return embeds, attn, labels, weights
