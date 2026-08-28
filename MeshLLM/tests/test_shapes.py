"""bridge 모듈의 텐서 계약(shape)과 손실 불변식 — 작은 더미 텐서로만.

GPU·LLM·체크포인트 없이 돈다. transformers/peft 는 일부러 import 하지 않는다
(느리고, 여기서 확인하려는 것과 무관하다).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from datasets.qa_dataset import IGNORE_INDEX                      # noqa: E402
from losses.lm_loss import weighted_causal_lm_loss           # noqa: E402
from models.stage2.bridge import (                           # noqa: E402
    CrossAttentionFusion,
    MlpBridge,
    splice_prefix,
)

# 실제 Stage-2 는 (mesh 당 정점 토큰 24 + latent 1) 이지만 여기서는 축소판을 쓴다.
B, M, N_M, C = 2, 2, 24, 16
LATENT, TEXT_DIM, LLM_HIDDEN = 32, 24, 64


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)


def _mesh_inputs():
    return (torch.randn(B, M, N_M, C), torch.randn(B, M, LATENT))


# --------------------------------------------------------------------------- #
# MlpBridge — 토큰별 사영 (LLaVA-1.5 방식). prefix 토큰 = M × (N_m + 1)
# --------------------------------------------------------------------------- #
def test_mlp_bridge_shape():
    bridge = MlpBridge(C, LATENT, LLM_HIDDEN)
    prefix = bridge(*_mesh_inputs())
    assert prefix.shape == (B, M * (N_M + 1), LLM_HIDDEN)


def test_mlp_bridge_accepts_single_mesh():
    """mesh_vertex 가 3차원이면 M=1 로 승격된다."""
    bridge = MlpBridge(C, LATENT, LLM_HIDDEN)
    prefix = bridge(torch.randn(B, N_M, C), torch.randn(B, LATENT))
    assert prefix.shape == (B, N_M + 1, LLM_HIDDEN)


def test_mlp_bridge_n_prefix_tokens():
    assert MlpBridge.n_prefix_tokens(M, N_M) == M * (N_M + 1)
    assert MlpBridge.n_prefix_tokens(1) == 25          # 기본 24 정점 토큰 + latent 1
    assert MlpBridge.n_prefix_tokens(6) == 150


def test_mlp_bridge_n_prefix_tokens_matches_forward():
    bridge = MlpBridge(C, LATENT, LLM_HIDDEN)
    prefix = bridge(*_mesh_inputs())
    assert prefix.shape[1] == MlpBridge.n_prefix_tokens(M, N_M)


def test_mlp_bridge_is_item_discriminative():
    """항목마다 다른 prefix 가 나와야 한다 (qformer 를 버린 이유 = 항목 변별력 붕괴)."""
    bridge = MlpBridge(C, LATENT, LLM_HIDDEN)
    prefix = bridge(*_mesh_inputs())
    assert not torch.allclose(prefix[0], prefix[1], atol=1e-4)


def test_mlp_bridge_without_prefix_norm():
    bridge = MlpBridge(C, LATENT, LLM_HIDDEN, prefix_norm=False)
    assert bridge.prefix_norm is None and bridge.prefix_scale is None
    assert bridge(*_mesh_inputs()).shape == (B, M * (N_M + 1), LLM_HIDDEN)


# --------------------------------------------------------------------------- #
# CrossAttentionFusion — Q-Former 식. 출력은 항상 (B, n_query, llm_hidden)
# --------------------------------------------------------------------------- #
def test_cross_attention_fusion_shape():
    n_query, L = 4, 7
    fusion = CrossAttentionFusion(C, LATENT, TEXT_DIM, LLM_HIDDEN,
                                  dim=32, n_query=n_query, layers=1, heads=2)
    prefix = fusion(*_mesh_inputs(), torch.randn(B, L, TEXT_DIM))
    assert prefix.shape == (B, n_query, LLM_HIDDEN)


def test_cross_attention_fusion_with_masks():
    n_query, L = 4, 7
    fusion = CrossAttentionFusion(C, LATENT, TEXT_DIM, LLM_HIDDEN,
                                  dim=32, n_query=n_query, layers=2, heads=2)
    text_mask = torch.ones(B, L, dtype=torch.long)
    text_mask[1, 4:] = 0                    # 패딩된 질문 토큰
    mesh_valid = torch.ones(B, M, dtype=torch.bool)
    mesh_valid[1, 1] = False                # 패딩된 mesh 슬롯
    prefix = fusion(*_mesh_inputs(), torch.randn(B, L, TEXT_DIM),
                    text_mask=text_mask, mesh_valid=mesh_valid)
    assert prefix.shape == (B, n_query, LLM_HIDDEN)
    assert torch.isfinite(prefix).all()


def test_cross_attention_fusion_single_mesh():
    n_query, L = 4, 5
    fusion = CrossAttentionFusion(C, LATENT, TEXT_DIM, LLM_HIDDEN,
                                  dim=32, n_query=n_query, layers=1, heads=2)
    prefix = fusion(torch.randn(B, N_M, C), torch.randn(B, LATENT),
                    torch.randn(B, L, TEXT_DIM))
    assert prefix.shape == (B, n_query, LLM_HIDDEN)


# --------------------------------------------------------------------------- #
# splice_prefix — inline(mesh_at) / prepend(fallback) 두 경로
# --------------------------------------------------------------------------- #
def _lm_batch(n_q, t_len=20):
    tok_embeds = torch.randn(B, t_len, LLM_HIDDEN)
    attn = torch.ones(B, t_len, dtype=torch.long)
    labels = torch.full((B, t_len), IGNORE_INDEX, dtype=torch.long)
    labels[:, -5:] = torch.randint(0, 100, (B, 5))
    weights = torch.zeros(B, t_len)
    weights[:, -5:] = 1.0
    return tok_embeds, attn, labels, weights


def test_splice_prefix_inline_keeps_length():
    n_q = M * (N_M + 1)
    prefix = torch.randn(B, n_q, LLM_HIDDEN)
    tok_embeds, attn, labels, weights = _lm_batch(n_q, t_len=n_q + 30)
    mesh_at = torch.tensor([3, 5])

    embeds, out_attn, out_labels, out_weights = splice_prefix(
        prefix, tok_embeds, attn, labels, weights, mesh_at=mesh_at)

    # 길이가 안 변하므로 attn/labels/weights 는 손대지 않는다
    assert embeds.shape == tok_embeds.shape
    assert out_attn.shape == attn.shape
    assert torch.equal(out_attn, attn)
    assert torch.equal(out_labels, labels)
    assert torch.equal(out_weights, weights)


def test_splice_prefix_inline_writes_prefix_at_mesh_at():
    n_q = 6
    prefix = torch.randn(B, n_q, LLM_HIDDEN)
    tok_embeds, attn, labels, weights = _lm_batch(n_q)
    mesh_at = torch.tensor([2, 7])

    embeds, *_ = splice_prefix(prefix, tok_embeds, attn, labels, weights,
                               mesh_at=mesh_at)
    for b in range(B):
        start = int(mesh_at[b])
        assert torch.allclose(embeds[b, start:start + n_q], prefix[b])
        assert torch.allclose(embeds[b, :start], tok_embeds[b, :start])
        assert torch.allclose(embeds[b, start + n_q:], tok_embeds[b, start + n_q:])
    # 원본 임베딩은 그대로여야 한다 (clone 후 scatter)
    assert not embeds.data_ptr() == tok_embeds.data_ptr()


def test_splice_prefix_prepend_path():
    n_q, t_len = 6, 20
    prefix = torch.randn(B, n_q, LLM_HIDDEN)
    tok_embeds, attn, labels, weights = _lm_batch(n_q, t_len=t_len)

    embeds, out_attn, out_labels, out_weights = splice_prefix(
        prefix, tok_embeds, attn, labels, weights, mesh_at=None)

    assert embeds.shape == (B, n_q + t_len, LLM_HIDDEN)
    assert out_attn.shape == (B, n_q + t_len)
    assert out_labels.shape == (B, n_q + t_len)
    assert out_weights.shape == (B, n_q + t_len)
    assert torch.allclose(embeds[:, :n_q], prefix)
    assert torch.allclose(embeds[:, n_q:], tok_embeds)
    assert (out_attn[:, :n_q] == 1).all()
    assert (out_labels[:, :n_q] == IGNORE_INDEX).all()      # prefix 는 감독하지 않는다
    assert (out_weights[:, :n_q] == 0).all()
    assert torch.equal(out_labels[:, n_q:], labels)


def test_splice_prefix_prepend_when_mesh_at_truncated():
    """mesh_at 이 -1(자리표시자가 잘림)이면 prepend 로 폴백한다."""
    n_q, t_len = 6, 20
    prefix = torch.randn(B, n_q, LLM_HIDDEN)
    tok_embeds, attn, labels, weights = _lm_batch(n_q, t_len=t_len)
    mesh_at = torch.tensor([3, -1])

    embeds, out_attn, *_ = splice_prefix(prefix, tok_embeds, attn, labels, weights,
                                         mesh_at=mesh_at)
    assert embeds.shape == (B, n_q + t_len, LLM_HIDDEN)
    assert out_attn.shape == (B, n_q + t_len)


def test_splice_prefix_prepend_allows_none_labels():
    n_q, t_len = 6, 20
    prefix = torch.randn(B, n_q, LLM_HIDDEN)
    tok_embeds = torch.randn(B, t_len, LLM_HIDDEN)
    attn = torch.ones(B, t_len, dtype=torch.long)

    embeds, out_attn, out_labels, out_weights = splice_prefix(
        prefix, tok_embeds, attn, None, None, mesh_at=None)
    assert embeds.shape == (B, n_q + t_len, LLM_HIDDEN)
    assert out_attn.shape == (B, n_q + t_len)
    assert out_labels is None and out_weights is None


def test_mlp_bridge_prefix_splices_end_to_end():
    """MlpBridge 출력 토큰 수와 inline 자리표시자 길이가 맞물리는지."""
    bridge = MlpBridge(C, LATENT, LLM_HIDDEN)
    prefix = bridge(*_mesh_inputs())
    n_q = MlpBridge.n_prefix_tokens(M, N_M)
    assert prefix.shape[1] == n_q
    tok_embeds, attn, labels, weights = _lm_batch(n_q, t_len=n_q + 16)
    embeds, *_ = splice_prefix(prefix, tok_embeds, attn, labels, weights,
                               mesh_at=torch.tensor([2, 4]))
    assert embeds.shape == tok_embeds.shape


# --------------------------------------------------------------------------- #
# weighted_causal_lm_loss — 독스트링이 주장하는 불변식
#   "With weights all 1.0 on supervised tokens this equals the standard mean loss."
# --------------------------------------------------------------------------- #
def _lm_logits_labels(t_len=12, vocab=37):
    logits = torch.randn(B, t_len, vocab)
    labels = torch.full((B, t_len), IGNORE_INDEX, dtype=torch.long)
    labels[:, 6:] = torch.randint(0, vocab, (B, t_len - 6))
    return logits, labels


def _reference_mean_ce(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def test_weighted_loss_none_weights_equals_mean_ce():
    logits, labels = _lm_logits_labels()
    got = weighted_causal_lm_loss(logits, labels, None)
    assert torch.allclose(got, _reference_mean_ce(logits, labels), atol=1e-6)


def test_weighted_loss_all_ones_equals_mean_ce():
    logits, labels = _lm_logits_labels()
    weights = torch.ones_like(labels, dtype=torch.float)
    got = weighted_causal_lm_loss(logits, labels, weights)
    assert torch.allclose(got, _reference_mean_ce(logits, labels), atol=1e-6)


def test_weighted_loss_upweights_fact_tokens():
    """fact 토큰 가중을 키우면 손실이 그 토큰 쪽으로 끌려간다."""
    logits, labels = _lm_logits_labels()
    weights = torch.ones_like(labels, dtype=torch.float)
    weights[:, -3:] = 5.0
    weighted = weighted_causal_lm_loss(logits, labels, weights)
    plain = weighted_causal_lm_loss(logits, labels, None)
    assert weighted.shape == plain.shape == ()
    assert not torch.allclose(weighted, plain)

    # sum(w*CE)/sum(w over supervised) 를 직접 계산한 값과 같아야 한다
    shift_logits, shift_labels = logits[:, :-1, :], labels[:, 1:]
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)).float(),
        shift_labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="none",
    ).view(B, -1)
    sup = (shift_labels != IGNORE_INDEX).float()
    w = weights[:, 1:] * sup
    assert torch.allclose(weighted, (ce * w).sum() / w.sum(), atol=1e-6)


def test_weighted_loss_stats_keys():
    """plateau 진단용 분해 (stats=True) 가 기대 키를 다 돌려주는지."""
    logits, labels = _lm_logits_labels()
    weights = torch.ones_like(labels, dtype=torch.float)
    weights[:, -3:] = 5.0
    loss, stats = weighted_causal_lm_loss(logits, labels, weights, stats=True)
    assert loss.shape == ()
    for key in ("unweighted_ce", "fact_ce", "nonfact_ce", "n_supervised", "n_fact",
                "fact_token_ratio", "empty_supervision"):
        assert key in stats, key
    # 가중 없는 CE 는 표준 평균 CE 와 같다
    assert torch.allclose(stats["unweighted_ce"], _reference_mean_ce(logits, labels),
                          atol=1e-6)
    assert float(stats["n_fact"]) == 3 * B          # 마지막 3 토큰 × 배치


def test_weighted_loss_all_ignored_does_not_nan():
    """감독 토큰이 하나도 없어도 NaN 이 나오면 안 된다 (denom clamp_min)."""
    logits, _ = _lm_logits_labels()
    labels = torch.full((B, logits.size(1)), IGNORE_INDEX, dtype=torch.long)
    loss = weighted_causal_lm_loss(logits, labels, None)
    assert torch.isfinite(loss) and float(loss) == 0.0

