"""Stage-2 loss: weighted causal-LM (next-token) cross-entropy.

fact 토큰에 가중을 걸어 학습하므로 손실은 단순 평균이 아니라
sum(w * CE) / sum(w over supervised) 다. 가중이 전부 1.0 이면 표준 평균 손실과 같다.

Moved verbatim from ``train_s2.py`` (weighted_causal_lm_loss).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.qa_dataset import IGNORE_INDEX


def _as_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return dict(OmegaConf.to_container(cfg, resolve=True))
    except Exception:
        pass
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {}


def weighted_causal_lm_loss(logits, labels, weights=None, stats: bool = False,
                            ignore_index: int = IGNORE_INDEX):
    """Next-token CE with optional per-token weights.

    logits (B, T, V), labels (B, T) with IGNORE_INDEX for unsupervised.
    weights (B, T) or None. Returns a scalar = sum(w * CE) / sum(w over supervised).
    With weights all 1.0 on supervised tokens this equals the standard mean loss.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    B, Tm1, V = shift_logits.shape
    ce = F.cross_entropy(
        shift_logits.view(-1, V).float(),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(B, Tm1)

    supervised = (shift_labels != ignore_index).float()
    if weights is None:
        w = supervised
    else:
        w = weights[:, 1:].to(ce.dtype) * supervised
    denom = w.sum().clamp_min(1.0)
    loss = (ce * w).sum() / denom
    if not stats:
        return loss

    # plateau 진단용 분해. 가중 손실만 보면 λ 가 흔들려 무엇이 줄었는지 알 수 없다.
    with torch.no_grad():
        n_sup = supervised.sum()
        fact = (w > 1.0).float() * supervised          # 가중이 걸린(=fact) 토큰
        nonfact = supervised - fact
        d = {
            "unweighted_ce": (ce * supervised).sum() / n_sup.clamp_min(1.0),
            "fact_ce": (ce * fact).sum() / fact.sum().clamp_min(1.0),
            "nonfact_ce": (ce * nonfact).sum() / nonfact.sum().clamp_min(1.0),
            "n_supervised": n_sup,
            "n_fact": fact.sum(),
            "fact_token_ratio": fact.sum() / n_sup.clamp_min(1.0),
            "empty_supervision": (supervised.sum(dim=1) == 0).float().mean(),
        }
    return loss, d


class WeightedLmLoss(nn.Module):
    """Prefer: ``WeightedLmLoss(cfg.losses)``.

    ``weighted_causal_lm_loss`` 의 얇은 래퍼 (stage-1 ``LossS1`` 관례).
    """

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        p = _as_dict(cfg)
        p.update(kwargs)
        self.ignore_index = int(p.get("ignore_index", IGNORE_INDEX))
        self.cfg = p

    def forward(self, logits, labels, weights=None, stats: bool = False):
        return weighted_causal_lm_loss(logits, labels, weights, stats=stats,
                                       ignore_index=self.ignore_index)
