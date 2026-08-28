"""Loss v1: SpiralNet++ reconstruction loss (L1).

Official SpiralNet++ training (reconstruction/train_eval.py):
    loss = F.l1_loss(out, x, reduction='mean')

Correspondence/classification demos use NLL after log_softmax, which does not
apply to continuous 11-D muscle targets — so we follow the mesh L1 loss.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn
import torch.nn.functional as F


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


class LossS1(nn.Module):
    """Prefer: ``LossS1(cfg.losses)``."""

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        p = _as_dict(cfg)
        p.update(kwargs)
        self.weight = float(p.get("weight", 1.0))
        self.reduction = p.get("reduction", "mean")
        self.cfg = p

    def forward(self, predictions, targets):
        # Same as SpiralNet++: F.l1_loss(out, x, reduction='mean')
        return self.weight * F.l1_loss(predictions, targets, reduction=self.reduction)
