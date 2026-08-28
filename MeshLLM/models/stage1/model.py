"""Model v1: SpiralNet++ encoder + classification head → 11 muscle activations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from models.stage1.spiralnet.blocks import ClassificationHead, SpiralEncoder
from models.stage1.spiralnet.preprocess import build_spiral_cache, load_spiral_tensors


def _as_dict(cfg: Any) -> dict:
    """OmegaConf DictConfig / Mapping / plain object → dict."""
    if cfg is None:
        return {}
    if hasattr(cfg, "items") and not isinstance(cfg, type):
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(cfg):
                return dict(OmegaConf.to_container(cfg, resolve=True))
        except Exception:
            pass
        return dict(cfg)
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}


class Stage1Model(nn.Module):
    """Displacement mesh (B, V, 3) → muscle activations (B, 11).

    Prefer constructing with the Hydra model config only::

        model = Stage1Model(cfg.models)
    """

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        p = _as_dict(cfg)
        p.update(kwargs)

        template_obj = Path(p.get("template_obj", "DATA/mesh/topology.obj"))
        transform_cache = p.get("transform_cache", None)
        if transform_cache is None:
            transform_cache = template_obj.with_name("spiral_transform.pkl")
        transform_cache = Path(transform_cache)

        in_channels = int(p.get("in_channels", 3))
        out_channels = list(p.get("out_channels", (16, 16)))
        ds_factors = tuple(p.get("ds_factors", (4, 4)))
        seq_lengths = tuple(p.get("seq_lengths", (9, 9)))
        dilations = tuple(p.get("dilations", (1, 1)))
        latent_channels = int(p.get("latent_channels", 32))
        hidden_channels = int(p.get("hidden_channels", 32))
        num_outputs = int(p.get("num_outputs", 11))
        dropout = float(p.get("dropout", 0.5))
        output_activation = p.get("output_activation", "sigmoid")
        device = p.get("device", None)

        if not transform_cache.is_file():
            print(f"[Stage1Model] Building spiral transforms → {transform_cache}")
            build_spiral_cache(
                template_obj,
                ds_factors=ds_factors,
                seq_lengths=seq_lengths,
                dilations=dilations,
                out_path=transform_cache,
            )

        spirals, downs, cache = load_spiral_tensors(transform_cache, device=None)
        self.n_verts = int(cache["n_verts_levels"][0])
        self.transform_cache = transform_cache
        self.cfg = p

        self.encoder = SpiralEncoder(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            spiral_indices=spirals,
            down_transform=downs,
        )
        self.head = ClassificationHead(
            in_channels=latent_channels,
            hidden_channels=hidden_channels,
            num_outputs=num_outputs,
            dropout=dropout,
            activation=output_activation,
        )

        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != 3:
            raise ValueError(f"expected (B, V, 3), got {tuple(x.shape)}")
        if x.size(1) != self.n_verts:
            raise ValueError(f"expected V={self.n_verts}, got {x.size(1)}")
        return self.head(self.encoder(x))

    def predict(self, batch: dict) -> torch.Tensor:
        return self.forward(batch["inputs"])

    def train_step(self, batch: dict, criterion: nn.Module | None = None):
        self.train()
        pred = self.predict(batch)
        target = batch["label"]
        loss = (
            criterion(pred, target)
            if criterion is not None
            else torch.nn.functional.mse_loss(pred, target)
        )
        return {"loss": loss, "predictions": pred, "targets": target}

    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module | None = None):
        self.eval()
        pred = self.predict(batch)
        target = batch["label"]
        loss = (
            criterion(pred, target)
            if criterion is not None
            else torch.nn.functional.mse_loss(pred, target)
        )
        return {"loss": float(loss), "predictions": pred, "targets": target}
