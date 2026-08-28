"""Stage-1 evaluation entrypoint (3D displacement → muscle activations).

Usage:
  python test_s1.py
  python test_s1.py checkpoint=outputs/stage1/checkpoint_best.pt
  python test_s1.py run.split=val
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from datasets import find_dataset_def
from losses import find_loss_def
from models import find_model_def
from trainers import find_trainer_def


def _load_checkpoint(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Train first with: python train_s1.py"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    meta = {
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "global_step": ckpt.get("global_step") if isinstance(ckpt, dict) else None,
        "path": str(ckpt_path),
    }
    return meta


@torch.no_grad()
def _collect_predictions(model, loader, device):
    model.eval()
    preds, labels, indices = [], [], []
    for batch in loader:
        inputs = batch["inputs"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        pred = model(inputs)
        preds.append(pred.cpu())
        labels.append(target.cpu())
        if "index" in batch and torch.is_tensor(batch["index"]):
            indices.append(batch["index"].cpu())
    out = {
        "predictions": torch.cat(preds, dim=0),
        "labels": torch.cat(labels, dim=0),
    }
    if indices:
        out["index"] = torch.cat(indices, dim=0)
    return out


@hydra.main(version_base=None, config_path="configs", config_name="test_s1")
def main(cfg: DictConfig):
    print("Config:\n" + OmegaConf.to_yaml(cfg))

    DatasetClass = find_dataset_def(
        cfg.datasets.name, getattr(cfg.datasets, "class_name", None)
    )
    ds_cfg = OmegaConf.to_container(cfg.datasets, resolve=True)
    ds_cfg["split"] = cfg.run.split
    dataset = DatasetClass(ds_cfg)

    ModelClass = find_model_def(cfg.models.name, cfg.models.class_name)
    model = ModelClass(cfg.models)

    LossClass = find_loss_def(cfg.losses.name, cfg.losses.class_name)
    loss_fn = LossClass(cfg.losses)

    TrainerClass = find_trainer_def(cfg.trainers.name, cfg.trainers.class_name)
    trainer_cfg = OmegaConf.to_container(cfg.trainers, resolve=True)
    wandb_cfg = dict(trainer_cfg.get("wandb") or {})
    if "enabled" not in wandb_cfg:
        wandb_cfg["enabled"] = False
    trainer_cfg["wandb"] = wandb_cfg
    trainer = TrainerClass(trainer_cfg, experiment_cfg=cfg)

    ckpt_path = Path(cfg.checkpoint)
    meta = _load_checkpoint(model, ckpt_path, trainer.device)
    model = model.to(trainer.device)
    loss_fn = loss_fn.to(trainer.device)

    print(
        f"[Stage-1] split={cfg.run.split}  n={len(dataset)}  "
        f"model={type(model).__name__}  loss={type(loss_fn).__name__}\n"
        f"checkpoint={meta['path']}  epoch={meta['epoch']}  step={meta['global_step']}"
    )

    test_loss = trainer.evaluate(model, loss_fn, dataset)
    print(f"test_loss (L1)={test_loss:.6f}")

    loader = trainer._build_loader(dataset, trainer.eval_batch_size, shuffle=False)
    packed = _collect_predictions(model, loader, trainer.device)
    per_muscle = (packed["predictions"] - packed["labels"]).abs().mean(dim=0)
    names = dataset[0]["muscle_names"] if len(dataset) > 0 else [f"m{i}" for i in range(11)]
    print("per-muscle L1:")
    for name, v in zip(names, per_muscle.tolist()):
        print(f"  {name:6s}  {v:.6f}")

    if cfg.run.save_predictions:
        pred_path = Path(cfg.run.pred_path)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": meta,
            "split": cfg.run.split,
            "loss": float(test_loss),
            "per_muscle_l1": {n: float(v) for n, v in zip(names, per_muscle.tolist())},
            **packed,
        }
        torch.save(payload, pred_path)
        print(f"saved predictions → {pred_path}")

    return 0


if __name__ == "__main__":
    main()
