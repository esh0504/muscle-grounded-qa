"""Trainer v1: Adam + L1 training loop for mesh → muscle models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def _as_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    try:
        if OmegaConf.is_config(cfg):
            return dict(OmegaConf.to_container(cfg, resolve=True))
    except Exception:
        pass
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {}


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _target_key(model) -> str:
    """손실의 타깃이 배치의 어느 키인가.

    기본은 `label` (근육 활성 11개, Stage1Model). mesh→mesh AE 처럼 입력 자신을 복원하는
    모델은 클래스에 `target_key = "inputs"` 를 선언한다.
    """
    return getattr(model, "target_key", "label")


class TrainerS1:
    """Prefer: ``TrainerS1(cfg.trainers)`` then ``trainer.fit(...)``.

    SpiralNet++-style defaults: Adam, lr=1e-3, StepLR decay 0.99 / epoch.
    Optional Weights & Biases logging via ``wandb.enabled: true``.
    """

    def __init__(self, cfg=None, experiment_cfg=None, **kwargs):
        p = _as_dict(cfg)
        p.update(kwargs)
        self.cfg = p
        self.experiment_cfg = _as_dict(experiment_cfg) if experiment_cfg is not None else None

        self.output_dir = Path(p.get("output_dir", "outputs/trainer_s1"))
        self.batch_size = int(p.get("batch_size", 32))
        self.eval_batch_size = int(p.get("eval_batch_size", self.batch_size))
        self.num_workers = int(p.get("num_workers", 0))
        self.epochs = int(p.get("epochs", 1))
        self.max_steps = p.get("max_steps", None)
        if self.max_steps is not None:
            self.max_steps = int(self.max_steps)

        self.optimizer_name = str(p.get("optimizer", "Adam"))
        self.lr = float(p.get("lr", 1e-3))
        self.weight_decay = float(p.get("weight_decay", 0.0))
        self.lr_decay = float(p.get("lr_decay", 0.99))
        self.decay_step = int(p.get("decay_step", 1))

        self.log_every = int(p.get("log_every", 50))
        self.eval_every_epochs = int(p.get("eval_every_epochs", 1))
        self.save_every_epochs = int(p.get("save_every_epochs", 1))
        self.seed = int(p.get("seed", 42))
        device = p.get("device", None)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # W&B
        wandb_cfg = p.get("wandb", {}) or {}
        if not isinstance(wandb_cfg, dict):
            wandb_cfg = _as_dict(wandb_cfg)
        self.wandb_enabled = bool(wandb_cfg.get("enabled", False))
        self.wandb_project = wandb_cfg.get("project", "tongue-muscle")
        self.wandb_entity = wandb_cfg.get("entity", None)
        self.wandb_run_name = wandb_cfg.get("name", None)
        self.wandb_mode = wandb_cfg.get("mode", "online")  # online | offline | disabled
        self.wandb_tags = list(wandb_cfg.get("tags", []) or [])
        self.wandb_notes = wandb_cfg.get("notes", None)
        self.wandb_log_model = bool(wandb_cfg.get("log_model", False))
        self._wandb = None
        self._wandb_run = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # W&B helpers
    # ------------------------------------------------------------------
    def _wandb_init(self, model: torch.nn.Module, train_dataset, val_dataset):
        if not self.wandb_enabled or self.wandb_mode == "disabled":
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb is not installed. Run: pip install wandb"
            ) from e

        self._wandb = wandb
        config = {
            "trainer": self.cfg,
            "n_train": len(train_dataset),
            "n_val": 0 if val_dataset is None else len(val_dataset),
            "device": str(self.device),
            "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        if self.experiment_cfg is not None:
            config["experiment"] = self.experiment_cfg

        try:
            self._wandb_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                name=self.wandb_run_name,
                mode=self.wandb_mode,
                tags=self.wandb_tags or None,
                notes=self.wandb_notes,
                config=config,
                dir=str(self.output_dir),
                reinit=True,
            )
            wandb.watch(model, log="gradients", log_freq=max(self.log_every, 100))
        except Exception as e:  # 로그인/네트워크 문제 — 기록만 끄고 학습은 계속한다
            print(f"[TrainerS1] wandb init 실패 ({e}) — 로깅 없이 계속합니다. "
                  "wandb 를 쓰려면 `wandb login` 후 trainers.wandb.mode=online.")
            self._wandb = None
            self._wandb_run = None

    def _wandb_log(self, metrics: dict, step: Optional[int] = None):
        if self._wandb_run is None:
            return
        self._wandb.log(metrics, step=step)

    def _wandb_finish(self):
        if self._wandb_run is None:
            return
        if self.wandb_log_model:
            # best checkpoint as artifact (optional)
            best = self.output_dir / "checkpoint_best.pt"
            if best.is_file():
                art = self._wandb.Artifact("model-best", type="model")
                art.add_file(str(best))
                self._wandb_run.log_artifact(art)
        self._wandb.finish()
        self._wandb_run = None

    # ------------------------------------------------------------------
    # optim / loaders
    # ------------------------------------------------------------------
    def _build_loader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        collate = getattr(dataset, "collate_fn", None)
        if collate is None and hasattr(dataset, "__class__"):
            collate = getattr(dataset.__class__, "collate_fn", None)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate,
        )

    def _build_optimizer(self, model: torch.nn.Module):
        params = [p for p in model.parameters() if p.requires_grad]
        if self.optimizer_name.lower() == "adam":
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer_name.lower() == "adamw":
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def _build_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.decay_step, gamma=self.lr_decay
        )

    # ------------------------------------------------------------------
    # train / eval
    # ------------------------------------------------------------------
    def fit(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        train_dataset,
        val_dataset=None,
    ) -> dict:
        torch.manual_seed(self.seed)
        model = model.to(self.device)
        loss_fn = loss_fn.to(self.device)

        train_loader = self._build_loader(train_dataset, self.batch_size, shuffle=True)
        val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_loader = self._build_loader(
                val_dataset, self.eval_batch_size, shuffle=False
            )

        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)

        self._wandb_init(model, train_dataset, val_dataset)

        history = {"train_loss": [], "val_loss": []}
        global_step = 0
        best_val = float("inf")

        print(
            f"[TrainerS1] device={self.device}  epochs={self.epochs}  "
            f"batch_size={self.batch_size}  lr={self.lr}  "
            f"train={len(train_dataset)}  val={0 if val_dataset is None else len(val_dataset)}  "
            f"wandb={self.wandb_enabled}"
        )

        try:
            for epoch in range(1, self.epochs + 1):
                t0 = time.time()
                train_loss, global_step, stop = self._train_epoch(
                    model, loss_fn, train_loader, optimizer, epoch, global_step
                )
                history["train_loss"].append(train_loss)
                lr = scheduler.get_last_lr()[0]
                msg = f"epoch {epoch}/{self.epochs}  train_loss={train_loss:.6f}"

                metrics = {
                    "epoch": epoch,
                    "train/loss_epoch": train_loss,
                    "train/lr": lr,
                    "time/epoch_sec": time.time() - t0,
                }

                val_loss = None
                if val_loader is not None and (epoch % self.eval_every_epochs == 0):
                    val_loss = self.evaluate(model, loss_fn, val_loader)
                    history["val_loss"].append(val_loss)
                    metrics["val/loss"] = val_loss
                    msg += f"  val_loss={val_loss:.6f}"
                    if val_loss < best_val:
                        best_val = val_loss
                        metrics["val/best_loss"] = best_val
                        self._save_checkpoint(
                            model, optimizer, epoch, global_step, tag="best"
                        )

                scheduler.step()
                msg += f"  lr={lr:.2e}  time={time.time() - t0:.1f}s"
                print(msg)
                self._wandb_log(metrics, step=global_step)

                if epoch % self.save_every_epochs == 0:
                    self._save_checkpoint(
                        model, optimizer, epoch, global_step, tag=f"epoch{epoch}"
                    )

                if stop:
                    print(f"[TrainerS1] reached max_steps={self.max_steps}, stopping.")
                    break

            self._save_checkpoint(model, optimizer, epoch, global_step, tag="last")
            # val 이 한 번도 안 돌았어도 (val 셋 없음 등) checkpoint_best 는 항상 남긴다 —
            # Stage-2 가 이 파일을 기본 경로로 읽는다.
            if not (self.output_dir / "checkpoint_best.pt").is_file():
                self._save_checkpoint(model, optimizer, epoch, global_step, tag="best")
        finally:
            self._wandb_finish()

        return history

    def _train_epoch(
        self,
        model,
        loss_fn,
        loader,
        optimizer,
        epoch: int,
        global_step: int,
    ):
        model.train()
        total = 0.0
        n = 0
        stop = False

        for batch_idx, batch in enumerate(loader, start=1):
            batch = _move_batch(batch, self.device)
            optimizer.zero_grad(set_to_none=True)

            pred = model(batch["inputs"])
            loss = loss_fn(pred, batch[_target_key(model)])
            loss.backward()
            optimizer.step()

            bs = batch["inputs"].size(0)
            total += float(loss.item()) * bs
            n += bs
            global_step += 1
            loss_val = float(loss.item())

            if batch_idx % self.log_every == 0 or batch_idx == 1:
                print(
                    f"  epoch {epoch}  step {batch_idx}/{len(loader)}  "
                    f"loss={loss_val:.6f}"
                )
                self._wandb_log(
                    {
                        "train/loss_step": loss_val,
                        "train/epoch": epoch,
                        "train/batch": batch_idx,
                    },
                    step=global_step,
                )

            if self.max_steps is not None and global_step >= self.max_steps:
                stop = True
                break

        return total / max(n, 1), global_step, stop

    @torch.no_grad()
    def evaluate(self, model, loss_fn, loader_or_dataset) -> float:
        model.eval()
        if isinstance(loader_or_dataset, DataLoader):
            loader = loader_or_dataset
        else:
            loader = self._build_loader(
                loader_or_dataset, self.eval_batch_size, shuffle=False
            )

        total = 0.0
        n = 0
        for batch in loader:
            batch = _move_batch(batch, self.device)
            pred = model(batch["inputs"])
            loss = loss_fn(pred, batch[_target_key(model)])
            bs = batch["inputs"].size(0)
            total += float(loss.item()) * bs
            n += bs
        return total / max(n, 1)

    def _save_checkpoint(self, model, optimizer, epoch, global_step, tag: str):
        path = self.output_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "cfg": self.cfg,
            },
            path,
        )
        print(f"  saved {path}")
