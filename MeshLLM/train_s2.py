#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-2 training: mesh-conditioned multimodal LLM (Qwen3-8B + LoRA).

Pipeline (see doc/stage2.md):
  3D mesh displacement --[FROZEN Stage-1 SpiralNet++ encoder]--> mesh tokens
  mesh tokens          --[bridge]--> LLM prefix (soft prompt)
  [mesh prefix ; chat prompt] --[Qwen3-8B + LoRA]--> Answer (causal LM)

bridge 기본은 **mlp**(MlpBridge — 토큰별 선형 사영, LLaVA-1.5 방식)다. qformer
(CrossAttentionFusion, 질문 텍스트까지 cross-attn 으로 융합)는 선택지로 남아 있지만
2026-08-01 진단에서 항목 간 변별력이 붕괴하는 것이 확인돼 기본에서 내려왔다.
근거는 configs/models/stage2_model.yaml 주석과 models/stage2/bridge.py 의 MlpBridge 독스트링.

Only the LoRA adapter, the bridge and the projector are trained.
Stage-1 encoder and Qwen3 base weights are frozen.

Examples:
  # mesh encoder + fusion path only, no LLM download (fast sanity check)
  python tools/smoke_s2.py +experiment=ours_ko datasets.max_records=200
  # 단일 GPU
  python train_s2.py +experiment=ours_ko
  # 여러 GPU (DDP). world 배로 유효 배치가 커지므로 grad_accum 을 그만큼 줄여 맞춘다.
  torchrun --nproc_per_node=2 train_s2.py +experiment=ours_ko trainers.grad_accum=4
"""

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from datasets import find_dataset_def
from losses import find_loss_def
from models import find_model_def
from trainers import find_trainer_def


@hydra.main(version_base=None, config_path="configs", config_name="train_s2")
def main(cfg: DictConfig):
    print("Config:\n" + OmegaConf.to_yaml(cfg))
    if cfg.run.smoke_test:
        from scripts.smoke_s2 import run_smoke
        return run_smoke(cfg)

    # ⚠️ 모델을 만들기 **전에** 시드를 건다. bridge(무작위 초기화)·LoRA A/B·
    # encoder_init=random 인코더가 전부 torch 전역 RNG 를 쓰므로, 이 호출이 뒤로 밀리면
    # 초기 가중치가 run 마다 달라진다. 랭크마다 같은 초기값 → DDP 가 broadcast 로 맞춘다.
    # (TrainerS2.fit 안에도 같은 호출이 있지만 그때는 이미 모델이 만들어진 뒤다)
    torch.manual_seed(cfg.seed)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.models.llm)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    DatasetClass = find_dataset_def(cfg.datasets.name, getattr(cfg.datasets, "class_name", None))
    train_dataset = DatasetClass(cfg.datasets)
    ModelClass = find_model_def(cfg.models.name, cfg.models.class_name)
    model = ModelClass(cfg.models)
    LossClass = find_loss_def(cfg.losses.name, cfg.losses.class_name)
    loss_fn = LossClass(cfg.losses)
    TrainerClass = find_trainer_def(cfg.trainers.name, cfg.trainers.class_name)
    trainer = TrainerClass(cfg.trainers, experiment_cfg=cfg)
    if cfg.run.do_train:
        trainer.fit(model, loss_fn, train_dataset, tokenizer=tokenizer, data_cfg=cfg.datasets)
    return 0


if __name__ == "__main__":
    main()
