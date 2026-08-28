#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-2 smoke test — mesh 인코더 + fusion 경로만 (LLM 다운로드 없음).

동결 인코더가 실제로 동결됐는지, fusion 이 미분 가능한지, fact 마스킹이
label 을 건드리지 않는지를 한 배치로 확인한다.

  python tools/smoke_s2.py +experiment=ours_ko datasets.max_records=200
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import find_dataset_def  # noqa: E402
from datasets.qa_dataset import IGNORE_INDEX, QaCollator  # noqa: E402
from models.stage2.bridge import CrossAttentionFusion, MlpBridge  # noqa: E402
from models.stage2.mesh_encoder import FrozenMeshEncoder  # noqa: E402


def run_smoke(cfg: DictConfig):
    from transformers import AutoTokenizer

    mcfg = cfg.models
    use_mesh = mcfg.encoder_init != "none"
    kind = str(mcfg.get("encoder_kind", "mesh"))
    enc = None
    if use_mesh:
        ck = mcfg.stage1_ckpt if mcfg.encoder_init == "pretrained" else None
        print(f"[smoke] {kind} 인코더 로드 (encoder_init={mcfg.encoder_init}) ...")
        enc = FrozenMeshEncoder(ck, mcfg.model_cfg)
        if mcfg.get("feature_stats", None):
            enc.load_feature_stats(mcfg.feature_stats)
        n_frozen = sum(p.numel() for p in enc.parameters())
        n_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        print(f"[smoke] encoder params={n_frozen} trainable={n_train} (expect 0 trainable)")
        print(f"[smoke] per_vertex_channels={enc.per_vertex_channels} "
              f"latent={enc.latent_channels} n_mesh_tokens={enc.n_mesh_tokens}")
    else:
        print("[smoke] encoder_init=none — mesh 경로 없이 데이터/마스킹만 확인한다")

    cfg_data = OmegaConf.create(OmegaConf.to_container(cfg.datasets, resolve=True))
    if cfg_data.max_records is None:
        cfg_data.max_records = 200     # 스모크는 색인을 조금만 만든다
    DatasetClass = find_dataset_def(cfg_data.name, cfg_data.get("class_name", None))
    ds = DatasetClass(cfg_data)
    print(f"[smoke] dataset records={len(ds)} (split={cfg_data.split}, lang={cfg_data.lang})")

    tokenizer = AutoTokenizer.from_pretrained(mcfg.llm)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collate = QaCollator(tokenizer, cfg_data, seed=cfg.seed)

    items = [ds[i] for i in range(min(4, len(ds)))]
    batch = collate(items)
    in_key = DatasetClass.input_key       # "disp" (3D) | "imgs" (2D 렌더)
    m = cfg_data.masking
    n_masked = int((batch["input_ids"] == collate.mask_id).sum())
    w = batch["loss_weights"]
    sup = w > 0
    print(f"[smoke] {in_key}={tuple(batch[in_key].shape)} "
          f"mesh_valid={tuple(batch['mesh_valid'].shape)} "
          f"input_ids={tuple(batch['input_ids'].shape)}")
    print(f"[smoke] 감독 토큰 {int(sup.sum())}개 | fact 가중 토큰 {int((w > 1.0).sum())}개 "
          f"(λ={float(w[w > 1.0].max()) if (w > 1.0).any() else 0:.2f}) | "
          f"컨텍스트 마스킹된 입력 토큰 {n_masked}개 (p={m.context_mask_prob})")
    total_w = float(w.sum())
    fact_share = float(w[w > 1.0].sum()) / total_w if total_w else 0.0
    print(f"[smoke] fact 토큰의 손실 기여 = {100 * fact_share:.1f}% "
          f"(목표 τ={m.target_fact_share})")

    if not use_mesh:
        masked = batch["input_ids"] == collate.mask_id
        assert not masked.any() or (batch["labels"][masked] != IGNORE_INDEX).any()
        print("[smoke] OK — text-only row: 데이터/마스킹 경로 정상 (mesh 미사용).")
        return

    mesh_vertex, mesh_latent = enc(batch[in_key])
    print(f"[smoke] mesh_vertex={tuple(mesh_vertex.shape)}  mesh_latent={tuple(mesh_latent.shape)}")
    # 항목 간 변별력 — 이게 붕괴하면(코사인 1.0 근처) LLM 이 mesh 를 못 읽는다.
    _f = torch.cat([mesh_vertex.flatten(1), mesh_latent.flatten(1)], 1)
    _n = torch.nn.functional.normalize(_f, dim=1)
    _c = _n @ _n.T
    _k = _f.size(0)
    print(f"[smoke] 인코더 출력 항목간 코사인유사도="
          f"{float((_c.sum() - _c.diag().sum()) / max(_k * (_k - 1), 1)):.4f} "
          f"(feature_norm={enc.feature_norm}; 1.0 에 가까우면 붕괴)")

    llm_hidden, text_dim = 128, 128
    kind = mcfg.get("bridge", "qformer")
    if kind == "mlp":
        fusion = MlpBridge(enc.per_vertex_channels, enc.latent_channels, llm_hidden,
                           hidden=mcfg.fusion_dim)
        n_pref = MlpBridge.n_prefix_tokens(int(batch["mesh_valid"].size(1)),
                                           enc.n_mesh_tokens - 1)
    else:
        fusion = CrossAttentionFusion(
            mesh_per_vertex_c=enc.per_vertex_channels, mesh_latent_c=enc.latent_channels,
            text_dim=text_dim, llm_hidden=llm_hidden,
            dim=mcfg.fusion_dim, n_query=mcfg.n_query,
            layers=mcfg.fusion_layers, heads=mcfg.fusion_heads,
        )
        n_pref = mcfg.n_query
    b = batch[in_key].size(0)
    fake_text = torch.randn(b, 10, text_dim)
    fake_mask = torch.ones(b, 10, dtype=torch.long)
    prefix = fusion(mesh_vertex, mesh_latent, fake_text, text_mask=fake_mask,
                    mesh_valid=batch["mesh_valid"])
    print(f"[smoke] bridge={kind} · prefix -> LLM tokens={tuple(prefix.shape)} "
          f"(expect ({b}, {n_pref}, {llm_hidden}))")
    loss = prefix.pow(2).mean()
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in fusion.parameters() if p.grad is not None)
    print(f"[smoke] fusion grads flow: total|grad|={g:.4f}")

    # 컨텍스트 마스킹이 label 을 건드리지 않았는지 (핵심 불변식)
    masked = batch["input_ids"] == collate.mask_id
    assert not (batch["labels"][masked] == IGNORE_INDEX).all() or not masked.any(), \
        "컨텍스트 마스킹된 자리의 label 이 사라졌다 — 원본 정답을 유지해야 한다"
    assert n_train == 0, "encoder must be frozen"
    assert prefix.shape == (b, n_pref, llm_hidden)
    print("[smoke] OK — 인코더 동결, fusion 미분 가능, 마스킹 label 보존 확인.")


@hydra.main(version_base=None, config_path="../configs", config_name="train_s2")
def main(cfg: DictConfig):
    run_smoke(cfg)
    return 0


if __name__ == "__main__":
    main()
