"""Stage-2 model: frozen mesh encoder + mesh→LLM bridge + LoRA-tuned Qwen3.

  3D mesh displacement --[FROZEN Stage-1 SpiralNet++ encoder]--> mesh tokens
  mesh tokens          --[MlpBridge | CrossAttentionFusion]----> LLM prefix
  [mesh prefix ; chat prompt] --[LLM + LoRA]--> Answer (causal LM)

Only the LoRA adapter, the bridge (fusion/projector/query tokens) and the optional
aux muscle head are trained. Stage-1 encoder and LLM base weights are frozen.

Moved verbatim from ``train_s2.py`` (Stage2Model); only `args` → config dict access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.mesh_dataset import MUSCLE_NAMES
from datasets.qa_dataset import IGNORE_INDEX  # noqa: F401  (re-export; splice_prefix 가 쓴다)
from losses.lm_loss import weighted_causal_lm_loss
from models.stage2.bridge import CrossAttentionFusion, MlpBridge, splice_prefix
from models.stage2.mesh_encoder import FrozenMeshEncoder


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


class Stage2Model(nn.Module):
    """Prefer: ``Stage2Model(cfg.models)``."""

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        p = _as_dict(cfg)
        p.update(kwargs)
        lora = _as_dict(p.get("lora", {}))
        aux = _as_dict(p.get("aux", {}))

        # encoder_init: pretrained(=Stage-1 동결) | random(=무작위 동결) | none(=mesh 미사용)
        self.encoder_init = p.get("encoder_init", "pretrained")
        self.encoder_kind = str(p.get("encoder_kind", "mesh"))
        self.use_mesh = self.encoder_init != "none"
        if self.use_mesh:
            stage1_ckpt = p.get("stage1_ckpt", "outputs/stage1/checkpoint_best.pt")
            model_cfg = p.get("model_cfg", "configs/models/stage1_model.yaml")
            ckpt = stage1_ckpt if self.encoder_init == "pretrained" else None
            if self.encoder_kind != "mesh":
                raise ValueError(f"encoder_kind must be 'mesh': {self.encoder_kind!r}")
            self.mesh_encoder = FrozenMeshEncoder(ckpt, model_cfg)
            # 인코더가 배치의 어느 키를 읽는지 (dataset 이 같은 키로 넣어 준다)
            self.input_key = self.mesh_encoder.input_key
            fs = p.get("feature_stats", None)
            if fs:
                self.mesh_encoder.load_feature_stats(fs)
        else:
            self.input_key = "disp"
            print("[stage2] encoder_init=none — text-only row (mesh prefix 없음)")

        dtype = torch.bfloat16 if bool(p.get("bf16", True)) else torch.float32
        self.llm = AutoModelForCausalLM.from_pretrained(
            p.get("llm", "Qwen/Qwen3-8B"), torch_dtype=dtype,
            attn_implementation=p.get("attn_impl", "eager"),
        )
        self.llm.config.use_cache = False
        lora_cfg = LoraConfig(
            r=int(lora.get("r", 16)), lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            bias="none", task_type="CAUSAL_LM",
            target_modules=lora.get(
                "targets",
                "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(","),
        )
        self.llm = get_peft_model(self.llm, lora_cfg)

        llm_hidden = self.llm.config.hidden_size
        self.bridge_kind = p.get("bridge", "qformer")
        if self.use_mesh and self.bridge_kind == "mlp":
            self.fusion = MlpBridge(
                mesh_per_vertex_c=self.mesh_encoder.per_vertex_channels,
                mesh_latent_c=self.mesh_encoder.latent_channels,
                llm_hidden=llm_hidden, hidden=int(p.get("fusion_dim", 512)),
                prefix_norm=bool(p.get("prefix_norm", True)),
                prefix_scale_init=float(p.get("prefix_scale_init", 0.022)),
            )
        elif self.use_mesh:
            self.fusion = CrossAttentionFusion(
                mesh_per_vertex_c=self.mesh_encoder.per_vertex_channels,
                mesh_latent_c=self.mesh_encoder.latent_channels,
                text_dim=llm_hidden, llm_hidden=llm_hidden,
                dim=int(p.get("fusion_dim", 512)), n_query=int(p.get("n_query", 32)),
                layers=int(p.get("fusion_layers", 4)), heads=int(p.get("fusion_heads", 8)),
                prefix_norm=bool(p.get("prefix_norm", True)),
                prefix_scale_init=float(p.get("prefix_scale_init", 0.022)),
            )
            # bridge 는 **FP32 로 유지**한다. 무작위 초기화된 모듈을 순수 bf16 파라미터로
            # 학습하면 작은 갱신이 반올림에 묻혀 초반 정렬이 잘 안 된다.
            # LLM 은 bf16 그대로 두고, prefix 를 넣기 직전에만 캐스팅한다.
            self.fusion.float()

        self.aux_muscle = bool(aux.get("muscle", False)) and self.use_mesh
        if self.aux_muscle:
            self.muscle_head = nn.Sequential(
                nn.Linear(self.mesh_encoder.latent_channels, 64), nn.GELU(),
                nn.Linear(64, len(MUSCLE_NAMES)), nn.Sigmoid(),
            ).float()
        self.aux_weight = float(aux.get("weight", 0.5))
        self.cfg = p

    def _embed_tokens(self, input_ids):
        return self.llm.get_input_embeddings()(input_ids)

    def forward(self, batch):
        llm_dtype = next(self.llm.parameters()).dtype
        mesh_latent = None
        extra = {}

        tok_embeds = self._embed_tokens(batch["input_ids"]).to(llm_dtype)  # (B, L, H)
        labels = batch["labels"]
        attn = batch["attention_mask"]
        weights = batch.get("loss_weights")

        if self.use_mesh:
            # frozen mesh feature (no grad). bridge 는 FP32 로 돈다.
            mesh_vertex, mesh_latent = self.mesh_encoder(batch[self.input_key])
            mesh_vertex = mesh_vertex.float()
            mesh_latent = mesh_latent.float()

            # question text embeddings (frozen embedding lookup, detached)
            with torch.no_grad():
                q_embeds = self._embed_tokens(batch["q_input_ids"]).float()

            prefix = self.fusion(mesh_vertex, mesh_latent, q_embeds,
                                 text_mask=batch["q_attention_mask"],
                                 mesh_valid=batch.get("mesh_valid"))  # (B, N_q, H) fp32
            with torch.no_grad():   # prefix 가 token embedding 과 스케일이 맞는지 감시
                p_rms = prefix.float().pow(2).mean().sqrt()
                t_rms = tok_embeds.float().pow(2).mean().sqrt()
                extra = {"prefix_rms": p_rms, "token_embed_rms": t_rms,
                         "prefix_to_token_rms": p_rms / t_rms.clamp_min(1e-6)}
            prefix = prefix.to(llm_dtype)      # LLM 에 넣기 직전에만 캐스팅

        if not self.use_mesh:
            # text-only row: prefix 없이 대화만 넣는다
            out = self.llm(inputs_embeds=tok_embeds, attention_mask=attn)
            lm_loss, st = weighted_causal_lm_loss(out.logits, labels, weights, stats=True)
            return {"loss": lm_loss, "lm_loss": lm_loss.detach(), "aux_loss": None, **st}

        inputs_embeds, attn, labels, weights = splice_prefix(
            prefix, tok_embeds, attn, labels, weights, batch.get("mesh_at"))

        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn)
        lm_loss, st = weighted_causal_lm_loss(out.logits, labels, weights, stats=True)

        loss = lm_loss
        aux = None
        if self.aux_muscle:
            # 여러 mesh 를 참조하는 레코드에서는 첫 mesh 의 라벨만 감독한다
            lat = mesh_latent[:, 0] if mesh_latent.dim() == 3 else mesh_latent
            pred = self.muscle_head(lat)
            aux = F.mse_loss(pred.float(), batch["muscle"].to(pred.device).float())
            loss = loss + self.aux_weight * aux
        return {"loss": loss, "lm_loss": lm_loss.detach(),
                "aux_loss": (aux.detach() if aux is not None else None), **st, **extra}

    @torch.no_grad()
    def generate_answer(self, batch, **gen_kwargs):
        """평가용 생성. batch 의 input_ids 는 **정답 없이** generation prompt 까지만 담겨야 한다.

        mesh prefix 를 임베딩 앞에 붙이고 inputs_embeds 로 생성한다. 반환은 새로 생성된
        토큰 id (B, T_new) — prompt 는 포함되지 않는다.
        """
        self.eval()
        dtype = next(self.llm.parameters()).dtype
        tok_embeds = self._embed_tokens(batch["input_ids"]).to(dtype)
        attn = batch["attention_mask"]

        if self.use_mesh:
            # forward() 와 같은 규약: bridge 는 FP32 로 돌리고 prefix 만 LLM dtype 으로 캐스팅한다.
            # (bridge dtype 으로 tok_embeds 를 올리면 LLM 이 bf16 이라 matmul dtype 이 어긋난다)
            mesh_vertex, mesh_latent = self.mesh_encoder(batch[self.input_key])
            prefix = self.fusion(mesh_vertex.float(), mesh_latent.float(),
                                 self._embed_tokens(batch["q_input_ids"]).float(),
                                 text_mask=batch["q_attention_mask"],
                                 mesh_valid=batch.get("mesh_valid"))
            tok_embeds, attn, _, _ = splice_prefix(prefix, tok_embeds, attn,
                                                   mesh_at=batch.get("mesh_at"))

        return self.llm.generate(inputs_embeds=tok_embeds, attention_mask=attn, **gen_kwargs)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def save(self, output_dir: str | Path):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.llm.save_pretrained(out / "lora")
        bridge = {"fusion": self.fusion.state_dict()} if self.use_mesh else {}
        if self.aux_muscle:
            bridge["muscle_head"] = self.muscle_head.state_dict()
        torch.save(bridge, out / "mm_projector.pt")
