"""Trainer s2: Stage-2 학습 루프 (DDP + grad accumulation + 주기적 생성 검증).

원본 ``train_s2.py`` 의 ``build_val`` / ``validate`` / ``_grad_norm`` / ``dist_setup`` /
``train`` 을 그대로 옮긴 것이다. 손실은 ``Stage2Model.forward`` 가 내부에서 계산하므로
``loss_fn`` 은 stage-1 트레이너와 시그니처를 맞추기 위한 자리다 (모델이 이미 들고 있다).

  TrainerS2(cfg.trainers, experiment_cfg=cfg).fit(
      model, loss_fn, train_dataset, tokenizer=tok, data_cfg=cfg.datasets)
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Dataset / collate — datasets/qa_dataset.py (설정: configs/datasets/qa_dataset.yaml)
#
# fact 마스킹(손실 가중 + 이전 턴 컨텍스트 마스킹)은 QA 레코드에 이미 붙어 있는
# `mask_spans` 를 쓴다. 예전의 한국어 정규식 휴리스틱은 자연화된 표현과 en 데이터를
# 놓쳐서 걷어냈다. 비율은 전부 config 에서 조정한다.
#
# 검증 세트는 **학습에 쓴 것과 같은 클래스**로 만든다 (2D 렌더 row 는 RenderQaV3).
# MeshQaDataset 는 그 클래스를 못 받았을 때의 기본값일 뿐이다.
from datasets.qa_dataset import MeshQaDataset, QaCollator, pad_inputs
from metrics.stage2_val import score_generations
from utils import move


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


def grad_norm(params) -> float:
    """clip 전 gradient L2 norm. bridge 와 LoRA 중 어디가 죽었는지 보려는 용도."""
    tot = 0.0
    for p in params:
        if p.grad is not None:
            tot += float(p.grad.detach().float().pow(2).sum())
    return tot ** 0.5


def dist_setup():
    """torchrun 으로 띄웠으면 process group 을 연다.

    반환 (rank, local_rank, world_size, is_dist). torchrun 없이 실행하면 단일 GPU 경로 그대로.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return 0, 0, 1, False
    import torch.distributed as dist
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), local_rank, world, True


class TrainerS2:
    """Prefer: ``TrainerS2(cfg.trainers, experiment_cfg=cfg)`` then ``.fit(...)``."""

    def __init__(self, cfg=None, experiment_cfg=None, **kwargs):
        p = _as_dict(cfg)
        p.update(kwargs)
        self.cfg = p
        self.experiment_cfg = _as_dict(experiment_cfg) if experiment_cfg is not None else None

        self.output_dir = Path(p.get("output_dir", "outputs/stage2"))
        self.epochs = int(p.get("epochs", 3))
        self.batch_size = int(p.get("batch_size", 2))
        self.grad_accum = int(p.get("grad_accum", 8))
        self.lr = float(p.get("lr", 2e-4))
        lr_bridge = p.get("lr_bridge", 5e-4)
        self.lr_bridge = None if lr_bridge is None else float(lr_bridge)
        self.weight_decay = float(p.get("weight_decay", 0.0))
        self.bridge_weight_decay = float(p.get("bridge_weight_decay", 1e-2))
        self.warmup_ratio = float(p.get("warmup_ratio", 0.05))
        self.max_grad_norm = float(p.get("max_grad_norm", 1.0))
        self.num_workers = int(p.get("num_workers", 2))
        self.log_every = int(p.get("log_every", 1))
        self.save_every = int(p.get("save_every", 0))
        self.seed = int(p.get("seed", 42))

        # validation — 일정 optimizer step 마다 (rank 0 에서만, 나머지는 barrier 대기)
        v = _as_dict(p.get("val", {}) or {})
        self.val_every = int(v.get("every", 0))              # 0 이면 끔
        self.val_split = str(v.get("split", "val"))
        self.val_max_records = int(v.get("max_records", 512))    # teacher-forced 손실용
        self.val_gen_records = int(v.get("gen_records", 64))     # 생성용 (0 이면 생략)
        self.val_max_new_tokens = int(v.get("max_new_tokens", 256))
        # 기권이 정답인 turn_type (쉼표 구분)
        self.abstain_turn_types = str(v.get("abstain_turn_types", "D1"))

        # wandb (rank 0 에서만 기록)
        w = _as_dict(p.get("wandb", {}) or {})
        self.wandb_enabled = bool(w.get("enabled", False))
        self.wandb_project = w.get("project", "tongue-muscle")
        self.wandb_name = w.get("name", None)          # 생략 시 실험 이름(output_dir 끝)
        self.wandb_mode = w.get("mode", "online")      # online | offline | disabled
        self.wandb_tags = w.get("tags", "stage2")      # 쉼표 구분

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def _build_val(self, tokenizer, cfg, val_dataset=None, dataset_cls=None):
        """검증용 dataset/collator.

        학습과 두 곳이 다르다.
        - **컨텍스트 마스킹 off** — 추론 시 이전 턴은 온전한 텍스트다. 검증은 추론 조건을 흉내내야 한다.
        - **span 전체 선택(r=1.0)** — 매번 다른 span 을 고르면 가중 손실이 스텝마다 흔들려 비교가 안 된다.
        unweighted CE 는 span 선택과 무관하므로 주 비교 지표로 쓴다.
        """
        vcfg = OmegaConf.merge(cfg, {
            "split": self.val_split,
            "max_records": int(self.val_max_records),
            "masking": {"span_select_ratio": 1.0, "context_mask_prob": 0.0},
        })
        ds = (val_dataset if val_dataset is not None
              else (dataset_cls or MeshQaDataset)(vcfg))
        col = QaCollator(tokenizer, vcfg, seed=self.seed)
        return ds, col, vcfg

    @torch.no_grad()
    def _validate(self, model, tokenizer, val_ds, val_col, device, lang="ko",
                  q_max_len: int = 256):
        """teacher-forced 손실 + 생성 기반 fact 지표.

        DDP 에서는 **rank 0 만** 호출하고 나머지는 barrier 로 기다린다 (원본 모델 사용, DDP 래퍼 아님).
        """
        was_training = model.training
        model.eval()
        out = {}

        # ---- 1) teacher-forced 손실 -------------------------------------------
        n = min(len(val_ds), int(self.val_max_records))
        loader = DataLoader([val_ds[i] for i in range(n)], batch_size=self.batch_size,
                            shuffle=False, collate_fn=val_col)
        tot_w = tot_u = tot_f = tot_nf = 0.0
        nb = 0
        for batch in loader:
            o = model(move(batch, device))
            tot_w += float(o["loss"]); tot_u += float(o["unweighted_ce"])
            tot_f += float(o["fact_ce"]); tot_nf += float(o["nonfact_ce"])
            nb += 1
        if nb:
            out.update({"val/weighted_loss": tot_w / nb, "val/unweighted_ce": tot_u / nb,
                        "val/fact_ce": tot_f / nb, "val/nonfact_ce": tot_nf / nb,
                        "val/n_loss_samples": n})

        # ---- 2) 생성 기반 지표 (고정 부분집합) ---------------------------------
        g = int(self.val_gen_records)
        if g > 0:
            # turn_type 층화. 앞에서 g 개를 그대로 자르면 안 된다 — 인덱스가 파일 round-robin
            # 이라 val_ds[0:32] 는 전부 A1 turn-0 이고 D1 이 한 건도 없다. 그러면
            # abstention_f1 이 prf(0,0,0)=0.0 인 축퇴 케이스가 되어 "기권을 못 한다"로 오독된다
            # (실측: 이 버그로 val/abstention_f1 이 항상 0.0 이었다).
            pool = min(len(val_ds), max(g * 12, int(self.val_max_records)))
            by_tt: dict[str, list[int]] = {}
            for i in range(pool):
                it = val_ds[i]
                k = it.get("target_turn")
                tts = it.get("turn_types") or []
                by_tt.setdefault(tts[k] if k is not None and k < len(tts) else "", []).append(i)
            picks, r = [], 0
            while len(picks) < g and any(len(v) > r for v in by_tt.values()):
                for tt in sorted(by_tt):                    # 라운드로빈으로 골고루
                    if len(by_tt[tt]) > r and len(picks) < g:
                        picks.append(by_tt[tt][r])
                r += 1
            items = [val_ds[i] for i in picks]
            preds, golds, abst = [], [], []
            bs = max(1, self.batch_size)
            tokenizer.padding_side = "left"      # 배치 생성은 왼쪽 패딩
            for st in range(0, len(items), bs):
                chunk = items[st:st + bs]
                prompts, q_txt = [], []
                for it in chunk:
                    msgs = it["messages"][:-1]   # 대상 assistant 답변을 떼고 generation prompt 로
                    try:
                        t = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                          add_generation_prompt=True,
                                                          enable_thinking=False)
                    except TypeError:
                        t = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                          add_generation_prompt=True)
                    prompts.append(t); q_txt.append(it["question"])
                enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
                q = tokenizer(q_txt, return_tensors="pt", padding=True, truncation=True,
                              max_length=q_max_len)
                key = chunk[0].get("input_key", "disp")
                inputs, valid = pad_inputs(chunk, key)
                batch = {"input_ids": enc["input_ids"].to(device),
                         "attention_mask": enc["attention_mask"].to(device),
                         "q_input_ids": q["input_ids"].to(device),
                         "q_attention_mask": q["attention_mask"].to(device),
                         key: inputs.to(device), "mesh_valid": valid.to(device)}
                seqs = model.generate_answer(batch, max_new_tokens=int(self.val_max_new_tokens),
                                             do_sample=False, num_beams=1,
                                             pad_token_id=tokenizer.pad_token_id)
                for it, s in zip(chunk, seqs):
                    preds.append(tokenizer.decode(s, skip_special_tokens=True).strip())
                    spans = it["answer_spans"][-1] if it["answer_spans"] else []
                    golds.append(spans)
                    tt = it["turn_types"]; k = it["target_turn"]
                    abst.append((tt[k] if k is not None and k < len(tt) else "") in
                                set(self.abstain_turn_types.split(",")))
            tts_of = [(it["turn_types"][it["target_turn"]]
                       if it.get("target_turn") is not None
                       and it["target_turn"] < len(it.get("turn_types") or []) else "")
                      for it in items]
            out.update(score_generations(preds, golds, tts_of, abst, lang))

        if was_training:
            model.train()
        return out

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------
    def _save(self, model, path):
        """``Stage2Model.save`` 래퍼 — DDP 래퍼가 아니라 원본 모듈에서 저장한다."""
        net = model.module if hasattr(model, "module") else model
        net.save(path)

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------
    def fit(self, model, loss_fn, train_dataset, val_dataset=None, *,
            tokenizer=None, data_cfg=None):
        rank, local_rank, world, is_dist = dist_setup()
        is_main = rank == 0
        log = print if is_main else (lambda *a, **k: None)

        torch.manual_seed(self.seed)   # 랭크마다 같은 초기값 → DDP 가 broadcast 로 맞춘다
        device = (torch.device(f"cuda:{local_rank}") if torch.cuda.is_available()
                  else torch.device("cpu"))

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        mp = _as_dict(getattr(model, "cfg", None))
        cfg = data_cfg
        dcfg = _as_dict(cfg)
        log(f"[stage2] 데이터 색인 중: {dcfg.get('qa_glob')} (split={dcfg.get('split')})")
        # Hydra 오버라이드까지 반영된 최종 데이터 설정을 남긴다 — 이 파일 하나로 데이터
        # 구성이 재현된다. (DDP 에서는 rank 0 만 쓴다. 모든 랭크가 같은 파일에 동시에
        # 쓰면 지저분해진다.) configs/train_s2.yaml 이 hydra.output_subdir=null 이라
        # .hydra/config.yaml 이 안 남으므로 이 파일이 유일한 학습 시작 시점 기록이다.
        if is_main:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            OmegaConf.save(OmegaConf.create(dcfg),
                           Path(self.output_dir) / "resolved_data_cfg.yaml")
        dataset = train_dataset
        # 랭크마다 collator 시드를 달리한다 — 같은 시드면 span 선택·컨텍스트 마스킹 난수가
        # 모든 랭크에서 같은 순서로 돌아 다양성이 준다.
        collate = QaCollator(tokenizer, cfg, seed=self.seed + rank)
        m = _as_dict(dcfg.get("masking", {}) or {})
        log(f"[stage2] records={len(dataset):,} | 마스킹: span={m.get('span_source')} "
            f"r={m.get('span_select_ratio')} τ={m.get('target_fact_share')} "
            f"p_ctx={m.get('context_mask_prob')} ({m.get('context_mask_token')})")

        val_ds = val_col = None
        if self.val_every > 0 and is_main:
            try:
                val_ds, val_col, vcfg = self._build_val(tokenizer, cfg, val_dataset,
                                                        dataset_cls=type(train_dataset))
                log(f"[stage2] validation: {len(val_ds):,}개 (split={self.val_split}) | "
                    f"{self.val_every} optimizer step 마다 | 손실 {self.val_max_records}개 · "
                    f"생성 {self.val_gen_records}개")
            except Exception as e:
                log(f"[stage2][경고] validation 세트 구성 실패 — 검증 없이 진행: {type(e).__name__}: {e}")
                val_ds = None

        sampler = None
        if is_dist:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True,
                                         seed=self.seed)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=(sampler is None),
                            sampler=sampler, num_workers=self.num_workers, collate_fn=collate)

        log(f"[stage2] building model (LLM={mp.get('llm')}, "
            f"LoRA r={_as_dict(mp.get('lora', {})).get('r')}) ...")
        model = model.to(device)
        if model.use_mesh:
            model.mesh_encoder.to(device)

        net = model
        if is_dist:
            from torch.nn.parallel import DistributedDataParallel as DDP
            # 동결 파라미터(mesh 인코더)는 requires_grad=False 라 DDP 동기화 대상이 아니다.
            # 학습 대상(LoRA·fusion·projector)은 매 스텝 전부 쓰이므로 find_unused_parameters 불필요.
            net = DDP(model, device_ids=[local_rank], output_device=local_rank,
                      find_unused_parameters=False)

        n_train = sum(p.numel() for p in model.trainable_parameters())
        n_total = sum(p.numel() for p in model.parameters())
        log(f"[stage2] trainable params={n_train:,} / total={n_total:,} "
            f"({100 * n_train / n_total:.3f}%)")

        # Effective accumulation is capped by the epoch length, so short epochs
        # (len(loader) < grad_accum) still take an optimizer step per epoch.
        accum = max(1, min(self.grad_accum, len(loader)))
        if accum != self.grad_accum:
            log(f"[stage2] grad_accum {self.grad_accum} > steps/epoch {len(loader)}; "
                f"using effective grad_accum={accum}")

        # bridge(무작위 초기화)와 LoRA(사전학습 위 얹은 저랭크)는 필요한 LR 이 다르다.
        # 같은 LR + warm-up 없음 조합은 초반에 정렬이 안 된 prefix 로 LLM 을 흔든다.
        bridge_p, lora_p = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (lora_p if "lora_" in n else bridge_p).append(p)
        lr_bridge = self.lr_bridge if self.lr_bridge is not None else self.lr
        groups = [{"params": bridge_p, "lr": lr_bridge, "weight_decay": self.bridge_weight_decay},
                  {"params": lora_p, "lr": self.lr, "weight_decay": self.weight_decay}]
        optim = torch.optim.AdamW([g for g in groups if g["params"]])
        total_steps = math.ceil(len(loader) / accum) * self.epochs
        warmup_steps = int(self.warmup_ratio * total_steps)
        try:
            from transformers import get_cosine_schedule_with_warmup
            sched = get_cosine_schedule_with_warmup(optim, warmup_steps, max(total_steps, 1))
        except Exception:
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(total_steps, 1))
            warmup_steps = 0
        log(f"[stage2] 파라미터 그룹: bridge {sum(p.numel() for p in bridge_p):,} @lr={lr_bridge:g} | "
            f"LoRA {sum(p.numel() for p in lora_p):,} @lr={self.lr:g} | "
            f"warm-up {warmup_steps}/{total_steps} step ({self.warmup_ratio:.0%})")

        eff_batch = self.batch_size * accum * world
        log(f"[stage2] device={device} world={world} epochs={self.epochs} "
            f"batch/GPU={self.batch_size} grad_accum={accum} → 유효 배치 {eff_batch} "
            f"| steps~={total_steps}")
        if is_dist:
            log(f"[stage2] DDP: GPU {world}장. 유효 배치가 world 배로 커지므로 단일 GPU 와 같게 두려면 "
                f"grad_accum 을 1/{world} 로 줄여라.")

        # ---- wandb (rank 0 만) --------------------------------------------------
        run = None
        if is_main and self.wandb_enabled:
            try:
                import wandb
                Path(self.output_dir).mkdir(parents=True, exist_ok=True)
                wandb_config = {**self.cfg, "world_size": world, "effective_batch": eff_batch,
                                "n_records": len(dataset), "steps_per_epoch": len(loader),
                                "total_optimizer_steps": total_steps,
                                "trainable_params": n_train, "total_params": n_total,
                                "model": mp, "data": dcfg}
                if self.experiment_cfg is not None:
                    wandb_config["experiment"] = self.experiment_cfg
                run = wandb.init(
                    project=self.wandb_project,
                    name=self.wandb_name or Path(self.output_dir).name,
                    mode=self.wandb_mode,
                    dir=str(self.output_dir),
                    tags=[t for t in str(self.wandb_tags).split(",") if t],
                    config=wandb_config,
                )
                log(f"[stage2] wandb: {run.url}")
            except Exception as e:      # 로깅 실패로 학습이 죽으면 안 된다
                log(f"[stage2][경고] wandb 초기화 실패 — 로깅 없이 계속한다: {type(e).__name__}: {e}")
                run = None

        q_max_len = int(dcfg.get("q_max_len", 256))
        lang = str(dcfg.get("lang", "ko"))

        global_step = 0
        best = {"score": float("-inf"), "step": -1, "metrics": {}}
        net.train()
        for epoch in range(1, self.epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)      # 에폭마다 셔플이 바뀌게
            optim.zero_grad(set_to_none=True)
            running = 0.0
            last_grad_norms: dict = {}
            for it, batch in enumerate(loader, start=1):
                batch = move(batch, device)
                boundary = (it % accum == 0 or it == len(loader))
                # 누적 중간 스텝에서는 gradient all-reduce 를 건너뛴다 (accum 배 통신 절약)
                ctx = net.no_sync() if (is_dist and not boundary) else contextlib.nullcontext()
                with ctx:
                    out = net(batch)
                    (out["loss"] / accum).backward()
                running += out["loss"].item()

                if boundary:
                    if run is not None:
                        last_grad_norms = {"grad_norm/bridge": grad_norm(bridge_p),
                                           "grad_norm/lora": grad_norm(lora_p)}
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), self.max_grad_norm)
                    optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)
                    global_step += 1

                    # ---- 주기적 validation ----------------------------------
                    if self.val_every > 0 and global_step % self.val_every == 0:
                        if is_main and val_ds is not None:
                            vt = time.time()
                            try:
                                vres = self._validate(model, tokenizer, val_ds, val_col, device,
                                                      lang=lang, q_max_len=q_max_len)
                            except Exception as e:
                                # 여기서 예외가 나가면 rank 0 이 barrier 에 못 와 다른 랭크가
                                # 영원히 멈춘다. 검증 실패로 학습을 죽이지 않는다.
                                log(f"  [val][경고] 검증 실패 — 건너뛴다: {type(e).__name__}: {e}")
                                vres = {}
                            sample = vres.pop("_sample", "")
                            # best 기준은 mesh 를 봐야만 맞출 수 있는 카테고리여야 한다.
                            # val/mask_f1(micro)는 movement/region 이 turn_type 당 gold-set 이
                            # 사실상 하나뿐이라 상수 문자열만 뱉어도 F1 1.0 이 나오고, 실측 no-mesh
                            # floor 가 0.479 다. 그 기준으로 고르면 template 모방을 고르게 된다.
                            best_key = next((vres[k] for k in ("val/mask_f1_number_informative",
                                                               "val/mask_f1_number")
                                             if vres.get(k) is not None), None)
                            msg = " ".join(f"{k.split('/')[-1]}={v:.4f}"
                                           for k, v in vres.items()
                                           if isinstance(v, float) and k.startswith("val/"))
                            log(f"  [val] step {global_step} ({time.time()-vt:.0f}s) {msg}")
                            if sample:
                                log(f"  [val] 예시 생성: {sample[:160]}")
                            if run is not None and vres:
                                run.log(vres, step=global_step)
                            # best 기준: mesh 의존 카테고리 F1 우선, 없으면 unweighted CE 최소
                            cur = (best_key if best_key is not None
                                   else -vres.get("val/unweighted_ce", float("inf")))
                            if vres and cur is not None and cur > best["score"]:
                                best.update(score=cur, step=global_step,
                                            metrics={k: v for k, v in vres.items()})
                                self._save(model, Path(self.output_dir) / "checkpoint_best")
                                (Path(self.output_dir) / "checkpoint_best" / "best.json").write_text(
                                    json.dumps(best, ensure_ascii=False, indent=2, default=str))
                                log(f"  [val] best 갱신 (score={cur:.4f}) → checkpoint_best/")
                        if is_dist:
                            import torch.distributed as dist
                            dist.barrier()      # 검증 동안 다른 랭크는 대기
                    if self.save_every > 0 and global_step % self.save_every == 0 and is_main:
                        self._save(model, Path(self.output_dir) / f"checkpoint_step{global_step:06d}")
                        log(f"  [ckpt] step {global_step} 저장")

                if it % self.log_every == 0:
                    aux = out["aux_loss"]
                    lrs = sched.get_last_lr()
                    lr_b, lr_l = lrs[0], lrs[-1]   # group0=bridge, group1=LoRA
                    msg = (f"  epoch {epoch} it {it}/{len(loader)} "
                           f"loss={out['loss'].item():.4f} lm={out['lm_loss'].item():.4f}")
                    if aux is not None:
                        msg += f" aux={aux.item():.4f}"
                    msg += f" lr(bridge/lora)={lr_b:.2e}/{lr_l:.2e}"
                    log(msg)   # rank 0 의 로컬 손실 (전체 평균 아님)
                    if run is not None:
                        rec = {"train/loss": out["loss"].item(),
                               "train/lm_loss": out["lm_loss"].item(),
                               "train/lr": lr_l, "train/lr_bridge": lr_b,
                               "train/epoch": epoch,
                               "train/progress": (epoch - 1 + it / len(loader)) / self.epochs}
                        if aux is not None:
                            rec["train/aux_loss"] = aux.item()
                        # 분해 지표 — 가중 손실만 보면 λ 변동에 묻혀 무엇이 줄었는지 모른다
                        for k in ("unweighted_ce", "fact_ce", "nonfact_ce", "n_supervised",
                                  "n_fact", "fact_token_ratio", "empty_supervision",
                                  "prefix_rms", "token_embed_rms", "prefix_to_token_rms"):
                            if k in out and out[k] is not None:
                                rec[f"train/{k}"] = float(out[k])
                        rec.update(last_grad_norms)
                        run.log(rec, step=global_step)

            mean_loss = running / max(len(loader), 1)
            log(f"[stage2] epoch {epoch} mean_loss={mean_loss:.4f}")
            if run is not None:
                run.log({"train/epoch_mean_loss": mean_loss, "train/epoch": epoch},
                        step=global_step)
            if is_main:
                self._save(model, self.output_dir)   # DDP 래퍼가 아니라 원본에서 저장
                log(f"[stage2] saved checkpoint -> {self.output_dir}")
            if is_dist:
                import torch.distributed as dist
                dist.barrier()                # 저장 중 다른 랭크가 앞서 나가지 않게

        if is_main:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            payload = {**self.cfg, "world_size": world, "effective_batch": eff_batch,
                       # 옛 vars(args) 는 lr/epochs/lora_r 가 전부 최상위 평면 키였다.
                       # 이전 run 과 키 단위로 비교하려면 model.* / data.* 를 봐야 한다.
                       "model": mp, "data": dcfg}
            if self.experiment_cfg is not None:
                payload["experiment"] = self.experiment_cfg
            (Path(self.output_dir) / "train_config.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str)
            )
        if run is not None:
            run.finish()
        if is_dist:
            import torch.distributed as dist
            dist.destroy_process_group()
        log("[stage2] done.")
        return best
