"""Stage-2 평가 — DATA/unseentest/readme.md 의 3개 세트를 돌려 표의 column 을 채운다.

  Set 1  in-distribution (새 메쉬)      → Mask F1 · Abstention F1 · Direction · Monotonicity
  Set 2  synergy (새 질문 family)       → Sign acc · Mask F1
  Set 3  diagnostic (새 질문 형식)      → Correction-sign F1 · GPT-judge 페이로드 · 사람 검토 리포트

산출물: <output_dir>/Set1, Set2, Set3 아래
  preds.jsonl    {"index", "turn", "turn_type", "question", "pred", "gold"}
  metrics.json   지표 결과 + 실행 메타(체크포인트·생성 파라미터)
  Set3/render/   항목별 PNG + index.html (메쉬 2뷰 + 질문 + 예측 + GT 근거)
  Set3/judge_payloads.jsonl

원본 ``eval.py`` 에서 그대로 옮겼다. `jobs_set1` / `score_set1` 등은 **모듈 레벨 함수**로
남겨 둔다 — dummy/scripts/trivial_baselines.py 가 모델 없이 그 둘만 import 해서 쓴다.
`EvaluatorUnseen` 는 그 함수들을 부르는 얇은 오케스트레이터다.
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

import metrics as M
from datasets.mesh_store import MESH_CONTROLS, MESH_ROOT, MeshStore, apply_mesh_control
from render_report import build_index, render_item
from utils import write_jsonl


def _as_dict(cfg: Any) -> dict:
    """OmegaConf DictConfig / Mapping / plain object → dict."""
    if cfg is None:
        return {}
    if hasattr(cfg, "items") and not isinstance(cfg, type):
        try:
            if OmegaConf.is_config(cfg):
                return dict(OmegaConf.to_container(cfg, resolve=True))
        except Exception:
            pass
        return dict(cfg)
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}


def _as_conf(cfg: Any):
    """dict / DictConfig / None → DictConfig. 원본 코드가 `cfg.x` 속성 접근을 쓴다."""
    if cfg is not None and OmegaConf.is_config(cfg):
        return cfg
    return OmegaConf.create(_as_dict(cfg))


def load_tokenizer(llm_id: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(llm_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"      # 배치 생성에는 왼쪽 패딩이어야 한다
    return tok


# --------------------------------------------------------------------------- #
# 생성
# --------------------------------------------------------------------------- #
def _render_prompt(tok, messages, enable_thinking: bool):
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=enable_thinking)
    except TypeError:      # 템플릿이 해당 인자를 모르면 그냥 기본 렌더
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_input_source(exp_models, store):
    """인코더 입력을 인덱스로 꺼내는 함수와 그 배치 키를 돌려준다.

    3D row 는 mesh 변위(disp), 2D 렌더 row 는 3뷰 이미지(imgs)다. 모델 쪽 규약은
    `Stage2Model.input_key` = 동결 인코더의 `input_key` 와 같다.
    """
    return "disp", (lambda i: torch.from_numpy(store.disp(int(i))))


@torch.no_grad()
def generate(model, tok, store, jobs, cfg, device, data_cfg, exp_models=None):
    """jobs: [{"messages", "question", "mesh_indices", ...}] → 각 job 에 'pred' 를 채워 돌려준다."""
    g = cfg.generation
    in_key, get_input = make_input_source(exp_models, store)
    bs = int(cfg.batch_size)
    gen_kwargs = dict(max_new_tokens=int(g.max_new_tokens), do_sample=bool(g.do_sample),
                      num_beams=int(g.num_beams), pad_token_id=tok.pad_token_id)

    for st in range(0, len(jobs), bs):
        chunk = jobs[st:st + bs]
        texts = [_render_prompt(tok, j["messages"], bool(g.enable_thinking)) for j in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        q = tok([j["question"] for j in chunk], return_tensors="pt", padding=True,
                truncation=True, max_length=int(getattr(data_cfg, "q_max_len", 256) or 256))

        # mesh 통제(run.mesh_control)가 걸려 있으면 'mesh_indices_eff' 가 다른 mesh 를 가리킨다.
        # mesh_valid 는 원래 mesh 개수를 따라간다 — 통제가 시퀀스 길이를 바꾸면 안 된다.
        m_max = max(len(j["mesh_indices"]) for j in chunk)
        first = get_input(chunk[0].get("mesh_indices_eff", chunk[0]["mesh_indices"])[0])
        inputs = torch.zeros(len(chunk), m_max, *first.shape)
        valid = torch.zeros(len(chunk), m_max, dtype=torch.bool)
        for bi, j in enumerate(chunk):
            src = j.get("mesh_indices_eff") or j["mesh_indices"]
            for mi in range(len(j["mesh_indices"])):
                d = get_input(src[mi % len(src)])
                pert = j.get("mesh_perturb")
                if pert == "rest":
                    # 3D 는 변위 0 = rest 형상. 렌더는 대응하는 rest 렌더가 캐시에 없어
                    # 빈 이미지(배경만)가 된다 — "형상 정보 없음" 통제로 읽을 것.
                    d = torch.zeros_like(d)
                elif pert == "noise":
                    d = torch.randn_like(d) * d.pow(2).mean().sqrt()
                inputs[bi, mi] = d
                valid[bi, mi] = True

        batch = {"input_ids": enc["input_ids"].to(device),
                 "attention_mask": enc["attention_mask"].to(device),
                 "q_input_ids": q["input_ids"].to(device),
                 "q_attention_mask": q["attention_mask"].to(device),
                 in_key: inputs.to(device), "mesh_valid": valid.to(device)}
        out = model.generate_answer(batch, **gen_kwargs)
        for j, seq in zip(chunk, out):
            j["pred"] = tok.decode(seq, skip_special_tokens=True).strip()
        print(f"  생성 {min(st + bs, len(jobs))}/{len(jobs)}", flush=True)
    return jobs


# --------------------------------------------------------------------------- #
# 세트별 job 구성
# --------------------------------------------------------------------------- #
def jobs_set1(cfg, sc, args, limit=None):
    """test-split QA 를 assistant 턴 단위로 편다. 앞 턴은 gold 로 채운 문맥으로 준다.

    소스(A1 / PH / B3)마다 따로 읽는다 — 한 glob 에 상한을 걸면 파일 순서대로 잘려
    A1 턴만 뽑히고 D1/B2/B3 이 사라진다.

    ⚠️ turn_wise 인덱스는 파일 round-robin 이라 **앞에서 자르면 각 시나리오의 첫 turn_type
    만** 뽑힌다. 실측: PH `max_records: 100` → 전부 `A2`, D1/B2/C1/prescriptive 0개
    (= Abstention F1 과 Direction acc 가 구조적으로 측정 불가). 그래서 넉넉히 읽은 뒤
    `per_turn_type` 만큼 **turn_type 별로 층화 추출**한다. 생성 비용은 층화 후 개수로 정해진다.
    """
    from datasets.qa_dataset import MeshQaDataset

    base = OmegaConf.load("configs/datasets/qa_dataset.yaml")
    jobs = []
    for src in sc.sources:
        n_max = int(limit) if limit else int(src.max_records)
        data = OmegaConf.merge(base, {"qa_glob": src.qa_glob, "split": sc.split,
                                      "lang": cfg.lang, "variants": sc.variants,
                                      "max_records": n_max,
                                      "split_policy": src.get("split_policy", "all")})
        ds = MeshQaDataset(data)
        src_jobs = []
        for i in range(len(ds)):
            it = ds[i]
            msgs, tt, ai = it["messages"], it["turn_types"], 0
            for pos, m in enumerate(msgs):
                if m["role"] != "assistant":
                    continue
                src_jobs.append({
                    "index": int(it["index"]), "turn": ai,
                    "turn_type": tt[ai] if ai < len(tt) else "",
                    "messages": msgs[:pos],                   # system + 이전 턴 + 이번 질문
                    "question": msgs[pos - 1]["content"],
                    "gold": m["content"],
                    "gold_spans": it["answer_spans"][ai] if ai < len(it["answer_spans"]) else [],
                    "mesh_indices": it["mesh_indices"],
                    "scenario": it["scenario"],
                })
                ai += 1

        per_tt = None if limit else src.get("per_turn_type", None)
        if per_tt:
            rng = random.Random(int(cfg.seed))
            by_tt: dict[str, list] = {}
            for j in src_jobs:
                by_tt.setdefault(j["turn_type"], []).append(j)
            kept = []
            for tt in sorted(by_tt):
                g = by_tt[tt]
                rng.shuffle(g)
                kept.extend(g[: int(per_tt)])
            src_jobs = kept
        counts = collections.Counter(j["turn_type"] for j in src_jobs)
        print(f"[Set1] {Path(str(src.qa_glob)).name}: 레코드 {len(ds)}개 "
              f"→ 턴 {len(src_jobs)}개 {dict(sorted(counts.items()))}")
        jobs.extend(src_jobs)
    print(f"[Set1] 채점 대상 턴 {len(jobs)}개")
    return jobs


def _load_records(path: Path):
    txt = Path(path).read_text(encoding="utf-8")
    s = txt.lstrip()
    if s.startswith("["):                     # JSON 배열
        return json.loads(s)
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


def jobs_gen_set(cfg, sc, system_prompt: str, tag: str):
    recs = _load_records(sc.path)
    if sc.max_records:
        recs = recs[: int(sc.max_records)]
    print(f"[{tag}] 레코드 {len(recs)}개 ← {sc.path}")
    jobs = []
    for r in recs:
        conv = r["conversations"]
        q = conv[0]["value"]
        gold_turn = next((t for t in conv if t["from"] == "gpt"), None)
        jobs.append({
            "index": int(r["index"]), "turn": 0, "turn_type": r.get("family", tag),
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": q}],
            "question": q,
            "gold": (gold_turn or {}).get("value", ""),
            "gold_spans": (gold_turn or {}).get("mask_spans", []),
            "mesh_indices": MeshStore.index_of(r),
            "score": r.get("score", {}),
            "judge_grounding": r.get("judge_grounding"),
            "section": r.get("section", ""),
        })
    return jobs


# --------------------------------------------------------------------------- #
# 채점
# --------------------------------------------------------------------------- #
def score_set1(jobs, sc, lang):
    out = {}
    names = list(sc.metrics)
    if "mask_f1" in names:
        # 기권이 정답인 턴은 fact 재현을 요구하지 않으므로 제외한다
        items = [{"gold_spans": j["gold_spans"], "pred": j["pred"],
                  "turn_type": j["turn_type"]}
                 for j in jobs if j["turn_type"] not in set(sc.abstain_turn_types)]
        out["mask_f1"] = M.mask_f1.score(items, lang)
    if "abstention_f1" in names:
        need = set(sc.abstain_turn_types)
        out["abstention_f1"] = M.abstention_f1.score(
            [{"pred": j["pred"], "should_abstain": j["turn_type"] in need} for j in jobs], lang)
    if "direction_acc" in names:
        tt = set(sc.direction_turn_types)
        out["direction_acc"] = M.direction_acc.score(
            [j for j in jobs if j["turn_type"] in tt], lang)
    if "monotonicity_acc" in names:
        tt = set(sc.monotonicity_turn_types)
        out["monotonicity_acc"] = M.monotonicity_acc.score(
            [j for j in jobs if j["turn_type"] in tt], lang)
    out["turn_type_counts"] = {t: sum(1 for j in jobs if j["turn_type"] == t)
                               for t in sorted({j["turn_type"] for j in jobs})}
    out["train_mesh_overlap"] = train_overlap(jobs)
    return out


def train_overlap(jobs) -> dict:
    """평가에 쓴 mesh 중 학습 split 에 들어 있던 비율 (오염도).

    split_policy=primary 로 완화한 소스(B3)는 나머지 mesh 가 train 에 있을 수 있다.
    "held-out 이다"라고 쓰기 전에 이 숫자를 확인하라.
    """
    from datasets.split_trainvaltest import load_split_indices
    train = set(load_split_indices(MESH_ROOT, "train"))
    per_type: dict[str, list[int]] = {}
    for j in jobs:
        seen = sum(1 for i in j["mesh_indices"] if i in train)
        a, b = per_type.setdefault(j["turn_type"], [0, 0])
        per_type[j["turn_type"]] = [a + seen, b + len(j["mesh_indices"])]
    return {t: {"in_train": a, "total": b, "rate": (a / b if b else 0.0)}
            for t, (a, b) in sorted(per_type.items())}


def score_set2(jobs, sc, lang):
    out = {}
    if "sign_acc" in sc.metrics:
        out["sign_acc"] = M.sign_acc.score(
            [{"gold": j["score"]["gold"], "pred": j["pred"]} for j in jobs])
    if "mask_f1" in sc.metrics:
        out["mask_f1"] = M.mask_f1.score(
            [{"gold_spans": j["gold_spans"], "pred": j["pred"]} for j in jobs], lang)
    return out


def score_set3(jobs, sc, out_dir: Path):
    out = {}
    if "correction_sign_f1" in sc.metrics:
        out["correction_sign_f1"] = M.correction_sign_f1.score(
            [{"gold_inc": j["score"].get("gold_inc", []),
              "gold_dec": j["score"].get("gold_dec", []), "pred": j["pred"]} for j in jobs])
    if "judge_payload" in sc.metrics:
        jd = sc.get("judge", {}) or {}
        out["judge_payload"] = M.judge_payload.build(
            [{"index": j["index"], "question": j["question"], "pred": j["pred"],
              "grounding": j["judge_grounding"]} for j in jobs if j.get("judge_grounding")],
            out_dir / "judge_payloads.jsonl",
            model=str(jd.get("model", "gpt-4o")),
            temperature=float(jd.get("temperature", 0.0)),
            n_votes=int(jd.get("n_votes", 3)))
    return out


def render_set3(jobs, sc, store, out_dir: Path):
    r = sc.get("render", {}) or {}
    if not r.get("enabled", False):
        return None
    n = int(r.get("max_items", 100))
    sel = jobs[:n]
    png_dir = out_dir / "render"
    entries = []
    print(f"[Set3] 사람 검토용 렌더 {len(sel)}개 → {png_dir}")
    for k, j in enumerate(sel):
        g = j.get("judge_grounding") or {}
        png = png_dir / f"{j['index']:06d}.png"
        render_item(store.rest, store.faces, store.disp(j["mesh_indices"][0]),
                    question=j["question"], pred=j.get("pred", ""), grounding=g,
                    title=f"#{j['index']}  section={j.get('section','')}  "
                          f"target=/{g.get('target','?')}/",
                    out_png=png)
        entries.append({"index": j["index"], "png": f"render/{png.name}",
                        "section": j.get("section", ""), "target": g.get("target", ""),
                        "question": j["question"], "pred": j.get("pred", ""),
                        "should_increase": g.get("should_increase", []),
                        "should_decrease": g.get("should_decrease", [])})
        if (k + 1) % 20 == 0:
            print(f"  렌더 {k + 1}/{len(sel)}", flush=True)
    idx = build_index(entries, out_dir / "index.html", f"Set 3 — 진단형 사람 검토 ({len(sel)}개)")
    human = int(r.get("human_subset", 0) or 0)
    if human:
        (out_dir / "human_subset.json").write_text(
            json.dumps([e["index"] for e in entries[:human]], indent=2), encoding="utf-8")
    return {"n_rendered": len(sel), "index_html": str(idx), "human_subset": human}


# --------------------------------------------------------------------------- #
class EvaluatorUnseen:
    """Prefer: ``EvaluatorUnseen(cfg.evaluators, experiment_cfg=cfg)`` then ``.run()``."""

    def __init__(self, cfg=None, experiment_cfg=None, **kwargs):
        p = _as_dict(cfg)
        p.update(kwargs)
        self.experiment_cfg = _as_conf(experiment_cfg)
        exp = self.experiment_cfg

        # 원본 eval yaml 은 seed 를 평가 설정과 같은 노드에 두었다. jobs_set1/apply_mesh_control
        # 이 `cfg.seed` 를 읽으므로 루트의 seed 를 같은 노드로 끌어온다.
        p.setdefault("seed", int(exp.get("seed", 42)))
        self.cfg = OmegaConf.create(p)

        # 원본 --output. 평가 설정에 output_dir 이 없으면 <실험 output_dir>/eval 로 간다.
        self.output_dir = p.get("output_dir", None)
        self.checkpoint = exp.get("checkpoint", None)
        self.device = str(exp.get("device", "cuda"))
        self.run_cfg = _as_conf(exp.get("run", {}))
        self.name = str(exp.get("name", ""))

    # ----------------------------------------------------------------------- #
    # 모델 로드
    # ----------------------------------------------------------------------- #
    def load_model(self):
        """실험 설정 + 저장된 가중치(LoRA · fusion)로 Stage-2 모델을 세운다."""
        from models.stage2.model_s2 import Stage2Model

        exp = self.experiment_cfg
        ckpt_dir = Path(self.checkpoint) if self.checkpoint else Path(exp.output_dir)
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        print(f"[eval] 모델 구성: {self.name} (encoder_init={exp.models.encoder_init})")
        model = Stage2Model(exp.models)

        bridge_fp = ckpt_dir / "mm_projector.pt"
        lora_dir = ckpt_dir / "lora"
        if lora_dir.is_dir():
            from peft import load_peft_weights, set_peft_model_state_dict
            set_peft_model_state_dict(model.llm, load_peft_weights(str(lora_dir)))
            print(f"[eval] LoRA 로드: {lora_dir}")
        elif self.run_cfg.get("require_weights"):
            # --require-weights: 미학습 가중치로 그럴듯한 쓰레기 숫자를 만드는 사고를 막는다.
            raise SystemExit(f"[eval] LoRA 가 없다: {lora_dir}\n"
                             f"       --checkpoint <output_dir>/checkpoint_best 를 지정하라.")
        else:
            print(f"[eval][경고] LoRA 가 없다: {lora_dir} — 학습 전 가중치로 평가한다")
        if bridge_fp.is_file():
            bridge = torch.load(bridge_fp, map_location="cpu", weights_only=False)
            if model.use_mesh and "fusion" in bridge:
                model.fusion.load_state_dict(bridge["fusion"])
            if getattr(model, "aux_muscle", False) and "muscle_head" in bridge:
                model.muscle_head.load_state_dict(bridge["muscle_head"])
            print(f"[eval] fusion/projector 로드: {bridge_fp}")
        elif model.use_mesh:
            print(f"[eval][경고] fusion 가중치가 없다: {bridge_fp}")

        model.to(device)
        if model.use_mesh:
            model.mesh_encoder.to(device)
        model.eval()
        return model, device, ckpt_dir

    # ----------------------------------------------------------------------- #
    def run(self):
        cfg = self.cfg                      # 평가 설정 (lang · batch_size · generation · sets)
        exp = self.experiment_cfg           # 실험 루트 (models · datasets · output_dir · run)
        run = self.run_cfg

        # mesh grounding 통제. real=그대로, shuffle=같은 turn_type 안에서 다른 mesh,
        # rest=변위 0, noise=RMS 맞춘 가우시안
        mesh_control = str(run.get("mesh_control", "real") or "real")
        if mesh_control not in MESH_CONTROLS:
            raise SystemExit(f"[eval] run.mesh_control 은 {sorted(MESH_CONTROLS)} 중 하나여야 "
                             f"한다: {mesh_control}")
        limit = run.get("limit", None)
        score_only = bool(run.get("score_only", False))
        render_only = bool(run.get("render_only", False))
        want_sets = run.get("sets", None)

        out_root = Path(self.output_dir or (Path(exp.output_dir) / "eval"))
        if mesh_control != "real" and not self.output_dir:
            # 통제 조건 결과가 기준(real) 결과를 덮어쓰지 않도록 분리한다.
            out_root = out_root.parent / f"{out_root.name}_mesh-{mesh_control}"
        out_root.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(int(cfg.seed))

        want = set(want_sets) if want_sets else {k for k in cfg.sets if cfg.sets[k].enabled}
        store = MeshStore()
        need_gen = not (score_only or render_only)

        model = tok = device = None
        data_cfg = _as_conf(exp.get("datasets", {}))
        ckpt_dir = Path(self.checkpoint) if self.checkpoint else Path(exp.output_dir)
        if need_gen:
            model, device, ckpt_dir = self.load_model()
            tok = load_tokenizer(str(exp.models.llm))

        system_prompt = None
        from datasets.qa_dataset import SYSTEM_PROMPT
        system_prompt = SYSTEM_PROMPT.get(str(cfg.lang), SYSTEM_PROMPT["en"])

        summary = {}
        for key, tag in (("set1", "Set1"), ("set2", "Set2"), ("set3", "Set3")):
            if key not in want:
                continue
            sc = cfg.sets[key]
            if limit:
                sc = OmegaConf.merge(sc, {"max_records": limit})
            out_dir = out_root / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            preds_fp = out_dir / "preds.jsonl"

            print(f"\n=== {tag} ===")
            if key == "set1":
                jobs = jobs_set1(cfg, sc, data_cfg, limit) if need_gen else None
            else:
                jobs = jobs_gen_set(cfg, sc, system_prompt, tag)

            if need_gen:
                jobs = apply_mesh_control(jobs, mesh_control, int(cfg.seed))
                jobs = generate(model, tok, store, jobs, cfg, device, data_cfg,
                                exp_models=exp.get("models", {}))
                write_jsonl(preds_fp, [{k: j[k] for k in
                                        ("index", "turn", "turn_type", "question", "pred", "gold")}
                                       for j in jobs])
            else:
                if not preds_fp.is_file():
                    print(f"[{tag}] preds.jsonl 이 없어 건너뛴다: {preds_fp}")
                    continue
                saved = {(int(r["index"]), int(r.get("turn", 0))): r["pred"]
                         for r in _load_records(preds_fp)}
                if key == "set1":
                    print(f"[{tag}] run.score_only 는 Set1 의 gold span 이 필요해 재구성한다")
                    jobs = jobs_set1(cfg, sc, data_cfg, limit)
                for j in jobs:
                    j["pred"] = saved.get((j["index"], j.get("turn", 0)), "")

            if key == "set1":
                res = score_set1(jobs, sc, str(cfg.lang))
            elif key == "set2":
                res = score_set2(jobs, sc, str(cfg.lang))
            else:
                res = score_set3(jobs, sc, out_dir)
                rep = render_set3(jobs, sc, store, out_dir)
                if rep:
                    res["render"] = rep

            res["_meta"] = {
                "experiment": self.name, "checkpoint": str(ckpt_dir),
                "lang": str(cfg.lang), "n_jobs": len(jobs),
                "mesh_control": mesh_control,
                "generation": OmegaConf.to_container(cfg.generation, resolve=True),
            }
            (out_dir / "metrics.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            summary[tag] = res
            print(json.dumps({k: v for k, v in res.items() if not k.startswith("_")},
                             ensure_ascii=False, indent=2)[:1400])

        (out_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n=== 완료 → {out_root} ===")
        return summary
