"""Set-1 probe evaluator — Metircs.md / SET1_PROBE_SPEC.md cloze probe.

Config만으로 `{output_dir}/checkpoint_best` 로드 → 추론 → 채점 → 렌더.

산출물 (`{output}/` = evaluator output_dir, 기본 `{exp.output_dir}/eval`):
  pred/render/{testdata_index}.png
      정면 Mesh + Q + A + (reference_answer) 큰 글씨
  pred/auto_metric_json/
      muscle_set.json · value_set.json · direction_set.json · abstention_set.json
  pred/judge_metric_json/gpt/
      imgs_preds.json      — Q+A+images (NO reference)
      mesh_preds.json      — Q+A+mesh_npy (NO reference)
      reference_answer.json — offline only; do NOT upload to GPT
  pred/judge_metric_json/human/{testdata_index}.png
      utility 항목: 정면 Mesh + Q + A + (reference_answer)
"""

from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

from datasets.mesh_store import MESH_CONTROLS, MeshStore, apply_mesh_control
from datasets.qa_dataset import SYSTEM_PROMPT
from evaluators.unseen import load_tokenizer, make_input_source
from metrics import set1_probe as M
from metrics.spans import is_abstention
from render_report import build_index
from utils import write_jsonl

FRONT_PNG_ROOT = Path("DATA/mesh_png/front")
REF_ANSWERS_TMPL = "DATA/unseentest/utility_judge_en/reference_answers_{lang}.jsonl"
# matplotlib 정면 대체뷰 (캐시 PNG 없을 때). 해부학 -X=앞 → azim=180.
FRONT_VIEW = (8.0, 180.0)


def _as_dict(cfg: Any) -> dict:
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
    return {}


def _as_conf(cfg: Any):
    if cfg is not None and OmegaConf.is_config(cfg):
        return cfg
    return OmegaConf.create(_as_dict(cfg))


def resolve_best_checkpoint(exp) -> Path:
    """Config만 넣어도 best epoch을 찾는다.

    우선순위:
      1) exp.checkpoint 가 디렉터리/파일로 존재
      2) {checkpoint}/checkpoint_best
      3) {output_dir}/checkpoint_best
      4) {output_dir}
    """
    raw = exp.get("checkpoint", None)
    out = Path(str(exp.get("output_dir", "outputs/stage2")))
    candidates = []
    if raw:
        p = Path(str(raw))
        candidates += [p, p / "checkpoint_best"]
    candidates += [out / "checkpoint_best", out]
    for c in candidates:
        if (c / "lora").is_dir() or (c / "mm_projector.pt").is_file():
            return c
    # 없어도 경로를 돌려 load_model 이 경고/중단하게 한다
    return out / "checkpoint_best" if not raw else Path(str(raw))


def load_probe(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def jobs_from_probe(records: list[dict], lang: str, system_prompt: str,
                    limit: int | None = None) -> list[dict]:
    jobs = []
    for rec in records:
        if limit is not None and len(jobs) >= int(limit):
            break
        leadin = rec.get("answer_leadin") or ""
        jobs.append({
            "uid": rec["uid"],
            "family": rec["family"],
            "index": int(rec["mesh_index"]),
            "mesh_index": int(rec["mesh_index"]),
            "mesh_indices": [int(rec["mesh_index"])],
            "n_act": rec.get("n_act"),
            "n_act_bin": rec.get("n_act_bin"),
            "turn": 0,
            "turn_type": rec["family"],
            "question": rec["question"],
            "answer_leadin": leadin,
            "gold": rec.get("gold"),
            "gold_text": (rec.get("gold") or {}).get("text"),
            "should_abstain": bool(rec.get("should_abstain")),
            "score_auto": bool(rec.get("score_auto", True)),
            "meta": rec.get("meta") or {},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": rec["question"]},
            ],
            # cloze: generation prompt 뒤에 leadin 을 붙인다 (unseen generate 확장)
            "prompt_suffix": leadin,
        })
    return jobs


@torch.no_grad()
def generate_cloze(model, tok, store, jobs, cfg, device, data_cfg, exp_models=None):
    """unseen.generate 와 같되 `prompt_suffix`(answer_leadin)를 붙인다."""
    g = cfg.generation
    in_key, get_input = make_input_source(exp_models, store)
    bs = int(cfg.batch_size)
    gen_kwargs = dict(max_new_tokens=int(g.max_new_tokens), do_sample=bool(g.do_sample),
                      num_beams=int(g.num_beams), pad_token_id=tok.pad_token_id)

    def _render(messages, suffix: str):
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=bool(g.enable_thinking))
        except TypeError:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        if suffix:
            # assistant 턴이 이미 leadin 으로 시작하도록 강제
            text = text + suffix
        return text

    for st in range(0, len(jobs), bs):
        chunk = jobs[st:st + bs]
        texts = [_render(j["messages"], j.get("prompt_suffix") or "") for j in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        q = tok([j["question"] for j in chunk], return_tensors="pt", padding=True,
                truncation=True, max_length=int(getattr(data_cfg, "q_max_len", 256) or 256))

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
            cont = tok.decode(seq, skip_special_tokens=True).strip()
            j["pred"] = cont
            lead = j.get("answer_leadin") or ""
            j["pred_display"] = (f"{lead} {cont}".strip() if lead else cont)
        print(f"  생성 {min(st + bs, len(jobs))}/{len(jobs)}", flush=True)
    return jobs


def _wrap_text(text: str, width: int = 72, max_lines: int = 10) -> str:
    lines: list[str] = []
    for para in (text or "").strip().splitlines() or [""]:
        lines.extend(textwrap.wrap(para, width=width) or [""])
    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["…"]
    return "\n".join(lines)


def load_reference_answers(path: str | Path | None, lang: str) -> dict[str, str]:
    """uid → reference_answer. utility gold.text 가 비어 있을 때 사용."""
    candidates = []
    if path:
        candidates.append(Path(str(path)))
    candidates.append(Path(REF_ANSWERS_TMPL.format(lang=lang)))
    for fp in candidates:
        if not fp.is_file():
            continue
        out: dict[str, str] = {}
        with fp.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                uid = row.get("uid")
                ref = row.get("reference_answer")
                if uid and ref is not None:
                    out[str(uid)] = str(ref)
        return out
    return {}


def reference_answer_of(job: dict) -> str:
    if job.get("reference_answer"):
        return str(job["reference_answer"])
    gold = job.get("gold") or {}
    text = gold.get("text")
    if text is not None and str(text).strip():
        return str(text)
    if job.get("family") == "muscle_set":
        muscles = gold.get("muscles") or []
        return ", ".join(muscles) if muscles else ""
    if job.get("family") == "value" and gold.get("value") is not None:
        return str(gold["value"])
    if job.get("family") == "direction":
        if gold.get("kind") == "correction" or gold.get("gold_inc") or gold.get("gold_dec"):
            inc = ", ".join(gold.get("gold_inc") or []) or "-"
            dec = ", ".join(gold.get("gold_dec") or []) or "-"
            return f"↑ {inc}; ↓ {dec}"
        dirs = gold.get("directions") or {}
        if dirs:
            parts = [f"{r}={'+'.join(v) if isinstance(v, (list, tuple)) else v}"
                     for r, v in dirs.items()]
            return "; ".join(parts)
    if job.get("should_abstain"):
        return "ABSTAIN"
    return ""


def attach_reference_answers(jobs: list[dict], refs: dict[str, str]) -> None:
    for j in jobs:
        if j["uid"] in refs:
            j["reference_answer"] = refs[j["uid"]]
        elif not j.get("reference_answer"):
            j["reference_answer"] = reference_answer_of(j)


def testdata_index_of(job: dict) -> str:
    """파일명용 인덱스 — probe uid (고유)."""
    return str(job.get("uid") or job.get("index") or job.get("mesh_index"))


def render_front_qa(
    mesh_index: int,
    *,
    question: str,
    answer: str,
    reference_answer: str,
    out_png: Path,
    title: str = "",
    front_png_root: Path | None = None,
    store: MeshStore | None = None,
) -> Path:
    """정면 Mesh + 큰 글씨 Q / A / (reference_answer)."""
    import matplotlib
    matplotlib.use("Agg")
    from render_report import _ensure_font
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    _ensure_font()
    root = Path(front_png_root or FRONT_PNG_ROOT)
    front_fp = root / f"{int(mesh_index)}.png"

    fig = plt.figure(figsize=(12.5, 11.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.08)
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)

    ax_m = fig.add_subplot(gs[0, 0])
    if front_fp.is_file():
        ax_m.imshow(np.asarray(Image.open(front_fp).convert("RGB")))
        ax_m.set_title("front", fontsize=14, pad=6)
        ax_m.axis("off")
    else:
        # 캐시 없으면 matplotlib 정면 3D
        if store is None:
            store = MeshStore()
        disp = store.disp(int(mesh_index))
        verts = store.rest + disp
        faces = store.faces
        mag = np.linalg.norm(disp, axis=-1)
        face_mag = mag[faces].mean(axis=1)
        ax_m.remove()
        ax_m = fig.add_subplot(gs[0, 0], projection="3d")
        tri = ax_m.plot_trisurf(
            verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces,
            cmap="viridis", linewidth=0.1, edgecolor="none", antialiased=True)
        tri.set_array(face_mag)
        vmin, vmax = float(face_mag.min()), float(face_mag.max())
        tri.set_clim(vmin, vmax if vmax > vmin else vmin + 1e-6)
        elev, azim = FRONT_VIEW
        ax_m.view_init(elev=elev, azim=azim)
        ax_m.set_title("front", fontsize=14)
        ax_m.set_box_aspect((np.ptp(verts[:, 0]), np.ptp(verts[:, 1]), np.ptp(verts[:, 2])))
        ax_m.set_xticks([]); ax_m.set_yticks([]); ax_m.set_zticks([])
        ax_m.grid(False)

    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    blocks = [
        ("Q", _wrap_text(question, 70, 6), "#0a7a3e"),
        ("A", _wrap_text(answer, 70, 8), "#1a3a6b"),
        ("(reference_answer)", _wrap_text(reference_answer or "(none)", 70, 8), "#6b3a1a"),
    ]
    y = 0.98
    for label, body, color in blocks:
        ax.text(0.0, y, label, fontsize=18, fontweight="bold", va="top",
                color=color, transform=ax.transAxes)
        ax.text(0.0, y - 0.07, body, fontsize=15, va="top", linespacing=1.35,
                color="#111", transform=ax.transAxes)
        nline = body.count("\n") + 1
        y -= 0.10 + 0.055 * nline

    fig.subplots_adjust(left=0.05, right=0.97, top=0.94, bottom=0.03)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def _family_item_record(job: dict, lang: str = "en") -> dict:
    """auto_metric_json 항목 한 줄."""
    s = job.get("item_score") or {}
    fam = job["family"]
    rec = {
        "testdata_index": testdata_index_of(job),
        "uid": job["uid"],
        "family": fam,
        "mesh_index": int(job["mesh_index"]),
        "n_act": job.get("n_act"),
        "n_act_bin": job.get("n_act_bin"),
        "question": job["question"],
        "answer_leadin": job.get("answer_leadin") or "",
        "A": job.get("pred_display") or job.get("pred") or "",
        "pred": job.get("pred") or "",
        "reference_answer": reference_answer_of(job),
        "gold": job.get("gold"),
        "item_score": s,
    }
    if fam == "abstention":
        rec["should_abstain"] = bool(job.get("should_abstain"))
        rec["pred_abstain"] = is_abstention(job.get("pred") or "", lang)
    return rec


def write_auto_metric_jsons(jobs: list[dict], metrics: dict, out_dir: Path,
                            lang: str = "en") -> dict[str, str]:
    """pred/auto_metric_json/{muscle,value,direction,abstention}_set.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by: dict[str, list] = {"muscle_set": [], "value": [], "direction": [], "abstention": []}
    for j in jobs:
        if j["family"] in by:
            by[j["family"]].append(_family_item_record(j, lang=lang))

    payloads = {
        "muscle_set": {
            "family": "muscle_set",
            "metric_key": "Muscle F1",
            "summary": metrics.get("muscle_f1", {}),
            "headline": (metrics.get("headline") or {}).get("Muscle F1"),
            "items": by["muscle_set"],
        },
        "value_set": {
            "family": "value",
            "metric_key": "Value acc",
            "summary": metrics.get("value_acc", {}),
            "headline": (metrics.get("headline") or {}).get("Value acc"),
            "delta_shuf": metrics.get("delta_shuf"),
            "items": by["value"],
        },
        "direction_set": {
            "family": "direction",
            "metric_key": "Direction acc",
            "summary": metrics.get("direction_acc", {}),
            "headline": (metrics.get("headline") or {}).get("Direction acc"),
            "items": by["direction"],
        },
        "abstention_set": {
            "family": "abstention",
            "metric_key": "Abstention F1",
            "summary": metrics.get("abstention_f1", {}),
            "headline": (metrics.get("headline") or {}).get("Abstention F1"),
            # 파일에는 probe 의 abstention 양성만; summary F1 은 answerable 음성 포함
            "items": by["abstention"],
        },
    }
    paths = {}
    for name, payload in payloads.items():
        fp = out_dir / f"{name}.json"
        fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        paths[name] = str(fp)
    return paths


def write_gpt_judge_payloads(jobs: list[dict], out_dir: Path, *,
                             model_name: str = "") -> dict:
    """GPT judge 페이로드 — preds 와 reference 를 물리적으로 분리.

    pred/judge_metric_json/gpt/
      imgs_preds.json       — question + model_answer + images  (NO reference)
      mesh_preds.json       — question + model_answer + mesh_npy (NO reference)
      reference_answer.json — uid + reference_answer only (offline; GPT에 올리지 말 것)

    Join on uid / testdata_index after GPT returns verdicts.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs_items: list[dict] = []
    mesh_items: list[dict] = []
    ref_items: list[dict] = []

    for j in jobs:
        if j.get("family") != "utility":
            continue
        idx = testdata_index_of(j)
        mi = int(j["mesh_index"])
        answer = j.get("pred_display") or j.get("pred") or ""
        pred = j.get("pred") or ""
        base = {
            "testdata_index": idx,
            "uid": j["uid"],
            "model": model_name or None,
            "mesh_index": mi,
            "question": j["question"],
            "A": answer,
            "model_answer": pred,
            "rubric": "utility_open",
        }
        # preds — reference_answer 필드를 절대 넣지 않는다
        imgs_items.append({
            **base,
            "images": {
                "front": f"images/{mi}/front.png",
                "left": f"images/{mi}/left.png",
                "up": f"images/{mi}/up.png",
            },
            "front_png": f"DATA/mesh_png/front/{mi}.png",
        })
        mesh_items.append({
            **base,
            "mesh_npy": {
                "disp": f"meshes/{mi}/disp.npy",
                "rest": "meshes/_shared/rest.npy",
                "faces": "meshes/_shared/faces.npy",
            },
        })
        ref_items.append({
            "testdata_index": idx,
            "uid": j["uid"],
            "question": j["question"],
            "reference_answer": reference_answer_of(j),
        })

    paths = {
        "imgs_preds": out_dir / "imgs_preds.json",
        "mesh_preds": out_dir / "mesh_preds.json",
        "reference_answer": out_dir / "reference_answer.json",
    }
    payloads = {
        "imgs_preds": {
            "n": len(imgs_items),
            "note": "GPT upload OK. No reference_answer. Image-conditioned utility preds.",
            "items": imgs_items,
        },
        "mesh_preds": {
            "n": len(mesh_items),
            "note": "GPT upload OK. No reference_answer. Mesh-npy-conditioned utility preds.",
            "items": mesh_items,
        },
        "reference_answer": {
            "n": len(ref_items),
            "note": "OFFLINE ONLY — do NOT upload to GPT. Join on uid after judging.",
            "items": ref_items,
        },
    }
    out_paths = {}
    for key, fp in paths.items():
        fp.write_text(json.dumps(payloads[key], ensure_ascii=False, indent=2),
                      encoding="utf-8")
        out_paths[key] = str(fp)

    # 옛 mesh.json 이 남아 GPT에 같이 올라가는 일을 막는다
    legacy = out_dir / "mesh.json"
    if legacy.is_file():
        legacy.unlink()

    return {"paths": out_paths, "n": len(imgs_items)}


# 하위 호환 별칭
write_gpt_mesh_json = write_gpt_judge_payloads


class EvaluatorSet1Probe:
    """Prefer: ``EvaluatorSet1Probe(cfg.evaluators, experiment_cfg=cfg).run()``."""

    def __init__(self, cfg=None, experiment_cfg=None, **kwargs):
        p = _as_dict(cfg)
        p.update(kwargs)
        self.experiment_cfg = _as_conf(experiment_cfg)
        exp = self.experiment_cfg
        p.setdefault("seed", int(exp.get("seed", 42)))
        self.cfg = OmegaConf.create(p)
        self.output_dir = p.get("output_dir", None)
        self.device = str(exp.get("device", "cuda"))
        self.run_cfg = _as_conf(exp.get("run", {}))
        self.name = str(exp.get("name", ""))

    def load_model(self, ckpt_dir: Path):
        from models.stage2.model_s2 import Stage2Model

        exp = self.experiment_cfg
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        print(f"[eval] 모델 구성: {self.name}  ckpt={ckpt_dir}")
        model = Stage2Model(exp.models)

        lora_dir = ckpt_dir / "lora"
        bridge_fp = ckpt_dir / "mm_projector.pt"
        if lora_dir.is_dir():
            from peft import load_peft_weights, set_peft_model_state_dict
            set_peft_model_state_dict(model.llm, load_peft_weights(str(lora_dir)))
            print(f"[eval] LoRA 로드: {lora_dir}")
        elif self.run_cfg.get("require_weights", True):
            raise SystemExit(
                f"[eval] LoRA 없음: {lora_dir}\n"
                f"       experiment output_dir 에 checkpoint_best 가 있는지 확인하라.")
        else:
            print(f"[eval][경고] LoRA 없음 — 미학습 가중치로 평가")

        if bridge_fp.is_file():
            bridge = torch.load(bridge_fp, map_location="cpu", weights_only=False)
            if model.use_mesh and "fusion" in bridge:
                model.fusion.load_state_dict(bridge["fusion"])
            if getattr(model, "aux_muscle", False) and "muscle_head" in bridge:
                model.muscle_head.load_state_dict(bridge["muscle_head"])
            print(f"[eval] fusion 로드: {bridge_fp}")
        elif model.use_mesh:
            print(f"[eval][경고] fusion 없음: {bridge_fp}")

        model.to(device)
        if model.use_mesh:
            model.mesh_encoder.to(device)
        model.eval()
        return model, device

    def run(self):
        cfg = self.cfg
        exp = self.experiment_cfg
        run = self.run_cfg
        lang = str(cfg.get("lang", "en"))
        seed = int(cfg.seed)
        torch.manual_seed(seed)
        random.seed(seed)

        probe_path = Path(str(cfg.probe_path))
        if not probe_path.is_file():
            raise SystemExit(
                f"[eval] probe 데이터 없음: {probe_path}\n"
                f"       python tools/build_set1_probe.py --lang {lang} 를 먼저 실행하라.")

        ckpt_dir = resolve_best_checkpoint(exp)
        out_root = Path(self.output_dir or (Path(exp.output_dir) / "eval"))
        mesh_control = str(run.get("mesh_control", "real") or "real")
        if mesh_control not in MESH_CONTROLS:
            raise SystemExit(f"run.mesh_control 은 {sorted(MESH_CONTROLS)} 중 하나")
        if mesh_control != "real" and not self.output_dir:
            out_root = out_root.parent / f"{out_root.name}_mesh-{mesh_control}"
        out_root.mkdir(parents=True, exist_ok=True)

        score_only = bool(run.get("score_only", False))
        render_only = bool(run.get("render_only", False))
        limit = run.get("limit", None)
        do_delta = bool(cfg.get("delta_shuf", True))

        records = load_probe(probe_path)
        system = SYSTEM_PROMPT.get(lang, SYSTEM_PROMPT["en"])
        data_cfg = _as_conf(exp.get("datasets", {}))
        store = MeshStore()
        pred_root = out_root / "pred"
        pred_root.mkdir(parents=True, exist_ok=True)

        refs = load_reference_answers(cfg.get("reference_answers_path", None), lang)
        preds_fp = out_root / "preds.jsonl"
        jobs = jobs_from_probe(records, lang, system, limit=limit)
        attach_reference_answers(jobs, refs)

        model = tok = device = None
        need_gen = not (score_only or render_only)
        if need_gen:
            model, device = self.load_model(ckpt_dir)
            tok = load_tokenizer(str(exp.models.llm))
            jobs = apply_mesh_control(jobs, mesh_control, seed)
            jobs = generate_cloze(model, tok, store, jobs, cfg, device, data_cfg,
                                  exp_models=exp.get("models", {}))
            write_jsonl(preds_fp, [{
                "uid": j["uid"], "family": j["family"], "index": j["index"],
                "n_act_bin": j.get("n_act_bin"), "question": j["question"],
                "answer_leadin": j.get("answer_leadin"),
                "pred": j["pred"], "pred_display": j.get("pred_display"),
                "gold": j.get("gold"), "should_abstain": j.get("should_abstain"),
            } for j in jobs])
        else:
            if not preds_fp.is_file():
                raise SystemExit(f"preds.jsonl 없음: {preds_fp}")
            saved = {r["uid"]: r for r in _load_jsonl(preds_fp)}
            for j in jobs:
                s = saved.get(j["uid"], {})
                j["pred"] = s.get("pred", "")
                j["pred_display"] = s.get("pred_display", j["pred"])

        # Δ_shuf: muscle_set family 를 shuffle mesh 로 한 번 더
        # (headline = MuscleF1_real − MuscleF1_shuf). legacy value shuf 파일도 읽음.
        shuf_jobs = None
        if do_delta and need_gen and mesh_control == "real":
            mus_jobs = [dict(j) for j in jobs if j["family"] == "muscle_set"]
            if mus_jobs:
                print(f"\n[Δ_shuf] muscle_set {len(mus_jobs)}개 shuffle 재추론")
                mus_jobs = apply_mesh_control(mus_jobs, "shuffle", seed + 1)
                mus_jobs = generate_cloze(model, tok, store, mus_jobs, cfg, device,
                                          data_cfg, exp_models=exp.get("models", {}))
                for j in mus_jobs:
                    j["pred_shuf"] = j["pred"]
                shuf_jobs = mus_jobs
                write_jsonl(out_root / "preds_muscle_shuf.jsonl", [{
                    "uid": j["uid"], "index": j["index"], "pred_shuf": j["pred"],
                    "mesh_indices_eff": j.get("mesh_indices_eff"),
                } for j in mus_jobs])
        elif do_delta and (out_root / "preds_muscle_shuf.jsonl").is_file():
            shuf_map = {r["uid"]: r["pred_shuf"]
                        for r in _load_jsonl(out_root / "preds_muscle_shuf.jsonl")}
            shuf_jobs = []
            for j in jobs:
                if j["family"] == "muscle_set":
                    jj = dict(j)
                    jj["pred_shuf"] = shuf_map.get(j["uid"], "")
                    shuf_jobs.append(jj)
        elif do_delta and (out_root / "preds_value_shuf.jsonl").is_file():
            # legacy: older evals that only stored value shuf
            shuf_map = {r["uid"]: r["pred_shuf"]
                        for r in _load_jsonl(out_root / "preds_value_shuf.jsonl")}
            shuf_jobs = []
            for j in jobs:
                if j["family"] == "value":
                    jj = dict(j)
                    jj["pred_shuf"] = shuf_map.get(j["uid"], "")
                    shuf_jobs.append(jj)

        # Value |pred−gold| threshold — configs/evaluators/set1_probe.yaml → value:
        #   source: gold|config · tol_mm · tol_norm · per_quantity
        # Legacy flat value_tol still accepted as tol_mm/default.
        value_cfg = _as_dict(cfg.get("value", {}))
        if cfg.get("value_tol") is not None and "tol_mm" not in value_cfg:
            value_cfg.setdefault("tol_mm", float(cfg.value_tol))
            value_cfg.setdefault("default_tol", float(cfg.value_tol))
        value_tol_cfg = M.resolve_value_tol_cfg(value_cfg or None)
        metrics = M.score_all(jobs, lang=lang, shuf_jobs=shuf_jobs,
                              value_tol=value_tol_cfg)

        auto_paths = write_auto_metric_jsons(
            jobs, metrics, pred_root / "auto_metric_json", lang=lang)
        gpt_info = write_gpt_judge_payloads(
            jobs, pred_root / "judge_metric_json" / "gpt",
            model_name=self.name)
        # 하위 호환: 옛 judge_payloads.jsonl 도 남김
        util_payload = M.build_utility_judge_payload(
            jobs, out_root / "judge_payloads.jsonl")
        metrics["utility"] = {
            **metrics.get("utility", {}),
            **util_payload,
            "gpt_judge": gpt_info,
        }
        metrics["_meta"] = {
            "experiment": self.name,
            "checkpoint": str(ckpt_dir),
            "probe_path": str(probe_path),
            "lang": lang,
            "n_jobs": len(jobs),
            "mesh_control": mesh_control,
            "pred_root": str(pred_root),
            "auto_metric_json": auto_paths,
            "value_tol": value_tol_cfg,
            "generation": OmegaConf.to_container(cfg.generation, resolve=True),
        }
        (out_root / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_root / "summary.json").write_text(
            json.dumps(metrics.get("headline", {}), ensure_ascii=False, indent=2),
            encoding="utf-8")
        (pred_root / "summary.json").write_text(
            json.dumps(metrics.get("headline", {}), ensure_ascii=False, indent=2),
            encoding="utf-8")

        print("\n=== Set-1 Probe metrics ===")
        print(json.dumps(metrics.get("headline", {}), ensure_ascii=False, indent=2))
        print(json.dumps({k: metrics[k] for k in
                          ("muscle_f1", "value_acc", "direction_acc",
                           "abstention_f1", "delta_shuf") if k in metrics},
                         ensure_ascii=False, indent=2)[:2000])

        # 렌더 (score_only 여부와 무관 — preds 가 있으면 그림)
        if bool(cfg.get("render", {}).get("enabled", True)):
            self._render_all(jobs, store, out_root, pred_root, cfg)

        print(f"\n=== 완료 → {out_root} ===")
        print(f"    pred/render/ · pred/auto_metric_json/ · pred/judge_metric_json/")
        return metrics

    def _render_all(self, jobs, store, out_root: Path, pred_root: Path, cfg):
        rcfg = _as_dict(cfg.get("render", {}))
        max_items = int(rcfg.get("max_items", 0) or 0)  # 0 = all
        front_root = Path(str(rcfg.get("front_png_root") or
                              cfg.get("front_png_root") or FRONT_PNG_ROOT))
        png_dir = pred_root / "render"
        human_dir = pred_root / "judge_metric_json" / "human"
        entries = []
        sel = jobs if not max_items else jobs[:max_items]
        print(f"[render] {len(sel)}개 → {png_dir}")
        print(f"[render/human] utility → {human_dir}")
        for k, j in enumerate(sel):
            idx = testdata_index_of(j)
            answer = j.get("pred_display") or j.get("pred") or ""
            ref = reference_answer_of(j)
            title = f"{idx}  n_act={j.get('n_act_bin')}  {j['family']}"
            png = png_dir / f"{idx}.png"
            render_front_qa(
                int(j["mesh_index"]),
                question=j["question"],
                answer=answer,
                reference_answer=ref,
                out_png=png,
                title=title,
                front_png_root=front_root,
                store=store,
            )
            if j.get("family") == "utility":
                render_front_qa(
                    int(j["mesh_index"]),
                    question=j["question"],
                    answer=answer,
                    reference_answer=ref,
                    out_png=human_dir / f"{idx}.png",
                    title=f"[human] {idx}",
                    front_png_root=front_root,
                    store=store,
                )
            entries.append({
                "index": idx, "png": f"pred/render/{png.name}",
                "section": j["family"], "target": j.get("n_act_bin", ""),
                "question": j["question"],
                "pred": answer,
                "should_increase": [], "should_decrease": [],
            })
            if (k + 1) % 25 == 0:
                print(f"  렌더 {k + 1}/{len(sel)}", flush=True)
        build_index(entries, out_root / "index.html",
                    f"Set-1 Probe — {len(sel)} items")
        write_jsonl(out_root / "item_scores.jsonl", [
            {"uid": j["uid"], "testdata_index": testdata_index_of(j),
             "family": j["family"], "index": j["index"],
             "item_score": j.get("item_score"), "pred": j.get("pred"),
             "reference_answer": reference_answer_of(j)}
            for j in jobs
        ])


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
