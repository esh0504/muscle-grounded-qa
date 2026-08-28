"""Stage-1 회귀 헤드 평가 — Set-1 의 500 mesh 에서 Muscle-EM · Activation MAE.

Stage-2 표(`Baseline.md` §6)는 LLM 이 **문장으로** 답한 것을 채점한다. 이 평가기는 같은
mesh 집합에서 Stage-1 모델이 **회귀 헤드로 바로 뱉은 11-D 활성**을 채점한다. LLM·프롬프트·
디코딩이 끼지 않으므로, "표현 자체가 근육 정보를 얼마나 담는가"만 남는다.

    row            입력                  모델                       체크포인트
    ─────────────  ────────────────────  ─────────────────────────  ───────────────────────
    mesh3d         mesh 변위 (370, 3)    Stage1Model (SpiralNet++)      outputs/stage1
    zeros          —                     전부 0 (trivial 기준선)    —
    trainmean      —                     train 평균 활성 (기준선)   —

평가 mesh 는 `DATA/unseentest/set1_probe.jsonl` 의 `family=muscle_set` 500개 —
Stage-2 표와 **같은 mesh** 다. gold 는 probe 와 같은 규칙(`a_i > 1e-4`)으로 원본 CSV 에서
다시 만든다.

**임계값은 val split 에서 고른다.** 활성 집합을 만들려면 회귀 출력을 이진화해야 하는데,
그 임계값을 test 500개에서 고르면 수치가 부풀려진다. `threshold.mode=val` (기본) 이면
val 에서 `n_select` 개를 뽑아 Muscle-EM 을 최대로 하는 값을 고르고 그대로 test 에 쓴다.

⚠️ **probe 500 은 `n_act` 균형 표본이다** (`build_set1_probe.stratified_sample`). 실제 test
split 은 3+ 가 79% 라 probe 수치가 held-out 성능보다 높게 나온다. 그래서 같은 임계값으로
test split **전체**를 한 번 더 재서 (`full_test`) 두 표를 같이 낸다.

    python eval_s1.py
    python eval_s1.py evaluators.rows='[mesh3d]'          # 3D 만
    python eval_s1.py evaluators.full_test.enabled=false  # probe 500 만 (빠름)
    python eval_s1.py evaluators.threshold.mode=fixed evaluators.threshold.value=0.05

산출물 (`{output_dir}/`):
    metrics.json   행별 headline + 분해 + 임계값 sweep + full_test + compare
    preds.jsonl    mesh 하나당 예측/gold 11-D 벡터와 활성 집합
    summary.md     논문에 붙일 표 (probe 500 · test split 전체 · 유의성)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from datasets.mesh_dataset import MUSCLE_NAMES, SHARD_SIZE
from datasets.mesh_store import MeshStore
from metrics import muscle_regression as mr
from models import find_model_def

MUSCLE_CSV = "pool_meta.csv"


def _as_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    if OmegaConf.is_config(cfg):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {}


def _as_conf(cfg: Any) -> DictConfig:
    if cfg is None:
        return OmegaConf.create({})
    return cfg if OmegaConf.is_config(cfg) else OmegaConf.create(_as_dict(cfg))


# --------------------------------------------------------------------------- #
# 데이터
# --------------------------------------------------------------------------- #
def load_probe_meshes(path: str | Path, family: str = "muscle_set") -> list[dict]:
    """probe jsonl → mesh 하나당 한 항목 (`mesh_index`, `n_act`, gold 근육 이름).

    `muscle_set` family 가 mesh 당 정확히 하나라서 이게 곧 평가 mesh 목록이다.
    """
    rows: list[dict] = []
    seen: set[int] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("family") != family or rec.get("mesh_index") is None:
            continue
        gidx = int(rec["mesh_index"])
        if gidx in seen:
            continue
        seen.add(gidx)
        rows.append({
            "uid": rec.get("uid", f"m{gidx:06d}"),
            "mesh_index": gidx,
            "n_act": int(rec.get("n_act", 0)),
            "n_act_bin": str(rec.get("n_act_bin", "")),
            "gold_muscles": list((rec.get("gold") or {}).get("muscles") or []),
        })
    if not rows:
        raise RuntimeError(f"{path} 에 family={family!r} 항목이 없다")
    return rows


_ACT_TABLE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def load_activation_table(mesh_root: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """CSV 전체를 (N, 11) 배열 + 존재 마스크로 한 번만 읽는다 (전역 인덱스로 바로 색인).

    probe · val · test 를 각각 훑으면 295k 행짜리 CSV 를 여러 번 파싱하게 된다.
    13 MB 짜리 배열이라 통째로 들고 있는 편이 싸다. 열 순서는 `MUSCLE_NAMES` 로 고정.
    """
    key = str(Path(mesh_root).resolve())
    cached = _ACT_TABLE.get(key)
    if cached is not None:
        return cached

    rows: list[tuple[int, list[float]]] = []
    with (Path(mesh_root) / MUSCLE_CSV).open() as fh:
        for row in csv.DictReader(fh):
            rows.append((int(row["index"]), [float(row[m]) for m in MUSCLE_NAMES]))
    n = max(i for i, _ in rows) + 1
    table = np.zeros((n, len(MUSCLE_NAMES)), dtype=np.float32)
    have = np.zeros(n, dtype=bool)
    for gidx, vals in rows:
        table[gidx] = vals
        have[gidx] = True
    _ACT_TABLE[key] = (table, have)
    return table, have


def load_activations(mesh_root: str | Path, indices) -> np.ndarray:
    """(N, 11) 활성. 없는 인덱스는 0 으로 조용히 채우지 않고 죽는다."""
    table, have = load_activation_table(mesh_root)
    ids = np.asarray([int(i) for i in indices], dtype=np.int64)
    bad = ids[(ids < 0) | (ids >= len(table)) | ~have[np.clip(ids, 0, len(table) - 1)]]
    if len(bad):
        raise RuntimeError(f"{MUSCLE_CSV} 에 활성이 없는 인덱스 {len(bad)}개 "
                           f"(예: {bad[:5].tolist()})")
    return table[ids]


def load_split_meshes(mesh_root: str | Path, split: str, n: int | None, seed: int) -> list[int]:
    """split 파일에서 인덱스를 n 개 뽑는다 (n=None/0 이면 전부)."""
    path = Path(mesh_root) / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"split 파일이 없다: {path}")
    ids = np.asarray([int(x) for x in path.read_text().split() if x.strip()], dtype=np.int64)
    if n and n < len(ids):
        ids = ids[np.random.default_rng(seed).permutation(len(ids))[:n]]
    return sorted(int(i) for i in ids)


# --------------------------------------------------------------------------- #
# 입력 공급 — Stage-1 학습 때와 같은 전처리여야 한다
# --------------------------------------------------------------------------- #
class MeshInput:
    """전역 인덱스 → mesh 변위 (370, 3). `MeshDataset.__getitem__["inputs"]` 와 동일."""

    kind = "mesh"

    def __init__(self, mesh_root: str | Path, **_):
        self.store = MeshStore(Path(mesh_root))
        # 마지막 shard 는 157행뿐이라 shard 수 × 1000 으로 잡으면 없는 인덱스를 있다고 한다.
        # 파일 크기에서 실제 행 수를 센다 (행 하나 = n_verts × 3 × 4 byte).
        row_bytes = self.store.rest.shape[0] * 3 * 4
        self.rows_in_shard = {
            int(p.stem.split("_")[1]): p.stat().st_size // row_bytes
            for p in (Path(mesh_root) / "verts").glob("shard_*.bin")
            if not p.name.endswith(".part")
        }

    def available(self, index: int) -> bool:
        shard, local = divmod(int(index), SHARD_SIZE)
        return 0 <= int(index) and local < self.rows_in_shard.get(shard, 0)

    def batch(self, indices) -> torch.Tensor:
        return torch.from_numpy(
            np.stack([self.store.disp(int(i)) for i in indices]).astype(np.float32))


def build_input(kind: str, cfg) -> MeshInput:
    data = _as_dict(cfg)
    if kind == "mesh":
        return MeshInput(**data)
    raise ValueError(f"input must be 'mesh', got {kind!r}")


# --------------------------------------------------------------------------- #
# 예측
# --------------------------------------------------------------------------- #
def load_stage1_model(model_cfg_path: str | Path, ckpt_path: str | Path,
                      device: torch.device) -> torch.nn.Module:
    """모델 yaml + 체크포인트 → eval 모드 모델. 가중치는 strict 로 싣는다.

    strict 를 푸는 순간 헤드가 무작위인 채로 점수가 나올 수 있다. 구조가 안 맞으면
    yaml 과 체크포인트 짝이 틀린 것이므로 여기서 죽는 게 맞다.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Stage-1 체크포인트가 없다: {ckpt_path}")

    model_cfg = OmegaConf.load(str(model_cfg_path))
    ModelClass = find_model_def(model_cfg.name, model_cfg.class_name)
    model = ModelClass(model_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    print(f"[stage1] {type(model).__name__} ← {ckpt_path} (epoch={epoch})")
    return model


@torch.no_grad()
def predict(model: torch.nn.Module, source, indices, device: torch.device,
            batch_size: int = 64) -> np.ndarray:
    """(N, 11) 예측 활성. dropout 이 꺼진 eval 모드에서만 부른다."""
    out = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        x = source.batch(chunk).to(device)
        out.append(model(x).float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, len(MUSCLE_NAMES)), np.float32)


# --------------------------------------------------------------------------- #
# 표
# --------------------------------------------------------------------------- #
def summary_markdown(rows: list[dict], n: int, probe_path: str,
                     compare: dict | None = None) -> str:
    head = ("| Row | Muscle-EM | Activation MAE | Muscle F1 | P | R | thr | thr src |\n"
            "|---|---:|---:|---:|---:|---:|---:|:--:|\n")
    body = "".join(
        f"| {r['label']} | **{r['muscle_em'] * 100:.1f}** | **{r['activation_mae']:.4f}** | "
        f"{r['muscle_f1'] * 100:.1f} | {r['em']['precision'] * 100:.1f} | "
        f"{r['em']['recall'] * 100:.1f} | {r['threshold']:.4g} | {r['threshold_source']} |\n"
        for r in rows)

    diag_head = ("| Row | MAE(active) | MAE(inactive) | pred n_act | dead heads | "
                 "EM ceiling (dead) | EM on reachable | EM@oracle thr | scorer ceiling |\n"
                 "|---|---:|---:|---:|---|---:|---:|---:|---:|\n")
    diag = "".join(
        f"| {r['label']} | {r['mae']['mae_active']:.4f} | {r['mae']['mae_inactive']:.4f} | "
        f"{r['em']['pred_n_act_mean']:.2f} | "
        f"{', '.join(r['dead_heads']['dead_muscles']) or '—'} | "
        f"{r['dead_heads']['em_ceiling'] * 100:.1f} | "
        f"{r['dead_heads']['muscle_em_reachable'] * 100:.1f} | "
        f"{r['oracle_sweep']['muscle_em'] * 100:.1f} | "
        f"{r['scorer_ceiling']['muscle_em'] * 100:.1f} |\n"
        for r in rows)

    per_n = "| Row | " + " | ".join(f"EM n_act={k}" for k in ("1", "2", "3+")) + " |\n"
    per_n += "|---|" + "---:|" * 3 + "\n"
    for r in rows:
        cells = [f"{r['em']['per_n_act'].get(k, 0.0) * 100:.1f}" for k in ("1", "2", "3+")]
        per_n += f"| {r['label']} | " + " | ".join(cells) + " |\n"

    ft_rows = [r for r in rows if r.get("full_test")]
    full = ""
    if ft_rows:
        f0 = ft_rows[0]["full_test"]
        full = (f"## Held-out split 전체 (`{f0['split']}`, n = {f0['n']:,})\n\n"
                "| Row | Muscle-EM | Activation MAE | Muscle F1 | P | R | thr |\n"
                "|---|---:|---:|---:|---:|---:|---:|\n")
        full += "".join(
            f"| {r['label']} | {r['full_test']['muscle_em'] * 100:.1f} | "
            f"{r['full_test']['activation_mae']:.4f} | "
            f"{r['full_test']['muscle_f1'] * 100:.1f} | "
            f"{r['full_test']['precision'] * 100:.1f} | "
            f"{r['full_test']['recall'] * 100:.1f} | {r['full_test']['threshold']:.4g} |\n"
            for r in ft_rows)
        full += (f"\n평균 n_act = {f0['gold_n_act_mean']:.2f} (probe 는 균형 표본이라 "
                 f"{rows[0]['em']['gold_n_act_mean']:.2f}).\n"
                 f"**\"held-out 에서 얼마?\"의 답은 이 표다.** 위 headline 은 Stage-2 표와\n"
                 f"mesh 를 맞추기 위한 균형 표본 수치라 더 높게 나온다.\n\n")

    cmp_md = ""
    if compare:
        c = compare
        em, mae = c["muscle_em"], c["activation_mae"]
        sig = "유의하지 않다" if em["mcnemar_exact_p"] >= 0.05 else "유의하다"
        cmp_md = (
            f"## {c['a']} vs {c['b']} — 격차가 잡음인가 (probe {c['n']})\n\n"
            f"| 지표 | Δ (a − b) | 95% CI | 검정 |\n|---|---:|---|---|\n"
            f"| Muscle-EM | {em['delta'] * 100:+.1f} pt | "
            f"[{em['ci95'][0] * 100:+.1f}, {em['ci95'][1] * 100:+.1f}] pt | "
            f"McNemar exact p = **{em['mcnemar_exact_p']:.3f}** |\n"
            f"| Activation MAE | {mae['delta']:+.4f} | "
            f"[{mae['ci95'][0]:+.4f}, {mae['ci95'][1]:+.4f}] | "
            f"쌍 부트스트랩 ({c['boot']['n_boot']:,}회) |\n\n"
            f"불일치 쌍: {c['a']} 만 맞힌 mesh {em['a_only']}개 / {c['b']} 만 "
            f"{em['b_only']}개 (둘 다 {em['both']}개). MAE 는 {c['a']} 가 "
            f"{mae['a_better_frac'] * 100:.0f}% 의 mesh 에서 더 낮다.\n\n"
            f"> **Muscle-EM 격차는 {sig}** (p = {em['mcnemar_exact_p']:.3f}). CI 가 0 을\n"
            f"> 걸치면 \"3D 가 낫다\"를 결론으로 쓰지 마라. MAE 쪽 CI 는 별개로 읽어라.\n\n")

    gold_nact = rows[0]["em"]["gold_n_act_mean"] if rows else 0.0
    return (
        f"# Stage-1 회귀 헤드 — Muscle-EM · Activation MAE\n\n"
        f"probe: `{probe_path}` · n = **{n} mesh** (test split, held-out in-distribution)\n"
        f"⚠️ 이 500개는 `n_act` 를 1/2/3+ 로 **균형 맞춰 뽑은** 표본이다 "
        f"(`scripts/build_set1_probe.stratified_sample`).\n"
        f"실제 test split 은 3+ 가 79% 라, EM 이 `n_act` 와 함께 떨어지는 이 과제에서\n"
        f"아래 headline 은 held-out split 성능보다 **높게** 나온다 — 두 표를 같이 봐라.\n\n"
        f"gold 활성 집합: `a_i > {mr.ACTIVE_EPS}` (probe 빌더와 동일) · gold 평균 n_act = "
        f"{gold_nact:.2f}\n\n"
        f"## Headline — Set-1 probe 500 (n_act 균형 표본, Stage-2 표와 같은 mesh)\n\n"
        f"{head}{body}\n"
        f"- **Muscle-EM** = 예측 활성 집합이 gold 와 **정확히** 같은 mesh 비율 (%). 근육\n"
        f"  하나만 어긋나도 0 이라 Muscle F1 보다 훨씬 엄격하다.\n"
        f"- **Activation MAE** = 11-D 활성의 평균 절대오차 (임계값과 무관, 낮을수록 좋음).\n"
        f"- **thr** = 회귀 출력 → 활성 집합 이진화 임계값. 모델 행(`thr src=val`)은 val\n"
        f"  split 에서 골라 여기 적용했다. 기준선 행(`oracle(baseline)`)은 예측이 상수라\n"
        f"  임계값이 \"몇 개를 켤까\"만 정하므로 이 500개 위에서 고른다 — 기준선에 유리한\n"
        f"  쪽이고, 그러고도 EM 은 0.0 / 2.0 이다.\n\n"
        f"{full}"
        f"{cmp_md}"
        f"## 진단 (probe 500)\n\n{diag_head}{diag}\n"
        f"- **dead heads** = 이 500 mesh 어디서도 `thr` 을 못 넘는 출력 차원. Stage-1 은\n"
        f"  L1 로 학습하는데(`losses/loss_s1.py`) L1 최적해가 조건부 **중앙값**이라, 절반\n"
        f"  넘게 0 인 근육은 헤드가 통째로 0 으로 눌린다.\n"
        f"- **EM ceiling (dead)** = 죽은 근육이 gold 에 없어 EM 이 가능한 mesh 비율.\n"
        f"  **EM on reachable** = 그 mesh 들만 놓고 잰 EM. headline 을 이 둘과 같이 읽어라.\n"
        f"- **scorer ceiling** = gold 를 그대로 예측해도 `thr` 이진화 때문에 깎이는 상한\n"
        f"  (활성값이 0.0002 까지 내려간다). **EM@oracle thr** = 이 500개에 임계값을\n"
        f"  맞췄을 때의 상한 — 진단용이지 보고 수치가 아니다.\n\n"
        f"## per n_act (probe 500, 재계산 `a > {mr.ACTIVE_EPS}` 기준)\n\n{per_n}\n"
        f"> probe jsonl 의 `n_act` 필드는 CSV `n_active`(= `a >= 0.05` 개수)를 복사한\n"
        f"> 값이라 여기서 다시 셌다. `gold.muscles` 자체는 두 기준이 같다.\n"
    )


# --------------------------------------------------------------------------- #
class EvaluatorStage1Muscle:
    """Prefer: ``EvaluatorStage1Muscle(cfg.evaluators, experiment_cfg=cfg).run()``."""

    def __init__(self, cfg=None, experiment_cfg=None, **kwargs):
        p = _as_dict(cfg)
        p.update(kwargs)
        self.experiment_cfg = _as_conf(experiment_cfg)
        exp = self.experiment_cfg
        p.setdefault("seed", int(exp.get("seed", 42)))
        self.cfg = OmegaConf.create(p)
        self.device = str(exp.get("device", "cuda"))
        self.run_cfg = _as_conf(exp.get("run", {}))
        self.name = str(exp.get("name", "stage1_muscle"))

    # ---------------------------------------------------------------- #
    def _rows(self) -> list[dict]:
        """`evaluators.rows` 가 id 목록이면 `evaluators.model_rows` 에서 골라 온다."""
        table = {str(r["id"]): dict(r) for r in _as_dict(self.cfg).get("model_rows", [])}
        want = self.cfg.get("rows", None)
        ids = [str(x) for x in want] if want else list(table)
        missing = [i for i in ids if i not in table]
        if missing:
            raise KeyError(f"model_rows 에 없는 row id {missing} (가능: {list(table)})")
        return [table[i] for i in ids]

    def _threshold(self, row: dict, source, model, device) -> tuple[float, str, dict]:
        """이진화 임계값. 기본은 **val split 에서 선택** (test 에서 고르지 않는다)."""
        thr_cfg = _as_dict(self.cfg.get("threshold", {}))
        mode = str(thr_cfg.get("mode", "val"))
        if mode == "fixed":
            return float(thr_cfg.get("value", 0.05)), "fixed", {}
        if mode != "val":
            raise ValueError(f"threshold.mode must be val|fixed, got {mode!r}")

        mesh_root = self.cfg.data.mesh_root
        ids = load_split_meshes(mesh_root, str(thr_cfg.get("split", "val")),
                                int(thr_cfg.get("n_select", 4000)), int(self.cfg.seed))
        ids = [i for i in ids if source.available(i)]
        if not ids:
            raise RuntimeError(f"[{row['id']}] 임계값 선택용 val 표본이 비었다")
        gold = load_activations(mesh_root, ids)
        pred = predict(model, source, ids, device, int(self.cfg.get("batch_size", 64)))
        sel = mr.select_threshold(pred, gold, objective=str(thr_cfg.get("objective",
                                                                       "muscle_em")))
        print(f"[stage1] {row['id']}: val n={len(ids)} → thr={sel['threshold']:.4g} "
              f"(val EM={sel['best']['muscle_em']:.3f})")
        return float(sel["threshold"]), "val", sel

    def _baseline_row(self, row: dict, n: int) -> np.ndarray:
        """학습 없는 기준선 (`SET1_METRICS_DETAIL.md` §8 majority baseline 에 대응).

        `zeros`  — 전부 0. gold 의 77% 가 0 이라 **MAE 만 보면 이게 꽤 좋다**. 학습된 행이
                   이걸 못 이기면 Activation MAE 는 아무 말도 하지 않는 것이다.
        `mean`   — train split 평균 활성을 모든 mesh 에 그대로 찍는다.
        """
        kind = str(row.get("baseline", "mean"))
        if kind == "zeros":
            return np.zeros((n, len(MUSCLE_NAMES)), dtype=np.float32)
        if kind != "mean":
            raise ValueError(f"baseline must be zeros|mean, got {kind!r}")
        ids = load_split_meshes(self.cfg.data.mesh_root, str(row.get("split", "train")),
                                int(row.get("n_fit", 20000)), int(self.cfg.seed))
        return np.tile(load_activations(self.cfg.data.mesh_root, ids).mean(axis=0), (n, 1))

    def _full_test(self, row: dict, thr: float, source, model, device) -> dict | None:
        """probe 500 개와 **같은 임계값**으로 test split 전체를 다시 잰다.

        probe 는 `n_act` 를 1/2/3+ 로 균형 맞춰 뽑은 세트다 (`build_set1_probe.stratified_sample`).
        실제 test split 은 3+ 가 79% 라, EM 이 `n_act` 와 함께 떨어지는 이 과제에서 probe
        수치는 **held-out split 성능보다 높게 나온다**. Stage-2 표와 mesh 를 맞추려면 probe
        가 맞지만, "held-out 에서 얼마?"의 답은 이 블록이다. 둘 다 보고한다.
        """
        cfg = _as_dict(self.cfg.get("full_test", {}))
        if not cfg.get("enabled", True):
            return None
        mesh_root = self.cfg.data.mesh_root
        ids = load_split_meshes(mesh_root, str(cfg.get("split", "test")),
                                cfg.get("n", None), int(self.cfg.seed))
        if source is not None:
            ids = [i for i in ids if source.available(i)]
        gold = load_activations(mesh_root, ids)
        pred = (self._baseline_row(row, len(ids)) if model is None
                else predict(model, source, ids, device, int(self.cfg.get("batch_size", 64))))
        em = mr.score_muscle_em(pred, gold, thr, n_act=mr.n_act_bins(gold))
        mae = mr.score_activation_mae(pred, gold)
        return {"split": str(cfg.get("split", "test")), "n": len(ids),
                "threshold": float(thr),
                "muscle_em": em["muscle_em"], "activation_mae": mae["mae"],
                "muscle_f1": em["muscle_f1"], "precision": em["precision"],
                "recall": em["recall"], "gold_n_act_mean": em["gold_n_act_mean"],
                "per_n_act": em["per_n_act"], "mae_active": mae["mae_active"]}

    # ---------------------------------------------------------------- #
    def run(self):
        cfg = self.cfg
        torch.manual_seed(int(cfg.seed))
        np.random.seed(int(cfg.seed))
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        probe = load_probe_meshes(cfg.probe_path, str(cfg.get("family", "muscle_set")))
        if cfg.get("limit", None):
            probe = probe[:int(cfg.limit)]
        idx = [r["mesh_index"] for r in probe]
        gold = load_activations(cfg.data.mesh_root, idx)
        # probe 의 n_act 필드는 (a >= 0.05) 개수라 ACTIVE_EPS 기준과 다르다 → 다시 센다.
        n_act_bin = mr.n_act_bins(gold)
        n_rebinned = sum(1 for r, b in zip(probe, n_act_bin) if r["n_act_bin"] != b)
        print(f"[stage1] probe mesh {len(idx)}개 · gold 활성 (N, 11) = {gold.shape}")
        if n_rebinned:
            print(f"[stage1] n_act_bin 재계산: probe 필드와 다른 mesh {n_rebinned}개 "
                  f"(probe 는 a>=0.05 로 셌다 — gold 집합 자체는 그대로다)")

        # gold 집합이 probe 가 구운 것과 같은지 대조 — 다르면 CSV/probe 가 어긋난 것이다.
        for rec, g in zip(probe, gold):
            got = set(mr.active_names(g > mr.ACTIVE_EPS))
            if got != set(rec["gold_muscles"]):
                raise RuntimeError(
                    f"gold 불일치 {rec['uid']}: CSV={sorted(got)} probe={sorted(rec['gold_muscles'])}")

        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        scored: list[dict] = []
        preds_by_row: dict[str, np.ndarray] = {}
        for row in self._rows():
            if row.get("input") == "constant":
                pred = self._baseline_row(row, len(idx))
                # 모든 mesh 에 같은 값을 찍으므로 임계값은 "몇 개를 켤까"만 정한다. val 에서
                # 골라도 test 에서 골라도 같은 계단 함수라, 상한 그대로가 majority baseline
                # 의 정의다 (SET1_METRICS_DETAIL.md §8).
                thr, thr_src = mr.select_threshold(pred, gold)["threshold"], "oracle(baseline)"
                thr_info, source, model = {}, None, None
            else:
                source = build_input(str(row["input"]), cfg.data)
                missing = [i for i in idx if not source.available(i)]
                if missing:
                    raise RuntimeError(
                        f"[{row['id']}] 입력이 없는 probe mesh {len(missing)}개 "
                        f"(예: {missing[:5]}). 500개가 모두 있어야 행끼리 비교된다.")
                model = load_stage1_model(row["model_cfg"], row["checkpoint"], device)
                thr, thr_src, thr_info = self._threshold(row, source, model, device)
                pred = predict(model, source, idx, device, int(cfg.get("batch_size", 64)))

            s = mr.score_all(pred, gold, thr=thr, n_act=n_act_bin)
            s.update({"id": row["id"], "label": row.get("label", row["id"]),
                      "input": row.get("input"), "checkpoint": row.get("checkpoint"),
                      "threshold_source": thr_src,
                      "threshold_selection": {k: v for k, v in thr_info.items()
                                              if k != "curve"},
                      "full_test": self._full_test(row, thr, source, model, device)})
            scored.append(s)
            preds_by_row[row["id"]] = pred
            ft = s["full_test"]
            print(f"[stage1] {s['label']:<26} Muscle-EM={s['muscle_em'] * 100:5.1f}  "
                  f"MAE={s['activation_mae']:.4f}  F1={s['muscle_f1'] * 100:5.1f}"
                  + (f"   | test split 전체(n={ft['n']}): EM={ft['muscle_em'] * 100:5.1f}  "
                     f"MAE={ft['activation_mae']:.4f}" if ft else ""))

        # 학습된 행이 둘 이상이면 앞의 두 행 격차에 유의성을 붙인다. 같은 500 mesh 를
        # 본 쌍 표본이라 McNemar 가 성립한다.
        model_rows = [s for s in scored if s["input"] != "constant"]
        compare = (mr.compare_rows(model_rows[0], model_rows[1], seed=int(cfg.seed))
                   if len(model_rows) >= 2 else None)
        if compare:
            em = compare["muscle_em"]
            print(f"[stage1] {compare['a']} vs {compare['b']}: "
                  f"ΔEM={em['delta'] * 100:+.1f} pt "
                  f"CI95=[{em['ci95'][0] * 100:+.1f}, {em['ci95'][1] * 100:+.1f}] "
                  f"McNemar p={em['mcnemar_exact_p']:.3f}")

        metrics = {
            "probe_path": str(cfg.probe_path),
            "n": len(idx), "seed": int(cfg.seed),
            "gold_eps": mr.ACTIVE_EPS,
            "muscles": MUSCLE_NAMES,
            "threshold": _as_dict(cfg.get("threshold", {})),
            "rows": scored,
            "compare": compare,
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

        with (out_dir / "preds.jsonl").open("w", encoding="utf-8") as fh:
            for i, rec in enumerate(probe):
                item = {"uid": rec["uid"], "mesh_index": rec["mesh_index"],
                        "n_act": rec["n_act"], "n_act_bin": rec["n_act_bin"],
                        "gold": {"activations": [round(float(v), 6) for v in gold[i]],
                                 "muscles": sorted(rec["gold_muscles"])},
                        "pred": {}}
                for s in scored:
                    p = preds_by_row[s["id"]][i]
                    item["pred"][s["id"]] = {
                        "activations": [round(float(v), 6) for v in p],
                        "muscles": sorted(mr.active_names(p > s["threshold"])),
                        "exact": bool(s["em"]["exact"][i]),
                    }
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

        summary = summary_markdown(scored, len(idx), str(cfg.probe_path), compare)
        (out_dir / "summary.md").write_text(summary, encoding="utf-8")

        print(f"\n{summary}")
        print(f"→ {out_dir}/  (metrics.json · preds.jsonl · summary.md)")
        return metrics
