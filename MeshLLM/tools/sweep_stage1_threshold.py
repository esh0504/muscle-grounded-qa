#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-1 이진화 임계값 0 → 1.0 전수 훑기 + val 표본 seed 안정성.

`eval_s1.py` 는 val 에서 임계값 하나를 골라 그 수치만 낸다. 이 스크립트는 그 선택이
**얼마나 민감한지**를 보여준다 — 0 에서 1.0 까지 전 구간의 Muscle-EM / F1 곡선과,
val 표본을 seed 5개로 바꿔가며 고른 임계값들이 어디에 떨어지는지.

  세트                     의미
  ───────────────────────  ──────────────────────────────────────────────
  Val split (14,752)       임계값을 고르는 곳. test 와 같은 분포다
  Set-1 probe (500)        Stage-2 표와 같은 mesh. n_act 균형 표본이라 쉽다
  Test split (14,752)      held-out 성능. 균형 없음 (3+ 가 79%)

Activation MAE 는 임계값과 무관하므로 곡선에 없다 (seed 를 바꿔도 안 변한다).

  python tools/sweep_stage1_threshold.py
  python tools/sweep_stage1_threshold.py --seeds 0 1 42 --step 0.01
  python tools/sweep_stage1_threshold.py --rows mesh3d      # 3D 만

산출물 (`--out-dir`, 기본 outputs/stage1_muscle/):
  threshold_sweep.json   곡선 전체 + seed 별 선택 결과
  threshold_sweep.png    3×2 그림 (행=EM 전구간/EM 확대/F1, 열=모델)
  threshold_sweep.md     값 표
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluators.stage1_muscle import (  # noqa: E402
    build_input, load_activations, load_probe_meshes, load_split_meshes,
    load_stage1_model, predict,
)
from metrics import muscle_regression as mr  # noqa: E402

# dataviz 기본 팔레트 슬롯 1–3 (blue / orange / aqua). 세 슬롯은 all-pairs 검증을 통과한
# 조합이라 그대로 쓴다. 색은 **세트 정체성**을 따르고, 모델은 패널로 가른다.
SERIES = [
    ("val", "Val split (n=14,752)", "#2a78d6"),
    ("probe", "Set-1 probe (n=500)", "#eb6834"),
    ("test", "Test split (n=14,752)", "#1baf7a"),
]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
SURFACE = "#fcfcfb"


def sweep(pred: np.ndarray, gold: np.ndarray, grid: np.ndarray) -> dict:
    """임계값 격자 전체에서 EM / F1 / P / R. gold 는 `a > ACTIVE_EPS` 로 이진화."""
    G = gold > mr.ACTIVE_EPS
    em, f1, prec, rec, nact = [], [], [], [], []
    for thr in grid:
        P = pred > thr
        em.append(float((P == G).all(axis=1).mean()))
        tp, fp, fn = int((P & G).sum()), int((P & ~G).sum()), int((~P & G).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
        prec.append(p)
        rec.append(r)
        nact.append(float(P.sum(axis=1).mean()))
    return {"muscle_em": em, "muscle_f1": f1, "precision": prec, "recall": rec,
            "pred_n_act": nact}


def select_on(pred: np.ndarray, gold: np.ndarray, grid: np.ndarray) -> tuple[float, float]:
    """격자에서 Muscle-EM 최대인 임계값 (동점이면 작은 쪽) 과 그때의 EM."""
    curve = sweep(pred, gold, grid)["muscle_em"]
    best = int(np.argmax(curve))          # argmax 는 첫 최대 → 작은 임계값
    return float(grid[best]), float(curve[best])


def at(grid: np.ndarray, curve: list[float], thr: float) -> float:
    return float(curve[int(np.argmin(np.abs(grid - thr)))])


# --------------------------------------------------------------------------- #
def figure(res: dict, out_png: Path, zoom: float = 0.10) -> None:
    """small multiples — 열=모델, 행=(EM 전 구간 / EM 확대 / F1 전 구간).

    전 구간(0→1.0)만 그리면 정작 임계값이 결정되는 0–0.1 구간이 한 픽셀로 뭉갠다.
    그래서 같은 곡선을 확대한 행을 하나 둔다 — seed 5개가 고른 값이 거기서만 보인다.
    패널 안의 3색은 **평가 세트** 정체성이다 (모델은 열로 갈랐다).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.asarray(res["grid"])
    rows = res["rows"]
    panels = [("muscle_em", "Muscle-EM", 1.0),
              ("muscle_em", f"Muscle-EM  (zoom 0–{zoom:g})", zoom),
              ("muscle_f1", "Muscle F1", 1.0)]

    fig, axes = plt.subplots(len(panels), len(rows), figsize=(11.5, 10.6),
                             sharey="row", facecolor=SURFACE)
    axes = np.asarray(axes).reshape(len(panels), len(rows))

    for ci, row in enumerate(rows):
        for ri, (key, pname, xmax) in enumerate(panels):
            ax = axes[ri, ci]
            ax.set_facecolor(SURFACE)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(GRID)
            ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
            ax.set_axisbelow(True)
            ax.tick_params(colors=INK_2, labelsize=9, length=0)
            keep = grid <= xmax
            is_zoom = xmax < 1.0

            # seed 5개가 고른 임계값 — 계열이 아니라 주석이라 회색 얇은 선.
            for s in row["seeds"]:
                ax.axvline(s["threshold"], color=INK_2, linewidth=0.8, alpha=0.4,
                           zorder=1)

            for si, (sk, slabel, color) in enumerate(SERIES):
                curve = np.asarray(row["curves"][sk][key])
                ax.plot(grid[keep], curve[keep], color=color, linewidth=2.0, zorder=3,
                        solid_capstyle="round", label=slabel)
                bi = int(np.argmax(curve))
                ax.plot(grid[bi], curve[bi], "o", ms=6, color=color,
                        markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=4)
                # 봉우리 한 점만 직접 라벨 (모든 점에 숫자 금지). val 과 test 는 거의
                # 겹치므로 확대 행에서만, 계열마다 다른 오프셋으로 찍는다.
                if is_zoom:
                    ax.annotate(f"{curve[bi] * 100:.1f} @ {grid[bi]:g}",
                                (grid[bi], curve[bi]), textcoords="offset points",
                                xytext=(9, (18, 6, -13)[si]), fontsize=8.5,
                                color=color, zorder=5)

            if ri == 0:
                ax.set_title(row["label"], fontsize=12, color=INK, pad=10)
            if ci == 0:
                ax.set_ylabel(pname, fontsize=10.5, color=INK)
            ax.set_xlabel("binarization threshold", fontsize=10, color=INK_2)
            ax.set_xlim(0, xmax)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}")
            if is_zoom:
                thrs = sorted({s["threshold"] for s in row["seeds"]})
                ax.set_xticks(list(np.arange(0, xmax + 1e-9, 0.02)))
                ax.annotate("thr chosen on val: "
                            + ", ".join(f"{t:g}" for t in thrs),
                            (0.5, 0.04), xycoords="axes fraction", ha="center",
                            fontsize=8.5, color=INK_2)

    handles = [plt.Line2D([], [], color=c, linewidth=2.4, label=lab)
               for _, lab, c in SERIES]
    handles.append(plt.Line2D([], [], color=INK_2, linewidth=0.8, alpha=0.5,
                              label=f"thr chosen on val ({len(rows[0]['seeds'])} seeds)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, labelcolor=INK_2, bbox_to_anchor=(0.5, 0.004))
    fig.suptitle("Stage-1 muscle set — metric vs binarization threshold (%)",
                 fontsize=13.5, color=INK, y=0.988)
    fig.tight_layout(rect=(0, 0.045, 1, 0.965), h_pad=2.6)
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def markdown(res: dict) -> str:
    grid = np.asarray(res["grid"])
    marks = [t for t in (0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0)
             if t <= grid.max()]
    out = ["# Stage-1 임계값 훑기 (0 → 1.0) · val seed 5개\n",
           f"격자 {grid[0]:.3g} … {grid[-1]:.3g}, 간격 {res['step']:.3g} "
           f"({len(grid)}점) · gold `a > {mr.ACTIVE_EPS}`\n",
           "Activation MAE 는 임계값과 무관하다 — seed 를 바꿔도 안 변한다.\n"]

    for row in res["rows"]:
        out.append(f"\n## {row['label']}\n")
        out.append(f"\n### val seed 별 선택 (n_select={res['n_select']:,})\n")
        out.append("\n| seed | 고른 thr | val EM | probe500 EM | test split EM |\n"
                   "|---:|---:|---:|---:|---:|\n")
        for s in row["seeds"]:
            out.append(f"| {s['seed']} | **{s['threshold']:.3g}** | "
                       f"{s['val_em'] * 100:.1f} | {s['probe_em'] * 100:.1f} | "
                       f"{s['test_em'] * 100:.1f} |\n")
        thrs = [s["threshold"] for s in row["seeds"]]
        pe = [s["probe_em"] for s in row["seeds"]]
        te = [s["test_em"] for s in row["seeds"]]
        out.append(f"\nthr {min(thrs):.3g}–{max(thrs):.3g} · probe500 EM "
                   f"{min(pe) * 100:.1f}–{max(pe) * 100:.1f} · test EM "
                   f"{min(te) * 100:.1f}–{max(te) * 100:.1f}"
                   f" · val 전체({res['n_val']:,})로 고르면 "
                   f"**{row['thr_val_full']:.3g}**\n")

        out.append(f"\n### 곡선 (Muscle-EM %, 굵은 값 = 각 세트의 봉우리)\n\n| thr |"
                   + "".join(f" {lab} |" for _, lab, _ in SERIES) + "\n|---:|"
                   + "---:|" * len(SERIES) + "\n")
        peaks = {sk: float(np.max(row["curves"][sk]["muscle_em"])) for sk, _, _ in SERIES}
        for t in marks:
            cells = []
            for sk, _, _ in SERIES:
                v = at(grid, row["curves"][sk]["muscle_em"], t)
                cells.append(f" **{v * 100:.1f}** |" if abs(v - peaks[sk]) < 1e-12
                             else f" {v * 100:.1f} |")
            out.append(f"| {t:.3g} |" + "".join(cells) + "\n")
        for sk, lab, _ in SERIES:
            c = row["curves"][sk]["muscle_em"]
            bi = int(np.argmax(c))
            out.append(f"\n- {lab}: 최대 EM **{c[bi] * 100:.1f}** @ thr {grid[bi]:.3g}")
        out.append("\n")
    return "".join(out)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/evaluators/stage1_muscle.yaml")
    ap.add_argument("--rows", nargs="*", default=["mesh3d"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 42, 123, 2024])
    ap.add_argument("--step", type=float, default=0.005, help="임계값 격자 간격")
    ap.add_argument("--n-select", type=int, default=4000, help="seed 당 val 표본 수")
    ap.add_argument("--out-dir", default="outputs/stage1_muscle")
    a = ap.parse_args()

    cfg = OmegaConf.load(a.config)
    mesh_root = str(cfg.data.mesh_root)
    grid = np.round(np.arange(0.0, 1.0 + a.step / 2, a.step), 6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = {str(r["id"]): dict(r) for r in OmegaConf.to_container(cfg.model_rows)}

    probe = load_probe_meshes(cfg.probe_path, str(cfg.get("family", "muscle_set")))
    sets = {"probe": [r["mesh_index"] for r in probe],
            "val": load_split_meshes(mesh_root, "val", None, 42),
            "test": load_split_meshes(mesh_root, "test", None, 42)}
    print(f"[sweep] 격자 {len(grid)}점 (0 → 1.0, {a.step}) · seed {a.seeds}")

    rows = []
    for rid in a.rows:
        row = table[rid]
        source = build_input(str(row["input"]), cfg.data)
        model = load_stage1_model(row["model_cfg"], row["checkpoint"], device)

        ids = {k: [i for i in v if source.available(i)] for k, v in sets.items()}
        gold = {k: load_activations(mesh_root, v) for k, v in ids.items()}
        pred = {k: predict(model, source, v, device, int(cfg.get("batch_size", 64)))
                for k, v in ids.items()}
        curves = {k: sweep(pred[k], gold[k], grid) for k in ids}
        print(f"[sweep] {rid}: " + " · ".join(f"{k} n={len(v)}" for k, v in ids.items()))

        # seed 별로 val 부분표본을 다시 뽑아 임계값을 고른다 (예측은 재사용).
        n_val = len(ids["val"])
        seeds = []
        for s in a.seeds:
            take = np.random.default_rng(s).permutation(n_val)[:min(a.n_select, n_val)]
            thr, val_em = select_on(pred["val"][take], gold["val"][take], grid)
            seeds.append({"seed": int(s), "threshold": thr, "val_em": val_em,
                          "probe_em": at(grid, curves["probe"]["muscle_em"], thr),
                          "test_em": at(grid, curves["test"]["muscle_em"], thr),
                          "probe_f1": at(grid, curves["probe"]["muscle_f1"], thr),
                          "test_f1": at(grid, curves["test"]["muscle_f1"], thr)})
            print(f"[sweep]   seed {s:>5}: thr={thr:.3g}  val EM={val_em:.3f}  "
                  f"probe500 EM={seeds[-1]['probe_em']:.3f}  "
                  f"test EM={seeds[-1]['test_em']:.3f}")
        thr_full, _ = select_on(pred["val"], gold["val"], grid)

        rows.append({"id": rid, "label": row.get("label", rid),
                     "checkpoint": row.get("checkpoint"),
                     "n": {k: len(v) for k, v in ids.items()},
                     "activation_mae": {k: float(np.abs(pred[k] - gold[k]).mean())
                                        for k in ids},
                     "curves": curves, "seeds": seeds, "thr_val_full": thr_full})

    res = {"grid": grid.tolist(), "step": a.step, "seeds": list(a.seeds),
           "n_select": a.n_select, "n_val": len(sets["val"]),
           "gold_eps": mr.ACTIVE_EPS, "rows": rows}

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "threshold_sweep.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "threshold_sweep.md").write_text(markdown(res), encoding="utf-8")
    figure(res, out / "threshold_sweep.png")
    print(f"\n→ {out}/threshold_sweep.{{json,md,png}}")


if __name__ == "__main__":
    main()
