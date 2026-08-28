"""Stage-1 회귀 헤드 지표 — **Muscle-EM** · **Activation MAE**.

Set-1 probe(`metrics/set1_probe.py`)가 *LLM 이 생성한 문장*을 채점한다면, 이쪽은 Stage-1
모델이 **회귀 헤드로 직접 뱉은 11-D 활성 벡터**를 채점한다. 같은 mesh, 같은 gold 를 쓰므로
두 표를 나란히 놓을 수 있다 (Stage-2 Muscle F1 ↔ Stage-1 Muscle-EM/F1).

| 지표 | 정의 | 임계값 |
|---|---|---|
| Muscle-EM      | 예측 활성 **집합**이 gold 집합과 정확히 같은 비율 | 필요 (`thr`) |
| Activation MAE | `mean |pred − gold|` (11-D 전체)                  | 불필요 |
| Muscle F1      | micro set-F1 (Stage-2 표와 비교용)                | 필요 (`thr`) |

**gold 집합**은 probe 빌더와 같은 규칙이다 — `a_i > ACTIVE_EPS (1e-4)`
(`tools/build_set1_probe.py`, `SET1_METRICS_DETAIL.md` §0).

⚠️ `thr` 는 **eval 세트에서 고르면 안 된다**. `select_threshold()` 를 val split 에서 돌려
고른 값을 test 500개에 적용한다 (`evaluators/stage1_muscle.py`). probe 위에서 낸 sweep 은
진단용이지 headline 이 아니다.

⚠️ Activation MAE 는 gold 활성값이 대부분 0 이라 **0 만 찍어도 낮게 나온다**. 그래서
`mae_active` / `mae_inactive` 를 함께 돌려준다 — 셋을 같이 봐야 한다.
"""

from __future__ import annotations

import math

import numpy as np

from datasets.mesh_dataset import MUSCLE_NAMES

# probe 빌더(tools/build_set1_probe.py)와 같은 값. 바꾸면 gold 가 어긋난다.
ACTIVE_EPS = 1e-4

# select_threshold 기본 격자. 활성값이 0.0002 까지 내려가서 아래쪽을 촘촘히 깐다.
DEFAULT_THR_GRID = (
    [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    + [round(0.01 * k, 4) for k in range(1, 20)]      # 0.01 … 0.19
    + [round(0.20 + 0.05 * k, 4) for k in range(0, 13)]  # 0.20 … 0.80
)


def _as_2d(a) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected (N, M), got {arr.shape}")
    return arr


def _group_mean(values: np.ndarray, keys) -> dict[str, float]:
    """`per_n_act` 류 분해. keys 가 None 이면 빈 dict."""
    if keys is None:
        return {}
    keys = [str(k) for k in keys]
    out: dict[str, float] = {}
    for key in sorted(set(keys)):
        sel = np.array([k == key for k in keys], dtype=bool)
        out[key] = float(values[sel].mean()) if sel.any() else 0.0
    return out


def active_mask(a, eps: float) -> np.ndarray:
    """`a > eps` 인 원소 마스크. gold 는 `eps=ACTIVE_EPS`, 예측은 고른 `thr`."""
    return _as_2d(a) > float(eps)


def active_names(mask_row, muscle_names=MUSCLE_NAMES) -> list[str]:
    return [n for n, on in zip(muscle_names, np.asarray(mask_row, dtype=bool)) if on]


def n_act_bins(gold, eps: float = ACTIVE_EPS) -> list[str]:
    """gold 에서 `n_act_bin` (1 / 2 / 3+) 을 **다시 계산**한다.

    ⚠️ probe jsonl 의 `n_act` 필드를 쓰면 안 된다. 그 값은 CSV 의 `n_active` 열을 그대로
    복사한 것이고, 그 열은 `(a >= 0.05).sum()` 이라 `ACTIVE_EPS(1e-4)` 기준과 다르다
    (500개 중 37개가 어긋나고 7개는 bin 이 바뀐다). `gold.muscles` 자체는 맞다.
    """
    counts = active_mask(gold, eps).sum(axis=1)
    return [("3+" if c >= 3 else str(int(c))) for c in counts]


def scorer_ceiling(gold, thr: float, gold_eps: float = ACTIVE_EPS) -> dict:
    """gold 를 **그대로 예측했을 때**의 EM — 이진화 임계값이 깎아먹는 몫.

    활성값이 0.0002 까지 내려가므로 `thr` 이 그보다 크면 완벽한 회귀기라도 그 항목을
    놓친다. 모델 수치 옆에 이 상한을 같이 적어야 "임계값 손실"과 "모델 오차"가 안 섞인다.
    """
    G = active_mask(gold, gold_eps)
    exact = (active_mask(gold, thr) == G).all(axis=1)
    lost = int((G & ~active_mask(gold, thr)).sum())
    return {"threshold": float(thr), "muscle_em": float(exact.mean()),
            "n_active_lost": lost, "n_active_entries": int(G.sum())}


# --------------------------------------------------------------------------- #
# Muscle-EM · Muscle F1 (임계값 필요)
# --------------------------------------------------------------------------- #
def score_muscle_em(pred, gold, thr: float, *, gold_eps: float = ACTIVE_EPS,
                    n_act=None, muscle_names=MUSCLE_NAMES) -> dict:
    """예측 활성 집합 == gold 활성 집합 비율 (+ 같은 이진화의 micro F1).

    EM 은 집합이 통째로 맞아야 1 이라 F1 보다 훨씬 엄격하다. 근육 하나만 어긋나도 0.
    """
    P = active_mask(pred, thr)
    G = active_mask(gold, gold_eps)
    if P.shape != G.shape:
        raise ValueError(f"shape mismatch: pred {P.shape} vs gold {G.shape}")

    exact = (P == G).all(axis=1)
    tp = int((P & G).sum())
    fp = int((P & ~G).sum())
    fn = int((~P & G).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    # 근육별 이진 정확도 — 어느 근육에서 집합이 깨지는지 본다.
    per_muscle = {name: float((P[:, i] == G[:, i]).mean())
                  for i, name in enumerate(muscle_names)}

    return {
        "muscle_em": float(exact.mean()),
        "muscle_f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn,
        "threshold": float(thr),
        "gold_eps": float(gold_eps),
        "n": int(len(P)),
        "n_exact": int(exact.sum()),
        "pred_n_act_mean": float(P.sum(axis=1).mean()),
        "gold_n_act_mean": float(G.sum(axis=1).mean()),
        "per_muscle_acc": per_muscle,
        "per_n_act": _group_mean(exact.astype(np.float64), n_act),
        "exact": exact.tolist(),
    }


def diagnose_dead_heads(pred, gold, thr: float, *, gold_eps: float = ACTIVE_EPS,
                        muscle_names=MUSCLE_NAMES) -> dict:
    """임계값을 **한 번도 못 넘는** 출력 차원(죽은 헤드)과 그로 인한 EM 상한.

    Stage-1 은 L1 로 학습한다 (`losses/loss_s1.py`). L1 최적해는 조건부 **중앙값**이라,
    절반 넘게 0 인 근육은 형상으로 잘 안 풀리는 순간 헤드가 통째로 0 으로 눌린다. 그런
    근육이 gold 에 하나라도 켜져 있는 mesh 는 EM 이 **구조적으로 불가능**하다.

    `em_ceiling` = 죽은 근육이 gold 에 없는 mesh 비율 = 달성 가능한 EM 의 상한.
    `muscle_em_reachable` = 그 mesh 들만 놓고 잰 EM. headline EM 을 이 둘과 같이 읽어라.
    """
    P, G = _as_2d(pred), active_mask(gold, gold_eps)
    dead = [i for i in range(P.shape[1]) if P[:, i].max() <= float(thr)]
    blocked = G[:, dead].any(axis=1) if dead else np.zeros(len(G), dtype=bool)
    exact = (active_mask(P, thr) == G).all(axis=1)
    reachable = ~blocked
    return {
        "dead_muscles": [muscle_names[i] for i in dead],
        "pred_max_per_muscle": {muscle_names[i]: float(P[:, i].max())
                                for i in range(P.shape[1])},
        "n_blocked": int(blocked.sum()),
        "em_ceiling": float(reachable.mean()),
        "muscle_em_reachable": float(exact[reachable].mean()) if reachable.any() else 0.0,
    }


def select_threshold(pred, gold, *, grid=None, gold_eps: float = ACTIVE_EPS,
                     objective: str = "muscle_em") -> dict:
    """`objective` 를 최대로 하는 임계값을 격자에서 고른다.

    **선택은 val split 에서만 한다.** test 500개 위에서 고르면 그 수치는 임계값을 맞춰준
    상한이지 성능이 아니다 (`SET1_PROBE_SPEC.md` §0 split honesty 와 같은 이유).
    동점이면 격자에서 먼저 나오는(=작은) 임계값을 쓴다.
    """
    if objective not in ("muscle_em", "muscle_f1"):
        raise ValueError(f"objective must be muscle_em|muscle_f1, got {objective!r}")
    grid = list(DEFAULT_THR_GRID if grid is None else grid)

    curve = []
    for thr in grid:
        s = score_muscle_em(pred, gold, thr, gold_eps=gold_eps)
        curve.append({"threshold": float(thr), "muscle_em": s["muscle_em"],
                      "muscle_f1": s["muscle_f1"]})
    best = max(curve, key=lambda r: (r[objective], -r["threshold"]))
    return {"threshold": best["threshold"], "objective": objective,
            "best": best, "curve": curve, "n": int(len(_as_2d(pred)))}


# --------------------------------------------------------------------------- #
# Activation MAE (임계값 불필요)
# --------------------------------------------------------------------------- #
def score_activation_mae(pred, gold, *, gold_eps: float = ACTIVE_EPS,
                         n_act=None, muscle_names=MUSCLE_NAMES) -> dict:
    """11-D 활성 회귀 오차. headline 은 `mae` (전체 원소 평균).

    gold 의 ~77% 가 0 이라 `mae` 하나만 보면 "전부 0" 예측이 좋아 보인다.
    `mae_active`(gold 가 켜진 원소) / `mae_inactive`(꺼진 원소) 를 같이 봐라.
    """
    P, G = _as_2d(pred), _as_2d(gold)
    if P.shape != G.shape:
        raise ValueError(f"shape mismatch: pred {P.shape} vs gold {G.shape}")

    err = np.abs(P - G)
    on = G > float(gold_eps)

    return {
        "mae": float(err.mean()),
        "mae_active": float(err[on].mean()) if on.any() else 0.0,
        "mae_inactive": float(err[~on].mean()) if (~on).any() else 0.0,
        "rmse": float(np.sqrt(((P - G) ** 2).mean())),
        "n": int(len(P)),
        "n_active_entries": int(on.sum()),
        "per_muscle_mae": {name: float(err[:, i].mean())
                           for i, name in enumerate(muscle_names)},
        "per_n_act": _group_mean(err.mean(axis=1), n_act),
        "item_mae": err.mean(axis=1).tolist(),
    }


# --------------------------------------------------------------------------- #
# 행끼리 비교 — 격차가 표본 잡음인지
# --------------------------------------------------------------------------- #
def _mcnemar_exact_p(b: int, c: int) -> float:
    """불일치 쌍 (b, c) 에 대한 McNemar 정확검정 양측 p (이항 n=b+c, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return float(min(1.0, 2 * tail))


def compare_rows(a: dict, b: dict, *, seed: int = 42, n_boot: int = 20000) -> dict:
    """두 행의 격차에 유의성을 붙인다. **같은 mesh 를 본 쌍 표본**이어야 한다.

    `a`, `b` 는 `score_all()` 결과. Muscle-EM 은 이진 결과라 McNemar 정확검정을,
    Activation MAE 는 mesh 별 연속값이라 쌍 부트스트랩 CI 를 쓴다.

    ⚠️ 유의하지 않은 격차를 결론으로 쓰지 마라. n=500 에서 EM 2 pt 차이는 잡음과
    구분되지 않는다 — 이 함수는 그걸 표에 강제로 적어 넣기 위해 있다.
    """
    ea, eb = np.asarray(a["em"]["exact"], bool), np.asarray(b["em"]["exact"], bool)
    ma, mb = np.asarray(a["mae"]["item_mae"]), np.asarray(b["mae"]["item_mae"])
    if not (len(ea) == len(eb) == len(ma) == len(mb)):
        raise ValueError("행끼리 mesh 수가 다르다 — 쌍 비교가 성립하지 않는다")

    a_only, b_only = int((ea & ~eb).sum()), int((eb & ~ea).sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ea), size=(int(n_boot), len(ea)))
    d_em = (ea[idx].mean(axis=1) - eb[idx].mean(axis=1))
    d_mae = (ma[idx].mean(axis=1) - mb[idx].mean(axis=1))

    return {
        "a": a["label"], "b": b["label"], "n": int(len(ea)),
        "muscle_em": {
            "delta": float(ea.mean() - eb.mean()),
            "a_only": a_only, "b_only": b_only, "both": int((ea & eb).sum()),
            "mcnemar_exact_p": _mcnemar_exact_p(a_only, b_only),
            "ci95": [float(np.percentile(d_em, 2.5)), float(np.percentile(d_em, 97.5))],
        },
        "activation_mae": {
            "delta": float(ma.mean() - mb.mean()),   # 음수 = a 가 더 낮다(좋다)
            "ci95": [float(np.percentile(d_mae, 2.5)), float(np.percentile(d_mae, 97.5))],
            "a_better_frac": float((ma < mb).mean()),
        },
        "boot": {"n_boot": int(n_boot), "seed": int(seed)},
    }


# --------------------------------------------------------------------------- #
# 묶음
# --------------------------------------------------------------------------- #
def score_all(pred, gold, *, thr: float, gold_eps: float = ACTIVE_EPS,
              n_act=None, muscle_names=MUSCLE_NAMES, thr_grid=None) -> dict:
    """headline 두 개 + 분해 + probe 위 임계값 sweep(진단).

    sweep 은 "임계값을 이 세트에 맞췄다면" 상한을 보여줄 뿐이다. headline `muscle_em` 은
    바깥(val)에서 고른 `thr` 로 계산한 값이다.
    """
    em = score_muscle_em(pred, gold, thr, gold_eps=gold_eps, n_act=n_act,
                         muscle_names=muscle_names)
    mae = score_activation_mae(pred, gold, gold_eps=gold_eps, n_act=n_act,
                               muscle_names=muscle_names)
    sweep = select_threshold(pred, gold, grid=thr_grid, gold_eps=gold_eps)
    dead = diagnose_dead_heads(pred, gold, thr, gold_eps=gold_eps,
                               muscle_names=muscle_names)
    return {
        "muscle_em": em["muscle_em"],
        "activation_mae": mae["mae"],
        "muscle_f1": em["muscle_f1"],
        "threshold": float(thr),
        "n": em["n"],
        "em": em,
        "mae": mae,
        "dead_heads": dead,
        "scorer_ceiling": scorer_ceiling(gold, thr, gold_eps),
        "oracle_sweep": {"threshold": sweep["threshold"],
                         "muscle_em": sweep["best"]["muscle_em"],
                         "muscle_f1": sweep["best"]["muscle_f1"],
                         "curve": sweep["curve"]},
    }
