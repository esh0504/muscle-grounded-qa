"""Mask F1 — 사실성(faithfulness).  [Set 1, Set 2]

정답 answer 의 fact span(근육명/수치/방향/부위)을 모델 답이 재현하는가.
micro-F1 을 카테고리별 + 전체로 보고한다. 환각(fp)과 누락(fn)이 둘 다 잡힌다.

카테고리에 채점할 사실이 하나도 없으면(gold·pred 모두 공집합) f1 을 0 이 아니라 **null**
로 보고한다. 0 으로 적으면 "그 카테고리를 다 틀렸다"로 읽히고 macro 까지 끌어내린다.
"""

from __future__ import annotations

from metrics import number_df
from metrics.spans import (
    CATEGORIES,
    extract_facts,
    facts_from_spans,
    prf,
    set_prf,
    untaggable_spans,
)


# 허용오차 기본값 — 고정 절대값이 아니라 **정답값 대비 상대**로 준다 (2026-08-03 변경).
TOL_REL = 0.02      # 정답값의 2%
TOL_FLOOR = 0.01    # 작은 값에서 무한히 엄격해지지 않게 하는 하한


def _tol_prf(gold: set, pred: set, rel: float, floor: float) -> tuple[int, int, int]:
    """수치를 **허용오차 안이면 맞은 것**으로 세는 그리디 매칭.

    exact string match 는 이 데이터에서 달성 불가능한 정의다: 동결 인코더 특징에서
    **완벽한 선형 판독기**조차 소수 3자리를 맞출 확률이 평균 0.194 뿐이다
    (2자리 0.398 / 1자리 0.831 — 연구 repo 진단 측정).
    QA 답변은 `cd_min=0.162` 처럼 3자리를 쓰므로, tolerance 0 인 지표는 모델 성능이 아니라
    인코더의 정보량 상한을 재게 된다. 허용오차판을 **함께** 보고한다.

    허용오차는 **정답값마다** `max(floor, rel * |gold|)` 로 정한다. 2026-08-03 이전의 고정
    절대값(0.05)은 양(quantity)마다 강도가 완전히 달랐다 — role 별 중앙값 대비 실측:
    `curv_peak` 0.0%(사실상 완전일치 요구) ~ `doming` 55.6%(사실상 무조건 통과), **편차 55,600배**.
    상대 2% + 하한 0.01 이면 2.0% ~ 11.1%(편차 6배)로 좁혀진다. 하한이 없으면 작은 값
    (`doming` 중앙값 0.09 → 0.0018)에서 허용오차가 인코더 해상도 밑으로 내려가 다시 측정 불가다.

    정답 하나가 예측 하나를 소비한다(`used`). 1차원 + 정답별 허용오차에서는 정답을 오름차순으로
    돌며 가장 가까운 미사용 예측을 집는 그리디가 최적 매칭과 일치한다.
    """
    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    g = sorted([v for v in (_f(x) for x in gold) if v is not None])
    p = sorted([v for v in (_f(x) for x in pred) if v is not None])
    used = [False] * len(p)
    tp = 0
    for gv in g:
        tol = max(floor, rel * abs(gv))
        best, bd = -1, None
        for i, pv in enumerate(p):
            if used[i]:
                continue
            d = abs(pv - gv)
            if d <= tol and (bd is None or d < bd):
                best, bd = i, d
        if best >= 0:
            used[best] = True
            tp += 1
    # 숫자로 못 읽은 값만 exact 규칙으로 따로 센다. 파싱된 값은 위 루프가 이미 처리했으므로,
    # 여기서 `gold & pred` 전체를 더하면 이중 계수가 되어 tp > |gold| 와 **음수 fp/fn** 이 나온다
    # (예: gold=pred={'0.84','1.2.3'} → tp=3, fp=-1). 현재 NUMRE 는 항상 float 파싱이 되는
    # 문자열만 뽑아서(실측 722,013 span 중 실패 0) 이 분기가 죽어 있지만, 정규식이나 주석
    # 방식을 바꾸면 살아난다.
    tp += len({x for x in gold if _f(x) is None} & {x for x in pred if _f(x) is None})
    return tp, len(pred) - tp, len(gold) - tp


def score(items: list[dict], lang: str = "en",
          tol_rel: float | None = TOL_REL, tol_floor: float = TOL_FLOOR) -> dict:
    """items: [{"gold_spans": [...], "pred": "...", "turn_type": "A1"}]

    (gold_spans 대신 gold_text 도 가능. turn_type 은 number_informative 계산에만 쓴다.)

    tol_rel=None 이면 허용오차판(`number_tol` · `number_informative_tol`)을 계산하지 않는다.
    """
    agg = {c: [0, 0, 0] for c in CATEGORIES}
    inf = [0, 0, 0]
    tol = [0, 0, 0]        # number_informative + 허용오차
    tol_all = [0, 0, 0]    # number 전체 + 허용오차
    n = ignored = 0
    df_table = number_df.load(lang)
    for it in items:
        pred = it.get("pred") or ""
        if it.get("gold_spans") is not None:
            gold = facts_from_spans(it["gold_spans"], lang)
            ignored += untaggable_spans(it["gold_spans"], lang)
        else:
            gold = extract_facts(it.get("gold_text", ""), lang)
        got = extract_facts(pred, lang)
        n += 1
        for c in CATEGORIES:
            tp, fp, fn = set_prf(gold[c], got[c])
            agg[c][0] += tp; agg[c][1] += fp; agg[c][2] += fn
        if df_table:
            bp = number_df.boilerplate(df_table, it.get("turn_type", ""))
            g_inf, p_inf = gold["number"] - bp, got["number"] - bp
            tp, fp, fn = set_prf(g_inf, p_inf)
            inf[0] += tp; inf[1] += fp; inf[2] += fn
            if tol_rel is not None:
                t, f, n_ = _tol_prf(g_inf, p_inf, tol_rel, tol_floor)
                tol[0] += t; tol[1] += f; tol[2] += n_
        if tol_rel is not None:
            t, f, n_ = _tol_prf(gold["number"], got["number"], tol_rel, tol_floor)
            tol_all[0] += t; tol_all[1] += f; tol_all[2] += n_

    per_cat = {c: (prf(*agg[c]) if sum(agg[c]) else None) for c in CATEGORIES}
    # number_informative 는 number 의 부분집합이라 micro/macro 에는 넣지 않는다(이중 계수).
    per_cat["number_informative"] = prf(*inf) if (df_table and sum(inf)) else None
    if tol_rel is not None:
        per_cat["number_tol"] = prf(*tol_all) if sum(tol_all) else None
        per_cat["number_informative_tol"] = prf(*tol) if (df_table and sum(tol)) else None
    scored = [v["f1"] for c, v in per_cat.items()
              if v is not None and c in CATEGORIES]
    tot = [sum(agg[c][i] for c in CATEGORIES) for i in range(3)]
    return {
        "n": n,
        "micro": prf(*tot),
        "macro_f1": sum(scored) / len(scored) if scored else None,
        "per_category": per_cat,
        "gold_spans_untaggable": ignored,   # 어휘표에 없어 채점에서 빠진 gold span 수
        # 2026-08-03 이전에는 float 하나(고정 절대 0.05)였다. 옛 metrics.json 과 구분된다.
        "number_tolerance": ({"rel": tol_rel, "floor": tol_floor}
                             if tol_rel is not None else None),
    }
