"""Abstention F1 — 정직성/캘리브레이션.  [Set 1]

역문제(D1: "형상만으로 근육을 알 수 있나?")처럼 답이 유일하지 않은 질문에서
기권하는가. 기권을 양성 클래스로 두고 P/R/F1.

  P = 정당한 기권 / 모델이 한 전체 기권
  R = 모델이 기권한 필요 질문 / 기권이 필요한 질문 전체
"""

from __future__ import annotations

from metrics.spans import is_abstention, prf


def score(items: list[dict], lang: str = "en") -> dict:
    """items: [{"pred": "...", "should_abstain": bool}]"""
    tp = fp = fn = tn = 0
    for it in items:
        pred_abs = is_abstention(it.get("pred") or "", lang)
        gold_abs = bool(it.get("should_abstain"))
        if pred_abs and gold_abs:
            tp += 1
        elif pred_abs and not gold_abs:
            fp += 1
        elif not pred_abs and gold_abs:
            fn += 1
        else:
            tn += 1
    out = prf(tp, fp, fn)
    n_should = tp + fn
    if n_should == 0:
        # 기권이 필요한 질문이 하나도 없으면 F1 은 정의되지 않는다.
        # prf(0,0,0) 이 돌려주는 0.0 을 그대로 쓰면 "모델이 기권을 못 한다"로 오독된다
        # (실제로 학습 중 val 지표가 이 축퇴 케이스였다 — 평가 부분집합에 D1 이 없었다).
        out.update({k: None for k in ("precision", "recall", "f1")})
    # 비-D1 에서의 과잉 기권. 기권 F1 은 질문 텍스트만으로도 높게 나오므로
    # 캘리브레이션은 이 값으로 봐야 한다.
    n_neg = fp + tn
    out.update({"n": len(items), "tn": tn,
                "n_should_abstain": n_should,
                "abstain_rate": (tp + fp) / len(items) if items else 0.0,
                "over_abstain_rate": (fp / n_neg) if n_neg else None})
    return out
