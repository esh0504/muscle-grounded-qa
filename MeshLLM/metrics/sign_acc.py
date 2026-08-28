"""Sign accuracy — 상호작용 비선형.  [Set 2, ★ 일반화 핵심 지표]

학습에 없던 협동/길항 프레이밍에서 super(+)/additive(0)/sub(−)를 맞히는가.
정답은 시뮬레이터 fact 에서 결정론적으로 계산된 값(`score.gold`)이라 LLM-judge 순환논리가 없다.

majority-class baseline 을 반드시 함께 본다 — gold 분포가 치우쳐 있어(현재 − 275 / 0 132 / + 93)
한 라벨만 외쳐도 0.55 가 나온다.

파싱 규칙은 DATA/unseentest/eval.py 의 parse_sign 과 동일하게 유지한다.
"""

from __future__ import annotations

import re
from collections import Counter

PAT_SUPER = re.compile(
    r"super-?additive|synerg|coopera|amplif|greater than the (linear )?sum|"
    r"exceeds? the (linear )?sum|more than the sum|초가법|협동|상승작용", re.I)
PAT_SUB = re.compile(
    r"sub-?additive|antagon|oppos|cancel|less than the (linear )?sum|offset|"
    r"smaller than the sum|길항|상쇄|저가법", re.I)
PAT_ADD = re.compile(
    r"\badditive\b|equal to the sum|about the sum|roughly the sum|"
    r"matches the (linear )?superposition|linear|가법적|선형 중첩", re.I)


def parse_sign(text: str) -> str:
    """'+' | '-' | '0' | '?'(파싱 실패). 구체적 주장(super/sub)을 additive 보다 먼저 본다."""
    if PAT_SUPER.search(text):
        return "+"
    if PAT_SUB.search(text):
        return "-"
    if PAT_ADD.search(text):
        return "0"
    return "?"


def score(items: list[dict]) -> dict:
    """items: [{"gold": "+|-|0", "pred": "..."}]"""
    n = cor = 0
    dist = Counter(); conf = Counter(); unparsed = 0
    for it in items:
        g = it["gold"]
        p = parse_sign(it.get("pred") or "")
        n += 1; dist[g] += 1; conf[f"{g}->{p}"] += 1
        unparsed += (p == "?")
        cor += (p == g)
    maj = max(dist.values()) / sum(dist.values()) if dist else 0.0
    return {"accuracy": cor / n if n else 0.0, "n": n, "correct": cor,
            "majority_baseline": maj, "unparsed_rate": unparsed / n if n else 0.0,
            "gold_dist": dict(dist), "confusion": dict(sorted(conf.items()))}
