"""Monotonicity accuracy — 용량-반응.  [Set 1, turn_type=B3]

활성 세기를 올릴 때의 추세를 맞히는가. 3-way 라벨:
  monotonic  — 단조 증가/감소
  saturating — 증가폭이 줄어드는 포화형
  reversal   — 방향이 뒤집힘
gold 는 gold answer 의 movement span 개념(mono/saturate/reverse)에서 결정론적으로 만든다.
majority-class baseline 을 함께 보고한다 (한 라벨만 외쳐도 나오는 점수).
"""

from __future__ import annotations

from collections import Counter

from metrics.spans import extract_facts, facts_from_spans

_ORDER = ("reverse", "saturate", "mono")   # 앞쪽이 더 구체적인 주장 → 우선
_LABEL = {"reverse": "reversal", "saturate": "saturating", "mono": "monotonic"}


def _label(concepts: set) -> str | None:
    for c in _ORDER:
        if c in concepts:
            return _LABEL[c]
    return None


def score(items: list[dict], lang: str = "en") -> dict:
    n = cor = 0
    dist = Counter(); conf = Counter()
    for it in items:
        gold = _label(set(facts_from_spans(it.get("gold_spans") or [], lang)["movement"]))
        if gold is None:
            continue
        pred = _label(set(extract_facts(it.get("pred") or "", lang)["movement"])) or "none"
        n += 1; dist[gold] += 1; conf[f"{gold}->{pred}"] += 1
        cor += (pred == gold)
    maj = max(dist.values()) / sum(dist.values()) if dist else 0.0
    return {"accuracy": cor / n if n else 0.0, "n": n, "correct": cor,
            "majority_baseline": maj, "gold_dist": dict(dist), "confusion": dict(conf)}
