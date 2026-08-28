"""Direction accuracy — 개입 추론.  [Set 1, turn_type=B2]

반사실(근육 한 단계 변경) 질문에서 형상 변화 **방향**을 맞히는가.
정답은 gold answer 의 movement span 개념(advance/retract/elevate/descend/...)이고,
예측 텍스트에서 같은 개념을 뽑아 비교한다.

정답 조건(둘 다 만족해야 correct):
  1) gold 방향 개념을 모두 언급했다
  2) 그 반대 개념을 함께 말하지 않았다  ← 양쪽을 다 나열해 요행으로 맞는 것을 막는다
"""

from __future__ import annotations

from metrics.spans import OPPOSITE, extract_facts, facts_from_spans

DIRECTIONAL = set(OPPOSITE)


def score(items: list[dict], lang: str = "en") -> dict:
    n = cor = skipped = 0
    contradictions = 0
    for it in items:
        gold = set(facts_from_spans(it.get("gold_spans") or [], lang)["movement"]) & DIRECTIONAL
        if not gold:
            skipped += 1
            continue
        got = set(extract_facts(it.get("pred") or "", lang)["movement"])
        n += 1
        has_all = gold <= got
        contra = any(OPPOSITE[g] in got for g in gold)
        contradictions += bool(contra)
        cor += bool(has_all and not contra)
    return {"accuracy": cor / n if n else 0.0, "n": n, "correct": cor,
            "contradiction_rate": contradictions / n if n else 0.0,
            "skipped_no_gold_direction": skipped}
