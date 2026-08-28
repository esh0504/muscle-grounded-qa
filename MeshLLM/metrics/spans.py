"""fact span 추출·대조 공통 유틸.

어휘는 **자연화 검증에 쓴 것과 같은 표**를 그대로 재사용한다(`natural_vllm.VOCAB`).
채점기가 데이터 주석과 다른 어휘를 쓰면 점수가 어휘 차이를 재게 되므로,
mask_spans 를 만든 코드와 단일 출처를 공유하는 것이 중요하다.
"""

from __future__ import annotations

import re

from natural_vllm import NUMRE, VOCAB, _concepts

CATEGORIES = ("muscle", "number", "movement", "region")

# 방향 개념의 반대쌍 — 방향 정확도에서 "정답 개념을 말했지만 반대말도 함께 말한" 경우를 잡는다.
OPPOSITE = {
    "advance": "retract", "retract": "advance",
    "elevate": "descend", "descend": "elevate",
    "inc": "dec", "dec": "inc",
    "flat": "arch", "arch": "flat",
}


def _muscles(text: str, lang: str) -> list[str]:
    return [n for n in VOCAB[lang]["names"] if n in text]


def _numbers(text: str) -> list[str]:
    return [x for x in NUMRE.findall(text) if x not in ("", "-", ".")]


def extract_facts(text: str, lang: str = "en") -> dict[str, set]:
    """모델 답변 문자열 → 카테고리별 fact **집합**.

    number 는 값 그대로, muscle 은 정식명칭, movement/region 은 표면형이 아니라
    **개념**으로 뽑는다 ("단조 증가"와 "monotonically increases"가 같은 것으로 세어지도록).

    집합(중복 제거)인 이유: readme 의 정의가 "예측 fact 집합과 정답 fact 집합의 micro-F1"이고,
    같은 사실을 여러 번 반복한다고 점수를 더 주거나 깎을 이유가 없다. 정답 쪽(mask_spans)은
    출현 횟수만큼 span 이 붙으므로, 양쪽 다 집합으로 맞추지 않으면 반복 언급된 부위어에서
    허위 recall 손실이 생긴다.
    """
    v = VOCAB[lang]
    return {
        "muscle": set(_muscles(text, lang)),
        "number": set(_numbers(text)),
        "movement": set(_concepts(text, v["move_c"])),
        "region": set(_concepts(text, v["region_c"])),
    }


def facts_from_spans(spans: list[dict], lang: str = "en") -> dict[str, set]:
    """정답 turn 의 mask_spans → 같은 형식의 fact 집합 (채점 기준)."""
    v = VOCAB[lang]
    out: dict[str, set] = {c: set() for c in CATEGORIES}
    for s in spans or []:
        t, val = s.get("type"), s.get("value", "")
        if t == "muscle":
            out["muscle"].add(val)
        elif t == "number":
            out["number"].add(val)
        elif t == "movement":
            out["movement"] |= v["move_c"].get(val, set())
        elif t == "region":
            out["region"] |= v["region_c"].get(val, set())
    return out


def untaggable_spans(spans: list[dict], lang: str = "en") -> int:
    """어휘표로 표현할 수 없어 채점에서 빠지는 gold span 수.

    movement/region span 중 개념표에 없는 표면형(예: L2 의 'superposition')은 양쪽 모두에서
    빠져 대칭이지만, 그만큼 지표가 덮지 못하는 사실이 있다는 뜻이라 함께 보고한다.
    """
    v = VOCAB[lang]
    n = 0
    for s in spans or []:
        t, val = s.get("type"), s.get("value", "")
        if t == "movement" and not v["move_c"].get(val):
            n += 1
        elif t == "region" and not v["region_c"].get(val):
            n += 1
    return n


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def set_prf(gold: set, pred: set) -> tuple[int, int, int]:
    """집합 기준 tp/fp/fn."""
    return len(gold & pred), len(pred - gold), len(gold - pred)


# --------------------------------------------------------------------------- #
# 기권 판정
# --------------------------------------------------------------------------- #
ABSTAIN_PATTERNS = {
    "en": [r"can(?:no|')?t be determined", r"cannot be", r"not unique", r"motor equivalen",
           r"no,? it can'?t", r"not possible to tell", r"insufficient", r"ambiguous",
           r"more than one .{0,20}combination", r"different .{0,25}combinations"],
    "ko": [r"판단할 수 없", r"판단 불가", r"알 수 없", r"유일하지 않", r"비유일",
           r"운동 등가", r"결정할 수 없", r"여러 .{0,15}조합"],
}


def is_abstention(text: str, lang: str = "en") -> bool:
    t = text.lower()
    pats = ABSTAIN_PATTERNS.get(lang, []) + ABSTAIN_PATTERNS["en"]
    return any(re.search(p, t) for p in pats)
