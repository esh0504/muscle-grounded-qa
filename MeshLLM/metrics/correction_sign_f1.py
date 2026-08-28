"""Correction-sign F1 — 진단형의 부분 자동 앵커.  [Set 3]

"어떤 근육을 올려야/내려야 하나"를 gold(should_increase/decrease)와 대조한다.
근육 언급마다 **가장 가까운** 방향 동사로 inc/dec 를 정하는 휴리스틱이라 프록시다.
주 지표는 GPT-judge / human 이고, 이 값은 그 판정이 엉뚱하지 않은지 보는 앵커다.

파싱 규칙은 DATA/unseentest/eval.py 의 muscle_directions 와 동일하게 유지한다.
"""

from __future__ import annotations

import re

from metrics.spans import prf

MUS = ["GGP", "GGM", "GGA", "STY", "GH", "MH", "HG", "VERT", "TRANS", "IL", "SL"]
FULLNAME = {
    "GGP": "genioglossus posterior", "GGM": "genioglossus medius",
    "GGA": "genioglossus anterior", "STY": "styloglossus", "GH": "geniohyoid",
    "MH": "mylohyoid", "HG": "hyoglossus", "VERT": "verticalis",
    "TRANS": "transversus", "IL": "inferior longitudinal", "SL": "superior longitudinal",
}
# "contract GGP" / "relax VERT" 도 ↑/↓ 와 동일하게 인정한다.
INC_VERB = (
    r"(increase|raise|contract(?:\s+more)?|activate|strengthen|tighten|"
    r"더\s*수축|증가|올려)"
)
DEC_VERB = (
    r"(decrease|relax|reduce|lower|release|deactivate|"
    r"이완|감소|내려|낮(?:추|춰|추어)|풀\b)"
)
WIN = 45


def _nearest(win: str, anchor: int, pat: str):
    best = None
    for mt in re.finditer(pat, win):
        d = abs(mt.start() - anchor)
        if best is None or d < best:
            best = d
    return best


def muscle_directions(text: str) -> set[tuple[str, str]]:
    """Nearest-verb heuristic. Conflicting inc+dec for the same muscle are dropped."""
    best: dict[str, tuple[str, int]] = {}  # mus → (tag, distance)
    for m in MUS:
        for pat in (r"\b" + m + r"\b", FULLNAME[m]):
            for mt in re.finditer(pat, text, flags=re.I):
                a = mt.start()
                win = text[max(0, a - WIN):mt.end() + WIN].lower()
                anc = a - max(0, a - WIN)
                di = _nearest(win, anc, INC_VERB)
                dd = _nearest(win, anc, DEC_VERB)
                if di is None and dd is None:
                    continue
                if dd is None or (di is not None and di <= dd):
                    tag, dist = "inc", int(di)
                else:
                    tag, dist = "dec", int(dd)
                prev = best.get(m)
                if prev is None or dist < prev[1]:
                    best[m] = (tag, dist)
                elif prev is not None and dist == prev[1] and prev[0] != tag:
                    # 동일 거리로 상충 → 제거 표시
                    best[m] = ("?", dist)
    return {(m, tag) for m, (tag, _) in best.items() if tag in ("inc", "dec")}


def score(items: list[dict]) -> dict:
    """items: [{"gold_inc": [...], "gold_dec": [...], "pred": "..."}]"""
    tp = fp = fn = 0
    for it in items:
        gold = {(m, "inc") for m in it.get("gold_inc", [])} | \
               {(m, "dec") for m in it.get("gold_dec", [])}
        pred = muscle_directions(it.get("pred") or "")
        tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
    out = prf(tp, fp, fn)
    out["n"] = len(items)
    out["note"] = "휴리스틱 프록시 — 주 지표는 GPT-judge / human"
    return out
