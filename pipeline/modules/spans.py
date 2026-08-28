# -*- coding: utf-8 -*-
"""mask_spans — verifiable-token tagging shared by every QA generator.

Each gold answer carries character-level spans typed as
    muscle | region | movement | number(+role)
so that (a) structured checks can verify facts survive naturalization and
(b) span-level metrics (Muscle EM, Value Acc, Direction EM) are computable.

One factory replaces the four near-identical copies that used to live in
scale_qa.py / scale_qa_en.py / feature_qa.py / feature_qa_en.py. Matching
behavior is preserved exactly per language config (see step4_qa_templates).
"""
import re

NUMRE = re.compile(r"-?\d+\.\d+|-?\d+")
_PRIORITY = {"muscle": 0, "region": 1, "movement": 2, "number": 3}


def _find_all(text, word):
    """Non-overlapping literal occurrences (str.find loop)."""
    out = []
    i = text.find(word)
    while i >= 0:
        out.append((i, i + len(word)))
        i = text.find(word, i + len(word))
    return out


def _word_all(text, word):
    r"""Non-overlapping \b-bounded occurrences."""
    return [(m.start(), m.end()) for m in re.finditer(r"\b" + re.escape(word) + r"\b", text)]


def make_mask_spans(muscle_full, moves, regions, roles,
                    move_mode="find", region_mode="find", ctx=12):
    """Build a mask_spans(text) function.

    muscle_full : iterable of full muscle display names (matched literally)
    moves       : movement keywords            move_mode:   'find' | 'word'
    regions     : region keywords              region_mode: 'find' | 'word'
    roles       : [(context_key, role), ...] — first key found in the `ctx`
                  chars before a number assigns its role (else 'value')
    """
    muscle_full = list(muscle_full)
    moves = list(moves)
    regions = list(regions)
    roles = list(roles)
    mmatch = _find_all if move_mode == "find" else _word_all
    rmatch = _find_all if region_mode == "find" else _word_all

    def mask_spans(text):
        sp = []
        for m in muscle_full:
            for a, b in _find_all(text, m):
                sp.append({"type": "muscle", "value": m, "start": a, "end": b})
        for w in moves:
            for a, b in mmatch(text, w):
                sp.append({"type": "movement", "value": w, "start": a, "end": b})
        for w in regions:
            for a, b in rmatch(text, w):
                sp.append({"type": "region", "value": w, "start": a, "end": b})
        for mt in NUMRE.finditer(text):
            st = mt.start()
            if text[max(0, st - 1):st] == "#":       # mesh ids (#123) are not values
                continue
            pre = text[max(0, st - ctx):st]
            role = next((r for k, r in roles if k in pre), "value")
            sp.append({"type": "number", "value": mt.group(), "start": st, "end": mt.end(), "role": role})
        sp.sort(key=lambda s: (s["start"], _PRIORITY[s["type"]]))
        out, occ = [], []
        for s in sp:
            if any(not (s["end"] <= a or s["start"] >= b) for a, b in occ):
                continue
            out.append(s)
            occ.append((s["start"], s["end"]))
        return sorted(out, key=lambda s: s["start"])

    return mask_spans
