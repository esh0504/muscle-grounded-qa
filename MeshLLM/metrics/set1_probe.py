"""Set-1 probe metrics — Metircs.md / SET1_PROBE_SPEC.md.

Muscle F1 · Value acc (trainlike scalars) · Direction · Abstention · Δ_shuf.
Utility (GPT-J / Human) is payload-only here. M+V removed.

Value = region_disp mm (front/mid/back) and/or A1 features (cl_t, peak_z, …).
Per-item `gold.tol` (mm vs normalized).
"""

from __future__ import annotations

import re
from collections import defaultdict

from datasets.mesh_dataset import MUSCLE_NAMES
from metrics.spans import OPPOSITE, is_abstention, prf, set_prf
from natural_vllm import FULL_EN, FULL_KO, NUMRE

VALUE_TOL = 2.0          # default / mm fallback
VALUE_TOL_NORM = 0.1     # normalized A1-feature fallback
VALUE_TOL_CD_MIN = 0.03  # tighter slot used at probe build time
REGIONS = ("tip", "body", "root", "dorsum")
# rest_desc 에서 front/mid/back 서술을 tip/body/root 로 인정 (dorsum 은 별칭 없음).
REGION_ALIASES = {
    "tip": "tip", "body": "body", "root": "root", "dorsum": "dorsum",
    "front": "tip", "mid": "body", "middle": "body", "back": "root",
    "anterior": "tip", "posterior": "root",
    "설첨": "tip", "설체": "body", "설근": "root", "설배": "dorsum",
    "앞": "tip", "중간": "body", "뒤": "root",
}
DIR_CONCEPTS = {"advance", "retract", "elevate", "descend", "preserve"}
MOVE_EPS_MM = 0.5  # numeric dx/dz → direction (builder MOVE_EPS_MM 과 동일)


def resolve_value_tol_cfg(raw=None, *, value_tol: float | None = None) -> dict:
    """Normalize evaluator config → value-tol dict.

    Config knobs (configs/evaluators/set1_probe.yaml → ``value:``):
      source: gold | config
        gold   — use probe ``gold.tol`` when present (default; build-time tols)
        config — ignore gold.tol; use unit / per_quantity from this config
      tol_mm / tol_norm / default_tol / per_quantity

    Legacy flat ``value_tol: 2.0`` still maps to ``tol_mm`` / ``default_tol``.
    """
    cfg: dict = {
        "source": "gold",
        "default_tol": float(VALUE_TOL),
        "tol_mm": float(VALUE_TOL),
        "tol_norm": float(VALUE_TOL_NORM),
        "per_quantity": {"cd_min": float(VALUE_TOL_CD_MIN)},
    }
    if isinstance(raw, dict):
        for k in ("source", "default_tol", "tol_mm", "tol_norm"):
            if raw.get(k) is not None:
                cfg[k] = raw[k] if k == "source" else float(raw[k])
        pq = raw.get("per_quantity") or {}
        if isinstance(pq, dict) and pq:
            merged = dict(cfg["per_quantity"])
            merged.update({str(k): float(v) for k, v in pq.items()})
            cfg["per_quantity"] = merged
        # allow nested alias value.tol → default
        if raw.get("tol") is not None and raw.get("default_tol") is None:
            cfg["default_tol"] = float(raw["tol"])
            cfg["tol_mm"] = float(raw.get("tol_mm", raw["tol"]))
    if value_tol is not None:
        # flat legacy override wins for mm/default when explicitly passed
        cfg["default_tol"] = float(value_tol)
        cfg["tol_mm"] = float(value_tol)
    src = str(cfg["source"]).lower().strip()
    if src not in ("gold", "config"):
        raise ValueError(f"value.source must be 'gold' or 'config', got {src!r}")
    cfg["source"] = src
    return cfg

# surface → concept (EN + KO). Longest match first when scanning.
_DIR_SURF = [
    # EN
    ("advance", "advance"), ("protrud", "advance"), ("forward", "advance"),
    ("retract", "retract"), ("backward", "retract"),
    ("elevate", "elevate"), ("rais", "elevate"), ("upward", "elevate"),
    ("descend", "descend"), ("depress", "descend"), ("lower", "descend"),
    ("preserve", "preserve"), ("unchanged", "preserve"), ("no change", "preserve"),
    ("negligible", "preserve"), ("stay", "preserve"),
    # KO
    ("전진", "advance"), ("전방", "advance"), ("돌출", "advance"),
    ("후퇴", "retract"), ("후방", "retract"),
    ("상승", "elevate"), ("올림", "elevate"), ("거상", "elevate"),
    ("하강", "descend"), ("내림", "descend"), ("하방", "descend"),
    ("유지", "preserve"), ("변화 없", "preserve"), ("거의 없", "preserve"),
]


def _abbr_aliases() -> dict[str, str]:
    """Map surface forms → muscle abbreviation."""
    m = {a: a for a in MUSCLE_NAMES}
    m.update({a.lower(): a for a in MUSCLE_NAMES})
    for src in (FULL_EN, FULL_KO):
        for abbr, full in src.items():
            m[full.lower()] = abbr
            # "genioglossus posterior (GGP)" → also bare full name before paren
            bare = re.sub(r"\s*\([^)]*\)\s*", "", full).strip().lower()
            if bare:
                m[bare] = abbr
    # common short forms from MUSCLE_INFO-style names
    extras = {
        "genioglossus posterior": "GGP", "genioglossus medius": "GGM",
        "genioglossus medial": "GGM", "genioglossus anterior": "GGA",
        "styloglossus": "STY", "geniohyoid": "GH", "mylohyoid": "MH",
        "hyoglossus": "HG", "verticalis": "VERT", "transversus": "TRANS",
        "inferior longitudinal": "IL", "superior longitudinal": "SL",
    }
    m.update(extras)
    return m


_ALIAS = _abbr_aliases()


def parse_muscles(text: str) -> set[str]:
    t = text or ""
    found: set[str] = set()
    # abbreviations as whole tokens
    for abbr in MUSCLE_NAMES:
        if re.search(rf"\b{abbr}\b", t, flags=re.IGNORECASE):
            found.add(abbr)
    # full-name / alias longest-first
    for surf, abbr in sorted(_ALIAS.items(), key=lambda kv: -len(kv[0])):
        if len(surf) < 3:
            continue
        if surf in t.lower():
            found.add(abbr)
    return found


def parse_value(text: str) -> float | None:
    nums = [x for x in NUMRE.findall(text or "") if x not in ("", "-", ".")]
    if not nums:
        return None
    try:
        return float(nums[0])
    except ValueError:
        return None


def _region_token_pat() -> str:
    """Longest-first alternation of region surface forms (aliases included)."""
    keys = sorted(REGION_ALIASES.keys(), key=len, reverse=True)
    return "|".join(re.escape(k) for k in keys)


def parse_directions(text: str) -> dict[str, set[str]]:
    """Split by region mentions (tip/body/root/dorsum **or** front/mid/back).

    Also accept numeric cues near a region: dx/dz (mm) → advance/retract,
    elevate/descend using the same sign convention as value-slot region_disp
    (dx>0 advance, dz>0 elevate) and ``MOVE_EPS_MM``.
    """
    t = (text or "").lower()
    per: dict[str, set[str]] = {r: set() for r in REGIONS}
    scoped = False
    tok = _region_token_pat()
    # "tip=advance+elevate; body=preserve" / "front: retract, raise"
    for m in re.finditer(rf"({tok})\s*[=:]\s*([^;|\n]+)", t, flags=re.I):
        canon = REGION_ALIASES.get(m.group(1).lower())
        if not canon:
            continue
        scoped = True
        per[canon] |= _merge_dir_concepts(m.group(2))
    # loose window: "front region … (retract, raise) … dx -0.04"
    for m in re.finditer(
            rf"({tok})\b(.{{0,80}}?)(?={tok}|$)", t, flags=re.I):
        canon = REGION_ALIASES.get(m.group(1).lower())
        if not canon:
            continue
        got = _merge_dir_concepts(m.group(0))
        if got:
            scoped = True
            per[canon] |= got
    if scoped:
        return per
    # unscoped fallback: global bag → every gold region (legacy behaviour)
    global_c = _merge_dir_concepts(t)
    return {r: set(global_c) for r in REGIONS}


def _merge_dir_concepts(text: str) -> set[str]:
    """Word concepts win; dx/dz signs only fill a missing AP/vertical axis."""
    words = _concepts_in(text)
    nums = _concepts_from_numbers(text)
    ap = {"advance", "retract"}
    vert = {"elevate", "descend"}
    out = set(words)
    if not (out & ap):
        out |= nums & ap
    if not (out & vert):
        out |= nums & vert
    # preserve from either
    if "preserve" in words or "preserve" in nums:
        out.add("preserve")
    return out & DIR_CONCEPTS


def _concepts_in(text: str) -> set[str]:
    t = (text or "").lower()
    got: set[str] = set()
    for surf, concept in sorted(_DIR_SURF, key=lambda x: -len(x[0])):
        if surf in t:
            got.add(concept)
    return got & DIR_CONCEPTS


def _concepts_from_numbers(text: str) -> set[str]:
    """dx/dz (and forward/up) signed magnitudes → direction concepts."""
    t = (text or "").lower()
    got: set[str] = set()
    # dx / forward / AP
    for m in re.finditer(
            r"(?:\bdx\b|forward(?:_mm)?|ap)\s*[=:]?\s*([+-]?\d+(?:\.\d+)?)", t):
        v = float(m.group(1))
        if abs(v) >= MOVE_EPS_MM:
            got.add("advance" if v > 0 else "retract")
    # dz / up / vertical
    for m in re.finditer(
            r"(?:\bdz\b|up(?:_mm)?|vertical)\s*[=:]?\s*([+-]?\d+(?:\.\d+)?)", t):
        v = float(m.group(1))
        if abs(v) >= MOVE_EPS_MM:
            got.add("elevate" if v > 0 else "descend")
    # parenthetical pair: (dx 0.68mm, dz -0.74mm) / (-0.95/0.25) after inc/dec talk
    for m in re.finditer(
            r"\bdx\b[^0-9+-]{0,6}([+-]?\d+(?:\.\d+)?)\s*(?:mm)?\s*[,/]\s*"
            r"(?:\bdz\b[^0-9+-]{0,6})?([+-]?\d+(?:\.\d+)?)", t):
        vx, vz = float(m.group(1)), float(m.group(2))
        if abs(vx) >= MOVE_EPS_MM:
            got.add("advance" if vx > 0 else "retract")
        if abs(vz) >= MOVE_EPS_MM:
            got.add("elevate" if vz > 0 else "descend")
    return got


# --------------------------------------------------------------------------- #
# scorers
# --------------------------------------------------------------------------- #
def score_muscle_f1(items: list[dict]) -> dict:
    tp = fp = fn = 0
    n = 0
    by_bin: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # tp,fp,fn,n
    for it in items:
        gold = set((it.get("gold") or {}).get("muscles") or [])
        pred = parse_muscles(it.get("pred") or "")
        t, f_p, f_n = set_prf(gold, pred)
        tp += t; fp += f_p; fn += f_n
        n += 1
        b = str(it.get("n_act_bin") or "")
        by_bin[b][0] += t; by_bin[b][1] += f_p; by_bin[b][2] += f_n; by_bin[b][3] += 1
        it["item_score"] = {
            "muscle_correct": gold == pred and bool(gold),
            "muscle_f1": _f1(t, f_p, f_n),
            "pred_muscles": sorted(pred),
        }
    out = prf(tp, fp, fn)
    out.update({"n": n, "per_n_act": {b: {**prf(v[0], v[1], v[2]), "n": v[3]}
                                      for b, v in sorted(by_bin.items())}})
    return out


def _item_value_tol(it: dict, tol_cfg: dict | float = VALUE_TOL) -> float:
    """Resolve |pred−gold| threshold for one value item.

    Priority when ``source=gold`` (default):
      1) gold.tol / meta.tol (baked at ``tools/build_set1_probe.py``)
      2) per_quantity[quantity] from config
      3) unit → tol_mm / tol_norm
      4) default_tol

    When ``source=config``: skip (1), use config only.
    """
    if not isinstance(tol_cfg, dict):
        tol_cfg = resolve_value_tol_cfg(value_tol=float(tol_cfg))
    g = it.get("gold") or {}
    meta = it.get("meta") or {}
    qty = str(g.get("quantity") or meta.get("quantity") or "")
    unit = g.get("unit") or meta.get("unit")
    per_q = tol_cfg.get("per_quantity") or {}

    if tol_cfg.get("source", "gold") == "gold":
        if g.get("tol") is not None:
            return float(g["tol"])
        if meta.get("tol") is not None:
            return float(meta["tol"])

    if qty and qty in per_q:
        return float(per_q[qty])
    if unit == "mm":
        return float(tol_cfg.get("tol_mm", VALUE_TOL))
    if unit == "norm":
        return float(tol_cfg.get("tol_norm", VALUE_TOL_NORM))
    return float(tol_cfg.get("default_tol", VALUE_TOL))


def score_value_acc(items: list[dict], tol: float | dict = VALUE_TOL) -> dict:
    """Value acc: correct iff |first_number − gold| ≤ tol (per item)."""
    if isinstance(tol, dict):
        tol_cfg = resolve_value_tol_cfg(tol)
    else:
        tol_cfg = resolve_value_tol_cfg(value_tol=float(tol))
    n = cor = 0
    by_bin: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_qty: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for it in items:
        gold_o = it.get("gold") or {}
        gold = gold_o.get("value")
        pred = parse_value(it.get("pred") or "")
        itol = _item_value_tol(it, tol_cfg)
        ok = gold is not None and pred is not None and abs(pred - float(gold)) <= itol
        n += 1
        cor += bool(ok)
        b = str(it.get("n_act_bin") or "")
        by_bin[b][0] += bool(ok); by_bin[b][1] += 1
        q = str(gold_o.get("quantity") or "?")
        by_qty[q][0] += bool(ok); by_qty[q][1] += 1
        it["item_score"] = {
            **(it.get("item_score") or {}),
            "value_correct": bool(ok),
            "pred_value": pred,
            "gold_value": gold,
            "tol": itol,
            "unit": gold_o.get("unit"),
            "quantity": q,
        }
    return {
        "accuracy": cor / n if n else 0.0, "n": n, "correct": cor,
        "tol_cfg": tol_cfg,
        "default_tol": tol_cfg.get("default_tol"),
        "per_n_act": {b: {"accuracy": v[0] / v[1] if v[1] else 0.0, "n": v[1],
                          "correct": v[0]} for b, v in sorted(by_bin.items())},
        "per_quantity": {q: {"accuracy": v[0] / v[1] if v[1] else 0.0, "n": v[1],
                             "correct": v[0]} for q, v in sorted(by_qty.items())},
    }


def _pred_uses_fmb(text: str) -> bool:
    t = (text or "").lower()
    return bool(re.search(r"\b(front|mid|middle|back)\b", t))


def score_direction_rest(items: list[dict]) -> dict:
    """rest→current descriptive region directions (tip/body/root/dorsum).

    front/mid/back (and dx/dz signs) are accepted as tip/body/root evidence.
    When the pred only uses FMB vocabulary, dorsum is not required.
    """
    n_reg = cor_reg = 0
    n_item = cor_item = 0
    contradictions = 0
    for it in items:
        gold_d = (it.get("gold") or {}).get("directions") or {}
        pred_text = it.get("pred") or ""
        pred_d = parse_directions(pred_text)
        fmb_only = _pred_uses_fmb(pred_text) and not re.search(
            r"\b(tip|body|root|dorsum)\b", pred_text, flags=re.I)
        item_ok = True
        region_scores = {}
        scored_any = False
        for r in REGIONS:
            g = set(gold_d.get(r) or [])
            p = set(pred_d.get(r) or [])
            if not g:
                continue
            # FMB-only answers: skip dorsum (no FMB alias)
            if fmb_only and r == "dorsum" and not p:
                region_scores[r] = {
                    "gold": sorted(g), "pred": sorted(p),
                    "correct": None, "skipped": "fmb_alias_no_dorsum",
                }
                continue
            scored_any = True
            n_reg += 1
            if g == {"preserve"}:
                ok = bool(p) and p <= {"preserve"}
            else:
                has_all = g <= p
                contra = any(OPPOSITE.get(x) in p for x in g if x in OPPOSITE)
                contradictions += bool(contra)
                ok = bool(has_all and not contra)
            cor_reg += bool(ok)
            region_scores[r] = {"gold": sorted(g), "pred": sorted(p), "correct": bool(ok)}
            item_ok = item_ok and ok
        if not scored_any:
            item_ok = False
        n_item += 1
        cor_item += bool(item_ok)
        it["item_score"] = {
            **(it.get("item_score") or {}),
            "direction_kind": "rest_desc",
            "direction_correct": bool(item_ok),
            "regions": region_scores,
        }
    return {
        "accuracy": cor_reg / n_reg if n_reg else 0.0,
        "item_accuracy": cor_item / n_item if n_item else 0.0,
        "n_regions": n_reg, "correct_regions": cor_reg,
        "n": n_item, "correct_items": cor_item,
        "contradiction_rate": contradictions / n_reg if n_reg else 0.0,
        "note": "front/mid/back→tip/body/root; dx/dz signs; FMB-only skips dorsum",
    }


def score_direction_correction(items: list[dict]) -> dict:
    """Muscle ↑/↓ corrections toward an anchor (Set3-style set F1).

    Accepts arrows and verbal paraphrases: contract/increase≈↑, relax/decrease≈↓.
    """
    tp = fp = fn = 0
    n_item = cor_item = 0
    for it in items:
        gold_o = it.get("gold") or {}
        gold = {(m, "inc") for m in (gold_o.get("gold_inc") or [])} | \
               {(m, "dec") for m in (gold_o.get("gold_dec") or [])}
        pred = parse_correction_pairs(it.get("pred") or "")
        t = len(gold & pred); f_p = len(pred - gold); f_n = len(gold - pred)
        tp += t; fp += f_p; fn += f_n
        item_ok = bool(gold) and gold == pred
        n_item += 1
        cor_item += item_ok
        it["item_score"] = {
            **(it.get("item_score") or {}),
            "direction_kind": "correction",
            "direction_correct": item_ok,
            "correction_f1": _f1(t, f_p, f_n),
            "pred_pairs": sorted([f"{m}:{d}" for m, d in pred]),
            "gold_pairs": sorted([f"{m}:{d}" for m, d in gold]),
        }
    out = prf(tp, fp, fn)
    out.update({
        "n": n_item, "correct_items": cor_item,
        "item_accuracy": cor_item / n_item if n_item else 0.0,
        "note": "↑/↓ + contract/relax + KO SOV (muscles→더 수축/이완) + full names",
    })
    return out


def parse_correction_pairs(text: str) -> set[tuple[str, str]]:
    """Merge arrow / verb-list / KO-SOV / nearest-verb parsers.

    Priority: arrow(3) > KO muscles-before-verb(2) = EN verb-list(2) > proximity(1).
    """
    from metrics.correction_sign_f1 import muscle_directions

    best: dict[str, tuple[str, int]] = {}

    def _put(m: str, tag: str, pri: int):
        prev = best.get(m)
        if prev is None or pri > prev[1]:
            best[m] = (tag, pri)
        elif prev[1] == pri and prev[0] != tag:
            best[m] = ("?", pri)  # ambiguous at same priority

    for m, tag in _parse_arrow_corrections(text):
        _put(m, tag, 3)
    # Korean SOV ("A, B 더 수축하고 … 이완") before EN list — same pri; both OK
    for m, tag in _parse_ko_sov_corrections(text):
        _put(m, tag, 2)
    for m, tag in _parse_verb_list_corrections(text):
        _put(m, tag, 2)
    for m, tag in muscle_directions(text):
        _put(m, tag, 1)
    return {(m, tag) for m, (tag, _) in best.items() if tag in ("inc", "dec")}


def _muscles_in_chunk(chunk: str) -> set[str]:
    """Abbreviations + English/Korean full names inside a text span."""
    found: set[str] = set()
    for m in MUSCLE_NAMES:
        if re.search(rf"\b{m}\b", chunk, flags=re.I):
            found.add(m)
    fulls = {
        "genioglossus posterior": "GGP", "genioglossus medius": "GGM",
        "genioglossus anterior": "GGA", "styloglossus": "STY",
        "geniohyoid": "GH", "mylohyoid": "MH", "hyoglossus": "HG",
        "verticalis": "VERT", "transversus": "TRANS",
        "inferior longitudinal": "IL", "superior longitudinal": "SL",
        # Korean surfaces (with and without English parenthetical)
        "이설근 후부": "GGP", "이설근 중부": "GGM", "이설근 전부": "GGA",
        "경돌설근": "STY", "이설골근": "GH", "악설골근": "MH",
        "설골설근": "HG", "수직근": "VERT", "횡근": "TRANS",
        "하종설근": "IL", "상종설근": "SL",
    }
    cl = chunk.lower()
    # keep original for Hangul match
    raw = chunk
    for surf, abbr in sorted(fulls.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"[가-힣]", surf):
            if surf in raw:
                found.add(abbr)
        elif surf in cl:
            found.add(abbr)
    return found


def _parse_arrow_corrections(text: str) -> set[tuple[str, str]]:
    """Parse '↑ GGP, STY; ↓ HG' / 'GGP↑' style.

    Trailing ``MUS↑/MUS↓`` wins over a leading-arrow list chunk, so
    ``STY↑, TRANS↓`` does not mark TRANS as inc from the STY↑ spill.
    """
    found: dict[str, str] = {}
    t = text or ""

    # 1) trailing form first (highest confidence)
    for m in MUSCLE_NAMES:
        has_up = bool(re.search(rf"\b{m}\s*↑", t, flags=re.I))
        has_dn = bool(re.search(rf"\b{m}\s*↓", t, flags=re.I))
        if has_up and has_dn:
            continue
        if has_up:
            found[m] = "inc"
        elif has_dn:
            found[m] = "dec"

    # 2) leading arrow + list (skip muscles already tagged by trailing form)
    for arrow, tag in (("↑", "inc"), ("↓", "dec"), ("⬆", "inc"), ("⬇", "dec")):
        for mt in re.finditer(
                rf"(?<![A-Za-z]){re.escape(arrow)}\s*([^↑↓⬆⬇\n;]+)", t):
            chunk = mt.group(1)
            # do not spill past a trailing-arrow muscle inside the chunk
            chunk = re.split(rf"(?:{'|'.join(MUSCLE_NAMES)})\s*[↑↓⬆⬇]",
                             chunk, maxsplit=1)[0]
            for m in _muscles_in_chunk(chunk):
                if m not in found:
                    found[m] = tag
    return set(found.items())


def _parse_verb_list_corrections(text: str) -> set[tuple[str, str]]:
    """Parse 'contract GGP, STY … relax VERT' — stop at opposing verb / sentence."""
    found: set[tuple[str, str]] = set()
    t = text or ""
    inc = (
        r"increase|raise|contract(?:\s+more)?|activate|strengthen|tighten|"
        r"더\s*수축|증가|올려"
    )
    dec = (
        r"decrease|relax|reduce|lower|release|deactivate|"
        r"이완|감소|내려|낮(?:추|춰|추어)|풀"
    )
    # verb-governed chunks; stop at opposing verb, sentence end, or length cap
    for lab, tag, stop in ((inc, "inc", dec), (dec, "dec", inc)):
        for mt in re.finditer(
                rf"(?:{lab})\b(.{{0,120}}?)(?=(?:{stop})\b|[.;\n]|$)",
                t, flags=re.I | re.S):
            chunk = mt.group(1)
            for m in _muscles_in_chunk(chunk):
                found.add((m, tag))
    return found


def _parse_ko_sov_corrections(text: str) -> set[tuple[str, str]]:
    """Parse Korean SOV: 'A, B 더 수축하고, C, D를 이완'.

    Muscles appear *before* the direction verb (unlike English verb-list order).
    """
    if not text or not re.search(r"[가-힣]", text):
        return set()
    found: set[tuple[str, str]] = set()
    t = text
    inc = r"더\s*수축|수축|증가|올려|활성"
    dec = r"이완|감소|내려|낮(?:추|춰|추어)|풀"
    for lab, tag, stop in ((inc, "inc", dec), (dec, "dec", inc)):
        for mt in re.finditer(rf"(.{{0,160}}?)(?:{lab})", t, flags=re.I | re.S):
            chunk = mt.group(1)
            # keep only the last clause
            chunk = re.split(r"[.。\n;；]", chunk)[-1]
            # drop muscles belonging to the opposing verb earlier in the clause
            parts = re.split(rf"(?:{stop})", chunk)
            chunk = parts[-1] if parts else chunk
            if len(chunk) > 160:
                chunk = chunk[-160:]
            for m in _muscles_in_chunk(chunk):
                found.add((m, tag))
    return found


B2_MAG_MIN_MM = 2.0  # filter weak nearest-mesh interventions
_AP_AXIS = ("advance", "retract")
_VERT_AXIS = ("elevate", "descend")


def _b2_motion_mag_mm(gold_o: dict, meta: dict) -> float | None:
    """Dominant-region displacement magnitude (mm) from gold/meta."""
    dx = gold_o.get("dx")
    dz = gold_o.get("dz")
    if dx is not None and dz is not None:
        return float((float(dx) ** 2 + float(dz) ** 2) ** 0.5)
    dom = gold_o.get("dominant_region") or meta.get("dominant_region")
    per = meta.get("delta_mm") or {}
    if dom and isinstance(per.get(dom), dict):
        d = per[dom]
        if "mag" in d:
            return float(d["mag"])
        if "dx" in d and "dz" in d:
            return float((float(d["dx"]) ** 2 + float(d["dz"]) ** 2) ** 0.5)
    return None


def _b2_pred_concepts(it: dict) -> set[str]:
    pred = it.get("pred") or ""
    gold_o = it.get("gold") or {}
    meta = it.get("meta") or {}
    dom = gold_o.get("dominant_region") or meta.get("dominant_region")
    got = _merge_dir_concepts(pred)
    scoped = parse_directions(pred)
    if dom:
        got |= set(scoped.get(dom) or [])
    return got


def _b2_gold_concepts(it: dict) -> set[str]:
    gold_o = it.get("gold") or {}
    meta = it.get("meta") or {}
    dom = gold_o.get("dominant_region") or meta.get("dominant_region")
    gold_d = gold_o.get("directions") or {}
    if dom and gold_d.get(dom):
        return set(gold_d[dom])
    g: set[str] = set()
    for labs in gold_d.values():
        g |= set(labs or [])
    return g


def _axis_sign_ok(g: set[str], p: set[str], axis: tuple[str, str]) -> bool | None:
    """None = axis not in gold; True/False = sign match on that axis.

    Extra axes in ``p`` are ignored. Hedging both signs on the gold axis → False.
    """
    g_ax = g & set(axis)
    if not g_ax:
        return None
    p_ax = p & set(axis)
    if not p_ax:
        return False
    if len(p_ax) > 1:
        return False  # hedge
    return p_ax == g_ax


def score_direction_b2(items: list[dict], *,
                       mag_min_mm: float = B2_MAG_MIN_MM) -> dict:
    """B2 diagnostic score — **axis sign only**, weak motions filtered.

    - Keep items with dominant |Δ| ≥ ``mag_min_mm`` (default 2.0).
    - For each gold AP/vertical axis: pred must state the correct sign
      (extra axes OK; opposite / hedge / missing → wrong).
    - Primary ``accuracy`` = fraction of kept items with all gold axes correct.
    """
    n = cor = 0
    n_skip_weak = n_skip_empty = 0
    n_ax = cor_ax = 0
    region_hits = 0
    for it in items:
        gold_o = it.get("gold") or {}
        meta = it.get("meta") or {}
        dom = gold_o.get("dominant_region") or meta.get("dominant_region")
        g = _b2_gold_concepts(it)
        mag = _b2_motion_mag_mm(gold_o, meta)
        pred = it.get("pred") or ""
        got = _b2_pred_concepts(it)
        rh = bool(dom and re.search(rf"\b{re.escape(str(dom))}\b", pred, flags=re.I))

        if not g or g == {"preserve"}:
            n_skip_empty += 1
            it["item_score"] = {
                **(it.get("item_score") or {}),
                "direction_kind": "b2",
                "direction_correct": None,
                "skipped": "no_gold_or_preserve",
                "mag_mm": mag,
            }
            continue
        if mag is None or mag < float(mag_min_mm):
            n_skip_weak += 1
            it["item_score"] = {
                **(it.get("item_score") or {}),
                "direction_kind": "b2",
                "direction_correct": None,
                "skipped": f"mag<{mag_min_mm}",
                "mag_mm": mag,
            }
            continue

        axis_ok = []
        for axis in (_AP_AXIS, _VERT_AXIS):
            ok_ax = _axis_sign_ok(g, got, axis)
            if ok_ax is None:
                continue
            n_ax += 1
            cor_ax += bool(ok_ax)
            axis_ok.append(bool(ok_ax))
        item_ok = bool(axis_ok) and all(axis_ok)
        n += 1
        cor += bool(item_ok)
        region_hits += rh
        it["item_score"] = {
            **(it.get("item_score") or {}),
            "direction_kind": "b2",
            "direction_correct": bool(item_ok),
            "gold_concepts": sorted(g),
            "pred_concepts": sorted(got),
            "dominant_region": dom,
            "region_mentioned": rh,
            "mag_mm": mag,
            "axis_ok": axis_ok,
            "scoring": "sign_only",
        }
    return {
        "accuracy": cor / n if n else 0.0,
        "n": n, "correct": cor,
        "axis_accuracy": cor_ax / n_ax if n_ax else 0.0,
        "n_axes": n_ax, "correct_axes": cor_ax,
        "region_mention_rate": region_hits / n if n else 0.0,
        "n_skipped_weak": n_skip_weak,
        "n_skipped_empty": n_skip_empty,
        "mag_min_mm": float(mag_min_mm),
        "note": (f"B2 sign-only on |Δ|≥{mag_min_mm}mm; "
                 "extra axes ignored; opposite/hedge/missing axis → wrong"),
    }


def score_direction_acc(items: list[dict]) -> dict:
    """Dispatch train-matched subtypes (prescriptive / B2) + legacy aliases.

    Headline Direction = **prescriptive F1** (main).
    B2 sign-only (filtered) is reported as a diagnostic appendix metric.
    """
    presc, b2, rest = [], [], []
    for it in items:
        kind = (it.get("gold") or {}).get("kind") or \
               (it.get("meta") or {}).get("direction_kind") or "b2"
        if kind in ("prescriptive", "correction"):
            presc.append(it)
        elif kind == "rest_desc":
            rest.append(it)
        else:
            b2.append(it)

    out_presc = score_direction_correction(presc) if presc else None
    out_b2 = score_direction_b2(b2) if b2 else None
    out_rest = score_direction_rest(rest) if rest else None

    # main headline = prescriptive; fall back to legacy mean if no presc
    if out_presc is not None:
        headline = float(out_presc.get("f1") or 0.0)
    else:
        parts = []
        if out_b2 is not None:
            parts.append(float(out_b2.get("accuracy") or 0.0))
        if out_rest is not None:
            parts.append(float(out_rest.get("accuracy") or 0.0))
        headline = sum(parts) / len(parts) if parts else 0.0

    return {
        "accuracy": headline,
        "n": len(items),
        "prescriptive": out_presc,
        "b2": out_b2,
        "rest_desc": out_rest,          # legacy only
        "correction": out_presc,        # alias for old readers
        "n_prescriptive": len(presc),
        "n_b2": len(b2),
        "n_rest_desc": len(rest),
        "n_correction": len(presc),
        "headline_is": "prescriptive_f1",
    }


def score_abstention(items: list[dict], lang: str = "en") -> dict:
    """items already mixed: should_abstain True/False."""
    from metrics import abstention_f1
    return abstention_f1.score(
        [{"pred": it.get("pred"), "should_abstain": bool(it.get("should_abstain"))}
         for it in items],
        lang=lang,
    )


def score_delta_shuf(real: dict, shuf: dict, *, kind: str = "muscle_f1") -> dict:
    """Δ_shuf = score(real mesh) − score(shuffled mesh).

    Default / headline kind is **Muscle F1** (mesh grounding). Legacy value-acc
    keys are still filled when ``kind="value_acc"``.
    """
    if kind == "value_acc":
        a = float(real.get("accuracy") or 0.0)
        b = float(shuf.get("accuracy") or 0.0)
        return {
            "kind": "value_acc",
            "value_acc_real": a,
            "value_acc_shuf": b,
            "delta_shuf": a - b,
            "n_real": real.get("n"),
            "n_shuf": shuf.get("n"),
        }
    a = float(real.get("f1") or 0.0)
    b = float(shuf.get("f1") or 0.0)
    return {
        "kind": "muscle_f1",
        "muscle_f1_real": a,
        "muscle_f1_shuf": b,
        "delta_shuf": a - b,
        "n_real": real.get("n"),
        "n_shuf": shuf.get("n"),
    }


def score_all(jobs: list[dict], lang: str = "en",
              shuf_jobs: list[dict] | None = None,
              value_tol: float | dict = VALUE_TOL) -> dict:
    by = defaultdict(list)
    for j in jobs:
        by[j["family"]].append(j)

    muscle = score_muscle_f1(by.get("muscle_set", []))
    value = score_value_acc(by.get("value", []), tol=value_tol)
    direction = score_direction_acc(by.get("direction", []))

    # abstention: positives + answerable battery as negatives (over-refusal)
    abs_items = list(by.get("abstention", []))
    for fam in ("muscle_set", "value", "direction"):
        for it in by.get(fam, []):
            abs_items.append({**it, "should_abstain": False})
    abstention = score_abstention(abs_items, lang=lang)

    out = {
        "muscle_f1": muscle,
        "value_acc": value,
        "direction_acc": direction,
        "abstention_f1": abstention,
        "utility": {"n": len(by.get("utility", [])),
                    "note": "GPT-J / Human — see judge_payloads.jsonl"},
    }

    if shuf_jobs is not None:
        shuf_mus = [j for j in shuf_jobs if j["family"] == "muscle_set"]
        shuf_val = [j for j in shuf_jobs if j["family"] == "value"]
        # Headline Δ_shuf = Muscle F1 drop under shuffled mesh.
        if shuf_mus:
            muscle_shuf = score_muscle_f1([
                {**j, "pred": j.get("pred_shuf", j.get("pred", ""))} for j in shuf_mus
            ])
            out["muscle_f1_shuf"] = muscle_shuf
            out["delta_shuf"] = score_delta_shuf(muscle, muscle_shuf, kind="muscle_f1")
        elif shuf_val:
            # legacy fallback if only value shuf preds are present
            value_shuf = score_value_acc([
                {**j, "pred": j.get("pred_shuf", j.get("pred", ""))} for j in shuf_val
            ], tol=value_tol)
            out["value_acc_shuf"] = value_shuf
            out["delta_shuf"] = score_delta_shuf(value, value_shuf, kind="value_acc")
        if shuf_val and "value_acc_shuf" not in out:
            out["value_acc_shuf"] = score_value_acc([
                {**j, "pred": j.get("pred_shuf", j.get("pred", ""))} for j in shuf_val
            ], tol=value_tol)

    b2 = direction.get("b2") or {}
    out["headline"] = {
        "Muscle F1": muscle.get("f1"),
        "Value acc": value.get("accuracy"),
        # Direction headline = Prescriptive F1 (train-matched main metric)
        "Direction acc": direction.get("accuracy"),
        "Direction(=prescriptive F1)": direction.get("accuracy"),
        "B2-sign": b2.get("accuracy"),
        "B2-sign n": b2.get("n"),
        "Abstention F1": abstention.get("f1"),
        "Δ_shuf": (out.get("delta_shuf") or {}).get("delta_shuf"),
    }
    return out


def _f1(tp, fp, fn) -> float:
    return prf(tp, fp, fn)["f1"]


def build_utility_judge_payload(items: list[dict], out_path) -> dict:
    """Write GPT-J / Human judging payloads (no API call)."""
    from pathlib import Path
    import json
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            if it.get("family") != "utility":
                continue
            fh.write(json.dumps({
                "uid": it["uid"],
                "mesh_index": it["mesh_index"],
                "question": it["question"],
                "pred": it.get("pred", ""),
                "rubric": "utility_open",
                "judge": ["gpt-j", "human"],
            }, ensure_ascii=False) + "\n")
            n += 1
    return {"path": str(path), "n": n}
