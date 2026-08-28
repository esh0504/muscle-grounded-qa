#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Set-1 probe dataset (Metircs.md / SET1_PROBE_SPEC.md).

  python tools/build_set1_probe.py
  python tools/build_set1_probe.py --n-mesh 500 --out DATA/unseentest/set1_probe.jsonl

Deterministic gold from 11-D activation + geometry.
Direction matches training distribution; both kinds per mesh:
  · prescriptive (contract more / relax) — like PH ``prescriptive`` turns
  · B2 intervention (front/mid/back + raise/lower + dx/dz) — like PH ``B2`` turns
  → with ``--n-mesh 500``: Dir = 500 presc + 500 B2
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from datasets.mesh_dataset import MUSCLE_NAMES
from datasets.mesh_store import MESH_ROOT, MeshStore
ACTIVE_EPS = 1e-4
MOVE_EPS_MM = 0.5
MM = 1000.0
ANAT = {"anterior_sign": -1, "vertical_sign": +1}
REGIONS = ("tip", "body", "root", "dorsum")  # legacy geometry masks
FMB = ("front", "mid", "back")  # training region_disp / B2 / Value
B2_DELTAS = (0.15, -0.15, 0.10, -0.10)  # training B2 intervention steps
# A1-style feature scalars (from properties.jsonl when feat_ok)
FEAT_SLOTS = (
    # loosened Value tols (mm ±2.0; norm ±0.1; cd_min kept tighter)
    ("cl_t", 0.1), ("cd_min", 0.03), ("peak_xn", 0.1),
    ("peak_z", 0.1), ("doming", 0.1), ("tilt", 0.1),
)
# region_disp.npz components → mm (same as scale_qa)
REGION_COMPS = ("dx", "dz", "mag")
REGION_TOL_MM = 2.0  # was 0.5; ≈50% Value acc on ours_en

# --------------------------------------------------------------------------- #
# geometry (same masks as dummy/qa_gen/fact_extraction.py)
# --------------------------------------------------------------------------- #
def region_masks(rest: np.ndarray) -> dict[str, np.ndarray]:
    x, y, z = rest[:, 0], rest[:, 1], rest[:, 2]
    tip = x <= np.percentile(x, 20)
    root = x >= np.percentile(x, 80)
    body = ~tip & ~root
    dorsum = (~tip) & (z >= np.percentile(z, 70))
    return {"tip": tip, "body": body, "root": root, "dorsum": dorsum}


def region_motion_mm(disp: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, dict]:
    """disp = (verts_b - verts_a) in metres → per-region forward/up mm."""
    out = {}
    for name in REGIONS:
        d = disp[masks[name]]
        if d.size == 0:
            out[name] = {"forward_mm": 0.0, "up_mm": 0.0, "mean_mag_mm": 0.0}
            continue
        mean = d.mean(axis=0)
        out[name] = {
            "forward_mm": round(float(ANAT["anterior_sign"] * mean[0] * MM), 2),
            "up_mm": round(float(ANAT["vertical_sign"] * mean[2] * MM), 2),
            "mean_mag_mm": round(float(np.linalg.norm(d, axis=1).mean() * MM), 2),
        }
    return out


def direction_labels(motion: dict[str, dict], eps: float = MOVE_EPS_MM) -> dict[str, list[str]]:
    """Per region: advance/retract and/or elevate/descend, else preserve."""
    labels = {}
    for name in REGIONS:
        m = motion[name]
        dirs: list[str] = []
        fw, up = m["forward_mm"], m["up_mm"]
        if abs(fw) >= eps:
            dirs.append("advance" if fw > 0 else "retract")
        if abs(up) >= eps:
            dirs.append("elevate" if up > 0 else "descend")
        labels[name] = dirs if dirs else ["preserve"]
    return labels


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_test_indices(mesh_root: Path, test_file: str | Path | None = None) -> list[int]:
    """평가 mesh 목록. 기본은 `<mesh_root>/test.txt`.

    E2(앵커 held-out)는 `mesh_e2/heldout.txt` 를 평가셋으로 써야 하는데, topology·verts·
    활성 CSV 는 원본 `DATA/mesh` 에서 읽어야 한다. 그래서 목록 파일만 따로 받는다.
    """
    path = Path(test_file) if test_file else (mesh_root / "test.txt")
    if not path.is_file():
        raise SystemExit(f"[probe] 평가 mesh 목록이 없다: {path}")
    return [int(x) for x in path.read_text().split() if x.strip()]


def load_muscle_table(mesh_root: Path) -> dict[int, dict]:
    out = {}
    with (mesh_root / "pool_meta.csv").open() as fh:
        for row in csv.DictReader(fh):
            idx = int(row["index"])
            act = {m: float(row[m]) for m in MUSCLE_NAMES}
            out[idx] = {
                "n_active": int(row["n_active"]),
                "section": row.get("section", ""),
                "phoneme": row.get("phoneme", ""),
                "detail": row.get("detail", ""),
                "activations": act,
                "active": [m for m in MUSCLE_NAMES if act[m] > ACTIVE_EPS],
                "vec": np.asarray([act[m] for m in MUSCLE_NAMES], dtype=np.float32),
            }
    return out


def load_centers(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as fh:  # strip BOM on first header
        for row in csv.DictReader(fh):
            vec = np.asarray([float(row[m]) for m in MUSCLE_NAMES], dtype=np.float32)
            rows.append({
                "key": row["key"],
                "ipa": row.get("ipa", row["key"]),
                "label_ko": row.get("label_ko", row["key"]),
                "category": row.get("category", ""),
                "vec": vec,
            })
    return rows


def named_anchors(centers: list[dict]) -> list[dict]:
    """IPA 가 있는 앵커만. `functional` 6행은 `ipa == "-"` 라 EN 질문이 망가진다.

    EN 템플릿이 `f"/{ipa}/"` 로 라벨을 만들기 때문에 functional 앵커는
    "…to produce **/-/**?" / "To get to /-/," 가 되어 **목표가 지정되지 않는다**.
    KO 는 `label_ko`(후퇴/돌출/…)를 쓰므로 멀쩡하다 — 그래서 EN 만 걸린다.
    """
    return [c for c in centers
            if str(c.get("ipa", "")).strip() not in ("", "-")
            and str(c.get("category", "")).strip() != "functional"]


def build_act_index(table: dict[int, dict]):
    idxs = np.asarray(sorted(table), dtype=np.int64)
    mat = np.stack([table[int(i)]["vec"] for i in idxs], axis=0)
    return idxs, mat


def nearest_mesh(vec: np.ndarray, idxs: np.ndarray, mat: np.ndarray,
                 exclude: set[int] | None = None) -> int:
    d = np.linalg.norm(mat - vec[None, :], axis=1)
    if exclude:
        for i in exclude:
            # rare; mask matching rows
            d[idxs == i] = np.inf
    return int(idxs[int(np.argmin(d))])


def n_act_bin(n: int) -> str:
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def stratified_sample(test_ids: list[int], table: dict[int, dict],
                      n_mesh: int, seed: int,
                      prefer: set[int] | None = None,
                      mode: str = "balanced") -> list[int]:
    """Sample evaluation meshes. Prefer `prefer` indices (e.g. A1 feat_ok).

    mode:
      balanced — n_act 1/2/3+ 를 1/3씩. 출고된 probe 의 규칙이므로 **기본값**이다.
      3plus    — n_act 3+ 만. 앵커 held-out 평가용: 앵커 폐포는 거의 전부 3+ 라
                 (mesh_e2/heldout.txt 40,611 중 bin1=0 · bin2=50) balanced 가 물리적으로
                 불가능하다. 대조군 probe 도 **반드시 같은 mode 로** 구울 것 — n_act 구간별
                 난이도가 크게 달라서(ours_en EM 65.7/88.3/65.1) 섞으면 Δ 가 오염된다.
      natural  — 풀의 자연 분포 그대로.
    """
    rng = random.Random(seed)
    prefer = prefer or set()
    buckets: dict[str, list[int]] = defaultdict(list)
    for i in test_ids:
        rec = table.get(i)
        if not rec or rec["n_active"] <= 0:
            continue
        buckets[n_act_bin(rec["n_active"])].append(i)

    def order(pool: list[int]) -> list[int]:
        pref = [i for i in pool if i in prefer]
        rest = [i for i in pool if i not in prefer]
        rng.shuffle(pref)
        rng.shuffle(rest)
        return pref + rest

    if mode in ("3plus", "natural"):
        pool = buckets["3+"] if mode == "3plus" else [i for b in buckets.values() for i in b]
        ordered = order(pool)
        if len(ordered) < n_mesh:
            raise RuntimeError(f"mode={mode}: need {n_mesh}, have {len(ordered)}")
        chosen = ordered[:n_mesh]
        rng.shuffle(chosen)
        return chosen

    per = n_mesh // 3
    leftover = n_mesh - 3 * per
    chosen = []
    for bi, b in enumerate(("1", "2", "3+")):
        ordered = order(buckets[b][:])
        take = per + (1 if bi < leftover else 0)
        if len(ordered) < take:
            raise RuntimeError(
                f"n_act bin {b}: need {take}, have {len(ordered)}. "
                f"앵커 held-out 평가라면 --nact-balance 3plus 를 쓰고 대조군도 같이 다시 구울 것")
        chosen.extend(ordered[:take])
    rng.shuffle(chosen)
    return chosen


# --------------------------------------------------------------------------- #
# question templates
# --------------------------------------------------------------------------- #
TEMPLATES = {
    "en": {
        "muscle_set": {
            "q": "Which muscles are active in this tongue shape?",
            "leadin": "The active muscles are",
        },
        "value": {
            "cl_t": {
                "q": ("Normalizing the dorsum by palate arc length: what is the "
                      "maximum-constriction location cl_t (0=front, 1=back)?"),
                "leadin": "cl_t =",
            },
            "cd_min": {
                "q": ("What is the minimum gap to the palate cd_min "
                      "(normalized by palate arc length)?"),
                "leadin": "cd_min =",
            },
            "peak_xn": {
                "q": ("What is the dorsal peak anteroposterior position peak_xn "
                      "(0=front, 1=back)?"),
                "leadin": "peak_xn =",
            },
            "peak_z": {
                "q": "What is the dorsal peak height peak_z (normalized)?",
                "leadin": "peak_z =",
            },
            "doming": {
                "q": "What is the tongue doming value for this shape?",
                "leadin": "doming =",
            },
            "tilt": {
                "q": "What is the front-to-back tilt of this tongue shape?",
                "leadin": "tilt =",
            },
            "region_mm": {
                "q": ("What is the {region}-region {comp_name} in millimetres "
                      "(rest → this shape)?"),
                "leadin": "{region}_{comp} =",
            },
        },
        "direction": {
            "prescriptive": {
                "q": (
                    "How should the tongue be adjusted from this state to "
                    "produce {label}?"
                ),
                "leadin": "To get to {label},",
            },
            "b2": {
                "q": (
                    "From this state, if the muscle activation is changed by "
                    "{muscle}{delta:+.2f}, how will the tongue shape change?"
                ),
                "leadin": "The",
            },
        },
        "abstention": [
            ("Can the unique muscle activation pattern be determined from this "
             "tongue shape alone?", ""),
            ("Is there only one muscle combination that could produce a tongue "
             "shape like this?", ""),
            ("From geometry alone, can you name the exact activation vector that "
             "created this shape?", ""),
            ("Does this shape uniquely identify which muscles were co-activated "
             "and at what levels?", ""),
            ("Can motor equivalence be ruled out for this articulation?", ""),
        ],
        "utility": [
            "In one sentence, how would you describe this tongue posture for a clinician?",
            "Which articulatory goal does this shape most likely serve?",
            "Name one speech-therapy cue that matches this tongue configuration.",
            "Is the tongue tip more anterior or posterior than a neutral rest pose?",
            "Would this posture help or hinder producing a high front vowel? Why?",
            "Summarize the dominant muscle action visible in this mesh.",
            "What should a learner feel if asked to imitate this tongue shape?",
            "Is the dorsum raised, lowered, or roughly neutral here?",
            "Give a short coaching prompt to reproduce this posture.",
            "Which region (tip/body/root/dorsum) moved the most from rest?",
        ],
    },
    "ko": {
        "muscle_set": {
            "q": "이 혀 모양에서 활성화된 근육은 무엇인가?",
            "leadin": "활성 근육은",
        },
        "value": {
            "cl_t": {
                "q": "구개 호길이로 정규화했을 때 최대 협착 위치 cl_t(0=앞, 1=뒤)는?",
                "leadin": "cl_t =",
            },
            "cd_min": {
                "q": "구개까지 최소 간격 cd_min(정규화)은 얼마인가?",
                "leadin": "cd_min =",
            },
            "peak_xn": {
                "q": "설배 피크 전후 위치 peak_xn(0=앞, 1=뒤)은?",
                "leadin": "peak_xn =",
            },
            "peak_z": {
                "q": "설배 피크 높이 peak_z(정규화)는?",
                "leadin": "peak_z =",
            },
            "doming": {
                "q": "이 혀 모양의 doming 값은?",
                "leadin": "doming =",
            },
            "tilt": {
                "q": "이 혀 모양의 front-to-back tilt는?",
                "leadin": "tilt =",
            },
            "region_mm": {
                "q": "rest 대비 {region} 부위 {comp_name}(mm)는?",
                "leadin": "{region}_{comp} =",
            },
        },
        "direction": {
            "prescriptive": {
                "q": "이 혀 상태에서 조음 «{label}» 를 만들려면 어떻게 조정해야 하는가?",
                "leadin": "«{label}» 로 가려면,",
            },
            "b2": {
                "q": (
                    "이 상태에서 근육 활성을 {muscle}{delta:+.2f}로 바꾸면 "
                    "혀 모양이 어떻게 변하는가?"
                ),
                "leadin": "",
            },
        },
        "abstention": [
            ("혀 모양만으로 근육 활성 패턴을 유일하게 결정할 수 있는가?", ""),
            ("이 모양을 만들 수 있는 근육 조합이 하나뿐인가?", ""),
            ("형상만 보고 정확한 활성 벡터를 말할 수 있는가?", ""),
            ("이 조음에서 운동 등가를 배제할 수 있는가?", ""),
            ("기하만으로 공동 활성 근육과 세기를 확정할 수 있는가?", ""),
        ],
        "utility": [
            "임상가가 이해하도록 이 혀 자세를 한 문장으로 설명하라.",
            "이 모양이 가장 잘 맞는 조음 목표는 무엇인가?",
            "이 혀 설정에 맞는 언어치료 큐를 하나 제시하라.",
            "혀 끝이 중립 자세보다 앞인가 뒤인가?",
            "고전설 모음 산출에 도움이 되는가, 방해가 되는가? 이유를 말하라.",
            "이 메쉬에서 보이는 지배적 근육 작용을 요약하라.",
            "학습자가 이 자세를 따라 할 때 느껴야 할 감각은?",
            "설배는 상승·하강·중립 중 어디에 가까운가?",
            "이 자세를 재현하도록 짧은 코칭 문장을 적어라.",
            "rest 대비 tip/body/root/dorsum 중 어디가 가장 많이 움직였는가?",
        ],
    },
}


def format_muscle_gold(muscles: list[str]) -> str:
    return ", ".join(muscles)


CORR_ACT_EPS = 0.05  # |Δactivation| below this → ignore


def correction_from_acts(cur: dict[str, float], tgt: dict[str, float],
                         eps: float = CORR_ACT_EPS) -> tuple[list[str], list[str]]:
    """Muscles to contract more / relax to go from current activation → target."""
    inc, dec = [], []
    for m in MUSCLE_NAMES:
        d = float(tgt.get(m, 0.0)) - float(cur.get(m, 0.0))
        if d >= eps:
            inc.append(m)
        elif d <= -eps:
            dec.append(m)
    return inc, dec


def format_prescriptive_gold(inc: list[str], dec: list[str]) -> str:
    """Training-like surface: ``contract more: A, B; relax: C``."""
    parts = []
    if inc:
        parts.append("contract more: " + ", ".join(inc))
    if dec:
        parts.append("relax: " + ", ".join(dec))
    return "; ".join(parts) if parts else "(no change)"


def format_prescriptive_gold_ko(inc: list[str], dec: list[str]) -> str:
    parts = []
    if inc:
        parts.append("더 수축: " + ", ".join(inc))
    if dec:
        parts.append("이완: " + ", ".join(dec))
    return "; ".join(parts) if parts else "(변화 없음)"


def _vert_surf(concept: str, lang: str) -> str:
    """Canonical elevate/descend → training surface raise/lower (EN) or 상승/하강."""
    if lang == "ko":
        return {"elevate": "상승", "descend": "하강", "advance": "전진",
                "retract": "후퇴", "preserve": "유지"}.get(concept, concept)
    return {"elevate": "raise", "descend": "lower"}.get(concept, concept)


def fmb_delta_from_region_disp(
        rd0: dict[str, dict[str, float]] | None,
        rd1: dict[str, dict[str, float]] | None,
        eps: float = MOVE_EPS_MM,
) -> tuple[str, dict[str, list[str]], dict[str, dict[str, float]]] | None:
    """Compare two region_disp rows → dominant FMB region + direction labels."""
    if not rd0 or not rd1:
        return None
    per: dict[str, dict[str, float]] = {}
    dirs: dict[str, list[str]] = {}
    best_r, best_mag = "front", -1.0
    for r in FMB:
        dx = float(rd1[r]["dx"]) - float(rd0[r]["dx"])
        dz = float(rd1[r]["dz"]) - float(rd0[r]["dz"])
        mag = (dx * dx + dz * dz) ** 0.5
        labs: list[str] = []
        if abs(dx) >= eps:
            labs.append("advance" if dx > 0 else "retract")
        if abs(dz) >= eps:
            labs.append("elevate" if dz > 0 else "descend")
        if not labs:
            labs = ["preserve"]
        dirs[r] = labs
        per[r] = {"dx": round(dx, 2), "dz": round(dz, 2), "mag": round(mag, 2)}
        if mag > best_mag:
            best_mag, best_r = mag, r
    if best_mag < eps and dirs[best_r] == ["preserve"]:
        return None
    return best_r, dirs, per


def format_b2_gold(dominant: str, dirs: dict[str, list[str]],
                   per: dict[str, dict[str, float]], lang: str,
                   *, base_mesh: int) -> str:
    labs = dirs[dominant]
    if labs == ["preserve"]:
        move = "preserve" if lang != "ko" else "유지"
    else:
        move = ", ".join(_vert_surf(c, lang) for c in labs)
    dx = per[dominant]["dx"]
    dz = per[dominant]["dz"]
    if lang == "ko":
        return (f"{dominant} 부위가 주로 {move} "
                f"(변화 dx {dx:.2f}mm, dz {dz:.2f}mm). "
                f"해당 근육 변화만의 순수 효과(다른 근육 고정, 비교 mesh #{base_mesh}).")
    return (f"{dominant} region mainly {move} "
            f"(change dx {dx:.2f}mm, dz {dz:.2f}mm). "
            f"This is the pure effect of that muscle change alone "
            f"(others fixed, vs base mesh #{base_mesh}).")


def pick_b2_intervention(
        cur_acts: dict[str, float],
        act_idxs: np.ndarray,
        act_mat: np.ndarray,
        mi: int,
        region_disp: dict[int, dict],
        rng: random.Random,
) -> dict | None:
    """Find a train-like B2 intervention via nearest-mesh activation edit."""
    cur_vec = np.asarray([cur_acts[m] for m in MUSCLE_NAMES], dtype=np.float32)
    cands: list[tuple[str, float]] = []
    for m in MUSCLE_NAMES:
        a0 = float(cur_acts[m])
        for delta in B2_DELTAS:
            a1 = a0 + delta
            if a1 < -1e-6 or a1 > 1.0 + 1e-6:
                continue
            cands.append((m, float(delta)))
    rng.shuffle(cands)
    rd0 = region_disp.get(mi)
    for m, delta in cands[:40]:
        tgt = cur_vec.copy()
        tgt[MUSCLE_NAMES.index(m)] = float(np.clip(cur_vec[MUSCLE_NAMES.index(m)] + delta, 0.0, 1.0))
        j = nearest_mesh(tgt, act_idxs, act_mat, exclude={mi})
        got = fmb_delta_from_region_disp(rd0, region_disp.get(j))
        if got is None:
            continue
        dom, dirs, per = got
        if dirs[dom] == ["preserve"]:
            continue
        return {
            "muscle": m, "delta": delta, "base_mesh": j,
            "dominant_region": dom, "directions": dirs, "delta_mm": per,
        }
    return None


def format_value_text(val: float) -> str:
    # match training QA precision: 2–3 decimals
    if abs(val) >= 10:
        return f"{val:.1f}"
    s = f"{val:.3f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-", "-0") else "0"


def load_region_disp_mm(path: Path) -> dict[int, dict[str, dict[str, float]]]:
    """index → {front/mid/back: {dx,dz,mag} in mm}."""
    d = np.load(path, allow_pickle=True)
    cols = [str(c) for c in d["cols"]]
    out = {}
    for i, row in zip(d["idxs"], d["disp"]):
        reg = {}
        for name in FMB:
            reg[name] = {
                "dx": round(float(row[cols.index(f"{name}_dx")]) * MM, 2),
                "dz": round(float(row[cols.index(f"{name}_dz")]) * MM, 2),
                "mag": round(float(row[cols.index(f"{name}_mag")]) * MM, 2),
            }
        out[int(i)] = reg
    return out


def load_a1_features(path: Path) -> dict[int, dict[str, float]]:
    """index → A1 feature dict (only feat_ok rows)."""
    out = {}
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if not r.get("feat_ok") or not r.get("features"):
                continue
            feats = r["features"]
            out[int(r["index"])] = {
                k: float(feats[k]) for k, _ in FEAT_SLOTS if k in feats and feats[k] is not None
            }
    return out


_COMP_NAME = {
    "en": {"dx": "AP displacement dx", "dz": "vertical displacement dz",
           "mag": "displacement magnitude"},
    "ko": {"dx": "전후 변위 dx", "dz": "상하 변위 dz", "mag": "변위 크기"},
}


def pick_trainlike_value(mesh_index: int, region_mm: dict, feats: dict | None,
                         rng: random.Random, lang: str) -> dict:
    """Pick one training-distribution scalar slot (A1 feature or region_disp mm)."""
    feat_cands = []
    if feats:
        for key, tol in FEAT_SLOTS:
            if key not in feats:
                continue
            feat_cands.append({
                "slot": key, "quantity": key, "region": None, "comp": None,
                "value": round(float(feats[key]), 3),
                "tol": tol, "unit": "norm", "q_kwargs": {},
            })
    region_cands = []
    if region_mm:
        for reg in FMB:
            for comp in REGION_COMPS:
                v = float(region_mm[reg][comp])
                if abs(v) < MOVE_EPS_MM and comp != "mag":
                    continue
                region_cands.append({
                    "slot": "region_mm", "quantity": f"{reg}_{comp}",
                    "region": reg, "comp": comp, "value": v,
                    "tol": REGION_TOL_MM, "unit": "mm",
                    "q_kwargs": {
                        "region": reg, "comp": comp,
                        "comp_name": _COMP_NAME[lang][comp],
                    },
                })
    # Prefer A1 features when available (~70%) — matches shape_desc training
    if feat_cands and (not region_cands or rng.random() < 0.70):
        return rng.choice(feat_cands)
    pool = region_cands or feat_cands
    if not pool:
        raise RuntimeError(f"no value slots for mesh {mesh_index}")
    return rng.choice(pool)


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build_records(args) -> list[dict]:
    mesh_root = Path(args.mesh_root)
    store = MeshStore(mesh_root)
    masks = region_masks(store.rest)
    table = load_muscle_table(mesh_root)
    test_ids = load_test_indices(mesh_root, getattr(args, "test_file", None))
    centers = load_centers(Path(args.centers))
    region_disp = load_region_disp_mm(Path(args.region_disp))
    a1_feats = load_a1_features(Path(args.properties))
    lang = args.lang
    T = TEMPLATES[lang]
    rng = random.Random(args.seed)
    # Direction B2 sampling must not advance ``rng`` — otherwise Value/utility
    # slots drift from the previous probe. Prescriptive keeps using ``rng``
    # with the same choice pattern as the old ↑/↓ correction branch.
    rng_dir = random.Random(args.seed + 91)
    # prescriptive 앵커 풀을 바꿔도 다른 family 가 밀리면 안 된다 (rng_dir 과 같은 이유).
    # 아래에서 `rng` 는 예전과 똑같이 소비하고, 실제 앵커만 이 스트림에서 다시 뽑는다.
    rng_anchor = random.Random(args.seed + 137)
    anchor_pool = centers if args.anchor_pool == "all" else named_anchors(centers)
    if args.anchor_pool != "all":
        dropped = sorted({c["key"] for c in centers} - {c["key"] for c in anchor_pool})
        print(f"  anchor_pool=named: {len(anchor_pool)}/{len(centers)} 앵커 사용 "
              f"(제외: {', '.join(dropped)})")

    meshes = stratified_sample(
        test_ids, table, args.n_mesh, args.seed, prefer=set(a1_feats),
        mode=getattr(args, "nact_balance", "balanced"))
    act_idxs, act_mat = build_act_index(table)
    print(f"  sampled meshes with A1 features: "
          f"{sum(1 for m in meshes if m in a1_feats)}/{len(meshes)}")
    records: list[dict] = []
    verts_cache: dict[int, np.ndarray] = {}

    def verts(i: int) -> np.ndarray:
        if i not in verts_cache:
            verts_cache[i] = store.rest + store.disp(i)
        return verts_cache[i]

    for mi in meshes:
        rec = table[mi]
        actives = list(rec["active"])
        n_bin = n_act_bin(rec["n_active"])

        # ---- muscle_set ----------------------------------------------------
        ms = T["muscle_set"]
        gold_m = sorted(actives, key=lambda m: (-rec["activations"][m], m))
        records.append({
            "uid": f"m{mi:06d}_muscle_set",
            "family": "muscle_set",
            "mesh_index": mi,
            "n_act": rec["n_active"],
            "n_act_bin": n_bin,
            "lang": lang,
            "question": ms["q"],
            "answer_leadin": ms["leadin"],
            "gold": {"muscles": gold_m, "value": None, "directions": None,
                     "text": format_muscle_gold(gold_m)},
            "should_abstain": False,
            "score_auto": True,
            "meta": {"section": rec["section"], "phoneme": rec["phoneme"]},
        })

        # ---- value (training-like scalars: region_disp mm + A1 features) ---
        rest_disp = store.disp(mi)
        self_motion = region_motion_mm(rest_disp, masks)  # tip/body/root/dorsum for rest_desc
        slot = pick_trainlike_value(
            mi, region_disp.get(mi), a1_feats.get(mi), rng, lang)
        if slot["slot"] == "region_mm":
            vt = T["value"]["region_mm"]
            q = vt["q"].format(**slot["q_kwargs"])
            leadin = vt["leadin"].format(**slot["q_kwargs"])
        else:
            vt = T["value"][slot["slot"]]
            q, leadin = vt["q"], vt["leadin"]
        gval = slot["value"]
        records.append({
            "uid": f"m{mi:06d}_value",
            "family": "value",
            "mesh_index": mi,
            "n_act": rec["n_active"],
            "n_act_bin": n_bin,
            "lang": lang,
            "question": q,
            "answer_leadin": leadin,
            "gold": {
                "muscles": None, "value": gval, "directions": None,
                "text": format_value_text(gval),
                "unit": slot["unit"], "quantity": slot["quantity"],
                "region": slot["region"], "tol": slot["tol"],
            },
            "should_abstain": False,
            "score_auto": True,
            "meta": {
                "quantity": slot["quantity"], "unit": slot["unit"],
                "tol": slot["tol"], "slot": slot["slot"],
                "region_disp_mm": region_disp.get(mi),
                "a1_features": a1_feats.get(mi),
                "section": rec["section"],
                "value_kind": "trainlike_scalar",
            },
        })

        # ---- direction: both kinds per mesh (prescriptive + B2) ------------
        # With n_mesh=500 → Dir = 500 presc + 500 B2.
        def draw_anchor(pool, r):
            a = r.choice(pool)
            k = nearest_mesh(a["vec"], act_idxs, act_mat, exclude={mi})
            i, d = correction_from_acts(rec["activations"], table[k]["activations"])
            for _ in range(5):
                if i or d:
                    break
                a = r.choice(pool)
                k = nearest_mesh(a["vec"], act_idxs, act_mat, exclude={mi})
                i, d = correction_from_acts(rec["activations"], table[k]["activations"])
            return a, k, i, d

        # `rng` 는 예전과 동일하게 소비한다 — 이후 value/abstention/utility 가 안 밀리게.
        anchor, k_idx, inc, dec = draw_anchor(centers, rng)
        if anchor_pool is not centers:
            anchor, k_idx, inc, dec = draw_anchor(anchor_pool, rng_anchor)
        label = (anchor["label_ko"] if lang == "ko"
                 else f"/{anchor['ipa']}/")
        dt = T["direction"]["prescriptive"]
        gtext = (format_prescriptive_gold_ko(inc, dec) if lang == "ko"
                 else format_prescriptive_gold(inc, dec))
        records.append({
            "uid": f"m{mi:06d}_direction_prescriptive",
            "family": "direction",
            "mesh_index": mi,
            "n_act": rec["n_active"],
            "n_act_bin": n_bin,
            "lang": lang,
            "question": dt["q"].format(label=label),
            "answer_leadin": dt["leadin"].format(label=label),
            "gold": {
                "kind": "prescriptive",
                "muscles": None, "value": None, "directions": None,
                "gold_inc": inc, "gold_dec": dec,
                "text": gtext,
            },
            "should_abstain": False,
            "score_auto": True,
            "meta": {
                "direction_kind": "prescriptive",
                "anchor_key": anchor["key"],
                "anchor_ipa": anchor["ipa"],
                "anchor_mesh": k_idx,
                "section": rec["section"],
            },
        })

        inter = pick_b2_intervention(
            rec["activations"], act_idxs, act_mat, mi, region_disp, rng_dir)
        if inter is None:
            # fallback: rest→current on FMB of this mesh (still train vocab)
            rd = region_disp.get(mi) or {r: {"dx": 0, "dz": 0, "mag": 0} for r in FMB}
            got = fmb_delta_from_region_disp(
                {r: {"dx": 0.0, "dz": 0.0, "mag": 0.0} for r in FMB}, rd)
            if got is None:
                dom, dirs, per = "front", {r: ["preserve"] for r in FMB}, {
                    r: {"dx": 0.0, "dz": 0.0, "mag": 0.0} for r in FMB}
                base_mesh = mi
                muscle, delta = "GGP", 0.15
            else:
                dom, dirs, per = got
                base_mesh = mi
                # synthetic delta label for the question only
                active = [m for m in MUSCLE_NAMES if rec["activations"][m] > ACTIVE_EPS]
                muscle = rng_dir.choice(active or list(MUSCLE_NAMES))
                delta = float(rng_dir.choice(B2_DELTAS))
        else:
            dom = inter["dominant_region"]
            dirs = inter["directions"]
            per = inter["delta_mm"]
            base_mesh = inter["base_mesh"]
            muscle, delta = inter["muscle"], inter["delta"]
        dt = T["direction"]["b2"]
        q = dt["q"].format(muscle=muscle, delta=float(delta))
        gtext = format_b2_gold(dom, dirs, per, lang, base_mesh=base_mesh)
        records.append({
            "uid": f"m{mi:06d}_direction_b2",
            "family": "direction",
            "mesh_index": mi,
            "n_act": rec["n_active"],
            "n_act_bin": n_bin,
            "lang": lang,
            "question": q,
            "answer_leadin": dt["leadin"],
            "gold": {
                "kind": "b2",
                "muscles": None, "value": None,
                "directions": dirs,
                "dominant_region": dom,
                "gold_inc": None, "gold_dec": None,
                "text": gtext,
                "dx": per[dom]["dx"], "dz": per[dom]["dz"],
            },
            "should_abstain": False,
            "score_auto": True,
            "meta": {
                "direction_kind": "b2",
                "muscle": muscle,
                "delta": delta,
                "base_mesh": base_mesh,
                "dominant_region": dom,
                "delta_mm": per,
                "section": rec["section"],
                "fallback_rest_fmb": inter is None,
            },
        })

    # ---- abstention (~50), mesh-independent; attach unused test meshes -----
    n_abs = args.n_abstention
    abs_pool = [i for i in test_ids if i not in set(meshes) and table.get(i, {}).get("n_active", 0) > 0]
    rng.shuffle(abs_pool)
    abs_qs = T["abstention"]
    for k in range(n_abs):
        q, leadin = abs_qs[k % len(abs_qs)]
        mi = abs_pool[k % len(abs_pool)]
        records.append({
            "uid": f"abs_{k:03d}",
            "family": "abstention",
            "mesh_index": mi,
            "n_act": table[mi]["n_active"],
            "n_act_bin": n_act_bin(table[mi]["n_active"]),
            "lang": lang,
            "question": q,
            "answer_leadin": leadin,
            "gold": {"muscles": None, "value": None, "directions": None,
                     "text": "cannot be uniquely determined"},
            "should_abstain": True,
            "score_auto": True,
            "meta": {"template_id": k % len(abs_qs)},
        })

    # ---- utility (30) — GPT-J / Human; no auto gold ------------------------
    n_util = args.n_utility
    util_qs = T["utility"]
    util_pool = abs_pool[n_abs:] or abs_pool
    for k in range(n_util):
        mi = util_pool[k % len(util_pool)]
        q = util_qs[k % len(util_qs)]
        records.append({
            "uid": f"util_{k:03d}",
            "family": "utility",
            "mesh_index": mi,
            "n_act": table[mi]["n_active"],
            "n_act_bin": n_act_bin(table[mi]["n_active"]),
            "lang": lang,
            "question": q,
            "answer_leadin": "",
            "gold": {"muscles": None, "value": None, "directions": None, "text": None},
            "should_abstain": False,
            "score_auto": False,
            "meta": {"judge": "gpt_j_human", "template_id": k % len(util_qs)},
        })

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-root", default=str(MESH_ROOT))
    ap.add_argument("--centers", default="DATA/aux/centers.csv")
    ap.add_argument("--region-disp", default="DATA/aux/region_disp.npz")
    ap.add_argument("--properties", default="DATA/aux/properties.jsonl")
    ap.add_argument("--out", default="DATA/unseentest/set1_probe.jsonl")
    ap.add_argument("--lang", choices=("en", "ko"), default="en")
    ap.add_argument("--n-mesh", type=int, default=500,
                    help="meshes → muscle_set / value / direction each get this many")
    ap.add_argument("--n-abstention", type=int, default=500)
    ap.add_argument("--n-utility", type=int, default=30)
    ap.add_argument("--test-file", default=None,
                    help="평가 mesh 목록 파일. 기본은 <mesh-root>/test.txt. "
                         "E2 는 DATA/mesh_e2/heldout.txt 를 준다")
    ap.add_argument("--anchor-pool", choices=("all", "named"), default="all",
                    help="prescriptive 목표 앵커 풀. named = IPA 가 있는 20개만 "
                         "(functional 6개 제외 — EN 에서 '/-/' 가 되어 목표가 "
                         "지정되지 않는다). all = 기존 26개, 출고된 probe 재현용")
    ap.add_argument("--nact-balance", choices=("balanced", "3plus", "natural"),
                    default="balanced",
                    help="mesh 표본의 n_act 규칙. balanced=1/2/3+ 를 1/3씩(출고 probe 규칙). "
                         "3plus=3+ 만 — 앵커 held-out 은 폐포에 bin1 이 0개라 balanced 가 "
                         "불가능하다. 이 경우 **대조군 probe 도 같은 값으로** 다시 구울 것")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = build_records(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    fam = Counter(r["family"] for r in records)
    bins = Counter(r["n_act_bin"] for r in records if r["family"] == "muscle_set")
    val_q = Counter((r.get("gold") or {}).get("quantity")
                    for r in records if r["family"] == "value")
    dir_k = Counter((r.get("gold") or {}).get("kind")
                    for r in records if r["family"] == "direction")
    print(f"[set1_probe] wrote {len(records)} → {out}")
    print(f"  families: {dict(fam)}")
    print(f"  mesh n_act_bin (muscle_set): {dict(bins)}")
    print(f"  value quantities: {dict(val_q)}")
    print(f"  direction kinds: {dict(dir_k)}")


if __name__ == "__main__":
    main()
