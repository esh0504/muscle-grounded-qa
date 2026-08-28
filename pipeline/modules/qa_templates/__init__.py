# -*- coding: utf-8 -*-
"""Step 4 — YAML-driven QA template engine.

Every natural-language surface form lives in settings/templates_<lang>.yaml
(user-editable); this module only loads those files, formats the structured
facts into the {placeholders}, and applies the fixed numeric formats that make
the gold answers deterministic and record-verifiable.

Adding a language = adding one settings/templates_<lang>.yaml file; it appears
in BY_LANG automatically and the step-5 generators accept it via --lang.
"""
import glob
import os

import yaml

from ..config import PIPELINE_ROOT
from ..spans import make_mask_spans


class LangTemplates:
    """One language's templates, exposing the interface the generators use."""

    def __init__(self, spec):
        self.LANG = str(spec["lang"])
        f = spec["files"]
        self.PHYSICS_PREFIX = f["physics_prefix"]
        self.FEATURE_A1_PREFIX = f["feature_a1_prefix"]
        self.B3_FILENAME = f["b3_filename"]
        self.FULL = dict(spec["muscle_names"])
        self._act = dict(spec["muscle_actions"])
        self._dir = dict(spec["direction_words"])
        self._p = spec["physics"]
        self._f = spec["feature"]
        self._b = spec["b3"]

        def spans_fn(c):
            return make_mask_spans(
                self.FULL.values(), c["movement_words"], c["region_words"],
                [tuple(x) for x in c["number_roles"]],
                move_mode=c.get("movement_mode", "find"),
                region_mode=c.get("region_mode", "find"),
                ctx=int(c.get("context_chars", 12)))

        self.physics_mask_spans = spans_fn(spec["spans"]["physics"])
        self.feature_mask_spans = spans_fn(spec["spans"]["feature"])

    # ------------------------------------------------------------- physics chain
    def hv(self, dx, dz):
        return (self._dir["advance"] if dx < 0 else self._dir["retract"],
                self._dir["raise"] if dz > 0 else self._dir["lower"])

    def t_attribution(self, active):
        t = self._p["attribution"]
        muscles = t["list_sep"].join(self.FULL[m] for m in active)
        actions = t["action_sep"].join(
            t["action_item"].format(muscle=self.FULL[m], action=self._act[m]) for m in active)
        return t["q"], t["a"].format(muscles=muscles, actions=actions)

    def t_identifiability(self):
        t = self._p["identifiability"]
        return t["q"], t["a"]

    def t_volume(self, vr, br, h, v):
        t = self._p["volume"]
        return t["q"], t["a"].format(vol_ratio=f"{vr:.3f}", region=br, h=h, v=v)

    def t_single(self, m, br, h, v, rd):
        t = self._p["single"]
        vals = dict(muscle=self.FULL[m], action=self._act[m], region=br, h=h, v=v,
                    front_dx=f"{rd[0]:.2f}", front_dz=f"{rd[1]:.2f}",
                    mid_dx=f"{rd[3]:.2f}", mid_dz=f"{rd[4]:.2f}",
                    back_dx=f"{rd[6]:.2f}", back_dz=f"{rd[7]:.2f}")
        return t["q"].format(**vals), t["a"].format(**vals)

    def t_counterfactual(self, dm, base_index, dd, k):
        t = self._p["counterfactual"]
        ki = k * 3
        vals = dict(delta=dm, base=base_index,
                    region=["front", "mid", "back"][k],
                    h=self._dir["retract"] if dd[ki] > 0 else self._dir["advance"],
                    v=self._dir["raise"] if dd[ki + 1] > 0 else self._dir["lower"],
                    dx=f"{dd[ki]:.2f}", dz=f"{dd[ki+1]:.2f}")
        negligible = abs(dd[ki]) + abs(dd[ki + 1]) < 0.1
        a = t["a_negligible"] if negligible else t["a_effect"]
        return t["q"].format(**vals), a.format(**vals)

    def t_prescriptive(self, tgt, target, inc, dec):
        t = self._p["prescriptive"]
        parts = []
        if inc:
            parts.append(t["increase_prefix"] + t["list_sep"].join(self.FULL[m] for m in inc))
        if dec:
            parts.append(t["decrease_prefix"] + t["list_sep"].join(self.FULL[m] for m in dec))
        body = t["part_sep"].join(parts) if parts else t["already_close"]
        return (t["q"].format(target=tgt),
                t["a"].format(target=tgt, label=target.get("label", ""), body=body))

    # --------------------------------------------------------------- feature QA
    def _place_key(self, clt):
        return "front" if clt < 0.34 else "mid" if clt < 0.67 else "back"

    def t_a1(self, f):
        t = self._f
        key = self._place_key(f["cl_t"])
        vals = dict(
            place=t["place_words"][key], place_example=t["place_examples"][key],
            cl_t=f"{f['cl_t']:.2f}", cd_min=f"{f['cd_min']:.3f}",
            peak_xn=f"{f['peak_xn']:.2f}", peak_z=f"{f['peak_z']:.2f}",
            doming=f"{f['doming']:.2f}", tilt=f"{f['tilt']:.2f}",
            h_front=f"{f['h_front']:.2f}", h_mid=f"{f['h_mid']:.2f}", h_back=f"{f['h_back']:.2f}",
            hi_reg=f["hi_reg"], arc_len=f"{f['arc_len']:.2f}", curv_peak=f"{f['curv_peak']:.1f}",
            curv_word=t["curv_words"]["gentle"] if f["curv_peak"] < 120 else t["curv_words"]["sharp"],
            doming_word=t["doming_words"]["flat"] if f["doming"] < 0.08 else t["doming_words"]["domed"])
        return [(t["q1"], t["a1"].format(**vals)),
                (t["q2"], t["a2"].format(**vals)),
                (t["q3"], t["a3"].format(**vals))]

    def q_b3_single(self, mname, lv0, lv1):
        return self._b["q_single"].format(muscle=mname, lv0=f"{lv0:.2f}", lv1=f"{lv1:.2f}")

    def q_b3_effort(self, names, lv0, lv1):
        return self._b["q_effort"].format(muscles=names, lv0=f"{lv0:.2f}", lv1=f"{lv1:.2f}")

    def b3_seq(self, levels, resps):
        t = self._b
        return t["seq_sep"].join(
            t["seq_item"].format(lv=f"{lv:.2f}", resp=f"{rp:.2f}") for lv, rp in zip(levels, resps))

    def a_b3(self, tier, lead, seq, rho):
        return self._b["a_" + tier].format(lead=lead, seq=seq, rho=f"{rho:.2f}")


def _load():
    langs = {}
    for path in sorted(glob.glob(os.path.join(PIPELINE_ROOT, "settings", "templates_*.yaml"))):
        with open(path, encoding="utf-8") as fh:
            spec = yaml.safe_load(fh)
        t = LangTemplates(spec)
        langs[t.LANG] = t
    if not langs:
        raise FileNotFoundError("no settings/templates_*.yaml found")
    return langs


BY_LANG = _load()
LANGS = sorted(BY_LANG)
