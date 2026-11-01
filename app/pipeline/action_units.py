"""Geometric, FACS-inspired facial-action estimation for static sculpture.

IMPORTANT WORDING: a sculpture has no muscles that move.  What we estimate
is the *implied facial-action configuration*: how far the depicted geometry
departs from a neutral template along directions that, in living faces,
correspond to specific Action Units.  Each AU therefore reports:

  intensity  0-5   (FACS-style A-E scale mapped to 0-5, 0 = absent)
  evidence   0-1   (how strongly the geometry supports it)
  the anatomical muscle *associated* with the AU (not "detected")

The neutral template (config/neutral_template.json) is the calibration
target: once expert-labelled neutral sculptures exist, refit it with
training/calibrate_neutral_template.py.  Version: geometric-au-v0.1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from .landmarks import MORPHO_LANDMARKS as L

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "neutral_template.json"

# Default neutral template (ratios relative to inter-exocanthion distance).
# Values are anthropometric approximations adjusted for MediaPipe landmark
# placement; refit them from labelled neutral sculptures when available.
DEFAULT_TEMPLATE = {
    "brow_inner_height": {"mean": 0.220, "sd": 0.035},
    "brow_outer_height": {"mean": 0.150, "sd": 0.035},
    "brow_mid_height":   {"mean": 0.240, "sd": 0.035},
    "brow_gap":          {"mean": 0.330, "sd": 0.040},
    "eye_opening":       {"mean": 0.105, "sd": 0.022},
    "mouth_width":       {"mean": 0.550, "sd": 0.060},
    "corner_lift":       {"mean": -0.010, "sd": 0.030},
    "lip_gap":           {"mean": 0.000, "sd": 0.030},
    "lower_face_height": {"mean": 0.680, "sd": 0.060},
    "upper_lip_height":  {"mean": 0.160, "sd": 0.030},
    "lower_lip_chin":    {"mean": 0.340, "sd": 0.045},
    "lip_thickness":     {"mean": 0.160, "sd": 0.035},
    "nose_width":        {"mean": 0.480, "sd": 0.050},
    "nose_length":       {"mean": 0.490, "sd": 0.050},
}


def load_template() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_TEMPLATE


def _p(lm: np.ndarray, name: str) -> np.ndarray:
    return lm[L[name], :2]


def geometric_features(lm: np.ndarray) -> dict[str, float]:
    """Scale-free facial ratios from roll-aligned landmarks (y grows downward)."""
    iod = np.linalg.norm(_p(lm, "exocanthion_R") - _p(lm, "exocanthion_L")) + 1e-6

    def dy(a, b):  # vertical distance a->b (positive when b is lower)
        return (_p(lm, b)[1] - _p(lm, a)[1]) / iod

    def dist(a, b):
        return np.linalg.norm(_p(lm, a) - _p(lm, b)) / iod

    eye_center_R = (_p(lm, "endocanthion_R") + _p(lm, "exocanthion_R")) / 2
    eye_center_L = (_p(lm, "endocanthion_L") + _p(lm, "exocanthion_L")) / 2
    f = {
        "brow_inner_height": np.mean([dy("brow_inner_R", "endocanthion_R"), dy("brow_inner_L", "endocanthion_L")]),
        "brow_outer_height": np.mean([dy("brow_outer_R", "exocanthion_R"), dy("brow_outer_L", "exocanthion_L")]),
        "brow_mid_height": np.mean([(eye_center_R[1] - _p(lm, "brow_mid_R")[1]) / iod,
                                    (eye_center_L[1] - _p(lm, "brow_mid_L")[1]) / iod]),
        "brow_gap": dist("brow_inner_R", "brow_inner_L"),
        "eye_opening": np.mean([dy("palpebrale_sup_R", "palpebrale_inf_R"), dy("palpebrale_sup_L", "palpebrale_inf_L")]),
        "mouth_width": dist("cheilion_R", "cheilion_L"),
        # positive when the mouth corners sit ABOVE the lip midline (smile-like)
        "corner_lift": ((_p(lm, "labiale_superius")[1] + _p(lm, "labiale_inferius")[1]) / 2
                        - (_p(lm, "cheilion_R")[1] + _p(lm, "cheilion_L")[1]) / 2) / iod,
        "lip_gap": dy("stomion_sup", "stomion_inf"),
        "lower_face_height": dy("subnasale", "gnathion"),
        "upper_lip_height": dy("subnasale", "labiale_superius"),
        "lower_lip_chin": dy("labiale_inferius", "gnathion"),
        "lip_thickness": dy("labiale_superius", "stomion_sup") + dy("stomion_inf", "labiale_inferius"),
        "nose_width": dist("alare_R", "alare_L"),
        "nose_length": dy("nasion", "pronasale"),
    }
    return {k: float(v) for k, v in f.items()}


def zscores(features: dict[str, float], template: dict | None = None) -> dict[str, float]:
    t = template or load_template()
    return {k: (features[k] - t[k]["mean"]) / t[k]["sd"] for k in features if k in t}


# AU definitions: name, associated musculature, list of (feature, sign) that
# increase evidence for the AU.  sign=+1 -> higher-than-neutral supports AU.
AU_DEFS = [
    ("AU1", "Inner brow raiser", "frontalis (pars medialis)", [("brow_inner_height", +1)]),
    ("AU2", "Outer brow raiser", "frontalis (pars lateralis)", [("brow_outer_height", +1)]),
    ("AU4", "Brow lowerer", "corrugator supercilii / depressor supercilii / procerus",
     [("brow_inner_height", -1), ("brow_mid_height", -1), ("brow_gap", -1)]),
    ("AU5", "Upper lid raiser", "levator palpebrae superioris", [("eye_opening", +1)]),
    ("AU6", "Cheek raiser", "orbicularis oculi (pars orbitalis)", [("eye_opening", -1), ("corner_lift", +1)]),
    ("AU7", "Lid tightener", "orbicularis oculi (pars palpebralis)", [("eye_opening", -1)]),
    ("AU9", "Nose wrinkler", "levator labii superioris alaeque nasi", [("nose_length", -1), ("nose_width", +1)]),
    ("AU10", "Upper lip raiser", "levator labii superioris", [("upper_lip_height", -1)]),
    ("AU12", "Lip corner puller", "zygomaticus major", [("corner_lift", +1), ("mouth_width", +1)]),
    ("AU15", "Lip corner depressor", "depressor anguli oris", [("corner_lift", -1)]),
    ("AU17", "Chin raiser", "mentalis", [("lower_lip_chin", -1)]),
    ("AU20", "Lip stretcher", "risorius", [("mouth_width", +1), ("corner_lift", -0.5)]),
    ("AU23", "Lip tightener", "orbicularis oris", [("lip_thickness", -1)]),
    ("AU25", "Lips part", "depressor labii inferioris / relaxed mentalis", [("lip_gap", +1)]),
    ("AU26", "Jaw drop", "masseter relaxed; temporalis / pterygoid relaxed",
     [("lip_gap", +1), ("lower_face_height", +1)]),
]


@dataclass
class ActionUnit:
    code: str
    name: str
    associated_musculature: str
    intensity: float      # 0-5
    evidence: float       # 0-1
    present: bool
    contributing_features: dict[str, float]

    def to_dict(self):
        return asdict(self)


def _evidence_from_z(z: float) -> float:
    """Map a supporting z-score to 0-1 evidence (0 below ~0.7 sd, saturates ~3.5 sd)."""
    return float(np.clip((z - 0.7) / 2.8, 0.0, 1.0))


def estimate_action_units(lm_aligned: np.ndarray, template: dict | None = None) -> tuple[list[ActionUnit], dict, dict]:
    feats = geometric_features(lm_aligned)
    z = zscores(feats, template)
    aus: list[ActionUnit] = []
    for code, name, muscle, rules in AU_DEFS:
        contribs = {}
        ev_terms = []
        weights = []
        for feat, sign in rules:
            zz = z.get(feat, 0.0) * np.sign(sign)
            wgt = abs(sign)
            contribs[feat] = round(float(z.get(feat, 0.0)), 3)
            ev_terms.append(_evidence_from_z(zz) * wgt)
            weights.append(wgt)
        evidence = float(np.sum(ev_terms) / np.sum(weights))
        intensity = round(5.0 * evidence, 2)
        aus.append(ActionUnit(code=code, name=name, associated_musculature=muscle,
                              intensity=intensity, evidence=round(evidence, 3),
                              present=evidence >= 0.2, contributing_features=contribs))
    return aus, {k: round(v, 4) for k, v in feats.items()}, {k: round(v, 3) for k, v in z.items()}
