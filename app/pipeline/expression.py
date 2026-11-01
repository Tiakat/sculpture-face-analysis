"""Depicted-expression inference from the AU-like profile (frozen v0.1).

The classifier is a transparent prototype matcher following the EMFACS
emotion-prediction table (Ekman & Friesen; Ekman & Rosenberg 1997):
each expression category is defined by the AUs that characterise it.
It outputs a full probability distribution, a separate confidence, an
explicit UNKNOWN / out-of-distribution flag, and a per-region contribution
map for explainability.

Terminology used everywhere: "depicted facial expression" — never the
emotional state of a historical person.

Once expert labels exist, training/train_expression_classifier.py fits a
supervised model on the same feature vector and registers it as v0.2+;
this module can then load it instead of the prototype table.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

from .action_units import ActionUnit

CATEGORIES = ["neutral", "happiness", "sadness", "anger", "fear", "disgust", "surprise"]
UNKNOWN = "unknown/ambiguous"

# Prototype AU weights per category (core AUs weighted 1.0, secondary 0.5).
PROTOTYPES: dict[str, dict[str, float]] = {
    "happiness": {"AU6": 0.8, "AU12": 1.0, "AU25": 0.3},
    "sadness":   {"AU1": 0.8, "AU4": 0.6, "AU15": 1.0, "AU17": 0.5},
    "anger":     {"AU4": 1.0, "AU5": 0.6, "AU7": 0.8, "AU23": 0.8, "AU17": 0.3},
    "fear":      {"AU1": 0.7, "AU2": 0.7, "AU4": 0.5, "AU5": 1.0, "AU20": 1.0, "AU26": 0.5},
    "disgust":   {"AU9": 1.0, "AU10": 0.9, "AU17": 0.6, "AU15": 0.4},
    "surprise":  {"AU1": 0.8, "AU2": 0.8, "AU5": 1.0, "AU26": 1.0, "AU25": 0.4},
}
# AUs whose presence contradicts a category (penalty weights)
CONTRADICTIONS: dict[str, dict[str, float]] = {
    "happiness": {"AU15": 0.8, "AU4": 0.4},
    "sadness":   {"AU12": 0.8, "AU5": 0.3},
    "anger":     {"AU12": 0.6, "AU1": 0.5, "AU26": 0.3},
    "fear":      {"AU12": 0.5, "AU15": 0.3},
    "disgust":   {"AU12": 0.4, "AU5": 0.4},
    "surprise":  {"AU4": 0.8, "AU12": 0.3, "AU15": 0.3},
}

REGION_OF_AU = {
    "AU1": "brow", "AU2": "brow", "AU4": "brow",
    "AU5": "eyes", "AU6": "eyes", "AU7": "eyes",
    "AU9": "nose", "AU10": "mouth", "AU12": "mouth", "AU15": "mouth", "AU17": "jaw",
    "AU20": "mouth", "AU23": "mouth", "AU25": "mouth", "AU26": "jaw",
}

MODEL_REGISTRY = Path(__file__).resolve().parents[2] / "data" / "models" / "expression_model.json"


@dataclass
class ExpressionResult:
    primary: str
    confidence: float
    probabilities: dict[str, float]
    is_unknown: bool
    unknown_reason: str | None
    total_activation: float
    region_contributions: dict[str, float]
    supporting_aus: list[str]
    model: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = (x - x.max()) / temp
    e = np.exp(z)
    return e / e.sum()


_LEARNED = None


def _load_learned_model():
    """Optional trained model (training/train_expression_classifier.py),
    activated with MFP_EXPRESSION_MODEL=/path/to/model.pkl."""
    global _LEARNED
    if _LEARNED is not None:
        return _LEARNED or None
    import os
    p = os.getenv("MFP_EXPRESSION_MODEL")
    if not p or not Path(p).exists():
        _LEARNED = False
        return None
    import pickle
    with open(p, "rb") as f:
        _LEARNED = pickle.load(f)
    return _LEARNED


def classify_expression(aus: list[ActionUnit], pose_quality: float = 1.0,
                        features: dict | None = None) -> ExpressionResult:
    ev = {a.code: a.evidence for a in aus}
    learned = _load_learned_model()
    if learned is not None and features is not None:
        return _classify_learned(learned, aus, ev, features, pose_quality)
    return _classify_prototype(aus, ev, pose_quality)


def _classify_learned(m, aus, ev, features, pose_quality) -> ExpressionResult:
    x = np.array([[features.get(k, 0.0) for k in m["features"] if not k.startswith("AU")] +
                  [ev.get(k, 0.0) for k in m["features"] if k.startswith("AU")]])
    proba = m["model"].predict_proba(x)[0]
    labels = list(m["model"].classes_)
    probs = {c: 0.0 for c in CATEGORIES}
    for c, p in zip(labels, proba):
        probs[c] = float(p)
    order = sorted(probs, key=probs.get, reverse=True)
    p1, p2 = probs[order[0]], probs[order[1]]
    conf = float(np.clip((p1 - p2) * 1.5 + p1 * 0.5, 0, 1) * (0.6 + 0.4 * pose_quality))
    is_unknown = conf < 0.2 or (p1 - p2) < 0.08
    region = {r: 0.0 for r in ("brow", "eyes", "nose", "mouth", "jaw")}
    for au, e in ev.items():
        region[REGION_OF_AU[au]] += e
    tot = sum(region.values()) or 1.0
    return ExpressionResult(
        primary=UNKNOWN if is_unknown else order[0], confidence=round(conf, 3),
        probabilities={c: round(v, 4) for c, v in probs.items()}, is_unknown=is_unknown,
        unknown_reason="insufficient margin between top classes" if is_unknown else None,
        total_activation=round(float(sum(ev.values())), 3),
        region_contributions={k: round(v / tot, 3) for k, v in region.items()},
        supporting_aus=[a.code for a in aus if a.present],
        model=f"learned-expression-classifier-v{m.get('version')}",
        notes=["Supervised model trained on expert-labelled sculpture data."])


def _classify_prototype(aus: list[ActionUnit], ev: dict, pose_quality: float) -> ExpressionResult:
    total_act = float(sum(ev.values()))

    scores = {}
    for cat, proto in PROTOTYPES.items():
        s = sum(w * ev.get(au, 0.0) for au, w in proto.items()) / sum(proto.values())
        pen = sum(w * ev.get(au, 0.0) for au, w in CONTRADICTIONS.get(cat, {}).items())
        scores[cat] = s - 0.6 * pen
    # Neutral score: high when every AU is weak
    neutral_score = float(np.clip(1.0 - 1.6 * max(ev.values(), default=0.0) - 0.15 * total_act, 0.0, 1.0))
    scores["neutral"] = neutral_score

    cats = CATEGORIES
    vec = np.array([scores[c] for c in cats])
    probs = _softmax(vec * 4.0)  # temperature chosen for a peaked-but-not-saturated distribution
    order = np.argsort(-probs)
    primary = cats[order[0]]
    p1, p2 = probs[order[0]], probs[order[1]]

    # Confidence: margin between top-2 scaled by pose reliability.
    confidence = float(np.clip((p1 - p2) * 1.5 + p1 * 0.5, 0.0, 1.0) * (0.6 + 0.4 * pose_quality))

    # Out-of-distribution logic: strong activation with no matching prototype,
    # or near-tie between incompatible categories.
    best_match = max(scores[c] for c in PROTOTYPES)
    notes: list[str] = []
    is_unknown = False
    reason = None
    if total_act > 2.5 and best_match < 0.25:
        is_unknown, reason = True, "strong facial-action configuration that matches no expression prototype"
    elif p1 - p2 < 0.04 and primary != "neutral":
        is_unknown, reason = True, f"near-tie between '{cats[order[0]]}' and '{cats[order[1]]}'"
    elif p1 - p2 < 0.10 and primary != "neutral":
        notes.append(f"Close alternative: '{cats[order[1]]}' ({p2:.2f}); treat the primary label as tentative.")
    elif confidence < 0.2:
        is_unknown, reason = True, "insufficient evidence for a reliable classification"
    if is_unknown:
        notes.append("Result flagged UNKNOWN/AMBIGUOUS; the ranked probabilities are shown for reference only.")

    # Region contributions (explainability): how much each face region drove the primary decision
    region = {r: 0.0 for r in ("brow", "eyes", "nose", "mouth", "jaw")}
    if primary == "neutral":
        for au, e in ev.items():
            region[REGION_OF_AU[au]] += (1.0 - e)
    else:
        for au, w in PROTOTYPES[primary].items():
            region[REGION_OF_AU[au]] += w * ev.get(au, 0.0)
    tot = sum(region.values()) or 1.0
    region = {k: round(v / tot, 3) for k, v in region.items()}

    supporting = [a.code for a in aus if a.present and (primary == "neutral" or a.code in PROTOTYPES.get(primary, {}))]
    if primary == "neutral":
        supporting = [a.code for a in aus if a.present]

    return ExpressionResult(
        primary=UNKNOWN if is_unknown else primary,
        confidence=round(confidence, 3),
        probabilities={c: round(float(p), 4) for c, p in zip(cats, probs)},
        is_unknown=is_unknown,
        unknown_reason=reason,
        total_activation=round(total_act, 3),
        region_contributions=region,
        supporting_aus=supporting,
        model="emfacs-prototype-classifier-v0.1 (frozen)",
        notes=notes,
    )
