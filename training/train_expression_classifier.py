#!/usr/bin/env python3
"""Train a supervised expression classifier on gold-standard / expert-labelled faces (any domain).

Feature vector = the same geometric features + AU evidences the frozen v0.1
pipeline computes, so the trained model is a drop-in replacement for the
prototype matcher (registered under data/models/ with a version stamp).

Input: a JSONL produced by  GET /api/admin/export/gold  (accepted human labels;
museum descriptions are never labels) and/or the reference index with expert `expression`
labels (records where expression_source != model).  Splits are group-aware
(object / mould) — see training/split.py.

Usage:
    python training/train_expression_classifier.py --gold gold_standard.jsonl \
        --reference data/reference_index/index.json --version 0.2.0

Model deployment is manual: compare metrics with the previous version in
data/models/registry.json and only promote when it is better on the held-out
group-split test set (no automatic continuous learning).
"""
from __future__ import annotations

import os
import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.evaluate import expression_metrics, save  # noqa: E402
from training.split import assert_no_leakage, grouped_split  # noqa: E402
from app.pipeline.action_units import AU_DEFS, DEFAULT_TEMPLATE  # noqa: E402

FEATURE_KEYS = list(DEFAULT_TEMPLATE.keys())
AU_KEYS = [d[0] for d in AU_DEFS]
MODELS_DIR = Path(os.environ.get("GFP_DATA_DIR", ROOT / "data")) / "models"


def load_samples(validation: Path | None, reference: Path | None) -> list[dict]:
    samples = []
    if validation and validation.exists():
        for line in validation.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("label_expression") and r.get("geometric_features"):
                samples.append({"id": r.get("face_id") or r.get("analysis_id"), "object_number": r.get("group_key") or r.get("object_id"),
                                "mold_group": r.get("group_key"), "sha256": r.get("sha256"), "institution": r.get("institution"),
                                "culture": r.get("culture"), "domain": r.get("domain"), "features": r["geometric_features"],
                                "au": r.get("au_evidence") or {}, "expression": r["label_expression"]})
    if reference and reference.exists():
        idx = json.loads(reference.read_text(encoding="utf-8"))
        for r in idx["records"]:
            src = r.get("expression_source") or ""
            if r.get("expression") and "model" not in src and r.get("geometric_features"):
                samples.append({"id": r["id"], "object_number": r.get("object_number"), "mold_group": r.get("mold_group"),
                                "features": r["geometric_features"], "au": {k: v / 5 for k, v in (r.get("au_profile") or {}).items()},
                                "expression": r["expression"]})
    return samples


def vectorise(s: dict) -> np.ndarray:
    f = [s["features"].get(k, 0.0) for k in FEATURE_KEYS]
    a = [s["au"].get(k, 0.0) for k in AU_KEYS]
    return np.array(f + a, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", "--validation", dest="validation", type=Path)
    ap.add_argument("--reference", type=Path, default=Path(os.environ.get("GFP_DATA_DIR", ROOT / "data")) / "reference_index" / "index.json")
    ap.add_argument("--version", required=True, help="semantic version for the new model, e.g. 0.2.0")
    ap.add_argument("--algo", choices=["logreg", "gbm"], default="logreg")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout", help="external test: hold out one value of --holdout-field (e.g. an institution or culture)")
    ap.add_argument("--holdout-field", default="institution", choices=["institution", "culture", "domain"])
    args = ap.parse_args()

    samples = load_samples(args.validation, args.reference)
    labels = sorted({s["expression"] for s in samples})
    if len(samples) < 30 or len(labels) < 2:
        sys.exit(f"Not enough expert-labelled samples ({len(samples)} samples, {len(labels)} classes). "
                 "Collect researcher validations first; a supervised model on <30 samples would be meaningless.")

    if args.holdout:
        split = {"train": [s for s in samples if s.get(args.holdout_field) != args.holdout],
                 "test": [s for s in samples if s.get(args.holdout_field) == args.holdout]}
        print(f"External test: {args.holdout_field} = {args.holdout} ({len(split['test'])} samples never seen in training)")
    else:
        split = grouped_split(samples, seed=args.seed)
    assert_no_leakage(split["train"], split["test"])
    if not split["test"]:
        sys.exit("Empty test split.")
    Xtr = np.stack([vectorise(s) for s in split["train"]]); ytr = [s["expression"] for s in split["train"]]
    Xte = np.stack([vectorise(s) for s in split["test"]]); yte = [s["expression"] for s in split["test"]]

    if args.algo == "logreg":
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"))
    else:
        clf = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=args.seed))
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    prob = clf.predict_proba(Xte).max(axis=1)
    metrics = expression_metrics(yte, list(pred), prob, labels)
    metrics.update({"n_train": len(ytr), "n_test": len(yte), "algorithm": args.algo, "features": FEATURE_KEYS + AU_KEYS})

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"expression-{args.algo}-v{args.version}"
    with open(MODELS_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump({"model": clf, "labels": labels, "features": FEATURE_KEYS + AU_KEYS, "version": args.version}, f)
    save(metrics, MODELS_DIR / f"{name}.metrics.json")

    reg_path = MODELS_DIR / "registry.json"
    reg = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.exists() else {"models": []}
    reg["models"].append({"name": name, "version": args.version, "created_at": datetime.now(timezone.utc).isoformat(),
                          "n_train": len(ytr), "n_test": len(yte), "balanced_accuracy": metrics["balanced_accuracy"],
                          "macro_f1": metrics["macro_f1"], "promoted": False})
    reg_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("n_train", "n_test", "accuracy", "balanced_accuracy", "macro_f1")}, indent=2))
    print(f"\nSaved {name}. Promote manually by setting GFP_EXPRESSION_MODEL={MODELS_DIR / (name + '.pkl')} "
          "after comparing with the previous version in registry.json.")


if __name__ == "__main__":
    main()
