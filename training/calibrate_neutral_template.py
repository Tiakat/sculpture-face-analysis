#!/usr/bin/env python3
"""Refit the neutral geometric template from expert-labelled NEUTRAL faces.

The AU estimator measures deviation from a neutral template (mean / sd of
14 scale-free facial ratios).  v0.1 ships anthropometric approximations; this
script replaces them with values measured on faces an expert labelled
"neutral" — the core of the "calibrate later" plan.  Millones used exactly
this criterion (neutral gesture, Ekman & Rosenberg 1997) to select vessels
for morphometrics.

Usage:
    python training/calibrate_neutral_template.py --reference data/reference_index/index.json \
        [--validation validation_set.jsonl] [--min-n 15]
Writes config/neutral_template.json (loaded automatically at next start).
"""
from __future__ import annotations

import os
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.pipeline.action_units import CONFIG_PATH, DEFAULT_TEMPLATE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=Path(os.environ.get("GFP_DATA_DIR", ROOT / "data")) / "reference_index" / "index.json")
    ap.add_argument("--validation", type=Path)
    ap.add_argument("--min-n", type=int, default=15)
    ap.add_argument("--shrink", type=float, default=0.3, help="shrinkage toward the prior template (0 = pure data)")
    args = ap.parse_args()

    feats = []
    if args.reference.exists():
        for r in json.loads(args.reference.read_text(encoding="utf-8"))["records"]:
            if r.get("expression") == "neutral" and "model" not in (r.get("expression_source") or "") and r.get("geometric_features"):
                feats.append(r["geometric_features"])
    if args.validation and args.validation.exists():
        for line in args.validation.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("label_expression") == "neutral" and r.get("geometric_features"):
                    feats.append(r["geometric_features"])
    if len(feats) < args.min_n:
        sys.exit(f"Only {len(feats)} expert-labelled neutral specimens (need >= {args.min_n}).")

    tmpl = {}
    for k, prior in DEFAULT_TEMPLATE.items():
        vals = np.array([f[k] for f in feats if k in f], dtype=float)
        mean = (1 - args.shrink) * vals.mean() + args.shrink * prior["mean"]
        sd = max((1 - args.shrink) * vals.std(ddof=1) + args.shrink * prior["sd"], 0.005)
        tmpl[k] = {"mean": round(float(mean), 5), "sd": round(float(sd), 5), "n": int(len(vals))}
    tmpl["_meta"] = {"calibrated_at": datetime.now(timezone.utc).isoformat(), "n_specimens": len(feats),
                     "shrinkage": args.shrink, "source": [str(args.reference), str(args.validation)]}
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(tmpl, indent=2), encoding="utf-8")
    print(f"Wrote {CONFIG_PATH} from {len(feats)} neutral specimens. Bump 'action_units' version in app/pipeline/versions.py.")


if __name__ == "__main__":
    main()
