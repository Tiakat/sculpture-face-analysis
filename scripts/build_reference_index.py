#!/usr/bin/env python3
"""Build the reference index used for retrieval and morphometrics from the
GlobalFaceCorpus (data/corpus/records.jsonl).  Records without a stored image
(rights-restricted, metadata-only) are skipped.  Runs the frozen pipeline (detection, landmarks, alignment,
AU profile, embedding) on every image and stores:

  data/reference_index/index.json      records + dataset_version + build info
  data/reference_index/embeddings.npy  (n, d)
  data/reference_index/landmarks.npy   (n, 33, 2) anthropometric subset (aligned coords)

Only the largest/primary face of each corpus image enters the index.

Images on which no face can be landmarked are listed in
data/reference_index/skipped.json for manual inspection.

If a record has no expert `expression` label, the model's own prediction is
stored under `expression` with `expression_source = "model-v0.1 (unverified)"`
so that "reference specimens classified as X" can still be shown, clearly
flagged as unverified in the UI.

Usage:
    python scripts/build_reference_index.py [--min-quality 40]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.pipeline import versions  # noqa: E402
from app.pipeline.action_units import estimate_action_units  # noqa: E402
from app.pipeline.embedding import face_embedding  # noqa: E402
from app.pipeline.expression import classify_expression  # noqa: E402
from app.pipeline.landmarks import localise_all, morpho_subset  # noqa: E402
from app.pipeline.normalization import NORMALISERS, DEFAULT_NORMALISER, align_face  # noqa: E402
from app.pipeline.pose import estimate_pose  # noqa: E402
from app.pipeline.quality import assess_quality  # noqa: E402

DB_DIR = Path(os.environ.get("GFP_DATA_DIR", ROOT / "data")) / "corpus"
OUT_DIR = Path(os.environ.get("GFP_DATA_DIR", ROOT / "data")) / "reference_index"


def process(path: Path, min_quality: float):
    img = cv2.imread(str(path))
    if img is None:
        return None, "unreadable"
    h, w = img.shape[:2]
    if max(h, w) > 1600:
        s = 1600 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    locs = [l for l in localise_all(img, max_faces=4) if l.landmarks is not None]
    if not locs:
        return None, "no landmarks"
    loc = locs[0]
    q = assess_quality(img, loc.box)
    if q.score < min_quality:
        return None, f"quality {q.score}"
    pose = estimate_pose(loc.landmarks, img.shape)
    norm = NORMALISERS[DEFAULT_NORMALISER](img)
    aligned, lm_al, _ = align_face(norm, loc.landmarks)
    aus, feats, _ = estimate_action_units(lm_al)
    expr = classify_expression(aus, pose.pose_quality, feats)
    emb = face_embedding(aligned, lm_al)
    return {"embedding": emb, "landmarks": morpho_subset(lm_al), "quality": q.score, "pose": pose.to_dict(),
            "expression": expr.primary, "probabilities": expr.probabilities,
            "au_profile": {a.code: a.intensity for a in aus if a.present}, "features": feats}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-quality", type=float, default=40)
    ap.add_argument("--db", default=str(DB_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    db_dir, out_dir = Path(args.db), Path(args.out)
    meta_file = db_dir / "records.jsonl"
    if not meta_file.exists():
        sys.exit(f"No metadata at {meta_file}. Run an ingestion script first.")
    records = [json.loads(l) for l in meta_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    records = [r for r in records if r.get("image_path")]
    ver = (db_dir / "VERSION").read_text(encoding="utf-8").strip() if (db_dir / "VERSION").exists() else "GlobalFaceCorpus v0.0"

    kept, embs, lms, skipped = [], [], [], []
    for i, rec in enumerate(records, 1):
        p = db_dir / rec["image_path"]
        res, why = process(p, args.min_quality)
        if res is None:
            skipped.append({"id": rec["id"], "reason": why})
            print(f"[{i}/{len(records)}] skip {rec['id']}: {why}")
            continue
        out = dict(rec)
        out["quality_score"] = res["quality"]
        out["pose"] = res["pose"]
        out["au_profile"] = res["au_profile"]
        out["geometric_features"] = res["features"]
        out["model_expression"] = res["expression"]
        out["model_probabilities"] = res["probabilities"]
        out["museum"] = out.get("institution")
        out["date"] = out.get("date_display")
        if not out.get("expression"):
            out["expression"] = res["expression"]
            out["expression_source"] = f"model {versions.COMPONENT_VERSIONS['expression_model']} (unverified)"
        kept.append(out)
        embs.append(res["embedding"])
        lms.append(res["landmarks"])
        print(f"[{i}/{len(records)}] ok   {rec['id']}: {out['expression']} q={res['quality']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", np.stack(embs).astype(np.float32) if embs else np.zeros((0, 1), np.float32))
    np.save(out_dir / "landmarks.npy", np.stack(lms) if lms else np.zeros((0, 33, 2)))
    payload = json.dumps({"dataset_version": ver, "built_at": datetime.now(timezone.utc).isoformat(),
                          "pipeline_versions": versions.COMPONENT_VERSIONS, "n_records": len(kept), "n_skipped": len(skipped),
                          "records": kept}, indent=1, ensure_ascii=False, default=str)
    tmp = out_dir / "index.json.tmp"          # write atomically so a crash never leaves a half-written index
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(out_dir / "index.json")
    (out_dir / "skipped.json").write_text(json.dumps(skipped, indent=1), encoding="utf-8")
    print(f"\nReference index built: {len(kept)} specimens ({len(skipped)} skipped) -> {out_dir}")
    print("Restart the server or POST /api/references/reload (admin) to load it.")


if __name__ == "__main__":
    main()
