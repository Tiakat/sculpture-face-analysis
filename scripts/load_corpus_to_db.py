#!/usr/bin/env python3
"""Load corpus records into the platform database as OBJECTS (+ IMAGES, optionally analysed).

    python scripts/load_corpus_to_db.py                # objects + image rows, no analysis (fast)
    python scripts/load_corpus_to_db.py --analyse      # also run the face pipeline on each image (slow)
    python scripts/load_corpus_to_db.py --source MET   # one source only

Objects already present (same source + source_object_id) are updated, not duplicated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GLOG_minloglevel", "2")

from sqlalchemy import select  # noqa: E402

from app import taxonomy  # noqa: E402
from app.config import ANALYSIS_DIR, CORPUS_DIR, RUN_STABILITY  # noqa: E402
from app.database import Base, SessionLocal, engine  # noqa: E402
from app.models import ArtObject, Face, Image  # noqa: E402
from app.pipeline import versions  # noqa: E402

OBJ_FIELDS = ("source", "source_object_id", "source_url", "institution", "object_number", "title", "artist", "domain", "region",
              "culture", "period", "date_start", "date_end", "date_display", "date_precision", "material", "technique", "object_type",
              "country", "site", "license", "rights_holder", "external_ids")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--limit", type=int, default=100000)
    args = ap.parse_args()
    Base.metadata.create_all(bind=engine)
    recs_file = CORPUS_DIR / "records.jsonl"
    if not recs_file.exists():
        sys.exit("No corpus. Run scripts/ingest.py first.")
    recs = [json.loads(l) for l in recs_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.source:
        recs = [r for r in recs if r["source"] == args.source]
    if args.analyse:
        from app.pipeline.runner import run_image_analysis
        from app.state import get_reference_index
        ref = get_reference_index()
    db = SessionLocal()
    n_new = n_upd = n_img = 0
    for r in recs[: args.limit]:
        o = db.scalar(select(ArtObject).where(ArtObject.source == r["source"], ArtObject.source_object_id == str(r["source_object_id"])))
        meta = {k: r.get(k) for k in OBJ_FIELDS if r.get(k) is not None}
        meta["provenance_text"] = r.get("provenance")
        meta["notes"] = f"culture_raw={r.get('culture_raw')}; material_raw={r.get('material_raw')}"
        for fld in ("culture", "region", "period", "material", "domain", "object_type", "license"):
            if meta.get(fld) and meta[fld] not in taxonomy.ids(fld):
                meta[fld] = None if fld != "license" else "UNKNOWN"
        if o is None:
            o = ArtObject(annotation_layer="corpus_auto")
            db.add(o)
            n_new += 1
        else:
            n_upd += 1
        for k, v in meta.items():
            setattr(o, k, v)
        o.training_allowed = taxonomy.training_eligible(meta)[0]
        db.flush()
        if r.get("image_path"):
            p = CORPUS_DIR / r["image_path"]
            if p.exists() and not any(im.stored_path == str(p) for im in o.images):
                sha = hashlib.sha256(p.read_bytes()).hexdigest()
                im = Image(object_id=o.id, original_filename=p.name, stored_path=str(p), sha256=sha, image_url=r.get("image_url"),
                           iiif_url=r.get("iiif_url"), image_license=r.get("license"), status="pending")
                db.add(im)
                db.flush()
                n_img += 1
                if args.analyse:
                    dterm = taxonomy.term("domain", o.domain) or {}
                    rep = run_image_analysis(p, ANALYSIS_DIR / im.id, ref, {"domain": o.domain, "depicted": bool(dterm.get("depicted", True))},
                                             run_stability=RUN_STABILITY)
                    im.status = rep["status"]; im.quality_score = rep["image_quality"]["score"]; im.phash = rep["input"]["phash"]
                    im.face_count = rep.get("face_count", 0); im.model_version = versions.COMPONENT_VERSIONS["expression_model"]
                    im.dataset_version = rep["versions"]["dataset"]; im.width, im.height = rep["input"]["working_size"]
                    for f in rep.get("faces", []):
                        face = Face(image_id=im.id, face_index=f["face_index"], bbox=list(f["box"]) if f.get("box") else None,
                                    quality_score=f["quality"]["score"], status=f.get("status", "rejected"))
                        if f.get("status") == "completed":
                            face.expression = f["expression"]["primary"]; face.expression_confidence = f["expression"]["confidence"]
                            face.pose_quality = f["pose"]["pose_quality"]; face.yaw = f["pose"]["yaw_deg"]; face.difficulty_level = f["difficulty_level"]
                        db.add(face)
        db.commit()
        print(f"  {r['id']}: {'new' if n_new else 'upd'}")
    print(f"Objects: {n_new} new, {n_upd} updated; {n_img} images attached{' and analysed' if args.analyse else ''}.")


if __name__ == "__main__":
    main()
