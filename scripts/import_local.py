#!/usr/bin/env python3
"""Import a local folder of images + CSV metadata into the corpus (agreements, own photographs, expert-labelled data).

CSV header — any subset of the canonical fields; `filename` is mandatory:
  filename,id,institution,object_number,title,artist,domain,region,culture,culture_subgroup,period,date_start,date_end,
  date_display,material,technique,object_type,subject_type,representation_type,country,site,provenance,license,
  rights_holder,photographer,view_angle,mold_group,expression,expression_source,cultural_sensitivity

Controlled fields are validated against config/taxonomy.json; the script stops on the first invalid value
and lists the allowed ids.

    python scripts/import_local.py --images ~/photos --csv metadata.csv --source LARCO --license PERMISSION
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import taxonomy  # noqa: E402
from connectors.base import CORPUS, IMG_DIR, RECORDS, RIGHTS, CorpusRecord, bump_version, existing_ids  # noqa: E402
import json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--license", default="UNKNOWN")
    args = ap.parse_args()
    seen = existing_ids()
    img_dir = Path(args.images)
    n = 0
    with open(args.csv, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), 1):
            src = img_dir / row["filename"]
            if not src.exists():
                print(f"  ! missing {src}")
                continue
            rid = f"{args.source}_{row.get('id') or Path(row['filename']).stem}"
            if rid in seen:
                continue
            ctrl = {k: row[k] for k in taxonomy.CONTROLLED_FIELDS if row.get(k)}
            for k in ("date_start", "date_end"):
                if row.get(k):
                    ctrl[k] = int(row[k])
            try:
                taxonomy.validate_metadata(dict(ctrl))
            except Exception as e:
                sys.exit(f"row {i} ({row['filename']}): {getattr(e, 'detail', e)}\nAllowed culture ids: {taxonomy.ids('culture')}")
            d = IMG_DIR / rid
            d.mkdir(parents=True, exist_ok=True)
            dst = d / f"original{src.suffix.lower()}"
            shutil.copy2(src, dst)
            im = Image.open(dst).convert("RGB")
            im.thumbnail((400, 400))
            im.save(d / "thumb.jpg", quality=88)
            rec = CorpusRecord(id=rid, source=args.source, source_object_id=row.get("id") or Path(row["filename"]).stem, source_url=row.get("source_url"),
                               institution=row.get("institution"), object_number=row.get("object_number"), title=row.get("title"), artist=row.get("artist"),
                               domain=row.get("domain") or "sculpture", region=row.get("region"), culture=row.get("culture"), period=row.get("period"),
                               date_start=int(row["date_start"]) if row.get("date_start") else None, date_end=int(row["date_end"]) if row.get("date_end") else None,
                               date_display=row.get("date_display"), material=row.get("material"), technique=row.get("technique"), object_type=row.get("object_type"),
                               country=row.get("country"), site=row.get("site"), provenance=row.get("provenance"), license=row.get("license") or args.license,
                               rights_holder=row.get("rights_holder"), image_path=str(dst.relative_to(CORPUS)), thumbnail_path=str((d / "thumb.jpg").relative_to(CORPUS)),
                               expression=row.get("expression") or None, expression_source=row.get("expression_source") or ("expert" if row.get("expression") else None))
            extra = {"culture_subgroup": row.get("culture_subgroup"), "mold_group": row.get("mold_group"), "photographer": row.get("photographer"),
                     "view_angle": row.get("view_angle"), "subject_type": row.get("subject_type"), "representation_type": row.get("representation_type"),
                     "cultural_sensitivity": row.get("cultural_sensitivity")}
            CORPUS.mkdir(parents=True, exist_ok=True)
            with RECORDS.open("a", encoding="utf-8") as fo:
                fo.write(json.dumps({**rec.to_dict(), **{k: v for k, v in extra.items() if v}}, ensure_ascii=False) + "\n")
            with RIGHTS.open("a", encoding="utf-8") as fo:
                fo.write(json.dumps({"id": rid, "license": rec.license, "attribution": row.get("photographer") or row.get("institution"),
                                     "source_url": rec.source_url, **taxonomy.license_policy(rec.license)}, ensure_ascii=False) + "\n")
            n += 1
    if n:
        bump_version(f"Local import {args.source}: +{n}")
    print(f"Imported {n} records.")


if __name__ == "__main__":
    main()
