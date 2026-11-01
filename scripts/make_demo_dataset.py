#!/usr/bin/env python3
"""SYNTHETIC demo corpus for offline smoke-testing (NOT real heritage data).

Warps a public-domain sample photograph (scikit-image astronaut) into N pseudo-specimens
with fake culture / region / period / material labels drawn from the controlled taxonomy,
so retrieval, Procrustes/PCA, affinity and the object browser can be exercised.
Delete data/corpus and data/reference_index before real ingestion.

    python scripts/make_demo_dataset.py --n 32 --reset
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import data
from skimage.transform import PiecewiseAffineTransform, warp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app import taxonomy  # noqa: E402
from connectors.base import CORPUS, IMG_DIR, RECORDS, RIGHTS, CorpusRecord, bump_version  # noqa: E402

FAKE = [("moche", "ceramic", "sculpture"), ("roman_imperial", "stone_marble", "sculpture"), ("egyptian", "stone_limestone", "sculpture"),
        ("benin", "metal_bronze", "sculpture"), ("renaissance", "canvas", "painting"), ("chinese_qin_han", "terracotta", "sculpture")]


def deform(img, rng, strength, bias):
    h, w = img.shape[:2]
    rows, cols = np.meshgrid(np.linspace(0, h, 6), np.linspace(0, w, 6))
    src = np.dstack([cols.flat, rows.flat])[0]
    dst = src + rng.normal(0, strength, src.shape) + bias
    edge = (src[:, 0] == 0) | (src[:, 0] == w) | (src[:, 1] == 0) | (src[:, 1] == h)
    dst[edge] = src[edge]
    tf = PiecewiseAffineTransform()
    tf.estimate(src, dst)
    return (warp(img, tf, output_shape=(h, w)) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    if args.reset and CORPUS.exists():
        shutil.rmtree(CORPUS)
    rng = np.random.default_rng(args.seed)
    base = data.astronaut()[20:300, 150:420]
    biases = {c: rng.normal(0, 6, (1, 2)) for c, _, _ in FAKE}
    CORPUS.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        culture, material, domain = FAKE[i % len(FAKE)]
        img = deform(base, rng, 4.0, biases[culture])
        img = np.clip(img.astype(np.float32) * rng.uniform(0.7, 1.25), 0, 255).astype(np.uint8)
        rid = f"DEMO_{i:03d}"
        d = IMG_DIR / rid
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(d / "original.jpg", quality=92)
        th = Image.fromarray(img)
        th.thumbnail((400, 400))
        th.save(d / "thumb.jpg", quality=88)
        ct = taxonomy.term("culture", culture)
        rec = CorpusRecord(id=rid, source="DEMO", source_object_id=rid, source_url="https://scikit-image.org", institution="Synthetic demo (not a museum)",
                           object_number=f"DEMO-{i:03d}", title=f"Synthetic demo specimen {i:03d} ({ct['label']})", domain=domain,
                           region=ct["region"], culture=culture, culture_raw="SYNTHETIC", period=None, date_start=ct.get("date_start"),
                           date_end=ct.get("date_end"), date_display=f"{ct.get('date_start')}–{ct.get('date_end')} (fake)", material=material,
                           object_type="portrait_head", provenance="Warped scikit-image astronaut sample (NASA, public domain) — synthetic test data",
                           license="PD", license_raw="NASA photograph, public domain", image_path=f"images/{rid}/original.jpg",
                           thumbnail_path=f"images/{rid}/thumb.jpg")
        with RECORDS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**rec.to_dict(), "mold_group": f"demo_mold_{i % 8}"}) + "\n")
        with RIGHTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": rid, "license": "PD", "attribution": "NASA / scikit-image", **taxonomy.license_policy("PD")}) + "\n")
    bump_version(f"SYNTHETIC demo corpus: {args.n} warped specimens (not heritage data)")
    print(f"Created {args.n} synthetic specimens in {CORPUS}")


if __name__ == "__main__":
    main()
