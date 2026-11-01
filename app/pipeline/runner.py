"""End-to-end analysis for ONE IMAGE → N FACE INSTANCES (v0.1, frozen models).

METADATA FIREWALL
-----------------
`run_image_analysis` receives only what the vision pipeline needs: the image
path and an `analysis_context` limited to {domain, depicted}.  Culture,
region, period, material, artist … are attached by the caller AFTER analysis,
under a separate top-level key, so no cultural label can influence — or be
learned from — the facial computation.

Per face: pose → illumination normalisation → alignment → AU-like actions →
expression (+UNKNOWN) → embedding → reference retrieval → morphometrics →
difficulty level → (optional) stability tests.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from . import versions
from .action_units import estimate_action_units
from .embedding import face_embedding
from .expression import classify_expression, CATEGORIES
from .landmarks import localise_all, landmarks_from_box, morpho_subset, MORPHO_LANDMARKS, FaceLocalisation
from .normalization import NORMALISERS, DEFAULT_NORMALISER, align_face
from .pose import estimate_pose
from .quality import assess_quality
from .reference_index import ReferenceIndex, suggest_attribution
from .visuals import muscle_map, morphometrics_figure

log = logging.getLogger(__name__)
MAX_SIDE = 1800
ALLOWED_CONTEXT_KEYS = {"domain", "depicted", "affinity_fields", "max_faces", "face_box"}  # face_box = geometric guidance only

WORDING_DEPICTED = {
    "expression": "Depicted facial expression of the represented face — not the emotional state of any real person.",
    "actions": "Implied facial-action configuration inferred from the represented geometry; the artwork has no moving "
               "muscles. Musculature listed is the anatomy associated with each Action Unit in living faces.",
}
WORDING_LIVING = {
    "expression": "Perceived facial expression category of a photographed/filmed person at one instant; a facial "
                  "configuration is not a verified emotional state (Barrett et al. 2019).",
    "actions": "Facial-action estimate (FACS-inspired) from a single still image; no temporal information.",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(img_bgr: np.ndarray) -> str:
    """64-bit perceptual hash (DCT) for near-duplicate detection."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8]
    med = np.median(d[1:, 1:])
    bits = (d > med).flatten()
    return format(int("".join("1" if b else "0" for b in bits), 2), "016x")


def hamming(h1: str, h2: str) -> int:
    return bin(int(h1, 16) ^ int(h2, 16)).count("1")


def _load_working_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unsupported or corrupt image file")
    scale = 1.0
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def _overlay(img, faces: list[FaceLocalisation]):
    out = img.copy()
    for i, f in enumerate(faces):
        if f.box is not None:
            cv2.rectangle(out, (f.box[0], f.box[1]), (f.box[2], f.box[3]), (0, 200, 255), 2)
            cv2.putText(out, f"#{i}", (f.box[0], max(12, f.box[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        if f.landmarks is not None:
            for (x, y, _) in f.landmarks:
                cv2.circle(out, (int(x), int(y)), 1, (80, 255, 80), -1)
            for idx in MORPHO_LANDMARKS.values():
                x, y = f.landmarks[idx, :2]
                cv2.circle(out, (int(x), int(y)), 2, (255, 60, 60), -1)
    return out


def _expression_from(img_bgr, landmarks, pose_quality):
    aligned, lm_al, _ = align_face(img_bgr, landmarks)
    aus, feats, z = estimate_action_units(lm_al)
    return classify_expression(aus, pose_quality, feats), aus, feats, z, aligned, lm_al


def difficulty_level(q_score: float, pose, completeness_hint: str | None = None) -> int:
    """0 clear frontal · 1 mild rotation · 2 three-quarter · 3 profile · 4 partial occlusion/poor quality · 5 damaged/stylised · 6 abstract."""
    yaw = abs(pose.yaw_deg)
    lvl = 0 if yaw < 12 else 1 if yaw < 25 else 2 if yaw < 50 else 3
    if q_score < 55:
        lvl = max(lvl, 4)
    if completeness_hint in ("heavily_damaged", "intentionally_distorted", "reconstructed"):
        lvl = max(lvl, 5)
    return lvl


def stability_tests(img_bgr, landmarks, pose_quality, base_probs) -> dict:
    results = {"illumination": {}, "perturbation": {}}
    base_vec = np.array([base_probs[c] for c in CATEGORIES])
    for name, fn in NORMALISERS.items():
        try:
            res, *_ = _expression_from(fn(img_bgr), landmarks, pose_quality)
            results["illumination"][name] = {"primary": res.primary, "probabilities": res.probabilities}
        except Exception as e:  # pragma: no cover
            results["illumination"][name] = {"error": str(e)}
    h, w = img_bgr.shape[:2]
    # perturb only the face region (crop with margin) so multi-face images stay tractable
    x0, y0 = int(max(0, landmarks[:, 0].min() - 0.4 * (landmarks[:, 0].max() - landmarks[:, 0].min()))), int(max(0, landmarks[:, 1].min() - 0.4 * (landmarks[:, 1].max() - landmarks[:, 1].min())))
    x1, y1 = int(min(w, landmarks[:, 0].max() + 0.4 * (landmarks[:, 0].max() - landmarks[:, 0].min()))), int(min(h, landmarks[:, 1].max() + 0.4 * (landmarks[:, 1].max() - landmarks[:, 1].min())))
    crop = img_bgr[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    perts = {
        "brightness_+10%": lambda im: np.clip(im.astype(np.float32) * 1.10, 0, 255).astype(np.uint8),
        "brightness_-10%": lambda im: np.clip(im.astype(np.float32) * 0.90, 0, 255).astype(np.uint8),
        "rotate_+3deg": lambda im: cv2.warpAffine(im, cv2.getRotationMatrix2D((cw / 2, ch / 2), 3, 1.0), (cw, ch), borderMode=cv2.BORDER_REPLICATE),
        "rotate_-3deg": lambda im: cv2.warpAffine(im, cv2.getRotationMatrix2D((cw / 2, ch / 2), -3, 1.0), (cw, ch), borderMode=cv2.BORDER_REPLICATE),
        "scale_95%": lambda im: cv2.resize(im, (max(8, int(cw * 0.95)), max(8, int(ch * 0.95))), interpolation=cv2.INTER_AREA),
        "crop_5%": lambda im: im[int(ch * 0.05):int(ch * 0.95), int(cw * 0.05):int(cw * 0.95)],
    }
    agree, div = [], []
    for name, fn in perts.items():
        try:
            variant = fn(crop)
            locs = localise_all(variant, max_faces=1)
            if not locs or locs[0].landmarks is None:
                results["perturbation"][name] = {"error": "landmarks lost"}
                continue
            res, *_ = _expression_from(variant, locs[0].landmarks, pose_quality)
            vec = np.array([res.probabilities[c] for c in CATEGORIES])
            d = float(np.abs(vec - base_vec).sum() / 2)
            same = max(res.probabilities, key=res.probabilities.get) == max(base_probs, key=base_probs.get)
            agree.append(bool(same)); div.append(d)
            results["perturbation"][name] = {"primary": res.primary, "tv_distance": round(d, 4), "same_top_class": bool(same)}
        except Exception as e:  # pragma: no cover
            results["perturbation"][name] = {"error": str(e)}
    ill = [v.get("primary") for v in results["illumination"].values() if "primary" in v]
    results["summary"] = {
        "illumination_top_class_consistent": len(set(ill)) <= 1 if ill else None,
        "perturbation_agreement_rate": round(float(np.mean(agree)), 3) if agree else None,
        "mean_tv_distance": round(float(np.mean(div)), 4) if div else None,
        "stability_score": round(float(np.mean(agree)) * (1 - min(1.0, float(np.mean(div)))), 3) if agree and div else None,
    }
    return results


def run_image_analysis(original_path: Path, out_dir: Path, reference: ReferenceIndex,
                       analysis_context: dict | None = None, run_stability: bool = True) -> dict:
    ctx = {k: v for k, v in (analysis_context or {}).items() if k in ALLOWED_CONTEXT_KEYS}
    leaked = set((analysis_context or {}).keys()) - ALLOWED_CONTEXT_KEYS
    if leaked:
        raise ValueError(f"metadata firewall: keys {sorted(leaked)} must not be passed to the analysis pipeline")
    depicted = bool(ctx.get("depicted", True))
    wording = WORDING_DEPICTED if depicted else WORDING_LIVING
    affinity_fields = tuple(ctx.get("affinity_fields") or ("culture", "region", "period", "material"))
    max_faces = int(ctx.get("max_faces") or 12)

    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    img, scale = _load_working_image(original_path)
    report: dict = {
        "schema": "global-face-analysis/1.0",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "input": {"original_file": original_path.name, "sha256": _sha256(original_path), "phash": phash(img),
                  "working_scale": round(scale, 4), "working_size": [int(img.shape[1]), int(img.shape[0])],
                  "analysis_context": {"domain": ctx.get("domain"), "depicted": depicted}},
        "versions": {**versions.COMPONENT_VERSIONS, "dataset": reference.dataset_version,
                     "python": platform.python_version(), "opencv": cv2.__version__},
        "parameters": {"working_max_side": MAX_SIDE, "illumination_normaliser": DEFAULT_NORMALISER,
                       "stability_tests": run_stability, "max_faces": max_faces, "affinity_fields": list(affinity_fields)},
        "wording": wording, "status": "started", "warnings": [], "faces": [],
    }

    face_box = ctx.get("face_box")
    if face_box:
        # user-drawn rectangle, given in ORIGINAL pixels -> working pixels
        fb = [float(v) * scale for v in face_box]
        faces = [landmarks_from_box(img, (fb[0], fb[1], fb[2], fb[3]))]
        report["parameters"]["face_box_original_px"] = [int(v) for v in face_box]
    else:
        faces = localise_all(img, max_faces=max_faces)
    manual_note = [n for f in faces for n in f.notes if f.landmarks is None]
    faces = [f for f in faces if f.landmarks is not None]
    if manual_note:
        report["warnings"].extend(manual_note)
    primary_box = faces[0].box if faces else None
    q_img = assess_quality(img, primary_box)
    report["image_quality"] = q_img.to_dict()
    report["face_count"] = len(faces)
    cv2.imwrite(str(out_dir / "working.jpg"), img)
    cv2.imwrite(str(out_dir / "overlay.jpg"), _overlay(img, faces))

    if not faces or (not q_img.accepted and len(faces) == 0):
        report["status"] = "rejected"
        report["rejection_reason"] = "no analysable face found" if not faces else "insufficient image quality: " + "; ".join(q_img.issues)
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["duration_s"] = round(time.time() - t0, 2)
        (out_dir / "analysis.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
        return report

    norm_img = NORMALISERS[DEFAULT_NORMALISER](img)
    cv2.imwrite(str(out_dir / "normalised.jpg"), norm_img)

    for i, loc in enumerate(faces):
        fdir = out_dir / f"face_{i}"
        fdir.mkdir(exist_ok=True)
        q = assess_quality(img, loc.box)
        face_rep: dict = {"face_index": i, "box": loc.box, "detector": loc.detector, "detection_confidence": loc.confidence,
                          "notes": loc.notes, "quality": q.to_dict()}
        if not q.accepted:
            face_rep["status"] = "rejected"
            face_rep["rejection_reason"] = "insufficient face quality: " + "; ".join(q.issues)
            report["faces"].append(face_rep)
            continue
        pose = estimate_pose(loc.landmarks, img.shape)
        expr, aus, feats, z, aligned, lm_al = _expression_from(norm_img, loc.landmarks, pose.pose_quality)
        emb = face_embedding(aligned, lm_al)
        np.save(fdir / "embedding.npy", emb)
        np.save(fdir / "landmarks_478.npy", loc.landmarks)
        cv2.imwrite(str(fdir / "aligned.jpg"), aligned)
        x0, y0, x1, y1 = loc.box
        pad = int(0.2 * max(x1 - x0, y1 - y0))
        crop = img[max(0, y0 - pad):min(img.shape[0], y1 + pad), max(0, x0 - pad):min(img.shape[1], x1 + pad)]
        if crop.size:
            cv2.imwrite(str(fdir / "crop.jpg"), crop)
        top_label = max(expr.probabilities, key=expr.probabilities.get)
        morph = reference.morpho.analyse(morpho_subset(lm_al), group_fields=affinity_fields)
        nearest = reference.retrieve(emb, k=12, expression=top_label)
        attribution = suggest_attribution(nearest, morph.group_affinity)
        same_group = {}
        for fld in ("culture", "region"):
            if fld in attribution:
                same_group[fld] = {"value": attribution[fld]["value"],
                                   "specimens": reference.retrieve_by_field(emb, fld, attribution[fld]["value"], k=6)}
        # explanatory images
        au_dicts = [a.to_dict() for a in aus]
        try:
            muscle_map(aligned, lm_al, au_dicts, fdir / "muscle_map.jpg")
        except Exception as e:  # pragma: no cover
            log.warning("muscle map failed: %s", e)
        try:
            cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
            lm_crop = loc.landmarks.copy()
            lm_crop[:, 0] -= cx0
            lm_crop[:, 1] -= cy0
            morphometrics_figure(crop if crop.size else img, lm_crop, morph.to_dict(), reference, fdir / "morphometrics.png",
                                 group_field=next(iter(morph.group_affinity), affinity_fields[0] if affinity_fields else "culture"))
        except Exception as e:  # pragma: no cover
            log.warning("morphometrics figure failed: %s", e)
        face_rep.update({
            "status": "completed",
            "pose": pose.to_dict(),
            "facial_actions": {"action_units": [a.to_dict() for a in aus], "geometric_features": feats,
                               "z_scores_vs_neutral_template": z},
            "expression": {**expr.to_dict(), "ranked": sorted(expr.probabilities.items(), key=lambda kv: -kv[1])},
            "reference_retrieval": {"reference_index_size": reference.size,
                                    "nearest_overall": nearest[:10],
                                    "nearest_same_expression": reference.retrieve_by_expression(emb, top_label, k=6),
                                    "nearest_same_group": same_group},
            "suggested_attribution": {"wording": "Estimated by similarity to reference specimens (appearance + shape) — a suggestion "
                                                 "to be confirmed by the researcher, not a determination of origin.",
                                      "fields": attribution},
            "morphometrics": morph.to_dict(),
            "difficulty_level": difficulty_level(q.score, pose),
            "stability": stability_tests(norm_img, loc.landmarks, pose.pose_quality, expr.probabilities)
            if (run_stability and i < 3) else None,
            "artefacts": {"crop": f"face_{i}/crop.jpg", "aligned": f"face_{i}/aligned.jpg",
                          "muscle_map": f"face_{i}/muscle_map.jpg", "morphometrics": f"face_{i}/morphometrics.png"},
        })
        report["warnings"] += [f"face #{i}: {w}" for w in pose.warnings]
        report["faces"].append(face_rep)

    done = [f for f in report["faces"] if f.get("status") == "completed"]
    report["summary"] = {
        "image_quality_score": q_img.score,
        "faces_detected": len(faces),
        "faces_analysed": len(done),
        "primary_face": None if not done else {
            "index": done[0]["face_index"], "expression": done[0]["expression"]["primary"],
            "confidence": done[0]["expression"]["confidence"], "pose_quality": done[0]["pose"]["pose_quality"],
            "difficulty_level": done[0]["difficulty_level"],
            "morphological_affinity": {k: v["most_similar"] for k, v in done[0]["morphometrics"]["group_affinity"].items()},
            "suggested_attribution": {k: v["value"] for k, v in done[0]["suggested_attribution"]["fields"].items()},
        },
    }
    report["status"] = "completed" if done else "rejected"
    if not done:
        report["rejection_reason"] = "all detected faces failed the quality gate"
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["duration_s"] = round(time.time() - t0, 2)
    (out_dir / "analysis.json").write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    return report


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
