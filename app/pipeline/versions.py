"""Version registry for every component that can influence a result.

Every analysis report records these values so that any result can be
traced back to the exact software/model/dataset combination that produced it.
"""
from __future__ import annotations

import json
from pathlib import Path

PLATFORM_VERSION = "0.1.0"

# Frozen v0.1 components. Nothing here learns online.
COMPONENT_VERSIONS = {
    "platform": PLATFORM_VERSION,
    "face_detector": "mediapipe-face_detection-0.10.14 (BlazeFace full-range)",
    "landmarker": "mediapipe-face_mesh-0.10.14 (468+10 iris landmarks)",
    "pose_estimator": "pnp-generic3d-v0.1",
    "quality_gate": "quality-gate-v0.1",
    "illumination": "clahe-grayworld-v0.1",
    "action_units": "geometric-au-v0.1 (heuristic, FACS-inspired, domain-aware wording)",
    "expression_model": "emfacs-prototype-classifier-v0.1 (frozen)",
    "embedding": "procrustes-shape+hog-v0.1",
    "morphometrics": "gpa-pca-ward-v0.1",
    "retrieval": "cosine-knn-v0.1",
}


def dataset_version(reference_dir: Path) -> str:
    """Read the dataset version stamped by scripts/build_reference_index.py."""
    meta = reference_dir / "index.json"
    if not meta.exists():
        return "GlobalFaceCorpus (no reference index built)"
    try:
        return json.loads(meta.read_text(encoding="utf-8")).get("dataset_version", "unknown")
    except Exception:
        return "unknown"
