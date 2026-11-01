"""Image-quality gate.

Runs BEFORE any inference. Produces a 0-100 quality score and a list of
issues. If the score is below the reject threshold the analysis is refused
(the model is never forced to produce an answer on unusable input).

Quality score and expression confidence are deliberately separate quantities.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

MIN_SIDE_PX = 160          # absolute minimum image side
GOOD_SIDE_PX = 800         # side length considered fully adequate
MIN_FACE_PX = 80           # face box side below this is unreliable for landmarks
GOOD_FACE_PX = 220
REJECT_THRESHOLD = 40.0


@dataclass
class QualityReport:
    score: float
    accepted: bool
    width: int
    height: int
    blur_laplacian_var: float
    mean_luminance: float
    contrast_std: float
    clipped_dark_fraction: float
    clipped_bright_fraction: float
    face_box_px: int | None
    face_fraction: float | None
    sub_scores: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _ramp(x: float, lo: float, hi: float) -> float:
    """Linear ramp 0..1 between lo and hi."""
    if hi == lo:
        return 1.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def assess_quality(image_bgr: np.ndarray, face_box: tuple[int, int, int, int] | None) -> QualityReport:
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Evaluate blur/exposure on the face region when available, else on the image.
    if face_box is not None:
        x0, y0, x1, y1 = face_box
        roi = gray[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
        if roi.size < 100:
            roi = gray
    else:
        roi = gray

    lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())
    mean_lum = float(roi.mean())
    contrast = float(roi.std())
    dark = float((roi < 8).mean())
    bright = float((roi > 247).mean())

    issues: list[str] = []
    sub: dict[str, float] = {}

    # Resolution
    side = min(w, h)
    sub["resolution"] = _ramp(side, MIN_SIDE_PX, GOOD_SIDE_PX)
    if side < MIN_SIDE_PX:
        issues.append(f"image too small ({w}x{h})")

    # Sharpness — Laplacian variance normalised for ROI size; thresholds empirical.
    sub["sharpness"] = _ramp(lap_var, 15.0, 150.0)
    if lap_var < 15.0:
        issues.append("image appears blurred")

    # Exposure: penalise very dark or very bright means.
    if mean_lum < 60:
        sub["exposure"] = _ramp(mean_lum, 20, 60)
        issues.append("under-exposed / very dark")
    elif mean_lum > 200:
        sub["exposure"] = 1.0 - _ramp(mean_lum, 200, 245)
        issues.append("over-exposed / very bright")
    else:
        sub["exposure"] = 1.0

    # Contrast
    sub["contrast"] = _ramp(contrast, 12.0, 45.0)
    if contrast < 12.0:
        issues.append("very low contrast")

    # Clipping (lost detail in shadows/highlights)
    clip = dark + bright
    sub["clipping"] = 1.0 - _ramp(clip, 0.05, 0.35)
    if clip > 0.2:
        issues.append("large clipped (pure black/white) areas")

    # Face size
    face_px = None
    face_frac = None
    if face_box is not None:
        x0, y0, x1, y1 = face_box
        face_px = int(max(x1 - x0, y1 - y0))
        face_frac = float(((x1 - x0) * (y1 - y0)) / (w * h))
        sub["face_size"] = _ramp(face_px, MIN_FACE_PX, GOOD_FACE_PX)
        if face_px < MIN_FACE_PX:
            issues.append(f"face too small ({face_px}px)")
        # Face touching border -> probably cropped
        margin = 2
        if x0 <= margin or y0 <= margin or x1 >= w - margin or y1 >= h - margin:
            issues.append("face touches the image border (possible cropping)")
            sub["face_size"] *= 0.8
    else:
        sub["face_size"] = 0.0
        issues.append("no face detected")

    weights = {
        "resolution": 0.15,
        "sharpness": 0.25,
        "exposure": 0.15,
        "contrast": 0.10,
        "clipping": 0.10,
        "face_size": 0.25,
    }
    score = 100.0 * sum(sub[k] * wgt for k, wgt in weights.items())
    if face_box is None:
        score = min(score, 25.0)

    return QualityReport(
        score=round(score, 1),
        accepted=score >= REJECT_THRESHOLD,
        width=w,
        height=h,
        blur_laplacian_var=round(lap_var, 2),
        mean_luminance=round(mean_lum, 2),
        contrast_std=round(contrast, 2),
        clipped_dark_fraction=round(dark, 4),
        clipped_bright_fraction=round(bright, 4),
        face_box_px=face_px,
        face_fraction=None if face_frac is None else round(face_frac, 4),
        sub_scores={k: round(v, 3) for k, v in sub.items()},
        issues=issues,
    )
