"""Head-pose (yaw / pitch / roll) estimation from 2-D landmarks via PnP."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from .landmarks import POSE_MODEL_POINTS


@dataclass
class PoseReport:
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    frontalness: float          # 1 = perfectly frontal, 0 = >= 60 deg off-axis
    pose_quality: float         # combined reliability for expression/morphometrics
    symmetry_index: float       # 1 = perfectly symmetric landmark layout
    warnings: list[str]

    def to_dict(self):
        return asdict(self)


def _rotation_to_euler(rvec: np.ndarray) -> tuple[float, float, float]:
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return float(np.degrees(pitch)), float(np.degrees(yaw)), float(np.degrees(roll))


def _wrap(a: float) -> float:
    """Wrap angle to [-90, 90] (pitch sign flips from PnP are common)."""
    while a > 90:
        a -= 180
    while a < -90:
        a += 180
    return a


def _mesh_axes_euler(lm: np.ndarray) -> tuple[float, float, float]:
    """Yaw/pitch/roll from the orientation of the landmark cloud's own axes.
    x-axis: right exocanthion -> left exocanthion; up-axis: chin -> forehead."""
    x = lm[263] - lm[33]
    up = lm[10] - lm[152]
    x = x / (np.linalg.norm(x) + 1e-9)
    up = up / (np.linalg.norm(up) + 1e-9)
    z = np.cross(x, up)
    z = z / (np.linalg.norm(z) + 1e-9)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    yaw = float(np.degrees(np.arcsin(np.clip(-R[0, 2], -1, 1))))
    pitch = _wrap(float(np.degrees(np.arctan2(R[1, 2], R[2, 2]))))
    roll = -_wrap(float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))))  # image y grows downward
    return yaw, pitch, roll


def estimate_pose(landmarks: np.ndarray, image_shape: tuple[int, int]) -> PoseReport:
    h, w = image_shape[:2]
    idx = list(POSE_MODEL_POINTS.keys())
    obj = np.array([POSE_MODEL_POINTS[i] for i in idx], dtype=np.float64)
    img_pts = landmarks[idx, :2].astype(np.float64)

    warnings: list[str] = []
    # Primary estimator: orientation of the face's own 3-D landmark axes
    # (MediaPipe supplies relative depth).  This is far less biased for
    # non-average morphologies than PnP against a generic 3-D template.
    try:
        yaw, pitch, roll = _mesh_axes_euler(landmarks)
    except Exception:
        yaw = pitch = roll = None

    # Secondary estimator (PnP) — used as fallback and recorded for QA.
    focal = w * 1.2
    cam = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(obj, img_pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if yaw is None:
        if not ok:
            warnings.append("pose estimation failed; pose set to zero")
            pitch = yaw = roll = 0.0
        else:
            pitch, yaw, roll = _rotation_to_euler(rvec)
            pitch, yaw, roll = _wrap(pitch), _wrap(yaw), _wrap(roll)
            warnings.append("mesh-axis pose failed; PnP fallback used")

    # Symmetry from bilateral landmark pairs relative to the midline
    pairs = [(33, 263), (133, 362), (61, 291), (234, 454), (172, 397), (129, 358)]
    mid_x = np.mean([landmarks[1, 0], landmarks[168, 0], landmarks[152, 0]])
    inter_oc = np.linalg.norm(landmarks[33, :2] - landmarks[263, :2]) + 1e-6
    asym = []
    for a, b in pairs:
        da = abs(landmarks[a, 0] - mid_x)
        db = abs(landmarks[b, 0] - mid_x)
        asym.append(abs(da - db) / inter_oc)
    symmetry = float(np.clip(1.0 - 2.5 * np.mean(asym), 0.0, 1.0))

    off_axis = np.sqrt(yaw ** 2 + pitch ** 2)
    frontalness = float(np.clip(1.0 - off_axis / 60.0, 0.0, 1.0))
    if abs(yaw) > 30:
        warnings.append(f"large yaw ({yaw:.0f} deg): expression and frontal morphometrics less reliable")
    if abs(pitch) > 25:
        warnings.append(f"large pitch ({pitch:.0f} deg)")
    if abs(roll) > 20:
        warnings.append(f"image rotated ({roll:.0f} deg roll)")
    pose_quality = float(np.clip(frontalness * (0.6 + 0.4 * symmetry), 0.0, 1.0))

    return PoseReport(
        yaw_deg=round(yaw, 1), pitch_deg=round(pitch, 1), roll_deg=round(roll, 1),
        frontalness=round(frontalness, 3), pose_quality=round(pose_quality, 3),
        symmetry_index=round(symmetry, 3), warnings=warnings,
    )
