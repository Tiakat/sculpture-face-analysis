"""Illumination normalisation and geometric alignment.

Several normalisation variants are exposed so the platform can measure
prediction *stability* under different photometric corrections rather
than assuming one is correct.
"""
from __future__ import annotations

import cv2
import numpy as np

ALIGNED_SIZE = 256


def grayworld(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0) + 1e-6
    gain = means.mean() / means
    out = np.clip(img * gain, 0, 255).astype(np.uint8)
    return out


def clahe(img_bgr: np.ndarray, clip: float = 2.0, grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    l2 = c.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def brightness_normalize(img_bgr: np.ndarray, target_mean: float = 128.0) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    m = gray.mean() + 1e-6
    gain = target_mean / m
    return np.clip(img_bgr.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def histogram_equalize(img_bgr: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


NORMALISERS = {
    "original": lambda im: im,
    "brightness": brightness_normalize,
    "histogram_eq": histogram_equalize,
    "clahe": clahe,
    "grayworld_clahe": lambda im: clahe(grayworld(im)),
}

DEFAULT_NORMALISER = "grayworld_clahe"


def align_face(img_bgr: np.ndarray, landmarks: np.ndarray, size: int = ALIGNED_SIZE):
    """Similarity-align the face so the eyes are horizontal and at fixed
    positions. Returns (aligned_image, transformed_landmarks, matrix)."""
    r_eye = landmarks[[33, 133], :2].mean(axis=0)
    l_eye = landmarks[[263, 362], :2].mean(axis=0)
    dst_r = np.array([size * 0.32, size * 0.40])
    dst_l = np.array([size * 0.68, size * 0.40])
    src = np.float32([r_eye, l_eye])
    dst = np.float32([dst_r, dst_l])
    M, _ = cv2.estimateAffinePartial2D(src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2))
    if M is None:
        M = np.float32([[1, 0, 0], [0, 1, 0]])
    aligned = cv2.warpAffine(img_bgr, M, (size, size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    pts = np.hstack([landmarks[:, :2], np.ones((len(landmarks), 1))])
    lm2 = (M @ pts.T).T
    lm_out = landmarks.copy()
    lm_out[:, :2] = lm2
    return aligned, lm_out, M
