"""Face embedding for reference retrieval (frozen v0.1).

Embedding = [Procrustes shape coordinates (weighted)] ++ [HOG of the aligned
grey face crop].  Deterministic, no learned weights, therefore fully
reproducible and independent of any human-face training set.  It can be
replaced by a learned embedding in a later model version without changing
the retrieval interface.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog

from .landmarks import morpho_subset
from .morphometrics import _center_scale

HOG_SIZE = 128


def face_embedding(aligned_face_bgr: np.ndarray, aligned_landmarks: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(aligned_face_bgr, cv2.COLOR_BGR2GRAY)
    # crop the central face region of the 256x256 aligned image
    h = gray.shape[0]
    crop = gray[int(h * 0.15): int(h * 0.95), int(h * 0.12): int(h * 0.88)]
    crop = cv2.resize(crop, (HOG_SIZE, HOG_SIZE))
    crop = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(crop)
    appearance = hog(crop, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    appearance = appearance / (np.linalg.norm(appearance) + 1e-9)

    shape = _center_scale(morpho_subset(aligned_landmarks)).ravel()
    shape = shape / (np.linalg.norm(shape) + 1e-9)

    # weight shape and appearance equally in the cosine space
    emb = np.concatenate([shape * np.sqrt(0.5), appearance * np.sqrt(0.5)]).astype(np.float32)
    return emb


def cosine_similarity(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-9)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return Bn @ a
