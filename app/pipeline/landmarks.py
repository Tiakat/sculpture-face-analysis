"""Face localisation and dense landmarking (frozen pretrained models).

Strategy (robust to sculpture photographs):
 1. Try MediaPipe FaceDetection (full-range BlazeFace).
 2. Fallback: OpenCV Haar cascade.
 3. Fallback: assume the whole image is the face (many museum photos are
    tight head shots) — flagged as low confidence.
 4. Run MediaPipe FaceMesh on the padded crop; map back to image coordinates.

All landmarks are returned in full-image pixel coordinates as an (N, 3)
array (x, y, z-relative).  Index semantics follow MediaPipe FaceMesh.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")
import mediapipe as mp  # noqa: E402

log = logging.getLogger(__name__)

_mp_fd = mp.solutions.face_detection
_mp_fm = mp.solutions.face_mesh

# Anthropometric landmark subset used for morphometrics (MediaPipe indices).
# Chosen to mirror the classical points used in Millones (Moche portrait
# vessels, frontal view): brow, orbit, nose, mouth, cheek and jaw points that
# are precise and replicable.
MORPHO_LANDMARKS: dict[str, int] = {
    "trichion_approx": 10,       # top of forehead midline
    "glabella": 9,
    "nasion": 168,
    "pronasale": 1,
    "subnasale": 2,
    "alare_R": 129,
    "alare_L": 358,
    "endocanthion_R": 133,
    "endocanthion_L": 362,
    "exocanthion_R": 33,
    "exocanthion_L": 263,
    "palpebrale_sup_R": 159,
    "palpebrale_inf_R": 145,
    "palpebrale_sup_L": 386,
    "palpebrale_inf_L": 374,
    "brow_inner_R": 107,
    "brow_inner_L": 336,
    "brow_mid_R": 105,
    "brow_mid_L": 334,
    "brow_outer_R": 70,
    "brow_outer_L": 300,
    "cheilion_R": 61,
    "cheilion_L": 291,
    "labiale_superius": 0,
    "stomion_sup": 13,
    "stomion_inf": 14,
    "labiale_inferius": 17,
    "sublabiale": 18,
    "gnathion": 152,
    "zygion_R": 234,
    "zygion_L": 454,
    "gonion_R": 172,
    "gonion_L": 397,
}
MORPHO_ORDER = list(MORPHO_LANDMARKS.keys())

# Generic 3-D face model (mm) for PnP head-pose estimation.
POSE_MODEL_POINTS = {
    1: (0.0, 0.0, 0.0),           # nose tip
    152: (0.0, -63.6, -12.5),     # chin
    33: (-43.3, 32.7, -26.0),     # right eye outer corner (image right = subject left)
    263: (43.3, 32.7, -26.0),     # left eye outer corner
    61: (-28.9, -28.9, -24.1),    # right mouth corner
    291: (28.9, -28.9, -24.1),    # left mouth corner
    168: (0.0, 36.0, -18.0),      # nasion
}


@dataclass
class FaceLocalisation:
    box: tuple[int, int, int, int] | None      # x0, y0, x1, y1 in image px
    detector: str
    confidence: float
    landmarks: np.ndarray | None = None        # (478, 3) full image px (z relative)
    notes: list[str] = field(default_factory=list)


def _detect_mediapipe(img_bgr: np.ndarray) -> tuple[tuple[int, int, int, int], float] | None:
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    best = None
    for model_sel in (1, 0):
        with _mp_fd.FaceDetection(model_selection=model_sel, min_detection_confidence=0.35) as fd:
            res = fd.process(rgb)
        if not res.detections:
            continue
        for det in res.detections:
            score = float(det.score[0])
            bb = det.location_data.relative_bounding_box
            x0 = int(bb.xmin * w)
            y0 = int(bb.ymin * h)
            x1 = int((bb.xmin + bb.width) * w)
            y1 = int((bb.ymin + bb.height) * h)
            if best is None or score > best[1]:
                best = ((x0, y0, x1, y1), score)
        if best:
            break
    return best


def _detect_haar(img_bgr: np.ndarray) -> tuple[tuple[int, int, int, int], float] | None:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(48, 48))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return (int(x), int(y), int(x + w), int(y + h)), 0.5


def _pad_box(box, w, h, pad=0.35):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    cx, cy = x0 + bw / 2, y0 + bh / 2
    side = max(bw, bh) * (1 + 2 * pad)
    nx0 = int(max(0, cx - side / 2))
    ny0 = int(max(0, cy - side / 2))
    nx1 = int(min(w, cx + side / 2))
    ny1 = int(min(h, cy + side / 2))
    return nx0, ny0, nx1, ny1


def _facemesh(crop_bgr: np.ndarray) -> np.ndarray | None:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    with _mp_fm.FaceMesh(static_image_mode=True, refine_landmarks=True,
                         max_num_faces=1, min_detection_confidence=0.3) as fm:
        res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None
    ch, cw = crop_bgr.shape[:2]
    lm = res.multi_face_landmarks[0].landmark
    arr = np.array([[p.x * cw, p.y * ch, p.z * cw] for p in lm], dtype=np.float64)
    return arr


def localise_and_landmark(img_bgr: np.ndarray) -> FaceLocalisation:
    h, w = img_bgr.shape[:2]
    notes: list[str] = []

    det = _detect_mediapipe(img_bgr)
    detector = "mediapipe_blazeface"
    if det is None:
        det = _detect_haar(img_bgr)
        detector = "opencv_haar"
        if det is not None:
            notes.append("primary detector failed; Haar cascade fallback used")
    if det is None:
        # Whole-image fallback (typical for tight museum head shots)
        det = ((0, 0, w, h), 0.2)
        detector = "whole_image_fallback"
        notes.append("no detector fired; whole image treated as face region (low confidence)")

    box, conf = det
    # Try landmarks on progressively larger padded crops.
    landmarks = None
    for pad in (0.35, 0.6, 1.0):
        px0, py0, px1, py1 = _pad_box(box, w, h, pad)
        crop = img_bgr[py0:py1, px0:px1]
        if crop.size == 0:
            continue
        lm = _facemesh(crop)
        if lm is not None:
            lm[:, 0] += px0
            lm[:, 1] += py0
            landmarks = lm
            break
    if landmarks is None and detector != "whole_image_fallback":
        lm = _facemesh(img_bgr)
        if lm is not None:
            landmarks = lm
            notes.append("landmarks obtained on full image rather than crop")

    if landmarks is not None:
        # Refine the box from the landmarks (tighter, more consistent).
        xs, ys = landmarks[:, 0], landmarks[:, 1]
        box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        if detector == "whole_image_fallback":
            conf = 0.5
            notes.append("landmarks found despite detector failure; confidence raised to 0.5")
    else:
        notes.append("dense landmarking failed")
        if detector == "whole_image_fallback":
            box = None

    return FaceLocalisation(box=box, detector=detector, confidence=round(float(conf), 3),
                            landmarks=landmarks, notes=notes)


def morpho_subset(landmarks: np.ndarray) -> np.ndarray:
    """(K, 2) array of the anthropometric subset, in MORPHO_ORDER."""
    idx = [MORPHO_LANDMARKS[k] for k in MORPHO_ORDER]
    return landmarks[idx, :2].copy()


# --------------------------------------------------------------- multi-face
def _detect_mediapipe_all(img_bgr: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
    """All faces found by BlazeFace (full-range first, short-range fallback), largest first."""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    for model_sel in (1, 0):
        with _mp_fd.FaceDetection(model_selection=model_sel, min_detection_confidence=0.35) as fd:
            res = fd.process(rgb)
        if not res.detections:
            continue
        out = []
        for det in res.detections:
            score = float(det.score[0])
            bb = det.location_data.relative_bounding_box
            x0, y0 = int(bb.xmin * w), int(bb.ymin * h)
            x1, y1 = int((bb.xmin + bb.width) * w), int((bb.ymin + bb.height) * h)
            out.append(((x0, y0, x1, y1), score))
        return sorted(out, key=lambda d: -(d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))
    return []


def _detect_haar_all(img_bgr: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
    gray = cv2.equalizeHist(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(48, 48))
    return [((int(x), int(y), int(x + w), int(y + h)), 0.5) for (x, y, w, h) in faces]


def localise_all(img_bgr: np.ndarray, max_faces: int = 12) -> list[FaceLocalisation]:
    """Multi-face localisation + landmarking (group paintings, reliefs, photographs).
    Falls back to the single-face strategy when no detector fires."""
    h, w = img_bgr.shape[:2]
    dets = _detect_mediapipe_all(img_bgr)
    detector = "mediapipe_blazeface"
    if not dets:
        dets = _detect_haar_all(img_bgr)
        detector = "opencv_haar"
    if not dets:
        single = localise_and_landmark(img_bgr)
        return [single] if single.landmarks is not None else []
    out: list[FaceLocalisation] = []
    for box, conf in dets[:max_faces]:
        landmarks = None
        for pad in (0.35, 0.6):
            px0, py0, px1, py1 = _pad_box(box, w, h, pad)
            crop = img_bgr[py0:py1, px0:px1]
            if crop.size == 0:
                continue
            lm = _facemesh(crop)
            if lm is not None:
                lm[:, 0] += px0
                lm[:, 1] += py0
                landmarks = lm
                break
        notes = [] if landmarks is not None else ["dense landmarking failed for this detection"]
        if landmarks is not None:
            xs, ys = landmarks[:, 0], landmarks[:, 1]
            box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        out.append(FaceLocalisation(box=box, detector=detector, confidence=round(float(conf), 3),
                                    landmarks=landmarks, notes=notes))
    return out


# ------------------------------------------------------------------ robust fallbacks for museum photographs
def _crop_search(img_bgr: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Museum photographs show the whole object centred with wide margins, so the face is small
    relative to the frame and BlazeFace misses it.  Try FaceMesh on centred / upper crops of the
    original and of a contrast-normalised copy."""
    h, w = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    normalised = cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)
    windows = []
    for frac in (0.7, 0.5, 0.35):
        cw, ch = int(w * frac), int(h * frac)
        windows.append((f"centre_{frac}", (w - cw) // 2, (h - ch) // 2, cw, ch))
    windows.append(("upper_half", int(w * 0.2), 0, int(w * 0.6), int(h * 0.55)))
    windows.append(("upper_third", int(w * 0.25), 0, int(w * 0.5), int(h * 0.4)))
    for name, x0, y0, cw, ch in windows:
        for vname, im in (("orig", img_bgr), ("clahe", normalised)):
            crop = im[y0:y0 + ch, x0:x0 + cw]
            if crop.size == 0:
                continue
            lm = _facemesh(crop)
            if lm is not None:
                lm[:, 0] += x0
                lm[:, 1] += y0
                return lm, f"facemesh_{name}_{vname}"
    return None, ""


_original_localise = localise_and_landmark


def localise_and_landmark(img_bgr: np.ndarray) -> FaceLocalisation:  # noqa: F811 — wraps the basic strategy
    loc = _original_localise(img_bgr)
    if loc.landmarks is not None:
        return loc
    lm, how = _crop_search(img_bgr)
    if lm is None:
        return loc
    xs, ys = lm[:, 0], lm[:, 1]
    return FaceLocalisation(box=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())), detector=how, confidence=0.45,
                            landmarks=lm, notes=["face found by crop search (object photographed with wide margins)"])


# ------------------------------------------------------------------ stylised faces (African, Oceanic, abstract) + manual box
def _facemesh_relaxed(crop_bgr: np.ndarray) -> np.ndarray | None:
    """FaceMesh tuned for stylised / non-naturalistic faces: the crop is upscaled so the face is
    at least ~400 px, and detection confidence is lowered step by step; the same is tried on a
    contrast-normalised copy and on a slightly blurred copy (masks and polished wood/bronze show
    specular highlights and carved lines that confuse the detector)."""
    if crop_bgr.size == 0:
        return None
    ch, cw = crop_bgr.shape[:2]
    original = crop_bgr
    scale = 1.0
    if max(ch, cw) < 400:
        scale = 400.0 / max(ch, cw)
        crop_bgr = cv2.resize(crop_bgr, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.cvtColor(cv2.merge([cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l), a, b]), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(clahe, (0, 0), 2.0)
    variants = (("orig", crop_bgr), ("clahe", clahe), ("blur", blurred))
    for conf in (0.3, 0.15, 0.05):
        for _name, im in variants:
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            with _mp_fm.FaceMesh(static_image_mode=True, refine_landmarks=True,
                                 max_num_faces=1, min_detection_confidence=conf) as fm:
                res = fm.process(rgb)
            if res.multi_face_landmarks:
                h2, w2 = im.shape[:2]
                lm = res.multi_face_landmarks[0].landmark
                arr = np.array([[p.x * w2, p.y * h2, p.z * w2] for p in lm], dtype=np.float64)
                arr[:, :2] /= scale
                arr[:, 2] /= scale
                if _plausible_face(original, arr):
                    return arr
    return None


def _plausible_face(img_bgr: np.ndarray, lm: np.ndarray) -> bool:
    """Reject FaceMesh hallucinations on featureless regions (a plain vase body, a circle, a wall).
    FaceMesh always returns a mesh once its detector fires; at relaxed confidence it can fire on
    nothing.  A real face — carved, painted or photographed — has edge energy around the eyes and
    the mouth that is clearly higher than on the cheeks."""
    h, w = img_bgr.shape[:2]
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    def region_mean(idx):
        pts = lm[idx, :2]
        x0, y0 = int(max(0, pts[:, 0].min())), int(max(0, pts[:, 1].min()))
        x1, y1 = int(min(w, pts[:, 0].max() + 1)), int(min(h, pts[:, 1].max() + 1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return 0.0
        return float(mag[y0:y1, x0:x1].mean())

    left_eye = [33, 133, 159, 145, 160, 144, 158, 153]
    right_eye = [362, 263, 386, 374, 385, 380, 387, 373]
    mouth = [61, 291, 0, 17, 13, 14, 78, 308]
    cheeks = [50, 280, 101, 330, 205, 425, 187, 411]
    regions = [region_mean(left_eye), region_mean(right_eye), region_mean(mouth)]
    feat = float(np.mean(regions))
    cheek = region_mean(cheeks) + 1e-6
    face_w = float(lm[:, 0].max() - lm[:, 0].min())
    # texture inside the face hull: a featureless surface has almost none
    hull = cv2.convexHull(lm[:, :2].astype(np.int32))
    mask = np.zeros_like(grey)
    cv2.fillConvexPoly(mask, hull, 255)
    inside = grey[mask > 0]
    texture = float(inside.std()) if inside.size > 50 else 0.0
    return feat >= 6.0 and min(regions) >= 3.0 and feat >= 1.3 * cheek and face_w >= 24 and texture >= 10.0


def landmarks_from_box(img_bgr: np.ndarray, box: tuple[int, int, int, int]) -> FaceLocalisation:
    """Manual fallback: the user draws the face rectangle (x0, y0, x1, y1 in original pixels).
    The rectangle is geometric guidance only — it carries no cultural information."""
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, x1 = sorted((max(0, x0), min(w, x1)))
    y0, y1 = sorted((max(0, y0), min(h, y1)))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return FaceLocalisation(box=(x0, y0, x1, y1), detector="manual_box", confidence=0.0, landmarks=None,
                                notes=["manual face box too small"])
    for pad in (0.25, 0.5, 0.9):
        px0, py0, px1, py1 = _pad_box((x0, y0, x1, y1), w, h, pad)
        lm = _facemesh_relaxed(img_bgr[py0:py1, px0:px1])
        if lm is not None:
            lm[:, 0] += px0
            lm[:, 1] += py0
            return FaceLocalisation(box=(x0, y0, x1, y1), detector="manual_box+facemesh_relaxed", confidence=0.5,
                                    landmarks=lm, notes=["face region given manually; landmarks fitted with relaxed detection"])
    return FaceLocalisation(box=(x0, y0, x1, y1), detector="manual_box", confidence=0.0, landmarks=None,
                            notes=["face region given manually but the landmark model could not fit this face (too abstract or too damaged)"])


_localise_with_crop_search = localise_and_landmark


def localise_and_landmark(img_bgr: np.ndarray) -> FaceLocalisation:  # noqa: F811 — adds the stylised-face pass
    loc = _localise_with_crop_search(img_bgr)
    if loc.landmarks is not None:
        return loc
    h, w = img_bgr.shape[:2]
    # Last automatic attempt: relaxed FaceMesh on the whole frame, the centre and the upper part.
    for name, (x0, y0, x1, y1) in (("whole", (0, 0, w, h)),
                                   ("centre", (int(w * 0.15), int(h * 0.05), int(w * 0.85), int(h * 0.75))),
                                   ("upper", (int(w * 0.2), 0, int(w * 0.8), int(h * 0.5)))):
        lm = _facemesh_relaxed(img_bgr[y0:y1, x0:x1])
        if lm is not None:
            lm[:, 0] += x0
            lm[:, 1] += y0
            xs, ys = lm[:, 0], lm[:, 1]
            return FaceLocalisation(box=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                                    detector=f"facemesh_relaxed_{name}", confidence=0.35, landmarks=lm,
                                    notes=["stylised face: landmarks fitted with relaxed detection (lower confidence)"])
    loc.notes.append("no face fitted automatically — draw the face rectangle on the image and run again")
    return loc
