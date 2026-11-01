import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from skimage import data

os.environ["GFP_DATA_DIR"] = str(Path("/tmp/mfp_pytest_data"))

from app.pipeline.action_units import estimate_action_units  # noqa: E402
from app.pipeline.expression import classify_expression, UNKNOWN  # noqa: E402
from app.pipeline.landmarks import localise_and_landmark, morpho_subset  # noqa: E402
from app.pipeline.morphometrics import gpa, procrustes_distance, MorphometricReference  # noqa: E402
from app.pipeline.normalization import align_face  # noqa: E402
from app.pipeline.pose import estimate_pose  # noqa: E402
from app.pipeline.quality import assess_quality  # noqa: E402
from app.pipeline.reference_index import ReferenceIndex  # noqa: E402
from app.pipeline.runner import run_image_analysis, phash, hamming
from app.pipeline.landmarks import localise_all  # noqa: E402
from training.split import grouped_split, assert_no_leakage  # noqa: E402


@pytest.fixture(scope="module")
def face_bgr():
    return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)


@pytest.fixture(scope="module")
def landmarks(face_bgr):
    loc = localise_and_landmark(face_bgr)
    assert loc.landmarks is not None and loc.landmarks.shape[0] >= 468
    return loc.landmarks


def test_quality_rejects_blank_image():
    q = assess_quality(np.zeros((300, 300, 3), np.uint8), None)
    assert not q.accepted and "no face detected" in q.issues


def test_quality_accepts_good_face(face_bgr):
    loc = localise_and_landmark(face_bgr)
    q = assess_quality(face_bgr, loc.box)
    assert q.accepted and q.score > 50


def test_pose_roll_tracks_image_rotation(face_bgr):
    h, w = face_bgr.shape[:2]
    base = estimate_pose(localise_and_landmark(face_bgr).landmarks, face_bgr.shape).roll_deg
    rot = cv2.warpAffine(face_bgr, cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1), (w, h))
    r = estimate_pose(localise_and_landmark(rot).landmarks, rot.shape).roll_deg
    assert abs(abs(r - base) - 15) < 6


def test_smile_yields_au12_and_happiness(face_bgr, landmarks):
    _, lm_al, _ = align_face(face_bgr, landmarks)
    aus, feats, z = estimate_action_units(lm_al)
    au = {a.code: a for a in aus}
    assert au["AU12"].present and au["AU12"].intensity > 2
    res = classify_expression(aus, 1.0, feats)
    assert res.primary == "happiness"
    assert abs(sum(res.probabilities.values()) - 1) < 1e-3  # probabilities are rounded to 4 dp
    assert set(res.region_contributions) == {"brow", "eyes", "nose", "mouth", "jaw"}


def test_neutral_template_gives_neutral():
    """Landmarks exactly at the template mean → no AU evidence → neutral."""
    from app.pipeline.action_units import AU_DEFS, ActionUnit
    aus = [ActionUnit(code=c, name=n, associated_musculature=m, intensity=0, evidence=0.0, present=False,
                      contributing_features={}) for c, n, m, _ in AU_DEFS]
    res = classify_expression(aus, 1.0)
    assert res.primary == "neutral" and res.confidence > 0.5


def test_unknown_when_contradictory():
    from app.pipeline.action_units import AU_DEFS, ActionUnit
    strong = {"AU12", "AU15", "AU4", "AU5", "AU26", "AU1", "AU9"}  # incompatible mixture
    aus = [ActionUnit(code=c, name=n, associated_musculature=m, intensity=5 if c in strong else 0,
                      evidence=1.0 if c in strong else 0.0, present=c in strong, contributing_features={})
           for c, n, m, _ in AU_DEFS]
    res = classify_expression(aus, 1.0)
    assert res.is_unknown and res.primary == UNKNOWN


def test_procrustes_invariance(landmarks):
    a = morpho_subset(landmarks)
    theta = 0.4
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    b = (a @ R.T) * 2.5 + np.array([100, -40])
    assert procrustes_distance(a, b) < 1e-6
    aligned, consensus = gpa(np.stack([a, b, a * 1.1]))
    assert aligned.shape == (3, a.shape[0], 2)


def test_morphometric_reference_affinity():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(33, 2))
    configs, meta = [], []
    for g, shift in (("A", 0.0), ("B", 0.6)):
        for i in range(8):
            c = base + rng.normal(scale=0.05, size=base.shape)
            c[:5] += shift
            configs.append(c)
            meta.append({"id": f"{g}{i}", "culture": g})
    ref = MorphometricReference(np.stack(configs), meta)
    q = base.copy(); q[:5] += 0.6
    res = ref.analyse(q)
    assert res.group_affinity["culture"]["most_similar"] == "B"
    assert res.reference_sample_size == 16 and len(res.pca_scores) > 0


def test_full_run_and_rejection(tmp_path, face_bgr):
    p = tmp_path / "face.jpg"
    cv2.imwrite(str(p), face_bgr)
    ref = ReferenceIndex(tmp_path / "no_index")
    rep = run_image_analysis(p, tmp_path / "out", ref, {"domain": "photograph_person", "depicted": False}, run_stability=True)
    assert rep["status"] == "completed" and rep["face_count"] == 1
    assert (tmp_path / "out" / "analysis.json").exists()
    f0 = rep["faces"][0]
    assert f0["stability"]["summary"]["perturbation_agreement_rate"] is not None
    assert f0["expression"]["primary"] == "happiness" and f0["difficulty_level"] in range(7)
    assert "Perceived" in rep["wording"]["expression"]
    with pytest.raises(ValueError):
        run_image_analysis(p, tmp_path / "out3", ref, {"culture": "moche"})
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((256, 256, 3), 40, np.uint8))
    rep2 = run_image_analysis(blank, tmp_path / "out2", ref, {}, run_stability=False)
    assert rep2["status"] == "rejected"


def test_multi_face_and_phash(face_bgr):
    two = np.concatenate([face_bgr, face_bgr[:, ::-1]], axis=1)
    locs = [l for l in localise_all(two) if l.landmarks is not None]
    assert len(locs) == 2
    h1, h2 = phash(face_bgr), phash(cv2.GaussianBlur(face_bgr, (5, 5), 0))
    assert hamming(h1, h2) <= 6
    assert hamming(h1, phash(np.zeros_like(face_bgr))) > 10


def test_grouped_split_has_no_leakage():
    recs = [{"id": f"img{i}", "object_number": f"obj{i // 3}", "expression": ["neutral", "happiness"][i % 2]} for i in range(60)]
    s = grouped_split(recs, seed=1)
    assert_no_leakage(s["train"], s["test"])
    assert len(s["test"]) > 0 and len(s["train"]) > len(s["test"])
