"""Explanatory images.

1. muscle_map()        — the aligned face with the musculature *associated* with
                         each detected Action Unit drawn as coloured paths anchored
                         on landmarks (thicker/brighter = stronger evidence), plus
                         a legend.  Wording on the image: "associated musculature",
                         never "muscle movement".
2. morphometrics_figure() — a 3-panel figure in the spirit of Millones' graphics:
                         (a) the 33 anthropometric landmarks digitised on the face
                         with the wireframe used for the indices, (b) the query
                         superimposed on the reference consensus with deviation
                         vectors (Procrustes residuals), (c) PCA scatter PC1×PC2 of
                         the reference sample with 95 % group ellipses and the
                         query as a star.  Panels (b)/(c) need a reference index.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .landmarks import MORPHO_LANDMARKS, MORPHO_ORDER

# Landmark-anchored muscle paths (MediaPipe indices). R = image-left side of the subject.
MUSCLE_PATHS = {
    "AU1": [("frontalis (medial)", [(107, 108, 69, 104), (336, 337, 299, 333)])],
    "AU2": [("frontalis (lateral)", [(70, 63, 105, 66), (300, 293, 334, 296)])],
    "AU4": [("corrugator supercilii", [(107, 55, 8), (336, 285, 8)]), ("procerus", [(8, 168, 6)])],
    "AU5": [("levator palpebrae", [(159, 160, 161, 246), (386, 387, 388, 466)])],
    "AU6": [("orbicularis oculi (orbital)", [(70, 63, 105, 66, 107, 55, 189, 245, 128, 121, 47, 100, 118, 50, 123, 116, 143, 156, 70),
                                              (300, 293, 334, 296, 336, 285, 413, 465, 357, 350, 277, 329, 347, 280, 352, 345, 372, 383, 300)])],
    "AU7": [("orbicularis oculi (palpebral)", [(33, 160, 158, 133, 153, 144, 33), (263, 387, 385, 362, 380, 373, 263)])],
    "AU9": [("levator labii sup. alaeque nasi", [(129, 49, 209, 128, 133), (358, 279, 429, 357, 362)])],
    "AU10": [("levator labii superioris", [(50, 205, 36, 37), (280, 425, 266, 267)])],
    "AU12": [("zygomaticus major", [(116, 123, 50, 205, 61), (345, 352, 280, 425, 291)])],
    "AU15": [("depressor anguli oris", [(61, 91, 182, 172), (291, 321, 406, 397)])],
    "AU17": [("mentalis", [(84, 181, 152), (314, 405, 152), (17, 18, 152)])],
    "AU20": [("risorius", [(61, 146, 132), (291, 375, 361)])],
    "AU23": [("orbicularis oris", [(61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91, 61)])],
    "AU25": [("depressor labii inferioris", [(84, 181, 148), (314, 405, 377)])],
    "AU26": [("masseter / jaw", [(234, 132, 172), (454, 361, 397)])],
}
PALETTE = [(60, 76, 231), (0, 165, 255), (0, 200, 80), (200, 120, 0), (180, 0, 160), (255, 80, 80), (0, 220, 220),
           (120, 200, 0), (255, 0, 130), (90, 90, 255), (0, 120, 200), (170, 100, 40), (255, 160, 0), (140, 0, 255), (0, 90, 0)]


def muscle_map(aligned_bgr: np.ndarray, lm_aligned: np.ndarray, action_units: list[dict], out_path: Path,
               min_evidence: float = 0.15) -> list[str]:
    """Draw associated musculature for active AUs. Returns the list of AUs drawn."""
    scale = 2 if aligned_bgr.shape[0] < 400 else 1
    face = cv2.resize(aligned_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else aligned_bgr.copy()
    lm_aligned = lm_aligned.copy()
    lm_aligned[:, :2] *= scale
    h, w = face.shape[:2]
    overlay = face.copy()
    active = [a for a in action_units if a["evidence"] >= min_evidence]
    active.sort(key=lambda a: -a["evidence"])
    drawn = []
    for i, au in enumerate(active):
        color = PALETTE[i % len(PALETTE)]
        ev = float(au["evidence"])
        thick = max(1, int(round(1 + 3 * ev)))
        for name, paths in MUSCLE_PATHS.get(au["code"], []):
            for path in paths:
                pts = np.array([lm_aligned[j, :2] for j in path], dtype=np.int32)
                closed = path[0] == path[-1]
                cv2.polylines(overlay, [pts], closed, color, thick, cv2.LINE_AA)
                for p in pts:
                    cv2.circle(overlay, tuple(p), thick + 1, color, -1, cv2.LINE_AA)
        drawn.append(au["code"])
    alpha = 0.75
    face = cv2.addWeighted(overlay, alpha, face, 1 - alpha, 0)

    # legend panel
    legend_w = 470
    canvas = np.full((max(h, 40 + 22 * (len(active) + 2)), w + legend_w, 3), 250, np.uint8)
    canvas[:h, :w] = face
    y = 24
    cv2.putText(canvas, "Associated musculature per AU", (w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    y += 8
    for i, au in enumerate(active):
        color = PALETTE[i % len(PALETTE)]
        y += 22
        cv2.rectangle(canvas, (w + 10, y - 12), (w + 26, y + 2), color, -1)
        muscles = "; ".join(n for n, _ in MUSCLE_PATHS.get(au["code"], []))
        cv2.putText(canvas, f"{au['code']} {au['name']}  (evidence {au['evidence']:.2f}, intensity {au['intensity']})", (w + 34, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        y += 15
        cv2.putText(canvas, muscles[:70], (w + 34, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (90, 90, 90), 1, cv2.LINE_AA)
    if not active:
        cv2.putText(canvas, "no AU above threshold (neutral configuration)", (w + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Inferred from geometry - an artwork has no muscle movement", (w + 10, canvas.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 60, 60), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)
    return drawn


# Wireframe edges between anthropometric landmarks (names from MORPHO_ORDER).
WIRE = [("trichion_approx", "glabella"), ("glabella", "nasion"), ("nasion", "pronasale"), ("pronasale", "subnasale"),
        ("subnasale", "labiale_superius"), ("labiale_superius", "stomion_sup"), ("stomion_inf", "labiale_inferius"),
        ("labiale_inferius", "sublabiale"), ("sublabiale", "gnathion"),
        ("exocanthion_R", "endocanthion_R"), ("exocanthion_L", "endocanthion_L"), ("endocanthion_R", "nasion"), ("endocanthion_L", "nasion"),
        ("brow_outer_R", "brow_mid_R"), ("brow_mid_R", "brow_inner_R"), ("brow_outer_L", "brow_mid_L"), ("brow_mid_L", "brow_inner_L"),
        ("alare_R", "pronasale"), ("alare_L", "pronasale"), ("alare_R", "subnasale"), ("alare_L", "subnasale"),
        ("cheilion_R", "labiale_superius"), ("cheilion_L", "labiale_superius"), ("cheilion_R", "labiale_inferius"), ("cheilion_L", "labiale_inferius"),
        ("zygion_R", "exocanthion_R"), ("zygion_L", "exocanthion_L"), ("zygion_R", "gonion_R"), ("zygion_L", "gonion_L"),
        ("gonion_R", "gnathion"), ("gonion_L", "gnathion"), ("zygion_R", "zygion_L")]


def _ellipse_95(points: np.ndarray):
    if len(points) < 3:
        return None
    mean = points.mean(axis=0)
    cov = np.cov(points.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    k = np.sqrt(5.991)  # chi2 95% 2 dof
    return mean, 2 * k * np.sqrt(max(vals[0], 1e-12)), 2 * k * np.sqrt(max(vals[1], 1e-12)), angle


def morphometrics_figure(crop_bgr: np.ndarray, lm_crop: np.ndarray, morph: dict, reference, out_path: Path,
                         group_field: str = "culture") -> None:
    """Three-panel figure (landmarks / Procrustes deviation / PCA scatter)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    has_ref = reference is not None and reference.morpho is not None and reference.morpho.pca is not None
    ncols = 3 if has_ref else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    axes = np.atleast_1d(axes)

    # (a) landmarks digitised on the face
    ax = axes[0]
    ax.imshow(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    P = {n: lm_crop[MORPHO_LANDMARKS[n], :2] for n in MORPHO_ORDER}
    for a, b in WIRE:
        ax.plot([P[a][0], P[b][0]], [P[a][1], P[b][1]], "-", color="#ffd166", lw=1.0, alpha=0.9)
    for i, n in enumerate(MORPHO_ORDER):
        ax.plot(P[n][0], P[n][1], "o", ms=4, color="#e63946", mec="white", mew=0.5)
        ax.annotate(str(i + 1), (P[n][0], P[n][1]), xytext=(3, -3), textcoords="offset points", fontsize=6, color="white",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, lw=0))
    ax.set_title("(a) Anthropometric landmarks (33) and index wireframe", fontsize=10)
    ax.axis("off")

    if has_ref:
        mr = reference.morpho
        q = np.array(morph["procrustes_coordinates"])
        cons = mr.consensus
        # display only: rotate both shapes so that the eye line is horizontal (GPA leaves an arbitrary
        # global orientation inherited from the first specimen)
        try:
            eR = np.mean([cons[MORPHO_ORDER.index(n)] for n in ("exocanthion_R", "endocanthion_R")], axis=0)
            eL = np.mean([cons[MORPHO_ORDER.index(n)] for n in ("exocanthion_L", "endocanthion_L")], axis=0)
            ang = np.arctan2(eL[1] - eR[1], eL[0] - eR[0])
            if np.cos(ang) < 0:
                ang += np.pi
            R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
            cons, q = cons @ R.T, q @ R.T
        except (ValueError, IndexError):
            pass
        # (b) consensus vs query with deviation vectors
        ax = axes[1]
        ax.plot(cons[:, 0], -cons[:, 1], "o", color="#457b9d", ms=4, label="reference consensus")
        ax.plot(q[:, 0], -q[:, 1], "*", color="#e63946", ms=7, label="this face")
        for c, p in zip(cons, q):
            ax.annotate("", xy=(p[0], -p[1]), xytext=(c[0], -c[1]), arrowprops=dict(arrowstyle="->", color="#e63946", lw=0.8, alpha=0.8))
        for a, b in WIRE:
            ia, ib = MORPHO_ORDER.index(a), MORPHO_ORDER.index(b)
            ax.plot([cons[ia, 0], cons[ib, 0]], [-cons[ia, 1], -cons[ib, 1]], "-", color="#457b9d", lw=0.5, alpha=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"(b) Procrustes superimposition — distance to consensus {morph['procrustes_distance_to_consensus']:.4f}", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xticks([]); ax.set_yticks([])

        # (c) PCA scatter with group ellipses
        ax = axes[2]
        scores = mr.scores[:, :2]
        groups = {}
        for i, m in enumerate(mr.metadata):
            groups.setdefault(str(m.get(group_field) or "unknown"), []).append(i)
        cmap = plt.get_cmap("tab10")
        for gi, (g, idx) in enumerate(sorted(groups.items())):
            pts = scores[idx]
            col = cmap(gi % 10)
            ax.scatter(pts[:, 0], pts[:, 1], s=14, color=col, alpha=0.75, label=f"{g} (n={len(idx)})")
            e = _ellipse_95(pts)
            if e:
                mean, wdt, hgt, ang = e
                ax.add_patch(Ellipse(mean, wdt, hgt, angle=ang, fc=col, ec=col, alpha=0.12, lw=1))
        qs = morph["pca_scores"][:2]
        if len(qs) == 2:
            ax.scatter([qs[0]], [qs[1]], marker="*", s=180, color="#e63946", ec="black", zorder=5, label="this face")
        ev = morph["pca_explained_variance"]
        ax.set_xlabel(f"PC1 ({ev[0] * 100:.1f} %)" if ev else "PC1")
        ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f} %)" if len(ev) > 1 else "PC2")
        ax.set_title(f"(c) PCA of Procrustes shape, 95 % ellipses by {group_field}", fontsize=9)
        ax.legend(fontsize=6, loc="best", ncol=2)
        ax.axhline(0, color="#999", lw=0.5); ax.axvline(0, color="#999", lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
