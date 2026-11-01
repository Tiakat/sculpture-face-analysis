"""Geometric morphometrics: Generalised Procrustes Analysis, PCA, clustering
and morphological-affinity estimation (GPA/PCA following Rohlf & Slice 1990; grouping
logic generalised from Millones' Moche portrait-vessel study to any taxonomy field).

Wording: results are *morphological affinity to reference samples*, never
"geographic origin of the person".
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA

from .landmarks import MORPHO_ORDER


# ---------------------------------------------------------------- Procrustes
def _center_scale(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0)
    return Xc / (np.linalg.norm(Xc) + 1e-12)


def _align_to(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rotate pre-shape X onto ref (orthogonal Procrustes, no reflection)."""
    U, _, Vt = np.linalg.svd(X.T @ ref)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # forbid reflection
        U[:, -1] *= -1
        R = U @ Vt
    return X @ R


def gpa(configs: np.ndarray, iters: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Generalised Procrustes Analysis.
    configs: (n, k, 2). Returns (aligned (n,k,2), consensus (k,2))."""
    X = np.stack([_center_scale(c) for c in configs])
    consensus = X[0].copy()
    for _ in range(iters):
        X = np.stack([_align_to(x, consensus) for x in X])
        new_c = _center_scale(X.mean(axis=0))
        if np.linalg.norm(new_c - consensus) < 1e-9:
            consensus = new_c
            break
        consensus = new_c
    return X, consensus


def procrustes_distance(a: np.ndarray, b: np.ndarray) -> float:
    A, B = _center_scale(a), _center_scale(b)
    A = _align_to(A, B)
    return float(np.sqrt(((A - B) ** 2).sum()))


def superimpose_query(query: np.ndarray, consensus: np.ndarray) -> np.ndarray:
    return _align_to(_center_scale(query), consensus)


# ---------------------------------------------------------------- Results
@dataclass
class MorphometricResult:
    landmark_names: list[str]
    procrustes_coordinates: list[list[float]]
    procrustes_distance_to_consensus: float
    reference_sample_size: int
    pca_scores: list[float]
    pca_explained_variance: list[float]
    pca_position_percentiles: list[float]
    cluster_id: int | None
    cluster_sizes: dict[str, int]
    nearest_reference_ids: list[dict]
    group_affinity: dict[str, dict]      # e.g. by valley / phase
    interpretation: str
    caveats: list[str] = field(default_factory=list)
    linear_measurements: dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def linear_measurements(lm2d: np.ndarray) -> dict[str, float]:
    """Classical facial indices from the anthropometric subset (scale-free)."""
    P = {n: lm2d[i] for i, n in enumerate(MORPHO_ORDER)}

    def d(a, b):
        return float(np.linalg.norm(P[a] - P[b]))

    face_w = d("zygion_R", "zygion_L")
    face_h = d("nasion", "gnathion")
    jaw_w = d("gonion_R", "gonion_L")
    m = {
        "facial_index_(nasion-gnathion/bizygomatic)": face_h / (face_w + 1e-9),
        "jaw_to_face_width": jaw_w / (face_w + 1e-9),
        "intercanthal_to_bizygomatic": d("endocanthion_R", "endocanthion_L") / (face_w + 1e-9),
        "eye_fissure_R_over_face_w": d("endocanthion_R", "exocanthion_R") / (face_w + 1e-9),
        "nasal_index_(alar_w/nasal_h)": d("alare_R", "alare_L") / (d("nasion", "subnasale") + 1e-9),
        "mouth_width_over_face_w": d("cheilion_R", "cheilion_L") / (face_w + 1e-9),
        "lower_face_over_face_h": d("subnasale", "gnathion") / (face_h + 1e-9),
        "upper_face_over_face_h": d("nasion", "subnasale") / (face_h + 1e-9),
        "forehead_over_face_h": d("trichion_approx", "nasion") / (face_h + 1e-9),
    }
    return {k: round(v, 4) for k, v in m.items()}


class MorphometricReference:
    """Reference sample of Procrustes-aligned configurations with metadata.

    Built by scripts/build_reference_index.py; loaded read-only at run time.
    """

    def __init__(self, configs: np.ndarray, metadata: list[dict], n_components: int = 8, n_clusters: int = 4):
        self.metadata = metadata
        self.n = len(metadata)
        if self.n >= 3:
            self.aligned, self.consensus = gpa(configs)
            flat = self.aligned.reshape(self.n, -1)
            self.flat_mean = flat.mean(axis=0)
            k = min(n_components, self.n - 1, flat.shape[1])
            self.pca = PCA(n_components=k).fit(flat)
            self.scores = self.pca.transform(flat)
            self.linkage = linkage(self.scores, method="ward")
            self.labels = fcluster(self.linkage, t=min(n_clusters, self.n), criterion="maxclust")
        else:
            self.aligned = None
            self.consensus = _center_scale(configs[0]) if len(configs) else None
            self.pca = None
            self.scores = None
            self.labels = None

    # ------------------------------------------------------------------
    def analyse(self, query_lm2d: np.ndarray, group_fields=("culture", "region", "period", "material")) -> MorphometricResult:
        caveats = [
            "Morphological affinity to reference samples is NOT a determination of geographic or biological origin.",
            "Artworks may be stylised, mould-made, workshop products, or depict social types rather than individuals; "
            "affinity is computed only within the current reference sample and its group labels.",
        ]
        meas = linear_measurements(query_lm2d)
        if self.pca is None:
            return MorphometricResult(
                landmark_names=MORPHO_ORDER,
                procrustes_coordinates=_center_scale(query_lm2d).round(5).tolist(),
                procrustes_distance_to_consensus=0.0,
                reference_sample_size=self.n,
                pca_scores=[], pca_explained_variance=[], pca_position_percentiles=[],
                cluster_id=None, cluster_sizes={}, nearest_reference_ids=[], group_affinity={},
                interpretation="Reference index too small (<3 specimens): only shape coordinates and linear "
                               "indices are reported. Build MocheFaceDB with scripts/build_reference_index.py.",
                caveats=caveats, linear_measurements=meas,
            )

        q = superimpose_query(query_lm2d, self.consensus)
        d_cons = float(np.sqrt(((q - self.consensus) ** 2).sum()))
        qflat = q.reshape(1, -1)
        qs = self.pca.transform(qflat)[0]
        pct = [float((self.scores[:, i] <= qs[i]).mean() * 100) for i in range(len(qs))]

        # Nearest references in Procrustes shape space
        dists = np.sqrt(((self.aligned - q[None]) ** 2).sum(axis=(1, 2)))
        order = np.argsort(dists)[:10]
        nearest = [{"reference_id": self.metadata[i].get("id"), "procrustes_distance": round(float(dists[i]), 4),
                    **{k: self.metadata[i].get(k) for k in ("museum", "culture", "region", "period", "material", "domain", "title")}}
                   for i in order]

        # Cluster membership: nearest centroid in PCA space
        centroids = {c: self.scores[self.labels == c].mean(axis=0) for c in np.unique(self.labels)}
        cid = int(min(centroids, key=lambda c: np.linalg.norm(centroids[c] - qs)))
        sizes = {str(int(c)): int((self.labels == c).sum()) for c in np.unique(self.labels)}

        # Group affinity with uncertainty (per metadata field)
        affinity = {}
        for fld in group_fields:
            groups = {}
            for i, m in enumerate(self.metadata):
                g = m.get(fld)
                if g:
                    groups.setdefault(str(g), []).append(i)
            groups = {g: idx for g, idx in groups.items() if len(idx) >= 2}
            if len(groups) < 2:
                continue
            entries = {}
            raw = {}
            for g, idx in groups.items():
                gd = dists[idx]
                # mean distance to group members + distance to group mean shape
                gmean = self.aligned[idx].mean(axis=0)
                d_mean = float(np.sqrt(((gmean - q) ** 2).sum()))
                raw[g] = d_mean
                entries[g] = {"n": len(idx), "distance_to_group_mean": round(d_mean, 4),
                              "mean_pairwise_distance": round(float(gd.mean()), 4),
                              "min_distance": round(float(gd.min()), 4)}
            # Convert distances to affinity scores via softmax on -d / scale
            ds = np.array(list(raw.values()))
            scale = ds.std() + 1e-6
            w = np.exp(-(ds - ds.min()) / scale)
            w = w / w.sum()
            for (g, e), wi in zip(entries.items(), w):
                e["affinity"] = round(float(wi), 3)
            ranked = sorted(entries.items(), key=lambda kv: -kv[1]["affinity"])
            top, second = ranked[0], ranked[1]
            margin = top[1]["affinity"] - second[1]["affinity"]
            affinity[fld] = {
                "ranking": [{"group": g, **e} for g, e in ranked],
                "most_similar": top[0],
                "margin_over_second": round(float(margin), 3),
                "reliability": "low" if margin < 0.1 else ("moderate" if margin < 0.25 else "higher"),
            }

        if affinity:
            first = next(iter(affinity.values()))
            interp = (f"The facial morphology is most similar to the reference sample labelled "
                      f"'{first['most_similar']}' (reliability: {first['reliability']}). "
                      "This is a morphological similarity estimate relative to the current reference "
                      "index, not a determination of origin.")
        else:
            interp = ("Reference index lacks group metadata with >=2 specimens per group; "
                      "only shape-space position, nearest specimens and clustering are reported.")
        if self.n < 30:
            caveats.append(f"Small reference sample (n={self.n}); PCA axes and clusters are unstable.")

        return MorphometricResult(
            landmark_names=MORPHO_ORDER,
            procrustes_coordinates=q.round(5).tolist(),
            procrustes_distance_to_consensus=round(d_cons, 4),
            reference_sample_size=self.n,
            pca_scores=[round(float(v), 5) for v in qs],
            pca_explained_variance=[round(float(v), 4) for v in self.pca.explained_variance_ratio_],
            pca_position_percentiles=[round(p, 1) for p in pct],
            cluster_id=cid, cluster_sizes=sizes, nearest_reference_ids=nearest,
            group_affinity=affinity, interpretation=interp, caveats=caveats, linear_measurements=meas,
        )
