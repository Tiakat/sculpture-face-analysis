"""Read-only access to the reference index (MocheFaceDB) built by
scripts/build_reference_index.py.

Files in data/reference_index/:
  index.json        dataset_version, build date, list of records (metadata)
  embeddings.npy    (n, d) float32
  landmarks.npy     (n, k, 2) anthropometric subset in aligned coordinates
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .embedding import cosine_similarity
from .morphometrics import MorphometricReference

log = logging.getLogger(__name__)


class ReferenceIndex:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.records: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.landmarks: np.ndarray | None = None
        self.dataset_version = "GlobalFaceCorpus (no reference index built)"
        self.morpho: MorphometricReference | None = None
        self.built_at: str | None = None
        self.load()

    def load(self):
        meta = self.dir / "index.json"
        if not meta.exists():
            log.warning("No reference index at %s", self.dir)
            self.morpho = MorphometricReference(np.zeros((0, 33, 2)), [])
            return
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            self.records = data["records"]
        except Exception as e:  # corrupt / half-written index must never prevent the app from starting
            log.error("Reference index at %s is unreadable (%s); starting with an empty index. Rebuild it from the admin page.", self.dir, e)
            self.records = []
            self.morpho = MorphometricReference(np.zeros((0, 33, 2)), [])
            self.dataset_version = "reference index unreadable — rebuild"
            return
        self.dataset_version = data.get("dataset_version", "unknown")
        self.built_at = data.get("built_at")
        self.embeddings = np.load(self.dir / "embeddings.npy")
        self.landmarks = np.load(self.dir / "landmarks.npy")
        self.morpho = MorphometricReference(self.landmarks, self.records)
        log.info("Loaded reference index %s with %d records", self.dataset_version, len(self.records))

    @property
    def size(self) -> int:
        return len(self.records)

    def retrieve(self, embedding: np.ndarray, k: int = 10, expression: str | None = None) -> list[dict]:
        if self.embeddings is None or len(self.records) == 0:
            return []
        sims = cosine_similarity(embedding, self.embeddings)
        order = np.argsort(-sims)
        out = []
        for i in order:
            rec = self.records[i]
            same_expr = expression is not None and rec.get("expression") == expression
            out.append({
                "reference_id": rec.get("id"),
                "similarity": round(float(sims[i]), 4),
                "same_expression_label": same_expr,
                "image": rec.get("image_path"),
                "thumbnail": rec.get("thumbnail_path"),
                "title": rec.get("title"),
                "museum": rec.get("museum"),
                "object_number": rec.get("object_number"),
                "culture": rec.get("culture"),
                "date": rec.get("date"),
                "domain": rec.get("domain"), "region": rec.get("region"), "period": rec.get("period"),
                "material": rec.get("material"), "object_type": rec.get("object_type"),
                "culture_subgroup": rec.get("culture_subgroup"), "provenance": rec.get("provenance"),
                "object_id": rec.get("object_id"),
                "expression": rec.get("expression"),
                "expression_source": rec.get("expression_source"),
                "au_profile": rec.get("au_profile"),
                "license": rec.get("license"),
                "source_url": rec.get("source_url"),
            })
            if len(out) >= k:
                break
        return out

    def retrieve_by_expression(self, embedding: np.ndarray, expression: str, k: int = 5) -> list[dict]:
        """Closest references that carry the same expression label."""
        if self.embeddings is None:
            return []
        idx = [i for i, r in enumerate(self.records) if r.get("expression") == expression]
        if not idx:
            return []
        sims = cosine_similarity(embedding, self.embeddings[idx])
        order = np.argsort(-sims)[:k]
        res = []
        for j in order:
            i = idx[j]
            rec = self.records[i]
            res.append({"reference_id": rec.get("id"), "similarity": round(float(sims[j]), 4),
                        "image": rec.get("image_path"), "thumbnail": rec.get("thumbnail_path"),
                        "title": rec.get("title"), "museum": rec.get("museum"),
                        "object_number": rec.get("object_number"), "culture": rec.get("culture"),
                        "date": rec.get("date"), "region": rec.get("region"), "period": rec.get("period"), "domain": rec.get("domain"),
                        "expression": rec.get("expression"), "expression_source": rec.get("expression_source"),
                        "license": rec.get("license"), "source_url": rec.get("source_url")})
        return res

    def retrieve_by_field(self, embedding: np.ndarray, field: str, value: str, k: int = 6) -> list[dict]:
        """Closest references sharing one metadata value (e.g. region == 'andes_north_coast')."""
        if self.embeddings is None:
            return []
        idx = [i for i, r in enumerate(self.records) if str(r.get(field)) == str(value)]
        if not idx:
            return []
        sims = cosine_similarity(embedding, self.embeddings[idx])
        order = np.argsort(-sims)[:k]
        out = []
        for j in order:
            rec = self.records[idx[j]]
            out.append({"reference_id": rec.get("id"), "similarity": round(float(sims[j]), 4), "title": rec.get("title"),
                        "museum": rec.get("museum") or rec.get("institution"), "culture": rec.get("culture"), "region": rec.get("region"),
                        "period": rec.get("period"), "material": rec.get("material"), "domain": rec.get("domain"), "date": rec.get("date") or rec.get("date_display"),
                        "expression": rec.get("expression"), "expression_source": rec.get("expression_source"),
                        "license": rec.get("license"), "source_url": rec.get("source_url")})
        return out


def suggest_attribution(nearest: list[dict], affinity: dict, fields=("culture", "region", "period", "material", "domain", "object_type")) -> dict:
    """Estimate metadata for an object WITHOUT any metadata input: similarity-weighted vote over the nearest
    reference specimens (appearance + shape embedding), combined with the morphometric group affinity.

    Output is a *suggestion* with a confidence and the supporting specimens; it is never a determination
    and is applied to the object record only when the researcher accepts it."""
    out = {}
    for fld in fields:
        votes: dict[str, float] = {}
        support: dict[str, list] = {}
        for r in nearest:
            v = r.get(fld)
            if not v:
                continue
            w = max(0.0, float(r.get("similarity", 0)))
            votes[v] = votes.get(v, 0.0) + w
            support.setdefault(v, []).append(r.get("reference_id"))
        aff = (affinity or {}).get(fld, {})
        for row in aff.get("ranking", []):
            votes[row["group"]] = votes.get(row["group"], 0.0) + 0.5 * float(row["affinity"]) * (sum(votes.values()) or 1.0)
        if not votes:
            continue
        total = sum(votes.values())
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        top, score = ranked[0]
        conf = score / total
        margin = conf - (ranked[1][1] / total if len(ranked) > 1 else 0.0)
        out[fld] = {"value": top, "confidence": round(conf, 3), "margin": round(margin, 3),
                    "reliability": "low" if margin < 0.15 or len(nearest) < 5 else ("moderate" if margin < 0.35 else "higher"),
                    "alternatives": [{"value": v, "share": round(s / total, 3)} for v, s in ranked[1:4]],
                    "supporting_specimens": support.get(top, [])[:6]}
    return out
