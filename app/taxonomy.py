"""Controlled taxonomy: load, validate, and expose the vocabularies.

Design rule (the "metadata firewall"): nothing in app/pipeline imports this
module. Culture, region, period, material etc. are stored WITH an object but
never enter the face-analysis model as features, so the model cannot learn
"Moche → sadness". They are used afterwards only for grouping, filtering and
for computing morphological affinity *within* the reference sample.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException

TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "config" / "taxonomy.json"

CONTROLLED_FIELDS = {
    "domain", "region", "culture", "period", "material", "technique", "object_type", "subject_type",
    "representation_type", "sex_gender_depicted", "age_depicted", "face_completeness", "restoration_status",
    "polychromy", "view_angle", "license", "cultural_sensitivity", "date_precision", "annotation_layer",
}


@lru_cache(maxsize=1)
def load() -> dict:
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def reload():
    load.cache_clear()
    return load()


def ids(field: str) -> list[str]:
    vals = load().get(field, [])
    return [v["id"] if isinstance(v, dict) else v for v in vals]


def term(field: str, term_id: str) -> dict | None:
    for v in load().get(field, []):
        if (v["id"] if isinstance(v, dict) else v) == term_id:
            return v if isinstance(v, dict) else {"id": v, "label": v}
    return None


def label(field: str, term_id: str | None) -> str | None:
    if term_id is None:
        return None
    t = term(field, term_id)
    return t.get("label", term_id) if t else term_id


def validate_metadata(meta: dict) -> dict:
    """Validate controlled fields; raise 422 with all problems at once."""
    problems = []
    for field in CONTROLLED_FIELDS & set(meta.keys()):
        val = meta[field]
        if val in (None, ""):
            continue
        vals = val if isinstance(val, list) else [val]
        allowed = set(ids(field))
        for v in vals:
            if v not in allowed:
                problems.append(f"{field}: '{v}' is not in the controlled vocabulary")
    # date consistency
    ds, de = meta.get("date_start"), meta.get("date_end")
    if ds is not None and de is not None and ds > de:
        problems.append("date_start must be <= date_end")
    for k in ("date_confidence",):
        if meta.get(k) is not None and not (0 <= float(meta[k]) <= 1):
            problems.append(f"{k} must be in [0,1]")
    if problems:
        raise HTTPException(422, {"taxonomy_errors": problems})
    # derive region from culture when missing
    if meta.get("culture") and not meta.get("region"):
        t = term("culture", meta["culture"])
        if t and t.get("region"):
            meta["region"] = t["region"]
    return meta


def region_path(region_id: str | None) -> list[str]:
    """['south_america', 'andes_north_coast'] — hierarchy from root to leaf."""
    path = []
    cur = region_id
    seen = set()
    while cur and cur not in seen:
        seen.add(cur)
        path.append(cur)
        t = term("region", cur)
        cur = t.get("parent") if t else None
    return list(reversed(path))


def license_policy(license_id: str | None) -> dict:
    t = term("license", license_id or "UNKNOWN") or term("license", "UNKNOWN")
    return {"license": t["id"], "training_allowed": t.get("training_allowed"), "derivatives": t.get("derivatives"),
            "attribution_required": t.get("attribution")}


def training_eligible(meta: dict) -> tuple[bool, str]:
    pol = license_policy(meta.get("license"))
    if pol["training_allowed"] is not True and pol["training_allowed"] != "non-commercial research only" \
            and pol["training_allowed"] != "per agreement":
        return False, f"licence {pol['license']} does not permit model training"
    sens = meta.get("cultural_sensitivity") or []
    sens = sens if isinstance(sens, list) else [sens]
    blocked = {"human_remains", "sacred_object", "community_protocol_applies"}
    if blocked & set(sens):
        return False, "cultural-sensitivity flag requires explicit ethics approval before training use"
    if meta.get("domain") in ("human_remains", "reconstruction", "synthetic"):
        return False, f"domain '{meta.get('domain')}' is excluded from the depicted-face training set by default"
    return True, "eligible"
