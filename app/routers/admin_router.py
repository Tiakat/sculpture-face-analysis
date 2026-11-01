from __future__ import annotations

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import jobs, taxonomy
from ..auth import require_admin
from ..config import ANALYSIS_DIR
from ..database import get_db
from ..models import Annotation, ArtObject, AuditLog, Collection, DuplicateGroup, Face, Image, Role, User
from ..pipeline import versions
from ..pipeline.runner import hamming
from ..schemas import AnnotationOut, CollectionIn, ReviewDecision, RoleUpdate, UserOut
from ..state import get_reference_index, reload_reference_index

router = APIRouter(prefix="/api/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/users", response_model=list[UserOut])
def users(db: Session = Depends(get_db)):
    """Users with their activity (images analysed, annotations made) — the administrator's overview."""
    out = []
    img_counts = dict(db.execute(select(Image.uploaded_by, func.count()).group_by(Image.uploaded_by)).all())
    ann_counts = dict(db.execute(select(Annotation.annotator_id, func.count()).group_by(Annotation.annotator_id)).all())
    for u in db.scalars(select(User).order_by(User.id)).all():
        d = UserOut.model_validate(u)
        d.images_count = img_counts.get(u.id, 0)
        d.annotations_count = ann_counts.get(u.id, 0)
        out.append(d)
    return out


@router.patch("/users/{user_id}", response_model=UserOut)
def update_user(user_id: int, data: RoleUpdate, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(404)
    if u.id == admin.id and data.role != "admin":
        raise HTTPException(400, "You cannot demote yourself")
    u.role = Role(data.role)
    if data.is_active is not None:
        u.is_active = data.is_active
    if data.facs_certified is not None:
        u.facs_certified = data.facs_certified
    db.add(AuditLog(user_id=admin.id, action="user_update", target=str(u.id), details=data.model_dump()))
    db.commit()
    db.refresh(u)
    return u


@router.get("/stats")
def stats(db: Session = Depends(get_db)):
    n = lambda model: db.scalar(select(func.count()).select_from(model)) or 0  # noqa: E731
    by_expr = dict(db.execute(select(Face.expression, func.count()).where(Face.expression.is_not(None)).group_by(Face.expression)).all())
    by_domain = dict(db.execute(select(ArtObject.domain, func.count()).group_by(ArtObject.domain)).all())
    by_layer = dict(db.execute(select(Annotation.layer, func.count()).group_by(Annotation.layer)).all())
    verdicts = dict(db.execute(select(Annotation.verdict, func.count()).group_by(Annotation.verdict)).all())
    total_ann = sum(verdicts.values())
    open_flags = db.scalar(select(func.count()).select_from(Annotation).where(Annotation.flag_for_review.is_(True), Annotation.review_status == "open")) or 0
    ref = get_reference_index()
    return {"users": n(User), "objects": n(ArtObject), "images": n(Image), "faces": n(Face), "annotations": total_ann,
            "annotations_by_layer": by_layer, "verdicts": verdicts, "open_flags": open_flags,
            "human_model_agreement": round(verdicts.get("agree", 0) / total_ann, 3) if total_ann else None,
            "faces_by_expression": by_expr, "objects_by_domain": by_domain,
            "training_eligible_objects": db.scalar(select(func.count()).select_from(ArtObject).where(ArtObject.training_allowed.is_(True))),
            "reference_index": {"dataset_version": ref.dataset_version, "size": ref.size}, "versions": versions.COMPONENT_VERSIONS}


@router.get("/flags")
def flags(status: str = "open", db: Session = Depends(get_db)):
    q = select(Annotation).where(Annotation.flag_for_review.is_(True))
    if status != "all":
        q = q.where(Annotation.review_status == status)
    out = []
    for a in db.scalars(q.order_by(Annotation.created_at.desc())).all():
        f = a.face
        out.append({**AnnotationOut.model_validate(a).model_dump(mode="json"), "image_id": f.image_id, "face_index": f.face_index,
                    "model_expression": f.expression, "model_confidence": f.expression_confidence,
                    "object_title": f.image.object.title, "object_id": f.image.object_id, "culture": f.image.object.culture})
    return out


@router.patch("/flags/{annotation_id}")
def review(annotation_id: int, data: ReviewDecision, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    a = db.get(Annotation, annotation_id)
    if not a:
        raise HTTPException(404)
    a.review_status = data.review_status
    a.reviewed_by = admin.id
    a.reviewed_at = datetime.now(timezone.utc)
    if data.review_status != "open":
        f = a.face
        if not any(x.flag_for_review and x.review_status == "open" for x in f.annotations if x.id != a.id):
            f.flagged = False
    db.add(AuditLog(user_id=admin.id, action="flag_review", target=str(a.id), details=data.model_dump()))
    db.commit()
    return {"ok": True}


@router.get("/audit")
def audit(limit: int = 500, db: Session = Depends(get_db)):
    rows = db.scalars(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(min(limit, 5000))).all()
    return [{"id": r.id, "created_at": r.created_at.isoformat(), "user_id": r.user_id, "action": r.action, "target": r.target,
             "details": r.details, "ip": r.ip} for r in rows]


# ------------------------------------------------------------------ collections (research subsets)
@router.get("/collections")
def collections(db: Session = Depends(get_db)):
    return [{"id": c.id, "name": c.name, "version": c.version, "description": c.description, "filter": c.filter_json,
             "n_objects": len(c.objects)} for c in db.scalars(select(Collection)).all()]


@router.post("/collections", status_code=201)
def create_collection(data: CollectionIn, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    c = Collection(name=data.name, version=data.version, description=data.description, filter_json=data.filter_json, created_by=admin.id)
    if data.filter_json:
        stmt = select(ArtObject)
        for k, v in data.filter_json.items():
            if hasattr(ArtObject, k) and v not in (None, ""):
                stmt = stmt.where(getattr(ArtObject, k) == v)
        c.objects = list(db.scalars(stmt).all())
    db.add(c)
    db.commit()
    return {"id": c.id, "name": c.name, "version": c.version, "n_objects": len(c.objects)}


@router.post("/collections/{cid}/refresh")
def refresh_collection(cid: int, db: Session = Depends(get_db)):
    c = db.get(Collection, cid)
    if not c:
        raise HTTPException(404)
    if c.filter_json:
        stmt = select(ArtObject)
        for k, v in c.filter_json.items():
            if hasattr(ArtObject, k) and v not in (None, ""):
                stmt = stmt.where(getattr(ArtObject, k) == v)
        c.objects = list(db.scalars(stmt).all())
        parts = c.version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        c.version = ".".join(parts)
    db.commit()
    return {"id": c.id, "version": c.version, "n_objects": len(c.objects)}


# ------------------------------------------------------------------ duplicates
@router.post("/duplicates/scan")
def scan_duplicates(max_hamming: int = 6, db: Session = Depends(get_db)):
    """Group objects whose images are exact or near duplicates (perceptual hash).  Groups are
    proposals for review — objects are never merged automatically."""
    imgs = [i for i in db.scalars(select(Image).where(Image.phash.is_not(None))).all()]
    parent = {i.object_id: i.object_id for i in imgs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = 0
    for a in range(len(imgs)):
        for b in range(a + 1, len(imgs)):
            if imgs[a].object_id == imgs[b].object_id:
                continue
            if imgs[a].sha256 == imgs[b].sha256 or hamming(imgs[a].phash, imgs[b].phash) <= max_hamming:
                ra, rb = find(imgs[a].object_id), find(imgs[b].object_id)
                if ra != rb:
                    parent[rb] = ra
                    pairs += 1
    groups: dict[str, list[str]] = {}
    for oid in parent:
        groups.setdefault(find(oid), []).append(oid)
    created = 0
    for root, members in groups.items():
        if len(members) < 2:
            continue
        g = DuplicateGroup(canonical_object_id=root, method=f"sha256|phash<={max_hamming}")
        db.add(g)
        db.flush()
        for oid in members:
            o = db.get(ArtObject, oid)
            o.duplicate_group_id = g.id
        created += 1
    db.commit()
    return {"pairs_linked": pairs, "groups_created": created}


# ------------------------------------------------------------------ export / splits
def _face_record(f: Face, rep_face: dict, label: str, label_source: str) -> dict:
    o = f.image.object
    return {"face_id": f.id, "image_id": f.image_id, "object_id": o.id, "group_key": o.mold_group or (f"dup{o.duplicate_group_id}" if o.duplicate_group_id else o.id),
            "institution": o.institution, "source": o.source, "domain": o.domain, "culture": o.culture, "region": o.region,
            "period": o.period, "material": o.material, "license": o.license, "training_allowed": o.training_allowed,
            "label_expression": label, "label_source": label_source, "model_expression": f.expression,
            "geometric_features": rep_face.get("facial_actions", {}).get("geometric_features"),
            "au_evidence": {a["code"]: a["evidence"] for a in rep_face.get("facial_actions", {}).get("action_units", [])},
            "difficulty_level": f.difficulty_level, "pose_quality": f.pose_quality, "quality_score": f.quality_score,
            "landmarks_file": str(ANALYSIS_DIR / f.image_id / f"face_{f.face_index}" / "landmarks_478.npy")}


@router.get("/export/gold")
def export_gold(layer: str = "all", db: Session = Depends(get_db)):
    """Gold-standard / expert labels as JSONL — the ONLY supervision used for training.
    Museum descriptions are never exported as labels.  Only training-eligible objects are included."""
    lines = []
    for a in db.scalars(select(Annotation)).all():
        if a.verdict == "uncertain" or (a.verdict == "disagree" and a.review_status != "accepted"):
            continue
        if layer != "all" and a.layer != layer:
            continue
        f = a.face
        if not f.image.object.training_allowed:
            continue
        rep_path = ANALYSIS_DIR / f.image_id / "analysis.json"
        rep = json.loads(rep_path.read_text(encoding="utf-8")) if rep_path.exists() else {}
        rep_face = next((x for x in rep.get("faces", []) if x.get("face_index") == f.face_index), {})
        label = a.expression if a.verdict == "disagree" else f.expression
        rec = _face_record(f, rep_face, label, f"{a.layer}:{a.annotator_id}")
        rec.update({"annotation_confidence": a.confidence, "soft_labels": a.expression_distribution, "aus_human": a.aus,
                    "representation_type": a.representation_type or f.image.object.representation_type})
        lines.append(json.dumps(rec))
    return Response("\n".join(lines), media_type="application/x-ndjson",
                    headers={"Content-Disposition": 'attachment; filename="gold_standard.jsonl"'})


@router.get("/splits/plan")
def split_plan(level: str = "object", test_frac: float = 0.2, seed: int = 42, db: Session = Depends(get_db)):
    """Preview a leakage-safe split at object / institution / culture / region level."""
    import numpy as np
    objs = db.scalars(select(ArtObject).where(ArtObject.training_allowed.is_(True))).all()
    key = {"object": lambda o: o.mold_group or (f"dup{o.duplicate_group_id}" if o.duplicate_group_id else o.id),
           "institution": lambda o: o.institution or o.source, "culture": lambda o: o.culture or "unknown",
           "region": lambda o: o.region or "unknown"}.get(level)
    if key is None:
        raise HTTPException(400, "level must be object|institution|culture|region")
    groups = sorted({key(o) for o in objs})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_frac))) if groups else 0
    test_groups = set(groups[:n_test])
    return {"level": level, "n_groups": len(groups), "test_groups": sorted(test_groups),
            "n_objects_train": sum(1 for o in objs if key(o) not in test_groups),
            "n_objects_test": sum(1 for o in objs if key(o) in test_groups),
            "note": "External test = hold out an entire institution or culture that the model never sees."}


@router.post("/references/reload")
def reload_ref():
    ref = reload_reference_index()
    return {"dataset_version": ref.dataset_version, "size": ref.size}


@router.post("/taxonomy/reload")
def reload_taxonomy():
    t = taxonomy.reload()
    return {"version": t["_meta"]["version"], "fields": sorted(taxonomy.CONTROLLED_FIELDS)}


# ------------------------------------------------------------------ corpus building (from inside the app)
from pydantic import BaseModel as _BM  # noqa: E402


class ApiKeysIn(_BM):
    HARVARD_API_KEY: str | None = None
    SMITHSONIAN_API_KEY: str | None = None
    EUROPEANA_API_KEY: str | None = None


@router.get("/corpus/keys")
def corpus_keys(admin: User = Depends(require_admin)):
    """Which museum API keys are saved (values are never returned)."""
    saved = jobs.load_keys()
    return {k: (k in saved) for k in jobs.KEY_NAMES}


@router.post("/corpus/keys")
def corpus_keys_save(data: ApiKeysIn, admin: User = Depends(require_admin)):
    jobs.save_keys({k: v for k, v in data.model_dump().items() if v is not None})
    saved = jobs.load_keys()
    return {k: (k in saved) for k in jobs.KEY_NAMES}


class CorpusBuildIn(_BM):
    sources: list[str] = ["met", "artic", "cleveland", "wikimedia"]
    queries: list[str] | None = None
    limit: int = 300
    analyse_into_db: bool = True


@router.post("/corpus/build")
def corpus_build(data: CorpusBuildIn, admin: User = Depends(require_admin)):
    """Download open-access images + metadata from the selected museum databases, build the reference
    index (retrieval + morphometrics) and load the objects into the platform. Runs in the background."""
    py = jobs.python()
    ingest = [py, "scripts/ingest.py", *data.sources, "--limit", str(data.limit)]
    for q in (data.queries or []):
        ingest += ["--query", q]
    steps = [("Downloading from museum databases", ingest),
             ("Building reference index (landmarks, embeddings, Procrustes)", [py, "scripts/build_reference_index.py"])]
    if data.analyse_into_db:
        steps.append(("Loading objects into the platform database", [py, "scripts/load_corpus_to_db.py"]))
    started = jobs.start("corpus_build", steps, on_done=lambda ok: reload_reference_index())
    if not started:
        raise HTTPException(409, "A job is already running")
    db_note = "Objects will appear in the Objects page." if data.analyse_into_db else ""
    return {"started": True, "steps": [s[0] for s in steps], "note": db_note}


@router.get("/corpus/status")
def corpus_status(admin: User = Depends(require_admin)):
    st = jobs.status()
    ref = get_reference_index()
    st["reference_index"] = {"dataset_version": ref.dataset_version, "size": ref.size}
    return st


@router.post("/users/{user_id}/reset-link")
def admin_reset_link(user_id: int, request: Request, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    """Create a password-reset link for a user (to hand over when e-mail is not configured)."""
    import secrets
    from datetime import timedelta
    from .. import mailer
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(404)
    u.reset_token = secrets.token_urlsafe(32)
    u.reset_expires = datetime.now(timezone.utc) + timedelta(hours=24)
    db.add(AuditLog(user_id=admin.id, action="admin_reset_link", target=str(u.id)))
    db.commit()
    return {"link": f"{mailer.public_url(request)}/?reset={u.reset_token}", "expires_hours": 24}


@router.post("/users/{user_id}/verify")
def admin_verify(user_id: int, db: Session = Depends(get_db), admin: User = Depends(require_admin)):
    u = db.get(User, user_id)
    if not u:
        raise HTTPException(404)
    u.email_verified = True
    u.verify_token = None
    db.add(AuditLog(user_id=admin.id, action="admin_verify_user", target=str(u.id)))
    db.commit()
    return {"ok": True}


@router.get("/mail-status")
def mail_status(admin: User = Depends(require_admin)):
    from .. import mailer
    return {"configured": mailer.configured(), "public_url": mailer.public_url()}
