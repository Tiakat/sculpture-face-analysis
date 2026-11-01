"""Objects → images → faces.  Upload = create/select an OBJECT, attach an IMAGE,
run the analysis, persist FACE instances.  Taxonomy metadata is validated and
stored on the object; the pipeline receives only {domain, depicted}."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import taxonomy
from ..auth import get_current_user
from ..config import ALLOWED_EXT, ANALYSIS_DIR, MAX_UPLOAD_MB, RUN_STABILITY, UPLOAD_DIR
from ..database import get_db
from ..models import Annotation, ArtObject, AuditLog, Face, Image, Role, User
from ..pipeline import versions
from ..pipeline.report import to_csv, to_pdf
from ..pipeline.runner import run_image_analysis, hamming
from ..schemas import AnnotationIn, AnnotationOut, FaceOut, ImageMetadata, ImageOut, ObjectMetadata, ObjectOut
from ..state import get_reference_index

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["objects"])


def _audit(db, user_id, action, request, target=None, details=None):
    db.add(AuditLog(user_id=user_id, action=action, target=target, details=details,
                    ip=request.client.host if request and request.client else None))


def _object_out(o: ArtObject) -> ObjectOut:
    d = {c: getattr(o, c) for c in ObjectMetadata.model_fields}
    d.update(id=o.id, created_at=o.created_at, training_allowed=o.training_allowed, annotation_layer=o.annotation_layer,
             image_count=len(o.images), face_count=sum(len(i.faces) for i in o.images), duplicate_group_id=o.duplicate_group_id)
    return ObjectOut(**d)


def _can_edit(o: ArtObject, user: User) -> bool:
    return user.role == Role.admin or o.created_by == user.id


def _can_view(o: ArtObject, user: User) -> bool:
    """Researchers see only their own objects; administrators see everything."""
    return user.role == Role.admin or o.created_by == user.id


def _apply_metadata(o: ArtObject, meta: dict):
    for k, v in meta.items():
        if hasattr(o, k):
            setattr(o, k, v)
    ok, _ = taxonomy.training_eligible(meta if meta.get("license") else {**meta, "license": o.license})
    o.training_allowed = ok
    o.metadata_version = (o.metadata_version or 1) + 1


# ------------------------------------------------------------------ objects
@router.post("/objects", response_model=ObjectOut, status_code=201)
def create_object(data: ObjectMetadata, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    meta = taxonomy.validate_metadata(data.model_dump(exclude_none=True))
    o = ArtObject(created_by=user.id)
    _apply_metadata(o, meta)
    db.add(o)
    db.flush()
    _audit(db, user.id, "object_create", request, o.id, {"source": o.source, "domain": o.domain})
    db.commit()
    db.refresh(o)
    return _object_out(o)


@router.get("/objects", response_model=list[ObjectOut])
def list_objects(q: str | None = None, culture: str | None = None, region: str | None = None, domain: str | None = None,
                 period: str | None = None, license: str | None = None, mine: bool = False, limit: int = 200, offset: int = 0,
                 user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    stmt = select(ArtObject).order_by(ArtObject.created_at.desc())
    if mine or user.role != Role.admin:
        stmt = stmt.where(ArtObject.created_by == user.id)
    for col, val in (("culture", culture), ("region", region), ("domain", domain), ("period", period), ("license", license)):
        if val:
            stmt = stmt.where(getattr(ArtObject, col) == val)
    if q:
        like = f"%{q}%"
        stmt = stmt.where((ArtObject.title.ilike(like)) | (ArtObject.object_number.ilike(like)) | (ArtObject.institution.ilike(like)))
    return [_object_out(o) for o in db.scalars(stmt.offset(offset).limit(min(limit, 1000))).all()]


@router.get("/objects/{object_id}")
def get_object(object_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    o = db.get(ArtObject, object_id)
    if not o or not _can_view(o, user):
        raise HTTPException(404)
    return {"object": _object_out(o).model_dump(mode="json"),
            "labels": {k: taxonomy.label(k, getattr(o, k)) for k in ("domain", "region", "culture", "period", "material", "object_type", "license")},
            "region_path": taxonomy.region_path(o.region),
            "images": [ImageOut.model_validate(i).model_dump(mode="json") for i in o.images],
            "faces": [FaceOut.model_validate(f).model_dump(mode="json") for i in o.images for f in i.faces],
            "collections": [{"id": c.id, "name": c.name, "version": c.version} for c in o.collections]}


@router.patch("/objects/{object_id}", response_model=ObjectOut)
def update_object(object_id: str, data: ObjectMetadata, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    o = db.get(ArtObject, object_id)
    if not o:
        raise HTTPException(404)
    if not _can_edit(o, user):
        raise HTTPException(403)
    meta = taxonomy.validate_metadata(data.model_dump(exclude_unset=True))
    _apply_metadata(o, meta)
    _audit(db, user.id, "object_update", request, o.id, meta)
    db.commit()
    db.refresh(o)
    return _object_out(o)


@router.delete("/objects/{object_id}", status_code=204)
def delete_object(object_id: str, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    o = db.get(ArtObject, object_id)
    if not o:
        raise HTTPException(404)
    if not _can_edit(o, user):
        raise HTTPException(403)
    for im in o.images:
        shutil.rmtree(ANALYSIS_DIR / im.id, ignore_errors=True)
        shutil.rmtree(UPLOAD_DIR / im.id, ignore_errors=True)
    _audit(db, user.id, "object_delete", request, o.id)
    db.delete(o)
    db.commit()
    return Response(status_code=204)


# ------------------------------------------------------------------ images / analysis
def _parse_face_box(raw: str) -> list[int] | None:
    """'x0,y0,x1,y1' in original pixels (drawn by the user when no face is found automatically)."""
    if not raw or not raw.strip():
        return None
    try:
        vals = [int(float(v)) for v in raw.replace(";", ",").split(",")]
    except ValueError:
        raise HTTPException(422, "face_box must be 'x0,y0,x1,y1'")
    if len(vals) != 4 or vals[2] - vals[0] < 8 or vals[3] - vals[1] < 8:
        raise HTTPException(422, "face_box must be 'x0,y0,x1,y1' with a positive size")
    return vals


@router.post("/objects/{object_id}/images", status_code=201)
async def upload_image(object_id: str, request: Request, file: UploadFile = File(...), image_metadata: str = Form("{}"),
                       face_box: str = Form(""),
                       user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    o = db.get(ArtObject, object_id)
    if not o or not _can_view(o, user):
        raise HTTPException(404, "Object not found")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported file type {ext!r}")
    fb = _parse_face_box(face_box)
    try:
        imeta = ImageMetadata.model_validate(json.loads(image_metadata or "{}")).model_dump(exclude_none=True)
        if imeta.get("view_angle"):
            taxonomy.validate_metadata({"view_angle": imeta["view_angle"]})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(422, f"Invalid image metadata: {e}")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"File exceeds {MAX_UPLOAD_MB} MB")
    sha = hashlib.sha256(data).hexdigest()
    dup = db.scalar(select(Image).where(Image.sha256 == sha))

    im = Image(object_id=o.id, uploaded_by=user.id, original_filename=Path(file.filename).name, stored_path="", sha256=sha, **imeta)
    db.add(im)
    db.flush()
    orig_dir = UPLOAD_DIR / im.id
    orig_dir.mkdir(parents=True, exist_ok=True)
    orig = orig_dir / f"original{ext}"
    orig.write_bytes(data)
    im.stored_path = str(orig)

    out_dir = ANALYSIS_DIR / im.id
    domain_term = taxonomy.term("domain", o.domain) or {}
    ctx = {"domain": o.domain, "depicted": bool(domain_term.get("depicted", True))}
    if fb:
        ctx["face_box"] = fb      # geometric guidance drawn by the user; not metadata
    try:
        report = run_image_analysis(orig, out_dir, get_reference_index(), ctx, run_stability=RUN_STABILITY)
        im.status = report["status"]
        im.quality_score = report["image_quality"]["score"]
        im.width, im.height = report["input"]["working_size"]
        im.phash = report["input"]["phash"]
        im.face_count = report.get("face_count", 0)
        im.error = report.get("rejection_reason")
        im.model_version = versions.COMPONENT_VERSIONS["expression_model"]
        im.dataset_version = report["versions"]["dataset"]
        for f in report.get("faces", []):
            face = Face(image_id=im.id, face_index=f["face_index"], bbox=list(f["box"]) if f.get("box") else None,
                        detection_confidence=f.get("detection_confidence"), quality_score=f["quality"]["score"],
                        status=f.get("status", "rejected"))
            if f.get("status") == "completed":
                face.pose_quality = f["pose"]["pose_quality"]
                face.yaw = f["pose"]["yaw_deg"]
                face.expression = f["expression"]["primary"]
                face.expression_confidence = f["expression"]["confidence"]
                face.difficulty_level = f["difficulty_level"]
            db.add(face)
    except Exception as e:
        log.exception("analysis failed")
        im.status, im.error = "error", str(e)
        report = {"status": "error", "error": str(e)}
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "analysis.json").write_text(json.dumps(report), encoding="utf-8")

    # near-duplicate hint (perceptual hash) — flagged, never auto-merged
    near = []
    if im.phash:
        for other in db.scalars(select(Image).where(Image.phash.is_not(None), Image.id != im.id)).all():
            if hamming(other.phash, im.phash) <= 6:
                near.append({"image_id": other.id, "object_id": other.object_id, "hamming": hamming(other.phash, im.phash)})
    _audit(db, user.id, "image_upload_analyse", request, im.id, {"object": o.id, "status": im.status, "sha256": sha,
                                                                  "exact_duplicate_of": dup.id if dup else None})
    db.commit()
    db.refresh(im)
    report["analysis_id"] = im.id
    report["object_metadata"] = _object_out(o).model_dump(mode="json")   # attached AFTER analysis (firewall)
    return {"image": ImageOut.model_validate(im).model_dump(mode="json"), "report": report,
            "faces": [FaceOut.model_validate(f).model_dump(mode="json") for f in im.faces],
            "duplicates": {"exact": dup.id if dup else None, "near": near}}


@router.post("/analyse", status_code=201)
async def quick_analyse(request: Request, file: UploadFile = File(...), metadata: str = Form("{}"), image_metadata: str = Form("{}"),
                        face_box: str = Form(""),
                        user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """One-shot: create the object from metadata and analyse the image."""
    try:
        meta = ObjectMetadata.model_validate(json.loads(metadata or "{}"))
    except Exception as e:
        raise HTTPException(422, f"Invalid metadata JSON: {e}")
    o = create_object(meta, request, user, db)
    return await upload_image(o.id, request, file, image_metadata, face_box, user, db)


def _load_report(im: Image) -> dict:
    p = ANALYSIS_DIR / im.id / "analysis.json"
    if not p.exists():
        raise HTTPException(404, "Report missing")
    rep = json.loads(p.read_text(encoding="utf-8"))
    rep["analysis_id"] = im.id
    rep["object_metadata"] = _object_out(im.object).model_dump(mode="json")
    return rep


@router.get("/images", response_model=list[ImageOut])
def list_images(mine: bool = False, limit: int = 300, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    stmt = select(Image).order_by(Image.created_at.desc()).limit(min(limit, 2000))
    if mine or user.role != Role.admin:
        stmt = stmt.where(Image.uploaded_by == user.id)
    return [ImageOut.model_validate(i) for i in db.scalars(stmt).all()]


@router.get("/images/{image_id}")
def get_image(image_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    im = db.get(Image, image_id)
    if not im or not _can_view(im.object, user):
        raise HTTPException(404)
    anns = []
    for f in im.faces:
        for a in f.annotations:
            d = AnnotationOut.model_validate(a).model_dump(mode="json")
            d["face_index"] = f.face_index
            anns.append(d)
    return {"image": ImageOut.model_validate(im).model_dump(mode="json"), "report": _load_report(im),
            "faces": [FaceOut.model_validate(f).model_dump(mode="json") for f in im.faces], "annotations": anns}


@router.get("/images/{image_id}/artefact/{name:path}")
def artefact(image_id: str, name: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    im = db.get(Image, image_id)
    if not im or not _can_view(im.object, user):
        raise HTTPException(404)
    if name == "original":
        p = Path(im.stored_path)
    else:
        base = (ANALYSIS_DIR / im.id).resolve()
        p = (base / name).resolve()
        if base not in p.parents or p.suffix.lower() not in (".jpg", ".png"):
            raise HTTPException(404)
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@router.get("/images/{image_id}/download")
def download(image_id: str, format: str = "json", request: Request = None, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    im = db.get(Image, image_id)
    if not im or not _can_view(im.object, user):
        raise HTTPException(404)
    rep = _load_report(im)
    anns = []
    for f in im.faces:
        for a in f.annotations:
            anns.append({"face_index": f.face_index, "created_at": a.created_at.isoformat(), "annotator": a.annotator_id, "layer": a.layer,
                         "verdict": a.verdict, "expression": a.expression, "confidence": a.confidence, "comment": a.comment})
    rep["annotations"] = anns
    _audit(db, user.id, "report_download", request, im.id, {"format": format})
    db.commit()
    base = f"face_heritage_{im.id[:12]}"
    if format == "json":
        return JSONResponse(rep, headers={"Content-Disposition": f'attachment; filename="{base}.json"'})
    if format == "csv":
        return Response(to_csv(rep), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{base}.csv"'})
    if format == "pdf":
        meta = {k: v for k, v in rep["object_metadata"].items() if k in ObjectMetadata.model_fields}
        return Response(to_pdf(rep, ANALYSIS_DIR / im.id, meta, anns), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{base}.pdf"'})
    raise HTTPException(400, "format must be json, csv or pdf")


@router.post("/images/{image_id}/rerun", status_code=201)
def rerun(image_id: str, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    im = db.get(Image, image_id)
    if not im or not _can_view(im.object, user):
        raise HTTPException(404)
    orig = Path(im.stored_path)
    if not orig.exists():
        raise HTTPException(404, "Original missing")
    o = im.object
    new = Image(object_id=o.id, uploaded_by=user.id, original_filename=im.original_filename, stored_path=im.stored_path,
                sha256=im.sha256, phash=im.phash, view_angle=im.view_angle, image_license=im.image_license,
                transformations=[{"op": "rerun", "source_image": im.id}])
    db.add(new)
    db.flush()
    domain_term = taxonomy.term("domain", o.domain) or {}
    report = run_image_analysis(orig, ANALYSIS_DIR / new.id, get_reference_index(),
                                {"domain": o.domain, "depicted": bool(domain_term.get("depicted", True))}, run_stability=RUN_STABILITY)
    new.status = report["status"]
    new.quality_score = report["image_quality"]["score"]
    new.face_count = report.get("face_count", 0)
    new.model_version = versions.COMPONENT_VERSIONS["expression_model"]
    new.dataset_version = report["versions"]["dataset"]
    for f in report.get("faces", []):
        face = Face(image_id=new.id, face_index=f["face_index"], bbox=list(f["box"]) if f.get("box") else None,
                    quality_score=f["quality"]["score"], status=f.get("status", "rejected"))
        if f.get("status") == "completed":
            face.expression = f["expression"]["primary"]; face.expression_confidence = f["expression"]["confidence"]
            face.pose_quality = f["pose"]["pose_quality"]; face.yaw = f["pose"]["yaw_deg"]; face.difficulty_level = f["difficulty_level"]
        db.add(face)
    _audit(db, user.id, "image_rerun", request, new.id, {"source": im.id})
    db.commit()
    db.refresh(new)
    report["analysis_id"] = new.id
    report["object_metadata"] = _object_out(o).model_dump(mode="json")
    return {"image": ImageOut.model_validate(new).model_dump(mode="json"), "report": report}


@router.delete("/images/{image_id}", status_code=204)
def delete_image(image_id: str, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    im = db.get(Image, image_id)
    if not im:
        raise HTTPException(404)
    if not (user.role == Role.admin or im.uploaded_by == user.id):
        raise HTTPException(403)
    shutil.rmtree(ANALYSIS_DIR / im.id, ignore_errors=True)
    shutil.rmtree(UPLOAD_DIR / im.id, ignore_errors=True)
    _audit(db, user.id, "image_delete", request, im.id)
    db.delete(im)
    db.commit()
    return Response(status_code=204)


# ------------------------------------------------------------------ annotations
@router.post("/faces/{face_id}/annotations", response_model=AnnotationOut, status_code=201)
def annotate(face_id: str, data: AnnotationIn, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    f = db.get(Face, face_id)
    if not f or not _can_view(f.image.object, user):
        raise HTTPException(404)
    if data.layer == "expert" and not (user.facs_certified or user.role == Role.admin):
        raise HTTPException(403, "Expert-layer annotations require a FACS-certified account (admin can grant it)")
    for fld in ("face_completeness", "representation_type", "subject_type"):
        if getattr(data, fld):
            taxonomy.validate_metadata({fld: getattr(data, fld)})
    a = Annotation(face_id=f.id, annotator_id=user.id, **data.model_dump())
    a.flag_for_review = data.flag_for_review or data.verdict == "disagree"
    if a.flag_for_review:
        f.flagged = True
    db.add(a)
    _audit(db, user.id, "annotation", request, f.id, data.model_dump())
    db.commit()
    db.refresh(a)
    return a


@router.get("/faces/{face_id}/annotations", response_model=list[AnnotationOut])
def list_annotations(face_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    f = db.get(Face, face_id)
    if not f or not _can_view(f.image.object, user):
        raise HTTPException(404)
    return f.annotations


@router.post("/objects/{object_id}/apply-suggestion")
def apply_suggestion(object_id: str, fields: dict, request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Accept the platform's suggested attribution (from reference-specimen similarity) into the object record.
    Only controlled fields are accepted; a note records that the values were suggested by the model."""
    o = db.get(ArtObject, object_id)
    if not o or not _can_edit(o, user):
        raise HTTPException(404)
    allowed = {k: v for k, v in fields.items() if k in ("culture", "region", "period", "material", "domain", "object_type") and v}
    meta = taxonomy.validate_metadata(dict(allowed))
    for k, v in meta.items():
        setattr(o, k, v)
    o.notes = ((o.notes or "") + f"\n[attribution suggested by {versions.COMPONENT_VERSIONS['retrieval']} and accepted by user {user.id}: {allowed}]").strip()
    o.metadata_version = (o.metadata_version or 1) + 1
    _audit(db, user.id, "apply_suggestion", request, o.id, allowed)
    db.commit()
    db.refresh(o)
    return _object_out(o)


@router.get("/taxonomy")
def get_taxonomy(user: User = Depends(get_current_user)):
    return taxonomy.load()


@router.get("/taxonomy/stats")
def taxonomy_stats(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    out = {}
    for col in ("domain", "region", "culture", "period", "material", "license"):
        rows = db.execute(select(getattr(ArtObject, col), func.count()).group_by(getattr(ArtObject, col))).all()
        out[col] = {(k or "unknown"): n for k, n in rows}
    out["objects"] = db.scalar(select(func.count()).select_from(ArtObject))
    out["images"] = db.scalar(select(func.count()).select_from(Image))
    out["faces"] = db.scalar(select(func.count()).select_from(Face))
    return out
