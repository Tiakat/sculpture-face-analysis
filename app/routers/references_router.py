from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ..auth import get_current_user, require_admin
from ..config import CORPUS_DIR
from ..models import User
from ..state import get_reference_index, reload_reference_index

router = APIRouter(prefix="/api/references", tags=["references"])


@router.get("/stats")
def stats(user: User = Depends(get_current_user)):
    ref = get_reference_index()
    recs = ref.records
    return {
        "dataset_version": ref.dataset_version,
        "size": ref.size,
        "built_at": ref.built_at,
        "storage_dir": str(ref.dir),
        "by_museum": Counter(r.get("museum") for r in recs),
        "by_domain": Counter(r.get("domain") or "unknown" for r in recs),
        "by_culture": Counter(r.get("culture") or "unknown" for r in recs),
        "by_region": Counter(r.get("region") or "unknown" for r in recs),
        "by_period": Counter(r.get("period") or "unknown" for r in recs),
        "by_expression": Counter(r.get("expression") or "unlabelled" for r in recs),
        "by_license": Counter(r.get("license") or "unknown" for r in recs),
    }


@router.get("")
def list_references(limit: int = 500, offset: int = 0, user: User = Depends(get_current_user)):
    recs = get_reference_index().records
    return {"total": len(recs), "items": recs[offset: offset + limit]}


@router.get("/{reference_id}/image")
def reference_image(reference_id: str, thumb: bool = True, user: User = Depends(get_current_user)):
    ref = get_reference_index()
    rec = next((r for r in ref.records if r.get("id") == reference_id), None)
    if rec is None:
        raise HTTPException(404)
    rel = rec.get("thumbnail_path" if thumb else "image_path") or rec.get("image_path")
    if not rel:
        raise HTTPException(404)
    p = (CORPUS_DIR / rel).resolve()
    if CORPUS_DIR.resolve() not in p.parents or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@router.post("/reload")
def reload(user: User = Depends(require_admin)):
    ref = reload_reference_index()
    return {"dataset_version": ref.dataset_version, "size": ref.size}
