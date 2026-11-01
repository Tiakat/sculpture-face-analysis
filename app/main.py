"""Global Face Heritage Platform — FastAPI application entry point.

Run:  uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs
UI:   http://localhost:8000/
"""
from __future__ import annotations

import logging
import os

os.environ.setdefault("GLOG_minloglevel", "2")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import models  # noqa: F401  (registers tables)
from .config import STATIC_DIR
from .database import Base, engine
from .pipeline import versions
from .routers import admin_router, auth_router, objects_router, references_router
from .state import get_reference_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="Global Face Heritage Platform",
    version=versions.PLATFORM_VERSION,
    description="Analysis of depicted facial expression and facial morphology in sculpture, painting, drawing, print and photograph from any culture, period or material — with archaeological metadata kept separate from the facial computation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("MFP_CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    resp.headers["Cache-Control"] = "no-store"
    if not request.url.path.startswith(("/docs", "/redoc", "/openapi")):
        resp.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'"
    return resp


def _migrate():
    """Add columns introduced after the first release to existing SQLite databases."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    if "users" not in insp.get_table_names():
        return
    cols = {c["name"] for c in insp.get_columns("users")}
    wanted = {"email_verified": "BOOLEAN DEFAULT 1", "verify_token": "VARCHAR(64)", "reset_token": "VARCHAR(64)",
              "reset_expires": "DATETIME", "reset_requested_at": "DATETIME"}
    with engine.begin() as conn:
        for name, ddl in wanted.items():
            if name not in cols:
                conn.execute(text(f"ALTER TABLE users ADD COLUMN {name} {ddl}"))


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)
    _migrate()
    get_reference_index()


app.include_router(auth_router.router)
app.include_router(objects_router.router)
app.include_router(references_router.router)
app.include_router(admin_router.router)


@app.get("/api/health")
def health():
    ref = get_reference_index()
    return {"status": "ok", "versions": versions.COMPONENT_VERSIONS,
            "reference_index": {"dataset_version": ref.dataset_version, "size": ref.size}}


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logging.getLogger("app").exception("unhandled error")
    return JSONResponse({"detail": "Internal server error"}, status_code=500)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")
