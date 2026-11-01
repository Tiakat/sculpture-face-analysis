"""Ontology-backed relational model.

OBJECT (artwork / artefact / photograph)  ─┬─ IMAGE (one view / one file)  ─┬─ FACE (one face instance)
                                           │                                 │      ├─ landmarks, AUs, expression, morphology (Analysis JSON)
                                           │                                 │      └─ ANNOTATION (human, audited)
                                           ├─ COLLECTION membership (e.g. "Moche research subset v1")
                                           └─ DUPLICATE GROUP (same object across sources)

Every face analysis belongs to exactly one IMAGE, which belongs to exactly one
OBJECT.  Splits, deduplication and evaluation always operate on OBJECT (or
duplicate-group) identity, never on images or faces.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Table, Text, JSON, Column
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _now():
    return datetime.now(timezone.utc)


def _uuid():
    return uuid.uuid4().hex


class Role(str, enum.Enum):
    researcher = "researcher"
    annotator = "annotator"
    admin = "admin"


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), default="")
    institution: Mapped[str] = mapped_column(String(255), default="")
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[Role] = mapped_column(Enum(Role), default=Role.researcher, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    facs_certified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verify_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reset_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reset_expires: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reset_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


collection_members = Table(
    "collection_members", Base.metadata,
    Column("collection_id", ForeignKey("collections.id"), primary_key=True),
    Column("object_id", ForeignKey("objects.id"), primary_key=True),
)


class Collection(Base):
    """A versioned research subset, e.g. 'Moche portrait vessels v1'."""
    __tablename__ = "collections"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    version: Mapped[str] = mapped_column(String(32), default="0.1")
    description: Mapped[str] = mapped_column(Text, default="")
    filter_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)   # taxonomy filter that defines it
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    objects: Mapped[list["ArtObject"]] = relationship(secondary=collection_members, back_populates="collections")


class DuplicateGroup(Base):
    __tablename__ = "duplicate_groups"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_object_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    method: Mapped[str] = mapped_column(String(64), default="phash+embedding")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ArtObject(Base):
    __tablename__ = "objects"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    # provenance
    source: Mapped[str] = mapped_column(String(64), default="LOCAL")          # institution code
    source_object_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    source_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    institution: Mapped[str | None] = mapped_column(String(255), nullable=True)
    object_number: Mapped[str | None] = mapped_column(String(255), nullable=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    artist: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # controlled taxonomy (validated by app.taxonomy)
    domain: Mapped[str] = mapped_column(String(64), default="sculpture", index=True)
    region: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    culture: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    culture_subgroup: Mapped[str | None] = mapped_column(String(64), nullable=True)   # e.g. Moche IV
    period: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    material: Mapped[str | None] = mapped_column(String(64), nullable=True)
    technique: Mapped[str | None] = mapped_column(String(64), nullable=True)
    object_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subject_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    representation_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    restoration_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    polychromy: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # chronology with uncertainty (astronomical years: -400 = 401 BCE)
    date_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    date_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    date_estimate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    date_precision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    date_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    date_display: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # geography
    country: Mapped[str | None] = mapped_column(String(128), nullable=True)
    site: Mapped[str | None] = mapped_column(String(255), nullable=True)
    provenance_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    mold_group: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    # rights & ethics
    license: Mapped[str] = mapped_column(String(32), default="UNKNOWN", index=True)
    rights_holder: Mapped[str | None] = mapped_column(String(255), nullable=True)
    training_allowed: Mapped[bool] = mapped_column(Boolean, default=False)
    cultural_sensitivity: Mapped[list | None] = mapped_column(JSON, nullable=True)
    access_restriction: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # misc
    external_ids: Mapped[dict | None] = mapped_column(JSON, nullable=True)   # wikidata, AAT, TGN…
    literature: Mapped[list | None] = mapped_column(JSON, nullable=True)     # attached scholarship
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    duplicate_group_id: Mapped[int | None] = mapped_column(ForeignKey("duplicate_groups.id"), nullable=True)
    annotation_layer: Mapped[str] = mapped_column(String(32), default="corpus_auto")
    metadata_version: Mapped[int] = mapped_column(Integer, default=1)

    images: Mapped[list["Image"]] = relationship(back_populates="object", cascade="all, delete-orphan")
    collections: Mapped[list[Collection]] = relationship(secondary=collection_members, back_populates="objects")

    def taxonomy_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "domain", "region", "culture", "culture_subgroup", "period", "material", "technique", "object_type",
            "subject_type", "representation_type", "restoration_status", "polychromy", "date_start", "date_end",
            "date_estimate", "date_precision", "date_confidence", "date_display", "country", "site", "license",
            "cultural_sensitivity", "institution", "object_number", "title", "artist", "source", "source_url")}


class Image(Base):
    __tablename__ = "images"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    object_id: Mapped[str] = mapped_column(ForeignKey("objects.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    uploaded_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    original_filename: Mapped[str] = mapped_column(String(512))
    stored_path: Mapped[str] = mapped_column(String(1024))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    phash: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    iiif_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    photographer: Mapped[str | None] = mapped_column(String(255), nullable=True)
    image_license: Mapped[str | None] = mapped_column(String(32), nullable=True)   # can differ from object metadata licence
    view_angle: Mapped[str | None] = mapped_column(String(32), nullable=True)
    transformations: Mapped[list | None] = mapped_column(JSON, nullable=True)      # provenance of any processing
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    derived_from_image_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # analysis summary
    status: Mapped[str] = mapped_column(String(32), default="pending")
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    face_count: Mapped[int] = mapped_column(Integer, default=0)
    model_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    dataset_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    object: Mapped[ArtObject] = relationship(back_populates="images")
    faces: Mapped[list["Face"]] = relationship(back_populates="image", cascade="all, delete-orphan")


class Face(Base):
    """One face instance inside one image (paintings may contain many)."""
    __tablename__ = "faces"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    image_id: Mapped[str] = mapped_column(ForeignKey("images.id"), index=True)
    face_index: Mapped[int] = mapped_column(Integer, default=0)
    bbox: Mapped[list | None] = mapped_column(JSON, nullable=True)
    detection_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    pose_quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    yaw: Mapped[float | None] = mapped_column(Float, nullable=True)
    expression: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expression_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    difficulty_level: Mapped[int | None] = mapped_column(Integer, nullable=True)   # 0 clear frontal … 6 abstract
    face_completeness: Mapped[str | None] = mapped_column(String(64), nullable=True)
    flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(32), default="completed")

    image: Mapped[Image] = relationship(back_populates="faces")
    annotations: Mapped[list["Annotation"]] = relationship(back_populates="face", cascade="all, delete-orphan")


class Annotation(Base):
    """Human annotation with full audit trail (gold-standard / expert layers)."""
    __tablename__ = "annotations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    face_id: Mapped[str] = mapped_column(ForeignKey("faces.id"), index=True)
    annotator_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    layer: Mapped[str] = mapped_column(String(32), default="gold_standard")
    verdict: Mapped[str] = mapped_column(String(16))   # agree / disagree / uncertain
    expression: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expression_distribution: Mapped[dict | None] = mapped_column(JSON, nullable=True)   # soft labels
    aus: Mapped[dict | None] = mapped_column(JSON, nullable=True)   # {"AU12": "present|absent|uncertain", ...}
    face_completeness: Mapped[str | None] = mapped_column(String(64), nullable=True)
    structures_visible: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    representation_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subject_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    difficulty_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    flag_for_review: Mapped[bool] = mapped_column(Boolean, default=False)
    review_status: Mapped[str] = mapped_column(String(16), default="open")
    reviewed_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    face: Mapped[Face] = relationship(back_populates="annotations")


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    target: Mapped[str | None] = mapped_column(String(128), nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
