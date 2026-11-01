from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)
    full_name: str = Field(default="", max_length=255)
    institution: str = Field(default="", max_length=255)


class UserOut(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    institution: str
    role: str
    is_active: bool
    facs_certified: bool = False
    created_at: datetime
    last_login: datetime | None = None
    images_count: int = 0
    annotations_count: int = 0
    email_verified: bool = True
    reset_requested_at: datetime | None = None
    model_config = {"from_attributes": True}


class ForgotIn(BaseModel):
    email: EmailStr


class ResetIn(BaseModel):
    token: str
    password: str = Field(min_length=10, max_length=128)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ObjectMetadata(BaseModel):
    """Controlled-vocabulary metadata for an OBJECT. Validated by app.taxonomy."""
    title: str | None = None
    artist: str | None = None
    institution: str | None = None
    object_number: str | None = None
    source: str = "LOCAL"
    source_object_id: str | None = None
    source_url: str | None = None
    domain: str = "sculpture"
    region: str | None = None
    culture: str | None = None
    culture_subgroup: str | None = None
    period: str | None = None
    material: str | None = None
    technique: str | None = None
    object_type: str | None = None
    subject_type: str | None = None
    representation_type: str | None = None
    restoration_status: str | None = None
    polychromy: str | None = None
    date_start: int | None = None
    date_end: int | None = None
    date_estimate: int | None = None
    date_precision: str | None = None
    date_confidence: float | None = None
    date_display: str | None = None
    country: str | None = None
    site: str | None = None
    provenance_text: str | None = None
    mold_group: str | None = None
    license: str = "UNKNOWN"
    rights_holder: str | None = None
    cultural_sensitivity: list[str] | None = None
    access_restriction: str | None = None
    external_ids: dict | None = None
    literature: list[str] | None = None
    notes: str | None = None


class ImageMetadata(BaseModel):
    image_url: str | None = None
    iiif_url: str | None = None
    photographer: str | None = None
    image_license: str | None = None
    view_angle: str | None = None


class ObjectOut(ObjectMetadata):
    id: str
    created_at: datetime
    training_allowed: bool
    annotation_layer: str
    image_count: int = 0
    face_count: int = 0
    duplicate_group_id: int | None = None
    model_config = {"from_attributes": True}


class ImageOut(BaseModel):
    id: str
    object_id: str
    created_at: datetime
    original_filename: str
    sha256: str
    phash: str | None
    width: int | None
    height: int | None
    status: str
    quality_score: float | None
    face_count: int
    model_version: str | None
    dataset_version: str | None
    error: str | None
    view_angle: str | None
    image_license: str | None
    model_config = {"from_attributes": True}


class FaceOut(BaseModel):
    id: str
    image_id: str
    face_index: int
    bbox: list | None
    quality_score: float | None
    pose_quality: float | None
    yaw: float | None
    expression: str | None
    expression_confidence: float | None
    difficulty_level: int | None
    flagged: bool
    status: str
    model_config = {"from_attributes": True}


class AnnotationIn(BaseModel):
    layer: Literal["gold_standard", "expert"] = "gold_standard"
    verdict: Literal["agree", "disagree", "uncertain"]
    expression: str | None = None
    expression_distribution: dict[str, float] | None = None
    aus: dict[str, Literal["present", "absent", "uncertain"]] | None = None
    face_completeness: str | None = None
    structures_visible: dict[str, bool] | None = None
    representation_type: str | None = None
    subject_type: str | None = None
    difficulty_level: int | None = Field(default=None, ge=0, le=6)
    confidence: float | None = Field(default=None, ge=0, le=1)
    comment: str | None = Field(default=None, max_length=4000)
    flag_for_review: bool = False


class AnnotationOut(AnnotationIn):
    id: int
    face_id: str
    annotator_id: int
    created_at: datetime
    review_status: str
    model_config = {"from_attributes": True}


class RoleUpdate(BaseModel):
    role: Literal["researcher", "annotator", "admin"]
    is_active: bool | None = None
    facs_certified: bool | None = None


class ReviewDecision(BaseModel):
    review_status: Literal["accepted", "rejected", "open"]


class CollectionIn(BaseModel):
    name: str
    version: str = "0.1"
    description: str = ""
    filter_json: dict | None = None
