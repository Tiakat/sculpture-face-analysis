"""Process-wide singletons (reference index)."""
from __future__ import annotations

from .config import REFERENCE_DIR
from .pipeline.reference_index import ReferenceIndex

_reference: ReferenceIndex | None = None


def get_reference_index() -> ReferenceIndex:
    global _reference
    if _reference is None:
        _reference = ReferenceIndex(REFERENCE_DIR)
    return _reference


def reload_reference_index() -> ReferenceIndex:
    global _reference
    _reference = ReferenceIndex(REFERENCE_DIR)
    return _reference
