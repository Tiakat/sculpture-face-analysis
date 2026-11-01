"""Shared helpers for MocheFaceDB ingestion.

Dataset layout (data/mochefacedb/):
  images/<SOURCE>_<ID>/original.<ext>      untouched download
  images/<SOURCE>_<ID>/thumb.jpg           400px thumbnail
  metadata.jsonl                           one JSON record per object
  licenses.jsonl                           licence + attribution per image
  VERSION                                  dataset version stamp

Only images whose licence permits research reuse are stored; every record
keeps museum / object number / culture / date / geography / licence / URL.
"""
from __future__ import annotations

import os
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = Path(os.environ.get("GFP_DATA_DIR", ROOT / "data")) / "mochefacedb"
IMG_DIR = DB_DIR / "images"
META_FILE = DB_DIR / "metadata.jsonl"
LIC_FILE = DB_DIR / "licenses.jsonl"

HEADERS = {"User-Agent": "MocheFacePlatform/0.1 (academic research; contact: set MFP_CONTACT env var)"}

MOCHE_KEYWORDS = ("moche", "mochica", "portrait", "head", "vessel", "bottle", "stirrup", "retrato")

# Peruvian north-coast valleys used for morphological grouping.
VALLEYS = ["Piura", "Lambayeque", "Zaña", "Jequetepeque", "Chicama", "Moche", "Virú", "Chao", "Santa", "Nepeña", "Casma", "Huarmey"]


def valley_from_text(*texts: str | None) -> str | None:
    blob = " ".join(t for t in texts if t).lower()
    for v in VALLEYS:
        if v.lower() in blob or v.lower().replace("ú", "u") in blob:
            return v
    return None


def phase_from_text(*texts: str | None) -> str | None:
    blob = " ".join(t for t in texts if t)
    m = re.search(r"\b(?:Moche|Mochica|Phase)\s*(I{1,3}|IV|V)\b", blob)
    if m:
        return m.group(1)
    return None


def load_existing_ids() -> set[str]:
    if not META_FILE.exists():
        return set()
    return {json.loads(l)["id"] for l in META_FILE.read_text(encoding="utf-8").splitlines() if l.strip()}


def client() -> httpx.Client:
    return httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True)


def get_json(c: httpx.Client, url: str, params=None, retries: int = 3):
    for i in range(retries):
        try:
            r = c.get(url, params=params)
            if r.status_code == 429:
                time.sleep(5 * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            if i == retries - 1:
                print(f"  ! failed {url}: {e}", file=sys.stderr)
                return None
            time.sleep(2 * (i + 1))
    return None


def save_image(c: httpx.Client, url: str, record_id: str) -> tuple[str, str] | None:
    d = IMG_DIR / record_id
    d.mkdir(parents=True, exist_ok=True)
    try:
        r = c.get(url)
        r.raise_for_status()
    except httpx.HTTPError as e:
        print(f"  ! image download failed {url}: {e}", file=sys.stderr)
        return None
    ext = Path(url.split("?")[0]).suffix.lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"):
        ext = ".jpg"
    orig = d / f"original{ext}"
    orig.write_bytes(r.content)
    try:
        im = Image.open(orig).convert("RGB")
        im.thumbnail((400, 400))
        im.save(d / "thumb.jpg", quality=88)
    except Exception as e:
        print(f"  ! not an image: {e}", file=sys.stderr)
        orig.unlink(missing_ok=True)
        return None
    return str(orig.relative_to(DB_DIR)), str((d / "thumb.jpg").relative_to(DB_DIR))


def append_record(record: dict, license_record: dict):
    DB_DIR.mkdir(parents=True, exist_ok=True)
    record.setdefault("ingested_at", datetime.now(timezone.utc).isoformat())
    with META_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with LIC_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(license_record, ensure_ascii=False) + "\n")


def bump_version(note: str):
    vf = DB_DIR / "VERSION"
    n = 0
    if vf.exists():
        try:
            n = int(vf.read_text(encoding="utf-8").strip().split(".")[-1])
        except ValueError:
            n = 0
    ver = f"MocheFaceDB v0.{n + 1}"
    vf.write_text(ver + "\n", encoding="utf-8")
    with (DB_DIR / "CHANGELOG.md").open("a", encoding="utf-8") as f:
        f.write(f"- {ver} — {datetime.now(timezone.utc).date()} — {note}\n")
    print(f"Dataset stamped {ver}")
    return ver


def looks_like_moche_face(*texts: str | None) -> bool:
    blob = " ".join(t for t in texts if t).lower()
    return ("moche" in blob or "mochica" in blob) and any(k in blob for k in ("portrait", "head", "face", "retrato", "cabeza", "rostro", "figure"))
