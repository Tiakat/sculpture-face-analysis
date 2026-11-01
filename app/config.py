from __future__ import annotations

import os
import secrets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("GFP_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
ANALYSIS_DIR = DATA_DIR / "analyses"
REFERENCE_DIR = DATA_DIR / "reference_index"
CORPUS_DIR = DATA_DIR / "corpus"
STATIC_DIR = BASE_DIR / "app" / "static"

DATABASE_URL = os.getenv("GFP_DATABASE_URL", f"sqlite:///{DATA_DIR / 'platform.db'}")

_secret_file = DATA_DIR / ".jwt_secret"


def _load_secret() -> str:
    env = os.getenv("GFP_JWT_SECRET")
    if env:
        return env
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if _secret_file.exists():
        return _secret_file.read_text(encoding="utf-8").strip()
    s = secrets.token_urlsafe(48)
    _secret_file.write_text(s, encoding="utf-8")
    try:
        os.chmod(_secret_file, 0o600)
    except OSError:
        pass
    return s


JWT_SECRET = _load_secret()
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = int(os.getenv("GFP_TOKEN_MINUTES", "480"))

MAX_UPLOAD_MB = int(os.getenv("GFP_MAX_UPLOAD_MB", "25"))

# Accounts that are administrators. Everyone else is a researcher and can only see their own work.
# Administrator e-mails: GFP_ADMIN_EMAILS (comma-separated) and/or data/admin_emails.txt.
# If neither is set, the first account registered becomes the administrator and is
# recorded in data/admin_emails.txt (kept local, never committed).
ADMIN_FILE = DATA_DIR / "admin_emails.txt"


def admin_emails() -> set[str]:
    emails = {e.strip().lower() for e in os.getenv("GFP_ADMIN_EMAILS", "").split(",") if e.strip()}
    if ADMIN_FILE.exists():
        emails |= {l.strip().lower() for l in ADMIN_FILE.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")}
    return emails


def remember_admin(email: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with ADMIN_FILE.open("a", encoding="utf-8") as f:
        f.write(email.strip().lower() + "\n")
# Login lockout: after LOGIN_MAX_ATTEMPTS failures from one IP within LOGIN_WINDOW_SECONDS the IP is blocked for that window.
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# Set GFP_RUN_STABILITY=0 to skip robustness tests (faster, less informative)
RUN_STABILITY = os.getenv("GFP_RUN_STABILITY", "1") == "1"

# Rate limiting for auth endpoints (per IP)
LOGIN_MAX_ATTEMPTS = 10
LOGIN_WINDOW_SECONDS = 300

for d in (UPLOAD_DIR, ANALYSIS_DIR, REFERENCE_DIR, CORPUS_DIR):
    d.mkdir(parents=True, exist_ok=True)
