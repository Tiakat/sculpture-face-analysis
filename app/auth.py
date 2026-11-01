"""Authentication & authorisation: bcrypt password hashing, JWT bearer
tokens, role-based access (researcher / admin), login rate limiting."""
from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .config import ACCESS_TOKEN_MINUTES, JWT_ALGORITHM, JWT_SECRET, LOGIN_MAX_ATTEMPTS, LOGIN_WINDOW_SECONDS
from .database import get_db
from .models import Role, User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

PASSWORD_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d).{10,}$")


def validate_password(pw: str) -> None:
    if not PASSWORD_RE.match(pw):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Password must be at least 10 characters and contain letters and digits.")


def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(pw, hashed)
    except Exception:
        return False


def create_access_token(user: User) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_MINUTES)
    payload = {"sub": str(user.id), "email": user.email, "role": user.role.value, "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    cred_exc = HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired credentials",
                             headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        uid = int(payload.get("sub"))
    except (JWTError, TypeError, ValueError):
        raise cred_exc
    user = db.get(User, uid)
    if user is None or not user.is_active:
        raise cred_exc
    return user


def require_role(*roles: Role):
    def dep(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Insufficient privileges")
        return user
    return dep


require_admin = require_role(Role.admin)


# ------------------------------------------------------------- rate limiting & account lockout
_attempts: dict[str, deque] = defaultdict(deque)
_failed_by_email: dict[str, deque] = defaultdict(deque)
LOCKOUT_FAILURES = 5
LOCKOUT_SECONDS = 900


def record_failed_login(email: str) -> None:
    _failed_by_email[email.lower()].append(time.time())


def check_lockout(email: str) -> None:
    dq = _failed_by_email[email.lower()]
    now = time.time()
    while dq and now - dq[0] > LOCKOUT_SECONDS:
        dq.popleft()
    if len(dq) >= LOCKOUT_FAILURES:
        raise HTTPException(status.HTTP_423_LOCKED, "Account temporarily locked after repeated failed sign-ins; try again in 15 minutes.")


def clear_failed(email: str) -> None:
    _failed_by_email.pop(email.lower(), None)


def check_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    dq = _attempts[ip]
    while dq and now - dq[0] > LOGIN_WINDOW_SECONDS:
        dq.popleft()
    if len(dq) >= LOGIN_MAX_ATTEMPTS:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many attempts; try again later.")
    dq.append(now)
