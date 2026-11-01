from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import admin_emails, remember_admin
from ..auth import (check_lockout, check_rate_limit, clear_failed, create_access_token, get_current_user, hash_password,
                    record_failed_login, validate_password, verify_password)
from ..database import get_db
from ..models import AuditLog, Role, User
from ..schemas import ForgotIn, RegisterIn, ResetIn, TokenOut, UserOut
from .. import mailer
import secrets
from datetime import timedelta
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])


def audit(db: Session, action: str, user_id: int | None, request: Request, target: str | None = None, details: dict | None = None):
    db.add(AuditLog(user_id=user_id, action=action, target=target, details=details,
                    ip=request.client.host if request.client else None))


@router.post("/register", response_model=UserOut, status_code=201)
def register(data: RegisterIn, request: Request, db: Session = Depends(get_db)):
    check_rate_limit(request)
    validate_password(data.password)
    email = data.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this e-mail already exists")
    admins = admin_emails()
    if not admins and not db.scalar(select(User).where(User.role == Role.admin)):
        remember_admin(email)          # fresh installation: the first account becomes the administrator
        admins = {email}
    role = Role.admin if email in admins else Role.researcher   # admin rights are tied to configured e-mails only
    user = User(email=email, full_name=data.full_name.strip(), institution=data.institution.strip(),
                hashed_password=hash_password(data.password), role=role)
    # E-mail confirmation when an SMTP account is configured; otherwise the account is active immediately.
    if mailer.configured() and role != Role.admin:
        user.email_verified = False
        user.verify_token = secrets.token_urlsafe(32)
        link = f"{mailer.public_url(request)}/api/auth/verify?token={user.verify_token}"
        mailer.send(email, "Confirm your account", f"Hello {user.full_name or ''},\n\nPlease confirm your account by opening this link:\n{link}\n\nIf you did not register, ignore this message.")
    else:
        user.email_verified = True
    db.add(user)
    db.flush()
    audit(db, "register", user.id, request, details={"role": role.value, "verification_email_sent": not user.email_verified})
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=TokenOut)
def login(request: Request, form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    check_rate_limit(request)
    check_lockout(form.username)
    user = db.scalar(select(User).where(User.email == form.username.lower()))
    if not user or not verify_password(form.password, user.hashed_password):
        record_failed_login(form.username)
        audit(db, "login_failed", None, request, details={"email": form.username.lower()})
        db.commit()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect e-mail or password")
    clear_failed(form.username)
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Account disabled")
    if not user.email_verified:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Please confirm your e-mail address first (check your inbox, or ask the administrator).")
    user.last_login = datetime.now(timezone.utc)
    # keep roles consistent with the configured administrator list
    admins = admin_emails()
    if user.email in admins and user.role != Role.admin:
        user.role = Role.admin
    elif admins and user.email not in admins and user.role == Role.admin:
        user.role = Role.researcher
    audit(db, "login", user.id, request)
    db.commit()
    return TokenOut(access_token=create_access_token(user), user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user


@router.get("/verify", response_class=HTMLResponse)
def verify(token: str, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.verify_token == token))
    if not user:
        return HTMLResponse("<h2>Invalid or already used confirmation link.</h2>", status_code=400)
    user.email_verified = True
    user.verify_token = None
    db.commit()
    return HTMLResponse("<h2>Your account is confirmed.</h2><p>You can close this page and sign in.</p>")


@router.post("/forgot")
def forgot(data: ForgotIn, request: Request, db: Session = Depends(get_db)):
    """Password forgotten: e-mails a reset link when SMTP is configured; otherwise the request is
    shown to the administrator, who hands the link to the person. Always answers the same way."""
    check_rate_limit(request)
    user = db.scalar(select(User).where(User.email == data.email.lower()))
    if user:
        user.reset_token = secrets.token_urlsafe(32)
        user.reset_expires = datetime.now(timezone.utc) + timedelta(hours=2)
        user.reset_requested_at = datetime.now(timezone.utc)
        link = f"{mailer.public_url(request)}/?reset={user.reset_token}"
        sent = mailer.send(user.email, "Password reset", f"Open this link within 2 hours to choose a new password:\n{link}")
        audit(db, "password_reset_requested", user.id, request, details={"email_sent": sent})
        db.commit()
    return {"ok": True, "message": "If this e-mail exists, a reset link has been sent"
            + ("." if mailer.configured() else " to the administrator, who will contact you (no e-mail service is configured).")}


@router.post("/reset")
def reset(data: ResetIn, request: Request, db: Session = Depends(get_db)):
    validate_password(data.password)
    user = db.scalar(select(User).where(User.reset_token == data.token))
    if not user or not user.reset_expires or user.reset_expires < datetime.now(timezone.utc):
        raise HTTPException(400, "Invalid or expired reset link")
    user.hashed_password = hash_password(data.password)
    user.reset_token = None
    user.reset_expires = None
    user.reset_requested_at = None
    user.email_verified = True
    audit(db, "password_reset", user.id, request)
    db.commit()
    return {"ok": True}
