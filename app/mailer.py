"""E-mail sending (verification and password-reset links).

Configured with environment variables (any SMTP provider; for Gmail create an
"app password" at https://myaccount.google.com/apppasswords):

    GFP_SMTP_HOST=smtp.gmail.com
    GFP_SMTP_PORT=587
    GFP_SMTP_USER=you@gmail.com
    GFP_SMTP_PASSWORD=xxxx xxxx xxxx xxxx
    GFP_SMTP_FROM=you@gmail.com
    GFP_PUBLIC_URL=https://your-public-address      (used in the links)

When SMTP is not configured the platform still works: new accounts are
activated immediately and password-reset requests are listed on the Admin
page, where the administrator can hand the reset link to the person.
"""
from __future__ import annotations

import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

log = logging.getLogger(__name__)
PREFIX = os.getenv("APP_ENV_PREFIX", "GFP")


def _cfg(name: str, default: str = "") -> str:
    return os.getenv(f"{PREFIX}_{name}", default)


def configured() -> bool:
    return bool(_cfg("SMTP_HOST") and _cfg("SMTP_USER") and _cfg("SMTP_PASSWORD"))


def public_url(request=None) -> str:
    url = _cfg("PUBLIC_URL")
    if url:
        return url.rstrip("/")
    if request is not None:
        return str(request.base_url).rstrip("/")
    return "http://127.0.0.1:8765"


def send(to: str, subject: str, body: str) -> bool:
    if not configured():
        log.info("SMTP not configured — e-mail to %s not sent (%s)", to, subject)
        return False
    msg = EmailMessage()
    msg["From"] = _cfg("SMTP_FROM") or _cfg("SMTP_USER")
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    host, port = _cfg("SMTP_HOST"), int(_cfg("SMTP_PORT", "587"))
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=ssl.create_default_context(), timeout=20) as s:
                s.login(_cfg("SMTP_USER"), _cfg("SMTP_PASSWORD"))
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as s:
                s.starttls(context=ssl.create_default_context())
                s.login(_cfg("SMTP_USER"), _cfg("SMTP_PASSWORD"))
                s.send_message(msg)
        return True
    except Exception as e:  # pragma: no cover
        log.error("e-mail to %s failed: %s", to, e)
        return False
