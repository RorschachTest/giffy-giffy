"""Sign in with Google, two roles, and a cap on how many people can use the app.

    GET  /login                   the sign-in page (and the waitlist / blocked message)
    GET  /auth/google             off to Google
    GET  /auth/google/callback    back from Google: the account is found or made, a session starts
    POST /auth/logout             ends the session
    GET  /me                      who is signed in

Roles:
    admin   search and share, add clips, Curate, manage users       (ADMIN_EMAILS)
    user    search and share (and what is added to those later)

The first MAX_USERS people who sign in become users; everyone after that is put on a
waitlist an admin can approve (Curate > Users). Admins never count against the cap.

What each request needs is decided in one place, access_for(): shared clip links and the
videos stay public so chat apps can preview them; everything else needs a signed-in,
active account, and the admin-only paths need an admin. When AUTH_ENABLED is off, or Google
is not configured, nobody is asked to sign in and every request is treated as an admin:
that is how the app behaved before, and how it runs in tests and on a laptop.

Google is used the standard OAuth way: the browser carries a one-time code back, the server
swaps it for a token directly with Google (over TLS, with the client secret), then asks
Google who the token belongs to. The session is a random token in an HttpOnly cookie; the
database keeps only its SHA-256, so a leaked table cannot be replayed.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from . import config

SESSION_COOKIE = "mc_session"
STATE_COOKIE = "mc_oauth_state"
GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO = "https://openidconnect.googleapis.com/v1/userinfo"

PUBLIC = "public"
USER = "user"
ADMIN = "admin"

# paths anyone can open: sign-in itself, shared clip links (chat previews), the files they show
_PUBLIC_PREFIXES = ("/login", "/auth/", "/c/", "/oembed", "/media/", "/static/", "/favicon.ico", "/legal", "/privacy")
# paths only admins can open: adding clips, Curate and everything it reads or changes
_ADMIN_PREFIXES = ("/upload", "/review", "/queries/", "/faces", "/people/", "/admin/",
                   "/understand/feedback", "/understand/stats", "/pilot")


def enabled() -> bool:
    return bool(config.AUTH_ENABLED and config.GOOGLE_CLIENT_ID and config.GOOGLE_CLIENT_SECRET)


def access_for(method: str, path: str) -> str:
    """What a request needs: PUBLIC, USER or ADMIN."""
    if path in ("/login",) or path.startswith(_PUBLIC_PREFIXES):
        return PUBLIC
    if path.startswith(_ADMIN_PREFIXES):
        return ADMIN
    if path.startswith("/clips/") and (method == "PATCH" or path.endswith("/state")):
        return ADMIN          # correcting a clip, and the model-input view, are Curate tools
    return USER


def admin_emails() -> set[str]:
    return {e.strip().lower() for e in config.ADMIN_EMAILS.split(",") if e.strip()}


# --------------------------------------------------------------------------- #
# Google
# --------------------------------------------------------------------------- #

def redirect_uri(base: str) -> str:
    return base.rstrip("/") + "/auth/google/callback"


def google_url(state: str, base: str) -> str:
    return GOOGLE_AUTH + "?" + urllib.parse.urlencode({
        "client_id": config.GOOGLE_CLIENT_ID, "redirect_uri": redirect_uri(base),
        "response_type": "code", "scope": "openid email profile", "state": state,
        "prompt": "select_account",
    })


def _post_form(url: str, data: dict) -> dict:
    req = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode(),
                                 headers={"content-type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _get_json(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def google_profile(code: str, base: str) -> dict:
    """The one-time code -> {email, name, picture} of a Google account with a verified email."""
    token = _post_form(GOOGLE_TOKEN, {
        "code": code, "client_id": config.GOOGLE_CLIENT_ID, "client_secret": config.GOOGLE_CLIENT_SECRET,
        "redirect_uri": redirect_uri(base), "grant_type": "authorization_code",
    })["access_token"]
    info = _get_json(GOOGLE_USERINFO, token)
    if not info.get("email") or not info.get("email_verified"):
        raise PermissionError("Google did not confirm this email address")
    # the basic profile, nothing else: these are all the scope "openid email profile" gives
    return {"sub": str(info.get("sub") or ""), "email": str(info["email"]).lower(),
            **{k: str(info.get(k) or "") for k in ("name", "given_name", "family_name", "picture", "locale")}}


# --------------------------------------------------------------------------- #
# Accounts and sessions
# --------------------------------------------------------------------------- #

def sign_in(conn, profile: dict) -> dict:
    """Find or make the account for a Google profile. A new account is an admin (ADMIN_EMAILS),
    an active user while fewer than MAX_USERS users are active, otherwise on the waitlist.
    Existing accounts keep their role and status (an admin may have changed them); an email
    added to ADMIN_EMAILS later is promoted on its next sign-in."""
    email = profile["email"]
    is_admin = email in admin_emails()
    conn.execute("SELECT pg_advisory_xact_lock(727275)")   # two first sign-ins at once must not both take the last seat
    row = conn.execute("SELECT * FROM users WHERE google_sub = %s OR email = %s ORDER BY google_sub IS NULL LIMIT 1",
                       (profile.get("sub") or None, email)).fetchone()
    fields = {k: profile.get(k, "") for k in ("name", "given_name", "family_name", "picture", "locale")}
    if row is None:
        active = conn.execute(
            "SELECT count(*) AS n FROM users WHERE role = 'user' AND status = 'active'").fetchone()["n"]
        role = ADMIN if is_admin else USER
        status = "active" if is_admin or active < config.MAX_USERS else "waitlist"
        row = conn.execute(
            """INSERT INTO users (google_sub, email, name, given_name, family_name, picture, locale, role, status)
               VALUES (%(sub)s, %(email)s, %(name)s, %(given_name)s, %(family_name)s, %(picture)s, %(locale)s,
                       %(role)s, %(status)s) RETURNING *""",
            {**fields, "sub": profile.get("sub") or None, "email": email, "role": role, "status": status}).fetchone()
    else:   # Google's copy is the truth for the profile; role and status stay ours
        row = conn.execute(
            """UPDATE users SET google_sub = coalesce(%(sub)s, google_sub), email = %(email)s, name = %(name)s,
                      given_name = %(given_name)s, family_name = %(family_name)s, picture = %(picture)s,
                      locale = %(locale)s, last_seen = now(),
                      role = CASE WHEN %(admin)s THEN 'admin' ELSE role END,
                      status = CASE WHEN %(admin)s THEN 'active' ELSE status END
                WHERE id = %(id)s RETURNING *""",
            {**fields, "sub": profile.get("sub") or None, "email": email, "admin": is_admin, "id": row["id"]}).fetchone()
    return row


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def start_session(conn, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    conn.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (%s, %s, %s)",
                 (_digest(token), user_id, datetime.now(timezone.utc) + timedelta(days=config.SESSION_DAYS)))
    return token


def end_session(conn, token: str | None) -> None:
    if token:
        conn.execute("DELETE FROM sessions WHERE token_hash = %s", (_digest(token),))


def user_for(conn, token: str | None) -> dict | None:
    """The account behind a session cookie, or None (no cookie, unknown, expired)."""
    if not token:
        return None
    return conn.execute(
        """SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id
            WHERE s.token_hash = %s AND s.expires_at > now()""", (_digest(token),)).fetchone()


def allowed(user: dict | None, need: str) -> bool:
    if need == PUBLIC:
        return True
    if not user or user["status"] != "active":
        return False
    return need == USER or user["role"] == ADMIN


def public_view(user: dict | None) -> dict | None:
    """What the page may know about the person signed in."""
    if user is None:
        return None
    return {"email": user["email"], "name": user["name"], "given_name": user["given_name"],
            "picture": user["picture"], "role": user["role"], "status": user["status"]}
