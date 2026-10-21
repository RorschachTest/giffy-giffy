"""Sign in with Google, roles and the user cap. Google itself is replaced by a stand-in."""
import os

import pytest

from . import conftest  # noqa: F401

from app import auth, config


# --------------------------------------------------------------------------- #
# Which paths need what (no database)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method,path,need", [
    ("GET", "/c/abc123", auth.PUBLIC), ("GET", "/media/clip.mp4", auth.PUBLIC), ("GET", "/login", auth.PUBLIC),
    ("GET", "/auth/google/callback", auth.PUBLIC), ("GET", "/legal", auth.PUBLIC), ("GET", "/privacy", auth.PUBLIC), ("GET", "/static/base.css", auth.PUBLIC),
    ("GET", "/", auth.USER), ("GET", "/search", auth.USER), ("POST", "/clips/5/share", auth.USER), ("GET", "/clips/5", auth.USER),
    ("POST", "/upload", auth.ADMIN), ("GET", "/review", auth.ADMIN), ("GET", "/faces", auth.ADMIN),
    ("POST", "/people/3/aliases", auth.ADMIN), ("PATCH", "/clips/5", auth.ADMIN), ("GET", "/clips/5/state", auth.ADMIN),
    ("GET", "/admin/users", auth.ADMIN), ("GET", "/queries/failed", auth.ADMIN),
])
def test_access_rules(method, path, need):
    assert auth.access_for(method, path) == need


def test_allowed():
    admin = {"role": "admin", "status": "active"}
    user = {"role": "user", "status": "active"}
    waiting = {"role": "user", "status": "waitlist"}
    assert auth.allowed(None, auth.PUBLIC) and not auth.allowed(None, auth.USER)
    assert auth.allowed(user, auth.USER) and not auth.allowed(user, auth.ADMIN)
    assert auth.allowed(admin, auth.ADMIN)
    assert not auth.allowed(waiting, auth.USER)


def test_off_without_google_keys(monkeypatch):
    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "")
    assert not auth.enabled()


# --------------------------------------------------------------------------- #
# The whole flow, against the test database
# --------------------------------------------------------------------------- #

needs_db = pytest.mark.skipif(not os.getenv("TEST_DATABASE_URL"), reason="TEST_DATABASE_URL not set")


@pytest.fixture
def app_client(monkeypatch):
    from fastapi.testclient import TestClient
    from app import api, db

    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    monkeypatch.setattr(config, "GOOGLE_CLIENT_ID", "test-client")
    monkeypatch.setattr(config, "GOOGLE_CLIENT_SECRET", "test-secret")
    monkeypatch.setattr(config, "ADMIN_EMAILS", "boss@example.com")
    monkeypatch.setattr(config, "MAX_USERS", 1)
    db.init_db()
    with db.session() as conn:
        conn.execute("TRUNCATE sessions, users RESTART IDENTITY CASCADE")
    people = {}
    monkeypatch.setattr(auth, "google_profile", lambda code, base: people[code])
    with TestClient(api.app) as client:
        client.people = people
        yield client


def sign_in_as(client, email, sub=None):
    """Go through /auth/google and back as `email`; returns where the callback sent us."""
    client.cookies.clear()
    client.people[email] = {"sub": sub or email, "email": email, "name": email.split("@")[0], "given_name": "",
                            "family_name": "", "picture": "", "locale": "en"}
    start = client.get("/auth/google", follow_redirects=False)
    assert start.status_code == 303 and "accounts.google.com" in start.headers["location"]
    assert "scope=openid+email+profile" in start.headers["location"]          # nothing more is asked for
    state = client.cookies.get(auth.STATE_COOKIE)
    back = client.get(f"/auth/google/callback?code={email}&state={state}", follow_redirects=False)
    assert back.status_code == 303
    return back.headers["location"]


@needs_db
def test_signed_out_people_see_the_sign_in_page_but_shared_links_stay_open(app_client):
    assert app_client.get("/", follow_redirects=False).headers["location"].startswith("/login")
    assert app_client.get("/search?q=x").status_code == 401
    assert app_client.get("/login").status_code == 200
    assert app_client.get("/c/doesnotexist").status_code in (200, 404)    # public route: no sign-in wall


@needs_db
def test_a_forged_callback_is_refused(app_client):
    app_client.people["x@example.com"] = {"sub": "x", "email": "x@example.com"}
    r = app_client.get("/auth/google/callback?code=x@example.com&state=made-up", follow_redirects=False)
    assert r.headers["location"] == "/login?state=error"


@needs_db
def test_roles_cap_and_waitlist(app_client):
    assert sign_in_as(app_client, "boss@example.com") == "/"               # admin from ADMIN_EMAILS
    assert app_client.get("/admin/users").status_code == 200
    me = app_client.get("/me").json()
    assert me["role"] == "admin" and "google_sub" not in me

    assert sign_in_as(app_client, "first@example.com") == "/"              # takes the one user place
    assert app_client.get("/search?q=x").status_code == 200
    assert app_client.get("/admin/users").status_code == 403               # users can't open Curate
    assert app_client.post("/upload").status_code == 403                   # ... or add clips

    assert sign_in_as(app_client, "second@example.com") == "/login?state=waitlist"   # cap reached
    assert app_client.get("/search?q=x").status_code == 403

    # the admin lets the second person in, and blocks the first (signed out everywhere)
    sign_in_as(app_client, "boss@example.com")
    users = {u["email"]: u for u in app_client.get("/admin/users").json()["users"]}
    assert app_client.post(f"/admin/users/{users['second@example.com']['id']}", json={"status": "active"}).status_code == 200
    assert app_client.post(f"/admin/users/{users['first@example.com']['id']}", json={"status": "blocked"}).status_code == 200
    assert app_client.post(f"/admin/users/{users['boss@example.com']['id']}", json={"role": "user"}).status_code == 400

    assert sign_in_as(app_client, "second@example.com") == "/"
    assert sign_in_as(app_client, "first@example.com") == "/login?state=blocked"


@needs_db
def test_sign_out_ends_the_session(app_client):
    sign_in_as(app_client, "boss@example.com")
    assert app_client.get("/search?q=x").status_code == 200
    app_client.post("/auth/logout")
    assert app_client.get("/search?q=x").status_code == 401
