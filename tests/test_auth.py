"""Registration, login, logout, and route protection tests."""

from app.extensions import db
from app.models import User


def registration_data(**overrides):
    data = {
        "full_name": "Avery Example",
        "email": "avery@example.com",
        "password": "SafePassphrase123",
        "confirm_password": "SafePassphrase123",
    }
    data.update(overrides)
    return data


def test_registration_stores_hash_and_redirects_to_login(client, app):
    response = client.post("/register", data=registration_data(), follow_redirects=True)
    assert response.status_code == 200
    assert b"Your account is ready" in response.data
    with app.app_context():
        user = db.session.scalar(db.select(User).where(User.email == "avery@example.com"))
        assert user is not None
        assert user.password_hash != "SafePassphrase123"
        assert user.check_password("SafePassphrase123")
        assert b"password_hash" not in response.data


def test_duplicate_email_is_rejected(client, registered_user, app):
    response = client.post("/register", data=registration_data())
    assert response.status_code == 200
    assert b"already exists" in response.data
    with app.app_context():
        assert db.session.scalar(db.select(db.func.count()).select_from(User)) == 1


def test_invalid_password_and_mismatched_confirmation_are_rejected(client, app):
    short_password = client.post("/register", data=registration_data(
        password="short", confirm_password="short",
    ))
    mismatch = client.post("/register", data=registration_data(
        confirm_password="DifferentPassphrase123",
    ))
    assert b"at least 12 characters" in short_password.data
    assert b"Passwords must match" in mismatch.data
    with app.app_context():
        assert db.session.scalar(db.select(db.func.count()).select_from(User)) == 0


def test_login_rejects_invalid_credentials(client, registered_user):
    response = client.post("/login", data={
        "email": registered_user,
        "password": "IncorrectPassphrase123",
    })
    assert response.status_code == 200
    assert b"Invalid email or password" in response.data


def test_login_and_logout(client, registered_user):
    login_response = client.post("/login", data={
        "email": registered_user,
        "password": "SafePassphrase123",
    })
    assert login_response.status_code == 302
    assert login_response.headers["Location"].endswith("/dashboard")
    assert client.get("/dashboard").status_code == 200

    logout_response = client.post("/logout")
    assert logout_response.status_code == 302
    assert client.get("/dashboard").status_code == 302


def test_dashboard_requires_authentication(client):
    response = client.get("/dashboard")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_csrf_protects_registration_when_enabled(client, app):
    app.config["WTF_CSRF_ENABLED"] = True
    response = client.post("/register", data=registration_data())
    assert response.status_code == 400


def test_logout_requires_authentication(client):
    response = client.post("/logout")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_registration_normalizes_email(client, app):
    client.post("/register", data=registration_data(email="  AVERY@EXAMPLE.COM  "))
    with app.app_context():
        assert db.session.scalar(db.select(User).where(User.email == "avery@example.com")) is not None
