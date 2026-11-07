"""Shared pytest fixtures for isolated application tests."""

import pytest

from app import create_app
from app.extensions import db


@pytest.fixture
def app():
    application = create_app("testing")
    application.config.update(TESTING=True, WTF_CSRF_ENABLED=False)
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def registered_user(client):
    response = client.post("/register", data={
        "full_name": "Avery Example",
        "email": "avery@example.com",
        "password": "SafePassphrase123",
        "confirm_password": "SafePassphrase123",
    }, follow_redirects=True)
    assert response.status_code == 200
    return "avery@example.com"
