"""Production configuration and deployment-readiness safety checks."""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile

import pytest
from flask import jsonify, request
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.middleware.proxy_fix import ProxyFix

from app import create_app
from app.extensions import db
from app.services.ml import model_store
from config import normalize_database_url


VALID_DATABASE_URL = "postgresql://finwise_user:very-secret-password@db.example.internal:5432/finwise_prod"
VALID_SECRET = "test-only-production-key-with-over-32-bytes"


def set_production_environment(monkeypatch, *, secret=VALID_SECRET, database=VALID_DATABASE_URL):
    if secret is None:
        monkeypatch.delenv("SECRET_KEY", raising=False)
    else:
        monkeypatch.setenv("SECRET_KEY", secret)
    if database is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", database)
    monkeypatch.delenv("RENDER_EXTERNAL_HOSTNAME", raising=False)
    monkeypatch.delenv("TRUSTED_HOSTS", raising=False)


def test_production_requires_secret_key(monkeypatch):
    set_production_environment(monkeypatch, secret=None)
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        create_app("production")


@pytest.mark.parametrize("secret", ["short", "   ", "x" * 31])
def test_production_rejects_weak_secret_keys(monkeypatch, secret):
    set_production_environment(monkeypatch, secret=secret)
    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        create_app("production")


def test_production_requires_database_url(monkeypatch):
    set_production_environment(monkeypatch, database=None)
    with pytest.raises(RuntimeError, match="PostgreSQL DATABASE_URL"):
        create_app("production")


@pytest.mark.parametrize("database", ["not-a-url", "sqlite:///production.sqlite3"])
def test_production_rejects_malformed_or_non_postgres_database_urls(monkeypatch, database):
    set_production_environment(monkeypatch, database=database)
    with pytest.raises(RuntimeError, match="PostgreSQL DATABASE_URL") as error:
        create_app("production")
    assert "very-secret-password" not in str(error.value)


def test_database_url_parser_uses_psycopg_3_without_losing_credentials():
    normalized = normalize_database_url("postgres://finwise:p%40ss@private-db:5432/finwise")
    parsed = make_url(normalized)
    assert parsed.drivername == "postgresql+psycopg"
    assert parsed.password == "p@ss"
    assert parsed.host == "private-db"
    assert parsed.database == "finwise"
    assert normalize_database_url("postgresql+psycopg://user:pass@host/db").startswith(
        "postgresql+psycopg://"
    )


def test_production_cookie_csrf_and_debug_settings(monkeypatch):
    set_production_environment(monkeypatch)
    app = create_app("production")
    assert app.config["DEBUG"] is False
    assert app.config["TESTING"] is False
    assert app.config["SECRET_KEY"] == VALID_SECRET
    assert app.config["SESSION_COOKIE_SECURE"] is True
    assert app.config["SESSION_COOKIE_HTTPONLY"] is True
    assert app.config["SESSION_COOKIE_SAMESITE"] == "Lax"
    assert app.config["SESSION_COOKIE_DOMAIN"] is None
    assert app.config["WTF_CSRF_ENABLED"] is True
    assert app.config["PREFERRED_URL_SCHEME"] == "https"
    assert isinstance(app.wsgi_app, ProxyFix)
    assert Path(".python-version").read_text(encoding="utf-8").strip() == "3.14.3"


def test_production_proxy_trust_uses_scheme_only(monkeypatch):
    set_production_environment(monkeypatch)
    app = create_app("production")

    @app.get("/_test/proxy")
    def proxy_probe():
        return jsonify(secure=request.is_secure, remote_addr=request.remote_addr)

    response = app.test_client().get("/_test/proxy", headers={
        "X-Forwarded-Proto": "https",
        "X-Forwarded-For": "203.0.113.88",
    })
    assert response.status_code == 200
    assert response.json["secure"] is True
    # Client-IP forwarding is deliberately untrusted and unused by auth.
    assert response.json["remote_addr"] != "203.0.113.88"


def test_production_errors_are_generic_and_not_logged_with_tracebacks(monkeypatch, caplog):
    set_production_environment(monkeypatch)
    app = create_app("production")

    @app.get("/_test/failure")
    def raise_private_error():
        raise RuntimeError("private transaction description and DB password")

    caplog.set_level(logging.ERROR, logger=app.logger.name)
    response = app.test_client().get("/_test/failure")
    assert response.status_code == 500
    assert "private transaction description" not in response.get_data(as_text=True)
    assert "DB password" not in response.get_data(as_text=True)
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "private transaction description" not in messages
    assert "DB password" not in messages
    assert "Traceback" not in messages


def test_health_endpoint_returns_only_status_and_sanitizes_database_errors(app, client, monkeypatch):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json == {"status": "ok", "database": "ok", "service": "finwise-ai"}

    def fail(_statement, *args, **kwargs):
        raise SQLAlchemyError("postgres://secret-user:secret-password/private-db")

    monkeypatch.setattr(db.session, "execute", fail)
    failed = client.get("/health")
    assert failed.status_code == 503
    assert failed.json == {"status": "error", "database": "unavailable"}
    assert "secret" not in failed.get_data(as_text=True)


def test_model_artifact_missing_after_ephemeral_restart_fails_closed(app):
    with tempfile.TemporaryDirectory(dir=app.instance_path, prefix="missing-model-test-") as model_dir:
        app.config["ML_MODEL_STORE_DIR"] = model_dir
        with app.app_context():
            loaded = model_store.load_model_artifact(987654, "expense_prediction")
        assert loaded == {"status": "missing", "model": None, "metadata": None}


def test_development_and_testing_configurations_remain_available(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "local-development-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://local:local@localhost:5432/finwise_dev")
    development = create_app("development")
    testing = create_app("testing")
    assert development.config["DEBUG"] is True
    assert development.config["SESSION_COOKIE_SECURE"] is False
    assert testing.config["TESTING"] is True
    assert testing.config["SQLALCHEMY_DATABASE_URI"] == "sqlite:///:memory:"
