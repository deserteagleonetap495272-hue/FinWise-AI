"""Application configuration loaded from environment variables."""

import os
from datetime import timedelta

from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


def normalize_database_url(database_url: str | None) -> str:
    """Validate a SQLAlchemy database URL and select the installed PG driver.

    Render commonly supplies ``postgresql://`` (and some older services may
    still use ``postgres://``); this project installs Psycopg 3, not Psycopg 2.
    Passwords are preserved in the runtime URL but are never rendered here.
    """
    if not isinstance(database_url, str) or not database_url.strip():
        raise ValueError("DATABASE_URL is required.")
    try:
        url = make_url(database_url.strip())
    except (ArgumentError, TypeError, ValueError):
        raise ValueError("DATABASE_URL is not a valid SQLAlchemy URL.") from None
    if url.drivername in {"postgres", "postgresql"}:
        url = url.set(drivername="postgresql+psycopg")
    return url.render_as_string(hide_password=False)


class Config:
    """Safe defaults shared by local and production configurations."""

    SECRET_KEY = os.getenv("SECRET_KEY")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"
    SESSION_REFRESH_EACH_REQUEST = False
    PERMANENT_SESSION_LIFETIME = timedelta(hours=12)
    WTF_CSRF_ENABLED = True
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024
    ML_MODEL_STORE_DIR = os.getenv("ML_MODEL_STORE_DIR") or None


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_DOMAIN = None
    PREFERRED_URL_SCHEME = "https"
    WTF_CSRF_ENABLED = True
    TRUSTED_HOSTS = None


class TestingConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = os.getenv("TEST_DATABASE_URL", "sqlite:///:memory:")


CONFIG_BY_NAME = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}
