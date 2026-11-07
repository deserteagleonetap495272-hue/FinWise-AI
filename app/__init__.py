"""FinWise AI Flask application factory."""

import os
import logging
from pathlib import Path

from flask import Flask, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
from sqlalchemy.engine import make_url
from sqlalchemy import text

from config import CONFIG_BY_NAME, normalize_database_url
from app.extensions import csrf, db, login_manager, migrate


def create_app(config_name: str | None = None) -> Flask:
    """Create and configure a FinWise AI application instance."""
    app = Flask(__name__, instance_relative_config=True)
    selected_config = config_name or os.getenv("FLASK_CONFIG", "development")
    if selected_config not in CONFIG_BY_NAME:
        raise ValueError(f"Unknown application configuration: {selected_config!r}.")
    app.config.from_object(CONFIG_BY_NAME[selected_config])
    is_production = selected_config == "production"
    if not is_production:
        app.config.from_pyfile("config.py", silent=True)
        if selected_config != "testing":
            # Resolve environment values at factory time too, so local process
            # managers and tests that set them after import still work.
            if os.getenv("SECRET_KEY"):
                app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
            if os.getenv("DATABASE_URL"):
                app.config["SQLALCHEMY_DATABASE_URI"] = normalize_database_url(
                    os.getenv("DATABASE_URL")
                )

    if is_production:
        # Production secrets and database credentials must come from the host's
        # environment, never a checked-in file or development fallback.
        secret_key = os.getenv("SECRET_KEY")
        if not secret_key or len(secret_key.encode("utf-8")) < 32:
            raise RuntimeError("Production requires a SECRET_KEY containing at least 32 bytes.")
        try:
            database_url = normalize_database_url(os.getenv("DATABASE_URL"))
            parsed_url = make_url(database_url)
        except ValueError:
            raise RuntimeError("Production requires a valid PostgreSQL DATABASE_URL.") from None
        if (parsed_url.drivername != "postgresql+psycopg" or not parsed_url.host
                or not parsed_url.database or not parsed_url.username or not parsed_url.password):
            raise RuntimeError("Production requires a credentialed PostgreSQL DATABASE_URL.")

        app.config.update(SECRET_KEY=secret_key, SQLALCHEMY_DATABASE_URI=database_url)
        trusted_hosts = [host.strip() for host in os.getenv("TRUSTED_HOSTS", "").split(",") if host.strip()]
        render_host = os.getenv("RENDER_EXTERNAL_HOSTNAME")
        if render_host and render_host not in trusted_hosts:
            trusted_hosts.append(render_host)
        if trusted_hosts:
            app.config["TRUSTED_HOSTS"] = trusted_hosts

        # Render terminates TLS at its edge and forwards X-Forwarded-Proto.
        # Trust only that scheme header; this app does not use forwarded client
        # IP or host data for authentication or authorization decisions.
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=0, x_proto=1, x_host=0, x_port=0, x_prefix=0)
        app.logger.setLevel(logging.INFO)

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    if not app.config.get("SECRET_KEY"):
        if not app.config.get("TESTING"):
            raise RuntimeError("Set SECRET_KEY in the environment before starting FinWise AI.")
        app.config["SECRET_KEY"] = "testing-only-secret"

    db.init_app(app)
    migrate.init_app(app, db, directory="migrations/finwise", compare_type=True)
    login_manager.init_app(app)
    csrf.init_app(app)
    login_manager.login_view = "web.login"
    login_manager.login_message_category = "warning"

    from app import models  # noqa: F401  Register models with SQLAlchemy metadata.
    from app.routes import analytics, budgets, ml, savings, transactions, web

    app.register_blueprint(web.bp)
    app.register_blueprint(transactions.bp)
    app.register_blueprint(budgets.bp)
    app.register_blueprint(savings.bp)
    app.register_blueprint(analytics.bp)
    app.register_blueprint(ml.bp)

    from app.services.transactions import format_money

    app.jinja_env.filters["money"] = format_money

    @app.get("/health")
    def health():
        try:
            db.session.execute(text("SELECT 1"))
        except Exception:
            db.session.rollback()
            return jsonify(status="error", database="unavailable"), 503
        return jsonify(status="ok", database="ok", service="finwise-ai")

    if is_production:
        @app.errorhandler(404)
        def production_not_found(_error):
            return "Page not found.", 404

        @app.errorhandler(Exception)
        def production_internal_error(_error):
            db.session.rollback()
            # Keep descriptions, credentials, SQL text, and tracebacks out of
            # responses and routine application logs.
            app.logger.error("Unhandled request error; response details withheld.")
            return "An internal error occurred. Please try again later.", 500

    return app
