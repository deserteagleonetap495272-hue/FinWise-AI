"""Account authentication and protected web pages."""

from sqlalchemy.exc import IntegrityError
from flask import Blueprint, flash, redirect, render_template, url_for
from flask_login import current_user, login_required, login_user, logout_user

from app.extensions import db
from app.forms import LoginForm, RegistrationForm
from app.models import User
from app.services.transactions import ensure_default_categories, financial_summary, recent_transactions
from app.services.budgets import budget_summary, ensure_budget_alerts
from app.services.savings import savings_summary

bp = Blueprint("web", __name__)


@bp.get("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("web.dashboard"))
    return render_template("index.html")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("web.dashboard"))

    form = LoginForm()
    if form.validate_on_submit():
        user = db.session.scalar(db.select(User).where(User.email == form.email.data))
        if user is None or not user.check_password(form.password.data):
            flash("Invalid email or password.", "error")
        else:
            # Flask's signed-cookie session is cleared before establishing a fresh
            # authenticated session. Flask-Login then records only the user ID.
            from flask import session

            ensure_default_categories(user.id)
            ensure_budget_alerts(user.id)
            session.clear()
            login_user(user, fresh=True)
            session.permanent = True
            return redirect(url_for("web.dashboard"))

    return render_template("auth/login.html", form=form), 200


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("web.dashboard"))

    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(email=form.email.data, full_name=form.full_name.data)
        try:
            user.set_password(form.password.data)
        except ValueError as error:
            form.password.errors.append(str(error))
        else:
            db.session.add(user)
            try:
                db.session.flush()
                ensure_default_categories(user.id)
                db.session.commit()
            except IntegrityError:
                # The unique DB constraint also protects against simultaneous signups.
                db.session.rollback()
                flash("An account with that email already exists.", "error")
            else:
                flash("Your account is ready. Please log in.", "success")
                return redirect(url_for("web.login"))

    return render_template("auth/register.html", form=form), 200


@bp.post("/logout")
@login_required
def logout():
    logout_user()
    from flask import session

    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("web.login"))


@bp.get("/dashboard")
@login_required
def dashboard():
    """Show financial totals and charts calculated for the current user."""
    summary = financial_summary(current_user.id)
    transactions = recent_transactions(current_user.id)
    budgets = budget_summary(current_user.id)
    savings = savings_summary(current_user.id)
    return render_template("dashboard.html", summary=summary, transactions=transactions,
                           budgets=budgets, savings=savings)
