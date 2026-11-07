"""Authenticated budget CRUD and alert dismissal routes."""

from calendar import monthrange
from datetime import date

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from wtforms import SelectChoice

from app.extensions import db
from app.forms import BudgetForm
from app.models import Budget, Category, Recommendation
from app.models.core import utc_now
from app.services.budgets import (
    InvalidBudgetCategory,
    OverlappingBudget,
    budget_alerts,
    budget_overview,
    budget_summary,
    create_budget,
    ensure_budget_alerts,
    update_budget,
)
from app.services.transactions import get_user_categories

bp = Blueprint("budgets", __name__, url_prefix="/budgets")


def _owned_budget_or_404(budget_id: int) -> Budget:
    budget = db.session.scalar(select(Budget).where(
        Budget.id == budget_id,
        Budget.user_id == current_user.id,
    ))
    if budget is None:
        abort(404)
    return budget


def _configure_form(form: BudgetForm, categories: list[Category]) -> None:
    form.category_id.choices = [
        SelectChoice(str(category.id), category.name) for category in categories
        if category.transaction_type == "expense"
    ]


def _render_budget_form(form: BudgetForm, heading: str, categories: list[Category]):
    return render_template("budgets/form.html", form=form, heading=heading, categories=categories)


@bp.get("")
@login_required
def list_budgets():
    categories = get_user_categories(current_user.id)
    entries = budget_overview(current_user.id)
    alerts = [alert for alert in budget_alerts(current_user.id) if alert.dismissed_at is None]
    return render_template("budgets/list.html", entries=entries, alerts=alerts, categories=categories)


@bp.route("/new", methods=["GET", "POST"])
@login_required
def new_budget():
    categories = get_user_categories(current_user.id)
    form = BudgetForm()
    _configure_form(form, categories)
    if request.method == "GET":
        today = date.today()
        form.period_start.data = today.replace(day=1)
        form.period_end.data = today.replace(day=monthrange(today.year, today.month)[1])
        form.period_type.data = "monthly"

    if form.validate_on_submit():
        try:
            create_budget(
                current_user.id, name=form.name.data, category_id=form.category_id.data,
                amount=form.amount.data, period_type=form.period_type.data,
                period_start=form.period_start.data, period_end=form.period_end.data,
                description=form.description.data,
            )
        except (InvalidBudgetCategory, OverlappingBudget, ValueError) as error:
            flash(str(error), "error")
        except SQLAlchemyError:
            db.session.rollback()
            flash("The budget could not be saved. Please try again.", "error")
        else:
            ensure_budget_alerts(current_user.id)
            flash("Budget created.", "success")
            return redirect(url_for("budgets.list_budgets"))
    return _render_budget_form(form, "Create budget", categories)


@bp.route("/<int:budget_id>/edit", methods=["GET", "POST"])
@login_required
def edit_budget(budget_id: int):
    budget = _owned_budget_or_404(budget_id)
    categories = get_user_categories(current_user.id)
    form = BudgetForm(obj=budget)
    _configure_form(form, categories)
    if request.method == "GET":
        form.period_type.data = "monthly" if (
            budget.period_start.day == 1
            and budget.period_end.day == monthrange(budget.period_end.year, budget.period_end.month)[1]
            and budget.period_start.month == budget.period_end.month
            and budget.period_start.year == budget.period_end.year
        ) else "custom"
    if form.validate_on_submit():
        try:
            update_budget(
                budget, current_user.id, name=form.name.data, category_id=form.category_id.data,
                amount=form.amount.data, period_type=form.period_type.data,
                period_start=form.period_start.data, period_end=form.period_end.data,
                description=form.description.data,
            )
        except (InvalidBudgetCategory, OverlappingBudget, ValueError) as error:
            flash(str(error), "error")
        except SQLAlchemyError:
            db.session.rollback()
            flash("The budget could not be updated. Please try again.", "error")
        else:
            ensure_budget_alerts(current_user.id)
            flash("Budget updated.", "success")
            return redirect(url_for("budgets.list_budgets"))
    return _render_budget_form(form, "Edit budget", categories)


@bp.post("/<int:budget_id>/delete")
@login_required
def delete_budget(budget_id: int):
    budget = _owned_budget_or_404(budget_id)
    try:
        db.session.delete(budget)
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        flash("The budget could not be deleted. Please try again.", "error")
    else:
        flash("Budget deleted.", "success")
    return redirect(url_for("budgets.list_budgets"))


@bp.post("/alerts/<int:alert_id>/dismiss")
@login_required
def dismiss_alert(alert_id: int):
    alert = db.session.scalar(select(Recommendation).where(
        Recommendation.id == alert_id,
        Recommendation.user_id == current_user.id,
        Recommendation.rationale["type"].as_string() == "budget_alert",
    ))
    if alert is None:
        abort(404)
    alert.dismissed_at = utc_now()
    db.session.commit()
    flash("Budget alert dismissed.", "success")
    return redirect(url_for("budgets.list_budgets"))
