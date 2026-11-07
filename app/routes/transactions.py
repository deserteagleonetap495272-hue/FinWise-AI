"""Authenticated transaction history and CRUD routes."""

from datetime import date
from decimal import Decimal, InvalidOperation

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from wtforms import SelectChoice

from app.extensions import db
from app.forms import TransactionForm
from app.models import Category, Transaction
from app.services.transactions import (
    InvalidCategoryError,
    create_transaction,
    get_user_categories,
    update_transaction,
)
from app.services.budgets import ensure_budget_alerts

bp = Blueprint("transactions", __name__, url_prefix="/transactions")


def _owned_transaction_or_404(transaction_id: int) -> Transaction:
    transaction = db.session.scalar(select(Transaction).where(
        Transaction.id == transaction_id,
        Transaction.user_id == current_user.id,
    ))
    if transaction is None:
        abort(404)
    return transaction


def _configure_form(form: TransactionForm, categories: list[Category]) -> None:
    form.category_id.choices = [
        SelectChoice(str(category.id), category.name, {"data-type": category.transaction_type})
        for category in categories
    ]


def _parse_date_parameter(name: str) -> date | None:
    value = request.args.get(name, "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        flash(f"{name.replace('_', ' ').title()} must be a valid date.", "error")
        return None


def _parse_decimal_parameter(name: str) -> Decimal | None:
    value = request.args.get(name, "").strip()
    if not value:
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        flash(f"{name.replace('_', ' ').title()} must be a valid amount.", "error")
        return None
    if not number.is_finite() or number < 0:
        flash(f"{name.replace('_', ' ').title()} cannot be negative.", "error")
        return None
    return number


@bp.get("")
@login_required
def list_transactions():
    """List current-user transactions with composable filters and pagination."""
    categories = get_user_categories(current_user.id)
    query = select(Transaction).where(Transaction.user_id == current_user.id)
    search = request.args.get("q", "").strip()[:120]
    transaction_type = request.args.get("type", "").strip().lower()
    category_id = request.args.get("category_id", type=int)
    start_date = _parse_date_parameter("start_date")
    end_date = _parse_date_parameter("end_date")
    minimum = _parse_decimal_parameter("min_amount")
    maximum = _parse_decimal_parameter("max_amount")

    if search:
        query = query.where(Transaction.description.ilike(f"%{search}%"))
    if transaction_type in {"income", "expense"}:
        query = query.where(Transaction.transaction_type == transaction_type)
    elif transaction_type:
        flash("Choose a valid transaction type.", "error")
        transaction_type = ""
    if category_id is not None:
        owned_category_ids = {category.id for category in categories}
        if category_id in owned_category_ids:
            query = query.where(Transaction.category_id == category_id)
        else:
            flash("Choose one of your categories.", "error")
            category_id = None
    if start_date:
        query = query.where(Transaction.transaction_date >= start_date)
    if end_date:
        query = query.where(Transaction.transaction_date <= end_date)
    if start_date and end_date and start_date > end_date:
        flash("Start date must be on or before end date.", "error")
        query = query.where(False)
    if minimum is not None:
        query = query.where(Transaction.amount >= minimum)
    if maximum is not None:
        query = query.where(Transaction.amount <= maximum)
    if minimum is not None and maximum is not None and minimum > maximum:
        flash("Minimum amount must be less than or equal to maximum amount.", "error")
        query = query.where(False)

    pagination = db.paginate(
        query.order_by(Transaction.transaction_date.desc(), Transaction.created_at.desc(), Transaction.id.desc()),
        page=max(request.args.get("page", 1, type=int), 1),
        per_page=15,
        error_out=False,
    )
    return render_template(
        "transactions/list.html", pagination=pagination, categories=categories,
        filters={"q": search, "type": transaction_type, "category_id": category_id,
                 "start_date": request.args.get("start_date", ""),
                 "end_date": request.args.get("end_date", ""),
                 "min_amount": request.args.get("min_amount", ""),
                 "max_amount": request.args.get("max_amount", "")},
    )


@bp.route("/new", methods=["GET", "POST"])
@login_required
def new_transaction():
    categories = get_user_categories(current_user.id)
    form = TransactionForm()
    _configure_form(form, categories)
    if request.method == "GET":
        initial_type = request.args.get("type", "expense")
        form.transaction_type.data = initial_type if initial_type in {"income", "expense"} else "expense"
        form.transaction_date.data = date.today()

    if form.validate_on_submit():
        try:
            create_transaction(
                user_id=current_user.id,
                amount=form.amount.data,
                transaction_type=form.transaction_type.data,
                category_id=form.category_id.data,
                description=form.description.data,
                transaction_date=form.transaction_date.data,
                notes=form.notes.data,
            )
        except InvalidCategoryError as error:
            flash(str(error), "error")
        except SQLAlchemyError:
            db.session.rollback()
            flash("The transaction could not be saved. Please try again.", "error")
        else:
            ensure_budget_alerts(current_user.id)
            flash("Transaction added.", "success")
            return redirect(url_for("transactions.list_transactions"))
    return render_template("transactions/form.html", form=form, heading="Add transaction")


@bp.route("/<int:transaction_id>/edit", methods=["GET", "POST"])
@login_required
def edit_transaction(transaction_id: int):
    transaction = _owned_transaction_or_404(transaction_id)
    categories = get_user_categories(current_user.id)
    form = TransactionForm(obj=transaction)
    _configure_form(form, categories)
    if form.validate_on_submit():
        try:
            update_transaction(
                transaction, user_id=current_user.id, amount=form.amount.data,
                transaction_type=form.transaction_type.data, category_id=form.category_id.data,
                description=form.description.data, transaction_date=form.transaction_date.data,
                notes=form.notes.data,
            )
        except InvalidCategoryError as error:
            flash(str(error), "error")
        except SQLAlchemyError:
            db.session.rollback()
            flash("The transaction could not be updated. Please try again.", "error")
        else:
            ensure_budget_alerts(current_user.id)
            flash("Transaction updated.", "success")
            return redirect(url_for("transactions.list_transactions"))
    return render_template("transactions/form.html", form=form, heading="Edit transaction")


@bp.post("/<int:transaction_id>/delete")
@login_required
def delete_transaction(transaction_id: int):
    transaction = _owned_transaction_or_404(transaction_id)
    try:
        db.session.delete(transaction)
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        flash("The transaction could not be deleted. Please try again.", "error")
    else:
        ensure_budget_alerts(current_user.id)
        flash("Transaction deleted.", "success")
    return redirect(url_for("transactions.list_transactions"))
