"""Authenticated ML dashboard, prediction, and review workflows."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required
from sqlalchemy import or_, select
from sqlalchemy.exc import SQLAlchemyError

from app.extensions import db
from app.models import Category, Transaction
from app.services.analytics import analyze_transactions, default_date_range
from app.services.budgets import ensure_budget_alerts
from app.services.ml.data_preprocessing import aggregate_expenses_by_month, fetch_user_transactions
from app.services.ml.forecasting import forecast_expenses
from app.services.ml import model_store

bp = Blueprint("ml", __name__, url_prefix="/ml")
SUGGESTION_SESSION_KEY = "ml_category_suggestion"


def _owned_transaction_or_404(transaction_id: int) -> Transaction:
    transaction = db.session.scalar(select(Transaction).where(
        Transaction.id == transaction_id,
        Transaction.user_id == current_user.id,
    ))
    if transaction is None:
        abort(404)
    return transaction


def _json_value(value):
    if isinstance(value, (date,)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _public_prediction(result: dict) -> dict:
    """Return prediction fields useful to a client without exposing internals."""
    keys = (
        "status", "data_sufficient", "predicted_expenses", "prediction_period",
        "selected_model", "model_based", "metrics", "baseline_metrics",
        "historical_months", "explanation", "model_artifact_status",
    )
    return _json_value({key: result.get(key) for key in keys})


def _public_forecast(result: dict) -> dict:
    keys = (
        "status", "data_sufficient", "historical_monthly_expenses", "forecasts",
        "forecast_month", "predicted_amount", "model_name", "evaluation_metrics",
        "forecast_horizon", "explanation", "limitations", "historical_month_count",
    )
    return _json_value({key: result.get(key) for key in keys})


def _dashboard_data() -> dict:
    """Assemble current-user ML data without training or mutating records."""
    user_id = current_user.id
    transactions = fetch_user_transactions(user_id)
    expense_status = model_store.get_model_status(user_id, "expense_prediction", transactions)
    category_status = model_store.get_model_status(user_id, "transaction_categorization", transactions)
    prediction = model_store.predict_expenses_for_user(user_id, transactions)

    if prediction.get("status") == "model_not_current":
        one_month_forecast = None
        three_month_forecast = None
    else:
        # Reuse this already-loaded Stage 3 result for both Stage 5 views.
        one_month_forecast = forecast_expenses(transactions, horizon=1, prediction_result=prediction)
        three_month_forecast = forecast_expenses(transactions, horizon=3, prediction_result=prediction)

    records = model_store.user_prediction_history(user_id)
    actual_months = aggregate_expenses_by_month(transactions)
    actual_by_period = {str(row.period): str(row.amount) for row in actual_months.itertuples(index=False)}
    predicted_by_period = {row.period_start.strftime("%Y-%m"): str(row.predicted_expense) for row in records}
    chart_periods = sorted(set(actual_by_period) | set(predicted_by_period))
    chart = {
        "labels": chart_periods,
        "actual": [actual_by_period.get(period) for period in chart_periods],
        "predicted": [predicted_by_period.get(period) for period in chart_periods],
        "currency": current_user.currency,
    }

    recent = []
    for row in transactions.tail(10).iloc[::-1].itertuples(index=False):
        recent.append({
            "id": int(row.id),
            "transaction_date": row.transaction_date.strftime("%d %b %Y"),
            "amount": Decimal(str(row.amount)),
            "transaction_type": str(row.transaction_type),
            "category": str(row.category),
        })

    today = date.today()
    start, end = default_date_range(today)
    analytics = analyze_transactions(user_id, start, end, today=today)
    anomalies = [
        {"date": row["date"], "category": row["category"], "amount": row["amount"], "label": row["label"]}
        for row in analytics["anomalies"][:5]
    ]

    saved_suggestion = session.get(SUGGESTION_SESSION_KEY)
    if not isinstance(saved_suggestion, dict):
        saved_suggestion = None
    categories = list(db.session.scalars(select(Category).where(
        Category.transaction_type.in_(["income", "expense"]),
        or_(Category.user_id == user_id, (Category.user_id.is_(None) & Category.is_system.is_(True))),
    ).order_by(Category.transaction_type, Category.name)))
    suggestion_choices = []
    if saved_suggestion:
        offered = {
            (item.get("category"), item.get("transaction_type"))
            for item in saved_suggestion.get("alternatives", []) if isinstance(item, dict)
        }
        suggestion_choices = [
            category for category in categories
            if (category.name, category.transaction_type) in offered
            and category.transaction_type == saved_suggestion.get("transaction_type")
        ]
    return {
        "transaction_count": len(transactions),
        "expense_status": expense_status,
        "category_status": category_status,
        "prediction": prediction,
        "one_month_forecast": one_month_forecast,
        "three_month_forecast": three_month_forecast,
        "chart": chart,
        "recent_transactions": recent,
        "anomalies": anomalies,
        "anomaly_methods": analytics["anomaly_methods"],
        "suggestion": saved_suggestion,
        "suggestion_choices": suggestion_choices,
        "categories": categories,
        "currency": current_user.currency,
    }


@bp.get("/")
@login_required
def dashboard():
    """Render prediction, forecast, categorization, and anomaly summaries."""
    try:
        return render_template("ml/index.html", **_dashboard_data())
    except SQLAlchemyError:
        db.session.rollback()
        flash("Machine learning information is temporarily unavailable.", "error")
        return render_template("ml/index.html", transaction_count=0, expense_status={"status": "unavailable"},
                               category_status={"status": "unavailable"}, prediction={},
                               one_month_forecast=None, three_month_forecast=None,
                               chart={"labels": [], "actual": [], "predicted": [], "currency": current_user.currency},
                               recent_transactions=[], anomalies=[], anomaly_methods={},
                               suggestion=None, categories=[], suggestion_choices=[], currency=current_user.currency)


@bp.get("/prediction")
@login_required
def prediction():
    """Return the signed-in user's current saved-model prediction as JSON."""
    transactions = fetch_user_transactions(current_user.id)
    return jsonify(_public_prediction(model_store.predict_expenses_for_user(current_user.id, transactions)))


@bp.get("/forecast")
@login_required
def forecast():
    """Return a one- or three-month forecast without implicit model training."""
    raw_horizon = request.args.get("horizon", "1").strip()
    if raw_horizon not in {"1", "3"}:
        return jsonify(error="Choose a forecast horizon of 1 or 3 months."), 400
    horizon = int(raw_horizon)
    transactions = fetch_user_transactions(current_user.id)
    prediction_result = model_store.predict_expenses_for_user(current_user.id, transactions)
    if prediction_result.get("status") == "model_not_current":
        return jsonify({
            "status": "model_not_current", "data_sufficient": False,
            "forecast_horizon": horizon, "historical_monthly_expenses": [],
            "forecasts": [], "predicted_amount": None,
            "explanation": "This user's expense model is missing or stale. Train or refresh it explicitly first.",
            "limitations": [],
        })
    return jsonify(_public_forecast(forecast_expenses(
        transactions, horizon=horizon, prediction_result=prediction_result
    )))


@bp.post("/train")
@login_required
def train():
    """Explicitly train or refresh one model for the authenticated user."""
    model_type = request.form.get("model_type", "")
    if model_type not in model_store.MODEL_TYPES:
        flash("Choose a supported model to train.", "error")
        return redirect(url_for("ml.dashboard"))
    force_refresh = request.form.get("force_refresh") == "1"
    try:
        transactions = fetch_user_transactions(current_user.id)
        result = model_store.train_user_model(
            current_user.id, model_type, transactions, force_refresh=force_refresh
        )
        history_saved = False
        training_result = result.get("training_result")
        if model_type == "expense_prediction" and training_result:
            try:
                history_saved = bool(model_store.save_prediction_history(current_user.id, training_result))
            except Exception:
                db.session.rollback()
    except Exception:
        # Do not send artifact paths, transaction data, or exception details to the browser.
        db.session.rollback()
        flash("Training could not be completed. Please try again later.", "error")
    else:
        if result["status"] == "already_current":
            flash("This model is already current; no retraining was needed.", "info")
        elif result["status"] in {"insufficient_data", "baseline_only"}:
            detail = (result.get("training_result") or {}).get("explanation")
            flash(detail or "The available history is insufficient for a fitted model. No artifact was saved.", "warning")
        else:
            message = "The model was trained and saved for your account."
            if model_type == "expense_prediction" and history_saved:
                message += " Its next-month estimate was added to your prediction history."
            flash(message, "success")
    return redirect(url_for("ml.dashboard"))


@bp.post("/transactions/<int:transaction_id>/suggest")
@login_required
def suggest_category(transaction_id: int):
    """Generate a suggestion for one owned transaction using a current model."""
    transaction = _owned_transaction_or_404(transaction_id)
    transactions = fetch_user_transactions(current_user.id)
    result = model_store.predict_category_for_user(current_user.id, transactions, transaction.description)
    if result.get("status") not in {"suggestion", "unvalidated_suggestion"}:
        session.pop(SUGGESTION_SESSION_KEY, None)
        flash(result.get("explanation") or "No category suggestion is available. Train a current categorizer first.", "warning")
    else:
        # Store labels, not descriptions or browser-supplied category IDs.
        session[SUGGESTION_SESSION_KEY] = {
            "transaction_id": transaction.id,
            "predicted_category": result["predicted_category"],
            "transaction_type": result["transaction_type"],
            "confidence": float(result["confidence"]),
            "model_name": result["model_name"],
            "alternatives": result.get("alternatives", []),
        }
        flash("Review the category suggestion before applying it.", "info")
    return redirect(url_for("ml.dashboard"))


@bp.post("/transactions/<int:transaction_id>/accept")
@login_required
def accept_category(transaction_id: int):
    """Apply only an offered, compatible category after rechecking ownership."""
    transaction = _owned_transaction_or_404(transaction_id)
    suggestion = session.get(SUGGESTION_SESSION_KEY)
    if not isinstance(suggestion, dict) or suggestion.get("transaction_id") != transaction.id:
        flash("Request a fresh category suggestion before accepting it.", "error")
        return redirect(url_for("ml.dashboard"))

    selected_id = request.form.get("category_id", type=int)
    if selected_id is None:
        flash("Choose a valid suggested category.", "error")
        return redirect(url_for("ml.dashboard"))
    category = db.session.scalar(select(Category).where(
        Category.id == selected_id,
        Category.transaction_type == transaction.transaction_type,
        or_(Category.user_id == current_user.id,
            (Category.user_id.is_(None) & Category.is_system.is_(True))),
    ))
    offered = {
        (item.get("category"), item.get("transaction_type"))
        for item in suggestion.get("alternatives", []) if isinstance(item, dict)
    }
    if (category is None or suggestion.get("transaction_type") != transaction.transaction_type
            or (category.name, category.transaction_type) not in offered):
        flash("That category is not an available, compatible suggestion for this transaction.", "error")
        return redirect(url_for("ml.dashboard"))
    try:
        transaction.category = category
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        flash("The category could not be updated. Please try again.", "error")
    else:
        ensure_budget_alerts(current_user.id)
        session.pop(SUGGESTION_SESSION_KEY, None)
        flash("Suggested category applied. Your model was not retrained.", "success")
    return redirect(url_for("ml.dashboard"))


@bp.post("/transactions/<int:transaction_id>/reject")
@login_required
def reject_category(transaction_id: int):
    """Dismiss a suggestion without modifying its transaction."""
    _owned_transaction_or_404(transaction_id)
    suggestion = session.get(SUGGESTION_SESSION_KEY)
    if isinstance(suggestion, dict) and suggestion.get("transaction_id") == transaction_id:
        session.pop(SUGGESTION_SESSION_KEY, None)
    flash("Suggestion dismissed. The transaction was not changed.", "info")
    return redirect(url_for("ml.dashboard"))
