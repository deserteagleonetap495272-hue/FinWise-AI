"""Authenticated advanced analytics page."""

from datetime import date

from flask import Blueprint, flash, render_template, request
from flask_login import current_user, login_required

from app.services.analytics import analyze_transactions, default_date_range

bp = Blueprint("analytics", __name__, url_prefix="/analytics")


def _date_filter(name: str) -> date | None:
    raw = request.args.get(name, "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        flash(f"{name.replace('_', ' ').title()} must be a valid date.", "error")
        return None


@bp.get("")
@login_required
def overview():
    """Render user-specific analytics for a validated date range."""
    today = date.today()
    default_start, default_end = default_date_range(today)
    start_date = _date_filter("start_date") or default_start
    end_date = _date_filter("end_date") or default_end
    if end_date > today:
        flash("End date cannot be in the future.", "error")
        end_date = today
    if start_date > end_date:
        flash("Start date must be on or before end date.", "error")
        start_date, end_date = default_start, default_end
    if (end_date - start_date).days > 3660:
        flash("Choose a date range of 10 years or less.", "error")
        start_date, end_date = default_start, default_end

    period = request.args.get("period", "monthly").strip().lower()
    if period not in {"monthly", "yearly"}:
        flash("Choose monthly or yearly chart grouping.", "error")
        period = "monthly"
    analytics = analyze_transactions(current_user.id, start_date, end_date, today=today)
    return render_template("analytics/index.html", analytics=analytics,
                           selected_period=period, currency=current_user.currency)
