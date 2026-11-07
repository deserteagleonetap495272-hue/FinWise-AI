"""Budget periods, spending utilization, summaries, and persistent alerts."""

from calendar import monthrange
from datetime import date
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.extensions import db
from app.models import Budget, Category, Recommendation, Transaction

ZERO = Decimal("0.00")


class InvalidBudgetCategory(ValueError):
    """A budget category must be an expense category owned by the user."""


class OverlappingBudget(ValueError):
    """A category already has a budget covering part of the requested period."""


def month_bounds(month_date: date) -> tuple[date, date]:
    """Return the inclusive boundaries for the month containing a date."""
    return month_date.replace(day=1), month_date.replace(day=monthrange(month_date.year, month_date.month)[1])


def _category_for_user(user_id: int, category_id: int) -> Category:
    category = db.session.scalar(select(Category).where(
        Category.id == category_id,
        Category.user_id == user_id,
        Category.transaction_type == "expense",
    ))
    if category is None:
        raise InvalidBudgetCategory("Choose one of your expense categories.")
    return category


def _assert_no_overlap(user_id: int, category_id: int, start: date, end: date,
                       exclude_id: int | None = None) -> None:
    query = select(Budget.id).where(
        Budget.user_id == user_id,
        Budget.category_id == category_id,
        Budget.period_start <= end,
        Budget.period_end >= start,
    )
    if exclude_id is not None:
        query = query.where(Budget.id != exclude_id)
    if db.session.scalar(query.limit(1)) is not None:
        raise OverlappingBudget("A budget already covers part of this category and period.")


def create_budget(user_id: int, *, name: str, category_id: int, amount: Decimal,
                  period_type: str, period_start: date, period_end: date,
                  description: str | None = None) -> Budget:
    """Create a monthly or custom-range budget for a user's expense category."""
    if period_type == "monthly":
        period_start, period_end = month_bounds(period_start)
    elif period_type != "custom":
        raise ValueError("Choose a valid budget period.")
    if amount <= ZERO:
        raise ValueError("Budget amount must be greater than zero.")
    if period_end < period_start:
        raise ValueError("End date must be on or after start date.")
    category = _category_for_user(user_id, category_id)
    _assert_no_overlap(user_id, category.id, period_start, period_end)
    budget = Budget(
        user_id=user_id, category=category, name=name.strip(), amount=amount,
        period_start=period_start, period_end=period_end, description=description or None,
    )
    db.session.add(budget)
    try:
        db.session.commit()
    except IntegrityError as error:
        db.session.rollback()
        raise OverlappingBudget("A budget already exists for this category and period.") from error
    return budget


def update_budget(budget: Budget, user_id: int, *, name: str, category_id: int,
                  amount: Decimal, period_type: str, period_start: date,
                  period_end: date, description: str | None = None) -> Budget:
    """Update a budget after rechecking its category and period."""
    if period_type == "monthly":
        period_start, period_end = month_bounds(period_start)
    elif period_type != "custom":
        raise ValueError("Choose a valid budget period.")
    if amount <= ZERO:
        raise ValueError("Budget amount must be greater than zero.")
    if period_end < period_start:
        raise ValueError("End date must be on or after start date.")
    category = _category_for_user(user_id, category_id)
    _assert_no_overlap(user_id, category.id, period_start, period_end, exclude_id=budget.id)
    budget.name = name.strip()
    budget.category = category
    budget.amount = amount
    budget.period_start = period_start
    budget.period_end = period_end
    budget.description = description or None
    try:
        db.session.commit()
    except IntegrityError as error:
        db.session.rollback()
        raise OverlappingBudget("A budget already exists for this category and period.") from error
    return budget


def budget_progress(budget: Budget, today: date | None = None) -> dict:
    """Calculate exact spent, remaining, utilization, and status values."""
    spent = db.session.scalar(select(db.func.coalesce(db.func.sum(Transaction.amount), 0)).where(
        Transaction.user_id == budget.user_id,
        Transaction.category_id == budget.category_id,
        Transaction.transaction_type == "expense",
        Transaction.transaction_date >= budget.period_start,
        Transaction.transaction_date <= budget.period_end,
    )) or ZERO
    spent = Decimal(spent)
    amount = Decimal(budget.amount)
    usage = spent / amount * Decimal("100") if amount > ZERO else ZERO
    if usage > Decimal("100"):
        status = "overspent"
    elif usage >= Decimal("90"):
        status = "critical"
    elif usage >= Decimal("70"):
        status = "warning"
    else:
        status = "normal"
    today = today or date.today()
    return {
        "spent": spent,
        "remaining": amount - spent,
        "usage": usage,
        "status": status,
        "active": budget.period_start <= today <= budget.period_end,
    }


def budget_overview(user_id: int, today: date | None = None) -> list[dict]:
    """Return progress for only the requesting user's budgets."""
    budgets = db.session.scalars(select(Budget).where(Budget.user_id == user_id).order_by(Budget.period_start.desc(), Budget.name))
    result = []
    for budget in budgets:
        start, end = month_bounds(budget.period_start)
        period_type = "monthly" if (budget.period_start, budget.period_end) == (start, end) else "custom"
        result.append({"budget": budget, **budget_progress(budget, today), "period_type": period_type})
    return result


def ensure_budget_alerts(user_id: int, today: date | None = None) -> None:
    """Persist one recommendation alert per budget threshold, without repeats."""
    entries = budget_overview(user_id, today)
    existing_keys = set(db.session.scalars(
        select(Recommendation.dedupe_key).where(Recommendation.user_id == user_id)
    ))
    alerts = []
    for entry in entries:
        budget = entry["budget"]
        usage = entry["usage"]
        thresholds = []
        if usage >= Decimal("70"):
            thresholds.append((70, "Budget reached 70%", "warning"))
        if usage >= Decimal("90"):
            thresholds.append((90, "Budget reached 90%", "critical"))
        if usage > Decimal("100"):
            thresholds.append((100, "Budget exceeded", "overspent"))
        for threshold, label, level in thresholds:
            title = f"Budget {budget.id}: {label}"
            dedupe_key = f"budget:{budget.id}:{threshold}"
            if dedupe_key in existing_keys:
                continue
            alerts.append(Recommendation(
                user_id=user_id,
                title=title,
                message=f"{budget.category.name} spending is {usage:.1f}% of “{budget.name}”.",
                dedupe_key=dedupe_key,
                rationale={"type": "budget_alert", "budget_id": budget.id,
                           "threshold": threshold, "category": budget.category.name,
                           "usage_percentage": str(usage), "level": level},
            ))
            existing_keys.add(dedupe_key)
    if alerts:
        db.session.add_all(alerts)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            persisted_keys = set(db.session.scalars(
                select(Recommendation.dedupe_key).where(Recommendation.user_id == user_id)
            ))
            if any(alert.dedupe_key not in persisted_keys for alert in alerts):
                raise


def budget_alerts(user_id: int) -> list[Recommendation]:
    """Return this user's budget alert recommendations, newest first."""
    return list(db.session.scalars(select(Recommendation).where(
        Recommendation.user_id == user_id,
        Recommendation.rationale["type"].as_string() == "budget_alert",
        Recommendation.rationale["budget_id"].as_integer().in_(
            select(Budget.id).where(Budget.user_id == user_id)
        ),
    ).order_by(Recommendation.created_at.desc())))


def budget_summary(user_id: int, today: date | None = None) -> dict:
    """Summarize active allocations, spend, remaining, and limit pressure."""
    active = [entry for entry in budget_overview(user_id, today) if entry["active"]]
    return {
        "allocated": sum((Decimal(entry["budget"].amount) for entry in active), ZERO),
        "spent": sum((entry["spent"] for entry in active), ZERO),
        "remaining": sum((entry["remaining"] for entry in active), ZERO),
        "overspent_count": sum(entry["status"] == "overspent" for entry in active),
        "nearing_limit": [entry for entry in active if entry["usage"] >= Decimal("70")],
        "entries": active,
    }
