"""Savings goal progress and contribution operations."""

from datetime import date
from decimal import Decimal

from sqlalchemy import select, func

from app.extensions import db
from app.models import GoalContribution, SavingsGoal
from app.models.core import utc_now

ZERO = Decimal("0.00")


def goal_progress(goal: SavingsGoal, today: date | None = None) -> dict:
    """Calculate saved amount, remaining balance, progress, and current status."""
    saved = db.session.scalar(select(func.coalesce(func.sum(GoalContribution.amount), 0)).where(
        GoalContribution.goal_id == goal.id,
    )) or ZERO
    saved = Decimal(saved)
    target = Decimal(goal.target_amount)
    remaining = max(target - saved, ZERO)
    percentage = min(saved / target * Decimal("100"), Decimal("100")) if target > ZERO else ZERO
    today = today or date.today()
    if saved >= target:
        status = "completed"
    elif goal.target_date is not None and goal.target_date < today:
        status = "overdue"
    elif saved > ZERO:
        status = "in_progress"
    else:
        status = "not_started"
    return {"saved": saved, "remaining": remaining, "percentage": percentage, "status": status}


def _refresh_completion(goal: SavingsGoal) -> None:
    """Keep the completion timestamp synchronized with contribution records."""
    progress = goal_progress(goal)
    if progress["status"] == "completed":
        goal.completed_at = goal.completed_at or utc_now()
    else:
        goal.completed_at = None


def goal_overview(user_id: int, today: date | None = None) -> list[dict]:
    """Return progress for this user's goals only."""
    goals = db.session.scalars(select(SavingsGoal).where(
        SavingsGoal.user_id == user_id,
    ).order_by(SavingsGoal.created_at.desc(), SavingsGoal.id.desc()))
    return [{"goal": goal, **goal_progress(goal, today)} for goal in goals]


def create_goal(user_id: int, *, name: str, target_amount: Decimal,
                target_date: date | None = None, description: str | None = None) -> SavingsGoal:
    """Create a goal belonging to the authenticated user."""
    if target_amount <= ZERO:
        raise ValueError("Target amount must be greater than zero.")
    goal = SavingsGoal(
        user_id=user_id, name=name.strip(), target_amount=target_amount,
        target_date=target_date, description=description or None,
    )
    db.session.add(goal)
    db.session.commit()
    return goal


def update_goal(goal: SavingsGoal, *, name: str, target_amount: Decimal,
                target_date: date | None = None, description: str | None = None) -> SavingsGoal:
    """Update goal metadata and recalculate its completion marker."""
    if target_amount <= ZERO:
        raise ValueError("Target amount must be greater than zero.")
    goal.name = name.strip()
    goal.target_amount = target_amount
    goal.target_date = target_date
    goal.description = description or None
    _refresh_completion(goal)
    db.session.commit()
    return goal


def add_contribution(goal: SavingsGoal, *, amount: Decimal, contributed_on: date,
                     note: str | None = None) -> GoalContribution:
    """Record a contribution and synchronize goal completion atomically."""
    if amount <= ZERO:
        raise ValueError("Contribution amount must be greater than zero.")
    contribution = GoalContribution(
        goal=goal, amount=amount, contributed_on=contributed_on, note=note or None,
    )
    db.session.add(contribution)
    db.session.flush()
    _refresh_completion(goal)
    db.session.commit()
    return contribution


def update_contribution(goal: SavingsGoal, contribution: GoalContribution, *, amount: Decimal,
                        contributed_on: date, note: str | None = None) -> GoalContribution:
    """Edit a contribution after route-level goal ownership checks."""
    if amount <= ZERO:
        raise ValueError("Contribution amount must be greater than zero.")
    if contribution.goal_id != goal.id:
        raise ValueError("Contribution does not belong to this goal.")
    contribution.amount = amount
    contribution.contributed_on = contributed_on
    contribution.note = note or None
    db.session.flush()
    _refresh_completion(goal)
    db.session.commit()
    return contribution


def delete_contribution(goal: SavingsGoal, contribution: GoalContribution) -> None:
    """Delete a contribution and recompute the goal's completion marker."""
    if contribution.goal_id != goal.id:
        raise ValueError("Contribution does not belong to this goal.")
    db.session.delete(contribution)
    db.session.flush()
    _refresh_completion(goal)
    db.session.commit()


def savings_summary(user_id: int, today: date | None = None) -> dict:
    """Summarize targets, contributions, remaining targets, and goal counts."""
    entries = goal_overview(user_id, today)
    return {
        "targets": sum((Decimal(entry["goal"].target_amount) for entry in entries), ZERO),
        "saved": sum((entry["saved"] for entry in entries), ZERO),
        "remaining": sum((entry["remaining"] for entry in entries), ZERO),
        "completed_count": sum(entry["status"] == "completed" for entry in entries),
        "active_count": sum(entry["status"] in {"not_started", "in_progress", "overdue"} for entry in entries),
        "entries": entries,
    }
