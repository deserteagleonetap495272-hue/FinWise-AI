"""User-scoped descriptive financial analytics and anomaly analysis."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal
from threading import RLock

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import func, select

from app.extensions import db
from app.models import Budget, Category, Transaction
from app.services.budgets import budget_progress

ZERO = Decimal("0.00")
MIN_STATISTICAL_RECORDS = 8
MIN_ISOLATION_RECORDS = 20
_ANOMALY_CACHE: dict[tuple, dict] = {}
_CACHE_LOCK = RLock()
_CACHE_LIMIT = 256


def _decimal_sum(values) -> Decimal:
    return sum((Decimal(value) for value in values), ZERO)


def _amount_string(value: Decimal) -> str:
    return f"{value:.2f}"


def _period_groups(frame: pd.DataFrame, period_column: str, periods: list,
                   labels: list[str]) -> list[dict]:
    """Aggregate income, expense, and net by a Pandas period column."""
    totals: dict[tuple, dict[str, Decimal]] = defaultdict(lambda: {"income": ZERO, "expense": ZERO})
    if not frame.empty:
        grouped = frame.groupby([period_column, "transaction_type"], sort=True)["amount"].agg(_decimal_sum)
        for (period, transaction_type), amount in grouped.items():
            totals[period][transaction_type] = amount
    rows = []
    for period, label in zip(periods, labels):
        income, expense = totals[period]["income"], totals[period]["expense"]
        rows.append({"period": label, "income": income, "expenses": expense, "net": income - expense})
    return rows


def _statistical_anomalies(expenses: pd.DataFrame) -> tuple[list[dict], dict]:
    if len(expenses) < MIN_STATISTICAL_RECORDS:
        return [], {"status": "insufficient_data", "minimum_records": MIN_STATISTICAL_RECORDS,
                    "records_analyzed": len(expenses)}

    flagged = []
    category_fences = {}
    # Compare a transaction with this user's own history in the same category.
    # A global amount distribution incorrectly calls a stable rent payment an
    # outlier simply because rent costs more than routine food/transport items.
    for category, group in expenses.groupby("category", sort=True):
        if len(group) < MIN_STATISTICAL_RECORDS:
            continue
        amounts = np.asarray([float(amount) for amount in group["amount"]], dtype=np.float64)
        mean = float(np.mean(amounts))
        standard_deviation = float(np.std(amounts, ddof=0))
        q1, q3 = np.percentile(amounts, [25, 75])
        iqr_fence = float(q3 + 1.5 * (q3 - q1))
        z_score_fence = mean + 3 * standard_deviation
        category_fences[str(category)] = {
            "records_analyzed": len(group), "iqr_upper_fence": f"{iqr_fence:.2f}",
            "z_score_upper_fence": f"{z_score_fence:.2f}",
        }
        for row in group.to_dict("records"):
            amount = float(row["amount"])
            z_score = (amount - mean) / standard_deviation if standard_deviation else 0.0
            reasons = []
            if amount > iqr_fence:
                reasons.append(f"Amount is above the {category} IQR upper fence ({iqr_fence:.2f}).")
            if z_score >= 3:
                reasons.append(f"Amount is {z_score:.1f} standard deviations above the {category} mean.")
            if reasons:
                flagged.append({
                    "transaction_id": int(row["id"]),
                    "date": row["transaction_date"].date().isoformat(),
                    "description": row["description"],
                    "category": row["category"],
                    "amount": _amount_string(row["amount"]),
                    "methods": ["statistical"],
                    "reasons": reasons,
                    "label": "Unusual spending · review",
                })
    return flagged, {
        "status": "analyzed", "minimum_records": MIN_STATISTICAL_RECORDS,
        "minimum_category_records": MIN_STATISTICAL_RECORDS,
        "records_analyzed": len(expenses), "category_fences": category_fences,
    }


def _anomaly_cache_key(user_id: int, start_date: date, end_date: date, frame: pd.DataFrame) -> tuple:
    latest_updated = None if frame.empty else frame["updated_at"].max().isoformat()
    latest_id = 0 if frame.empty else int(frame["id"].max())
    return id(db.engine), user_id, start_date.isoformat(), end_date.isoformat(), len(frame), latest_updated, latest_id


def _isolation_anomalies(user_id: int, start_date: date, end_date: date,
                         expenses: pd.DataFrame, frame: pd.DataFrame) -> tuple[list[dict], dict]:
    if len(expenses) < MIN_ISOLATION_RECORDS:
        return [], {"status": "insufficient_data", "minimum_records": MIN_ISOLATION_RECORDS,
                    "records_analyzed": len(expenses)}

    cache_key = _anomaly_cache_key(user_id, start_date, end_date, frame)
    with _CACHE_LOCK:
        cached = _ANOMALY_CACHE.get(cache_key)
    if cached is not None:
        return deepcopy(cached)

    category_features = pd.get_dummies(expenses["category"], dtype=float)
    features = category_features.copy()
    features.insert(0, "log_amount", np.log1p([float(value) for value in expenses["amount"]]))
    features.insert(1, "weekday", expenses["transaction_date"].dt.day_of_week.to_numpy(dtype=float))
    scaled = StandardScaler().fit_transform(features.to_numpy(dtype=np.float64))
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    predictions = model.fit_predict(scaled)
    flagged = []
    for (_, row), prediction in zip(expenses.iterrows(), predictions):
        if prediction != -1:
            continue
        flagged.append({
            "transaction_id": int(row["id"]),
            "date": row["transaction_date"].date().isoformat(),
            "description": row["description"],
            "category": row["category"],
            "amount": _amount_string(row["amount"]),
            "methods": ["isolation_forest"],
            "reasons": ["The transaction differs from this user's amount, category, and weekday patterns."],
            "label": "Unusual spending · review",
        })
    result = flagged, {"status": "analyzed", "minimum_records": MIN_ISOLATION_RECORDS,
                      "records_analyzed": len(expenses), "method": "Isolation Forest"}
    with _CACHE_LOCK:
        if len(_ANOMALY_CACHE) >= _CACHE_LIMIT:
            _ANOMALY_CACHE.pop(next(iter(_ANOMALY_CACHE)))
        _ANOMALY_CACHE[cache_key] = result
    return result


def _merge_anomalies(statistical: list[dict], isolation: list[dict]) -> list[dict]:
    merged = {row["transaction_id"]: row for row in statistical}
    for row in isolation:
        existing = merged.get(row["transaction_id"])
        if existing:
            existing["methods"].extend(row["methods"])
            existing["reasons"].extend(row["reasons"])
        else:
            merged[row["transaction_id"]] = row
    return sorted(merged.values(), key=lambda row: (row["date"], Decimal(row["amount"])), reverse=True)


def _recurring_patterns(expenses: pd.DataFrame) -> list[dict]:
    if expenses.empty:
        return []
    rows = []
    normalized = expenses.assign(
        normalized_description=expenses["description"].str.strip().str.casefold(),
        month=expenses["transaction_date"].dt.to_period("M"),
    )
    for (description, category), group in normalized.groupby(["normalized_description", "category"]):
        if not description or len(group) < 3 or group["month"].nunique() < 3:
            continue
        amounts = [Decimal(value) for value in group["amount"]]
        average = _decimal_sum(amounts) / Decimal(len(amounts))
        if average <= ZERO:
            continue
        spread = (max(amounts) - min(amounts)) / average
        if spread <= Decimal("0.05"):
            rows.append({"description": group.iloc[0]["description"], "category": category,
                         "occurrences": len(group), "months": group["month"].nunique(),
                         "average_amount": _amount_string(average)})
    return sorted(rows, key=lambda row: (-row["months"], -row["occurrences"]))[:10]


def _build_patterns(user_id: int, frame: pd.DataFrame, monthly_rows: list[dict], weekly_rows: list[dict],
                    category_rows: list[dict], start_date: date, end_date: date,
                    today: date) -> dict:
    expenses = frame[frame["transaction_type"] == "expense"] if not frame.empty else frame
    category_counts = {}
    if not expenses.empty:
        for category, group in expenses.groupby("category"):
            category_counts[category] = {"transactions": int(len(group)),
                                         "amount": _amount_string(_decimal_sum(group["amount"]))}
    frequent = sorted(
        ({"category": category, **values} for category, values in category_counts.items()),
        key=lambda row: (-row["transactions"], -Decimal(row["amount"])),
    )[:5]

    weekday_total = ZERO
    weekend_total = ZERO
    weekday_count = 0
    weekend_count = 0
    if not expenses.empty:
        for row in expenses.to_dict("records"):
            if row["transaction_date"].day_of_week >= 5:
                weekend_total += row["amount"]
                weekend_count += 1
            else:
                weekday_total += row["amount"]
                weekday_count += 1

    complete_months = []
    for row in monthly_rows:
        period = pd.Period(row["period"], freq="M")
        month_start = period.start_time.date()
        month_end = period.end_time.date()
        if month_start >= start_date and month_end <= end_date and month_end <= today:
            complete_months.append((period, row))
    month_comparison = None
    category_changes = []
    if len(complete_months) >= 2:
        previous_period, previous = complete_months[-2]
        current_period, current = complete_months[-1]
        delta = current["expenses"] - previous["expenses"]
        pct = (delta / previous["expenses"] * Decimal("100")) if previous["expenses"] else None
        month_comparison = {
            "current_month": str(current_period), "previous_month": str(previous_period),
            "current_expenses": _amount_string(current["expenses"]),
            "previous_expenses": _amount_string(previous["expenses"]),
            "change": _amount_string(delta), "change_percentage": None if pct is None else str(pct),
            "direction": "increase" if delta > ZERO else "reduction" if delta < ZERO else "unchanged",
        }
        if not expenses.empty:
            recent_month = expenses[expenses["transaction_date"].dt.to_period("M") == current_period]
            prior_month = expenses[expenses["transaction_date"].dt.to_period("M") == previous_period]
            current_by_category = recent_month.groupby("category")["amount"].agg(_decimal_sum).to_dict()
            previous_by_category = prior_month.groupby("category")["amount"].agg(_decimal_sum).to_dict()
            for category in set(current_by_category) | set(previous_by_category):
                current_amount = current_by_category.get(category, ZERO)
                previous_amount = previous_by_category.get(category, ZERO)
                if previous_amount > ZERO and current_amount > previous_amount:
                    change = (current_amount - previous_amount) / previous_amount * Decimal("100")
                    category_changes.append({"category": category, "current": _amount_string(current_amount),
                                             "previous": _amount_string(previous_amount),
                                             "change_percentage": str(change)})
        category_changes.sort(key=lambda row: Decimal(row["change_percentage"]), reverse=True)

    unusual_weeks = []
    weekly_values = [float(row["expenses"]) for row in weekly_rows]
    if len(weekly_values) >= 4:
        mean = float(np.mean(weekly_values))
        std = float(np.std(weekly_values, ddof=0))
        fence = mean + 2 * std
        unusual_weeks = [{"period": row["period"], "expenses": row["expenses"], "threshold": f"{fence:.2f}"}
                         for row in weekly_rows if float(row["expenses"]) > fence]

    budget_overruns = analyze_budget_patterns(user_id, start_date, end_date)
    return {
        "frequent_categories": frequent,
        "month_over_month": month_comparison,
        "category_changes": category_changes[:5],
        "weekday_vs_weekend": {
            "weekday_expenses": _amount_string(weekday_total),
            "weekend_expenses": _amount_string(weekend_total),
            "weekday_transactions": weekday_count,
            "weekend_transactions": weekend_count,
        },
        "recurring_expenses": _recurring_patterns(expenses),
        "large_transactions": [
            {"date": row["transaction_date"].date().isoformat(), "description": row["description"],
             "category": row["category"], "amount": _amount_string(row["amount"])}
            for row in expenses.sort_values("amount", ascending=False).head(5).to_dict("records")
        ] if not expenses.empty else [],
        "unusual_weeks": unusual_weeks,
        "repeated_budget_overruns": budget_overruns,
        "category_rows": category_rows,
    }


def _financial_insights(totals: dict, patterns: dict, monthly_rows: list[dict],
                        category_rows: list[dict]) -> list[dict]:
    insights = []
    comparison = patterns["month_over_month"]
    if comparison and comparison["direction"] in {"increase", "reduction"}:
        pct = comparison["change_percentage"]
        amount = comparison["change"]
        if pct is None:
            message = f"Spending was {_amount_string(abs(Decimal(amount)))} in {comparison['current_month']}; the previous month had no expenses."
        else:
            message = (f"Expenses {comparison['direction']}d by {Decimal(pct):.1f}% "
                       f"({amount}) in {comparison['current_month']} compared with {comparison['previous_month']}.")
        insights.append({"kind": "month_over_month", "message": message,
                         "metrics": comparison})
    if patterns["category_changes"]:
        change = patterns["category_changes"][0]
        insights.append({"kind": "category_change",
                         "message": f"{change['category']} spending increased {Decimal(change['change_percentage']):.1f}% between complete months.",
                         "metrics": change})
    if totals["savings_rate"] is not None and totals["savings_rate"] < 0:
        insights.append({"kind": "negative_cash_flow",
                         "message": "Expenses exceeded income during the selected period.",
                         "metrics": {"income": _amount_string(totals["income"]),
                                     "expenses": _amount_string(totals["expenses"]),
                                     "savings_rate": str(totals["savings_rate"])}})
    if len(monthly_rows) >= 2:
        rates = []
        for row in monthly_rows:
            rates.append(None if row["income"] == ZERO else (row["net"] / row["income"] * Decimal("100")))
        valid_rates = [rate for rate in rates if rate is not None]
        if len(valid_rates) >= 2 and valid_rates[-1] < valid_rates[-2]:
            insights.append({"kind": "savings_rate_change",
                             "message": f"Savings rate fell from {valid_rates[-2]:.1f}% to {valid_rates[-1]:.1f}% in the latest months with income.",
                             "metrics": {"previous_rate": str(valid_rates[-2]), "current_rate": str(valid_rates[-1])}})
    for week in patterns["unusual_weeks"][:2]:
        insights.append({"kind": "unusual_week",
                         "message": f"Expense total in the week of {week['period']} was above its selected-period baseline.",
                         "metrics": week})
    for budget in patterns["repeated_budget_overruns"]:
        insights.append({"kind": "repeated_budget_overrun",
                         "message": f"{budget['category']} spending exceeded its budget in {budget['overspent_periods']} periods.",
                         "metrics": budget})
    return insights


def analyze_transactions(user_id: int, start_date: date, end_date: date,
                         today: date | None = None) -> dict:
    """Analyze one authenticated user's transactions in a bounded date range."""
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    if (end_date - start_date).days > 3660:
        raise ValueError("Choose a date range of 10 years or less.")
    today = today or date.today()
    records = db.session.execute(
        select(
            Transaction.id,
            Transaction.transaction_date,
            Transaction.amount,
            Transaction.transaction_type,
            Transaction.description,
            Transaction.updated_at,
            Category.name.label("category"),
        )
        .join(Category, Category.id == Transaction.category_id)
        .where(
            Transaction.user_id == user_id,
            Category.user_id == user_id,
            Transaction.transaction_date >= start_date,
            Transaction.transaction_date <= end_date,
        )
        .order_by(Transaction.transaction_date, Transaction.id)
    ).mappings().all()
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        frame = pd.DataFrame(columns=["id", "transaction_date", "amount", "transaction_type",
                                      "description", "updated_at", "category"])
    else:
        frame["transaction_date"] = pd.to_datetime(frame["transaction_date"])
        frame["amount"] = pd.Series([Decimal(value) for value in frame["amount"]], dtype=object)
        frame["description"] = frame["description"].fillna("").astype(str)
        frame["category"] = frame["category"].fillna("Uncategorized").astype(str)

    expenses = frame[frame["transaction_type"] == "expense"].copy()
    income_frame = frame[frame["transaction_type"] == "income"]
    income = _decimal_sum(income_frame["amount"]) if not income_frame.empty else ZERO
    expense_total = _decimal_sum(expenses["amount"]) if not expenses.empty else ZERO
    net = income - expense_total
    savings_rate = (net / income * Decimal("100")) if income else None

    day_count = (end_date - start_date).days + 1
    month_periods = list(pd.period_range(start_date.replace(day=1), end_date, freq="M"))
    year_periods = list(pd.period_range(start_date.replace(month=1, day=1), end_date, freq="Y"))
    if frame.empty:
        frame["day_period"] = pd.Series(dtype="period[D]")
        frame["week_period"] = pd.Series(dtype="period[W-SUN]")
        frame["month_period"] = pd.Series(dtype="period[M]")
        frame["year_period"] = pd.Series(dtype="period[Y-DEC]")
    else:
        frame["day_period"] = frame["transaction_date"].dt.to_period("D")
        frame["week_period"] = frame["transaction_date"].dt.to_period("W-SUN").apply(lambda period: period.start_time.to_period("D"))
        frame["month_period"] = frame["transaction_date"].dt.to_period("M")
        frame["year_period"] = frame["transaction_date"].dt.to_period("Y")
    daily_totals = expenses.groupby(expenses["transaction_date"].dt.to_period("D"))["amount"].agg(_decimal_sum).to_dict() if not expenses.empty else {}
    weekly_totals = expenses.groupby(expenses["transaction_date"].dt.to_period("W-SUN"))["amount"].agg(_decimal_sum).to_dict() if not expenses.empty else {}
    daily_rows = [{"period": period.start_time.date().isoformat(), "expenses": _decimal_sum([amount])}
                  for period, amount in sorted(daily_totals.items())]
    week_periods = list(pd.period_range(start_date, end_date, freq="W-SUN"))
    weekly_rows = [{"period": period.start_time.date().isoformat(),
                    "expenses": weekly_totals.get(period, ZERO)} for period in week_periods]
    monthly_rows = _period_groups(frame, "month_period", month_periods, [str(period) for period in month_periods])
    yearly_rows = _period_groups(frame, "year_period", year_periods, [str(period) for period in year_periods])

    if expenses.empty:
        category_rows = []
    else:
        category_rows = []
        for category, group in expenses.groupby("category"):
            amount = _decimal_sum(group["amount"])
            category_rows.append({"category": category, "expenses": amount,
                                  "transactions": int(len(group))})
        category_rows.sort(key=lambda row: (-row["expenses"], row["category"].casefold()))

    day_averages = expense_total / Decimal(day_count) if day_count else ZERO
    month_count = len(month_periods)
    average_monthly = expense_total / Decimal(month_count) if month_count else ZERO
    totals = {
        "income": income, "expenses": expense_total, "net_cash_flow": net,
        "savings_rate": savings_rate, "average_daily_expenses": day_averages,
        "average_monthly_expenses": average_monthly,
        "highest_spending_category": category_rows[0] if category_rows else None,
        "lowest_spending_category": category_rows[-1] if category_rows else None,
        "days_in_range": day_count,
    }

    statistical_flags, statistical_status = _statistical_anomalies(expenses)
    isolation_flags, isolation_status = _isolation_anomalies(user_id, start_date, end_date, expenses, frame)
    anomalies = _merge_anomalies(statistical_flags, isolation_flags)
    patterns = _build_patterns(user_id, frame, monthly_rows, weekly_rows, category_rows,
                               start_date, end_date, today)
    insights = _financial_insights(totals, patterns, monthly_rows, category_rows)

    chart_data = {
        "monthly_labels": [row["period"] for row in monthly_rows],
        "monthly_income": [str(row["income"]) for row in monthly_rows],
        "monthly_expenses": [str(row["expenses"]) for row in monthly_rows],
        "weekly_labels": [row["period"] for row in weekly_rows],
        "weekly_expenses": [str(row["expenses"]) for row in weekly_rows],
        "category_labels": [row["category"] for row in category_rows],
        "category_expenses": [str(row["expenses"]) for row in category_rows],
        "savings_rate_labels": [row["period"] for row in monthly_rows],
        "savings_rates": [None if row["income"] == ZERO else str(row["net"] / row["income"] * Decimal("100"))
                          for row in monthly_rows],
        "yearly_labels": [row["period"] for row in yearly_rows],
        "yearly_income": [str(row["income"]) for row in yearly_rows],
        "yearly_expenses": [str(row["expenses"]) for row in yearly_rows],
    }
    return {
        "start_date": start_date,
        "end_date": end_date,
        "totals": totals,
        "daily": daily_rows,
        "weekly": weekly_rows,
        "monthly": monthly_rows,
        "yearly": yearly_rows,
        "categories": category_rows,
        "patterns": patterns,
        "anomalies": anomalies,
        "anomaly_methods": {"statistical": statistical_status, "isolation_forest": isolation_status},
        "insights": insights,
        "chart_data": chart_data,
        "transaction_count": len(frame),
    }


def analyze_budget_patterns(user_id: int, start_date: date, end_date: date) -> list[dict]:
    """Identify categories with repeated overspending in the selected range."""
    budgets = db.session.scalars(select(Budget).where(
        Budget.user_id == user_id,
        Budget.period_start >= start_date,
        Budget.period_end <= end_date,
    ))
    overspent_by_category: dict[str, int] = defaultdict(int)
    for budget in budgets:
        if budget_progress(budget)["status"] == "overspent":
            overspent_by_category[budget.category.name] += 1
    return [{"category": category, "overspent_periods": count}
            for category, count in sorted(overspent_by_category.items()) if count >= 2]


def default_date_range(today: date | None = None) -> tuple[date, date]:
    """Return the trailing 12-month analytics range, inclusive of today."""
    today = today or date.today()
    return today - timedelta(days=364), today
