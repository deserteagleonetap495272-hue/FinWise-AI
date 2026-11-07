"""User-scoped and reusable financial data preparation helpers.

Financial aggregations retain ``Decimal`` values. NumPy is used only for
calendar/time features, never to sum or round monetary amounts.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from app.extensions import db
from app.models import Category, Transaction

MONEY_ZERO = Decimal("0.00")
TRANSACTION_COLUMNS = (
    "id", "user_id", "category_id", "amount", "transaction_type",
    "description", "transaction_date", "category", "category_type",
)


def _empty_transactions() -> pd.DataFrame:
    """Return a frame with the stable schema used by this module."""
    return pd.DataFrame({
        "id": pd.Series(dtype="Int64"),
        "user_id": pd.Series(dtype="Int64"),
        "category_id": pd.Series(dtype="Int64"),
        "amount": pd.Series(dtype=object),
        "transaction_type": pd.Series(dtype="string"),
        "description": pd.Series(dtype="string"),
        "transaction_date": pd.Series(dtype="datetime64[ns]"),
        "category": pd.Series(dtype="string"),
        "category_type": pd.Series(dtype="string"),
    })


def fetch_user_transactions(user_id: int) -> pd.DataFrame:
    """Fetch one user's transactions in date order as a normalized DataFrame.

    ``user_id`` must come from trusted application code (normally
    ``current_user.id``). The query independently filters both transaction
    ownership and category ownership; global categories are represented by a
    null category owner in the existing schema and are excluded here because
    the current application assigns user-owned categories to transactions.
    This function is a service, not an HTTP authorization mechanism.
    """
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise ValueError("A valid authenticated user's ID is required.")

    records = db.session.execute(
        select(
            Transaction.id,
            Transaction.user_id,
            Transaction.category_id,
            Transaction.amount,
            Transaction.transaction_type,
            Transaction.description,
            Transaction.transaction_date,
            Category.name.label("category"),
            Category.transaction_type.label("category_type"),
        )
        .join(Category, Category.id == Transaction.category_id)
        .where(
            Transaction.user_id == user_id,
            Category.user_id == user_id,
        )
        .order_by(Transaction.transaction_date, Transaction.id)
    ).mappings().all()
    return clean_transaction_data(pd.DataFrame.from_records(records))


def clean_transaction_data(data: pd.DataFrame | list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize tabular transaction data and safely drop malformed rows.

    Invalid/missing dates, non-positive/non-finite/invalid amounts, unknown
    transaction types, and rows without a valid category label are excluded
    with their monetary data unavailable for modeling. Blank descriptions are
    normalized to an empty string, which allows aggregation while letting text
    models choose to filter them later. Exact duplicate source rows are
    removed; rows with distinct IDs are preserved even when their values match.
    Rejections are summarized in ``DataFrame.attrs`` to keep the model-ready
    schema simple.
    """
    frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    frame = frame.reindex(columns=list(TRANSACTION_COLUMNS))
    if frame.empty:
        result = _empty_transactions()
        result.attrs["rejected_rows"] = 0
        result.attrs["rejection_reasons"] = {}
        return result

    rejected: dict[str, int] = {}
    valid = pd.Series(True, index=frame.index)

    def reject(mask: pd.Series, reason: str) -> None:
        nonlocal valid
        count = int((valid & mask).sum())
        if count:
            rejected[reason] = rejected.get(reason, 0) + count
        valid &= ~mask

    raw_dates = frame["transaction_date"]

    def parse_date(value: Any) -> date | pd.NaT:
        # The persisted model uses SQL DATE, so retain the source calendar day
        # for timestamps instead of shifting it through UTC.
        if isinstance(value, bool) or not isinstance(value, (str, date, datetime, pd.Timestamp)):
            return pd.NaT
        try:
            return pd.Timestamp(value).date()
        except (TypeError, ValueError, OverflowError):
            return pd.NaT

    parsed_dates = pd.to_datetime(raw_dates.map(parse_date), errors="coerce")
    reject(parsed_dates.isna(), "invalid_date")

    amounts: list[Decimal | None] = []
    invalid_amount = []
    for value in frame["amount"]:
        try:
            amount = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            amount = None
        if amount is None or not amount.is_finite() or amount <= 0:
            invalid_amount.append(True)
            amounts.append(None)
        else:
            invalid_amount.append(False)
            amounts.append(amount)
    reject(pd.Series(invalid_amount, index=frame.index), "invalid_amount")

    kinds = frame["transaction_type"].astype("string").str.strip().str.lower()
    reject(~kinds.isin(["income", "expense"]).fillna(False), "invalid_transaction_type")

    labels = frame["category"].astype("string").str.strip()
    label_missing = labels.isna() | labels.eq("")
    reject(label_missing, "missing_category")
    category_types = frame["category_type"].astype("string").str.strip().str.lower()
    category_mismatch = category_types.notna() & category_types.ne("") & category_types.ne(kinds)
    reject(category_mismatch, "category_type_mismatch")

    frame = frame.loc[valid].copy()
    if frame.empty:
        result = _empty_transactions()
        result.attrs["rejected_rows"] = sum(rejected.values())
        result.attrs["rejection_reasons"] = rejected
        return result

    valid_indices = frame.index
    frame["amount"] = pd.Series(
        [amounts[index] for index in valid_indices], index=valid_indices, dtype=object
    )
    frame["transaction_type"] = kinds.loc[valid_indices].astype("string")
    frame["category"] = labels.loc[valid_indices].astype("string")
    frame["description"] = frame["description"].fillna("").astype("string").str.strip()
    frame["transaction_date"] = parsed_dates.loc[valid_indices].dt.normalize()

    # Remove only exact duplicate records. Database-backed records include IDs,
    # so legitimate repeated purchases with separate IDs remain distinct.
    frame = frame.drop_duplicates(subset=list(TRANSACTION_COLUMNS), keep="first")
    frame = frame.sort_values(["transaction_date", "id"], na_position="last", kind="stable")
    frame = frame.reset_index(drop=True)
    result = frame.reindex(columns=list(TRANSACTION_COLUMNS))
    result.attrs["rejected_rows"] = sum(rejected.values())
    result.attrs["rejection_reasons"] = rejected
    return result


def separate_income_expenses(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return independent income and expense frames from cleaned data."""
    frame = clean_transaction_data(data)
    income = frame.loc[frame["transaction_type"].eq("income")].copy().reset_index(drop=True)
    expenses = frame.loc[frame["transaction_type"].eq("expense")].copy().reset_index(drop=True)
    return income, expenses


def prepare_labeled_transactions(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare non-empty descriptions and valid category labels for NLP.

    Preserves one row per source record with columns ``id``, ``description``,
    and ``category``. Blank descriptions and missing/blank labels are omitted.
    Both income and expense records remain available to categorization; callers
    can use ``transaction_type`` if they choose separate category vocabularies.
    """
    frame = clean_transaction_data(data)
    mask = frame["description"].fillna("").str.strip().ne("") & frame["category"].fillna("").str.strip().ne("")
    columns = [column for column in ("id", "description", "category", "transaction_type") if column in frame]
    return frame.loc[mask, columns].copy().reset_index(drop=True)


def _period_range(start: date | datetime | pd.Timestamp,
                  end: date | datetime | pd.Timestamp, frequency: str) -> pd.PeriodIndex:
    """Build inclusive calendar periods without shifting date boundaries."""
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    return pd.period_range(start=start_date, end=end_date, freq=frequency)


def aggregate_expenses_by_month(data: pd.DataFrame, *, start_date: date | None = None,
                                end_date: date | None = None) -> pd.DataFrame:
    """Aggregate expense amounts by calendar month using exact Decimals.

    Optional bounds are inclusive and are applied to individual transaction
    dates before grouping. When either bound is provided, all intersecting
    months are included (zero-filled); omitted bounds use the observed range.
    Income is always excluded.
    """
    _, expenses = separate_income_expenses(data)
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    if start_date is not None:
        expenses = expenses.loc[expenses["transaction_date"].dt.date >= start_date]
    if end_date is not None:
        expenses = expenses.loc[expenses["transaction_date"].dt.date <= end_date]

    if expenses.empty:
        if start_date is None or end_date is None:
            return pd.DataFrame({"period": pd.Series(dtype="period[M]"), "amount": pd.Series(dtype=object), "transaction_count": pd.Series(dtype="int64")})
        periods = _period_range(start_date, end_date, "M")
        return pd.DataFrame({"period": periods, "amount": [MONEY_ZERO] * len(periods), "transaction_count": np.zeros(len(periods), dtype=np.int64)})

    expenses = expenses.copy()
    expenses["period"] = expenses["transaction_date"].dt.to_period("M")
    grouped = expenses.groupby("period", sort=True).agg(
        amount=("amount", lambda values: sum(values, MONEY_ZERO)),
        transaction_count=("amount", "size"),
    )
    first = start_date or expenses["transaction_date"].min().date()
    last = end_date or expenses["transaction_date"].max().date()
    periods = _period_range(first, last, "M")
    grouped = grouped.reindex(periods)
    grouped["amount"] = grouped["amount"].map(lambda value: MONEY_ZERO if pd.isna(value) else value).astype(object)
    grouped["transaction_count"] = grouped["transaction_count"].fillna(0).astype(np.int64)
    grouped.index.name = "period"
    return grouped.reset_index()


def aggregate_expenses_by_day(data: pd.DataFrame, *, start_date: date | None = None,
                              end_date: date | None = None) -> pd.DataFrame:
    """Aggregate expense amounts by calendar date using exact Decimals.

    Bounds are inclusive and applied to individual dates before aggregation.
    If both bounds are provided, days without expenses are included with zero
    amounts. Income is always excluded.
    """
    _, expenses = separate_income_expenses(data)
    if start_date is not None and end_date is not None and start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    if start_date is not None:
        expenses = expenses.loc[expenses["transaction_date"].dt.date >= start_date]
    if end_date is not None:
        expenses = expenses.loc[expenses["transaction_date"].dt.date <= end_date]

    if expenses.empty:
        if start_date is None or end_date is None:
            return pd.DataFrame({"period": pd.Series(dtype="datetime64[ns]"), "amount": pd.Series(dtype=object), "transaction_count": pd.Series(dtype="int64")})
        periods = pd.date_range(start_date, end_date, freq="D")
        return pd.DataFrame({"period": periods, "amount": [MONEY_ZERO] * len(periods), "transaction_count": np.zeros(len(periods), dtype=np.int64)})

    expenses = expenses.copy()
    expenses["period"] = expenses["transaction_date"].dt.normalize()
    grouped = expenses.groupby("period", sort=True).agg(
        amount=("amount", lambda values: sum(values, MONEY_ZERO)),
        transaction_count=("amount", "size"),
    )
    first = pd.Timestamp(start_date) if start_date is not None else expenses["period"].min()
    last = pd.Timestamp(end_date) if end_date is not None else expenses["period"].max()
    periods = pd.date_range(first, last, freq="D")
    grouped = grouped.reindex(periods)
    grouped["amount"] = grouped["amount"].map(lambda value: MONEY_ZERO if pd.isna(value) else value).astype(object)
    grouped["transaction_count"] = grouped["transaction_count"].fillna(0).astype(np.int64)
    grouped.index.name = "period"
    return grouped.reset_index()


def generate_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe calendar features derived only from each row's date.

    The output preserves source dates and adds year, month, quarter, day,
    weekday, weekend, month-start/end, and cyclic month encodings. No target,
    future date, or transaction amount is used to derive these features.
    """
    frame = clean_transaction_data(data)
    if frame.empty:
        for name, dtype in (("year", "Int64"), ("month", "Int64"), ("quarter", "Int64"),
                            ("day", "Int64"), ("day_of_week", "Int64"), ("is_weekend", "bool"),
                            ("is_month_start", "bool"), ("is_month_end", "bool"),
                            ("month_sin", "float64"), ("month_cos", "float64")):
            frame[name] = pd.Series(dtype=dtype)
        return frame

    dates = frame["transaction_date"]
    month = dates.dt.month.astype(np.int64)
    weekday = dates.dt.day_of_week.astype(np.int64)
    frame["year"] = dates.dt.year.astype(np.int64)
    frame["month"] = month
    frame["quarter"] = dates.dt.quarter.astype(np.int64)
    frame["day"] = dates.dt.day.astype(np.int64)
    frame["day_of_week"] = weekday
    frame["is_weekend"] = weekday.ge(5)
    frame["is_month_start"] = dates.dt.is_month_start
    frame["is_month_end"] = dates.dt.is_month_end
    frame["month_sin"] = np.sin(2 * np.pi * month / 12)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12)
    return frame


def prepare_expense_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned expense-only rows with safe calendar features.

    This establishes the modeling boundary for later prediction stages:
    income is excluded before any expense training dataset is returned.
    """
    _, expenses = separate_income_expenses(data)
    return generate_time_features(expenses)
