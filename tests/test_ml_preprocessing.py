"""Tests for user-scoped, leakage-safe ML preprocessing."""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from app.extensions import db
from app.models import Category, Transaction, User
from app.services.ml.data_preprocessing import (
    aggregate_expenses_by_day,
    aggregate_expenses_by_month,
    clean_transaction_data,
    fetch_user_transactions,
    generate_time_features,
    prepare_expense_training_data,
    prepare_labeled_transactions,
    separate_income_expenses,
)


def create_owner(app, email):
    with app.app_context():
        user = User(email=email, full_name="ML Test", password_hash="unused")
        db.session.add(user)
        db.session.flush()
        salary = Category(user_id=user.id, name="Salary", transaction_type="income")
        food = Category(user_id=user.id, name="Food", transaction_type="expense")
        db.session.add_all([salary, food])
        db.session.flush()
        result = user.id, salary.id, food.id
        db.session.commit()
        return result


def add_transaction(app, user_id, category_id, kind, amount, day, description=""):
    with app.app_context():
        db.session.add(Transaction(
            user_id=user_id,
            category_id=category_id,
            transaction_type=kind,
            amount=Decimal(str(amount)),
            transaction_date=day,
            description=description,
        ))
        db.session.commit()


@pytest.fixture
def owners(app):
    owner_a = create_owner(app, "preprocess-a@example.com")
    owner_b = create_owner(app, "preprocess-b@example.com")
    return owner_a, owner_b


def test_empty_history_has_stable_schema_and_empty_aggregates(app, owners):
    user_id, _, _ = owners[0]
    with app.app_context():
        frame = fetch_user_transactions(user_id)
    assert frame.empty
    assert "transaction_date" in frame.columns
    assert prepare_expense_training_data(frame).empty
    assert aggregate_expenses_by_month(frame).empty
    assert aggregate_expenses_by_day(frame).empty


def test_fetch_is_scoped_to_the_requested_user_and_validates_identity(app, owners):
    (user_a, _, food_a), (user_b, _, food_b) = owners
    add_transaction(app, user_a, food_a, "expense", "12.30", date(2026, 1, 4), "A purchase")
    add_transaction(app, user_b, food_b, "expense", "999.99", date(2026, 1, 4), "B purchase")
    with app.app_context():
        rows = fetch_user_transactions(user_a)
        assert rows["user_id"].tolist() == [user_a]
        assert rows["description"].tolist() == ["A purchase"]
        for invalid_id in (None, 0, -1, True, "1"):
            with pytest.raises(ValueError, match="authenticated"):
                fetch_user_transactions(invalid_id)


def test_income_and_expenses_are_separated_and_training_excludes_income(app, owners):
    (user_id, salary_id, food_id), _ = owners
    add_transaction(app, user_id, salary_id, "income", "2500.00", date(2026, 2, 1), "Pay")
    add_transaction(app, user_id, food_id, "expense", "15.75", date(2026, 2, 2), "Lunch")
    with app.app_context():
        rows = fetch_user_transactions(user_id)
    income, expenses = separate_income_expenses(rows)
    training = prepare_expense_training_data(rows)
    assert income["transaction_type"].tolist() == ["income"]
    assert expenses["transaction_type"].tolist() == ["expense"]
    assert set(training["transaction_type"]) == {"expense"}
    assert training["amount"].tolist() == [Decimal("15.75")]


@pytest.fixture
def transaction_frame():
    return clean_transaction_data(pd.DataFrame([
        {"id": 1, "user_id": 7, "category_id": 2, "amount": "10.10", "transaction_type": "expense",
         "description": "  Coffee  ", "transaction_date": "2026-01-31", "category": "Food", "category_type": "expense"},
        {"id": 2, "user_id": 7, "category_id": 2, "amount": "2.20", "transaction_type": "expense",
         "description": "Tea", "transaction_date": "2026-02-01", "category": "Food", "category_type": "expense"},
        {"id": 3, "user_id": 7, "category_id": 1, "amount": "1000.00", "transaction_type": "income",
         "description": "Pay", "transaction_date": "2026-02-01", "category": "Salary", "category_type": "income"},
        {"id": 4, "user_id": 7, "category_id": 2, "amount": "3.33", "transaction_type": "expense",
         "description": "Snack", "transaction_date": "2026-02-03", "category": "Food", "category_type": "expense"},
    ]))


def test_monthly_aggregation_is_exact_income_free_and_zero_fills(transaction_frame):
    monthly = aggregate_expenses_by_month(
        transaction_frame, start_date=date(2026, 1, 1), end_date=date(2026, 3, 31)
    )
    assert monthly["period"].astype(str).tolist() == ["2026-01", "2026-02", "2026-03"]
    assert monthly["amount"].tolist() == [Decimal("10.10"), Decimal("5.53"), Decimal("0.00")]
    assert monthly["transaction_count"].tolist() == [1, 2, 0]


def test_daily_aggregation_uses_inclusive_dates_and_preserves_decimal(transaction_frame):
    daily = aggregate_expenses_by_day(
        transaction_frame, start_date=date(2026, 2, 1), end_date=date(2026, 2, 3)
    )
    assert daily["period"].dt.date.tolist() == [date(2026, 2, day) for day in (1, 2, 3)]
    assert daily["amount"].tolist() == [Decimal("2.20"), Decimal("0.00"), Decimal("3.33")]
    assert daily["transaction_count"].tolist() == [1, 0, 1]


def test_cleaning_normalizes_missing_descriptions_and_filters_missing_labels():
    rows = pd.DataFrame([
        {"id": 1, "amount": "1.25", "transaction_type": "EXPENSE", "description": None,
         "transaction_date": date(2026, 4, 5), "category": " Food "},
        {"id": 2, "amount": "2.00", "transaction_type": "expense", "description": "No label",
         "transaction_date": date(2026, 4, 6), "category": None},
    ])
    cleaned = clean_transaction_data(rows)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["description"] == ""
    assert cleaned.iloc[0]["category"] == "Food"
    assert cleaned.attrs["rejection_reasons"] == {"missing_category": 1}
    assert prepare_labeled_transactions(cleaned).empty


def test_labeled_data_requires_description_and_category(transaction_frame):
    labeled = prepare_labeled_transactions(transaction_frame)
    assert labeled["description"].tolist() == ["Coffee", "Tea", "Pay", "Snack"]
    assert labeled["category"].tolist() == ["Food", "Food", "Salary", "Food"]


def test_dates_are_normalized_sorted_and_bad_dates_rejected():
    cleaned = clean_transaction_data(pd.DataFrame([
        {"id": 1, "amount": "1", "transaction_type": "expense", "description": "late",
         "transaction_date": "2026-07-02T19:30:00+05:30", "category": "Food"},
        {"id": 2, "amount": "1", "transaction_type": "expense", "description": "bad",
         "transaction_date": "not-a-date", "category": "Food"},
        {"id": 3, "amount": "1", "transaction_type": "expense", "description": "early",
         "transaction_date": "2026-07-01", "category": "Food"},
    ]))
    assert cleaned["description"].tolist() == ["early", "late"]
    assert cleaned["transaction_date"].dt.date.tolist() == [date(2026, 7, 1), date(2026, 7, 2)]
    assert cleaned.attrs["rejection_reasons"] == {"invalid_date": 1}


def test_time_features_derive_only_from_transaction_dates(transaction_frame):
    featured = generate_time_features(transaction_frame)
    first = featured.iloc[0]
    assert (first["year"], first["month"], first["quarter"], first["day"]) == (2026, 1, 1, 31)
    assert first["is_month_end"]
    assert first["is_weekend"]
    assert featured["month_sin"].between(-1, 1).all()
    assert featured["month_cos"].between(-1, 1).all()
    assert "amount" in featured and featured["amount"].tolist() == transaction_frame["amount"].tolist()


def test_malformed_amounts_types_and_duplicates_are_handled():
    rows = pd.DataFrame([
        {"id": 1, "amount": "NaN", "transaction_type": "expense", "description": "bad amount",
         "transaction_date": "2026-01-01", "category": "Food"},
        {"id": 2, "amount": "Infinity", "transaction_type": "expense", "description": "bad amount",
         "transaction_date": "2026-01-01", "category": "Food"},
        {"id": 3, "amount": "-5", "transaction_type": "expense", "description": "bad amount",
         "transaction_date": "2026-01-01", "category": "Food"},
        {"id": 4, "amount": "3.40", "transaction_type": "transfer", "description": "bad type",
         "transaction_date": "2026-01-01", "category": "Food"},
        {"id": 5, "amount": "3.40", "transaction_type": "expense", "description": "good",
         "transaction_date": "2026-01-01", "category": "Food"},
        {"id": 5, "amount": "3.40", "transaction_type": "expense", "description": "good",
         "transaction_date": "2026-01-01", "category": "Food"},
    ])
    cleaned = clean_transaction_data(rows)
    assert cleaned["id"].tolist() == [5]
    assert cleaned["amount"].tolist() == [Decimal("3.40")]
    assert cleaned.attrs["rejection_reasons"] == {"invalid_amount": 3, "invalid_transaction_type": 1}


def test_invalid_aggregation_bounds_raise_clear_error(transaction_frame):
    with pytest.raises(ValueError, match="on or before"):
        aggregate_expenses_by_month(transaction_frame, start_date=date(2026, 3, 1), end_date=date(2026, 2, 1))
    with pytest.raises(ValueError, match="on or before"):
        aggregate_expenses_by_day(transaction_frame, start_date=date(2026, 3, 1), end_date=date(2026, 2, 1))
