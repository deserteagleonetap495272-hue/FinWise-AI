"""Tests for descriptive analytics, anomaly detection, and user isolation."""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from app.extensions import db
from app.models import Category, Transaction, User
from app.services.analytics import analyze_transactions


def add_user(app, email):
    with app.app_context():
        user = User(email=email, full_name=email.split("@")[0], password_hash="unused")
        db.session.add(user)
        db.session.flush()
        categories = {
            ("Salary", "income"): Category(user_id=user.id, name="Salary", transaction_type="income"),
            ("Food", "expense"): Category(user_id=user.id, name="Food", transaction_type="expense"),
            ("Transport", "expense"): Category(user_id=user.id, name="Transport", transaction_type="expense"),
        }
        db.session.add_all(categories.values())
        db.session.flush()
        result = user.id, {key: category.id for key, category in categories.items()}
        db.session.commit()
        return result


def add_record(app, user_id, category_id, amount, kind, day, description="record"):
    with app.app_context():
        db.session.add(Transaction(
            user_id=user_id, category_id=category_id, amount=Decimal(str(amount)),
            transaction_type=kind, transaction_date=day, description=description,
        ))
        db.session.commit()


def test_monthly_weekly_category_and_savings_rate_analytics(app):
    user_id, categories = add_user(app, "analytics@example.com")
    rows = [
        (categories[("Salary", "income")], "1000.00", "income", date(2026, 1, 5), "salary"),
        (categories[("Food", "expense")], "100.10", "expense", date(2026, 1, 6), "food 1"),
        (categories[("Transport", "expense")], "50.20", "expense", date(2026, 1, 10), "transport 1"),
        (categories[("Salary", "income")], "1000.00", "income", date(2026, 2, 5), "salary"),
        (categories[("Food", "expense")], "300.30", "expense", date(2026, 2, 6), "food 2"),
        (categories[("Transport", "expense")], "40.40", "expense", date(2026, 2, 14), "transport 2"),
    ]
    for category_id, amount, kind, day, description in rows:
        add_record(app, user_id, category_id, amount, kind, day, description)
    with app.app_context():
        result = analyze_transactions(user_id, date(2026, 1, 1), date(2026, 2, 28), today=date(2026, 3, 20))
    assert result["totals"]["income"] == Decimal("2000.00")
    assert result["totals"]["expenses"] == Decimal("491.00")
    assert result["totals"]["net_cash_flow"] == Decimal("1509.00")
    assert result["totals"]["savings_rate"] == Decimal("75.45")
    assert result["totals"]["highest_spending_category"]["category"] == "Food"
    assert result["monthly"][1]["expenses"] == Decimal("340.70")
    assert result["patterns"]["month_over_month"]["direction"] == "increase"
    assert result["patterns"]["weekday_vs_weekend"]["weekend_transactions"] == 2
    assert result["chart_data"]["monthly_labels"] == ["2026-01", "2026-02"]
    assert result["weekly"]


def test_date_range_filtering_and_cross_user_isolation(app):
    user_a, categories_a = add_user(app, "analytics-a@example.com")
    user_b, categories_b = add_user(app, "analytics-b@example.com")
    add_record(app, user_a, categories_a[("Food", "expense")], "20.00", "expense", date(2026, 4, 5))
    add_record(app, user_a, categories_a[("Food", "expense")], "30.00", "expense", date(2026, 5, 5))
    add_record(app, user_b, categories_b[("Food", "expense")], "999999.00", "expense", date(2026, 5, 5))
    with app.app_context():
        result = analyze_transactions(user_a, date(2026, 5, 1), date(2026, 5, 31), today=date(2026, 5, 31))
    assert result["totals"]["expenses"] == Decimal("30.00")
    assert result["transaction_count"] == 1
    assert result["categories"][0]["expenses"] == Decimal("30.00")


def test_empty_range_and_zero_income_are_safe(app):
    user_id, _ = add_user(app, "empty-analytics@example.com")
    with app.app_context():
        empty = analyze_transactions(user_id, date(2026, 1, 1), date(2026, 1, 31), today=date(2026, 1, 31))
    assert empty["totals"]["income"] == Decimal("0.00")
    assert empty["totals"]["expenses"] == Decimal("0.00")
    assert empty["totals"]["savings_rate"] is None
    assert empty["categories"] == []
    assert empty["anomalies"] == []
    assert empty["anomaly_methods"]["statistical"]["status"] == "insufficient_data"
    assert empty["anomaly_methods"]["isolation_forest"]["status"] == "insufficient_data"


def test_statistical_anomaly_flags_large_transaction_with_explanation(app):
    user_id, categories = add_user(app, "anomaly@example.com")
    for index in range(8):
        add_record(app, user_id, categories[("Food", "expense")], "100.00", "expense",
                   date(2026, 1, 1) + timedelta(days=index), f"usual {index}")
    add_record(app, user_id, categories[("Food", "expense")], "10000.00", "expense",
               date(2026, 1, 12), "large purchase")
    with app.app_context():
        result = analyze_transactions(user_id, date(2026, 1, 1), date(2026, 1, 31), today=date(2026, 1, 31))
    flag = next(row for row in result["anomalies"] if row["description"] == "large purchase")
    assert "statistical" in flag["methods"]
    assert any("IQR" in reason for reason in flag["reasons"])
    assert "fraud" not in flag["label"].lower()
    assert result["anomaly_methods"]["isolation_forest"]["status"] == "insufficient_data"


def test_isolation_forest_runs_only_with_sufficient_data(app):
    user_id, categories = add_user(app, "isolation@example.com")
    for index in range(24):
        amount = "10000.00" if index == 23 else str(40 + index % 4)
        add_record(app, user_id, categories[("Food", "expense")], amount, "expense",
                   date(2026, 1, 1) + timedelta(days=index), f"expense {index}")
    with app.app_context():
        result = analyze_transactions(user_id, date(2026, 1, 1), date(2026, 2, 1), today=date(2026, 2, 1))
    assert result["anomaly_methods"]["isolation_forest"]["status"] == "analyzed"
    assert result["anomaly_methods"]["isolation_forest"]["records_analyzed"] == 24


def test_invalid_date_ranges_are_rejected(app):
    user_id, _ = add_user(app, "invalid-analytics@example.com")
    with app.app_context(), pytest.raises(ValueError, match="on or before"):
        analyze_transactions(user_id, date(2026, 2, 1), date(2026, 1, 1))
    with app.app_context(), pytest.raises(ValueError, match="10 years"):
        analyze_transactions(user_id, date(2000, 1, 1), date(2026, 1, 1))


def test_protected_route_validates_dates_and_renders_empty_state(client):
    password = "SafePassphrase123"
    client.post("/register", data={"full_name": "Analytics User", "email": "route-analytics@example.com",
                                   "password": password, "confirm_password": password})
    assert client.get("/analytics").status_code == 302
    client.post("/login", data={"email": "route-analytics@example.com", "password": password})
    page = client.get("/analytics")
    assert page.status_code == 200
    assert b"No transactions in this date range" in page.data
    assert b"Financial analytics" in page.data
    invalid = client.get("/analytics?start_date=not-a-date&period=invalid")
    assert invalid.status_code == 200
    assert b"must be a valid date" in invalid.data
    assert b"Choose monthly or yearly" in invalid.data
