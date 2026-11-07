"""Stage 9 validation using reproducible fictional INR ledger data."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from decimal import Decimal
import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.extensions import db
from app.models import Budget, Category, GoalContribution, Recommendation, SavingsGoal, Transaction, User
from app.services.analytics import analyze_transactions
from app.services.budgets import budget_progress, ensure_budget_alerts
from app.services.ml.data_preprocessing import aggregate_expenses_by_month, fetch_user_transactions
from app.services.ml.expense_prediction import build_monthly_features, clip_nonnegative_money, predict_next_month
from app.services.ml.forecasting import forecast_expenses
from app.services.ml.transaction_categorization import train_categorizer, predict_category
from app.services.savings import goal_progress
from app.services.transactions import financial_summary, format_money
from scripts.synthetic_financial_data import build_dataset

ZERO = Decimal("0.00")


def seed_synthetic_dataset(app):
    """Load only deterministic fictional records into the test app's SQLite DB."""
    dataset = build_dataset()
    users: dict[str, User] = {}
    categories: dict[tuple[str, str, str], Category] = {}
    for user_row in dataset.users:
        user = User(email=user_row["email"], full_name=user_row["full_name"], currency="INR")
        user.set_password("SyntheticTestPassword123")
        db.session.add(user)
        users[user_row["key"]] = user
    db.session.flush()

    for row in dataset.transactions:
        key = (row["user_key"], row["category"], row["transaction_type"])
        if key not in categories:
            category = Category(user_id=users[row["user_key"]].id, name=row["category"],
                                transaction_type=row["category_type"])
            db.session.add(category)
            categories[key] = category
    db.session.flush()

    transactions = []
    for row in dataset.transactions:
        transaction = Transaction(
            user_id=users[row["user_key"]].id,
            category_id=categories[(row["user_key"], row["category"], row["transaction_type"])].id,
            amount=row["amount"], transaction_type=row["transaction_type"],
            description=row["description"], transaction_date=row["transaction_date"],
        )
        db.session.add(transaction)
        transactions.append(transaction)

    goals: dict[str, SavingsGoal] = {}
    for contribution in dataset.contributions:
        key = contribution["user_key"]
        if key not in goals:
            goal = SavingsGoal(user_id=users[key].id, name=contribution["goal_name"],
                               target_amount=contribution["target_amount"])
            db.session.add(goal)
            goals[key] = goal
    db.session.flush()
    for row in dataset.contributions:
        db.session.add(GoalContribution(
            goal_id=goals[row["user_key"]].id, amount=row["amount"],
            contributed_on=row["contributed_on"], note=row["note"],
        ))
    db.session.commit()
    return dataset, users, categories, goals


def _expected_total(rows, *, user_key: str, kind: str, start=None, end=None):
    selected = [row["amount"] for row in rows
                if row["user_key"] == user_key and row["transaction_type"] == kind
                and (start is None or row["transaction_date"] >= start)
                and (end is None or row["transaction_date"] <= end)]
    return sum(selected, ZERO)


def test_synthetic_fixture_is_reproducible_and_contains_no_real_identifiers():
    first, second = build_dataset(), build_dataset()
    assert first == second
    assert len(first.transactions) == 481
    assert len(first.contributions) == 60
    assert {row["user_key"] for row in first.transactions} == {"primary", "secondary", "scenario"}
    assert all(row["email"].endswith("@example.test") for row in first.users)
    assert all(isinstance(row["amount"], Decimal) and row["amount"].as_tuple().exponent == -2
               for row in first.transactions)


def test_database_and_analytics_reconcile_exact_totals_and_inclusive_dates(app):
    dataset, users, _, _ = seed_synthetic_dataset(app)
    primary = users["primary"]
    expected_income = _expected_total(dataset.transactions, user_key="primary", kind="income")
    expected_expense = _expected_total(dataset.transactions, user_key="primary", kind="expense")

    summary = financial_summary(primary.id, today=date(2025, 3, 31))
    assert summary["total_income"] == expected_income
    assert summary["total_expenses"] == expected_expense
    assert summary["balance"] == expected_income - expected_expense
    march = [row for row in dataset.transactions if row["user_key"] == "primary"
             and row["transaction_date"].year == 2025 and row["transaction_date"].month == 3]
    assert summary["monthly_income"] == sum((r["amount"] for r in march if r["transaction_type"] == "income"), ZERO)
    assert summary["monthly_expenses"] == sum((r["amount"] for r in march if r["transaction_type"] == "expense"), ZERO)

    start, end = date(2025, 3, 1), date(2025, 3, 31)
    report = analyze_transactions(primary.id, start, end, today=end)
    assert report["totals"]["income"] == _expected_total(
        dataset.transactions, user_key="primary", kind="income", start=start, end=end)
    assert report["totals"]["expenses"] == _expected_total(
        dataset.transactions, user_key="primary", kind="expense", start=start, end=end)
    assert report["totals"]["net_cash_flow"] == report["totals"]["income"] - report["totals"]["expenses"]
    assert report["monthly"][0]["period"] == "2025-03"
    assert report["monthly"][0]["expenses"] == report["totals"]["expenses"]
    expected_categories = defaultdict(lambda: ZERO)
    for row in dataset.transactions:
        if row["user_key"] == "primary" and row["transaction_type"] == "expense" \
                and start <= row["transaction_date"] <= end:
            expected_categories[row["category"]] += row["amount"]
    assert {row["category"]: row["expenses"] for row in report["categories"]} == dict(expected_categories)


def test_ml_preprocessing_is_user_scoped_and_monthly_amounts_match_ledger(app):
    dataset, users, _, _ = seed_synthetic_dataset(app)
    primary = fetch_user_transactions(users["primary"].id)
    secondary = fetch_user_transactions(users["secondary"].id)
    assert len(primary) == sum(row["user_key"] == "primary" for row in dataset.transactions)
    assert len(secondary) == sum(row["user_key"] == "secondary" for row in dataset.transactions)
    assert set(primary["user_id"]) == {users["primary"].id}
    assert set(secondary["user_id"]) == {users["secondary"].id}
    assert not set(primary["id"]) & set(secondary["id"])

    monthly = aggregate_expenses_by_month(primary)
    expected_by_month: dict[str, Decimal] = defaultdict(lambda: ZERO)
    for row in dataset.transactions:
        if row["user_key"] == "primary" and row["transaction_type"] == "expense":
            expected_by_month[row["transaction_date"].strftime("%Y-%m")] += row["amount"]
    actual_by_month = {str(row.period): row.amount for row in monthly.itertuples(index=False)}
    assert actual_by_month == dict(expected_by_month)
    assert list(actual_by_month) == sorted(actual_by_month)


def test_budget_savings_and_inr_dashboard_values_reconcile(app):
    dataset, users, categories, goals = seed_synthetic_dataset(app)
    primary = users["primary"]
    rent = categories[("primary", "Rent", "expense")]
    budget = Budget(user_id=primary.id, category_id=rent.id, amount=Decimal("18000.00"),
                    name="Synthetic rent", period_start=date(2025, 3, 1), period_end=date(2025, 3, 31))
    db.session.add(budget)
    db.session.commit()
    progress = budget_progress(budget, today=date(2025, 3, 15))
    rent_amounts = [row["amount"] for row in dataset.transactions
                    if row["user_key"] == "primary" and row["category"] == "Rent"
                    and row["transaction_date"].month == 3 and row["transaction_date"].year == 2025]
    expected_rent = sum(rent_amounts, ZERO)
    assert progress["spent"] == expected_rent == Decimal("18000.00")
    assert progress["remaining"] == ZERO
    assert progress["usage"] == Decimal("100")
    assert progress["status"] == "critical"
    ensure_budget_alerts(primary.id, today=date(2025, 3, 15))
    ensure_budget_alerts(primary.id, today=date(2025, 3, 15))
    alerts = list(db.session.scalars(select(Recommendation).where(
        Recommendation.user_id == primary.id, Recommendation.dedupe_key.like(f"budget:{budget.id}:%")
    )))
    assert len(alerts) == 2

    food = categories[("primary", "Food delivery", "expense")]
    overspent_budget = Budget(user_id=primary.id, category_id=food.id, amount=Decimal("10000.00"),
                              name="Synthetic food", period_start=date(2025, 3, 1),
                              period_end=date(2025, 3, 31))
    db.session.add(overspent_budget)
    db.session.commit()
    overspent = budget_progress(overspent_budget, today=date(2025, 3, 15))
    actual_food = sum((row["amount"] for row in dataset.transactions if row["user_key"] == "primary"
                       and row["category"] == "Food delivery" and row["transaction_date"].year == 2025
                       and row["transaction_date"].month == 3), ZERO)
    assert actual_food == Decimal("19023.50")
    assert overspent["spent"] == actual_food
    assert overspent["remaining"] == Decimal("-9023.50")
    assert overspent["usage"] == Decimal("190.235")
    assert overspent["status"] == "overspent"
    ensure_budget_alerts(primary.id, today=date(2025, 3, 15))
    ensure_budget_alerts(primary.id, today=date(2025, 3, 15))
    over_budget_alerts = list(db.session.scalars(select(Recommendation).where(
        Recommendation.user_id == primary.id,
        Recommendation.dedupe_key.like(f"budget:{overspent_budget.id}:%"),
    )))
    assert len(over_budget_alerts) == 3

    goal = goals["primary"]
    goal_summary = goal_progress(goal, today=date(2025, 12, 31))
    assert goal_summary["saved"] == Decimal("60000.00")
    assert goal_summary["remaining"] == ZERO
    assert goal_summary["status"] == "completed"
    history = list(db.session.scalars(select(GoalContribution).where(GoalContribution.goal_id == goal.id)
                                      .order_by(GoalContribution.contributed_on)))
    assert len(history) == 24
    assert sum((row.amount for row in history), ZERO) == goal_summary["saved"]
    assert format_money(Decimal("123456.78"), "INR") == "₹ 123,456.78"


def test_exact_documented_savings_scenario_reconciles(app):
    dataset, users, _, goals = seed_synthetic_dataset(app)
    user = users["scenario"]
    summary = financial_summary(user.id, today=date(2024, 12, 31))
    assert summary["total_income"] == Decimal("600000.00")
    assert summary["total_expenses"] == Decimal("360000.00")
    assert summary["balance"] == Decimal("240000.00")
    assert summary["monthly_income"] == Decimal("50000.00")
    assert summary["monthly_expenses"] == Decimal("30000.00")
    assert summary["monthly_savings"] == Decimal("20000.00")
    assert goal_progress(goals["scenario"], today=date(2024, 12, 31)) == {
        "saved": Decimal("60000.00"), "remaining": ZERO, "percentage": Decimal("100"),
        "status": "completed",
    }
    assert len([row for row in dataset.transactions if row["user_key"] == "scenario"]) == 36


def test_expense_models_are_chronological_and_forecast_labels_actuals(app):
    dataset, users, _, _ = seed_synthetic_dataset(app)
    frame = fetch_user_transactions(users["primary"].id)
    result = predict_next_month(frame)
    assert result["historical_months"] == 24
    assert result["data_sufficient"] is True
    assert result["predicted_expenses"] >= ZERO
    assert clip_nonnegative_money(Decimal("-123.456")) == ZERO
    assert result["prediction_period"] == date(2026, 1, 1)
    history = result["monthly_history"]
    assert list(history["period"].astype(str)) == sorted(history["period"].astype(str))

    supervised = build_monthly_features(history)
    expected_holdout = [str(period) for period in supervised.iloc[result["training_observations"]:]["period"]]
    assert result["test_periods"] == expected_holdout
    assert all(period <= "2025-12" for period in result["test_periods"])
    assert result["baseline_metrics"]["mae"] is not None
    if result["model_based"]:
        selected = next(row for row in result["model_comparison"] if row["model"] == result["selected_model"])
        assert selected["mae"] <= result["baseline_metrics"]["mae"] * 0.95
        assert result["metrics"] == selected
    else:
        assert result["selected_model"] == "Naive baseline (previous month)"

    forecast = forecast_expenses(frame, horizon=3, prediction_result=result)
    assert len(forecast["forecasts"]) == 3
    assert all(row["value_type"] == "predicted" and row["amount"] >= ZERO
               for row in forecast["forecasts"])
    assert all(row["value_type"] == "actual" for row in forecast["historical_monthly_expenses"])


def test_anomaly_review_finds_known_spikes_and_does_not_flag_recurring_rent(app):
    dataset, users, _, _ = seed_synthetic_dataset(app)
    result = analyze_transactions(users["primary"].id, date(2024, 1, 1), date(2025, 12, 31),
                                  today=date(2025, 12, 31))
    flagged = {row["transaction_id"]: row for row in result["anomalies"]}
    injected = [row for row in dataset.transactions if row["user_key"] == "primary" and row["description"] in {
        "one time laptop purchase electronics store", "hospital diagnostic and treatment invoice",
        "family celebration catering and dinner order",
    }]
    assert len(injected) == 3
    assert all(row["id"] in flagged for row in injected)
    rents = [row for row in dataset.transactions if row["user_key"] == "primary" and row["category"] == "Rent"]
    assert len(rents) == 24
    assert not any(row["id"] in flagged for row in rents)
    assert all(row["label"] == "Unusual spending · review" for row in result["anomalies"])
    assert all("fraud" not in " ".join(row["reasons"]).lower() for row in result["anomalies"])
    statistical = [row for row in result["anomalies"] if "statistical" in row["methods"]]
    assert {row["transaction_id"] for row in statistical} == {row["id"] for row in injected}


def test_short_history_and_empty_owner_report_insufficient_data(app):
    _, users, _, _ = seed_synthetic_dataset(app)
    frame = fetch_user_transactions(users["primary"].id)
    short = frame[frame["transaction_date"] < pd.Timestamp("2024-09-01")]
    result = predict_next_month(short)
    assert result["data_sufficient"] is False
    assert result["model_based"] is False
    assert result["selected_model"] == "Naive baseline (previous month)"
    assert result["metrics"] is None
    empty = fetch_user_transactions(999999)
    empty_result = predict_next_month(empty)
    assert empty_result["status"] == "insufficient_data"
    assert empty_result["predicted_expenses"] is None
    empty_summary = financial_summary(999999, today=date(2025, 1, 15))
    assert empty_summary["total_income"] == empty_summary["total_expenses"] == ZERO


def test_income_only_and_expense_only_users_have_exact_zero_boundaries(app):
    income_user = User(email="income.only@example.test", full_name="Income Only", currency="INR")
    income_user.set_password("SyntheticTestPassword123")
    expense_user = User(email="expense.only@example.test", full_name="Expense Only", currency="INR")
    expense_user.set_password("SyntheticTestPassword123")
    db.session.add_all([income_user, expense_user])
    db.session.flush()
    salary = Category(user_id=income_user.id, name="Salary", transaction_type="income")
    food = Category(user_id=expense_user.id, name="Food", transaction_type="expense")
    db.session.add_all([salary, food])
    db.session.flush()
    db.session.add_all([
        Transaction(user_id=income_user.id, category_id=salary.id, amount=Decimal("123456789.01"),
                    transaction_type="income", transaction_date=date(2025, 1, 1), description="first day income"),
        Transaction(user_id=expense_user.id, category_id=food.id, amount=Decimal("9999999999.99"),
                    transaction_type="expense", transaction_date=date(2025, 1, 31), description="last day expense"),
    ])
    db.session.commit()
    db.session.add(Transaction(user_id=expense_user.id, category_id=food.id, amount=ZERO,
                               transaction_type="expense", transaction_date=date(2025, 1, 15),
                               description="invalid zero amount"))
    with pytest.raises(IntegrityError):
        db.session.flush()
    db.session.rollback()
    assert db.session.scalar(select(db.func.count(Transaction.id)).where(
        Transaction.user_id == expense_user.id
    )) == 1
    income_summary = financial_summary(income_user.id, today=date(2025, 1, 31))
    expense_summary = financial_summary(expense_user.id, today=date(2025, 1, 31))
    assert income_summary["total_income"] == Decimal("123456789.01")
    assert income_summary["total_expenses"] == ZERO
    assert expense_summary["total_income"] == ZERO
    assert expense_summary["total_expenses"] == Decimal("9999999999.99")


def test_categorization_has_chronological_external_validation(app):
    dataset, users, _, _ = seed_synthetic_dataset(app)
    frame = fetch_user_transactions(users["primary"].id)
    cutoff = pd.Timestamp("2025-07-01")
    train = frame[frame["transaction_date"] < cutoff]
    test = frame[frame["transaction_date"] >= cutoff]
    trained = train_categorizer(train)
    assert trained["sufficiently_trained"]
    eligible = {(row["category"], row["transaction_type"])
                for row in trained["eligible_category_distribution"]}
    evaluation_rows = test[
        test["description"].fillna("").str.strip().ne("")
        & test.apply(lambda row: (row["category"], row["transaction_type"]) in eligible, axis=1)
    ]
    actual = [json.dumps([kind, category], ensure_ascii=False)
              for category, kind in zip(evaluation_rows["category"], evaluation_rows["transaction_type"])]
    predicted = []
    for row in evaluation_rows.itertuples(index=False):
        suggestion = predict_category(row.description, trained)
        predicted.append(json.dumps([suggestion["transaction_type"], suggestion["predicted_category"]],
                                    ensure_ascii=False)
                         if suggestion["predicted_category"] else "__no_suggestion__")
    assert len(actual) >= 30
    assert all(pd.Timestamp(day) >= cutoff for day in evaluation_rows["transaction_date"])
    labels = sorted(set(actual))
    scores = {
        "accuracy": accuracy_score(actual, predicted),
        "precision_macro": precision_score(actual, predicted, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(actual, predicted, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(actual, predicted, labels=labels, average="macro", zero_division=0),
    }
    assert all(0 <= score <= 1 for score in scores.values())
    assert all(value is not None for value in predicted)
    empty = predict_category("", trained)
    short = predict_category("x", trained)
    unknown = predict_category("zyxwv qwerty blorf unlisted merchant", trained)
    assert empty["status"] == "invalid_description"
    assert short["status"] == "low_information"
    assert unknown["status"] == "unknown_description"
    valid_suggestion = predict_category("monthly salary payroll credit", trained)
    assert valid_suggestion["predicted_category"] == "Salary"
    assert valid_suggestion["transaction_type"] == "income"
    assert 0 <= valid_suggestion["confidence"] <= 1
    insufficient = train_categorizer(train.iloc[:2])
    assert insufficient["status"] == "insufficient_data"
    assert all(value is not None for value in predicted)


def test_dashboard_and_analytics_routes_render_synthetic_user_data_without_database_side_effects(app, client):
    dataset, users, _, _ = seed_synthetic_dataset(app)
    primary = users["primary"]
    response = client.post("/login", data={"email": primary.email,
                                           "password": "SyntheticTestPassword123"},
                           follow_redirects=True)
    assert response.status_code == 200
    for path in ("/dashboard", "/analytics?start_date=2024-01-01&end_date=2025-12-31",
                 "/budgets", "/savings", "/ml/"):
        page = client.get(path)
        assert page.status_code == 200, path
    dashboard = client.get("/dashboard").get_data(as_text=True)
    assert "₹" in dashboard
    assert "Synthetic Primary" in dashboard or "Dashboard" in dashboard
    assert "dashboard-chart-data" in dashboard and "category_values" in dashboard
    assert "Emergency savings" in dashboard
    analytics_page = client.get("/analytics?start_date=2024-01-01&end_date=2025-12-31")
    assert "Why flagged" in analytics_page.get_data(as_text=True)
    assert db.session.scalar(select(db.func.count(Transaction.id)).where(
        Transaction.user_id == primary.id
    )) == sum(row["user_key"] == "primary" for row in dataset.transactions)

    # Explicit training is exercised in a private, temporary artifact directory.
    with tempfile.TemporaryDirectory(dir=app.instance_path, prefix="stage9-models-") as model_dir:
        app.config["ML_MODEL_STORE_DIR"] = model_dir
        train_response = client.post("/ml/train", data={"model_type": "expense_prediction"},
                                     follow_redirects=True)
        assert train_response.status_code == 200
        assert "training" in train_response.get_data(as_text=True).lower() or "model" in train_response.get_data(as_text=True).lower()
        ml_page = client.get("/ml/")
        assert ml_page.status_code == 200
        assert "2025-12" in ml_page.get_data(as_text=True)
        ml_text = ml_page.get_data(as_text=True)
        assert "Actual vs predicted expenses" in ml_text
        assert "Predicted" in ml_text or "predicted" in ml_text
        artifacts = list(Path(model_dir).rglob("*.joblib"))
        assert artifacts and all(model_dir in str(path) for path in artifacts)


def test_empty_dashboard_reports_insufficient_history_without_fabricated_forecasts(app, client):
    user = User(email="empty.user@example.test", full_name="Empty User", currency="INR")
    user.set_password("SyntheticTestPassword123")
    db.session.add(user)
    db.session.commit()
    login = client.post("/login", data={"email": user.email, "password": "SyntheticTestPassword123"},
                        follow_redirects=True)
    assert login.status_code == 200
    ml_page = client.get("/ml/")
    assert ml_page.status_code == 200
    html = ml_page.get_data(as_text=True)
    assert "No valid expense history" in html or "insufficient" in html.lower()
    assert "No actual or saved prediction history is available yet." in html
    prediction = client.get("/ml/prediction").get_json()
    assert prediction["status"] == "model_not_current"
    assert not prediction.get("predicted_expenses")
    forecast = client.get("/ml/forecast?horizon=3").get_json()
    assert forecast["status"] == "model_not_current"
    assert forecast["forecasts"] == []


def test_fixture_cli_output_is_strictly_limited_to_documented_test_files():
    root = Path(__file__).resolve().parents[1]
    fixture_dir = root / "tests" / "fixtures"
    expected = {"synthetic_financial_transactions.csv", "synthetic_savings_contributions.csv",
                "synthetic_financial_scenario.json"}
    present = {path.name for path in fixture_dir.iterdir() if path.is_file()}
    assert present == expected
    scenario = json.loads((fixture_dir / "synthetic_financial_scenario.json").read_text(encoding="utf-8"))
    assert scenario["monthly_income"] == "50000.00"
    with (fixture_dir / "synthetic_financial_transactions.csv").open(encoding="utf-8") as stream:
        rows = list(__import__("csv").DictReader(stream))
    assert len(rows) == 481
    assert {row["user_key"] for row in rows} == {"primary", "secondary", "scenario"}
