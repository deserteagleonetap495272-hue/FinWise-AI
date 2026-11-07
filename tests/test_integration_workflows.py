"""Cross-module financial workflows exercised through authenticated routes."""

from calendar import monthrange
from datetime import date
from decimal import Decimal

from sqlalchemy import func, select

from app.extensions import db
from app.models import Budget, Category, GoalContribution, Recommendation, SavingsGoal, Transaction, User


PASSWORD = "SafePassphrase123"


def register_and_login(client, email="workflow@example.com"):
    response = client.post("/register", data={
        "full_name": "Workflow User", "email": email,
        "password": PASSWORD, "confirm_password": PASSWORD,
    })
    assert response.status_code == 302
    response = client.post("/login", data={"email": email, "password": PASSWORD})
    assert response.status_code == 302


def category_id(app, email, name, kind):
    with app.app_context():
        user = db.session.scalar(select(User).where(User.email == email))
        return db.session.scalar(select(Category.id).where(
            Category.user_id == user.id,
            Category.name == name,
            Category.transaction_type == kind,
        ))


def test_full_financial_workflow_reaches_analytics_and_ml_dashboard(app, client):
    email = "workflow@example.com"
    register_and_login(client, email)
    today = date.today()
    expense_category = category_id(app, email, "Food", "expense")
    income_category = category_id(app, email, "Salary", "income")

    assert client.post("/transactions/new", data={
        "transaction_type": "income", "amount": "5000.00",
        "category_id": str(income_category), "description": "Monthly salary",
        "transaction_date": today.isoformat(), "notes": "",
    }).status_code == 302

    start = today.replace(day=1)
    end = today.replace(day=monthrange(today.year, today.month)[1])
    assert client.post("/budgets/new", data={
        "name": "Food plan", "category_id": str(expense_category), "amount": "1000.00",
        "period_type": "monthly", "period_start": start.isoformat(),
        "period_end": end.isoformat(), "description": "Monthly food budget",
    }).status_code == 302

    assert client.post("/transactions/new", data={
        "transaction_type": "expense", "amount": "250.00",
        "category_id": str(expense_category), "description": "Grocery market purchase",
        "transaction_date": today.isoformat(), "notes": "",
    }).status_code == 302

    assert client.post("/savings/new", data={
        "name": "Emergency fund", "target_amount": "1000.00",
        "target_date": "", "description": "",
    }).status_code == 302
    with app.app_context():
        goal_id = db.session.scalar(select(SavingsGoal.id).where(SavingsGoal.name == "Emergency fund"))
    assert client.post(f"/savings/{goal_id}/contributions", data={
        "amount": "125.00", "contributed_on": today.isoformat(), "note": "First deposit",
    }).status_code == 302

    analytics = client.get("/analytics")
    assert analytics.status_code == 200
    assert b"5,000.00" in analytics.data
    assert b"250.00" in analytics.data

    budget_page = client.get("/budgets")
    savings_page = client.get("/savings")
    assert budget_page.status_code == savings_page.status_code == 200
    assert b"Food plan" in budget_page.data
    assert b"Emergency fund" in savings_page.data

    ml_page = client.get("/ml/")
    assert ml_page.status_code == 200
    assert b"Actual vs predicted expenses" in ml_page.data
    assert b"Train or refresh" in ml_page.data
    prediction = client.get("/ml/prediction")
    assert prediction.status_code == 200
    assert prediction.json["status"] == "model_not_current"

    with app.app_context():
        user = db.session.scalar(select(User).where(User.email == email))
        transaction_total = db.session.scalar(select(func.sum(Transaction.amount)).where(Transaction.user_id == user.id))
        contribution_total = db.session.scalar(select(func.sum(GoalContribution.amount)).join(SavingsGoal).where(SavingsGoal.user_id == user.id))
        budget = db.session.scalar(select(Budget).where(Budget.user_id == user.id))
        assert transaction_total == Decimal("5250.00")
        assert contribution_total == Decimal("125.00")
        assert budget.user_id == user.id


def test_read_only_pages_do_not_persist_budget_alerts(app, client):
    email = "readonly-workflow@example.com"
    register_and_login(client, email)
    user_id = db.session.scalar(select(User.id).where(User.email == email))
    category = db.session.scalar(select(Category).where(
        Category.user_id == user_id, Category.name == "Food", Category.transaction_type == "expense"
    ))
    today = date.today()
    with app.app_context():
        db.session.add(Budget(
            user_id=user_id, category_id=category.id, amount=Decimal("10.00"), name="Small food plan",
            period_start=today.replace(day=1),
            period_end=today.replace(day=monthrange(today.year, today.month)[1]),
        ))
        db.session.add(Transaction(
            user_id=user_id, category_id=category.id, amount=Decimal("20.00"),
            transaction_type="expense", description="Over budget", transaction_date=today,
        ))
        db.session.commit()
        before = db.session.scalar(select(func.count()).select_from(Recommendation))

    assert client.get("/dashboard").status_code == 200
    assert client.get("/budgets").status_code == 200
    assert client.get("/transactions").status_code == 200
    with app.app_context():
        after = db.session.scalar(select(func.count()).select_from(Recommendation))
        assert before == after == 0
