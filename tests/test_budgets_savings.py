"""Budget, savings goal, alerts, dashboard, and ownership tests."""

from calendar import monthrange
from datetime import date, timedelta
from decimal import Decimal

from app.extensions import db
from app.models import Budget, Category, GoalContribution, Recommendation, SavingsGoal, Transaction, User
from app.services.budgets import budget_progress
from app.services.savings import goal_progress


def register_login(client, email="budget-owner@example.com", name="Budget Owner"):
    password = "SafePassphrase123"
    response = client.post("/register", data={
        "full_name": name, "email": email, "password": password, "confirm_password": password,
    })
    assert response.status_code == 302
    response = client.post("/login", data={"email": email, "password": password})
    assert response.status_code == 302
    return email


def get_category_id(app, email, name="Food", kind="expense"):
    with app.app_context():
        user = db.session.scalar(db.select(User).where(User.email == email))
        return db.session.scalar(db.select(Category.id).where(
            Category.user_id == user.id,
            Category.name == name,
            Category.transaction_type == kind,
        ))


def budget_payload(category_id, **changes):
    today = date.today()
    start = today.replace(day=1)
    end = today.replace(day=monthrange(today.year, today.month)[1])
    payload = {
        "name": "Food plan", "category_id": str(category_id), "amount": "500.00",
        "period_type": "monthly", "period_start": start.isoformat(),
        "period_end": end.isoformat(), "description": "Monthly food spending",
    }
    payload.update(changes)
    return payload


def create_budget(client, category_id, **changes):
    return client.post("/budgets/new", data=budget_payload(category_id, **changes))


def create_goal(client, **changes):
    payload = {
        "name": "Emergency fund", "target_amount": "1000.00",
        "target_date": "", "description": "Three months of essentials",
    }
    payload.update(changes)
    return client.post("/savings/new", data=payload)


def goal_id_for(app, email, name="Emergency fund"):
    with app.app_context():
        user = db.session.scalar(db.select(User).where(User.email == email))
        return db.session.scalar(db.select(SavingsGoal.id).where(
            SavingsGoal.user_id == user.id, SavingsGoal.name == name,
        ))


def add_contribution(client, goal_id, amount, note="", contributed_on=None):
    return client.post(f"/savings/{goal_id}/contributions", data={
        "amount": amount,
        "contributed_on": (contributed_on or date.today()).isoformat(),
        "note": note,
    })


def test_budget_create_monthly_and_reject_zero_negative_invalid(client, app):
    email = register_login(client)
    client.get("/transactions/new")
    category_id = get_category_id(app, email)
    response = create_budget(client, category_id)
    assert response.status_code == 302
    with app.app_context():
        budget = db.session.scalar(db.select(Budget))
        assert budget.user_id == db.session.scalar(db.select(User.id).where(User.email == email))
        assert budget.period_start.day == 1
        assert budget.period_end.day == monthrange(budget.period_end.year, budget.period_end.month)[1]
        assert budget.name == "Food plan"
    for amount in ("0", "-5", "oops", "10000000000"):
        response = create_budget(client, category_id, amount=amount,
                                 period_start="2032-01-01", period_end="2032-01-31")
        assert response.status_code == 200
    invalid_range = create_budget(client, category_id, period_type="custom",
                                  period_start="2032-02-01", period_end="2032-01-31")
    assert b"End date must be on or after start date" in invalid_range.data


def test_budget_custom_date_range_actual_spending_remaining_and_percentage(client, app):
    email = register_login(client)
    client.get("/transactions/new")
    category_id = get_category_id(app, email)
    today = date.today()
    start = today - timedelta(days=2)
    end = today + timedelta(days=2)
    response = create_budget(client, category_id, name="Short food plan", amount="100.00",
                             period_type="custom", period_start=start.isoformat(), period_end=end.isoformat())
    assert response.status_code == 302
    add_transaction = lambda value, desc, category=category_id, day=today: client.post("/transactions/new", data={
        "transaction_type": "expense", "amount": value, "category_id": str(category),
        "description": desc, "transaction_date": day.isoformat(), "notes": "",
    })
    add_transaction("70.00", "Inside date range")
    add_transaction("40.00", "Outside date range", day=end + timedelta(days=1))
    transportation_id = get_category_id(app, email, "Transportation")
    add_transaction("250.00", "Different category", category=transportation_id)
    with app.app_context():
        budget = db.session.scalar(db.select(Budget))
        progress = budget_progress(budget, today)
        assert progress["spent"] == Decimal("70.00")
        assert progress["remaining"] == Decimal("30.00")
        assert progress["usage"] == Decimal("70.00")
        assert progress["status"] == "warning"


def test_budget_thresholds_and_overspending(client, app):
    email = register_login(client)
    client.get("/transactions/new")
    category_id = get_category_id(app, email)
    create_budget(client, category_id, amount="100.00")
    with app.app_context():
        budget = db.session.scalar(db.select(Budget))
        for amount, status in (("69.99", "normal"), ("70.00", "warning"),
                               ("90.00", "critical"), ("100.00", "critical"),
                               ("100.01", "overspent")):
            db.session.add(Transaction(
                user_id=budget.user_id, category_id=category_id, amount=Decimal(amount),
                transaction_type="expense", description="Threshold test", transaction_date=date.today(),
            ))
            db.session.flush()
            progress = budget_progress(budget)
            assert progress["status"] == status
            db.session.rollback()
            budget = db.session.get(Budget, budget.id)


def test_budget_edit_delete_and_conflicting_period(client, app):
    email = register_login(client)
    client.get("/transactions/new")
    category_id = get_category_id(app, email)
    created = create_budget(client, category_id)
    assert created.status_code == 302
    with app.app_context():
        budget_id = db.session.scalar(db.select(Budget.id))
    today = date.today()
    start = today.replace(day=1)
    end = today.replace(day=monthrange(today.year, today.month)[1])
    overlap = client.post("/budgets/new", data=budget_payload(
        category_id, name="Duplicate", period_type="custom",
        period_start=start.isoformat(), period_end=end.isoformat(),
    ))
    assert b"already covers" in overlap.data
    edited = client.post(f"/budgets/{budget_id}/edit", data=budget_payload(
        category_id, name="Updated budget", amount="650.00",
    ))
    assert edited.status_code == 302
    with app.app_context():
        budget = db.session.get(Budget, budget_id)
        assert budget.name == "Updated budget"
        assert budget.amount == Decimal("650.00")
    assert client.post(f"/budgets/{budget_id}/delete").status_code == 302
    with app.app_context():
        assert db.session.get(Budget, budget_id) is None


def test_budget_category_ownership_and_idor(client, app):
    owner = register_login(client)
    client.get("/transactions/new")
    foreign_category_id = get_category_id(app, owner)
    create_budget(client, foreign_category_id)
    with app.app_context():
        budget_id = db.session.scalar(db.select(Budget.id))
    client.post("/logout")
    other = register_login(client, "another-budget@example.com", "Another Owner")
    client.get("/transactions/new")
    foreign_category_response = create_budget(client, foreign_category_id,
                                              period_start="2035-01-01", period_end="2035-01-31")
    assert b"Not a valid choice" in foreign_category_response.data
    assert client.get(f"/budgets/{budget_id}/edit").status_code == 404
    assert client.post(f"/budgets/{budget_id}/delete").status_code == 404
    with app.app_context():
        budget = db.session.get(Budget, budget_id)
        assert budget.user_id == db.session.scalar(db.select(User.id).where(User.email == owner))


def test_budget_alerts_are_persisted_once_and_dismissible(client, app):
    email = register_login(client)
    client.get("/transactions/new")
    category_id = get_category_id(app, email)
    create_budget(client, category_id, amount="100.00")
    with app.app_context():
        user_id = db.session.scalar(db.select(User.id).where(User.email == email))
    assert client.post("/transactions/new", data={
        "transaction_type": "expense", "amount": "110.00", "category_id": str(category_id),
        "description": "Alert trigger", "transaction_date": date.today().isoformat(), "notes": "",
    }).status_code == 302
    page = client.get("/budgets")
    assert b"Food spending is 110.0%" in page.data
    client.get("/budgets")
    with app.app_context():
        alerts = list(db.session.scalars(db.select(Recommendation).where(
            Recommendation.user_id == user_id,
            Recommendation.rationale["type"].as_string() == "budget_alert",
        )))
        assert len(alerts) == 3
        alert_id = alerts[0].id
    assert client.post(f"/budgets/alerts/{alert_id}/dismiss").status_code == 302
    with app.app_context():
        assert db.session.get(Recommendation, alert_id).dismissed_at is not None


def test_savings_goal_contributions_progress_edit_delete_completion_and_status(client, app):
    email = register_login(client)
    assert create_goal(client).status_code == 302
    goal_id = goal_id_for(app, email)
    invalid_contribution = add_contribution(client, goal_id, "0")
    assert invalid_contribution.status_code == 200
    invalid_contribution = add_contribution(client, goal_id, "-1")
    assert invalid_contribution.status_code == 200
    assert add_contribution(client, goal_id, "400.00", "First save").status_code == 302
    with app.app_context():
        goal = db.session.get(SavingsGoal, goal_id)
        progress = goal_progress(goal)
        assert progress["saved"] == Decimal("400.00")
        assert progress["remaining"] == Decimal("600.00")
        assert progress["percentage"] == Decimal("40.000")
        assert progress["status"] == "in_progress"
        contribution_id = db.session.scalar(db.select(GoalContribution.id))
    edited = client.post(f"/savings/{goal_id}/contributions/{contribution_id}/edit", data={
        "amount": "1000.00", "contributed_on": date.today().isoformat(), "note": "Goal reached",
    })
    assert edited.status_code == 302
    with app.app_context():
        goal = db.session.get(SavingsGoal, goal_id)
        assert goal_progress(goal)["status"] == "completed"
        assert goal.completed_at is not None
    assert client.post(f"/savings/{goal_id}/contributions/{contribution_id}/delete").status_code == 302
    with app.app_context():
        goal = db.session.get(SavingsGoal, goal_id)
        assert goal_progress(goal)["saved"] == Decimal("0.00")
        assert goal.completed_at is None
        goal.target_date = date.today() - timedelta(days=1)
        db.session.commit()
        assert goal_progress(goal)["status"] == "overdue"


def test_goal_create_edit_delete_and_invalid_amounts(client, app):
    register_login(client)
    for amount in ("0", "-1", "wrong", "10000000000.00"):
        response = create_goal(client, target_amount=amount)
        assert response.status_code == 200
    created = create_goal(client)
    assert created.status_code == 302
    with app.app_context():
        goal_id = db.session.scalar(db.select(SavingsGoal.id))
    edit = client.post(f"/savings/{goal_id}/edit", data={
        "name": "Laptop fund", "target_amount": "2500.00", "target_date": "", "description": "New laptop",
    })
    assert edit.status_code == 302
    with app.app_context():
        goal = db.session.get(SavingsGoal, goal_id)
        assert goal.name == "Laptop fund"
        assert goal.target_amount == Decimal("2500.00")
    assert client.post(f"/savings/{goal_id}/delete").status_code == 302
    with app.app_context():
        assert db.session.get(SavingsGoal, goal_id) is None


def test_savings_goal_and_contribution_are_user_scoped(client, app):
    owner = register_login(client)
    create_goal(client)
    goal_id = goal_id_for(app, owner)
    add_contribution(client, goal_id, "100.00")
    with app.app_context():
        contribution_id = db.session.scalar(db.select(GoalContribution.id))
    client.post("/logout")
    register_login(client, "another-saver@example.com", "Another Saver")
    assert client.get(f"/savings/{goal_id}/edit").status_code == 404
    assert client.get(f"/savings/{goal_id}/contributions").status_code == 404
    assert client.get(f"/savings/{goal_id}/contributions/{contribution_id}/edit").status_code == 404
    assert client.post(f"/savings/{goal_id}/contributions/{contribution_id}/delete").status_code == 404
    with app.app_context():
        assert db.session.scalar(db.select(SavingsGoal.user_id).where(SavingsGoal.id == goal_id)) == db.session.scalar(
            db.select(User.id).where(User.email == owner)
        )


def test_dashboard_budget_savings_summaries_and_empty_states(client, app):
    email = register_login(client)
    empty = client.get("/dashboard")
    assert b"No active budgets" in empty.data
    assert b"Create a savings goal" in empty.data
    client.get("/transactions/new")
    category_id = get_category_id(app, email)
    create_budget(client, category_id)
    create_goal(client)
    page = client.get("/dashboard")
    assert page.status_code == 200
    assert b"Budget vs actual spending" in page.data
    assert b"Savings goal progress" in page.data
    assert b"budget-chart" in page.data
    assert b"savings-chart" in page.data


def test_dashboard_summaries_are_isolated_by_user(client, app):
    first_email = register_login(client)
    client.get("/transactions/new")
    first_category = get_category_id(app, first_email)
    create_budget(client, first_category, amount="100.00", name="First private budget")
    create_goal(client, name="First user's goal", target_amount="200.00")
    client.post("/logout")
    second_email = register_login(client, "dashboard-second@example.com", "Second User")
    client.get("/transactions/new")
    second_category = get_category_id(app, second_email)
    assert create_budget(client, second_category, amount="9000.00", name="Second private budget").status_code == 302
    create_goal(client, name="Second private goal", target_amount="8000.00")
    response = client.get("/dashboard")
    assert b"9,000.00" in response.data
    assert b"First user's goal" not in response.data
    assert b"First private budget" not in response.data
    assert b"Second private goal" in response.data


def test_csrf_protects_budget_and_savings_writes(client, app):
    register_login(client)
    app.config["WTF_CSRF_ENABLED"] = True
    assert client.post("/budgets/new", data={}).status_code == 400
    assert client.post("/savings/new", data={}).status_code == 400
