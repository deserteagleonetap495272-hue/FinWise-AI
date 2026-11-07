"""Transaction management, filtering, and per-user isolation tests."""

from datetime import date
from decimal import Decimal

from app.extensions import db
from app.models import Category, Transaction, User
from app.services.transactions import financial_summary


def create_and_login(client, email="owner@example.com", name="Test Owner"):
    password = "SafePassphrase123"
    response = client.post("/register", data={
        "full_name": name,
        "email": email,
        "password": password,
        "confirm_password": password,
    })
    assert response.status_code == 302
    response = client.post("/login", data={"email": email, "password": password})
    assert response.status_code == 302
    return email


def category_id(app, email, name, transaction_type):
    with app.app_context():
        user = db.session.scalar(db.select(User).where(User.email == email))
        category = db.session.scalar(db.select(Category).where(
            Category.user_id == user.id,
            Category.name == name,
            Category.transaction_type == transaction_type,
        ))
        return category.id


def add_transaction(client, app, email, *, amount="100.00", transaction_type="expense",
                    category_name="Food", description="Test purchase", transaction_date=None,
                    notes=""):
    # Visiting the form initializes per-user default categories.
    client.get("/transactions/new")
    selected_category = category_id(app, email, category_name, transaction_type)
    return client.post("/transactions/new", data={
        "transaction_type": transaction_type,
        "amount": amount,
        "category_id": str(selected_category),
        "description": description,
        "transaction_date": (transaction_date or date.today()).isoformat(),
        "notes": notes,
    })


def test_add_income_and_expense(client, app):
    email = create_and_login(client)
    income = add_transaction(client, app, email, amount="2500.50", transaction_type="income",
                             category_name="Salary", description="Monthly pay")
    expense = add_transaction(client, app, email, amount="120.25", description="Lunch", notes="Team lunch")
    assert income.status_code == expense.status_code == 302
    with app.app_context():
        rows = list(db.session.scalars(db.select(Transaction).order_by(Transaction.id)))
        assert [row.amount for row in rows] == [Decimal("2500.50"), Decimal("120.25")]
        assert rows[0].user_id == rows[1].user_id
        assert rows[1].notes == "Team lunch"


def test_rejects_nonpositive_or_invalid_amount(client, app):
    email = create_and_login(client)
    for amount in ("0", "-1", "not-a-number", "10000000000.00"):
        response = add_transaction(client, app, email, amount=amount)
        assert response.status_code == 200
        assert b"Amount" in response.data
    with app.app_context():
        assert db.session.scalar(db.select(db.func.count()).select_from(Transaction)) == 0


def test_rejects_another_users_category(client, app):
    email_a = create_and_login(client)
    client.get("/transactions/new")
    category_a = category_id(app, email_a, "Food", "expense")
    client.post("/logout")
    email_b = create_and_login(client, "other@example.com", "Other Owner")
    response = client.post("/transactions/new", data={
        "transaction_type": "expense", "amount": "10.00", "category_id": str(category_a),
        "description": "Invalid category", "transaction_date": date.today().isoformat(), "notes": "",
    })
    assert response.status_code == 200
    assert b"valid category" in response.data
    with app.app_context():
        assert db.session.scalar(db.select(db.func.count()).select_from(Transaction)) == 0


def test_history_search_and_category_date_filters(client, app):
    email = create_and_login(client)
    add_transaction(client, app, email, description="Coffee run")
    add_transaction(client, app, email, description="Bus ticket", category_name="Transportation")
    food_id = category_id(app, email, "Food", "expense")
    response = client.get(f"/transactions?q=Coffee&category_id={food_id}&start_date={date.today().isoformat()}")
    assert response.status_code == 200
    assert b"Coffee run" in response.data
    assert b"Bus ticket" not in response.data


def test_edit_and_delete_own_transaction(client, app):
    email = create_and_login(client)
    add_transaction(client, app, email, description="Old description")
    with app.app_context():
        transaction = db.session.scalar(db.select(Transaction))
        transaction_id = transaction.id
    response = client.post(f"/transactions/{transaction_id}/edit", data={
        "transaction_type": "expense", "amount": "300.10",
        "category_id": str(category_id(app, email, "Groceries", "expense")),
        "description": "Updated purchase", "transaction_date": date.today().isoformat(), "notes": "",
    })
    assert response.status_code == 302
    with app.app_context():
        transaction = db.session.get(Transaction, transaction_id)
        assert transaction.description == "Updated purchase"
        assert transaction.amount == Decimal("300.10")
    assert client.post(f"/transactions/{transaction_id}/delete").status_code == 302
    with app.app_context():
        assert db.session.get(Transaction, transaction_id) is None


def test_transaction_idor_is_hidden_from_other_user(client, app):
    email_a = create_and_login(client)
    add_transaction(client, app, email_a)
    with app.app_context():
        transaction_id = db.session.scalar(db.select(Transaction.id))
    client.post("/logout")
    create_and_login(client, "other@example.com", "Other Owner")
    assert client.get(f"/transactions/{transaction_id}/edit").status_code == 404
    assert client.post(f"/transactions/{transaction_id}/delete").status_code == 404


def test_history_hides_other_users_records(client, app):
    owner_email = create_and_login(client)
    add_transaction(client, app, owner_email, description="Private owner purchase")
    client.post("/logout")
    create_and_login(client, "other@example.com", "Other Owner")
    response = client.get("/transactions")
    assert response.status_code == 200
    assert b"Private owner purchase" not in response.data


def test_filters_compose_and_pagination(client, app):
    email = create_and_login(client)
    for number in range(17):
        add_transaction(client, app, email, description=f"Groceries {number:02d}")
    first_page = client.get("/transactions?type=expense&min_amount=100&page=1")
    second_page = client.get("/transactions?type=expense&min_amount=100&page=2")
    assert b"Page 1 of 2" in first_page.data
    assert b"Page 2 of 2" in second_page.data


def test_financial_summary_is_decimal_and_user_scoped(client, app):
    email_a = create_and_login(client)
    add_transaction(client, app, email_a, amount="1000.25", transaction_type="income",
                    category_name="Salary", description="Salary")
    add_transaction(client, app, email_a, amount="200.10", description="Food")
    client.post("/logout")
    email_b = create_and_login(client, "other@example.com", "Other Owner")
    add_transaction(client, app, email_b, amount="99999.00", transaction_type="income",
                    category_name="Salary", description="Other salary")
    with app.app_context():
        user_a = db.session.scalar(db.select(User).where(User.email == email_a))
        summary = financial_summary(user_a.id)
        assert summary["total_income"] == Decimal("1000.25")
        assert summary["total_expenses"] == Decimal("200.10")
        assert summary["balance"] == Decimal("800.15")
        assert isinstance(summary["monthly_savings"], Decimal)
        empty_user = User(email="empty@example.com", full_name="Empty User", password_hash="unused")
        db.session.add(empty_user)
        db.session.flush()
        empty = financial_summary(empty_user.id)
        assert empty["total_income"] == Decimal("0.00")
        assert not empty["category_labels"]


def test_dashboard_shows_real_user_totals_recent_records_and_empty_state(client, app):
    email = create_and_login(client)
    empty_response = client.get("/dashboard")
    assert b"No transactions yet" in empty_response.data
    add_transaction(client, app, email, amount="1250.75", transaction_type="income",
                    category_name="Salary", description="Payday")
    add_transaction(client, app, email, amount="31.25", description="Breakfast")
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"1,250.75" in response.data
    assert b"31.25" in response.data
    assert b"Breakfast" in response.data
    assert b"dashboard-chart-data" in response.data
    assert b"Expense categories" in response.data


def test_csrf_rejects_transaction_write_when_enabled(client, app):
    email = create_and_login(client)
    client.get("/transactions/new")
    app.config["WTF_CSRF_ENABLED"] = True
    response = client.post("/transactions/new", data={
        "transaction_type": "expense", "amount": "10.00", "category_id": "1",
        "description": "Missing token", "transaction_date": date.today().isoformat(),
    })
    assert response.status_code == 400
