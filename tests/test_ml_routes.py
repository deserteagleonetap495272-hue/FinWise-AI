"""Protected ML dashboard, model, and categorization route tests."""

from datetime import date
from decimal import Decimal

from sqlalchemy import select

from app.extensions import db
from app.models import Category, Transaction, User
from app.routes import ml as ml_routes
from app.services.transactions import ensure_default_categories


PASSWORD = "SafePassphrase123"


def create_user(app, email):
    with app.app_context():
        user = User(email=email, full_name="ML Test", password_hash="unused")
        user.set_password(PASSWORD)
        db.session.add(user)
        db.session.flush()
        ensure_default_categories(user.id)
        user_id = user.id
        db.session.commit()
        return user_id


def create_transaction(app, user_id, *, day=date(2026, 1, 5), amount="42.50",
                       kind="expense", description="weekly grocery market", category_name=None):
    with app.app_context():
        category_name = category_name or ("Food" if kind == "expense" else "Salary")
        category = db.session.scalar(select(Category).where(
            Category.user_id == user_id,
            Category.transaction_type == kind,
            Category.name == category_name,
        ))
        transaction = Transaction(
            user_id=user_id, category_id=category.id, amount=Decimal(amount),
            transaction_type=kind, transaction_date=day, description=description,
        )
        db.session.add(transaction)
        db.session.commit()
        return transaction.id, category.id


def login_as(client, email):
    response = client.post("/login", data={"email": email, "password": PASSWORD})
    assert response.status_code == 302


def test_all_ml_pages_and_mutations_require_login(client):
    assert client.get("/ml/").status_code == 302
    assert client.get("/ml/prediction").status_code == 302
    assert client.get("/ml/forecast").status_code == 302
    assert client.post("/ml/train", data={"model_type": "expense_prediction"}).status_code == 302
    assert client.post("/ml/transactions/1/suggest").status_code == 302
    assert client.post("/ml/transactions/1/accept").status_code == 302
    assert client.post("/ml/transactions/1/reject").status_code == 302


def test_dashboard_renders_empty_state_without_automatic_training(app, client, monkeypatch):
    create_user(app, "ml-empty@example.com")
    login_as(client, "ml-empty@example.com")
    calls = []
    monkeypatch.setattr(ml_routes.model_store, "train_user_model", lambda *args, **kwargs: calls.append(args))
    response = client.get("/ml/")
    assert response.status_code == 200
    assert b"Add transactions to build your financial history" in response.data
    assert b"No actual or saved prediction history" in response.data
    assert calls == []


def test_prediction_route_uses_only_authenticated_users_transactions(app, client, monkeypatch):
    user_a = create_user(app, "ml-a@example.com")
    user_b = create_user(app, "ml-b@example.com")
    transaction_a, _ = create_transaction(app, user_a, amount="15.00")
    create_transaction(app, user_b, amount="99999.00")
    login_as(client, "ml-a@example.com")
    calls = []

    def fake_prediction(user_id, data):
        calls.append((user_id, data["id"].tolist(), data["user_id"].unique().tolist()))
        return {
            "status": "model_not_current", "data_sufficient": False,
            "predicted_expenses": None, "prediction_period": None,
            "selected_model": None, "model_based": False,
            "metrics": None, "baseline_metrics": None, "historical_months": 1,
            "explanation": "Refresh required", "model_artifact_status": "missing",
        }

    monkeypatch.setattr(ml_routes.model_store, "predict_expenses_for_user", fake_prediction)
    response = client.get("/ml/prediction")
    assert response.status_code == 200
    assert response.json["status"] == "model_not_current"
    assert calls == [(user_a, [transaction_a], [user_a])]


def test_forecast_route_does_not_silently_use_or_train_stale_model(app, client, monkeypatch):
    user_id = create_user(app, "ml-stale@example.com")
    create_transaction(app, user_id)
    login_as(client, "ml-stale@example.com")
    calls = []
    monkeypatch.setattr(ml_routes.model_store, "predict_expenses_for_user", lambda owner, data: (
        calls.append(owner) or {"status": "model_not_current"}
    ))
    response = client.get("/ml/forecast?horizon=3")
    assert response.status_code == 200
    assert response.json["status"] == "model_not_current"
    assert "explicitly" in response.json["explanation"]
    assert calls == [user_id]
    assert client.get("/ml/forecast?horizon=2").status_code == 400


def test_forecast_route_reuses_single_stage3_result(app, client, monkeypatch):
    user_id = create_user(app, "ml-forecast@example.com")
    create_transaction(app, user_id)
    login_as(client, "ml-forecast@example.com")
    prediction = {"status": "evaluated", "predicted_expenses": Decimal("55.00")}
    prediction_calls = []
    forecast_calls = []
    monkeypatch.setattr(ml_routes.model_store, "predict_expenses_for_user", lambda owner, data: (
        prediction_calls.append(owner) or prediction
    ))
    monkeypatch.setattr(ml_routes, "forecast_expenses", lambda data, *, horizon, prediction_result: (
        forecast_calls.append((horizon, prediction_result)) or {
            "status": "forecasted", "data_sufficient": True,
            "historical_monthly_expenses": [], "forecasts": [],
            "forecast_month": None, "predicted_amount": None, "model_name": "Saved model",
            "evaluation_metrics": None, "forecast_horizon": horizon,
            "explanation": "Built from supplied model", "limitations": [], "historical_month_count": 1,
        }
    ))
    response = client.get("/ml/forecast?horizon=1")
    assert response.status_code == 200
    assert response.json["model_name"] == "Saved model"
    assert prediction_calls == [user_id]
    assert forecast_calls == [(1, prediction)]


def test_explicit_training_is_user_scoped_and_current_models_are_skipped(app, client, monkeypatch):
    user_id = create_user(app, "ml-train@example.com")
    create_transaction(app, user_id)
    login_as(client, "ml-train@example.com")
    calls = []

    def fake_train(owner, model_type, data, *, force_refresh=False):
        calls.append((owner, model_type, data["user_id"].unique().tolist(), force_refresh))
        return {"status": "already_current", "trained": False, "training_result": None}

    monkeypatch.setattr(ml_routes.model_store, "train_user_model", fake_train)
    response = client.post("/ml/train", data={"model_type": "expense_prediction"}, follow_redirects=True)
    assert response.status_code == 200
    assert b"already current" in response.data
    assert calls == [(user_id, "expense_prediction", [user_id], False)]


def test_explicit_training_handles_insufficient_history(app, client, monkeypatch):
    user_id = create_user(app, "ml-insufficient@example.com")
    create_transaction(app, user_id)
    login_as(client, "ml-insufficient@example.com")
    monkeypatch.setattr(ml_routes.model_store, "train_user_model", lambda *args, **kwargs: {
        "status": "insufficient_data", "trained": False,
        "training_result": {"explanation": "More expense history is needed."},
    })
    response = client.post("/ml/train", data={"model_type": "transaction_categorization"}, follow_redirects=True)
    assert b"More expense history is needed" in response.data


def test_explicit_expense_training_records_prediction_history(app, client, monkeypatch):
    user_id = create_user(app, "ml-history@example.com")
    create_transaction(app, user_id)
    login_as(client, "ml-history@example.com")
    training_result = {
        "predicted_expenses": Decimal("61.25"), "prediction_period": date(2026, 2, 1),
        "selected_model": "Linear Regression", "metrics": {"mae": 3.0},
    }
    recorded = []
    monkeypatch.setattr(ml_routes.model_store, "train_user_model", lambda *args, **kwargs: {
        "status": "trained", "trained": True, "training_result": training_result,
    })
    monkeypatch.setattr(ml_routes.model_store, "save_prediction_history", lambda owner, result: (
        recorded.append((owner, result)) or [object()]
    ))
    response = client.post("/ml/train", data={"model_type": "expense_prediction"}, follow_redirects=True)
    assert response.status_code == 200
    assert b"added to your prediction history" in response.data
    assert recorded == [(user_id, training_result)]


def test_stale_model_is_shown_and_no_training_occurs_on_dashboard(app, client, monkeypatch):
    user_id = create_user(app, "ml-stale-dashboard@example.com")
    create_transaction(app, user_id)
    login_as(client, "ml-stale-dashboard@example.com")
    monkeypatch.setattr(ml_routes.model_store, "get_model_status", lambda owner, kind, data: {
        "status": "stale", "is_current": False, "model_type": kind, "user_id": owner,
        "metadata": None, "source_fingerprint": "0" * 64,
    })
    calls = []
    monkeypatch.setattr(ml_routes.model_store, "train_user_model", lambda *args, **kwargs: calls.append(args))
    response = client.get("/ml/")
    assert response.status_code == 200
    assert b"saved model is stale" in response.data
    assert calls == []


def test_cross_user_transaction_suggestion_and_acceptance_are_rejected(app, client):
    user_a = create_user(app, "ml-owner@example.com")
    user_b = create_user(app, "ml-other@example.com")
    foreign_transaction, _ = create_transaction(app, user_b)
    login_as(client, "ml-owner@example.com")
    assert client.post(f"/ml/transactions/{foreign_transaction}/suggest").status_code == 404
    assert client.post(f"/ml/transactions/{foreign_transaction}/accept", data={"category_id": 1}).status_code == 404


def test_valid_category_suggestion_can_be_accepted(app, client, monkeypatch):
    user_id = create_user(app, "ml-accept@example.com")
    transaction_id, current_category_id = create_transaction(app, user_id, category_name="Food")
    with app.app_context():
        target = db.session.scalar(select(Category).where(
            Category.user_id == user_id, Category.transaction_type == "expense", Category.name == "Groceries"
        ))
        target_category_id = target.id
    login_as(client, "ml-accept@example.com")
    monkeypatch.setattr(ml_routes.model_store, "predict_category_for_user", lambda *args: {
        "status": "suggestion", "predicted_category": "Groceries", "transaction_type": "expense",
        "confidence": 0.81, "model_name": "Logistic Regression",
        "alternatives": [
            {"category": "Groceries", "transaction_type": "expense", "score": 0.81},
            {"category": "Food", "transaction_type": "expense", "score": 0.19},
        ],
    })
    suggested = client.post(f"/ml/transactions/{transaction_id}/suggest", follow_redirects=True)
    assert suggested.status_code == 200
    assert b"Suggested: Groceries" in suggested.data
    accepted = client.post(f"/ml/transactions/{transaction_id}/accept",
                           data={"category_id": str(target_category_id)}, follow_redirects=True)
    assert accepted.status_code == 200
    with app.app_context():
        transaction = db.session.get(Transaction, transaction_id)
        assert transaction.category_id == target_category_id
        assert transaction.category_id != current_category_id


def _set_suggestion(client, transaction_id, alternatives, *, predicted_type="expense"):
    with client.session_transaction() as browser_session:
        browser_session[ml_routes.SUGGESTION_SESSION_KEY] = {
            "transaction_id": transaction_id, "predicted_category": "Groceries",
            "transaction_type": predicted_type, "confidence": 0.8,
            "model_name": "test", "alternatives": alternatives,
        }


def test_rejection_does_not_modify_transaction(app, client, monkeypatch):
    user_id = create_user(app, "ml-reject@example.com")
    transaction_id, category_id = create_transaction(app, user_id)
    login_as(client, "ml-reject@example.com")
    monkeypatch.setattr(ml_routes.model_store, "predict_category_for_user", lambda *args: {
        "status": "suggestion", "predicted_category": "Groceries", "transaction_type": "expense",
        "confidence": 0.8, "model_name": "test",
        "alternatives": [{"category": "Groceries", "transaction_type": "expense", "score": 0.8}],
    })
    client.post(f"/ml/transactions/{transaction_id}/suggest")
    rejected = client.post(f"/ml/transactions/{transaction_id}/reject", follow_redirects=True)
    assert rejected.status_code == 200
    with app.app_context():
        assert db.session.get(Transaction, transaction_id).category_id == category_id


def test_invalid_foreign_and_wrong_type_category_ids_are_rejected(app, client):
    user_a = create_user(app, "ml-category-a@example.com")
    user_b = create_user(app, "ml-category-b@example.com")
    transaction_id, current_category_id = create_transaction(app, user_a)
    _, foreign_category_id = create_transaction(app, user_b)
    with app.app_context():
        income_category_id = db.session.scalar(select(Category.id).where(
            Category.user_id == user_a, Category.transaction_type == "income"
        ).limit(1))
    login_as(client, "ml-category-a@example.com")
    alternatives = [{"category": "Groceries", "transaction_type": "expense", "score": 0.8}]
    for category_id in (999999, foreign_category_id, income_category_id):
        _set_suggestion(client, transaction_id, alternatives)
        response = client.post(f"/ml/transactions/{transaction_id}/accept",
                               data={"category_id": str(category_id)}, follow_redirects=True)
        assert response.status_code == 200
        assert b"not an available, compatible suggestion" in response.data
        with app.app_context():
            assert db.session.get(Transaction, transaction_id).category_id == current_category_id


def test_suggestion_with_mismatched_transaction_type_cannot_be_accepted(app, client):
    user_id = create_user(app, "ml-type-mismatch@example.com")
    transaction_id, category_id = create_transaction(app, user_id, kind="expense")
    with app.app_context():
        income_category_id = db.session.scalar(select(Category.id).where(
            Category.user_id == user_id, Category.transaction_type == "income"
        ).limit(1))
    login_as(client, "ml-type-mismatch@example.com")
    _set_suggestion(client, transaction_id,
                    [{"category": "Salary", "transaction_type": "income", "score": 0.9}],
                    predicted_type="income")
    response = client.post(f"/ml/transactions/{transaction_id}/accept",
                           data={"category_id": str(income_category_id)}, follow_redirects=True)
    assert b"not an available, compatible suggestion" in response.data
    with app.app_context():
        assert db.session.get(Transaction, transaction_id).category_id == category_id


def test_state_changing_ml_endpoints_reject_requests_without_csrf(app, client):
    user_id = create_user(app, "ml-csrf@example.com")
    transaction_id, category_id = create_transaction(app, user_id)
    login_as(client, "ml-csrf@example.com")
    with client.session_transaction() as browser_session:
        browser_session["ml_category_suggestion"] = {
            "transaction_id": transaction_id, "transaction_type": "expense",
            "alternatives": [{"category": "Food", "transaction_type": "expense", "score": 1.0}],
        }
    app.config["WTF_CSRF_ENABLED"] = True
    endpoints = [
        ("/ml/train", {"model_type": "expense_prediction"}),
        (f"/ml/transactions/{transaction_id}/suggest", {}),
        (f"/ml/transactions/{transaction_id}/accept", {"category_id": str(category_id)}),
        (f"/ml/transactions/{transaction_id}/reject", {}),
    ]
    for path, payload in endpoints:
        response = client.post(path, data=payload)
        assert response.status_code == 400
