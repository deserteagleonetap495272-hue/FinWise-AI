"""Private artifact storage, freshness, and prediction history tests."""

from datetime import date
from decimal import Decimal
import json
import tempfile

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sqlalchemy import select

from app.extensions import db
from app.models import Category, Prediction, Transaction, User
from app.services.ml import model_store
from app.services.ml.expense_prediction import _make_estimator
from app.services.ml.transaction_categorization import _new_pipeline


@pytest.fixture(autouse=True)
def private_store(app):
    with tempfile.TemporaryDirectory(dir=app.instance_path, prefix="test-ml-store-") as directory:
        app.config["ML_MODEL_STORE_DIR"] = directory
        yield


def transaction_frame(months=12):
    rows = []
    start = pd.Period("2025-01", freq="M")
    for index in range(months):
        period = start + index
        rows.append({
            "id": index + 1, "user_id": 1, "category_id": 1,
            "amount": Decimal(str(30 + index)), "transaction_type": "expense",
            "description": f"private fixture purchase {index}",
            "transaction_date": period.start_time.date(),
            "category": "Fixture expense", "category_type": "expense",
        })
    return pd.DataFrame(rows)


def fitted_expense_pipeline():
    model = _make_estimator("Linear Regression")
    features = [[float(i), float(i + 1), float(i + 2), float(i + 1), 0.0, 1.0] for i in range(4)]
    model.fit(features, [float(10 + i) for i in range(4)])
    return model


def fitted_category_pipeline():
    model = _new_pipeline("Logistic Regression")
    model.fit(
        ["silver orchard produce basket", "silver green market fruit", "navy rail commute ticket",
         "navy transit train journey"],
        ['["expense","Groceries"]', '["expense","Groceries"]',
         '["expense","Transit"]', '["expense","Transit"]'],
    )
    return model


def feature_configuration(model_type):
    return model_store._feature_configuration(model_type)


def test_model_save_and_load_is_private_and_metadata_has_no_raw_descriptions(app):
    data = transaction_frame()
    fingerprint = model_store.transaction_fingerprint(1, data)
    with app.app_context():
        metadata = model_store.save_model_artifact(
            1, "expense_prediction", fitted_expense_pipeline(),
            source_fingerprint=fingerprint, training_data_count=9,
            feature_configuration=feature_configuration("expense_prediction"),
            training_period={"start": "2025-01-01", "end": "2025-12-01"},
            evaluation_metrics={"mae": 2.0}, selected_model="Linear Regression",
        )
        paths = model_store.model_artifact_paths(1, "expense_prediction")
        loaded = model_store.load_model_artifact(1, "expense_prediction", expected_fingerprint=fingerprint)
        assert paths[0].is_file() and paths[1].is_file()
        assert "private fixture purchase" not in paths[1].read_text(encoding="utf-8")
        assert metadata["user_id"] == 1
        assert metadata["model_version"] == model_store.MODEL_VERSIONS["expense_prediction"]
        assert loaded["status"] == "current"
        assert isinstance(loaded["model"], Pipeline)
        assert loaded["metadata"]["training_data_count"] == 9


def test_missing_artifact_and_missing_metadata_are_reported(app):
    with app.app_context():
        loaded = model_store.load_model_artifact(5, "transaction_categorization")
        assert loaded == {"status": "missing", "model": None, "metadata": None}


def test_corrupt_artifact_or_metadata_fails_closed(app, monkeypatch):
    data = transaction_frame()
    fingerprint = model_store.transaction_fingerprint(1, data)
    with app.app_context():
        model_store.save_model_artifact(
            1, "expense_prediction", fitted_expense_pipeline(),
            source_fingerprint=fingerprint, training_data_count=9,
            feature_configuration=feature_configuration("expense_prediction"), selected_model="Linear Regression",
        )
        artifact, metadata = model_store.model_artifact_paths(1, "expense_prediction")
        artifact.write_bytes(b"not a joblib artifact")
        assert model_store.load_model_artifact(1, "expense_prediction")["status"] == "corrupt"

        artifact.write_bytes(b"restore checksum mismatch")
        metadata.write_text('{"user_id": 999, "artifact_location": "../../bad.joblib"}', encoding="utf-8")
        called = False

        def forbidden_load(_path):
            nonlocal called
            called = True
            raise AssertionError("Joblib must not run for invalid metadata")

        monkeypatch.setattr(model_store.joblib, "load", forbidden_load)
        assert model_store.load_model_artifact(1, "expense_prediction")["status"] == "corrupt"
        assert called is False


def test_metadata_is_signed_and_detects_tampering(app):
    data = transaction_frame()
    fingerprint = model_store.transaction_fingerprint(1, data)
    with app.app_context():
        _, metadata_path = model_store.model_artifact_paths(1, "expense_prediction")
        model_store.save_model_artifact(
            1, "expense_prediction", fitted_expense_pipeline(),
            source_fingerprint=fingerprint, training_data_count=9,
            feature_configuration=feature_configuration("expense_prediction"), selected_model="Linear Regression",
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["user_id"] = 2
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        assert model_store.get_model_status(1, "expense_prediction", data)["status"] == "corrupt"


def test_user_models_are_separate_and_mixed_owner_input_is_rejected(app):
    data_a = transaction_frame()
    data_b = transaction_frame().assign(user_id=2, description="another user's private purchase")
    fingerprint_a = model_store.transaction_fingerprint(1, data_a)
    fingerprint_b = model_store.transaction_fingerprint(2, data_b)
    with app.app_context():
        model_store.save_model_artifact(
            1, "expense_prediction", fitted_expense_pipeline(),
            source_fingerprint=fingerprint_a, training_data_count=9,
            feature_configuration=feature_configuration("expense_prediction"), selected_model="Linear Regression",
        )
        model_store.save_model_artifact(
            2, "expense_prediction", fitted_expense_pipeline(),
            source_fingerprint=fingerprint_b, training_data_count=9,
            feature_configuration=feature_configuration("expense_prediction"), selected_model="Linear Regression",
        )
        path_a = model_store.model_artifact_paths(1, "expense_prediction")[0]
        path_b = model_store.model_artifact_paths(2, "expense_prediction")[0]
        assert path_a != path_b
        assert path_a.parent.name == "user_1"
        assert path_b.parent.name == "user_2"
        with pytest.raises(ValueError, match="only to the requested user"):
            model_store.transaction_fingerprint(1, pd.concat([data_a, data_b], ignore_index=True))


def test_path_traversal_and_public_static_storage_are_rejected(app):
    with app.app_context():
        for user_id, model_type in ((1, "../escape"), (-1, "expense_prediction"), (True, "expense_prediction")):
            with pytest.raises(ValueError):
                model_store.model_artifact_paths(user_id, model_type)
        app.config["ML_MODEL_STORE_DIR"] = app.static_folder
        with pytest.raises(model_store.ModelStoreError, match="public static"):
            model_store.model_artifact_paths(1, "expense_prediction")


def test_changed_transactions_make_model_stale_without_loading_it(app):
    data = transaction_frame()
    fingerprint = model_store.transaction_fingerprint(1, data)
    with app.app_context():
        model_store.save_model_artifact(
            1, "expense_prediction", fitted_expense_pipeline(),
            source_fingerprint=fingerprint, training_data_count=9,
            feature_configuration=feature_configuration("expense_prediction"), selected_model="Linear Regression",
        )
        changed = data.copy()
        changed.loc[0, "amount"] = Decimal("900.00")
        status = model_store.get_model_status(1, "expense_prediction", changed)
        assert status["status"] == "stale"
        assert model_store.load_current_user_model(1, "expense_prediction", changed)["status"] == "stale"


def test_model_version_change_marks_artifact_stale(app, monkeypatch):
    data = transaction_frame()
    fingerprint = model_store.transaction_fingerprint(1, data)
    with app.app_context():
        model_store.save_model_artifact(
            1, "expense_prediction", fitted_expense_pipeline(),
            source_fingerprint=fingerprint, training_data_count=9,
            feature_configuration=feature_configuration("expense_prediction"), selected_model="Linear Regression",
        )
        monkeypatch.setitem(model_store.MODEL_VERSIONS, "expense_prediction", "2.0.0")
        assert model_store.get_model_status(1, "expense_prediction", data)["status"] == "stale_version"


def test_explicit_refresh_trains_once_for_unchanged_data(app, monkeypatch):
    data = transaction_frame()
    pipeline = fitted_category_pipeline()
    calls = []

    def fake_train(_data):
        calls.append(1)
        return {
            "sufficiently_trained": True, "training_records": 10,
            "models": {"Logistic Regression": pipeline}, "selected_model": "Logistic Regression",
            "evaluation_status": "evaluated", "evaluation_reason": None,
            "evaluation_training_records": 8, "evaluation_test_records": 2,
            "model_comparison": {"Logistic Regression": {"accuracy": 0.9}},
            "category_distribution": [{"category": "Fixture expense", "count": 10}],
        }

    monkeypatch.setattr(model_store, "train_categorizer", fake_train)
    with app.app_context():
        first = model_store.train_user_model(1, "transaction_categorization", data)
        second = model_store.train_user_model(1, "transaction_categorization", data)
        assert first["status"] == "trained" and first["trained"] is True
        assert second["status"] == "already_current" and second["trained"] is False
        assert calls == [1]

        changed = data.copy()
        changed.loc[0, "description"] = "changed merchant text"
        third = model_store.train_user_model(1, "transaction_categorization", changed)
        assert third["status"] == "trained"
        assert calls == [1, 1]


def test_baseline_decision_is_cached_without_joblib_artifact(app, monkeypatch):
    data = transaction_frame()
    calls = []

    def fake_predict(_data, *, include_fitted_model=False):
        calls.append(1)
        return {
            "data_sufficient": True, "model_based": False,
            "predicted_expenses": Decimal("40.00"), "selected_model": "Naive baseline (previous month)",
            "historical_months": 12, "monthly_history": pd.DataFrame(),
            "metrics": {"mae": 1}, "baseline_metrics": {"mae": 1},
            "model_comparison": [], "test_periods": [],
            "training_observations": 0, "test_observations": 0,
            "status": "evaluated", "prediction_period": date(2026, 1, 1),
        }

    monkeypatch.setattr(model_store, "predict_next_month", fake_predict)
    with app.app_context():
        first = model_store.train_user_model(1, "expense_prediction", data)
        second = model_store.train_user_model(1, "expense_prediction", data)
        artifact, _ = model_store.model_artifact_paths(1, "expense_prediction")
        assert first["status"] == "baseline_only"
        assert second["status"] == "already_current"
        assert calls == [1]
        assert not artifact.exists()


def test_expense_model_training_and_prediction_reuse_saved_pipeline(app, monkeypatch):
    data = transaction_frame()
    pipeline = fitted_expense_pipeline()
    calls = []

    def fake_predict(_data, *, include_fitted_model=False):
        calls.append(include_fitted_model)
        monthly = pd.DataFrame({
            "period": pd.period_range("2025-01", periods=12, freq="M"),
            "amount": [Decimal("30.00")] * 12,
        })
        return {
            "data_sufficient": True, "model_based": True,
            "predicted_expenses": Decimal("40.00"), "selected_model": "Linear Regression",
            "historical_months": 12, "monthly_history": monthly,
            "metrics": {"mae": 1}, "baseline_metrics": {"mae": 3},
            "model_comparison": [], "test_periods": ["2025-12"],
            "training_observations": 6, "test_observations": 3,
            "status": "evaluated", "prediction_period": date(2026, 1, 1),
            "fitted_model": pipeline if include_fitted_model else None,
        }

    monkeypatch.setattr(model_store, "predict_next_month", fake_predict)
    with app.app_context():
        trained = model_store.train_user_model(1, "expense_prediction", data)
        predicted = model_store.predict_expenses_for_user(1, data)
        assert trained["status"] == "trained"
        assert predicted["status"] == "evaluated"
        assert predicted["model_artifact_status"] == "current"
        assert predicted["selected_model"] == "Linear Regression"
        assert predicted["predicted_expenses"] >= Decimal("0.00")
        assert calls == [True]


def test_category_prediction_uses_current_saved_user_model(app, monkeypatch):
    data = transaction_frame()
    pipeline = fitted_category_pipeline()
    monkeypatch.setattr(model_store, "train_categorizer", lambda _data: {
        "sufficiently_trained": True, "training_records": 4,
        "models": {"Logistic Regression": pipeline}, "selected_model": "Logistic Regression",
        "evaluation_status": "evaluated", "evaluation_reason": None,
        "evaluation_training_records": 2, "evaluation_test_records": 2,
        "model_comparison": {},
    })
    with app.app_context():
        model_store.train_user_model(1, "transaction_categorization", data)
        result = model_store.predict_category_for_user(1, data, "silver orchard produce")
        assert result["status"] == "suggestion"
        assert result["model_name"] == "Logistic Regression"


def test_prediction_history_is_upserted_and_scoped_to_owner(app):
    with app.app_context():
        for email in ("history-a@example.com", "history-b@example.com"):
            db.session.add(User(email=email, full_name="History", password_hash="unused"))
        db.session.commit()
        user_a, user_b = list(db.session.scalars(select(User).order_by(User.id)))
        entry = {
            "prediction_period": date(2026, 1, 1), "predicted_expenses": Decimal("75.25"),
            "selected_model": "Naive baseline (previous month)", "metrics": None,
            "status": "insufficient_data",
        }
        saved_first = model_store.save_prediction_history(user_a.id, entry)
        entry["predicted_expenses"] = Decimal("80.00")
        saved_second = model_store.save_prediction_history(user_a.id, entry)
        assert saved_first[0].id == saved_second[0].id
        assert len(model_store.user_prediction_history(user_a.id)) == 1
        assert model_store.user_prediction_history(user_a.id)[0].predicted_expense == Decimal("80.00")
        assert model_store.user_prediction_history(user_b.id) == []


def test_prediction_history_rejects_invalid_user_and_bad_period():
    with pytest.raises(ValueError, match="authenticated"):
        model_store.user_prediction_history(0)
    with pytest.raises(ValueError, match="invalid"):
        model_store.save_prediction_history(1, {
            "prediction_period": "2026-01-01", "predicted_expenses": Decimal("5.00"),
        })
