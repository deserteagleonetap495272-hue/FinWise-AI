"""Tests for isolated, user-trained transaction category suggestions."""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from app.extensions import db
from app.models import Category, Transaction, User
from app.services.ml.data_preprocessing import fetch_user_transactions
from app.services.ml.transaction_categorization import (
    MODEL_NAMES,
    normalize_description,
    predict_category,
    train_categorizer,
)


def labeled_rows(*, categories=("Groceries", "Transit"), per_category=12, user_id=1):
    """Build synthetic, test-only labeled examples rather than prod training records."""
    rows = []
    for category_index, category in enumerate(categories):
        count = per_category if isinstance(per_category, int) else per_category[category_index]
        for item in range(count):
            description = (
                f"copper rail commute route ticket marker{item}"
                if category_index % 2 else f"amber produce market pantry basket marker{item}"
            )
            rows.append({
                "id": len(rows) + 1, "user_id": user_id, "category_id": category_index + 1,
                "amount": Decimal("12.50"), "transaction_type": "expense",
                "description": description, "transaction_date": date(2026, 1, 1),
                "category": category, "category_type": "expense",
            })
    return pd.DataFrame(rows)


def test_train_creates_tfidf_nb_and_logistic_pipelines_with_metrics():
    trained = train_categorizer(labeled_rows())
    assert trained["status"] == "ready"
    assert trained["sufficiently_trained"] is True
    assert trained["training_records"] == 24
    assert set(trained["models"]) == set(MODEL_NAMES)
    for model_name in MODEL_NAMES:
        pipeline = trained["models"][model_name]
        assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
        assert pipeline.named_steps["tfidf"].vocabulary_
        assert set(trained["model_comparison"][model_name]) >= {
            "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted", "confusion_matrix",
        }
        assert trained["model_comparison"][model_name]["confusion_labels"]
    assert trained["selected_model"] in MODEL_NAMES
    assert trained["selection_basis"] == "stratified_holdout_macro_f1"


def test_low_support_categories_are_excluded_and_evaluation_is_not_invented():
    rows = labeled_rows(categories=("Groceries", "Transit", "Rare"), per_category=(5, 5, 2))
    trained = train_categorizer(rows)
    assert trained["sufficiently_trained"] is True
    assert trained["training_records"] == 10
    assert trained["excluded_categories"] == [
        {"category": "Rare", "transaction_type": "expense", "count": 2}
    ]
    assert trained["evaluation_status"] == "insufficient_evaluation_data"
    assert trained["model_comparison"] == {}
    assert "No evaluation metrics" in trained["evaluation_reason"]
    assert trained["selection_basis"] == "unvalidated_fallback"


def test_insufficient_categories_return_no_models_or_prediction():
    trained = train_categorizer(labeled_rows(categories=("Only one",), per_category=8))
    assert trained["status"] == "insufficient_data"
    assert trained["sufficiently_trained"] is False
    assert trained["models"] == {}
    suggestion = predict_category("some purchase", trained)
    assert suggestion["status"] == "insufficient_data"
    assert suggestion["predicted_category"] is None


def test_empty_descriptions_and_missing_labels_are_excluded():
    rows = labeled_rows()
    rows.loc[0, "description"] = None
    rows.loc[1, "description"] = "!!! ###"
    rows.loc[14, "category"] = None
    trained = train_categorizer(rows)
    assert trained["dropped_empty_normalized_descriptions"] == 1
    assert trained["training_records"] == 21
    assert trained["status"] == "ready"
    assert normalize_description("  CAFÉ,  Market!!! ") == "café market"


def test_prediction_returns_confidence_and_ranked_alternatives():
    trained = train_categorizer(labeled_rows())
    result = predict_category("amber produce pantry market", trained)
    assert result["status"] == "suggestion"
    assert result["predicted_category"] == "Groceries"
    assert result["transaction_type"] == "expense"
    assert result["model_name"] in MODEL_NAMES
    assert 0 <= result["confidence"] <= 1
    assert result["confidence"] == result["alternatives"][0]["score"]
    assert sum(option["score"] for option in result["alternatives"]) == pytest.approx(1)
    assert "not a guarantee" in result["explanation"]
    assert "confirm" in result["explanation"]


@pytest.mark.parametrize("description", ["", "   ", None])
def test_empty_input_returns_no_category(description):
    result = predict_category(description, train_categorizer(labeled_rows()))
    assert result["status"] == "invalid_description"
    assert result["predicted_category"] is None
    assert result["alternatives"] == []


def test_very_short_and_unseen_vocabulary_descriptions_are_handled():
    trained = train_categorizer(labeled_rows())
    short = predict_category("x", trained)
    unknown = predict_category("xylophone quasar unobserved", trained)
    assert short["status"] == "low_information"
    assert short["predicted_category"] is None
    assert unknown["status"] == "unknown_description"
    assert unknown["predicted_category"] is None


def test_unvalidated_training_can_suggest_but_states_metrics_are_unavailable():
    trained = train_categorizer(labeled_rows(per_category=4))
    assert trained["status"] == "trained_not_evaluated"
    assert trained["model_comparison"] == {}
    result = predict_category("amber produce pantry", trained)
    assert result["status"] == "unvalidated_suggestion"
    assert "not been evaluated" in result["explanation"]


def test_tfidf_evaluation_fit_uses_training_split_only(monkeypatch):
    rows = labeled_rows()
    original_fit_transform = TfidfVectorizer.fit_transform
    fit_sizes = []

    def record_training_documents(vectorizer, raw_documents, y=None):
        documents = list(raw_documents)
        fit_sizes.append(len(documents))
        return original_fit_transform(vectorizer, documents, y)

    monkeypatch.setattr(TfidfVectorizer, "fit_transform", record_training_documents)
    trained = train_categorizer(rows)
    assert trained["evaluation_training_records"] < trained["training_records"]
    assert fit_sizes[:2] == [trained["evaluation_training_records"]] * 2
    assert fit_sizes[2:] == [trained["training_records"]] * 2


def test_mixed_user_training_data_is_rejected():
    rows = pd.concat([labeled_rows(user_id=1), labeled_rows(user_id=2)], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one user"):
        train_categorizer(rows)


def test_database_training_data_and_manual_categories_remain_user_scoped(app):
    with app.app_context():
        owners = []
        for email, category_names in (
            ("categorizer-a@example.com", ("Groceries", "Transit")),
            ("categorizer-b@example.com", ("Private B", "Secret B")),
        ):
            user = User(email=email, full_name="Categorizer", password_hash="unused")
            db.session.add(user)
            db.session.flush()
            categories = [
                Category(user_id=user.id, name=name, transaction_type="expense")
                for name in category_names
            ]
            db.session.add_all(categories)
            db.session.flush()
            owners.append((user.id, categories))
            for category in categories:
                for index in range(3):
                    db.session.add(Transaction(
                        user_id=user.id, category_id=category.id, amount=Decimal("5.00"),
                        transaction_type="expense", description=f"label {category.name} item {index}",
                        transaction_date=date(2026, 1, index + 1),
                    ))
        db.session.commit()

        owner_a, _ = owners
        manual_category_id = owner_a[1][0].id
        manual_transaction = Transaction(
            user_id=owner_a[0], category_id=manual_category_id, amount=Decimal("20.00"),
            transaction_type="expense", description="my manually assigned purchase",
            transaction_date=date(2026, 2, 1),
        )
        db.session.add(manual_transaction)
        db.session.commit()
        transaction_id = manual_transaction.id

        user_a_rows = fetch_user_transactions(owner_a[0])
        trained = train_categorizer(user_a_rows)
        assert all(row["category"] not in {"Private B", "Secret B"}
                   for row in trained["category_distribution"])
        suggestion = predict_category("label groceries item", trained)
        assert suggestion["status"] in {"unvalidated_suggestion", "suggestion"}
        db.session.expire_all()
        unchanged = db.session.get(Transaction, transaction_id)
        assert unchanged.category_id == manual_category_id
        assert unchanged.user_id == owner_a[0]
