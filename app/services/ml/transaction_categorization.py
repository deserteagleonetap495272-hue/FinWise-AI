"""Per-user TF-IDF transaction category suggestions and evaluation.

Pass only a pre-scoped frame from
``fetch_user_transactions(current_user.id)``. This module neither queries the
database nor changes transactions: predicted labels are suggestions that must
be explicitly accepted by a later user-facing workflow.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from app.services.ml.data_preprocessing import prepare_labeled_transactions

MIN_CATEGORY_TRAINING_SAMPLES = 3
MIN_CATEGORY_EVALUATION_SAMPLES = 10
EVALUATION_TEST_FRACTION = 0.20
RANDOM_STATE = 42
MODEL_NAMES = ("Multinomial Naive Bayes", "Logistic Regression")
_TOKEN_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_description(description: str) -> str:
    """Lowercase and normalize punctuation/spacing without storing the text."""
    if not isinstance(description, str):
        return ""
    value = description.casefold()
    value = _TOKEN_RE.sub(" ", value)
    return _WHITESPACE_RE.sub(" ", value).strip()


def _class_label(category: str, transaction_type: str) -> str:
    """Encode category and transaction kind without ambiguous same-name labels."""
    return json.dumps([transaction_type, category], ensure_ascii=False, separators=(",", ":"))


def _decode_label(label: str) -> tuple[str, str]:
    transaction_type, category = json.loads(label)
    return category, transaction_type


def _new_pipeline(model_name: str) -> Pipeline:
    """Create a TF-IDF text pipeline with a reproducible classifier."""
    if model_name == "Multinomial Naive Bayes":
        classifier = MultinomialNB(alpha=1.0)
    elif model_name == "Logistic Regression":
        classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unsupported categorization model: {model_name}")
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=normalize_description,
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("classifier", classifier),
    ])


def _metrics(actual: list[str], predicted: list[str], labels: list[str]) -> dict[str, Any]:
    """Return held-out classification metrics and a labeled confusion matrix."""
    report = classification_report(
        actual, predicted, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "precision_weighted": float(report["weighted avg"]["precision"]),
        "recall_weighted": float(report["weighted avg"]["recall"]),
        "f1_weighted": float(report["weighted avg"]["f1-score"]),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
        "confusion_labels": [
            {"category": _decode_label(label)[0], "transaction_type": _decode_label(label)[1]}
            for label in labels
        ],
    }


def _prepare_training_rows(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reuse Stage 2 cleaning and exclude empty normalized descriptions."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Categorization training data must be a pandas DataFrame.")
    if "user_id" in data.columns:
        owners = data["user_id"].dropna().unique()
        if data["user_id"].isna().any() or len(owners) > 1:
            raise ValueError("Training data must contain records for exactly one user.")

    prepared = prepare_labeled_transactions(data)
    prepared = prepared.copy()
    prepared["normalized_description"] = prepared["description"].map(normalize_description)
    before_text_filter = len(prepared)
    prepared = prepared.loc[prepared["normalized_description"].str.len().gt(0)].copy()
    dropped_empty_text = before_text_filter - len(prepared)
    prepared["class_label"] = [
        _class_label(str(category).strip(), str(kind).strip().lower())
        for category, kind in zip(prepared["category"], prepared["transaction_type"])
    ]
    counts = Counter(prepared["class_label"])
    distribution = [
        {"category": _decode_label(label)[0], "transaction_type": _decode_label(label)[1], "count": count}
        for label, count in sorted(counts.items(), key=lambda pair: (pair[0]))
    ]
    eligible_labels = {label for label, count in counts.items() if count >= MIN_CATEGORY_TRAINING_SAMPLES}
    eligible = prepared.loc[prepared["class_label"].isin(eligible_labels)].copy().reset_index(drop=True)
    excluded_categories = [row for row in distribution if row["count"] < MIN_CATEGORY_TRAINING_SAMPLES]
    return eligible, {
        "records_after_text_cleaning": len(prepared),
        "dropped_empty_normalized_descriptions": dropped_empty_text,
        "eligible_training_records": len(eligible),
        "eligible_category_distribution": [row for row in distribution if row["count"] >= MIN_CATEGORY_TRAINING_SAMPLES],
        "excluded_categories": excluded_categories,
        "category_distribution": distribution,
    }


def _stratified_split(labels: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    """Return deterministic stratified train/test indices when support permits."""
    counts = labels.value_counts()
    if len(counts) < 2 or int(counts.min()) < MIN_CATEGORY_EVALUATION_SAMPLES:
        return None
    test_count = max(int(np.ceil(len(labels) * EVALUATION_TEST_FRACTION)), 2 * len(counts))
    if test_count >= len(labels) or len(labels) - test_count < 2 * len(counts):
        return None
    positions = np.arange(len(labels))
    return train_test_split(
        positions,
        test_size=test_count,
        random_state=RANDOM_STATE,
        stratify=labels.to_numpy(),
    )


def train_categorizer(data: pd.DataFrame) -> dict[str, Any]:
    """Evaluate candidate classifiers and fit final models on eligible rows.

    Categories with fewer than three labeled descriptions are excluded. When
    each eligible class has at least ten examples, a deterministic stratified
    holdout compares the candidates; each Pipeline fits TF-IDF on the training
    subset only. Otherwise evaluation metrics are omitted with an explanation.
    If at least two eligible classes remain, both final pipelines are trained
    on all eligible rows; without evaluation, Logistic Regression is an
    explicitly marked unvalidated fallback rather than an evaluated winner.
    """
    prepared, summary = _prepare_training_rows(data)
    labels = sorted(prepared["class_label"].unique().tolist())
    if len(labels) < 2:
        return {
            "status": "insufficient_data", "sufficiently_trained": False,
            "explanation": "At least two categories with three usable labeled descriptions each are needed to train a classifier.",
            "training_records": summary["eligible_training_records"],
            "category_distribution": summary["category_distribution"],
            "eligible_category_distribution": summary["eligible_category_distribution"],
            "excluded_categories": summary["excluded_categories"],
            "dropped_empty_normalized_descriptions": summary["dropped_empty_normalized_descriptions"],
            "evaluation_status": "not_evaluated", "evaluation_reason": "Insufficient eligible categories.",
            "evaluation_training_records": 0, "evaluation_test_records": 0,
            "model_comparison": {}, "selected_model": None, "selection_basis": None,
            "models": {},
        }

    raw_texts = prepared["description"].astype(str).tolist()
    target = prepared["class_label"].astype(str)
    split = _stratified_split(target)
    comparison: dict[str, dict[str, Any]] = {}
    evaluation_reason = None
    evaluation_status = "evaluated" if split else "insufficient_evaluation_data"
    evaluation_train_count = 0
    evaluation_test_count = 0
    if split is None:
        smallest = min(summary["eligible_category_distribution"], key=lambda row: row["count"])
        evaluation_reason = (
            f"Stratified evaluation requires at least {MIN_CATEGORY_EVALUATION_SAMPLES} examples per eligible category; "
            f"the smallest eligible category has {smallest['count']}. No evaluation metrics are reported."
        )
    else:
        train_indices, test_indices = split
        evaluation_train_count, evaluation_test_count = len(train_indices), len(test_indices)
        train_texts = [raw_texts[index] for index in train_indices]
        test_texts = [raw_texts[index] for index in test_indices]
        train_labels = target.iloc[train_indices].tolist()
        test_labels = target.iloc[test_indices].tolist()
        for model_name in MODEL_NAMES:
            evaluation_pipeline = _new_pipeline(model_name)
            evaluation_pipeline.fit(train_texts, train_labels)
            predicted = evaluation_pipeline.predict(test_texts).tolist()
            comparison[model_name] = _metrics(test_labels, predicted, labels)

    final_models = {}
    for model_name in MODEL_NAMES:
        pipeline = _new_pipeline(model_name)
        pipeline.fit(raw_texts, target.tolist())
        final_models[model_name] = pipeline

    if comparison:
        selected_model = max(
            MODEL_NAMES,
            key=lambda name: (
                comparison[name]["f1_macro"],
                comparison[name]["f1_weighted"],
                comparison[name]["accuracy"],
                -MODEL_NAMES.index(name),
            ),
        )
        selection_basis = "stratified_holdout_macro_f1"
        explanation = f"Both candidate models were compared on a stratified held-out sample; {selected_model} had the best macro F1."
        status = "ready"
    else:
        selected_model = "Logistic Regression"
        selection_basis = "unvalidated_fallback"
        explanation = (
            "The eligible data can train a classifier, but there are not enough examples per category for a stratified evaluation. "
            "Logistic Regression is provided as an unvalidated fallback; no test metrics are claimed."
        )
        status = "trained_not_evaluated"

    return {
        "status": status, "sufficiently_trained": True,
        "explanation": explanation,
        "training_records": len(prepared),
        "category_distribution": summary["category_distribution"],
        "eligible_category_distribution": summary["eligible_category_distribution"],
        "excluded_categories": summary["excluded_categories"],
        "dropped_empty_normalized_descriptions": summary["dropped_empty_normalized_descriptions"],
        "evaluation_status": evaluation_status,
        "evaluation_reason": evaluation_reason,
        "evaluation_training_records": evaluation_train_count,
        "evaluation_test_records": evaluation_test_count,
        "model_comparison": comparison,
        "selected_model": selected_model,
        "selection_basis": selection_basis,
        "models": final_models,
    }


def predict_category(description: str, trained: dict[str, Any]) -> dict[str, Any]:
    """Suggest a category for a description without modifying any transaction."""
    if not isinstance(description, str) or not description.strip():
        return {
            "status": "invalid_description", "predicted_category": None,
            "transaction_type": None, "confidence": None, "model_name": trained.get("selected_model"),
            "alternatives": [], "sufficiently_trained": bool(trained.get("sufficiently_trained")),
            "explanation": "Enter a transaction description before requesting a category suggestion.",
        }
    if not trained.get("sufficiently_trained") or trained.get("selected_model") not in trained.get("models", {}):
        return {
            "status": "insufficient_data", "predicted_category": None,
            "transaction_type": None, "confidence": None, "model_name": trained.get("selected_model"),
            "alternatives": [], "sufficiently_trained": False,
            "explanation": trained.get("explanation", "More labeled transactions are needed."),
        }

    normalized = normalize_description(description)
    if len(normalized.split()) < 2:
        return {
            "status": "low_information", "predicted_category": None,
            "transaction_type": None, "confidence": None, "model_name": trained["selected_model"],
            "alternatives": [], "sufficiently_trained": True,
            "explanation": "This description is too short to provide a useful category suggestion. Add more descriptive words.",
        }

    model_name = trained["selected_model"]
    pipeline: Pipeline = trained["models"][model_name]
    vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
    if vectorizer.transform([normalized]).nnz == 0:
        return {
            "status": "unknown_description", "predicted_category": None,
            "transaction_type": None, "confidence": None, "model_name": model_name,
            "alternatives": [], "sufficiently_trained": True,
            "explanation": "The description contains no terms found in this user's labeled history, so no category was suggested.",
        }

    probabilities = pipeline.predict_proba([normalized])[0]
    classes = pipeline.named_steps["classifier"].classes_
    ranked = sorted(zip(classes, probabilities), key=lambda item: (-float(item[1]), item[0]))
    alternatives = []
    for label, score in ranked:
        category, transaction_type = _decode_label(label)
        alternatives.append({
            "category": category,
            "transaction_type": transaction_type,
            "score": float(score),
        })
    predicted = alternatives[0]
    validated = trained.get("evaluation_status") == "evaluated"
    return {
        "status": "suggestion" if validated else "unvalidated_suggestion",
        "predicted_category": predicted["category"],
        "transaction_type": predicted["transaction_type"],
        "confidence": predicted["score"],
        "model_name": model_name,
        "alternatives": alternatives,
        "sufficiently_trained": True,
        "explanation": (
            f"Suggested from this user's labeled descriptions using {model_name}. "
            f"The confidence score ({predicted['score']:.1%}) is a model score, not a guarantee of correctness. "
            + ("This model has not been evaluated on a stratified holdout." if not validated else "Please review and confirm before applying this category.")
        ),
    }
