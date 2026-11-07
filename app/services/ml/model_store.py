"""Private per-user model artifacts and explicit training management.

Only application-trained scikit-learn pipelines are serialized. Paths are
derived from validated integer user IDs and an allowlisted model type; there
is no API for loading an arbitrary path. Artifacts are stored under Flask's
private instance directory, integrity checked, and authenticated before
Joblib deserialization.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import tempfile
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from threading import RLock
from typing import Any

import joblib
import pandas as pd
from flask import current_app
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sqlalchemy import select

from app.extensions import db
from app.models import Prediction
from app.services.ml.data_preprocessing import clean_transaction_data
from app.services.ml.expense_prediction import (
    FEATURE_COLUMNS,
    build_monthly_features,
    predict_next_month,
    predict_next_month_with_cached_baseline,
    predict_next_month_with_saved_model,
)
from app.services.ml.transaction_categorization import (
    MODEL_NAMES as CATEGORIZATION_MODEL_NAMES,
    normalize_description,
    predict_category,
    train_categorizer,
)

MODEL_VERSIONS = {
    "expense_prediction": "1.0.0",
    "transaction_categorization": "1.0.0",
}
MODEL_TYPES = frozenset(MODEL_VERSIONS)
EXPENSE_MODEL_NAMES = ("Linear Regression", "Random Forest", "Gradient Boosting")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCK = RLock()


class ModelStoreError(RuntimeError):
    """Raised when a trusted application model cannot be safely persisted."""


def _validate_user_id(user_id: int) -> int:
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise ValueError("A valid authenticated user's ID is required.")
    return user_id


def _validate_model_type(model_type: str) -> str:
    if model_type not in MODEL_TYPES:
        raise ValueError("Unsupported model type.")
    return model_type


def model_artifact_paths(user_id: int, model_type: str) -> tuple[Path, Path]:
    """Return the app-generated artifact/metadata paths for one owner/type."""
    user_id = _validate_user_id(user_id)
    model_type = _validate_model_type(model_type)
    configured_root = current_app.config.get("ML_MODEL_STORE_DIR")
    root = Path(configured_root) if configured_root else Path(current_app.instance_path) / "ml_models"
    if not root.is_absolute():
        root = Path(current_app.instance_path) / root
    root = root.resolve()
    static_root = Path(current_app.static_folder).resolve()
    if root == static_root or static_root in root.parents:
        raise ModelStoreError("Model storage cannot be located in a public static directory.")
    owner_dir = root / f"user_{user_id}"
    artifact = owner_dir / f"{model_type}.joblib"
    metadata = owner_dir / f"{model_type}.json"
    if root not in artifact.resolve().parents or root not in metadata.resolve().parents:
        raise ModelStoreError("Resolved model path escaped the private model directory.")
    return artifact, metadata


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _json_safe(value: Any) -> Any:
    """Convert trusted metric/config values to JSON primitives only."""
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return None
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise TypeError(f"Unsupported metadata value type: {type(value).__name__}")


def _secret_bytes() -> bytes:
    secret = current_app.config.get("SECRET_KEY")
    if not isinstance(secret, str) or not secret:
        raise ModelStoreError("A configured application secret is required for model integrity checks.")
    return secret.encode("utf-8")


def _sign_metadata(metadata: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in metadata.items() if key != "signature"}
    return hmac.new(_secret_bytes(), _canonical_json(unsigned), hashlib.sha256).hexdigest()


def _pipeline_matches_model_type(model_type: str, pipeline: Any) -> bool:
    if not isinstance(pipeline, Pipeline):
        return False
    if model_type == "expense_prediction":
        return (
            list(pipeline.named_steps) == ["imputer", "regressor"]
            and type(pipeline.named_steps["imputer"]) is SimpleImputer
            and pipeline.named_steps["imputer"].strategy == "median"
            and type(pipeline.named_steps["regressor"]) in {
                LinearRegression, RandomForestRegressor, GradientBoostingRegressor,
            }
        )
    vectorizer = pipeline.named_steps.get("tfidf")
    classifier = pipeline.named_steps.get("classifier")
    return (
        list(pipeline.named_steps) == ["tfidf", "classifier"]
        and type(vectorizer) is TfidfVectorizer
        and vectorizer.preprocessor is normalize_description
        and vectorizer.tokenizer is None
        and vectorizer.analyzer == "word"
        and vectorizer.token_pattern == r"(?u)\b\w+\b"
        and vectorizer.ngram_range == (1, 2)
        and vectorizer.lowercase is False
        and vectorizer.sublinear_tf is True
        and type(classifier) in {MultinomialNB, LogisticRegression}
    )


def transaction_fingerprint(user_id: int, data: pd.DataFrame) -> str:
    """Hash canonical cleaned input without saving descriptions or amounts."""
    user_id = _validate_user_id(user_id)
    _validate_user_data(user_id, data)
    cleaned = clean_transaction_data(data)
    columns = ["id", "user_id", "category_id", "amount", "transaction_type",
               "description", "transaction_date", "category", "category_type"]
    records = []
    for row in cleaned.loc[:, columns].to_dict("records"):
        records.append({
            "id": None if pd.isna(row["id"]) else int(row["id"]),
            "user_id": None if pd.isna(row["user_id"]) else int(row["user_id"]),
            "category_id": None if pd.isna(row["category_id"]) else int(row["category_id"]),
            "amount": str(row["amount"]),
            "transaction_type": str(row["transaction_type"]),
            "description": str(row["description"]),
            "transaction_date": pd.Timestamp(row["transaction_date"]).date().isoformat(),
            "category": str(row["category"]),
            "category_type": None if pd.isna(row["category_type"]) else str(row["category_type"]),
        })
    return hashlib.sha256(_canonical_json({"user_id": user_id, "records": records})).hexdigest()


def _validate_user_data(user_id: int, data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Model training data must be a pandas DataFrame.")
    if "user_id" not in data.columns:
        # The caller must still source this frame from a user-scoped service.
        return
    owners = data["user_id"].dropna().unique()
    if data["user_id"].isna().any() or len(owners) > 1 or (len(owners) == 1 and int(owners[0]) != user_id):
        raise ValueError("Training data must belong only to the requested user.")


def _training_period(data: pd.DataFrame) -> dict[str, str] | None:
    cleaned = clean_transaction_data(data)
    if cleaned.empty:
        return None
    dates = cleaned["transaction_date"].dt.date
    return {"start": min(dates).isoformat(), "end": max(dates).isoformat()}


def _write_atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        # Windows ACLs govern access; chmod is best-effort there.
        pass
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tmp-", delete=False) as stream:
            temporary = Path(stream.name)
        writer(temporary)
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)


def save_model_artifact(user_id: int, model_type: str, pipeline: Pipeline, *,
                        source_fingerprint: str, training_data_count: int,
                        feature_configuration: dict[str, Any],
                        training_period: dict[str, str] | None = None,
                        evaluation_metrics: dict[str, Any] | None = None,
                        selected_model: str,
                        evaluation_status: str = "evaluated") -> dict[str, Any]:
    """Persist a trusted, fitted pipeline and signed non-sensitive metadata."""
    user_id = _validate_user_id(user_id)
    model_type = _validate_model_type(model_type)
    if not _pipeline_matches_model_type(model_type, pipeline):
        raise ModelStoreError("Pipeline type does not match its declared model type.")
    if not isinstance(source_fingerprint, str) or not _SHA256_RE.fullmatch(source_fingerprint):
        raise ValueError("A SHA-256 source fingerprint is required.")
    if isinstance(training_data_count, bool) or not isinstance(training_data_count, int) or training_data_count < 0:
        raise ValueError("Training data count must be a non-negative integer.")
    if feature_configuration != _feature_configuration(model_type):
        raise ValueError("Feature configuration must use the application's approved model features.")

    artifact_path, metadata_path = model_artifact_paths(user_id, model_type)
    with _LOCK:
        try:
            _write_atomic(artifact_path, lambda path: joblib.dump(pipeline, path, compress=3))
            checksum = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            metadata = {
                "schema_version": 1,
                "model_type": model_type,
                "model_version": MODEL_VERSIONS[model_type],
                "training_timestamp": datetime.now(timezone.utc).isoformat(),
                "training_data_count": training_data_count,
                "feature_configuration": _json_safe(feature_configuration),
                "training_period": _json_safe(training_period),
                "evaluation_metrics": _json_safe(evaluation_metrics or {}),
                "user_id": user_id,
                "artifact_location": artifact_path.relative_to(artifact_path.parents[1]).as_posix(),
                "artifact_sha256": checksum,
                "source_fingerprint": source_fingerprint,
                "model_status": "trained",
                "selected_model": selected_model,
                "evaluation_status": evaluation_status,
            }
            metadata["signature"] = _sign_metadata(metadata)
            _write_atomic(metadata_path, lambda path: path.write_bytes(_canonical_json(metadata)))
        except Exception as error:
            raise ModelStoreError("The application model artifact could not be saved safely.") from error
    return metadata


def save_model_decision(user_id: int, model_type: str, *, source_fingerprint: str,
                        model_status: str, training_data_count: int,
                        feature_configuration: dict[str, Any],
                        training_period: dict[str, str] | None = None,
                        evaluation_metrics: dict[str, Any] | None = None,
                        selected_model: str | None = None,
                        evaluation_status: str = "not_evaluated") -> dict[str, Any]:
    """Cache baseline/insufficient training decisions without a Joblib file."""
    user_id = _validate_user_id(user_id)
    model_type = _validate_model_type(model_type)
    if model_status not in {"baseline_only", "insufficient_data"}:
        raise ValueError("Only non-model training decisions can use this function.")
    if not isinstance(source_fingerprint, str) or not _SHA256_RE.fullmatch(source_fingerprint):
        raise ValueError("A SHA-256 source fingerprint is required.")
    if (isinstance(training_data_count, bool) or not isinstance(training_data_count, int)
            or training_data_count < 0):
        raise ValueError("Training data count must be a non-negative integer.")
    if feature_configuration != _feature_configuration(model_type):
        raise ValueError("Feature configuration must use the application's approved model features.")
    artifact_path, metadata_path = model_artifact_paths(user_id, model_type)
    metadata = {
        "schema_version": 1,
        "model_type": model_type,
        "model_version": MODEL_VERSIONS[model_type],
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_data_count": training_data_count,
        "feature_configuration": _json_safe(feature_configuration),
        "training_period": _json_safe(training_period),
        "evaluation_metrics": _json_safe(evaluation_metrics or {}),
        "user_id": user_id,
        "artifact_location": None,
        "artifact_sha256": None,
        "source_fingerprint": source_fingerprint,
        "model_status": model_status,
        "selected_model": selected_model,
        "evaluation_status": evaluation_status,
    }
    metadata["signature"] = _sign_metadata(metadata)
    with _LOCK:
        try:
            artifact_path.unlink(missing_ok=True)
            _write_atomic(metadata_path, lambda path: path.write_bytes(_canonical_json(metadata)))
        except Exception as error:
            raise ModelStoreError("The application model status could not be saved safely.") from error
    return metadata


def _read_metadata(user_id: int, model_type: str) -> tuple[str, dict[str, Any] | None]:
    artifact_path, metadata_path = model_artifact_paths(user_id, model_type)
    if not metadata_path.is_file():
        return "missing", None
    try:
        if metadata_path.stat().st_size > 1_000_000:
            return "corrupt", None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or not _valid_metadata(metadata, user_id, model_type):
            return "corrupt", None
        if not hmac.compare_digest(str(metadata.get("signature", "")), _sign_metadata(metadata)):
            return "corrupt", None
        if metadata["model_version"] != MODEL_VERSIONS[model_type]:
            return "stale_version", metadata
        state = metadata["model_status"]
        if state == "trained":
            if not artifact_path.is_file():
                return "corrupt", metadata
            if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != metadata["artifact_sha256"]:
                return "corrupt", metadata
            return "current", metadata
        if state in {"baseline_only", "insufficient_data"}:
            if artifact_path.exists():
                return "corrupt", metadata
            return state, metadata
        return "corrupt", None
    except Exception:
        # Missing, malformed, tampered, or unreadable files fail closed.
        return "corrupt", None


def _valid_metadata(metadata: dict[str, Any], user_id: int, model_type: str) -> bool:
    required = {
        "schema_version", "model_type", "model_version", "training_timestamp",
        "training_data_count", "feature_configuration", "training_period",
        "evaluation_metrics", "user_id", "artifact_location", "artifact_sha256",
        "source_fingerprint", "model_status", "selected_model", "evaluation_status", "signature",
    }
    if set(metadata) != required:
        return False
    if metadata["schema_version"] != 1 or metadata["model_type"] != model_type:
        return False
    if (isinstance(metadata["user_id"], bool) or not isinstance(metadata["user_id"], int)
            or metadata["user_id"] != user_id):
        return False
    if not isinstance(metadata["training_data_count"], int) or metadata["training_data_count"] < 0:
        return False
    if not isinstance(metadata["feature_configuration"], dict) or not isinstance(metadata["evaluation_metrics"], dict):
        return False
    if not isinstance(metadata["training_timestamp"], str):
        return False
    try:
        datetime.fromisoformat(metadata["training_timestamp"])
    except ValueError:
        return False
    if not isinstance(metadata["source_fingerprint"], str) or not _SHA256_RE.fullmatch(metadata["source_fingerprint"]):
        return False
    state = metadata["model_status"]
    if state == "trained":
        allowed_names = EXPENSE_MODEL_NAMES if model_type == "expense_prediction" else CATEGORIZATION_MODEL_NAMES
        if metadata["selected_model"] not in allowed_names:
            return False
        expected_location = f"user_{user_id}/{model_type}.joblib"
        if metadata["artifact_location"] != expected_location:
            return False
        if not isinstance(metadata["artifact_sha256"], str) or not _SHA256_RE.fullmatch(metadata["artifact_sha256"]):
            return False
        if not isinstance(metadata["selected_model"], str) or not metadata["selected_model"]:
            return False
    elif state in {"baseline_only", "insufficient_data"}:
        if metadata["artifact_location"] is not None or metadata["artifact_sha256"] is not None:
            return False
    else:
        return False
    return isinstance(metadata["signature"], str)


def get_model_status(user_id: int, model_type: str, data: pd.DataFrame) -> dict[str, Any]:
    """Check model version and source fingerprint without training/loading."""
    user_id = _validate_user_id(user_id)
    model_type = _validate_model_type(model_type)
    fingerprint = transaction_fingerprint(user_id, data)
    status, metadata = _read_metadata(user_id, model_type)
    if status in {"current", "baseline_only", "insufficient_data"} and metadata["source_fingerprint"] != fingerprint:
        status = "stale"
    return {
        "status": status,
        "is_current": status in {"current", "baseline_only", "insufficient_data"},
        "model_type": model_type,
        "user_id": user_id,
        "source_fingerprint": fingerprint,
        "metadata": metadata,
    }


def load_model_artifact(user_id: int, model_type: str, *,
                        expected_fingerprint: str | None = None) -> dict[str, Any]:
    """Load only this user's internally resolved, signed, integrity-checked artifact."""
    user_id = _validate_user_id(user_id)
    model_type = _validate_model_type(model_type)
    if expected_fingerprint is not None and (
        not isinstance(expected_fingerprint, str) or not _SHA256_RE.fullmatch(expected_fingerprint)
    ):
        raise ValueError("Expected fingerprint must be a SHA-256 hex digest.")
    status, metadata = _read_metadata(user_id, model_type)
    if expected_fingerprint is not None and status in {"current", "baseline_only", "insufficient_data"}:
        if not hmac.compare_digest(metadata["source_fingerprint"], expected_fingerprint):
            return {"status": "stale", "model": None, "metadata": metadata}
    if status != "current":
        return {"status": status, "model": None, "metadata": metadata}

    artifact_path, _ = model_artifact_paths(user_id, model_type)
    try:
        # The path is internal, the signed metadata and checksum were verified,
        # and no caller-supplied filename reaches Joblib.
        model = joblib.load(artifact_path)
        if not _pipeline_matches_model_type(model_type, model):
            return {"status": "corrupt", "model": None, "metadata": None}
        return {"status": "current", "model": model, "metadata": metadata}
    except Exception:
        return {"status": "corrupt", "model": None, "metadata": None}


def load_current_user_model(user_id: int, model_type: str, data: pd.DataFrame) -> dict[str, Any]:
    """Return a loaded model only when its version and source data are current."""
    status = get_model_status(user_id, model_type, data)
    loaded = load_model_artifact(
        user_id, model_type, expected_fingerprint=status["source_fingerprint"]
    )
    if status["status"] not in {"current", "baseline_only", "insufficient_data"}:
        loaded["status"] = status["status"]
    return loaded


def _feature_configuration(model_type: str) -> dict[str, Any]:
    if model_type == "expense_prediction":
        return {
            "columns": list(FEATURE_COLUMNS),
            "aggregation": "calendar_month_expense_total",
            "lag_months": [1, 2, 3],
            "rolling_months": [3],
        }
    return {
        "vectorizer": "TfidfVectorizer",
        "ngram_range": [1, 2],
        "text_normalization": "casefold_punctuation_whitespace",
        "minimum_examples_per_category": 3,
    }


def _safe_expense_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected": result.get("metrics"),
        "baseline": result.get("baseline_metrics"),
        "comparison": result.get("model_comparison", []),
        "test_periods": result.get("test_periods", []),
        "training_observations": result.get("training_observations", 0),
        "test_observations": result.get("test_observations", 0),
    }


def train_user_model(user_id: int, model_type: str, data: pd.DataFrame, *,
                     force_refresh: bool = False) -> dict[str, Any]:
    """Explicitly train/refresh one user model, reusing a current artifact.

    This function is intentionally explicit; prediction/page rendering should
    call ``load_current_user_model`` and handle stale status instead of
    retraining implicitly. A baseline or insufficient-data decision is signed
    as metadata only and never saved as a Joblib model.
    """
    user_id = _validate_user_id(user_id)
    model_type = _validate_model_type(model_type)
    _validate_user_data(user_id, data)
    fingerprint = transaction_fingerprint(user_id, data)
    if not force_refresh:
        cached = load_model_artifact(user_id, model_type, expected_fingerprint=fingerprint)
        if cached["status"] in {"current", "baseline_only", "insufficient_data"}:
            return {
                "status": "already_current", "trained": False,
                "model": cached["model"], "metadata": cached["metadata"],
                "training_result": None,
            }

    training_period = _training_period(data)
    if model_type == "expense_prediction":
        result = predict_next_month(data, include_fitted_model=True)
        if not result.get("data_sufficient") or not result.get("model_based"):
            decision = (
                "baseline_only"
                if result.get("data_sufficient") and not result.get("model_based")
                else "insufficient_data"
            )
            metadata = save_model_decision(
                user_id, model_type, source_fingerprint=fingerprint,
                model_status=decision, training_data_count=int(result.get("historical_months", 0)),
                feature_configuration=_feature_configuration(model_type),
                training_period=training_period,
                evaluation_metrics=_safe_expense_metrics(result),
                selected_model=result.get("selected_model"),
                evaluation_status=result.get("status", "insufficient_data"),
            )
            clean_result = {key: value for key, value in result.items() if key != "fitted_model"}
            return {"status": decision, "trained": False, "model": None,
                    "metadata": metadata, "training_result": clean_result}

        model = result.get("fitted_model")
        if not _pipeline_matches_model_type(model_type, model):
            raise ModelStoreError("Stage 3 did not return a supported fitted expense model.")
        supervised = build_monthly_features(result["monthly_history"])
        metadata = save_model_artifact(
            user_id, model_type, model, source_fingerprint=fingerprint,
            training_data_count=len(supervised),
            feature_configuration=_feature_configuration(model_type),
            training_period=training_period,
            evaluation_metrics=_safe_expense_metrics(result),
            selected_model=result["selected_model"],
        )
        clean_result = {key: value for key, value in result.items() if key != "fitted_model"}
        return {"status": "trained", "trained": True, "model": model,
                "metadata": metadata, "training_result": clean_result}

    result = train_categorizer(data)
    if not result.get("sufficiently_trained"):
        metadata = save_model_decision(
            user_id, model_type, source_fingerprint=fingerprint,
            model_status="insufficient_data", training_data_count=int(result.get("training_records", 0)),
            feature_configuration=_feature_configuration(model_type),
            training_period=training_period,
            evaluation_metrics={"model_comparison": result.get("model_comparison", {})},
            selected_model=None,
            evaluation_status=result.get("evaluation_status", "not_evaluated"),
        )
        clean_result = {key: value for key, value in result.items() if key != "models"}
        return {"status": "insufficient_data", "trained": False, "model": None,
                "metadata": metadata, "training_result": clean_result}

    selected_model = result["selected_model"]
    model = result["models"][selected_model]
    metrics = {
        "model_comparison": result["model_comparison"],
        "evaluation_status": result["evaluation_status"],
        "evaluation_reason": result["evaluation_reason"],
        "evaluation_training_records": result["evaluation_training_records"],
        "evaluation_test_records": result["evaluation_test_records"],
    }
    metadata = save_model_artifact(
        user_id, model_type, model, source_fingerprint=fingerprint,
        training_data_count=int(result["training_records"]),
        feature_configuration=_feature_configuration(model_type),
        training_period=training_period,
        evaluation_metrics=metrics,
        selected_model=selected_model,
        evaluation_status=result["evaluation_status"],
    )
    clean_result = {key: value for key, value in result.items() if key != "models"}
    return {"status": "trained", "trained": True, "model": model,
            "metadata": metadata, "training_result": clean_result}


def predict_category_for_user(user_id: int, data: pd.DataFrame, description: str) -> dict[str, Any]:
    """Use a current saved categorizer; never trigger training implicitly."""
    loaded = load_current_user_model(user_id, "transaction_categorization", data)
    if loaded["status"] != "current":
        return {"status": "model_not_current", "model_status": loaded["status"],
                "predicted_category": None, "confidence": None,
                "explanation": "A current user-specific categorization model must be explicitly trained or refreshed first."}
    metadata = loaded["metadata"]
    trained = {
        "sufficiently_trained": True,
        "selected_model": metadata["selected_model"],
        "evaluation_status": metadata["evaluation_status"],
        "models": {metadata["selected_model"]: loaded["model"]},
    }
    return predict_category(description, trained)


def predict_expenses_for_user(user_id: int, data: pd.DataFrame) -> dict[str, Any]:
    """Predict using a current saved expense model or baseline decision."""
    loaded = load_current_user_model(user_id, "expense_prediction", data)
    metadata = loaded.get("metadata")
    if loaded["status"] == "current":
        result = predict_next_month_with_saved_model(
            data, model_name=metadata["selected_model"], model=loaded["model"], metadata=metadata
        )
        result["model_artifact_status"] = "current"
        return result
    if loaded["status"] in {"baseline_only", "insufficient_data"}:
        result = predict_next_month_with_cached_baseline(data, metadata)
        if loaded["status"] == "insufficient_data":
            result["status"] = "insufficient_data"
            result["data_sufficient"] = False
        result["model_artifact_status"] = loaded["status"]
        return result
    return {
        "status": "model_not_current", "data_sufficient": False,
        "model_artifact_status": loaded["status"],
        "predicted_expenses": None, "prediction_period": None,
        "selected_model": None, "model_based": False,
        "explanation": "No current user-specific expense model is available. Explicit training or refresh is required.",
    }


def forecast_expenses_for_user(user_id: int, data: pd.DataFrame, *, horizon: int = 1) -> dict[str, Any]:
    """Run Stage 5 forecasting with a currently validated Stage 3 result."""
    from app.services.ml.forecasting import forecast_expenses

    prediction = predict_expenses_for_user(user_id, data)
    if prediction.get("status") == "model_not_current":
        return {"status": "model_not_current", "data_sufficient": False,
                "forecast_horizon": horizon, "forecasts": [],
                "explanation": prediction["explanation"], "limitations": [
                    "Explicitly train or refresh the user's expense model before requesting a persisted-model forecast."
                ]}
    return forecast_expenses(data, horizon=horizon, prediction_result=prediction)


def save_prediction_history(user_id: int, prediction_result: dict[str, Any]) -> list[Prediction]:
    """Upsert predictions into the existing table, always scoped by owner/date."""
    user_id = _validate_user_id(user_id)
    entries = prediction_result.get("forecasts")
    if entries is None:
        amount = prediction_result.get("predicted_expenses")
        period_start = prediction_result.get("prediction_period")
        entries = [] if amount is None or period_start is None else [{
            "period_start": period_start,
            "amount": amount,
            "model_name": prediction_result.get("selected_model") or "Unknown",
            "value_type": "predicted",
            "evaluation_metrics": prediction_result.get("metrics"),
        }]
    saved = []
    try:
        for entry in entries:
            if entry.get("value_type", "predicted") != "predicted":
                continue
            period_start = entry.get("period_start")
            amount = Decimal(str(entry.get("amount")))
            if not isinstance(period_start, date) or not amount.is_finite() or amount < 0:
                raise ValueError("Prediction history entry is invalid.")
            model_name = str(entry.get("model_name") or entry.get("method") or "Unknown")[:80]
            metrics = _json_safe({
                "forecast_horizon": prediction_result.get("forecast_horizon", 1),
                "status": prediction_result.get("status"),
                "value_type": "predicted",
                "evaluation_metrics": entry.get("evaluation_metrics", prediction_result.get("metrics")),
            })
            record = db.session.scalar(select(Prediction).where(
                Prediction.user_id == user_id,
                Prediction.period_start == period_start,
            ))
            if record is None:
                record = Prediction(user_id=user_id, period_start=period_start)
                db.session.add(record)
            record.predicted_expense = amount
            record.model_name = model_name
            record.metrics = metrics
            saved.append(record)
        db.session.commit()
        return saved
    except Exception:
        db.session.rollback()
        raise


def user_prediction_history(user_id: int, *, limit: int = 24) -> list[Prediction]:
    """Return only the requested owner's most recent prediction records."""
    user_id = _validate_user_id(user_id)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("History limit must be between 1 and 100.")
    return list(db.session.scalars(
        select(Prediction)
        .where(Prediction.user_id == user_id)
        .order_by(Prediction.period_start.desc(), Prediction.id.desc())
        .limit(limit)
    ))
