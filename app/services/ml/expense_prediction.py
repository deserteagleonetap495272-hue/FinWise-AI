"""Monthly expense prediction with strictly chronological validation.

The service consumes transaction rows already retrieved for one authorized
user. It delegates data cleaning and aggregation to Stage 2 preprocessing,
keeps monetary aggregation exact with ``Decimal``, and converts values to
floating point only at the scikit-learn boundary.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from app.services.ml.data_preprocessing import (
    MONEY_ZERO,
    aggregate_expenses_by_month,
    separate_income_expenses,
)

CENT = Decimal("0.01")
FEATURE_COLUMNS = (
    "lag_1", "lag_2", "lag_3", "rolling_mean_3", "month_sin", "month_cos",
)
MIN_HISTORY_MONTHS = 12
MIN_TRAINING_OBSERVATIONS = 3
MIN_TEST_OBSERVATIONS = 3
MEANINGFUL_IMPROVEMENT = 0.05
MODEL_NAMES = ("Linear Regression", "Random Forest", "Gradient Boosting")


def build_monthly_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Build target-month features from strictly earlier monthly totals.

    Input is the ``period``/``amount`` DataFrame from Stage 2's monthly
    aggregator. Missing calendar months are represented as zero by that
    aggregator. Every lag and rolling average is shifted before calculation,
    so no row's features contain its target month's expense value.
    """
    if not {"period", "amount"}.issubset(monthly.columns):
        raise ValueError("Monthly totals must contain period and amount columns.")
    if monthly.empty:
        return pd.DataFrame(columns=["period", "actual", *FEATURE_COLUMNS])

    periods = pd.PeriodIndex(monthly["period"], freq="M")
    amounts = [Decimal(str(value)) for value in monthly["amount"]]
    series = pd.Series(amounts, index=periods, dtype=object).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    complete_periods = pd.period_range(series.index.min(), series.index.max(), freq="M")
    series = series.reindex(complete_periods, fill_value=MONEY_ZERO)

    rows: list[dict[str, Any]] = []
    for position, period in enumerate(complete_periods):
        if position < 3:
            continue
        prior = list(series.iloc[:position])
        lag_1, lag_2, lag_3 = prior[-1], prior[-2], prior[-3]
        prior_three_mean = sum(prior[-3:], MONEY_ZERO) / Decimal(3)
        month_angle = 2 * np.pi * period.month / 12
        rows.append({
            "period": period,
            "actual": series.iloc[position],
            "lag_1": float(lag_1),
            "lag_2": float(lag_2),
            "lag_3": float(lag_3),
            "rolling_mean_3": float(prior_three_mean),
            "month_sin": float(np.sin(month_angle)),
            "month_cos": float(np.cos(month_angle)),
        })
    return pd.DataFrame(rows, columns=["period", "actual", *FEATURE_COLUMNS])


def _make_estimator(name: str) -> Pipeline:
    """Create a deterministic, modest-capacity regression pipeline."""
    if name == "Linear Regression":
        regressor = LinearRegression()
    elif name == "Random Forest":
        regressor = RandomForestRegressor(
            n_estimators=120, max_depth=3, min_samples_leaf=2,
            random_state=42, n_jobs=1,
        )
    elif name == "Gradient Boosting":
        regressor = GradientBoostingRegressor(
            n_estimators=80, learning_rate=0.05, max_depth=2,
            min_samples_leaf=2, loss="huber", random_state=42,
        )
    else:
        raise ValueError(f"Unsupported expense model: {name}")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("regressor", regressor),
    ])


def calculate_metrics(actual: list[float], predicted: list[float], *,
                      training_observations: int) -> dict[str, int | float | None]:
    """Calculate MAE/RMSE and meaningful R² for a chronological test window."""
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted values must have equal lengths.")
    if not actual:
        return {
            "mae": None, "rmse": None, "r2": None,
            "training_observations": training_observations,
            "test_observations": 0,
        }
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = None
    if len(actual) >= MIN_TEST_OBSERVATIONS and len(set(actual)) > 1:
        r2 = float(r2_score(actual, predicted))
    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "training_observations": training_observations,
        "test_observations": len(actual),
    }


def _validation_start(observations: int) -> int:
    """Choose a chronological holdout leaving at least three train/test rows."""
    test_count = max(MIN_TEST_OBSERVATIONS, int(np.ceil(observations * 0.25)))
    return max(MIN_TRAINING_OBSERVATIONS, observations - test_count)


def evaluate_monthly_models(supervised: pd.DataFrame) -> dict[str, Any]:
    """Compare candidate models with naive lag-1 predictions walk-forward.

    For each test month, fit each model on rows strictly before that month.
    This expanding-window evaluation never shuffles or trains on future rows.
    If there are too few examples for a fair holdout, structured insufficiency
    is returned rather than fabricated metrics.
    """
    if supervised.empty:
        return {
            "status": "insufficient_data", "reason": "No supervised monthly rows are available.",
            "selected_model": None, "model_comparison": [],
            "baseline_metrics": calculate_metrics([], [], training_observations=0),
            "training_observations": 0, "test_observations": 0,
            "test_periods": [],
            "validation_method": "expanding_window",
        }

    first_test = _validation_start(len(supervised))
    if first_test >= len(supervised) or first_test < MIN_TRAINING_OBSERVATIONS:
        return {
            "status": "insufficient_data",
            "reason": "At least three training and three chronological test months are required for model comparison.",
            "selected_model": None, "model_comparison": [],
            "baseline_metrics": calculate_metrics([], [], training_observations=first_test),
            "training_observations": first_test, "test_observations": 0,
            "test_periods": [],
            "validation_method": "expanding_window",
        }

    actual = [float(value) for value in supervised.iloc[first_test:]["actual"]]
    baseline_predictions = [float(value) for value in supervised.iloc[first_test:]["lag_1"]]
    baseline_metrics = calculate_metrics(
        actual, baseline_predictions, training_observations=first_test
    )

    comparison = []
    for name in MODEL_NAMES:
        predictions = []
        for test_position in range(first_test, len(supervised)):
            training = supervised.iloc[:test_position]
            estimator = _make_estimator(name)
            estimator.fit(training.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64),
                          np.asarray([float(value) for value in training["actual"]], dtype=np.float64))
            row = supervised.iloc[[test_position]].loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            predictions.append(float(estimator.predict(row)[0]))
        metrics = calculate_metrics(
            actual, predictions, training_observations=first_test
        )
        comparison.append({"model": name, **metrics})

    baseline_mae = baseline_metrics["mae"]
    eligible = [
        row for row in comparison
        if baseline_mae is not None and baseline_mae > 0
        and row["mae"] is not None
        and row["mae"] <= baseline_mae * (1 - MEANINGFUL_IMPROVEMENT)
    ]
    selected = min(eligible, key=lambda row: (row["mae"], row["rmse"], row["model"])) if eligible else None
    return {
        "status": "evaluated",
        "reason": None if selected else "No tested model beat the previous-month baseline by at least 5% MAE; the baseline is preferred.",
        "selected_model": selected["model"] if selected else "Naive baseline (previous month)",
        "model_comparison": comparison,
        "baseline_metrics": baseline_metrics,
        "selected_metrics": selected,
        "training_observations": first_test,
        "test_observations": len(actual),
        "test_periods": [str(period) for period in supervised.iloc[first_test:]["period"]],
        "validation_method": "expanding_window",
    }


def clip_nonnegative_money(value: float | Decimal) -> Decimal:
    """Convert a model output to non-negative currency precision."""
    amount = Decimal(str(value))
    if not amount.is_finite():
        amount = MONEY_ZERO
    return max(amount, MONEY_ZERO).quantize(CENT, rounding=ROUND_HALF_UP)


def predict_next_month(data: pd.DataFrame, *, include_fitted_model: bool = False) -> dict[str, Any]:
    """Evaluate models and estimate next month's expenses for supplied rows.

    ``data`` must be the authorized user's transaction frame, normally from
    ``fetch_user_transactions(current_user.id)``. It is cleaned again at the
    service boundary, and income is excluded by Stage 2 aggregation.
    Insufficient history returns no model claim; one or more observed expense
    months may still receive an explicitly marked naive estimate.
    """
    _, expenses = separate_income_expenses(data)
    monthly = aggregate_expenses_by_month(expenses)
    historical_months = len(monthly)
    empty_metrics = calculate_metrics([], [], training_observations=0)

    if monthly.empty:
        return {
            "status": "insufficient_data", "data_sufficient": False,
            "explanation": "No valid expense history is available for a monthly prediction.",
            "predicted_expenses": None, "prediction_period": None,
            "selected_model": None, "model_based": False,
            "metrics": None, "baseline_metrics": empty_metrics,
            "model_comparison": [], "training_observations": 0,
            "test_observations": 0, "historical_months": 0,
            "validation_method": "expanding_window",
            "monthly_history": monthly,
        }

    last_period = pd.Period(monthly.iloc[-1]["period"], freq="M")
    next_period = last_period + 1
    baseline = clip_nonnegative_money(monthly.iloc[-1]["amount"])
    supervised = build_monthly_features(monthly)
    if historical_months < MIN_HISTORY_MONTHS:
        evaluation = {
            "status": "insufficient_data",
            "reason": f"At least {MIN_HISTORY_MONTHS} calendar months are required for model comparison.",
            "selected_model": None, "model_comparison": [],
            "baseline_metrics": calculate_metrics([], [], training_observations=0),
            "training_observations": 0, "test_observations": 0,
            "test_periods": [], "validation_method": "expanding_window",
        }
    else:
        evaluation = evaluate_monthly_models(supervised)

    if evaluation["status"] != "evaluated":
        explanation = (
            f"Only {historical_months} calendar months of expense history are available. "
            f"At least {MIN_HISTORY_MONTHS} months are required for model comparison; "
            "the estimate uses the previous month's expenses as a transparent baseline."
        )
        return {
            "status": "insufficient_data", "data_sufficient": False,
            "explanation": explanation, "predicted_expenses": baseline,
            "prediction_period": next_period.start_time.date(),
            "selected_model": "Naive baseline (previous month)", "model_based": False,
            "metrics": None, "baseline_metrics": evaluation["baseline_metrics"],
            "model_comparison": evaluation["model_comparison"],
            "training_observations": evaluation["training_observations"],
            "test_observations": evaluation["test_observations"],
            "test_periods": evaluation["test_periods"],
            "historical_months": historical_months,
            "validation_method": evaluation["validation_method"],
            "monthly_history": monthly,
        }

    selected = evaluation["selected_model"]
    if selected == "Naive baseline (previous month)":
        prediction = baseline
        metrics = evaluation["baseline_metrics"]
        model_based = False
    else:
        final_model = _make_estimator(selected)
        final_model.fit(
            supervised.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64),
            np.asarray([float(value) for value in supervised["actual"]], dtype=np.float64),
        )
        next_features = _next_month_features(monthly)
        output = final_model.predict(next_features.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64))[0]
        prediction = clip_nonnegative_money(output)
        metrics = evaluation["selected_metrics"]
        model_based = True

    explanation = (
        f"The {selected} was selected using expanding-window validation over past months. "
        "The next-month amount is an estimate based on historical expenses, not a guarantee."
        if model_based else
        "The previous-month baseline was selected because no tested model improved MAE by at least 5%. "
        "The next-month amount is an estimate, not a guarantee."
    )
    return {
        "status": "evaluated", "data_sufficient": True,
        "explanation": explanation, "predicted_expenses": prediction,
        "prediction_period": next_period.start_time.date(),
        "selected_model": selected, "model_based": model_based,
        "metrics": metrics, "baseline_metrics": evaluation["baseline_metrics"],
        "model_comparison": evaluation["model_comparison"],
        "training_observations": evaluation["training_observations"],
        "test_observations": evaluation["test_observations"],
        "test_periods": evaluation["test_periods"],
        "historical_months": historical_months,
        "validation_method": evaluation["validation_method"],
        "monthly_history": monthly,
        **({"fitted_model": final_model if model_based else None} if include_fitted_model else {}),
    }


def predict_next_month_with_saved_model(data: pd.DataFrame, *, model_name: str,
                                        model: Pipeline, metadata: dict[str, Any]) -> dict[str, Any]:
    """Predict using a previously validated Stage 3 model artifact.

    The model store must first verify user ownership, source fingerprint,
    metadata signature, artifact integrity, and model version. This function
    reuses Stage 2 aggregation/features and Stage 3 money formatting while
    skipping candidate evaluation and refitting on an unchanged-data request.
    """
    if model_name not in MODEL_NAMES:
        raise ValueError("Saved expense model name is not supported.")
    if not isinstance(model, Pipeline):
        raise ValueError("A supported fitted expense pipeline is required.")

    _, expenses = separate_income_expenses(data)
    monthly = aggregate_expenses_by_month(expenses)
    if len(monthly) < MIN_HISTORY_MONTHS:
        # A current saved model cannot normally reach this branch because the
        # source fingerprint identifies changes. Fail closed if called directly.
        raise ValueError("Saved expense model requires at least 12 months of history.")
    supervised = build_monthly_features(monthly)
    last_period = pd.Period(monthly.iloc[-1]["period"], freq="M")
    next_period = last_period + 1
    features = _next_month_features(monthly).loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    amount = clip_nonnegative_money(model.predict(features)[0])
    evaluation = metadata.get("evaluation_metrics", {})
    training_count = len(supervised)
    return {
        "status": "evaluated", "data_sufficient": True,
        "explanation": f"The saved {model_name} was trained on the unchanged transaction history and passed source validation.",
        "predicted_expenses": amount,
        "prediction_period": next_period.start_time.date(),
        "selected_model": model_name, "model_based": True,
        "metrics": evaluation.get("selected"),
        "baseline_metrics": evaluation.get("baseline"),
        "model_comparison": evaluation.get("comparison", []),
        "training_observations": training_count,
        "test_observations": (evaluation.get("selected") or {}).get("test_observations", 0),
        "test_periods": evaluation.get("test_periods", []),
        "historical_months": len(monthly),
        "validation_method": "expanding_window",
        "monthly_history": monthly,
    }


def predict_next_month_with_cached_baseline(data: pd.DataFrame,
                                            metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the previous-month baseline without fitting candidates again."""
    _, expenses = separate_income_expenses(data)
    monthly = aggregate_expenses_by_month(expenses)
    if monthly.empty:
        return predict_next_month(data)
    last_period = pd.Period(monthly.iloc[-1]["period"], freq="M")
    evaluation = metadata.get("evaluation_metrics", {})
    return {
        "status": "insufficient_data" if metadata.get("model_status") == "insufficient_data" else "evaluated",
        "data_sufficient": metadata.get("model_status") != "insufficient_data",
        "explanation": "The previous-month baseline was selected during training management; no model artifact is stored.",
        "predicted_expenses": clip_nonnegative_money(monthly.iloc[-1]["amount"]),
        "prediction_period": (last_period + 1).start_time.date(),
        "selected_model": "Naive baseline (previous month)", "model_based": False,
        "metrics": evaluation.get("baseline"),
        "baseline_metrics": evaluation.get("baseline"),
        "model_comparison": evaluation.get("comparison", []),
        "training_observations": evaluation.get("training_observations", 0),
        "test_observations": evaluation.get("test_observations", 0),
        "test_periods": evaluation.get("test_periods", []),
        "historical_months": len(monthly),
        "validation_method": "expanding_window",
        "monthly_history": monthly,
    }


def _next_month_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Construct one future row using only observed months before it."""
    supervised_source = monthly.copy()
    last_period = pd.Period(supervised_source.iloc[-1]["period"], freq="M")
    next_period = last_period + 1
    prior_values = [Decimal(value) for value in supervised_source["amount"]]
    if len(prior_values) < 3:
        raise ValueError("At least three monthly totals are required for model features.")
    angle = 2 * np.pi * next_period.month / 12
    return pd.DataFrame([{
        "lag_1": float(prior_values[-1]),
        "lag_2": float(prior_values[-2]),
        "lag_3": float(prior_values[-3]),
        "rolling_mean_3": float(sum(prior_values[-3:], MONEY_ZERO) / Decimal(3)),
        "month_sin": float(np.sin(angle)),
        "month_cos": float(np.cos(angle)),
    }], columns=FEATURE_COLUMNS)
