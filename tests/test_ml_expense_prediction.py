"""Monthly regression prediction and chronological validation tests."""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from app.services.ml import expense_prediction as prediction
from app.services.ml.data_preprocessing import aggregate_expenses_by_month


def monthly_transactions(values, *, first_month="2024-01", omit=(), kind="expense"):
    """Build explicit test-only transaction rows from month totals."""
    start = pd.Period(first_month, freq="M")
    rows = []
    for offset, amount in enumerate(values):
        if offset in omit:
            continue
        period = start + offset
        rows.append({
            "id": len(rows) + 1, "user_id": 1, "category_id": 1,
            "amount": Decimal(str(amount)), "transaction_type": kind,
            "description": f"fixture row {offset}", "transaction_date": period.start_time.date(),
            "category": "Fixture category", "category_type": kind,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def long_history():
    values = [Decimal(100 + 7 * month + (month % 4) * 11) for month in range(18)]
    return monthly_transactions(values)


def test_monthly_aggregation_is_reused_and_handles_missing_months():
    rows = monthly_transactions(["10.10", "30.00", "99.99", "20.20"], omit={1})
    monthly = aggregate_expenses_by_month(rows)
    assert monthly["period"].astype(str).tolist() == ["2024-01", "2024-02", "2024-03", "2024-04"]
    assert monthly["amount"].tolist() == [
        Decimal("10.10"), Decimal("0.00"), Decimal("99.99"), Decimal("20.20")
    ]


def test_lag_and_rolling_features_use_only_previous_months():
    monthly = pd.DataFrame({
        "period": pd.period_range("2024-01", periods=5, freq="M"),
        "amount": [Decimal(value) for value in (10, 20, 30, 40, 50)],
    })
    features = prediction.build_monthly_features(monthly)
    april = features.loc[features["period"].astype(str) == "2024-04"].iloc[0]
    assert (april["lag_1"], april["lag_2"], april["lag_3"]) == (30.0, 20.0, 10.0)
    assert april["rolling_mean_3"] == 20.0
    assert april["actual"] == Decimal(40)


def test_changing_future_targets_cannot_change_earlier_features():
    original = pd.DataFrame({
        "period": pd.period_range("2024-01", periods=8, freq="M"),
        "amount": [Decimal(value) for value in (10, 20, 30, 40, 50, 60, 70, 80)],
    })
    changed = original.copy()
    changed.loc[7, "amount"] = Decimal("999999.99")
    first = prediction.build_monthly_features(original)
    second = prediction.build_monthly_features(changed)
    columns = list(prediction.FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(
        first.loc[first["period"] < pd.Period("2024-08"), columns],
        second.loc[second["period"] < pd.Period("2024-08"), columns],
    )


def test_walk_forward_evaluation_is_chronological_and_compares_all_models(long_history):
    monthly = aggregate_expenses_by_month(long_history)
    supervised = prediction.build_monthly_features(monthly)
    result = prediction.evaluate_monthly_models(supervised)
    assert result["status"] == "evaluated"
    assert result["validation_method"] == "expanding_window"
    assert result["training_observations"] >= prediction.MIN_TRAINING_OBSERVATIONS
    assert result["test_observations"] >= prediction.MIN_TEST_OBSERVATIONS
    assert result["test_periods"] == sorted(result["test_periods"])
    assert [row["model"] for row in result["model_comparison"]] == list(prediction.MODEL_NAMES)
    assert all(row["mae"] is not None and row["rmse"] is not None for row in result["model_comparison"])


def test_baseline_prediction_and_model_selection_prefer_baseline_for_constant_spending():
    result = prediction.predict_next_month(monthly_transactions(["50.00"] * 18))
    assert result["predicted_expenses"] == Decimal("50.00")
    assert result["selected_model"] == "Naive baseline (previous month)"
    assert result["model_based"] is False
    assert result["baseline_metrics"]["mae"] == 0
    assert result["prediction_period"] == date(2025, 7, 1)


def test_candidate_is_selected_when_walk_forward_mae_meaningfully_beats_baseline(monkeypatch):
    class KnownTrendRegressor:
        def fit(self, features, target):
            return self

        def predict(self, features):
            # The fixture has a deterministic +10 monthly trend. This test
            # double proves the selection path; it is not a production model.
            return features[:, 0] + 10

    monkeypatch.setattr(prediction, "_make_estimator", lambda _name: KnownTrendRegressor())
    result = prediction.predict_next_month(monthly_transactions([100 + 10 * month for month in range(18)]))
    assert result["data_sufficient"] is True
    assert result["model_based"] is True
    assert result["selected_model"] in prediction.MODEL_NAMES
    assert result["metrics"]["mae"] == 0
    assert result["baseline_metrics"]["mae"] == pytest.approx(10)
    assert result["predicted_expenses"] == Decimal("280.00")
    assert "expanding-window validation" in result["explanation"]


@pytest.mark.parametrize("model_name", prediction.MODEL_NAMES)
def test_each_supported_regressor_can_be_created_and_fit(model_name, long_history):
    monthly = aggregate_expenses_by_month(long_history)
    supervised = prediction.build_monthly_features(monthly)
    model = prediction._make_estimator(model_name)
    model.fit(supervised.loc[:, prediction.FEATURE_COLUMNS].to_numpy(dtype=float),
              [float(value) for value in supervised["actual"]])
    output = model.predict(supervised.loc[:, prediction.FEATURE_COLUMNS].iloc[[-1]].to_numpy(dtype=float))
    assert len(output) == 1
    assert pd.notna(output[0])


def test_prediction_is_nonnegative_and_rounded_to_cents():
    assert prediction.clip_nonnegative_money(-12.349) == Decimal("0.00")
    assert prediction.clip_nonnegative_money("12.345") == Decimal("12.35")
    assert prediction.clip_nonnegative_money(float("nan")) == Decimal("0.00")


@pytest.mark.parametrize("months", [1, 5, 11])
def test_insufficient_history_returns_explicit_baseline_without_model_metrics(months):
    rows = monthly_transactions([Decimal(10 + month) for month in range(months)])
    result = prediction.predict_next_month(rows)
    assert result["status"] == "insufficient_data"
    assert result["data_sufficient"] is False
    assert result["predicted_expenses"] == Decimal(9 + months).quantize(Decimal("0.01"))
    assert result["model_based"] is False
    assert result["metrics"] is None
    assert "at least 12 months" in result["explanation"].lower()
    assert result["model_comparison"] == []


def test_empty_and_income_only_inputs_do_not_fabricate_predictions():
    empty = prediction.predict_next_month(pd.DataFrame())
    assert empty["predicted_expenses"] is None
    assert empty["status"] == "insufficient_data"
    income = monthly_transactions(["1000.00"] * 12, kind="income")
    result = prediction.predict_next_month(income)
    assert result["predicted_expenses"] is None
    assert result["historical_months"] == 0


def test_metrics_are_correct_and_r2_omitted_when_not_meaningful():
    metrics = prediction.calculate_metrics(
        [10.0, 20.0, 40.0], [12.0, 18.0, 35.0], training_observations=8
    )
    assert metrics["mae"] == pytest.approx(3.0)
    assert metrics["rmse"] == pytest.approx((33 / 3) ** 0.5)
    assert metrics["r2"] is not None
    assert metrics["training_observations"] == 8
    assert metrics["test_observations"] == 3
    assert prediction.calculate_metrics([10.0, 10.0], [9.0, 11.0], training_observations=4)["r2"] is None
