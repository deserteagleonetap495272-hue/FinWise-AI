"""Tests for forecasts built on the Stage 3 monthly prediction service."""

from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

from app.services.ml import forecasting
from app.services.ml.expense_prediction import predict_next_month as stage3_predict_next_month


def transaction_rows(months=24, *, omit=(), amount="50.00"):
    """Build synthetic test-only expenses; never used as production records."""
    rows = []
    for offset in range(months):
        if offset in omit:
            continue
        period = pd.Period("2024-01", freq="M") + offset
        rows.append({
            "id": len(rows) + 1,
            "user_id": 1,
            "category_id": 1,
            "amount": Decimal(str(amount)),
            "transaction_type": "expense",
            "description": f"fixture {offset}",
            "transaction_date": period.start_time.date(),
            "category": "Fixture expense",
            "category_type": "expense",
        })
    return pd.DataFrame(rows)


def stub_stage3(months, *, amount=Decimal("50.00"), sufficient=True, model="Naive baseline (previous month)"):
    """Shape a Stage 3 result around synthetic monthly history for focused tests."""
    monthly = pd.DataFrame({
        "period": pd.period_range("2024-01", periods=months, freq="M"),
        "amount": [Decimal("50.00")] * months,
        "transaction_count": [1] * months,
    })
    prediction_period = (monthly.iloc[-1]["period"] + 1).start_time.date() if months else None
    return {
        "monthly_history": monthly,
        "predicted_expenses": amount if months else None,
        "prediction_period": prediction_period,
        "selected_model": model if months else None,
        "metrics": {"mae": 2.0, "rmse": 3.0, "r2": 0.8} if sufficient and months else None,
        "data_sufficient": sufficient,
        "status": "evaluated" if sufficient else "insufficient_data",
    }


def test_next_month_forecast_reuses_stage3_and_returns_structured_output(monkeypatch):
    data = transaction_rows(12)
    calls = []
    stage3_result = stub_stage3(12, amount=Decimal("72.35"), model="Linear Regression")

    def fake_prediction(received):
        calls.append(received)
        return stage3_result

    monkeypatch.setattr(forecasting, "predict_next_month", fake_prediction)
    result = forecasting.forecast_next_month(data)
    assert calls == [data]
    assert result["status"] == "forecasted"
    assert result["data_sufficient"] is True
    assert result["predicted_amount"] == Decimal("72.35")
    assert result["forecast_month"] == date(2025, 1, 1)
    assert result["model_name"] == "Linear Regression"
    assert result["evaluation_metrics"] == stage3_result["metrics"]
    assert result["forecast_horizon"] == 1
    assert result["forecasts"][0]["value_type"] == "predicted"
    assert len(result["historical_monthly_expenses"]) == 12


def test_three_month_forecast_uses_flat_hold_without_recursive_calls(monkeypatch):
    data = transaction_rows(24)
    calls = []
    stage3_result = stub_stage3(24, amount=Decimal("88.40"))

    def fake_prediction(received):
        calls.append(received)
        return stage3_result

    monkeypatch.setattr(forecasting, "predict_next_month", fake_prediction)
    result = forecasting.forecast_three_months(data)
    assert calls == [data]
    assert result["status"] == "forecasted"
    assert result["data_sufficient"] is True
    assert [row["period"] for row in result["forecasts"]] == ["2026-01", "2026-02", "2026-03"]
    assert [row["amount"] for row in result["forecasts"]] == [Decimal("88.40")] * 3
    assert [row["value_type"] for row in result["forecasts"]] == ["predicted"] * 3
    assert [row["depends_on_prior_forecast"] for row in result["forecasts"]] == [False, True, True]
    assert result["forecasts"][1]["evaluation_metrics"] is None
    assert "flat level scenario" in result["explanation"]
    assert len(result["values"]) == 27


def test_three_month_horizon_is_withheld_below_minimum_history(monkeypatch):
    data = transaction_rows(23)
    monkeypatch.setattr(forecasting, "predict_next_month", lambda _data: stub_stage3(23))
    result = forecasting.forecast_expenses(data, horizon=3)
    assert result["status"] == "insufficient_data"
    assert result["data_sufficient"] is False
    assert result["historical_month_count"] == 23
    assert result["forecasts"] == []
    assert result["predicted_amount"] is None
    assert result["forecast_month"] is None
    assert "at least 24" in result["explanation"]


def test_empty_history_returns_no_forecast(monkeypatch):
    monkeypatch.setattr(forecasting, "predict_next_month", lambda _data: stub_stage3(0))
    result = forecasting.forecast_next_month(pd.DataFrame())
    assert result["status"] == "insufficient_data"
    assert result["historical_monthly_expenses"] == []
    assert result["forecasts"] == []
    assert result["predicted_amount"] is None
    assert result["forecast_month"] is None


def test_missing_months_are_preserved_as_zero_actuals_and_ordered(monkeypatch):
    data = transaction_rows(24, omit={5})
    original_prediction = stage3_predict_next_month
    monkeypatch.setattr(forecasting, "predict_next_month", original_prediction)
    result = forecasting.forecast_three_months(data)
    assert result["historical_month_count"] == 24
    assert result["historical_monthly_expenses"][5]["period"] == "2024-06"
    assert result["historical_monthly_expenses"][5]["amount"] == Decimal("0.00")
    assert result["historical_monthly_expenses"][5]["value_type"] == "actual"
    assert [row["period"] for row in result["values"]] == sorted(row["period"] for row in result["values"])


def test_actual_and_predicted_values_have_explicit_distinct_labels(monkeypatch):
    monkeypatch.setattr(forecasting, "predict_next_month", lambda _data: stub_stage3(24))
    result = forecasting.forecast_three_months(transaction_rows(24))
    assert {row["value_type"] for row in result["historical_monthly_expenses"]} == {"actual"}
    assert {row["value_type"] for row in result["forecasts"]} == {"predicted"}
    assert result["values"][-3:][0]["method"].startswith("Stage 3")
    assert result["values"][-3:][1]["method"] == "Flat hold of Stage 3 first-month estimate"


def test_stage3_is_called_once_with_original_actuals_and_future_values_do_not_leak(monkeypatch):
    data = transaction_rows(24)
    original_ids = data["id"].tolist()
    calls = []

    def fake_prediction(received):
        calls.append(received["id"].tolist())
        return stub_stage3(24, amount=Decimal("70.00"))

    monkeypatch.setattr(forecasting, "predict_next_month", fake_prediction)
    result = forecasting.forecast_three_months(data)
    assert calls == [original_ids]
    assert data["id"].tolist() == original_ids
    assert result["historical_month_count"] == 24
    assert result["forecasts"][1]["amount"] == result["forecasts"][0]["amount"]
    assert result["forecasts"][2]["amount"] == result["forecasts"][0]["amount"]


def test_negative_stage3_estimate_is_clamped(monkeypatch):
    monkeypatch.setattr(
        forecasting, "predict_next_month",
        lambda _data: stub_stage3(12, amount=Decimal("-42.00")),
    )
    result = forecasting.forecast_next_month(transaction_rows(12))
    assert result["predicted_amount"] == Decimal("0.00")
    assert result["forecasts"][0]["amount"] >= Decimal("0.00")


def test_insufficient_stage3_result_does_not_enable_three_month_horizon(monkeypatch):
    monkeypatch.setattr(
        forecasting, "predict_next_month",
        lambda _data: stub_stage3(24, sufficient=False),
    )
    result = forecasting.forecast_three_months(transaction_rows(24))
    assert result["status"] == "insufficient_data"
    assert result["forecasts"] == []
    assert result["data_sufficient"] is False


def test_horizon_validation():
    with pytest.raises(ValueError, match="1 or 3"):
        forecasting.forecast_expenses(pd.DataFrame(), horizon=2)
    with pytest.raises(ValueError, match="1 or 3"):
        forecasting.forecast_expenses(pd.DataFrame(), horizon=True)
