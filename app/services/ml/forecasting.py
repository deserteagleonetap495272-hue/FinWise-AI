"""One- and three-month expense forecasts built on Stage 3 predictions.

This module does not fit additional models. Stage 3 provides the one-month
estimate and its chronological evaluation; longer horizons use an explicitly
flat level assumption instead of recursively treating predictions as actuals.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd

from app.services.ml.expense_prediction import clip_nonnegative_money, predict_next_month

MIN_HISTORY_MONTHS_FOR_THREE_MONTH = 24
MAX_FORECAST_HORIZON = 3


def _history_rows(monthly_history: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert Stage 3 monthly totals into labeled actual history rows."""
    if monthly_history.empty:
        return []
    ordered = monthly_history.sort_values("period", kind="stable")
    return [
        {
            "period": str(row.period),
            "amount": Decimal(str(row.amount)),
            "value_type": "actual",
            "method": "recorded expense total",
        }
        for row in ordered.itertuples(index=False)
    ]


def forecast_expenses(data: pd.DataFrame, *, horizon: int = 1,
                     prediction_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return historical actuals and a next-month or conservative 3-month view.

    The supplied transaction frame must already be scoped to the authenticated
    user by the caller (normally with Stage 2's ``fetch_user_transactions``).
    A one-month request uses Stage 3 exactly. A three-month request requires
    at least 24 calendar months and repeats the Stage 3 first-month level for
    the following two periods; future estimates are never fed back as actuals.
    """
    if isinstance(horizon, bool) or horizon not in (1, 3):
        raise ValueError("Forecast horizon must be 1 or 3 months.")

    # Callers in the model-management layer may supply a result generated from
    # a currently validated user-specific artifact. Normal Stage 5 callers keep
    # the original behavior and ask Stage 3 directly.
    prediction = prediction_result if prediction_result is not None else predict_next_month(data)
    monthly_history = prediction["monthly_history"]
    history = _history_rows(monthly_history)
    month_count = len(history)
    limitations = [
        "Forecasts are estimates based on this user's historical expenses, not guaranteed future spending.",
        "Stage 3 evaluation metrics describe one-month-ahead validation only and do not measure multi-month accuracy.",
    ]
    if prediction["predicted_expenses"] is None or prediction["prediction_period"] is None:
        return _result(
            status="insufficient_data", sufficient=False, horizon=horizon,
            history=history, forecasts=[], prediction=prediction,
            explanation="No valid expense history is available to generate a forecast.",
            limitations=limitations + ["Add valid expense transactions before requesting a forecast."],
        )

    if horizon == 3 and month_count < MIN_HISTORY_MONTHS_FOR_THREE_MONTH:
        return _result(
            status="insufficient_data", sufficient=False, horizon=horizon,
            history=history, forecasts=[], prediction=prediction,
            explanation=(
                f"A three-month forecast requires at least {MIN_HISTORY_MONTHS_FOR_THREE_MONTH} "
                f"calendar months of actual expenses; {month_count} are available. No future values were generated."
            ),
            limitations=limitations + [
                f"The three-month horizon is withheld below {MIN_HISTORY_MONTHS_FOR_THREE_MONTH} historical months."
            ],
        )

    if horizon == 3 and not prediction["data_sufficient"]:
        return _result(
            status="insufficient_data", sufficient=False, horizon=horizon,
            history=history, forecasts=[], prediction=prediction,
            explanation="Stage 3 did not find enough history for a supported one-month estimate, so the longer forecast is withheld.",
            limitations=limitations + ["A reliable three-month estimate cannot be formed from the current Stage 3 result."],
        )

    first_amount = clip_nonnegative_money(prediction["predicted_expenses"])
    first_period = pd.Period(prediction["prediction_period"], freq="M")
    forecasts = [{
        "period": str(first_period),
        "period_start": first_period.start_time.date(),
        "amount": first_amount,
        "value_type": "predicted",
        "method": f"Stage 3 · {prediction['selected_model']}",
        "model_name": prediction["selected_model"],
        "evaluation_metrics": prediction["metrics"],
        "depends_on_prior_forecast": False,
    }]

    if horizon == 3:
        for month_offset in (2, 3):
            period = first_period + (month_offset - 1)
            forecasts.append({
                "period": str(period),
                "period_start": period.start_time.date(),
                "amount": first_amount,
                "value_type": "predicted",
                "method": "Flat hold of Stage 3 first-month estimate",
                "model_name": "Flat hold assumption",
                "evaluation_metrics": None,
                "depends_on_prior_forecast": True,
            })
        limitations.append(
            "Months two and three hold the first forecast amount flat; this is a conservative scenario assumption, not a separate model prediction."
        )
        limitations.append(
            "Later months depend on the first forecast level but are not recursively used as new observations, limiting error compounding."
        )
        explanation = (
            "The first month uses the Stage 3 next-month estimate. Months two and three repeat that amount as a flat level scenario; "
            "forecast values are not fed back into the prediction model."
        )
    else:
        explanation = (
            "The next-month forecast reuses Stage 3's selected model or previous-month baseline and its chronological validation."
        )
        if not prediction["data_sufficient"]:
            limitations.append(
                "Stage 3 returned a transparent previous-month estimate because model comparison had insufficient history."
            )

    combined = history + [
        {key: row[key] for key in ("period", "amount", "value_type", "method")}
        for row in forecasts
    ]
    combined.sort(key=lambda row: row["period"])
    return _result(
        status="forecasted" if prediction["data_sufficient"] else "insufficient_data",
        sufficient=bool(prediction["data_sufficient"] and (horizon == 1 or month_count >= MIN_HISTORY_MONTHS_FOR_THREE_MONTH)),
        horizon=horizon,
        history=history,
        forecasts=forecasts,
        prediction=prediction,
        explanation=explanation,
        limitations=limitations,
        combined=combined,
    )


def forecast_next_month(data: pd.DataFrame) -> dict[str, Any]:
    """Convenience wrapper for a one-month-ahead expense forecast."""
    return forecast_expenses(data, horizon=1)


def forecast_three_months(data: pd.DataFrame) -> dict[str, Any]:
    """Convenience wrapper for the guarded three-month expense scenario."""
    return forecast_expenses(data, horizon=3)


def _result(*, status: str, sufficient: bool, horizon: int,
            history: list[dict[str, Any]], forecasts: list[dict[str, Any]],
            prediction: dict[str, Any], explanation: str,
            limitations: list[str], combined: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Assemble the stable public forecast result structure."""
    first = forecasts[0] if forecasts else None
    return {
        "status": status,
        "data_sufficient": sufficient,
        "historical_monthly_expenses": history,
        "forecasts": forecasts,
        "forecast_month": first["period_start"] if first else None,
        "predicted_amount": first["amount"] if first else None,
        "model_name": prediction["selected_model"],
        "evaluation_metrics": prediction["metrics"],
        "forecast_horizon": horizon,
        "explanation": explanation,
        "limitations": limitations,
        "values": combined if combined is not None else history,
        "historical_month_count": len(history),
        "minimum_history_months_for_three_month": MIN_HISTORY_MONTHS_FOR_THREE_MONTH,
    }
