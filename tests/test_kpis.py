import pandas as pd
import pytest

from core import kpis


def _fixture_df() -> pd.DataFrame:
    """Two campaigns x 14 days, hand-built so ratios are easy to check by hand."""
    rows = []
    dates = pd.date_range("2026-01-01", periods=14, freq="D")
    for i, d in enumerate(dates):
        # Campaign A: steady performer.
        rows.append(dict(
            date=d, campaign_id=1, campaign_name="A", channel="Search", objective="Conversion",
            planned_daily_budget=100, impressions=10_000, clicks=500, spend=100,
            conversions=25, revenue=500,
        ))
        # Campaign B: second week revenue collapses (WoW drop should be detected).
        revenue_b = 400 if i < 7 else 40
        rows.append(dict(
            date=d, campaign_id=2, campaign_name="B", channel="Display", objective="Awareness",
            planned_daily_budget=50, impressions=20_000, clicks=200, spend=50,
            conversions=10, revenue=revenue_b,
        ))
    return pd.DataFrame(rows)


def test_portfolio_daily_ratios():
    df = _fixture_df()
    daily = kpis.portfolio_daily(df)
    row = daily.iloc[0]
    assert row["ctr"] == pytest.approx((500 + 200) / (10_000 + 20_000))
    assert row["roas"] == pytest.approx((500 + 400) / (100 + 50))


def test_campaign_summary_wow_delta_detects_collapse():
    df = _fixture_df()
    summary = kpis.campaign_summary(df, days=7, end_date=df["date"].max())
    row_b = summary[summary["campaign_name"] == "B"].iloc[0]
    # last 7 days revenue = 40*7=280, prior 7 days = 400*7=2800 -> -90%
    assert row_b["revenue_wow_pct"] == pytest.approx(-90.0, abs=0.5)

    row_a = summary[summary["campaign_name"] == "A"].iloc[0]
    assert row_a["revenue_wow_pct"] == pytest.approx(0.0, abs=0.5)


def test_forecast_metric_flat_series_has_zero_slope():
    df = _fixture_df()
    fc = kpis.forecast_metric(df, metric="revenue", campaign_id=1, history_days=14, horizon_days=7)
    assert fc.direction == "flat"
    assert fc.last_actual == pytest.approx(500.0)


def test_forecast_metric_detects_downward_trend():
    df = _fixture_df()
    fc = kpis.forecast_metric(df, metric="revenue", campaign_id=2, history_days=14, horizon_days=7)
    assert fc.direction == "down"
    assert fc.trend_pct_per_day < 0
