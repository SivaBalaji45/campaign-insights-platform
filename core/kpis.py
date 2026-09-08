"""KPI aggregation, week-over-week deltas, anomaly detection, and forecasting.

Everything here operates on the raw row-level dataframe produced by
data/generate_dataset.py. Ratios are always recomputed from summed numerators
and denominators (never averaged directly) so aggregation stays mathematically
correct at any grain (portfolio / channel / campaign).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "data" / "campaign_performance.csv"

METRIC_COLS = ["impressions", "clicks", "spend", "conversions", "revenue"]


def load_data(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No dataset found at {path}. Run `python main.py generate-data` first."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.where(b > 0, a / b, np.nan)


def _add_ratios(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.copy()
    agg["ctr"] = _safe_div(agg["clicks"], agg["impressions"])
    agg["cvr"] = _safe_div(agg["conversions"], agg["clicks"])
    agg["cpc"] = _safe_div(agg["spend"], agg["clicks"])
    agg["cpm"] = _safe_div(agg["spend"], agg["impressions"]) * 1000
    agg["cpa"] = _safe_div(agg["spend"], agg["conversions"])
    agg["roas"] = _safe_div(agg["revenue"], agg["spend"])
    agg["aov"] = _safe_div(agg["revenue"], agg["conversions"])
    return agg


def date_window(df: pd.DataFrame, days: int = 7, end_date: Optional[pd.Timestamp] = None):
    end_date = end_date or df["date"].max()
    start_date = end_date - pd.Timedelta(days=days - 1)
    prev_end = start_date - pd.Timedelta(days=1)
    prev_start = prev_end - pd.Timedelta(days=days - 1)
    return start_date, end_date, prev_start, prev_end


def portfolio_daily(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("date", as_index=False)[METRIC_COLS].sum()
    agg = _add_ratios(agg)
    return agg.sort_values("date")


def rolling_portfolio(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    daily = portfolio_daily(df)
    for col in ["spend", "revenue", "roas", "ctr", "cvr", "cpa"]:
        daily[f"{col}_roll{window}"] = daily[col].rolling(window, min_periods=1).mean()
    return daily


def portfolio_summary(
    df: pd.DataFrame,
    days: int = 7,
    end_date: Optional[pd.Timestamp] = None,
) -> dict:
    """Portfolio-wide (all-campaigns) KPI rollup with week-over-week deltas."""
    start, end, prev_start, prev_end = date_window(df, days=days, end_date=end_date)

    cur = df[(df["date"] >= start) & (df["date"] <= end)][METRIC_COLS].sum()
    prev = df[(df["date"] >= prev_start) & (df["date"] <= prev_end)][METRIC_COLS].sum()

    def ratios(s: pd.Series) -> dict:
        spend, revenue, impressions, clicks, conversions = (
            s["spend"], s["revenue"], s["impressions"], s["clicks"], s["conversions"]
        )
        return {
            "spend": spend,
            "revenue": revenue,
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "ctr": (clicks / impressions) if impressions else float("nan"),
            "cvr": (conversions / clicks) if clicks else float("nan"),
            "cpa": (spend / conversions) if conversions else float("nan"),
            "roas": (revenue / spend) if spend else float("nan"),
        }

    cur_r = ratios(cur)
    prev_r = ratios(prev)

    out = {f"{k}": round(float(v), 4) for k, v in cur_r.items()}
    for metric in ["spend", "revenue", "conversions", "ctr", "cvr", "cpa", "roas"]:
        p = prev_r[metric]
        out[f"{metric}_wow_pct"] = round((cur_r[metric] - p) / p * 100, 2) if p else None
    out["window_start"] = start.date().isoformat()
    out["window_end"] = end.date().isoformat()
    return out


def campaign_summary(
    df: pd.DataFrame,
    days: int = 7,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Per-campaign KPI rollup for the trailing `days` window plus the equal-length
    prior window, with week-over-week (or N-day-over-N-day) percent deltas."""
    start, end, prev_start, prev_end = date_window(df, days=days, end_date=end_date)

    cur = df[(df["date"] >= start) & (df["date"] <= end)]
    prev = df[(df["date"] >= prev_start) & (df["date"] <= prev_end)]

    dims = ["campaign_id", "campaign_name", "channel", "objective"]

    cur_agg = cur.groupby(dims, as_index=False)[METRIC_COLS].sum()
    cur_agg["planned_daily_budget"] = cur.groupby(dims)["planned_daily_budget"].mean().values
    cur_agg["budget_pacing_ratio"] = _safe_div(cur_agg["spend"], cur_agg["planned_daily_budget"] * days)
    cur_agg = _add_ratios(cur_agg)

    prev_agg = prev.groupby(dims, as_index=False)[METRIC_COLS].sum()
    prev_agg = _add_ratios(prev_agg)

    merged = cur_agg.merge(
        prev_agg[dims + METRIC_COLS + ["ctr", "cvr", "cpa", "roas", "cpc"]],
        on=dims,
        how="left",
        suffixes=("", "_prev"),
    )

    for metric in ["spend", "revenue", "conversions", "ctr", "cvr", "cpa", "roas"]:
        prev_col = f"{metric}_prev"
        delta_col = f"{metric}_wow_pct"
        merged[delta_col] = np.where(
            (merged[prev_col].notna()) & (merged[prev_col] != 0),
            (merged[metric] - merged[prev_col]) / merged[prev_col] * 100,
            np.nan,
        )

    merged["window_start"] = start
    merged["window_end"] = end
    return merged.sort_values("spend", ascending=False).reset_index(drop=True)


@dataclass
class Anomaly:
    date: str
    campaign_id: int
    campaign_name: str
    metric: str
    value: float
    expected: float
    z_score: float
    direction: str  # "spike" | "drop"


def detect_anomalies(
    df: pd.DataFrame,
    metrics: tuple[str, ...] = ("spend", "conversions", "roas", "ctr"),
    z_thresh: float = 2.5,
    lookback: int = 21,
) -> list[Anomaly]:
    """Rolling z-score anomaly detection per campaign per metric.

    For each day, compares the value against the mean/std of the preceding
    `lookback` days (excluding the day itself) so a real deviation is required,
    not just a naturally high day inside its own baseline.
    """
    anomalies: list[Anomaly] = []
    per_campaign = _add_ratios(
        df.groupby(["campaign_id", "campaign_name", "date"], as_index=False)[METRIC_COLS].sum()
    )

    for cid, g in per_campaign.groupby("campaign_id"):
        g = g.sort_values("date").reset_index(drop=True)
        name = g["campaign_name"].iloc[0]
        for metric in metrics:
            series = g[metric]
            rolling_mean = series.shift(1).rolling(lookback, min_periods=5).mean()
            rolling_std = series.shift(1).rolling(lookback, min_periods=5).std(ddof=0)
            z = (series - rolling_mean) / rolling_std.replace(0, np.nan)
            flagged = z.abs() >= z_thresh
            for idx in g.index[flagged.fillna(False)]:
                anomalies.append(
                    Anomaly(
                        date=g.loc[idx, "date"].date().isoformat(),
                        campaign_id=int(cid),
                        campaign_name=name,
                        metric=metric,
                        value=round(float(series.loc[idx]), 4),
                        expected=round(float(rolling_mean.loc[idx]), 4) if pd.notna(rolling_mean.loc[idx]) else float("nan"),
                        z_score=round(float(z.loc[idx]), 2),
                        direction="spike" if z.loc[idx] > 0 else "drop",
                    )
                )
    anomalies.sort(key=lambda a: abs(a.z_score), reverse=True)
    return anomalies


@dataclass
class Forecast:
    metric: str
    campaign_id: Optional[int]
    history_days: int
    horizon_days: int
    last_actual: float
    forecast_next: float
    forecast_total_horizon: float
    trend_pct_per_day: float
    direction: str  # "up" | "down" | "flat"
    confidence: str  # "low" | "medium" | "high" based on fit noise


def forecast_metric(
    df: pd.DataFrame,
    metric: str = "revenue",
    campaign_id: Optional[int] = None,
    history_days: int = 21,
    horizon_days: int = 7,
) -> Forecast:
    """Lightweight linear-trend forecast (ordinary least squares on the trailing
    `history_days`). Deliberately simple and auditable rather than a black box,
    since stakeholder reports need to be able to explain *why* a number moved.
    """
    scope = df if campaign_id is None else df[df["campaign_id"] == campaign_id]
    daily = scope.groupby("date", as_index=False)[METRIC_COLS].sum()
    daily = _add_ratios(daily).sort_values("date")
    daily = daily.tail(history_days).reset_index(drop=True)

    y = daily[metric].astype(float).values
    x = np.arange(len(y), dtype=float)

    if len(y) < 5 or np.all(y == y[0]):
        slope, intercept = 0.0, float(y[-1]) if len(y) else 0.0
    else:
        slope, intercept = np.polyfit(x, y, 1)

    fitted = slope * x + intercept
    residuals = y - fitted
    resid_std = float(np.std(residuals)) if len(residuals) else 0.0
    mean_y = float(np.mean(y)) if len(y) else 0.0
    noise_ratio = (resid_std / mean_y) if mean_y else 1.0

    if noise_ratio < 0.12:
        confidence = "high"
    elif noise_ratio < 0.30:
        confidence = "medium"
    else:
        confidence = "low"

    next_x = len(y)
    forecast_next = slope * next_x + intercept
    horizon_x = np.arange(next_x, next_x + horizon_days)
    horizon_vals = slope * horizon_x + intercept
    forecast_total = float(np.sum(np.clip(horizon_vals, a_min=0, a_max=None)))

    trend_pct_per_day = (slope / mean_y * 100) if mean_y else 0.0
    if abs(trend_pct_per_day) < 0.3:
        direction = "flat"
    else:
        direction = "up" if trend_pct_per_day > 0 else "down"

    return Forecast(
        metric=metric,
        campaign_id=campaign_id,
        history_days=len(y),
        horizon_days=horizon_days,
        last_actual=round(float(y[-1]), 2) if len(y) else 0.0,
        forecast_next=round(max(forecast_next, 0.0), 2),
        forecast_total_horizon=round(forecast_total, 2),
        trend_pct_per_day=round(trend_pct_per_day, 2),
        direction=direction,
        confidence=confidence,
    )
