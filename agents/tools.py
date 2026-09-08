"""Tool functions the agents can call. Every function returns plain
JSON-serializable data (dict / list / str) — no dataframes, no numpy types —
because AutoGen serializes tool results into the conversation as text.

These are the *only* way agent output is grounded in real numbers: the model
client never invents metrics, it only narrates what these functions return.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import pandas as pd

from core import kpis
from core.business_rules import evaluate_rules, insights_to_dicts, load_rules

_RULES_CONFIG = load_rules()


def _clean(obj):
    """Recursively replace NaN/inf with None so json.dumps never chokes,
    and convert numpy/py scalars to plain floats/ints."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else round(obj, 4)
    if hasattr(obj, "item"):  # numpy scalar
        return _clean(obj.item())
    return obj


def get_portfolio_summary(days: int = 7) -> dict:
    """Portfolio-wide spend/revenue/ROAS/CTR/CVR/CPA for the trailing `days`,
    with week-over-week percent change for each metric."""
    df = kpis.load_data()
    return _clean(kpis.portfolio_summary(df, days=days))


def get_campaign_breakdown(days: int = 7, top_n: int = 12, sort_by: str = "spend") -> list[dict]:
    """Per-campaign KPI table (spend, revenue, ROAS, CPA, CTR, CVR, budget
    pacing, week-over-week deltas) for the trailing `days`, sorted by `sort_by`
    descending. Valid sort_by values: spend, revenue, roas, cpa, roas_wow_pct,
    revenue_wow_pct."""
    df = kpis.load_data()
    cs = kpis.campaign_summary(df, days=days)
    cols = [
        "campaign_id", "campaign_name", "channel", "objective",
        "spend", "revenue", "conversions", "roas", "cpa", "ctr", "cvr",
        "budget_pacing_ratio", "spend_wow_pct", "revenue_wow_pct", "roas_wow_pct",
    ]
    if sort_by in cs.columns:
        cs = cs.sort_values(sort_by, ascending=False)
    records = cs[cols].head(top_n).to_dict(orient="records")
    return _clean(records)


def get_underperformers(days: int = 7, roas_threshold: float = 2.0) -> list[dict]:
    """Campaigns whose trailing-`days` ROAS is below `roas_threshold`,
    sorted by spend descending (biggest wasted spend first)."""
    df = kpis.load_data()
    cs = kpis.campaign_summary(df, days=days)
    under = cs[cs["roas"] < roas_threshold].sort_values("spend", ascending=False)
    cols = ["campaign_id", "campaign_name", "channel", "spend", "revenue", "roas", "roas_wow_pct"]
    return _clean(under[cols].to_dict(orient="records"))


def get_business_rule_flags(days: int = 7) -> list[dict]:
    """Evaluate config/business_rules.yaml against the trailing-`days`
    campaign summary. Returns actionable insights: severity, campaign,
    title, recommended_action, business_impact. This is the primary source
    of *actionable* recommendations — always prefer these over ad-hoc advice."""
    df = kpis.load_data()
    cs = kpis.campaign_summary(df, days=days)
    insights = evaluate_rules(cs, _RULES_CONFIG)
    return _clean(insights_to_dicts(insights))


def get_anomalies(lookback_days: int = 21, z_thresh: float = 2.5, top_n: int = 10) -> list[dict]:
    """Statistically detected anomalies (rolling z-score spikes/drops) in
    spend, conversions, ROAS, or CTR per campaign, most extreme first."""
    df = kpis.load_data()
    anomalies = kpis.detect_anomalies(df, z_thresh=z_thresh, lookback=lookback_days)
    return _clean([dataclasses.asdict(a) for a in anomalies[:top_n]])


def forecast_kpi(metric: str = "revenue", campaign_name: Optional[str] = None, horizon_days: int = 7) -> dict:
    """Simple linear-trend forecast for a metric (revenue, spend, conversions,
    clicks, or impressions) over the next `horizon_days`. Pass campaign_name
    to scope to one campaign, or omit for the whole portfolio. Use this
    whenever the user asks for a prediction, forecast, or 'what will happen
    next week' style question."""
    df = kpis.load_data()
    campaign_id = None
    resolved_name = None
    if campaign_name:
        matches = df[df["campaign_name"].str.contains(campaign_name, case=False, na=False)]
        if matches.empty:
            return {"error": f"No campaign matching '{campaign_name}'."}
        campaign_id = int(matches["campaign_id"].iloc[0])
        resolved_name = str(matches["campaign_name"].iloc[0])
    fc = kpis.forecast_metric(df, metric=metric, campaign_id=campaign_id, horizon_days=horizon_days)
    result = dataclasses.asdict(fc)
    result["campaign_name"] = resolved_name
    return _clean(result)


def list_campaigns() -> list[dict]:
    """List all campaigns with their channel and objective, for name lookups."""
    df = kpis.load_data()
    out = df[["campaign_id", "campaign_name", "channel", "objective"]].drop_duplicates()
    return _clean(out.to_dict(orient="records"))


ALL_TOOL_FUNCS = [
    get_portfolio_summary,
    get_campaign_breakdown,
    get_underperformers,
    get_business_rule_flags,
    get_anomalies,
    forecast_kpi,
    list_campaigns,
]
