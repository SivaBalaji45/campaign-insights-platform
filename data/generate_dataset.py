"""Synthetic campaign performance dataset generator.

Produces a daily, campaign-level marketing performance dataset with realistic
seasonality, budget pacing, and a handful of deliberately injected anomalies
(creative fatigue, a tracking outage, a viral spike) so the insight agent and
business-rule engine have something real to catch.

Row-level derived features (ratios that are meaningful per-row) are baked in
here. Time-series derived features (rolling averages, week-over-week deltas,
z-score anomalies, forecasts) are computed on demand in core/kpis.py because
they depend on the grouping/date-range chosen at analysis time.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_PATH = Path(__file__).parent / "campaign_performance.csv"

RNG_SEED = 42

CHANNELS = ["Search", "Social", "Display", "Video", "Email", "Affiliate"]
OBJECTIVES = ["Awareness", "Consideration", "Conversion"]
REGIONS = ["NA", "EMEA", "APAC", "LATAM"]
DEVICES = ["Mobile", "Desktop", "Tablet"]

# (campaign_name, channel, objective, daily_budget, base_ctr, target_roas, base_cpm, aov_mean)
#
# base_cvr is *derived* from target_roas (not hand-picked) via
# ROAS = 1000 * ctr * cvr * aov / cpm  =>  cvr = target_roas * cpm / (1000 * ctr * aov)
# so the dataset tells a coherent, realistic performance story instead of
# ROAS numbers that are an accident of independently-chosen inputs.
CAMPAIGNS = [
    ("Brand Search - Core",        "Search",    "Conversion",  1200, 0.055, 4.5, 9.5,   85),
    ("Brand Search - Competitor",  "Search",    "Conversion",   600, 0.038, 2.8, 11.0,  85),
    ("Prospecting - Lookalike",    "Social",    "Consideration", 900, 0.012, 1.4, 7.5,   78),
    ("Retargeting - Cart Abandon", "Social",    "Conversion",   500, 0.021, 5.5, 8.2,   92),
    ("Always-On Display",         "Display",   "Awareness",    700, 0.004, 0.9, 3.1,   70),
    ("Holiday Display Push",      "Display",   "Awareness",    650, 0.005, 0.85, 3.4,   70),
    ("YouTube Skippable",         "Video",     "Awareness",    850, 0.009, 1.8, 6.0,   75),
    ("Connected TV",              "Video",     "Awareness",    950, 0.006, 0.7, 14.0,  75),
    ("Lifecycle Newsletter",      "Email",     "Conversion",   150, 0.032, 9.0, 4.0,   68),
    ("Winback Sequence",          "Email",     "Conversion",   120, 0.045, 7.0, 4.0,   65),
    ("Affiliate Network - Tier1", "Affiliate", "Conversion",   800, 0.030, 3.6, 5.0,   80),
    ("Influencer Partnerships",   "Social",    "Awareness",    750, 0.015, 1.6, 9.0,   82),
]


def _cvr_from_target_roas(target_roas: float, ctr: float, cpm: float, aov: float) -> float:
    cvr = target_roas * cpm / (1000 * ctr * aov)
    return float(np.clip(cvr, 0.003, 0.5))

DOW_MULTIPLIER = {0: 1.00, 1: 1.03, 2: 1.05, 3: 1.02, 4: 0.98, 5: 0.80, 6: 0.78}  # Mon..Sun


def _clip_positive(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, a_min=0.0, a_max=None)


def generate(days: int = 120, end_date: datetime | None = None, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end_date = end_date or datetime.now(timezone.utc).date()
    dates = [end_date - timedelta(days=i) for i in range(days)][::-1]

    rows = []
    for cid, (name, channel, objective, budget, base_ctr, target_roas, base_cpm, aov_mean) in enumerate(CAMPAIGNS, start=1):
        base_cvr = _cvr_from_target_roas(target_roas, base_ctr, base_cpm, aov_mean)
        # Slow trend drift per campaign so nothing is perfectly flat.
        trend = rng.normal(loc=0.0002, scale=0.0015)

        # Inject one anomaly type per a few campaigns for the agent to catch.
        fatigue_start = None
        outage_day = None
        spike_day = None
        if cid == 3:  # Prospecting - Lookalike: creative fatigue over last 2 weeks
            fatigue_start = len(dates) - 14
        if cid == 1:  # Brand Search - Core: single-day tracking outage
            outage_day = len(dates) - 9
        if cid == 11:  # Affiliate: viral one-day spike
            spike_day = len(dates) - 5

        for i, d in enumerate(dates):
            dow = d.weekday()
            seasonality = DOW_MULTIPLIER[dow]
            noise = rng.normal(loc=1.0, scale=0.07)
            drift = 1.0 + trend * i

            planned_daily_budget = budget
            spend_factor = seasonality * noise * drift
            spend = _clip_positive(np.array([planned_daily_budget * spend_factor]))[0]

            cpm = max(0.3, base_cpm * rng.normal(1.0, 0.05))
            impressions = max(1.0, spend / cpm * 1000.0)

            ctr = max(0.0005, base_ctr * rng.normal(1.0, 0.08))
            if fatigue_start is not None and i >= fatigue_start:
                decay = 1.0 - 0.035 * (i - fatigue_start + 1)
                ctr *= max(0.35, decay)

            clicks = impressions * ctr

            cvr = max(0.0005, base_cvr * rng.normal(1.0, 0.10))
            conversions = clicks * cvr

            if outage_day is not None and i == outage_day:
                conversions = 0.0  # tracking outage: clicks still happen, conversions vanish

            if spike_day is not None and i == spike_day:
                clicks *= 3.4
                conversions *= 4.1
                impressions *= 2.2

            aov = max(5.0, rng.normal(aov_mean, aov_mean * 0.15))
            revenue = conversions * aov

            frequency = max(1.0, rng.normal(2.6 if channel != "Video" else 3.4, 0.6))
            reach = impressions / frequency

            region = REGIONS[(cid + i) % len(REGIONS)]
            device = DEVICES[(cid * 2 + i) % len(DEVICES)]

            social_engagements = None
            if channel in ("Social",):
                social_engagements = (clicks * rng.uniform(1.8, 3.2))

            rows.append(
                dict(
                    date=d.isoformat(),
                    campaign_id=cid,
                    campaign_name=name,
                    channel=channel,
                    objective=objective,
                    region=region,
                    device=device,
                    planned_daily_budget=round(planned_daily_budget, 2),
                    impressions=round(impressions),
                    clicks=round(clicks, 2),
                    spend=round(spend, 2),
                    conversions=round(conversions, 3),
                    revenue=round(revenue, 2),
                    frequency=round(frequency, 2),
                    reach=round(reach),
                    social_engagements=round(social_engagements, 2) if social_engagements is not None else np.nan,
                )
            )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["campaign_id", "date"]).reset_index(drop=True)

    df = add_derived_features(df)
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level ratio features. Safe against div-by-zero via replace-with-NaN."""
    out = df.copy()

    def safe_div(a, b):
        return np.where(b > 0, a / b, np.nan)

    out["ctr"] = safe_div(out["clicks"], out["impressions"])
    out["cpc"] = safe_div(out["spend"], out["clicks"])
    out["cpm"] = safe_div(out["spend"], out["impressions"]) * 1000
    out["cvr"] = safe_div(out["conversions"], out["clicks"])
    out["cpa"] = safe_div(out["spend"], out["conversions"])
    out["roas"] = safe_div(out["revenue"], out["spend"])
    out["aov"] = safe_div(out["revenue"], out["conversions"])
    out["mer"] = out["roas"]  # portfolio-level MER == ROAS at this grain
    out["budget_pacing_ratio"] = safe_div(out["spend"], out["planned_daily_budget"])
    out["engagement_rate"] = safe_div(out["social_engagements"], out["impressions"])
    out["day_of_week"] = out["date"].dt.day_name()
    out["is_weekend"] = out["date"].dt.weekday >= 5
    out["week"] = out["date"].dt.to_period("W").apply(lambda p: p.start_time.date().isoformat())
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic campaign performance dataset")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--out", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    df = generate(days=args.days, seed=args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows x {len(df.columns)} columns -> {out_path}")
    print(f"Date range: {df['date'].min().date()} .. {df['date'].max().date()}")
    print(f"Campaigns: {df['campaign_name'].nunique()}  Channels: {df['channel'].nunique()}")


if __name__ == "__main__":
    main()
