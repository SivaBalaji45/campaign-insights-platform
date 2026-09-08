import pandas as pd
import pytest

from core.business_rules import evaluate_rules, load_rules

RULES = load_rules()


def _row(**overrides) -> pd.DataFrame:
    base = dict(
        campaign_id=1, campaign_name="Test Campaign", channel="Search",
        roas=3.0, cpa=10.0, ctr_wow_pct=0.0, cvr_wow_pct=0.0,
        revenue_wow_pct=0.0, budget_pacing_ratio=1.0, spend_wow_pct=0.0, roas_wow_pct=0.0,
    )
    base.update(overrides)
    return pd.DataFrame([base])


def test_roas_critical_fires_below_1x():
    df = _row(roas=0.5)
    insights = evaluate_rules(df, RULES)
    assert any(i.rule_id == "roas_critical" for i in insights)


def test_roas_warning_suppressed_when_critical_fires():
    """Same group (roas_health): only the more severe rule should survive."""
    df = _row(roas=0.5)  # triggers both roas_critical (<1.0) and roas_warning (<2.0)
    insights = evaluate_rules(df, RULES)
    roas_insights = [i for i in insights if i.group == "roas_health"]
    assert len(roas_insights) == 1
    assert roas_insights[0].severity == "critical"


def test_healthy_campaign_fires_no_roas_rule():
    df = _row(roas=5.0, budget_pacing_ratio=1.0)
    insights = evaluate_rules(df, RULES)
    assert not any(i.group == "roas_health" for i in insights)


def test_cpa_over_target_uses_channel_specific_threshold():
    # Search target is 45 in config; 45*1.25 = 56.25 -> 60 should fire, 50 should not.
    over = _row(channel="Search", cpa=60.0)
    under = _row(channel="Search", cpa=50.0)
    assert any(i.rule_id == "cpa_over_target" for i in evaluate_rules(over, RULES))
    assert not any(i.rule_id == "cpa_over_target" for i in evaluate_rules(under, RULES))


def test_multi_condition_rule_requires_all_conditions():
    only_roas_high = _row(roas=5.0, budget_pacing_ratio=1.2)  # pacing not under threshold
    both = _row(roas=5.0, budget_pacing_ratio=0.5)
    assert not any(i.rule_id == "high_performer_scale_opportunity" for i in evaluate_rules(only_roas_high, RULES))
    assert any(i.rule_id == "high_performer_scale_opportunity" for i in evaluate_rules(both, RULES))


def test_recommended_action_is_formatted_not_a_raw_template():
    df = _row(roas=0.5)
    insight = next(i for i in evaluate_rules(df, RULES) if i.rule_id == "roas_critical")
    assert "{value" not in insight.recommended_action
    assert "0.50x" in insight.recommended_action
