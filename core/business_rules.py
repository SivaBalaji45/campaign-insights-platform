"""Business-rule engine: turns the per-campaign KPI summary into concrete,
actionable insights. This is the layer that keeps the agents' output grounded
in explicit, editable business logic instead of free-floating LLM judgment —
marketing leads can open config/business_rules.yaml and change a threshold
without touching any agent code.
"""
from __future__ import annotations

import operator as op
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

RULES_PATH = Path(__file__).parent.parent / "config" / "business_rules.yaml"

SEVERITY_RANK = {"critical": 3, "warning": 2, "opportunity": 1}

OPERATORS = {
    "lt": op.lt,
    "lte": op.le,
    "gt": op.gt,
    "gte": op.ge,
    "eq": op.eq,
}


class _SafeDict(dict):
    def __missing__(self, key):
        return f"{{{key}}}"


def load_rules(path: Optional[Path] = None) -> dict:
    path = path or RULES_PATH
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class Insight:
    rule_id: str
    group: str
    severity: str
    campaign_id: int
    campaign_name: str
    channel: str
    title: str
    message: str
    recommended_action: str
    business_impact: str
    metric_value: float = field(default=float("nan"))

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "group": self.group,
            "severity": self.severity,
            "campaign_id": self.campaign_id,
            "campaign_name": self.campaign_name,
            "channel": self.channel,
            "title": self.title,
            "message": self.message,
            "recommended_action": self.recommended_action,
            "business_impact": self.business_impact,
            "metric_value": self.metric_value,
        }


def _resolve_threshold(rule: dict, row: pd.Series, config: dict) -> Optional[float]:
    if "threshold" in rule:
        return float(rule["threshold"])
    if "threshold_by_channel" in rule:
        table_name = rule["threshold_by_channel"]
        table = config.get(table_name, {})
        base = table.get(row.get("channel"), table.get("default"))
        if base is None:
            return None
        return float(base) * float(rule.get("multiplier", 1.0))
    return None


def _check_condition(cond: dict, row: pd.Series, config: dict) -> tuple[bool, Optional[float], Optional[float]]:
    metric = cond["metric"]
    if metric not in row or pd.isna(row[metric]):
        return False, None, None
    value = float(row[metric])
    threshold = _resolve_threshold(cond, row, config)
    if threshold is None:
        return False, value, None
    comparator = OPERATORS[cond["operator"]]
    return bool(comparator(value, threshold)), value, threshold


def evaluate_rules(campaign_df: pd.DataFrame, rules_config: Optional[dict] = None) -> list[Insight]:
    """Evaluate every rule against every campaign row. Within each `group`,
    only the highest-severity firing rule per campaign is kept."""
    rules_config = rules_config or load_rules()
    rules = rules_config.get("rules", [])

    candidates: dict[tuple[int, str], Insight] = {}

    for _, row in campaign_df.iterrows():
        for rule in rules:
            fmt: dict[str, Any] = {
                "channel": row.get("channel"),
                "campaign_name": row.get("campaign_name"),
            }
            fired = False
            primary_value = float("nan")

            if "conditions" in rule:
                all_true = True
                for cond in rule["conditions"]:
                    ok, value, threshold = _check_condition(cond, row, rules_config)
                    all_true = all_true and ok
                    key = cond.get("as")
                    if key and value is not None:
                        fmt[key] = value
                    if key == "value" and value is not None:
                        primary_value = value
                fired = all_true
            else:
                ok, value, threshold = _check_condition(rule, row, rules_config)
                fired = ok
                if value is not None:
                    fmt["value"] = value
                    primary_value = value
                if threshold is not None:
                    fmt["threshold"] = threshold
                    if threshold:
                        fmt["pct_over_target"] = (value / threshold - 1) * 100
            if not fired:
                continue

            fmt.setdefault("value", primary_value)
            fmt["value_per_1000"] = fmt.get("value", 0) * 1000

            insight = Insight(
                rule_id=rule["id"],
                group=rule.get("group", rule["id"]),
                severity=rule["severity"],
                campaign_id=int(row["campaign_id"]),
                campaign_name=row["campaign_name"],
                channel=row["channel"],
                title=rule["title"],
                message=rule["title"],
                recommended_action=rule["recommended_action"].format_map(_SafeDict(fmt)),
                business_impact=rule.get("business_impact", "").format_map(_SafeDict(fmt)),
                metric_value=round(primary_value, 4) if primary_value == primary_value else float("nan"),
            )

            key = (insight.campaign_id, insight.group)
            existing = candidates.get(key)
            if existing is None or SEVERITY_RANK[insight.severity] > SEVERITY_RANK[existing.severity]:
                candidates[key] = insight

    insights = list(candidates.values())
    insights.sort(key=lambda i: (SEVERITY_RANK[i.severity], i.campaign_name), reverse=True)
    return insights


def insights_to_dicts(insights: list[Insight]) -> list[dict]:
    return [i.to_dict() for i in insights]
