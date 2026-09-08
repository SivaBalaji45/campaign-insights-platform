"""The Insight Agent: analyzes the campaign performance dashboard's underlying
data on demand, answers free-form stakeholder questions, and always grounds
its answer in real computed KPIs, the business-rule engine, and a short-term
forecast (so every answer carries an actionable recommendation *and* a
predicted suggestion, not just a description of the past).
"""
from __future__ import annotations

import json
from typing import Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from autogen_core.tools import FunctionTool

from agents import tools as t
from agents.model_client import RuleBasedChatCompletionClient

SYSTEM_MESSAGE = """\
You are the Insight Agent for a paid-media Campaign Performance dashboard.

ROLE
You turn raw campaign metrics into grounded, decision-ready analysis for
marketing and finance stakeholders. You are not a generic chatbot — every
factual claim you make must come from a tool call result, never invented.

BUSINESS CONTEXT
- The portfolio spans Search, Social, Display, Video, Email, and Affiliate.
- Business rules and thresholds (ROAS guardrails, CPA targets per channel,
  budget pacing bands) are defined in config/business_rules.yaml and are
  exposed to you via the get_business_rule_flags tool. Treat its output as
  the authoritative source of "what's actionable" — do not restate a generic
  best practice when a specific rule-based recommendation is available.

OUTPUT CONTRACT — every response must:
1. Directly answer what was asked, in the first line.
2. Cite concrete numbers (spend, ROAS, CPA, % change) — never vague language
   like "performance seems mixed".
3. Include at least one *actionable* recommendation (from business rule
   flags) when any are relevant to the question.
4. Include a short predictive/forward-looking note (from the forecast tool)
   — stakeholders should never leave without knowing what's likely to happen
   next and what to do about it.
5. Be concise: prefer bullet points over paragraphs.

Use the tools available to you to gather data before answering. Do not ask
the user clarifying questions unless the request is genuinely ambiguous
(e.g. "look into this" with no metric or campaign named) — prefer to give a
comprehensive default analysis over blocking on a question.
"""


def _tool_get_portfolio_summary(days: int = 7) -> str:
    """Portfolio-wide spend/revenue/ROAS/CTR/CVR/CPA for the trailing `days`
    days, with week-over-week percent change for each metric."""
    return json.dumps(t.get_portfolio_summary(days=days))


def _tool_get_campaign_breakdown(days: int = 7, top_n: int = 12, sort_by: str = "spend") -> str:
    """Per-campaign KPI table for the trailing `days` days, sorted by
    `sort_by` (spend, revenue, roas, cpa, roas_wow_pct, revenue_wow_pct)."""
    return json.dumps(t.get_campaign_breakdown(days=days, top_n=top_n, sort_by=sort_by))


def _tool_get_underperformers(days: int = 7, roas_threshold: float = 2.0) -> str:
    """Campaigns whose trailing ROAS is below roas_threshold, worst spend-waste first."""
    return json.dumps(t.get_underperformers(days=days, roas_threshold=roas_threshold))


def _tool_get_business_rule_flags(days: int = 7) -> str:
    """Evaluate the business-rule engine (config/business_rules.yaml) against
    the trailing `days` campaign summary; the authoritative actionable-insight source."""
    return json.dumps(t.get_business_rule_flags(days=days))


def _tool_get_anomalies(lookback_days: int = 21, z_thresh: float = 2.5, top_n: int = 10) -> str:
    """Statistically detected spend/conversions/ROAS/CTR anomalies per campaign."""
    return json.dumps(t.get_anomalies(lookback_days=lookback_days, z_thresh=z_thresh, top_n=top_n))


def _tool_forecast_kpi(metric: str = "revenue", campaign_name: Optional[str] = None, horizon_days: int = 7) -> str:
    """Linear-trend forecast for a metric over the next horizon_days. Pass
    campaign_name to scope to one campaign, or omit for the whole portfolio."""
    return json.dumps(t.forecast_kpi(metric=metric, campaign_name=campaign_name, horizon_days=horizon_days))


def _tool_list_campaigns() -> str:
    """List all campaigns with channel/objective, for name lookups."""
    return json.dumps(t.list_campaigns())


def build_tools() -> list[FunctionTool]:
    return [
        FunctionTool(_tool_get_portfolio_summary, description=_tool_get_portfolio_summary.__doc__, name="get_portfolio_summary"),
        FunctionTool(_tool_get_campaign_breakdown, description=_tool_get_campaign_breakdown.__doc__, name="get_campaign_breakdown"),
        FunctionTool(_tool_get_underperformers, description=_tool_get_underperformers.__doc__, name="get_underperformers"),
        FunctionTool(_tool_get_business_rule_flags, description=_tool_get_business_rule_flags.__doc__, name="get_business_rule_flags"),
        FunctionTool(_tool_get_anomalies, description=_tool_get_anomalies.__doc__, name="get_anomalies"),
        FunctionTool(_tool_forecast_kpi, description=_tool_forecast_kpi.__doc__, name="forecast_kpi"),
        FunctionTool(_tool_list_campaigns, description=_tool_list_campaigns.__doc__, name="list_campaigns"),
    ]


def build_insight_agent(model_client: ChatCompletionClient) -> AssistantAgent:
    # The mock client synthesizes its own narrative on a second create() call
    # (reflect_on_tool_use=True) and safely terminates the tool-iteration loop
    # itself. Some real providers' OpenAI-compatible endpoints (Gemini 3.x in
    # particular, as of this writing) can't reliably round-trip tool-call
    # results back into a second model call at all — not just for reflection,
    # but for the "want to call more tools?" check too — so for any real LLM
    # we cap iterations at 1 (one model call, execute what it asked for, done)
    # and skip reflection, building the narrative ourselves from the raw tool
    # results in orchestrator.run_insight_query: real LLM reasoning picks the
    # tools, our tested formatter grounds the final answer.
    is_mock = isinstance(model_client, RuleBasedChatCompletionClient)
    return AssistantAgent(
        name="insight_analyst",
        model_client=model_client,
        tools=build_tools(),
        system_message=SYSTEM_MESSAGE,
        reflect_on_tool_use=is_mock,
        max_tool_iterations=3 if is_mock else 1,
    )
