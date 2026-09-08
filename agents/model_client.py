"""Model client layer for the AutoGen agents.

Two things live here:

1. `RuleBasedChatCompletionClient` — a real `ChatCompletionClient` implementation
   (satisfies AutoGen's full model-client protocol: tool-call decisions,
   reflection/synthesis, token accounting) whose "reasoning" is a deterministic
   business-rule + KPI-grounded decision engine instead of a hosted LLM. It lets
   the whole multi-agent pipeline (tool calling, team orchestration, scheduling)
   run and be demoed/tested with zero API keys and zero cost.

2. `build_model_client(role)` — a factory that is the *only* place the swap to
   a real hosted LLM happens. Set MODEL_PROVIDER=anthropic or openai (plus the
   matching API key) in the environment and every agent starts reasoning with
   a real model instead — no other code changes anywhere in agents/ or
   dashboard/.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

from dotenv import load_dotenv

from autogen_core import CancellationToken, FunctionCall
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    ModelCapabilities,
    ModelFamily,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import Tool, ToolSchema

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Intent parsing helpers (shared by the mock client's tool-call decisions and
# its final synthesis, so "what the question asked for" stays consistent).
# ---------------------------------------------------------------------------

_FORECAST_WORDS = ("forecast", "predict", "next week", "next 7", "outlook", "projection", "going to")
_UNDERPERFORM_WORDS = ("underperform", "worst", "losing money", "struggling", "bad campaign", "waste")
_ANOMALY_WORDS = ("anomal", "unusual", "spike", "drop", "weird", "outage", "glitch")
_TOP_WORDS = ("top", "best", "winning", "scale", "double down")
_METRIC_WORDS = {
    "spend": ["spend", "budget"],
    "conversions": ["conversion", "leads"],
    "clicks": ["click"],
    "revenue": ["revenue", "sales"],
}


def _extract_metric(text: str) -> str:
    low = text.lower()
    for metric, words in _METRIC_WORDS.items():
        if any(w in low for w in words):
            return metric
    return "revenue"


def _extract_campaign_name(text: str) -> Optional[str]:
    try:
        from data.generate_dataset import CAMPAIGNS
    except Exception:
        return None
    low = text.lower()
    for row in CAMPAIGNS:
        name = row[0]
        # match on any distinctive word (2+ chars) from the campaign name
        for token in re.split(r"[\s-]+", name):
            if len(token) > 3 and token.lower() in low:
                return name
    return None


def _latest_user_text(messages: Sequence[LLMMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, UserMessage) and isinstance(msg.content, str):
            return msg.content
    return ""


def _tool_name(tool: Union[Tool, ToolSchema]) -> str:
    if hasattr(tool, "schema"):
        return tool.schema["name"]  # type: ignore[index]
    return tool["name"]  # type: ignore[index]


def _rough_token_count(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Synthesis templates
# ---------------------------------------------------------------------------

SEVERITY_EMOJI = {"critical": "🔴", "warning": "🟡", "opportunity": "🟢"}


def _fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"${x:,.0f}"


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.1f}%"


def _synthesize_insight_response(question: str, tool_results: dict[str, Any]) -> str:
    low = question.lower()
    lines: list[str] = []

    portfolio = tool_results.get("get_portfolio_summary") or {}
    flags = tool_results.get("get_business_rule_flags") or []
    anomalies = tool_results.get("get_anomalies") or []
    forecast = tool_results.get("forecast_kpi") or {}
    breakdown = tool_results.get("get_campaign_breakdown") or []
    underperformers = tool_results.get("get_underperformers")

    lines.append(f"**Answering:** {question.strip()}" if question.strip() else "**Weekly campaign performance analysis**")
    lines.append("")

    if portfolio:
        lines.append("### Executive Summary")
        lines.append(
            f"Portfolio spent **{_fmt_money(portfolio.get('spend'))}** and generated "
            f"**{_fmt_money(portfolio.get('revenue'))}** in revenue over "
            f"{portfolio.get('window_start')} → {portfolio.get('window_end')} "
            f"(**{portfolio.get('roas', 0):.2f}x ROAS**, {_fmt_pct(portfolio.get('roas_wow_pct'))} WoW). "
            f"Conversions {_fmt_pct(portfolio.get('conversions_wow_pct'))} WoW, "
            f"CPA {_fmt_pct(portfolio.get('cpa_wow_pct'))} WoW."
        )
        lines.append("")

    if underperformers is not None:
        lines.append("### Underperforming Campaigns")
        if underperformers:
            for c in underperformers[:6]:
                lines.append(
                    f"- **{c['campaign_name']}** ({c['channel']}): {c['roas']:.2f}x ROAS on "
                    f"{_fmt_money(c['spend'])} spend ({_fmt_pct(c.get('roas_wow_pct'))} WoW)"
                )
        else:
            lines.append("- No campaigns are currently below the ROAS threshold. ✅")
        lines.append("")

    if flags:
        lines.append("### Key Findings & Actionable Recommendations")
        for f in sorted(flags, key=lambda x: {"critical": 0, "warning": 1, "opportunity": 2}[x["severity"]])[:8]:
            emoji = SEVERITY_EMOJI[f["severity"]]
            lines.append(f"- {emoji} **{f['campaign_name']}** — {f['title']}: {f['recommended_action']}")
        lines.append("")

    if anomalies and (any(w in low for w in _ANOMALY_WORDS) or not flags):
        lines.append("### Detected Anomalies")
        for a in anomalies[:5]:
            lines.append(
                f"- **{a['campaign_name']}** — {a['metric']} {a['direction']} on {a['date']} "
                f"(value {a['value']}, expected ~{a['expected']}, z={a['z_score']})"
            )
        lines.append("")

    if breakdown and any(w in low for w in _TOP_WORDS + ("campaign", "breakdown", "compare", "channel")):
        lines.append("### Campaign Snapshot")
        for c in breakdown[:6]:
            lines.append(
                f"- **{c['campaign_name']}** ({c['channel']}): {_fmt_money(c['spend'])} spend, "
                f"{c['roas']:.2f}x ROAS, {_fmt_pct(c.get('revenue_wow_pct'))} revenue WoW"
            )
        lines.append("")

    if forecast:
        arrow = {"up": "📈", "down": "📉", "flat": "➡️"}.get(forecast.get("direction"), "➡️")
        scope = f"for **{forecast.get('campaign_name')}**" if forecast.get("campaign_name") else "portfolio-wide"
        lines.append("### Predicted Outlook (next 7 days)")
        lines.append(
            f"{arrow} {forecast.get('metric', 'revenue').title()} {scope} is trending "
            f"**{forecast.get('direction')}** ({forecast.get('trend_pct_per_day', 0):+.2f}%/day, "
            f"{forecast.get('confidence')} confidence). Projected next-7-day total: "
            f"**{_fmt_money(forecast.get('forecast_total_horizon'))}**."
        )
        if forecast.get("direction") == "down":
            lines.append(
                "→ *Suggested action:* address the flagged critical/warning campaigns above before "
                "next week's spend commits, or the downward trend is likely to continue."
            )
        elif forecast.get("direction") == "up":
            lines.append(
                "→ *Suggested action:* current trajectory supports maintaining or increasing budget "
                "on the top performers identified above."
            )
        lines.append("")

    return "\n".join(lines).strip()


def _synthesize_report(conversation_text: str) -> str:
    has_critical = "🔴" in conversation_text or "CRITICAL" in conversation_text.upper()
    has_warning = "🟡" in conversation_text or "WARNING" in conversation_text.upper()
    flag = "🔴 Action Needed" if has_critical else ("🟡 Review Recommended" if has_warning else "🟢 On Track")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    header = (
        f"# Weekly Campaign Performance Report — {today}\n\n"
        f"**Status: {flag}**\n\n"
        "---\n\n"
    )
    footer = (
        "\n\n---\n"
        f"*Prepared automatically by the Campaign Insights Agent (AutoGen orchestration) on {today}. "
        "Business rules and thresholds live in `config/business_rules.yaml` — update them there, not in code.*"
        "\n\nREPORT_COMPLETE"
    )
    return header + conversation_text.strip() + footer


# ---------------------------------------------------------------------------
# The mock ChatCompletionClient
# ---------------------------------------------------------------------------


class RuleBasedChatCompletionClient(ChatCompletionClient):
    """Deterministic, KPI-grounded stand-in for a hosted LLM.

    persona="insight_analyst": receives tools, makes one round of tool calls
    driven by intent parsing of the user's question, then synthesizes a
    grounded narrative from the real tool outputs.

    persona="report_writer": receives no tools; reformats whatever the
    previous agent said into a polished, stakeholder-ready report.
    """

    def __init__(self, persona: str = "insight_analyst"):
        self._persona = persona
        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Union[Tool, ToolSchema]] = [],
        tool_choice: Union[Tool, str] = "auto",
        json_output: Optional[Union[bool, type]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        prompt_tokens = sum(_rough_token_count(str(getattr(m, "content", ""))) for m in messages)

        if self._persona == "report_writer" or not tools:
            source_text = ""
            for msg in reversed(messages):
                if isinstance(msg, AssistantMessage) and isinstance(msg.content, str):
                    source_text = msg.content
                    break
            if not source_text:
                source_text = _latest_user_text(messages)
            content = _synthesize_report(source_text)
            usage = RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=_rough_token_count(content))
            self._total_usage.prompt_tokens += usage.prompt_tokens
            self._total_usage.completion_tokens += usage.completion_tokens
            return CreateResult(finish_reason="stop", content=content, usage=usage, cached=False)

        already_called = any(isinstance(m, FunctionExecutionResultMessage) for m in messages)

        if not already_called:
            question = _latest_user_text(messages)
            calls = self._decide_tool_calls(question, tools)
            usage = RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=len(calls) * 12)
            self._total_usage.prompt_tokens += usage.prompt_tokens
            self._total_usage.completion_tokens += usage.completion_tokens
            return CreateResult(finish_reason="function_calls", content=calls, usage=usage, cached=False)

        question = _latest_user_text(messages)
        tool_results: dict[str, Any] = {}
        for msg in messages:
            if isinstance(msg, FunctionExecutionResultMessage):
                for result in msg.content:
                    try:
                        tool_results[result.name] = json.loads(result.content)
                    except (json.JSONDecodeError, TypeError):
                        tool_results[result.name] = result.content

        content = _synthesize_insight_response(question, tool_results)
        usage = RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=_rough_token_count(content))
        self._total_usage.prompt_tokens += usage.prompt_tokens
        self._total_usage.completion_tokens += usage.completion_tokens
        return CreateResult(finish_reason="stop", content=content, usage=usage, cached=False)

    def _decide_tool_calls(self, question: str, tools: Sequence[Union[Tool, ToolSchema]]) -> list[FunctionCall]:
        available = {_tool_name(t) for t in tools}
        low = question.lower()
        wanted: list[tuple[str, dict]] = []

        if "get_portfolio_summary" in available:
            wanted.append(("get_portfolio_summary", {}))
        if "get_business_rule_flags" in available:
            wanted.append(("get_business_rule_flags", {}))
        if "get_anomalies" in available:
            wanted.append(("get_anomalies", {}))
        if "get_campaign_breakdown" in available:
            sort_by = "roas" if any(w in low for w in _TOP_WORDS) else "spend"
            wanted.append(("get_campaign_breakdown", {"sort_by": sort_by, "top_n": 6}))
        if "get_underperformers" in available and any(w in low for w in _UNDERPERFORM_WORDS):
            wanted.append(("get_underperformers", {}))
        if "forecast_kpi" in available:
            metric = _extract_metric(question)
            campaign_name = _extract_campaign_name(question)
            args = {"metric": metric}
            if campaign_name:
                args["campaign_name"] = campaign_name
            wanted.append(("forecast_kpi", args))

        return [
            FunctionCall(id=f"call_{i}", name=name, arguments=json.dumps(args))
            for i, (name, args) in enumerate(wanted)
        ]

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Union[Tool, ToolSchema]] = [],
        tool_choice: Union[Tool, str] = "auto",
        json_output: Optional[Union[bool, type]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ):
        result = await self.create(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        if isinstance(result.content, str):
            yield result.content
        yield result

    async def close(self) -> None:
        return None

    def actual_usage(self) -> RequestUsage:
        return self._total_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Union[Tool, ToolSchema]] = []) -> int:
        return sum(_rough_token_count(str(getattr(m, "content", ""))) for m in messages)

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Union[Tool, ToolSchema]] = []) -> int:
        return 200_000 - self.count_tokens(messages, tools=tools)

    @property
    def capabilities(self) -> ModelCapabilities:
        return {"vision": False, "function_calling": self._persona == "insight_analyst", "json_output": False}

    @property
    def model_info(self) -> ModelInfo:
        return {
            "vision": False,
            "function_calling": self._persona == "insight_analyst",
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": False,
            "multiple_system_messages": True,
        }


# ---------------------------------------------------------------------------
# Factory — the single swap point for a real hosted LLM.
# ---------------------------------------------------------------------------


def build_model_client(role: str) -> ChatCompletionClient:
    """role: 'insight_analyst' or 'report_writer'.

    MODEL_PROVIDER=anthropic + ANTHROPIC_API_KEY -> real Claude.
    MODEL_PROVIDER=openai    + OPENAI_API_KEY    -> real GPT.
    MODEL_PROVIDER=gemini    + GEMINI_API_KEY    -> real Gemini (via Google AI Studio's
                                                     OpenAI-compatible endpoint).
    anything else (default)                     -> RuleBasedChatCompletionClient (no key needed).
    """
    provider = os.getenv("MODEL_PROVIDER", "mock").lower()

    if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        from autogen_ext.models.anthropic import AnthropicChatCompletionClient

        return AnthropicChatCompletionClient(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        return OpenAIChatCompletionClient(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        return OpenAIChatCompletionClient(
            model=model,
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            # Gemini 3.x's "thinking" mode requires a thought_signature to be
            # echoed back on every tool-call turn, which the generic OpenAI-
            # compatible client doesn't preserve -> multi-turn tool calls 400.
            # reasoning_effort="minimal" disables thinking, avoiding this.
            reasoning_effort="minimal",
            model_info={
                "vision": True,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.UNKNOWN,
                "structured_output": True,
                "multiple_system_messages": True,
            },
        )

    return RuleBasedChatCompletionClient(persona=role)
