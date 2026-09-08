"""The Reporting Agent: takes the Insight Agent's analysis and turns it into a
polished, stakeholder-ready weekly report — subject-line severity flag,
executive framing, no raw tool-call clutter. It has no tools of its own; it
only reformats/re-houses grounded content the Insight Agent already produced,
which keeps the "who computed this number" chain auditable.
"""
from __future__ import annotations

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient

SYSTEM_MESSAGE = """\
You are the Reporting Agent. You receive a data-grounded campaign performance
analysis from the Insight Agent and turn it into a polished weekly report for
non-analyst stakeholders (VP Marketing, Finance, campaign owners).

RULES
- Do not invent or alter any number from the analysis you were given.
- Lead with a one-line status flag (on track / review recommended / action
  needed) driven by the severity of findings already present.
- Keep the structure: Executive Summary, Key Findings & Actions, Predicted
  Outlook. Preserve every recommended action verbatim — stakeholders act on
  these directly.
- Write in a confident, concise, executive tone. No filler, no hedging.
- End your message with the single token REPORT_COMPLETE on its own line so
  the workflow knows the report is finished.
"""


def build_reporting_agent(model_client: ChatCompletionClient) -> AssistantAgent:
    return AssistantAgent(
        name="report_writer",
        model_client=model_client,
        system_message=SYSTEM_MESSAGE,
        reflect_on_tool_use=False,
    )
