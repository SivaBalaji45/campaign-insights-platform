"""Real multi-agent orchestration (not a single scripted LLM call):

  Insight Agent  --(tool calls into core/kpis + core/business_rules)-->
  grounded analysis  -->  Reporting Agent  -->  polished stakeholder report

Two entry points:
- run_insight_query(question)   -> single-agent Q&A, used by the dashboard chat.
- run_weekly_workflow()          -> two-agent RoundRobinGroupChat team run,
                                     used by the scheduler for the weekly report.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import markdown as md
import yaml
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat

from agents.insight_agent import build_insight_agent
from agents.model_client import build_model_client
from agents.reporting_agent import build_reporting_agent

OUTBOX_DIR = Path(__file__).parent.parent / "outbox"
STAKEHOLDERS_PATH = Path(__file__).parent.parent / "config" / "stakeholders.yaml"

REPORT_HTML_SHELL = """\
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; background: #f4f5f7; margin: 0; padding: 32px 16px; color: #1a1a2e; }}
  .card {{ max-width: 720px; margin: 0 auto; background: #ffffff; border-radius: 12px; padding: 32px 40px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  h3 {{ font-size: 15px; color: #333; margin-top: 28px; border-bottom: 1px solid #eee; padding-bottom: 6px; }}
  p, li {{ font-size: 14px; line-height: 1.55; }}
  .status {{ display: inline-block; padding: 4px 12px; border-radius: 999px; font-size: 13px; font-weight: 600; background: #fdecea; color: #a4262c; }}
  .status.ok {{ background: #e6f4ea; color: #1e7e34; }}
  .status.warn {{ background: #fff4e5; color: #a15c00; }}
  hr {{ border: none; border-top: 1px solid #eee; margin: 20px 0; }}
  footer {{ font-size: 12px; color: #888; margin-top: 24px; }}
  .recipients {{ font-size: 12px; color: #666; margin-top: 8px; }}
</style>
</head>
<body>
<div class="card">
{body}
<footer>
  <hr>
  <div class="recipients"><strong>Distributed to:</strong> {recipients}</div>
</footer>
</div>
</body>
</html>
"""


def _load_stakeholders() -> dict:
    with open(STAKEHOLDERS_PATH) as f:
        return yaml.safe_load(f)


async def run_insight_query(question: str) -> str:
    """Single-turn Q&A used by the dashboard's 'Ask the Insight Agent' chat."""
    client = build_model_client("insight_analyst")
    agent = build_insight_agent(client)
    try:
        result = await agent.run(task=question)
        for msg in reversed(result.messages):
            if getattr(msg, "source", None) == "insight_analyst" and isinstance(msg.content, str):
                return msg.content
        return "No analysis was produced."
    finally:
        await client.close()


@dataclass
class WeeklyReportResult:
    markdown_path: str
    html_path: str
    manifest_path: str
    status_flag: str
    generated_at: str


async def run_weekly_workflow() -> WeeklyReportResult:
    """Real two-agent orchestration: Insight Agent analyzes -> Reporting Agent
    writes the stakeholder report. Uses AutoGen's RoundRobinGroupChat team so
    each agent's turn is a genuine, separately-modeled step, not one prompt
    doing both jobs."""
    insight_client = build_model_client("insight_analyst")
    report_client = build_model_client("report_writer")

    insight_agent = build_insight_agent(insight_client)
    reporting_agent = build_reporting_agent(report_client)

    termination = TextMentionTermination("REPORT_COMPLETE") | MaxMessageTermination(6)
    team = RoundRobinGroupChat(
        [insight_agent, reporting_agent],
        termination_condition=termination,
    )

    task = (
        "Produce the weekly campaign performance stakeholder report for the "
        "trailing 7 days: overall portfolio health, every actionable business-"
        "rule flag, any anomalies, and the predicted 7-day revenue outlook."
    )

    try:
        result = await team.run(task=task)
    finally:
        await insight_client.close()
        await report_client.close()

    report_text = ""
    for msg in reversed(result.messages):
        if getattr(msg, "source", None) == "report_writer" and isinstance(msg.content, str):
            report_text = msg.content
            break

    report_text = report_text.replace("REPORT_COMPLETE", "").strip()

    status_flag = "ok"
    if "🔴" in report_text or "Action Needed" in report_text:
        status_flag = "critical"
    elif "🟡" in report_text or "Review Recommended" in report_text:
        status_flag = "warning"

    stakeholders_cfg = _load_stakeholders()
    recipients = ", ".join(s["name"] for s in stakeholders_cfg.get("stakeholders", []))

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y-%m-%d_%H%M%S")
    OUTBOX_DIR.mkdir(parents=True, exist_ok=True)

    md_path = OUTBOX_DIR / f"weekly_report_{stamp}.md"
    md_path.write_text(report_text)

    status_class = {"critical": "", "warning": "warn", "ok": "ok"}[status_flag]
    body_html = md.markdown(report_text, extensions=["extra"])
    body_html = re.sub(
        r"<p><strong>Status:\s*(.*?)</strong></p>",
        rf'<p><span class="status {status_class}">\1</span></p>',
        body_html,
        count=1,
    )
    html_path = OUTBOX_DIR / f"weekly_report_{stamp}.html"
    html_path.write_text(
        REPORT_HTML_SHELL.format(
            title="Weekly Campaign Performance Report",
            body=body_html,
            recipients=recipients,
        )
    )

    manifest = {
        "generated_at": now.isoformat(),
        "status": status_flag,
        "delivery_mode": stakeholders_cfg.get("delivery_mode", "outbox"),
        "recipients": stakeholders_cfg.get("stakeholders", []),
        "markdown_path": str(md_path),
        "html_path": str(html_path),
    }
    manifest_path = OUTBOX_DIR / f"weekly_report_{stamp}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return WeeklyReportResult(
        markdown_path=str(md_path),
        html_path=str(html_path),
        manifest_path=str(manifest_path),
        status_flag=status_flag,
        generated_at=now.isoformat(),
    )
