# Campaign Insights Platform

A synthetic-data campaign performance dashboard, driven by a real two-agent
**AutoGen** pipeline: an **Insight Agent** that analyzes the dashboard's data
on demand (Q&A + predictive suggestions), and a **Reporting Agent** that
turns that analysis into a weekly stakeholder report on a real schedule.

This was built to a specific bar: real orchestration (not one prompt doing
everything), explicit business rules (not vibe-coded thresholds), and a
genuine swap point to a hosted LLM (not a toy that only works with fake data).

## Architecture

```
data/generate_dataset.py   synthetic campaign-level dataset + row-level derived features
core/kpis.py               aggregation, week-over-week deltas, anomaly detection, forecasting
core/business_rules.py     config/business_rules.yaml -> concrete actionable insights
agents/tools.py            thin JSON-serializable wrappers around core/* for the agents
agents/model_client.py     ChatCompletionClient: RuleBasedChatCompletionClient (default,
                            zero-key) + build_model_client() factory (real LLM swap point)
agents/insight_agent.py    AssistantAgent + tools, answers questions with grounded numbers
agents/reporting_agent.py  AssistantAgent, reformats analysis into a stakeholder report
agents/orchestrator.py     RoundRobinGroupChat team: Insight Agent -> Reporting Agent
agents/scheduler.py        APScheduler cron job that runs the weekly workflow for real
dashboard/app.py           Streamlit dashboard (charts, KPIs, insights, chat, report button)
config/business_rules.yaml editable thresholds/actions -- the single source of truth
config/stakeholders.yaml   distribution list + delivery mode
outbox/                    generated weekly reports land here (delivery_mode: outbox)
```

```
Insight Agent  --(FunctionTool calls)-->  core/kpis.py + core/business_rules.py
     |                                          (the only source of numbers/claims)
     v
grounded analysis (Executive Summary, Key Findings, Predicted Outlook)
     |
     v  (RoundRobinGroupChat hands off)
Reporting Agent  -->  polished stakeholder report  -->  outbox/*.html + *.md + manifest.json
```

Every number the agents state comes from a tool call into `core/kpis.py` or
`core/business_rules.py` — the model client never invents a metric. This is
what "real orchestration, not vibe coding" means in practice: you can open
the transcript and see exactly which function produced which number.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py generate-data          # synthetic dataset -> data/campaign_performance.csv
python main.py dashboard               # streamlit dashboard on localhost:8501
python main.py ask "which campaigns should we pause this week?"
python main.py report                  # run the two-agent weekly workflow once, now
python main.py schedule --daemon       # real weekly cron schedule (Mon 08:00 by default)
```

Run the test suite:

```bash
pytest tests/
```

## The business rules (config/business_rules.yaml)

This is the layer that makes the output *actionable*, not just descriptive.
Marketing/finance can edit thresholds without touching agent code:

| Rule | Fires when | Action |
|---|---|---|
| `roas_critical` / `roas_warning` | ROAS < 1.0x / < 2.0x | pause & rework / review bids |
| `cpa_over_target` | CPA > 1.25x the channel's target CPA | reallocate budget |
| `ctr_drop_wow` / `cvr_drop_wow` | CTR/CVR down >15%/20% WoW | refresh creative / check tracking |
| `revenue_drop_wow` | Revenue down >25% WoW | escalate same day |
| `budget_overspend_pacing` / `budget_underspend_pacing` | spend pacing >115% / <70% of plan | cap budget / raise bids |
| `high_performer_scale_opportunity` | ROAS > 4x **and** under-pacing | scale budget +20-30% |
| `efficiency_drop_while_scaling` | spend up >15% WoW **and** ROAS down >20% WoW | pause scaling, audit |

Rules that share a `group` (e.g. `roas_health`) are mutually exclusive — a
campaign in critical ROAS trouble gets one flag, the most severe one, not a
redundant stack.

## Swapping in a real LLM

Every agent gets its model client from `agents/model_client.build_model_client(role)`.
By default (`MODEL_PROVIDER` unset) it returns `RuleBasedChatCompletionClient` —
a deterministic, KPI-grounded stand-in that implements the *entire* AutoGen
model-client protocol (tool-call decisions, reflection, token accounting) so
the whole pipeline runs with zero API keys and zero cost.

To use a real hosted model instead, set in `.env` (copy `.env.example`):

```bash
MODEL_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

or

```bash
MODEL_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

No other code changes anywhere in `agents/` or `dashboard/` — this is the
single swap point, by design.

## Scheduling (the "report each week's insights to stakeholders" agent)

`agents/scheduler.py` is a real APScheduler cron job (not a chat-side
scheduled task) — it's meant to run as a long-lived process:

```bash
python -m agents.scheduler --daemon --day-of-week mon --hour 8 --minute 0
```

Each run drives the same `RoundRobinGroupChat` team used by `main.py report`,
then writes `outbox/weekly_report_<timestamp>.{html,md}` plus a
`.manifest.json` listing who it was "delivered" to. `config/stakeholders.yaml`
controls `delivery_mode` (`outbox` today) and the distribution list — swapping
in real email/Slack delivery is a matter of reading that manifest and calling
an SMTP/webhook client from the scheduler job, once credentials are available.

## Enhancements already built in

- **Answers free-form questions**: the dashboard's "Ask the Insight Agent"
  chat calls the same agent/tools live, not canned answers.
- **Predicted suggestions on every answer**: `core/kpis.forecast_metric` (a
  transparent linear-trend forecast) is included in every response, with a
  direction, confidence, and a suggested action tied to that direction —
  never just a description of the past.
