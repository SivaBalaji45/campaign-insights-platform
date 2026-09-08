"""Campaign Insights Platform — CLI entrypoint.

    python main.py generate-data [--days 120]
    python main.py dashboard                     # launches Streamlit
    python main.py ask "which campaigns should we pause?"
    python main.py report --run-once              # generate this week's stakeholder report now
    python main.py schedule --daemon               # start the weekly cron scheduler
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def cmd_generate_data(args):
    from data.generate_dataset import generate

    df = generate(days=args.days, seed=args.seed)
    out = ROOT / "data" / "campaign_performance.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows -> {out}")


def cmd_dashboard(args):
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py")])


def cmd_ask(args):
    from agents.orchestrator import run_insight_query

    answer = asyncio.run(run_insight_query(args.question))
    print(answer)


def cmd_report(args):
    from agents.orchestrator import run_weekly_workflow

    result = asyncio.run(run_weekly_workflow())
    print(f"Status: {result.status_flag}")
    print(f"HTML:   {result.html_path}")
    print(f"MD:     {result.markdown_path}")
    print(f"Manifest: {result.manifest_path}")


def cmd_schedule(args):
    from agents.scheduler import main as scheduler_main

    forwarded = []
    if args.daemon:
        forwarded.append("--daemon")
    if args.run_once:
        forwarded.append("--run-once")
    sys.argv = ["scheduler"] + forwarded
    scheduler_main()


def main():
    parser = argparse.ArgumentParser(description="Campaign Insights Platform")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate-data", help="Generate the synthetic dataset")
    p_gen.add_argument("--days", type=int, default=120)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.set_defaults(func=cmd_generate_data)

    p_dash = sub.add_parser("dashboard", help="Launch the Streamlit dashboard")
    p_dash.set_defaults(func=cmd_dashboard)

    p_ask = sub.add_parser("ask", help="Ask the Insight Agent a question")
    p_ask.add_argument("question")
    p_ask.set_defaults(func=cmd_ask)

    p_report = sub.add_parser("report", help="Generate the weekly stakeholder report now")
    p_report.set_defaults(func=cmd_report)

    p_sched = sub.add_parser("schedule", help="Run the weekly report scheduler")
    p_sched.add_argument("--daemon", action="store_true")
    p_sched.add_argument("--run-once", action="store_true")
    p_sched.set_defaults(func=cmd_schedule)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
