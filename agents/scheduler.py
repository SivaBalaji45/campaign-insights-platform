"""Weekly report scheduler.

This is the actual, real scheduling mechanism for the "report each week's
insights to stakeholders" agent — a cron-triggered APScheduler job, not a
Claude Code-side scheduled task. It's meant to run as a long-lived process
(e.g. under systemd/launchd/pm2, or containerized) independent of any chat
session.

    python -m agents.scheduler --run-once     # run the weekly workflow immediately, then exit
    python -m agents.scheduler --daemon        # start the weekly cron schedule and block
"""
from __future__ import annotations

import argparse
import asyncio
import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from agents.orchestrator import run_weekly_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("scheduler")


def _run_weekly_job():
    logger.info("Running weekly campaign insights workflow…")
    result = asyncio.run(run_weekly_workflow())
    logger.info(
        "Weekly report generated: status=%s html=%s manifest=%s",
        result.status_flag, result.html_path, result.manifest_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Weekly stakeholder report scheduler")
    parser.add_argument("--run-once", action="store_true", help="Run the workflow immediately and exit")
    parser.add_argument("--daemon", action="store_true", help="Start the weekly cron schedule and block")
    parser.add_argument("--day-of-week", default="mon", help="APScheduler cron day_of_week (default: mon)")
    parser.add_argument("--hour", type=int, default=8, help="Hour (24h) to run (default: 8)")
    parser.add_argument("--minute", type=int, default=0, help="Minute to run (default: 0)")
    args = parser.parse_args()

    if args.run_once:
        _run_weekly_job()
        return

    if args.daemon:
        scheduler = BlockingScheduler()
        trigger = CronTrigger(day_of_week=args.day_of_week, hour=args.hour, minute=args.minute)
        scheduler.add_job(_run_weekly_job, trigger, id="weekly_campaign_report")
        logger.info(
            "Scheduled weekly report for day_of_week=%s hour=%02d minute=%02d. Press Ctrl+C to stop.",
            args.day_of_week, args.hour, args.minute,
        )
        try:
            scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
