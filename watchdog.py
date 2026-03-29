#!/usr/bin/env python3
"""
watchdog.py — Background job that periodically checks for stale RunPod pods
and terminates any that have been running longer than MAX_AGE_MINUTES.

Usage:
    # Run once:
    python watchdog.py --once

    # Run as a loop (default every 10 minutes):
    python watchdog.py --interval 600

    # Custom max age:
    python watchdog.py --max-age 40 --interval 600

Environment:
    RUNPOD_API_KEY  — required
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

# Allow importing from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runpod_helpers import list_pods, terminate_pod, get_pod, RUNPOD_API_KEY


def check_and_terminate_stale(max_age_minutes: int = 40) -> list[str]:
    """
    Check all running pods. Terminate any older than max_age_minutes.
    Returns list of terminated pod IDs.
    """
    now = datetime.now(timezone.utc)
    pods = list_pods()
    terminated = []

    if not pods:
        print(f"[{now.isoformat()}] No pods running.")
        return terminated

    print(f"[{now.isoformat()}] Found {len(pods)} pod(s):")

    for pod in pods:
        pid = pod["id"]
        name = pod.get("name", "unnamed")
        status = pod.get("desiredStatus", "?")
        started_str = pod.get("lastStartedAt")

        if not started_str:
            print(f"  {pid} ({name}): no start time — skipping")
            continue

        started_at = datetime.fromisoformat(started_str.replace("Z", "+00:00"))
        age_minutes = (now - started_at).total_seconds() / 60.0

        if age_minutes > max_age_minutes:
            print(f"  {pid} ({name}): {age_minutes:.1f}min > {max_age_minutes}min → TERMINATING")
            try:
                terminate_pod(pid)
                terminated.append(pid)
                print(f"    ✓ Terminated")
            except Exception as e:
                print(f"    ✗ Failed: {e}")
        else:
            remaining = max_age_minutes - age_minutes
            print(f"  {pid} ({name}): {age_minutes:.1f}min — {remaining:.1f}min remaining")

    return terminated


def run_watchdog(interval_seconds: int = 600, max_age_minutes: int = 40):
    """Run the watchdog in a loop."""
    print(f"Watchdog started: checking every {interval_seconds}s, max age {max_age_minutes}min")
    print(f"Press Ctrl+C to stop.\n")

    while True:
        try:
            terminated = check_and_terminate_stale(max_age_minutes)
            if terminated:
                print(f"  → Terminated {len(terminated)} pod(s): {terminated}\n")
            else:
                print(f"  → No stale pods.\n")
        except Exception as e:
            print(f"  → Error during check: {e}\n")

        time.sleep(interval_seconds)


if __name__ == "__main__":
    if not RUNPOD_API_KEY:
        # Check if it's set in environment for this process
        if not os.environ.get("RUNPOD_API_KEY"):
            print("ERROR: Set RUNPOD_API_KEY environment variable.")
            sys.exit(1)

    parser = argparse.ArgumentParser(description="RunPod stale-pod watchdog")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=600, help="Seconds between checks (default 600 = 10 min)")
    parser.add_argument("--max-age", type=int, default=40, help="Max pod age in minutes before termination (default 40)")
    args = parser.parse_args()

    if args.once:
        terminated = check_and_terminate_stale(args.max_age)
        print(f"\nDone. Terminated {len(terminated)} pod(s).")
    else:
        try:
            run_watchdog(args.interval, args.max_age)
        except KeyboardInterrupt:
            print("\nWatchdog stopped.")
