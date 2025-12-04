#!/usr/bin/env python3
"""
Monitor experiment queue progress.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

PROGRESS_FILE = Path("experiment_queue_progress.json")
QUEUE_FILE = Path("experiment_queue.json")
LOG_FILE = Path("experiment_queue.log")


def main():
    # Load queue
    if not QUEUE_FILE.exists():
        print("ERROR: Queue file not found")
        sys.exit(1)

    with open(QUEUE_FILE) as f:
        queue = json.load(f)

    total = len(queue)

    # Load progress
    if not PROGRESS_FILE.exists():
        print("Queue not started yet")
        sys.exit(0)

    with open(PROGRESS_FILE) as f:
        progress = json.load(f)

    completed = len(progress.get("completed", []))
    failed = len(progress.get("failed", []))
    current_index = progress.get("current_index", 0)
    remaining = total - current_index

    # Display summary
    print("=" * 80)
    print("EXPERIMENT QUEUE STATUS")
    print("=" * 80)
    print(f"Total experiments:     {total}")
    print(f"Completed:             {completed} ({completed/total*100:.1f}%)")
    print(f"Failed:                {failed} ({failed/total*100:.1f}%)")
    print(f"Remaining:             {remaining} ({remaining/total*100:.1f}%)")
    print(f"Current index:         {current_index}")
    print("=" * 80)

    # Show current experiment
    if current_index < total:
        current_exp = queue[current_index]
        print(f"\nCurrent: {current_exp['aggregation']} alpha={current_exp['alpha']} "
              f"adv={current_exp['adv_pct']}% seed={current_exp['seed']}")

    # Show next 5
    if current_index + 1 < total:
        print("\nNext 5 experiments:")
        for i in range(current_index + 1, min(current_index + 6, total)):
            exp = queue[i]
            print(f"  {i+1}. {exp['aggregation']} alpha={exp['alpha']} "
                  f"adv={exp['adv_pct']}% seed={exp['seed']}")

    # Show recent failures
    if failed > 0:
        print(f"\nRecent failures ({min(failed, 5)} of {failed}):")
        for exp in progress["failed"][-5:]:
            print(f"  - {exp['aggregation']} alpha={exp['alpha']} "
                  f"adv={exp['adv_pct']}% seed={exp['seed']}")

    # Show last log lines
    if LOG_FILE.exists():
        print("\n" + "=" * 80)
        print("Recent log (last 10 lines):")
        print("=" * 80)
        with open(LOG_FILE) as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(line.rstrip())

    print()


if __name__ == "__main__":
    main()
