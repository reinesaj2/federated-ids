#!/bin/bash
# Launch heterogeneity rerun experiments sequentially

QUEUE_FILE="heterogeneity_rerun_queue.json"
PROGRESS_FILE="heterogeneity_rerun_progress.json"
LOG_FILE="heterogeneity_rerun.log"

# Verify queue exists
if [ ! -f "$QUEUE_FILE" ]; then
    echo "ERROR: Queue file not found: $QUEUE_FILE"
    exit 1
fi

# Launch sequential runner
nohup python scripts/run_full_queue_direct.py \
    --queue "$QUEUE_FILE" \
    --progress "$PROGRESS_FILE" \
    --log "$LOG_FILE" \
    > nohup_heterogeneity_rerun.out 2>&1 &

RUNNER_PID=$!
echo "Experiment runner started with PID: $RUNNER_PID"
echo $RUNNER_PID > heterogeneity_rerun.pid

echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo "  tail -f nohup_heterogeneity_rerun.out"
echo ""
echo "To check progress file:"
echo "  cat $PROGRESS_FILE | jq '.current_index, .completed | length'"
