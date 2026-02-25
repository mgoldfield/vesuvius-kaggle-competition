#!/bin/bash
# Monitor eval chain every 5 minutes, report errors immediately
LOG="/workspace/vesuvius-kaggle-competition/logs/eval_external_data_chain_continue.log"
LAST_CHECK=""

while true; do
    # Check if eval process is still running
    if ! pgrep -f eval_external_data_chain > /dev/null 2>&1; then
        echo "[$(date)] EVAL CHAIN PROCESS DIED"
        # Check if it completed normally
        if grep -q "Eval chain complete" "$LOG" 2>/dev/null; then
            echo "[$(date)] Chain completed normally."
            grep "SUMMARY" -A 20 "$LOG" | tail -25
        else
            echo "[$(date)] ERROR: Chain died unexpectedly!"
            echo "Last 10 lines of log:"
            tail -10 "$LOG"
        fi
        break
    fi

    # Check for errors
    if grep -qi "error\|traceback\|exception\|OOM\|killed" "$LOG" 2>/dev/null; then
        ERRORS=$(grep -ci "error\|traceback\|exception\|OOM\|killed" "$LOG" 2>/dev/null)
        echo "[$(date)] WARNING: Found $ERRORS error-like lines in log"
        grep -i "error\|traceback\|exception\|OOM\|killed" "$LOG" | tail -5
    fi

    # Check for completed evals
    COMPLETED=$(grep -c "comp_score:" /workspace/vesuvius-kaggle-competition/logs/eval_external_data_ep*.log 2>/dev/null || echo 0)
    CURRENT_EP=$(grep "=== \[Pass" "$LOG" | tail -1)

    # Get latest per-volume result
    LATEST_VOL=$(grep "\[.*\/20\]" /workspace/vesuvius-kaggle-competition/logs/eval_external_data_ep*.log 2>/dev/null | tail -1)

    echo "[$(date)] Status: $COMPLETED epochs done | $CURRENT_EP | Latest: $LATEST_VOL"

    # Print any completed epoch summaries
    for ep in 1 4 7 10 13 15 2 3 5 6 8 9 11 12 14; do
        EPLOG="/workspace/vesuvius-kaggle-competition/logs/eval_external_data_ep${ep}.log"
        if [ -f "$EPLOG" ] && grep -q "comp_score:" "$EPLOG" 2>/dev/null; then
            COMP=$(grep "comp_score:" "$EPLOG" | tail -1 | awk '{print $2}')
            if [ "$COMP" != "" ] && ! echo "$LAST_CHECK" | grep -q "ep${ep}"; then
                SDICE=$(grep "sdice:" "$EPLOG" | tail -1 | awk '{print $2}')
                TOPO=$(grep "topo:" "$EPLOG" | tail -1 | awk '{print $2}')
                echo "  NEW RESULT: ep${ep} comp=$COMP sdice=$SDICE topo=$TOPO"
                LAST_CHECK="$LAST_CHECK ep${ep}"
            fi
        fi
    done

    sleep 300
done
