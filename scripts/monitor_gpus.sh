#!/bin/bash
# Monitor all 3 GPUs every 5 minutes
# Usage: bash scripts/monitor_gpus.sh

GPU1_SSH="ssh -i ~/.ssh/remote-gpu -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@74.2.96.43 -p 10816"
GPU2_SSH="ssh -i ~/.ssh/remote-gpu -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@82.221.170.234 -p 31488"

while true; do
    echo ""
    echo "========== GPU Monitor: $(TZ=America/New_York date '+%I:%M %p ET') ($(date -u '+%H:%M UTC')) =========="
    
    # GPU0 (local)
    echo ""
    echo "--- gpu0 (local 5090) ---"
    PROC=$(ps aux | grep -E 'train_transunet|eval_transunet' | grep -v grep | head -3)
    if [ -n "$PROC" ]; then
        echo "$PROC" | awk '{print "  PID=" $2, "CPU=" $3 "%", "MEM=" $4 "%", $11, $12, $13}'
    else
        echo "  No training/eval processes running"
    fi
    # Latest log line
    for f in /workspace/vesuvius-kaggle-competition/logs/train_eval_dist_sq_lowlr.log; do
        if [ -f "$f" ]; then
            LAST=$(grep -E '^Epoch|comp=|DONE|Error' "$f" | tail -1)
            echo "  Log: $LAST"
        fi
    done
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | while read line; do
        echo "  GPU: $line"
    done

    # GPU1 (remote)
    echo ""
    echo "--- gpu1 (remote 5090 @ 74.2.96.43:10816) ---"
    $GPU1_SSH "
        PROC=\$(ps aux | grep -E 'train_transunet|eval_transunet|rsync|scp' | grep -v grep | head -3)
        if [ -n \"\$PROC\" ]; then
            echo \"\$PROC\" | awk '{print \"  PID=\" \$2, \"CPU=\" \$3 \"%\", \"MEM=\" \$4 \"%\", \$11, \$12, \$13}'
        else
            echo '  No training/eval/transfer processes running'
        fi
        for f in /workspace/vesuvius-kaggle-competition/logs/train_eval_discrim_dist_sq.log; do
            if [ -f \"\$f\" ]; then
                LAST=\$(grep -E '^Epoch|comp=|DONE|Error|Dry run' \"\$f\" | tail -1)
                echo \"  Log: \$LAST\"
            fi
        done
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | while read line; do
            echo \"  GPU: \$line\"
        done
        echo \"  Disk: \$(df -h /workspace | tail -1 | awk '{print \$3 \"/\" \$2, \"used\"}')\"
    " 2>/dev/null || echo "  UNREACHABLE"

    # GPU2 (remote)
    echo ""
    echo "--- gpu2 (remote 5090 @ 82.221.170.234:31488) ---"
    $GPU2_SSH "
        PROC=\$(ps aux | grep -E 'train_transunet|eval_transunet|rsync|scp' | grep -v grep | head -3)
        if [ -n \"\$PROC\" ]; then
            echo \"\$PROC\" | awk '{print \"  PID=\" \$2, \"CPU=\" \$3 \"%\", \"MEM=\" \$4 \"%\", \$11, \$12, \$13}'
        else
            echo '  No training/eval/transfer processes running'
        fi
        for f in /workspace/vesuvius-kaggle-competition/logs/train_eval_frozen_dist_sq.log; do
            if [ -f \"\$f\" ]; then
                LAST=\$(grep -E '^Epoch|comp=|DONE|Error|Dry run' \"\$f\" | tail -1)
                echo \"  Log: \$LAST\"
            fi
        done
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null | while read line; do
            echo \"  GPU: \$line\"
        done
        echo \"  Disk: \$(df -h /workspace | tail -1 | awk '{print \$3 \"/\" \$2, \"used\"}')\"
    " 2>/dev/null || echo "  UNREACHABLE"

    echo ""
    echo "========== Next check in 5 minutes =========="
    sleep 300
done
