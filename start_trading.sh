#!/bin/bash
# Auto-start trading system — called by cron on weekday mornings
# The system auto-shuts down when all markets close (~21:00 CET)

TRADING_DIR="/Users/kova/Desktop/Kova/claude/Claude trading"
PYTHON="$TRADING_DIR/venv/bin/python"
LOG_DIR="$TRADING_DIR/logs"
STARTUP_LOG="/tmp/trading_startup.log"

echo "$(date '+%Y-%m-%d %H:%M:%S') — Trading launcher triggered" >> "$STARTUP_LOG"

# Check if already running
if pgrep -f "python.*main\.py" > /dev/null 2>&1; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') — Already running (PID: $(pgrep -f 'python.*main\.py')), skipping" >> "$STARTUP_LOG"
    exit 0
fi

# Start trading system
cd "$TRADING_DIR"
nohup "$PYTHON" main.py >> "$LOG_DIR/startup_$(date +%Y%m%d).log" 2>&1 &
NEW_PID=$!

echo "$(date '+%Y-%m-%d %H:%M:%S') — Started trading system (PID: $NEW_PID)" >> "$STARTUP_LOG"
