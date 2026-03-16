"""
Google Chat Trading Update
Posts a trading performance summary to a Google Chat space via webhook.
Designed to run every 4 hours via scheduled task.
"""

import sqlite3
import json
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# ─── Config ─────────────────────────────────────────────────
DB_PATH = Path("/Users/kova/Desktop/Kova/claude/Claude trading/knowledge/trade_history.db")
INSIGHTS_PATH = Path("/Users/kova/Desktop/Kova/claude/Claude trading/knowledge/learning_insights.json")
WEBHOOK_URL = (
    "https://chat.googleapis.com/v1/spaces/AAQAk7ov2y4/messages"
    "?key=AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI"
    "&token=7_wZ9PbJYL9w-FIOwntTdkEpJv1FNO-8MyrxAq6qhPk"
)
TZ = ZoneInfo("Europe/Berlin")


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def query_today(conn):
    """Today's trading stats."""
    today = datetime.now(TZ).strftime("%Y-%m-%d")
    rows = conn.execute(
        """SELECT trade_id, symbol, direction, pnl, net_pnl, commission,
                  exit_reason, entry_price, exit_price, size,
                  signal_confidence, open_timestamp, close_timestamp,
                  strategy, regime
           FROM trades
           WHERE date(open_timestamp) = ? AND exit_reason != 'orphaned_cleanup'
           ORDER BY close_timestamp DESC""",
        (today,),
    ).fetchall()
    return rows


def query_period(conn, days):
    """Stats for the last N days."""
    cutoff = (datetime.now(TZ) - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = conn.execute(
        """SELECT pnl, net_pnl, commission, exit_reason, symbol, direction,
                  strategy, regime, open_hour
           FROM trades
           WHERE is_open = 0 AND exit_reason != 'orphaned_cleanup'
                 AND date(open_timestamp) >= ?
           ORDER BY close_timestamp""",
        (cutoff,),
    ).fetchall()
    return rows


def query_open_positions(conn):
    """Currently open positions."""
    rows = conn.execute(
        """SELECT symbol, direction, entry_price, size, open_timestamp,
                  COALESCE(stop_loss, 0) as sl, COALESCE(take_profit, 0) as tp,
                  signal_confidence
           FROM trades
           WHERE is_open = 1
           ORDER BY open_timestamp DESC""",
    ).fetchall()
    return rows


def compute_stats(rows):
    """Compute win rate, total PnL, avg trade, profit factor from rows."""
    if not rows:
        return {"count": 0, "wr": 0, "pnl": 0, "avg": 0, "pf": 0, "wins": 0, "losses": 0}

    pnls = [r["pnl"] for r in rows if r["pnl"] is not None]
    if not pnls:
        return {"count": 0, "wr": 0, "pnl": 0, "avg": 0, "pf": 0, "wins": 0, "losses": 0}

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_pnl = sum(pnls)
    gross_loss = abs(sum(losses)) if losses else 0.001

    return {
        "count": len(pnls),
        "wins": len(wins),
        "losses": len(losses),
        "wr": len(wins) / len(pnls) * 100 if pnls else 0,
        "pnl": total_pnl,
        "avg": total_pnl / len(pnls) if pnls else 0,
        "pf": sum(wins) / gross_loss if gross_loss > 0 else 0,
        "best": max(pnls) if pnls else 0,
        "worst": min(pnls) if pnls else 0,
        "commission": sum(r["commission"] for r in rows if r["commission"]) if rows else 0,
    }


def get_account_balance(conn):
    """Get latest account balance from risk state."""
    state_path = Path("/Users/kova/Desktop/Kova/claude/Claude trading/data/risk_state.json")
    if state_path.exists():
        state = json.loads(state_path.read_text())
        return state.get("balance", 10000)
    return 10000


def is_system_running():
    """Check if trading system is running."""
    import subprocess
    result = subprocess.run(["pgrep", "-f", "main.py"], capture_output=True, text=True)
    return bool(result.stdout.strip())


def get_learner_summary():
    """Get DB self-learner summary."""
    if not INSIGHTS_PATH.exists():
        return None
    try:
        data = json.loads(INSIGHTS_PATH.read_text())
        return data
    except Exception:
        return None


def build_message():
    """Build the Google Chat message card."""
    conn = get_db()
    now = datetime.now(TZ)

    # ─── Gather data ──────────────────────────────────────
    today_trades = query_today(conn)
    closed_today = [t for t in today_trades if t["pnl"] is not None]
    open_today = [t for t in today_trades if t["pnl"] is None]

    open_positions = query_open_positions(conn)
    period_7d = query_period(conn, 7)
    period_30d = query_period(conn, 30)

    stats_today = compute_stats(closed_today)
    stats_7d = compute_stats(period_7d)
    stats_30d = compute_stats(period_30d)

    balance = get_account_balance(conn)
    running = is_system_running()
    insights = get_learner_summary()

    conn.close()

    # ─── Format message ──────────────────────────────────
    status_emoji = "🟢" if running else "🔴"
    status_text = "RUNNING" if running else "STOPPED"

    lines = []
    lines.append(f"*🤖 Trading System Update*  —  {now.strftime('%a %d %b %H:%M CET')}")
    lines.append(f"System: {status_emoji} {status_text}  |  Balance: *€{balance:,.2f}*")
    lines.append("")

    # Today
    lines.append("*📊 Today*")
    if stats_today["count"] > 0:
        pnl_emoji = "📈" if stats_today["pnl"] >= 0 else "📉"
        lines.append(
            f"{pnl_emoji} P&L: *€{stats_today['pnl']:+.2f}*  |  "
            f"Trades: {stats_today['count']}  |  "
            f"Win Rate: {stats_today['wr']:.0f}%  |  "
            f"PF: {stats_today['pf']:.2f}"
        )
        lines.append(
            f"   Best: €{stats_today['best']:+.2f}  |  "
            f"Worst: €{stats_today['worst']:+.2f}  |  "
            f"Avg: €{stats_today['avg']:+.2f}"
        )
    else:
        lines.append("No closed trades today yet")

    # Open positions
    if open_positions:
        lines.append("")
        lines.append(f"*📂 Open Positions ({len(open_positions)})*")
        for pos in open_positions[:5]:
            direction_emoji = "🟢" if pos["direction"] == "LONG" else "🔴"
            lines.append(
                f"{direction_emoji} {pos['symbol']} {pos['direction']} "
                f"x{pos['size']} @ {pos['entry_price']:.2f}"
            )
    lines.append("")

    # 7-day rolling
    lines.append("*📅 7-Day Rolling*")
    if stats_7d["count"] > 0:
        pnl_emoji = "📈" if stats_7d["pnl"] >= 0 else "📉"
        lines.append(
            f"{pnl_emoji} P&L: *€{stats_7d['pnl']:+.2f}*  |  "
            f"Trades: {stats_7d['count']}  |  "
            f"WR: {stats_7d['wr']:.0f}%  |  "
            f"PF: {stats_7d['pf']:.2f}"
        )
    else:
        lines.append("No trades in last 7 days")
    lines.append("")

    # 30-day
    lines.append("*📆 30-Day*")
    if stats_30d["count"] > 0:
        pnl_emoji = "📈" if stats_30d["pnl"] >= 0 else "📉"
        lines.append(
            f"{pnl_emoji} P&L: *€{stats_30d['pnl']:+.2f}*  |  "
            f"Trades: {stats_30d['count']}  |  "
            f"WR: {stats_30d['wr']:.0f}%  |  "
            f"PF: {stats_30d['pf']:.2f}"
        )
    else:
        lines.append("No trades in last 30 days")

    # Today's trade details (last 5)
    if closed_today:
        lines.append("")
        lines.append("*🔍 Recent Trades*")
        for t in closed_today[:5]:
            result_emoji = "✅" if t["pnl"] and t["pnl"] > 0 else "❌"
            exit_tag = t["exit_reason"] or "?"
            pnl_val = t["pnl"] or 0
            lines.append(
                f"{result_emoji} {t['symbol']} {t['direction']} "
                f"€{pnl_val:+.2f} ({exit_tag})"
            )

    # DB Learner insights
    if insights:
        meta = insights.get("meta_notes", [])
        improving = [n for n in meta if "IMPROVING" in n.upper()]
        declining = [n for n in meta if "DECLINING" in n.upper() or "unprofitable" in n.lower()]

        if improving or declining:
            lines.append("")
            lines.append("*🧠 Self-Learner*")
            for note in (improving + declining)[:3]:
                lines.append(f"• {note}")

    # Top/bottom symbols today
    if closed_today:
        by_symbol = {}
        for t in closed_today:
            sym = t["symbol"]
            if sym not in by_symbol:
                by_symbol[sym] = 0
            by_symbol[sym] += t["pnl"] or 0

        if by_symbol:
            sorted_syms = sorted(by_symbol.items(), key=lambda x: x[1], reverse=True)
            best = sorted_syms[0]
            worst = sorted_syms[-1]
            if best[0] != worst[0]:
                lines.append("")
                lines.append(
                    f"🏆 Best: {best[0]} €{best[1]:+.2f}  |  "
                    f"💀 Worst: {worst[0]} €{worst[1]:+.2f}"
                )

    return "\n".join(lines)


def send_to_gchat(message: str):
    """Send a text message to Google Chat via webhook."""
    payload = json.dumps({"text": message}).encode("utf-8")

    req = urllib.request.Request(
        WEBHOOK_URL,
        data=payload,
        headers={"Content-Type": "application/json; charset=UTF-8"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.status
            body = resp.read().decode("utf-8")
            if status == 200:
                print(f"[GChat] Message sent successfully to Daily Trading space")
            else:
                print(f"[GChat] Unexpected status {status}: {body}")
    except urllib.error.HTTPError as e:
        print(f"[GChat] HTTP Error {e.code}: {e.read().decode()}")
    except urllib.error.URLError as e:
        print(f"[GChat] URL Error: {e.reason}")


def main():
    """Generate and send trading update."""
    print(f"[GChat] Generating trading update at {datetime.now(TZ).strftime('%Y-%m-%d %H:%M CET')}")
    message = build_message()
    print("─" * 50)
    print(message)
    print("─" * 50)
    send_to_gchat(message)


if __name__ == "__main__":
    main()
