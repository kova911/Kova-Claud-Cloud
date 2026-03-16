"""
Skill Document Auto-Updater
Automatically updates skill.md with performance metrics, parameter changes,
learning reports, and operational insights every 60 minutes during trading sessions.

This is the system's self-improvement mechanism — it creates a persistent record
of what works, what doesn't, and what was changed, enabling continuous refinement
across sessions.
"""

import logging
import time
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from analytics import Analytics

logger = logging.getLogger(__name__)

# Markers in skill.md for the auto-updated section
SECTION_START = "<!-- AUTO-UPDATED SECTION START -->"
SECTION_END = "<!-- AUTO-UPDATED SECTION END -->"


class SkillUpdater:
    """
    Automatically updates skill.md with live performance data every 60 minutes.

    Updates include:
    - Current session performance (trades, win rate, PnL, drawdown)
    - Active strategy parameters (if changed by self-learner)
    - Learning history (optimizations applied)
    - Market filter activity (blocked trades, VIX levels)
    - Operational log (errors, connection status)
    """

    def __init__(self, config, risk_mgr, learner):
        self.config = config
        self.risk_mgr = risk_mgr
        self.learner = learner
        self.skill_path = Path("skill.md")
        self._last_update: float = 0
        self.update_interval: int = 3600  # 60 minutes in seconds
        self._session_start = datetime.now()
        self._update_count = 0
        self._errors: list[str] = []
        # IBKR execution tracking (v2.6.0)
        self._bracket_orders_placed = 0
        self._bracket_orders_filled = 0
        self._tick_sizes_resolved: dict[str, float] = {}  # symbol → tick size
        self._delayed_data_symbols: set[str] = set()       # symbols using delayed data
        self._ibkr_errors: dict[str, int] = {}             # error_code → count

    def should_update(self) -> bool:
        """Check if 60 minutes have passed since last update."""
        return (time.time() - self._last_update) >= self.update_interval

    def update(self, force: bool = False):
        """
        Generate and write the auto-updated section of skill.md.

        Args:
            force: If True, update regardless of interval
        """
        if not force and not self.should_update():
            return

        try:
            self._update_count += 1
            content = self._generate_section()
            self._write_section(content)
            self._last_update = time.time()
            logger.info(f"skill.md auto-updated (update #{self._update_count})")
        except Exception as e:
            error_msg = f"Skill update failed: {e}"
            self._errors.append(error_msg)
            logger.error(error_msg)

    def log_error(self, error: str):
        """Record an error for inclusion in the next skill update."""
        self._errors.append(f"{datetime.now().strftime('%H:%M')} {error}")
        # Keep only last 20 errors
        self._errors = self._errors[-20:]

    def log_bracket_order(self, symbol: str, filled: bool, tick_size: float = 0):
        """Track bracket order execution for the dashboard."""
        self._bracket_orders_placed += 1
        if filled:
            self._bracket_orders_filled += 1
        if tick_size > 0:
            self._tick_sizes_resolved[symbol] = tick_size

    def log_delayed_data(self, symbol: str):
        """Track symbols using delayed market data."""
        self._delayed_data_symbols.add(symbol)

    def log_ibkr_error(self, error_code: int):
        """Track IBKR error frequency."""
        self._ibkr_errors[str(error_code)] = self._ibkr_errors.get(str(error_code), 0) + 1

    def _generate_section(self) -> str:
        """Generate the complete auto-updated section content."""
        now = datetime.now()
        stats = self.risk_mgr.get_stats()
        learning_history = self.learner.get_learning_history()
        sc = self.config.strategy
        rc = self.config.risk

        lines = []
        lines.append("")
        lines.append(SECTION_START)
        lines.append("")
        lines.append("## Live Performance Dashboard")
        lines.append("")
        lines.append(f"> **Last Updated**: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"> **Session Started**: {self._session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"> **Update Count**: {self._update_count}")
        lines.append(f"> **Mode**: {'PAPER' if self.config.ibkr.paper_trading else 'LIVE'}")
        lines.append("")

        # ─── Performance Metrics ──────────────────────────
        lines.append("### Session Performance")
        lines.append("")
        if stats.get("total_trades", 0) > 0:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Total Trades | {stats['total_trades']} |")
            lines.append(f"| Win Rate | {stats['win_rate']:.1%} |")
            lines.append(f"| Total PnL | {stats['total_pnl']:+.2f} EUR |")
            lines.append(f"| Avg Win | {stats['avg_win']:.2f} EUR |")
            lines.append(f"| Avg Loss | {stats['avg_loss']:.2f} EUR |")
            lines.append(f"| Profit Factor | {stats['profit_factor']:.2f} |")
            lines.append(f"| Max Drawdown | {stats['max_drawdown']:.2%} |")
            lines.append(f"| Balance | {stats['balance']:.2f} EUR |")
            lines.append(f"| Return | {stats['return_pct']:+.2%} |")
            lines.append(f"| Commission | {stats.get('total_commission', 0):.2f} EUR |")
            lines.append(f"| Avg Bars Held | {stats['avg_bars_held']:.1f} |")
            lines.append("")

            # Performance assessment
            lines.append("**Assessment**: ")
            if stats['win_rate'] >= 0.60 and stats['profit_factor'] >= 1.5:
                lines.append("EXCELLENT — strategy performing above targets.")
            elif stats['win_rate'] >= 0.55 and stats['profit_factor'] >= 1.3:
                lines.append("GOOD — strategy meeting minimum targets.")
            elif stats['win_rate'] >= 0.50:
                lines.append("WARNING — win rate below 55% target. Learning cycle recommended.")
            else:
                lines.append("CRITICAL — win rate below 50%. Immediate parameter review needed.")
            lines.append("")
        else:
            lines.append("*No trades executed yet this session.*")
            lines.append("")

        # ─── Institutional Analytics ─────────────────────
        closed_all = self.risk_mgr.closed_positions
        if len(closed_all) >= 10:
            lines.append("### Institutional Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")

            sortino = Analytics.sortino_ratio(closed_all)
            calmar = Analytics.calmar_ratio(closed_all, rc.initial_balance)
            lines.append(f"| Sortino Ratio | {sortino:.3f} |")
            lines.append(f"| Calmar Ratio | {calmar:.3f} |")

            vc = Analytics.var_cvar(closed_all)
            if "var" in vc:
                lines.append(f"| VaR (95%) | {vc['var']:.2f} EUR |")
                lines.append(f"| CVaR (Expected Shortfall) | {vc['cvar']:.2f} EUR |")

            tt = Analytics.t_test(closed_all)
            if "t_stat" in tt:
                sig_label = "YES" if tt["significant"] else "NO"
                lines.append(f"| t-statistic | {tt['t_stat']:.3f} (p={tt['p_value']:.4f}) |")
                lines.append(f"| Statistically Significant | {sig_label} |")

            streaks = Analytics.streak_analysis(closed_all)
            if streaks:
                lines.append(f"| Max Win Streak | {streaks.get('max_win_streak', 0)} |")
                lines.append(f"| Max Loss Streak | {streaks.get('max_loss_streak', 0)} |")

            # MAE/MFE summary
            mae_mfe = Analytics.mae_mfe_analysis(closed_all)
            if "avg_mae" in mae_mfe:
                lines.append(f"| Avg MAE | {mae_mfe['avg_mae']:.3%} |")
                lines.append(f"| Avg MFE | {mae_mfe['avg_mfe']:.3%} |")
                if "stop_optimization" in mae_mfe:
                    lines.append(f"| Stop Optimization | {mae_mfe['stop_optimization']} |")

            # Execution quality
            eq = Analytics.execution_quality(closed_all)
            if "avg_slippage" in eq:
                lines.append(f"| Avg Slippage | {eq['avg_slippage']:.4f} EUR ({eq['avg_slippage_pct']:.4%}) |")
                lines.append(f"| Total Slippage Cost | {eq['total_slippage_cost']:.2f} EUR |")

            lines.append("")

            # Signal decay recommendation
            sd = Analytics.signal_decay(closed_all)
            if "optimal_hold_bars" in sd:
                lines.append(f"**Signal Decay**: Optimal hold = {sd['optimal_hold_bars']} bars. ")
                lines.append(f"Winners avg {sd.get('avg_hold_winners', 0)} bars, losers avg {sd.get('avg_hold_losers', 0)} bars. ")
                lines.append(f"Recommendation: {sd.get('hold_recommendation', 'N/A')}")
                lines.append("")

        # ─── IBKR Execution Quality ─────────────────────
        if self._bracket_orders_placed > 0 or self._tick_sizes_resolved or self._ibkr_errors:
            lines.append("### IBKR Execution")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")

            if self._bracket_orders_placed > 0:
                fill_rate = self._bracket_orders_filled / self._bracket_orders_placed
                lines.append(f"| Bracket Orders | {self._bracket_orders_filled}/{self._bracket_orders_placed} filled ({fill_rate:.0%}) |")

            if self._tick_sizes_resolved:
                tick_summary = ", ".join(
                    f"{sym}: {tick}" for sym, tick in sorted(self._tick_sizes_resolved.items())
                )
                lines.append(f"| Tick Sizes | {tick_summary} |")

            if self._delayed_data_symbols:
                lines.append(f"| Delayed Data | {', '.join(sorted(self._delayed_data_symbols))} |")

            if self._ibkr_errors:
                error_summary = ", ".join(
                    f"E{code}: {count}x" for code, count in sorted(self._ibkr_errors.items())
                )
                lines.append(f"| IBKR Errors | {error_summary} |")

            lines.append("")

        # ─── Open Positions ──────────────────────────────
        open_positions = [p for p in self.risk_mgr.positions
                         if p.status.value == "open"]
        if open_positions:
            lines.append("### Open Positions")
            lines.append("")
            lines.append("| Symbol | Side | Entry | Current | PnL | Bars |")
            lines.append("|--------|------|-------|---------|-----|------|")
            for pos in open_positions:
                lines.append(
                    f"| {pos.symbol} | {pos.side.value} | "
                    f"{pos.entry_price:.2f} | {pos.current_price:.2f} | "
                    f"{pos.unrealized_pnl:+.2f} | {pos.bars_held} |"
                )
            lines.append("")

        # ─── Active Strategy Parameters ──────────────────
        lines.append("### Active Strategy Parameters")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append(f"| Primary Strategy | {sc.primary_strategy} |")
        lines.append(f"| Adaptive Regime | {'ON' if sc.adaptive_regime else 'OFF'} |")
        lines.append(f"| EMA Fast/Slow/Trend | {sc.ema_fast}/{sc.ema_slow}/{sc.ema_trend} |")
        lines.append(f"| RSI Period | {sc.rsi_period} |")
        lines.append(f"| RSI Oversold/Overbought | {sc.rsi_oversold}/{sc.rsi_overbought} |")
        lines.append(f"| BB Period/Std | {sc.bb_period}/{sc.bb_std} |")
        lines.append(f"| Volume Multiplier | {sc.volume_multiplier}x |")
        lines.append(f"| Min Confluence | {sc.min_signal_confluence} |")
        lines.append(f"| Stop-Loss | {rc.stop_loss_pct:.2%} |")
        lines.append(f"| Take-Profit | {rc.take_profit_pct:.2%} |")
        lines.append(f"| Risk per Trade | {rc.risk_per_trade:.2%} |")
        lines.append(f"| Max Daily Loss | {rc.max_daily_loss_pct:.2%} |")
        lines.append("")

        # ─── Risk Status ─────────────────────────────────
        lines.append("### Risk Status")
        lines.append("")
        lines.append(f"- Daily PnL: {self.risk_mgr.daily_pnl:+.2f} EUR")
        lines.append(f"- Daily Trades: {self.risk_mgr.daily_trades}/{rc.max_trades_per_day}")
        lines.append(f"- Consecutive Losses: {self.risk_mgr.consecutive_losses}/{rc.max_consecutive_losses}")
        lines.append(f"- Open Positions: {len(open_positions)}/{rc.max_open_positions}")
        lines.append(f"- Balance: {self.risk_mgr.balance:.2f} EUR")
        lines.append("")

        # ─── Learning History ─────────────────────────────
        if learning_history:
            lines.append("### Learning History")
            lines.append("")
            lines.append("| Time | Trigger | Score Delta | Applied |")
            lines.append("|------|---------|-------------|---------|")
            for entry in learning_history[-10:]:  # Last 10 entries
                timestamp = entry["timestamp"][:16]  # Trim to minute
                lines.append(
                    f"| {timestamp} | {entry['trigger']} | "
                    f"{entry['score_delta']:+.3f} | "
                    f"{'YES' if entry['applied'] else 'no'} |"
                )
            lines.append("")

            # Check if parameters were changed
            applied_count = sum(1 for e in learning_history if e["applied"])
            if applied_count > 0:
                lines.append(f"*{applied_count} parameter optimizations applied this session.*")
                lines.append("")

        # ─── Recent Closed Trades ─────────────────────────
        closed = self.risk_mgr.closed_positions[-10:]  # Last 10
        if closed:
            lines.append("### Recent Closed Trades")
            lines.append("")
            lines.append("| Symbol | Side | PnL | Exit Reason | Bars |")
            lines.append("|--------|------|-----|-------------|------|")
            for pos in closed:
                lines.append(
                    f"| {pos.symbol} | {pos.side.value} | "
                    f"{pos.pnl:+.2f} | {pos.status.value} | {pos.bars_held} |"
                )
            lines.append("")

        # ─── Errors & Incidents ───────────────────────────
        if self._errors:
            lines.append("### Operational Incidents")
            lines.append("")
            for error in self._errors[-10:]:  # Last 10
                lines.append(f"- {error}")
            lines.append("")

        # ─── Improvement Notes ────────────────────────────
        lines.append("### Self-Improvement Notes")
        lines.append("")
        lines.append(self._generate_improvement_notes(stats))
        lines.append("")

        lines.append(SECTION_END)
        lines.append("")

        return "\n".join(lines)

    def _generate_improvement_notes(self, stats: dict) -> str:
        """Generate actionable improvement recommendations based on current data."""
        notes = []

        if stats.get("total_trades", 0) < 10:
            notes.append("- Insufficient trade data for analysis. Continue collecting samples.")
            return "\n".join(notes) if notes else "*Collecting data...*"

        win_rate = stats.get("win_rate", 0)
        pf = stats.get("profit_factor", 0)
        avg_bars = stats.get("avg_bars_held", 0)
        max_dd = stats.get("max_drawdown", 0)

        # Win rate analysis
        if win_rate < 0.50:
            notes.append("- CRITICAL: Win rate below 50%. Consider increasing min_signal_confluence to 4.")
        elif win_rate < 0.55:
            notes.append("- Win rate below target (55%). Review RSI thresholds — oversold may be too aggressive.")
        elif win_rate > 0.70:
            notes.append("- Win rate very high (>70%). Check if take-profit is too tight — may be leaving money on table.")

        # Profit factor
        if pf < 1.0:
            notes.append("- CRITICAL: Profit factor below 1.0 — system is losing money. Stop trading and re-optimize.")
        elif pf < 1.3:
            notes.append("- Profit factor below 1.3 target. Consider widening take-profit or tightening stop-loss.")

        # Hold duration
        if avg_bars > 12:
            notes.append(f"- Average hold ({avg_bars:.0f} bars) approaching max_hold_candles. Signals may be too early.")
        elif avg_bars < 3:
            notes.append(f"- Average hold ({avg_bars:.0f} bars) very short. Check if stops are too tight.")

        # Drawdown
        if max_dd > 0.05:
            notes.append(f"- Max drawdown {max_dd:.1%} exceeds 5% threshold. Consider reducing risk_per_trade.")

        # Exit analysis
        closed = self.risk_mgr.closed_positions
        if len(closed) >= 10:
            sl_exits = sum(1 for p in closed if p.status.value == "closed_sl")
            tp_exits = sum(1 for p in closed if p.status.value == "closed_tp")
            trail_exits = sum(1 for p in closed if p.status.value == "closed_trail")
            time_exits = sum(1 for p in closed if p.status.value == "closed_time")

            sl_pct = sl_exits / len(closed)
            if sl_pct > 0.50:
                notes.append(f"- {sl_pct:.0%} of exits are stop-losses. Entry timing may be poor — consider adding confirmation bar.")
            if trail_exits > tp_exits and trail_exits > 5:
                notes.append(f"- Trailing stop closing more than take-profit ({trail_exits} vs {tp_exits}). TP may be too aggressive.")
            if time_exits > len(closed) * 0.20:
                notes.append(f"- {time_exits} time-based exits ({time_exits/len(closed):.0%}). Signals may lack conviction.")

        # IBKR execution health
        if self._bracket_orders_placed > 0:
            fill_rate = self._bracket_orders_filled / self._bracket_orders_placed
            if fill_rate < 0.80:
                notes.append(
                    f"- IBKR bracket fill rate low ({fill_rate:.0%}). "
                    f"Check tick sizes and limit price positioning."
                )

        if self._ibkr_errors:
            total_errors = sum(self._ibkr_errors.values())
            if total_errors > 10:
                notes.append(
                    f"- High IBKR error count ({total_errors}). "
                    f"Review error codes: {dict(self._ibkr_errors)}"
                )

        if not notes:
            notes.append("- System performing within all targets. No adjustments needed.")

        return "\n".join(notes)

    def _write_section(self, content: str):
        """Write the auto-updated section to skill.md."""
        if not self.skill_path.exists():
            logger.warning(f"skill.md not found at {self.skill_path}")
            return

        skill_text = self.skill_path.read_text(encoding="utf-8")

        # Check if section markers exist
        if SECTION_START in skill_text and SECTION_END in skill_text:
            # Replace existing section
            pattern = re.compile(
                re.escape(SECTION_START) + r".*?" + re.escape(SECTION_END),
                re.DOTALL,
            )
            new_text = pattern.sub(content.strip(), skill_text)
        else:
            # Append section at the end
            new_text = skill_text.rstrip() + "\n\n---\n" + content

        self.skill_path.write_text(new_text, encoding="utf-8")

    def get_summary(self) -> dict:
        """Return a summary of skill updater state."""
        return {
            "update_count": self._update_count,
            "last_update": datetime.fromtimestamp(self._last_update).isoformat() if self._last_update else None,
            "session_start": self._session_start.isoformat(),
            "errors": len(self._errors),
            "interval_minutes": self.update_interval // 60,
            "bracket_orders": f"{self._bracket_orders_filled}/{self._bracket_orders_placed}",
            "tick_sizes": dict(self._tick_sizes_resolved),
            "delayed_data_symbols": list(self._delayed_data_symbols),
            "ibkr_errors": dict(self._ibkr_errors),
        }
