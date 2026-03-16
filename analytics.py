"""
Institutional-Grade Analytics Engine
Sortino, Calmar, VaR/CVaR, MAE/MFE, Monte Carlo, t-test, attribution,
correlation, signal decay, execution quality.

Modeled after Goldman Sachs / JP Morgan quant desk analytics.
"""

import logging
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class Analytics:
    """
    Institutional analytics suite for trading system performance analysis.

    All methods accept a list of closed Position objects and return
    structured results suitable for logging, dashboards, and skill.md.
    """

    # ─── Risk-Adjusted Returns ──────────────────────────────────

    @staticmethod
    def sortino_ratio(positions: list, risk_free_rate: float = 0.035, annualization: int = 252) -> float:
        """
        Sortino Ratio — like Sharpe but only penalizes downside volatility.
        Preferred by institutional desks because upside variance is desirable.

        Formula: (Annualized Return - Rf) / Downside Deviation
        """
        if len(positions) < 2:
            return 0.0

        returns = [p.pnl_pct for p in positions]
        mean_return = np.mean(returns)
        downside = [r for r in returns if r < 0]

        if not downside:
            return float("inf")

        downside_dev = np.sqrt(np.mean([r ** 2 for r in downside]))
        if downside_dev == 0:
            return float("inf")

        # Annualize: assume ~trades_per_day * 252 trading days
        trades_per_day = max(1, len(positions) / max(1, Analytics._trading_days(positions)))
        annual_factor = trades_per_day * annualization

        annualized_return = mean_return * annual_factor
        annualized_dd = downside_dev * math.sqrt(annual_factor)

        return (annualized_return - risk_free_rate) / annualized_dd

    @staticmethod
    def calmar_ratio(positions: list, initial_balance: float = 10000.0) -> float:
        """
        Calmar Ratio — annualized return / max drawdown.
        Used by hedge fund allocators to assess drawdown-adjusted returns.
        """
        if not positions:
            return 0.0

        total_return = sum(p.pnl for p in positions) / initial_balance
        max_dd = Analytics._max_drawdown(positions, initial_balance)

        if max_dd == 0:
            return float("inf") if total_return > 0 else 0.0

        trading_days = max(1, Analytics._trading_days(positions))
        annualized_return = total_return * (252 / trading_days)

        return annualized_return / max_dd

    # ─── Value at Risk ──────────────────────────────────────────

    @staticmethod
    def var_cvar(positions: list, confidence: float = 0.95) -> dict:
        """
        Value at Risk (VaR) and Conditional VaR (Expected Shortfall).

        VaR: Maximum expected loss at given confidence level.
        CVaR: Average loss in the worst (1-confidence)% of cases.
        CVaR is preferred by Basel III because it captures tail risk.
        """
        if len(positions) < 10:
            return {"var": 0.0, "cvar": 0.0, "confidence": confidence, "note": "insufficient data"}

        pnls = sorted([p.pnl for p in positions])
        n = len(pnls)
        cutoff_idx = int(n * (1 - confidence))
        cutoff_idx = max(1, cutoff_idx)

        var = abs(pnls[cutoff_idx - 1])  # Loss at confidence percentile
        tail = pnls[:cutoff_idx]
        cvar = abs(np.mean(tail)) if tail else var

        return {
            "var": var,
            "cvar": cvar,
            "confidence": confidence,
            "worst_trade": abs(min(pnls)),
            "tail_trades": cutoff_idx,
        }

    # ─── MAE / MFE Analysis ────────────────────────────────────

    @staticmethod
    def mae_mfe_analysis(positions: list) -> dict:
        """
        Maximum Adverse Excursion / Maximum Favorable Excursion.

        MAE: How far a trade went against you before exit.
        MFE: How far a trade went in your favor before exit.

        Used to optimize stop-loss and take-profit placement.
        Requires Position.mae_pct and .mfe_pct fields.
        """
        trades_with_data = [p for p in positions if hasattr(p, "mae_pct") and p.mae_pct is not None]

        if len(trades_with_data) < 5:
            return {"note": "insufficient MAE/MFE data", "count": len(trades_with_data)}

        wins = [p for p in trades_with_data if p.pnl > 0]
        losses = [p for p in trades_with_data if p.pnl <= 0]

        result = {
            "count": len(trades_with_data),
            "avg_mae": np.mean([p.mae_pct for p in trades_with_data]),
            "avg_mfe": np.mean([p.mfe_pct for p in trades_with_data]),
            "median_mae": float(np.median([p.mae_pct for p in trades_with_data])),
            "median_mfe": float(np.median([p.mfe_pct for p in trades_with_data])),
            "max_mae": max(p.mae_pct for p in trades_with_data),
            "max_mfe": max(p.mfe_pct for p in trades_with_data),
        }

        if wins:
            result["winners_avg_mae"] = np.mean([p.mae_pct for p in wins])
            result["winners_avg_mfe"] = np.mean([p.mfe_pct for p in wins])
        if losses:
            result["losers_avg_mae"] = np.mean([p.mae_pct for p in losses])
            result["losers_avg_mfe"] = np.mean([p.mfe_pct for p in losses])

        # Stop-loss optimization insight
        if wins:
            # If winners' average MAE is less than current stop, stop may be too wide
            result["stop_optimization"] = (
                "TIGHTEN" if result.get("winners_avg_mae", 0) < result["avg_mae"] * 0.5
                else "OK"
            )
        # Take-profit optimization insight
        if losses:
            # If losers went significantly in our favor before reversing, TP may be too wide
            losers_mfe = [p.mfe_pct for p in losses]
            if np.mean(losers_mfe) > 0.003:  # Losers averaged 0.3%+ in our favor
                result["tp_optimization"] = "TIGHTEN — losers had significant MFE before reversing"
            else:
                result["tp_optimization"] = "OK"

        return result

    # ─── Monte Carlo Simulation ─────────────────────────────────

    @staticmethod
    def monte_carlo(
        positions: list,
        initial_balance: float = 10000.0,
        simulations: int = 1000,
        horizon: int = 252,
    ) -> dict:
        """
        Monte Carlo simulation of future equity paths.

        Randomly resamples historical trade PnLs to project forward.
        Returns confidence intervals for future performance.
        """
        if len(positions) < 20:
            return {"note": "need 20+ trades for Monte Carlo", "count": len(positions)}

        pnls = [p.pnl for p in positions]
        trades_per_day = max(1, len(positions) / max(1, Analytics._trading_days(positions)))
        total_trades = int(trades_per_day * horizon)

        rng = np.random.default_rng()  # Non-deterministic seed for true randomness
        terminal_balances = []
        ruin_threshold = initial_balance * 0.5
        ruin_count = 0

        # Single simulation loop for both terminal balances and ruin probability
        for _ in range(simulations):
            sampled = rng.choice(pnls, size=total_trades, replace=True)
            equity_path = initial_balance + np.cumsum(sampled)
            terminal_balances.append(equity_path[-1])
            if np.min(equity_path) <= ruin_threshold:
                ruin_count += 1

        terminal_balances.sort()
        n = len(terminal_balances)

        return {
            "simulations": simulations,
            "horizon_days": horizon,
            "trades_simulated": total_trades,
            "median_balance": float(terminal_balances[n // 2]),
            "p5_balance": float(terminal_balances[int(n * 0.05)]),
            "p25_balance": float(terminal_balances[int(n * 0.25)]),
            "p75_balance": float(terminal_balances[int(n * 0.75)]),
            "p95_balance": float(terminal_balances[int(n * 0.95)]),
            "prob_profit": sum(1 for b in terminal_balances if b > initial_balance) / n,
            "prob_ruin": ruin_count / simulations,
            "expected_return": (float(np.mean(terminal_balances)) - initial_balance) / initial_balance,
        }

    # ─── Statistical Significance ───────────────────────────────

    @staticmethod
    def t_test(positions: list) -> dict:
        """
        One-sample t-test: Is mean trade return significantly > 0?

        If p-value < 0.05, we can say with 95% confidence that the
        strategy generates positive returns (not just luck).
        """
        if len(positions) < 10:
            return {"significant": False, "note": "need 10+ trades", "count": len(positions)}

        returns = [p.pnl_pct for p in positions]
        n = len(returns)
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)

        if std == 0:
            return {"significant": mean > 0, "t_stat": float("inf"), "p_value": 0.0}

        t_stat = mean / (std / math.sqrt(n))

        # Use scipy t-distribution if available, else erfc approximation
        try:
            from scipy.stats import t as t_dist
            p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 1))  # Two-tailed
        except ImportError:
            from math import erfc
            p_value = erfc(abs(t_stat) / math.sqrt(2))  # Normal approximation

        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "mean_return": float(mean),
            "std_return": float(std),
            "n_trades": n,
            "confidence": "95%" if p_value < 0.05 else ("90%" if p_value < 0.10 else "not significant"),
        }

    # ─── Performance Attribution ────────────────────────────────

    @staticmethod
    def attribution(positions: list) -> dict:
        """
        Multi-dimensional performance attribution.
        Breaks down PnL by: regime, strategy, time of day, day of week, symbol.

        This is what Goldman's PnL attribution desk produces daily.
        """
        if not positions:
            return {}

        result = {}

        # By Market Regime
        by_regime = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
        for p in positions:
            regime = p.signal.regime.value if hasattr(p.signal, "regime") else "unknown"
            by_regime[regime]["pnl"] += p.pnl
            by_regime[regime]["trades"] += 1
            if p.pnl > 0:
                by_regime[regime]["wins"] += 1

        result["by_regime"] = {
            k: {
                "pnl": round(v["pnl"], 2),
                "trades": v["trades"],
                "win_rate": v["wins"] / v["trades"] if v["trades"] else 0,
            }
            for k, v in by_regime.items()
        }

        # By Strategy (mean_reversion vs momentum)
        by_strategy = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
        for p in positions:
            strat = p.signal.strategy if hasattr(p.signal, "strategy") else "unknown"
            by_strategy[strat]["pnl"] += p.pnl
            by_strategy[strat]["trades"] += 1
            if p.pnl > 0:
                by_strategy[strat]["wins"] += 1

        result["by_strategy"] = {
            k: {
                "pnl": round(v["pnl"], 2),
                "trades": v["trades"],
                "win_rate": v["wins"] / v["trades"] if v["trades"] else 0,
            }
            for k, v in by_strategy.items()
        }

        # By Hour of Day
        by_hour = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
        for p in positions:
            hour = p.entry_time.hour if hasattr(p, "entry_time") and p.entry_time else 0
            by_hour[hour]["pnl"] += p.pnl
            by_hour[hour]["trades"] += 1
            if p.pnl > 0:
                by_hour[hour]["wins"] += 1

        result["by_hour"] = {
            k: {
                "pnl": round(v["pnl"], 2),
                "trades": v["trades"],
                "win_rate": v["wins"] / v["trades"] if v["trades"] else 0,
            }
            for k, v in sorted(by_hour.items())
        }

        # By Day of Week (0=Mon, 4=Fri)
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        by_day = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
        for p in positions:
            if hasattr(p, "entry_time") and p.entry_time:
                dow = p.entry_time.weekday()
                by_day[day_names[dow]]["pnl"] += p.pnl
                by_day[day_names[dow]]["trades"] += 1
                if p.pnl > 0:
                    by_day[day_names[dow]]["wins"] += 1

        result["by_day"] = {
            k: {
                "pnl": round(v["pnl"], 2),
                "trades": v["trades"],
                "win_rate": v["wins"] / v["trades"] if v["trades"] else 0,
            }
            for k, v in by_day.items()
        }

        # By Symbol
        by_symbol = defaultdict(lambda: {"pnl": 0, "trades": 0, "wins": 0})
        for p in positions:
            by_symbol[p.symbol]["pnl"] += p.pnl
            by_symbol[p.symbol]["trades"] += 1
            if p.pnl > 0:
                by_symbol[p.symbol]["wins"] += 1

        # Sort by PnL descending
        result["by_symbol"] = {
            k: {
                "pnl": round(v["pnl"], 2),
                "trades": v["trades"],
                "win_rate": v["wins"] / v["trades"] if v["trades"] else 0,
            }
            for k, v in sorted(by_symbol.items(), key=lambda x: x[1]["pnl"], reverse=True)
        }

        # By Exit Type
        by_exit = defaultdict(lambda: {"pnl": 0, "trades": 0})
        for p in positions:
            by_exit[p.status.value]["pnl"] += p.pnl
            by_exit[p.status.value]["trades"] += 1

        result["by_exit"] = {
            k: {"pnl": round(v["pnl"], 2), "trades": v["trades"]}
            for k, v in sorted(by_exit.items(), key=lambda x: x[1]["trades"], reverse=True)
        }

        return result

    # ─── Cross-Asset Correlation ────────────────────────────────

    @staticmethod
    def correlation_matrix(positions: list) -> dict:
        """
        Cross-asset return correlation matrix.
        Used for portfolio diversification and risk concentration analysis.
        """
        by_symbol = defaultdict(list)
        for p in positions:
            by_symbol[p.symbol].append(p.pnl_pct)

        symbols = [s for s, returns in by_symbol.items() if len(returns) >= 5]
        if len(symbols) < 2:
            return {"note": "need 2+ symbols with 5+ trades each"}

        # Pad to equal length using mean return
        max_len = max(len(by_symbol[s]) for s in symbols)
        matrix = {}

        for s1 in symbols:
            matrix[s1] = {}
            r1 = by_symbol[s1]
            for s2 in symbols:
                r2 = by_symbol[s2]
                # Use overlapping length
                min_len = min(len(r1), len(r2))
                if min_len < 3:
                    matrix[s1][s2] = 0.0
                    continue
                corr = float(np.corrcoef(r1[:min_len], r2[:min_len])[0, 1])
                matrix[s1][s2] = round(corr, 3) if not np.isnan(corr) else 0.0

        # Identify high correlations (potential risk concentration)
        high_corr_pairs = []
        seen = set()
        for s1 in symbols:
            for s2 in symbols:
                if s1 != s2 and (s2, s1) not in seen:
                    seen.add((s1, s2))
                    c = matrix[s1][s2]
                    if abs(c) > 0.7:
                        high_corr_pairs.append({"pair": f"{s1}/{s2}", "correlation": c})

        return {
            "matrix": matrix,
            "high_correlations": high_corr_pairs,
            "symbols": symbols,
        }

    # ─── Signal Decay ───────────────────────────────────────────

    @staticmethod
    def signal_decay(positions: list) -> dict:
        """
        Signal decay curve — how does trade PnL evolve over hold time?

        If PnL peaks early and decays, we're holding too long.
        If PnL keeps improving, we may exit too early.
        """
        if len(positions) < 10:
            return {"note": "need 10+ trades for signal decay analysis"}

        # Group PnL by bars_held
        by_bars = defaultdict(list)
        for p in positions:
            by_bars[p.bars_held].append(p.pnl_pct)

        # Compute average PnL at each bar count
        decay_curve = {}
        for bars in sorted(by_bars.keys()):
            if len(by_bars[bars]) >= 2:
                decay_curve[bars] = {
                    "avg_pnl_pct": round(float(np.mean(by_bars[bars])), 5),
                    "count": len(by_bars[bars]),
                }

        # Find optimal hold duration (highest avg PnL)
        if decay_curve:
            optimal_bars = max(decay_curve.keys(), key=lambda b: decay_curve[b]["avg_pnl_pct"])
        else:
            optimal_bars = 0

        # Average hold of winners vs losers
        wins = [p for p in positions if p.pnl > 0]
        losses = [p for p in positions if p.pnl <= 0]

        return {
            "decay_curve": decay_curve,
            "optimal_hold_bars": optimal_bars,
            "avg_hold_winners": round(np.mean([p.bars_held for p in wins]), 1) if wins else 0,
            "avg_hold_losers": round(np.mean([p.bars_held for p in losses]), 1) if losses else 0,
            "hold_recommendation": (
                "SHORTEN" if wins and losses and
                np.mean([p.bars_held for p in wins]) < np.mean([p.bars_held for p in losses])
                else "OK"
            ),
        }

    # ─── Execution Quality ──────────────────────────────────────

    @staticmethod
    def execution_quality(positions: list) -> dict:
        """
        Execution quality analysis — slippage and fill rate.
        Requires Position.slippage field (signal price vs fill price).
        """
        trades_with_slip = [p for p in positions if hasattr(p, "slippage") and p.slippage is not None]

        if not trades_with_slip:
            return {"note": "no slippage data recorded"}

        slippages = [p.slippage for p in trades_with_slip]
        slippage_pcts = [p.slippage / p.entry_price if p.entry_price else 0 for p in trades_with_slip]

        total_slippage_cost = sum(abs(s) * p.size for s, p in zip(slippages, trades_with_slip))

        return {
            "trades_measured": len(trades_with_slip),
            "avg_slippage": round(float(np.mean(slippages)), 4),
            "avg_slippage_pct": round(float(np.mean(slippage_pcts)), 6),
            "median_slippage": round(float(np.median(slippages)), 4),
            "max_slippage": round(float(max(slippages, key=abs)), 4),
            "total_slippage_cost": round(total_slippage_cost, 2),
            "positive_slippage_pct": round(
                sum(1 for s in slippages if s > 0) / len(slippages), 3
            ),
        }

    # ─── Streak Analysis ───────────────────────────────────────

    @staticmethod
    def streak_analysis(positions: list) -> dict:
        """
        Win/loss streak analysis.
        Long losing streaks can indicate regime change.
        """
        if not positions:
            return {}

        streaks = []
        current_streak = 0
        current_type = None

        for p in positions:
            is_win = p.pnl > 0
            if current_type is None:
                current_type = is_win
                current_streak = 1
            elif is_win == current_type:
                current_streak += 1
            else:
                streaks.append((current_type, current_streak))
                current_type = is_win
                current_streak = 1
        streaks.append((current_type, current_streak))

        win_streaks = [s[1] for s in streaks if s[0]]
        loss_streaks = [s[1] for s in streaks if not s[0]]

        return {
            "max_win_streak": max(win_streaks) if win_streaks else 0,
            "max_loss_streak": max(loss_streaks) if loss_streaks else 0,
            "avg_win_streak": round(np.mean(win_streaks), 1) if win_streaks else 0,
            "avg_loss_streak": round(np.mean(loss_streaks), 1) if loss_streaks else 0,
            "current_streak": current_streak,
            "current_streak_type": "WIN" if current_type else "LOSS",
        }

    # ─── Comprehensive Report ──────────────────────────────────

    @staticmethod
    def generate_report(
        positions: list,
        initial_balance: float = 10000.0,
        run_monte_carlo: bool = True,
    ) -> dict:
        """
        Generate the complete institutional analytics report.
        Called at end of day from main.py.
        """
        if not positions:
            return {"note": "no trades to analyze"}

        report = {
            "generated_at": datetime.now().isoformat(),
            "total_trades": len(positions),
        }

        # Risk-adjusted returns
        report["sortino_ratio"] = Analytics.sortino_ratio(positions)
        report["calmar_ratio"] = Analytics.calmar_ratio(positions, initial_balance)

        # Risk metrics
        report["var_cvar"] = Analytics.var_cvar(positions)

        # Trade quality
        report["mae_mfe"] = Analytics.mae_mfe_analysis(positions)
        report["signal_decay"] = Analytics.signal_decay(positions)
        report["execution"] = Analytics.execution_quality(positions)
        report["streaks"] = Analytics.streak_analysis(positions)

        # Statistical significance
        report["t_test"] = Analytics.t_test(positions)

        # Attribution
        report["attribution"] = Analytics.attribution(positions)

        # Correlation
        report["correlation"] = Analytics.correlation_matrix(positions)

        # Monte Carlo (optional, computationally expensive)
        if run_monte_carlo and len(positions) >= 20:
            report["monte_carlo"] = Analytics.monte_carlo(positions, initial_balance)

        return report

    # ─── Report Formatting (for logs) ──────────────────────────

    @staticmethod
    def format_report(report: dict) -> str:
        """Format the analytics report for log output."""
        if "note" in report:
            return f"Analytics: {report['note']}"

        lines = []
        lines.append("")
        lines.append("=" * 70)
        lines.append("INSTITUTIONAL ANALYTICS REPORT")
        lines.append("=" * 70)
        lines.append(f"Generated: {report.get('generated_at', 'N/A')}")
        lines.append(f"Total Trades: {report.get('total_trades', 0)}")
        lines.append("")

        # Risk-Adjusted Returns
        lines.append("─── Risk-Adjusted Returns ───")
        lines.append(f"  Sortino Ratio:  {report.get('sortino_ratio', 0):.3f}")
        lines.append(f"  Calmar Ratio:   {report.get('calmar_ratio', 0):.3f}")
        lines.append("")

        # VaR / CVaR
        vc = report.get("var_cvar", {})
        if "var" in vc:
            lines.append("─── Value at Risk (95%) ───")
            lines.append(f"  VaR:   {vc['var']:.2f} EUR")
            lines.append(f"  CVaR:  {vc['cvar']:.2f} EUR (Expected Shortfall)")
            lines.append(f"  Worst: {vc.get('worst_trade', 0):.2f} EUR")
            lines.append("")

        # t-test
        tt = report.get("t_test", {})
        if "t_stat" in tt:
            lines.append("─── Statistical Significance ───")
            lines.append(f"  t-statistic: {tt['t_stat']:.3f}")
            lines.append(f"  p-value:     {tt['p_value']:.4f}")
            lines.append(f"  Verdict:     {'SIGNIFICANT' if tt.get('significant') else 'NOT SIGNIFICANT'}")
            lines.append("")

        # Signal Decay
        sd = report.get("signal_decay", {})
        if "optimal_hold_bars" in sd:
            lines.append("─── Signal Decay ───")
            lines.append(f"  Optimal Hold: {sd['optimal_hold_bars']} bars")
            lines.append(f"  Avg Hold (wins):   {sd.get('avg_hold_winners', 0)} bars")
            lines.append(f"  Avg Hold (losses): {sd.get('avg_hold_losers', 0)} bars")
            lines.append(f"  Recommendation:    {sd.get('hold_recommendation', 'N/A')}")
            lines.append("")

        # Streaks
        sk = report.get("streaks", {})
        if sk:
            lines.append("─── Streaks ───")
            lines.append(f"  Max Win Streak:  {sk.get('max_win_streak', 0)}")
            lines.append(f"  Max Loss Streak: {sk.get('max_loss_streak', 0)}")
            lines.append(f"  Current:         {sk.get('current_streak', 0)} {sk.get('current_streak_type', '')}")
            lines.append("")

        # Attribution by Strategy
        attr = report.get("attribution", {})
        by_strat = attr.get("by_strategy", {})
        if by_strat:
            lines.append("─── Attribution by Strategy ───")
            for strat, data in by_strat.items():
                lines.append(
                    f"  {strat:20s}  PnL: {data['pnl']:+8.2f}  "
                    f"Trades: {data['trades']:3d}  WR: {data['win_rate']:.0%}"
                )
            lines.append("")

        # Attribution by Symbol (top 5)
        by_sym = attr.get("by_symbol", {})
        if by_sym:
            lines.append("─── Top/Bottom Symbols ───")
            items = list(by_sym.items())
            for sym, data in items[:3]:
                lines.append(
                    f"  {sym:10s}  PnL: {data['pnl']:+8.2f}  "
                    f"Trades: {data['trades']:3d}  WR: {data['win_rate']:.0%}"
                )
            if len(items) > 3:
                lines.append("  ...")
                for sym, data in items[-2:]:
                    lines.append(
                        f"  {sym:10s}  PnL: {data['pnl']:+8.2f}  "
                        f"Trades: {data['trades']:3d}  WR: {data['win_rate']:.0%}"
                    )
            lines.append("")

        # Monte Carlo
        mc = report.get("monte_carlo", {})
        if "median_balance" in mc:
            lines.append("─── Monte Carlo (1000 sims, 252 days) ───")
            lines.append(f"  P5  (worst case):  {mc['p5_balance']:,.2f} EUR")
            lines.append(f"  P25 (conservative):{mc['p25_balance']:,.2f} EUR")
            lines.append(f"  P50 (median):      {mc['median_balance']:,.2f} EUR")
            lines.append(f"  P75 (optimistic):  {mc['p75_balance']:,.2f} EUR")
            lines.append(f"  P95 (best case):   {mc['p95_balance']:,.2f} EUR")
            lines.append(f"  Prob of Profit:    {mc['prob_profit']:.0%}")
            lines.append(f"  Prob of Ruin:      {mc['prob_ruin']:.1%}")
            lines.append("")

        # Execution Quality
        eq = report.get("execution", {})
        if "avg_slippage" in eq:
            lines.append("─── Execution Quality ───")
            lines.append(f"  Avg Slippage:   {eq['avg_slippage']:.4f} EUR ({eq['avg_slippage_pct']:.4%})")
            lines.append(f"  Total Cost:     {eq['total_slippage_cost']:.2f} EUR")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    # ─── Helper Methods ─────────────────────────────────────────

    @staticmethod
    def _max_drawdown(positions: list, initial_balance: float) -> float:
        """Calculate max drawdown from position list."""
        if not positions:
            return 0.0

        equity = initial_balance
        peak = initial_balance
        max_dd = 0.0

        for p in positions:
            equity += p.pnl
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return max_dd

    @staticmethod
    def _trading_days(positions: list) -> int:
        """Count unique trading days from positions."""
        if not positions:
            return 0

        days = set()
        for p in positions:
            if hasattr(p, "entry_time") and p.entry_time:
                days.add(p.entry_time.date())

        return max(1, len(days))
