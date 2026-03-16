"""
Self-Learning Module
Continuously evaluates strategy performance and optimizes parameters.
Uses walk-forward optimization and parameter grid search.
Logs everything for accountability and improvement tracking.
"""

import json
import logging
import copy
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from itertools import product

import pandas as pd
import numpy as np

from config import TradingConfig, StrategyConfig, RiskConfig
from backtester import Backtester, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class LearningReport:
    """Report from a learning cycle."""
    timestamp: str
    trigger: str                        # Why learning was triggered
    current_performance: dict           # Stats before optimization
    best_parameters: dict               # Winning parameter set
    best_performance: dict              # Stats with best parameters
    improvement: dict                   # Delta between current and best
    all_tested: list[dict]              # All parameter combinations tested
    recommendation: str                 # Human-readable recommendation
    applied: bool = False               # Whether params were auto-applied


class SelfLearner:
    """
    Automated strategy optimization through systematic parameter search.

    Learning triggers:
    1. Scheduled: every N trades (evaluation_interval)
    2. Performance: win rate or profit factor drops below threshold
    3. Drawdown: max drawdown exceeds threshold
    4. Manual: on-demand optimization request

    Process:
    1. Collect recent trade data and market data
    2. Define parameter grid (bounded by safe ranges)
    3. Run walk-forward backtest for each parameter combination
    4. Select best parameters based on composite score
    5. Generate report with recommendation
    6. Optionally auto-apply new parameters
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.lc = config.learning
        self.reports: list[LearningReport] = []
        self.trade_count_since_last_eval: int = 0

        # Parameter search ranges (bounded for safety)
        self.param_grid = {
            "ema_fast": [5, 7, 9, 12],
            "ema_slow": [15, 21, 26, 30],
            "rsi_period": [5, 7, 9, 14],
            "rsi_oversold": [20, 25, 30],
            "rsi_overbought": [70, 75, 80],
            "bb_period": [10, 15, 20, 25],
            "bb_std": [1.5, 2.0, 2.5],
            "volume_multiplier": [1.0, 1.2, 1.5],
            "min_signal_confluence": [2, 3, 4],
        }

    def should_learn(self, current_stats: dict) -> tuple[bool, str]:
        """Check if learning should be triggered."""
        if not self.lc.enabled:
            return False, "Learning disabled"

        # Not enough data yet
        total_trades = current_stats.get("total_trades", 0)
        if total_trades < self.lc.min_trades_for_learning:
            return False, f"Need {self.lc.min_trades_for_learning} trades, have {total_trades}"

        # Scheduled evaluation — use actual trade count, not scan cycles
        trades_since_last = total_trades - self.trade_count_since_last_eval
        if trades_since_last >= self.lc.evaluation_interval:
            self.trade_count_since_last_eval = total_trades
            return True, "scheduled_evaluation"

        # Performance drop
        win_rate = current_stats.get("win_rate", 1.0)
        if win_rate < self.lc.min_win_rate:
            return True, f"win_rate_low ({win_rate:.1%})"

        profit_factor = current_stats.get("profit_factor", 999)
        if profit_factor < self.lc.min_profit_factor:
            return True, f"profit_factor_low ({profit_factor:.2f})"

        # Drawdown
        max_dd = current_stats.get("max_drawdown", 0)
        if max_dd > self.lc.max_drawdown_trigger:
            return True, f"drawdown_high ({max_dd:.2%})"

        return False, "No trigger"

    def optimize(
        self,
        df: pd.DataFrame,
        symbol: str,
        current_stats: dict,
        trigger: str = "manual",
        max_combinations: int = 100,
    ) -> LearningReport:
        """
        Run parameter optimization and generate a learning report.

        Uses a smart subset of the parameter grid to keep runtime manageable.
        """
        logger.info(f"Learning cycle started. Trigger: {trigger}")
        # trade_count_since_last_eval is now updated in should_learn() using actual trade count

        backtester = Backtester(self.config)
        all_results = []

        # Generate parameter combinations (limited for speed)
        param_combos = self._generate_param_combos(max_combinations)
        logger.info(f"Testing {len(param_combos)} parameter combinations")

        for i, params in enumerate(param_combos):
            try:
                # Create modified strategy config
                test_sc = copy.deepcopy(self.config.strategy)
                for key, value in params.items():
                    if hasattr(test_sc, key):
                        setattr(test_sc, key, value)

                # Skip invalid combos (ema_fast must be < ema_slow)
                if test_sc.ema_fast >= test_sc.ema_slow:
                    continue

                # Run backtest
                result = backtester.run(df, symbol, strategy_config=test_sc)

                score = self._composite_score(result.stats)
                all_results.append({
                    "parameters": params,
                    "stats": result.stats,
                    "score": score,
                })

                if (i + 1) % 20 == 0:
                    logger.info(f"  Progress: {i + 1}/{len(param_combos)} combinations tested")

            except Exception as e:
                logger.warning(f"  Combo {i} failed: {e}")

        if not all_results:
            logger.warning("No valid results from optimization")
            return self._empty_report(trigger, current_stats)

        # Sort by composite score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        best = all_results[0]

        # Calculate improvement
        current_score = self._composite_score(current_stats)
        improvement = {
            "score_delta": best["score"] - current_score,
            "win_rate_delta": best["stats"].get("win_rate", 0) - current_stats.get("win_rate", 0),
            "pf_delta": best["stats"].get("profit_factor", 0) - current_stats.get("profit_factor", 0),
        }

        # Generate recommendation
        if best["score"] > current_score * 1.1:
            recommendation = (
                f"STRONG RECOMMENDATION: Switch to new parameters. "
                f"Score improved by {improvement['score_delta']:.2f} "
                f"({best['score']:.2f} vs {current_score:.2f}). "
                f"Win rate: {best['stats'].get('win_rate', 0):.1%}, "
                f"PF: {best['stats'].get('profit_factor', 0):.2f}"
            )
        elif best["score"] > current_score:
            recommendation = (
                f"MODERATE: Marginal improvement found. "
                f"Consider switching if current performance continues to degrade."
            )
        else:
            recommendation = "KEEP CURRENT: No improvement found. Current parameters are optimal."

        report = LearningReport(
            timestamp=datetime.now().isoformat(),
            trigger=trigger,
            current_performance=current_stats,
            best_parameters=best["parameters"],
            best_performance=best["stats"],
            improvement=improvement,
            all_tested=[
                {"params": r["parameters"], "score": r["score"]}
                for r in all_results[:10]  # Top 10 only
            ],
            recommendation=recommendation,
        )

        self.reports.append(report)
        self._save_report(report, symbol)

        logger.info(f"Learning complete: {recommendation}")
        return report

    def _composite_score(self, stats: dict) -> float:
        """
        Calculate composite score for parameter comparison.
        Weights: win rate (30%), profit factor (30%), Sharpe (20%), low drawdown (20%).
        """
        if not stats or stats.get("total_trades", 0) < 10:
            return -999

        win_rate = stats.get("win_rate", 0)
        pf = min(stats.get("profit_factor", 0), 5.0)  # Cap at 5 to avoid outliers
        sharpe = min(stats.get("sharpe_ratio", 0), 5.0)
        max_dd = stats.get("max_drawdown", 1.0)

        # Penalize low trade count (need statistical significance)
        trade_count = stats.get("total_trades", 0)
        significance_factor = min(trade_count / 100, 1.0)

        # Drawdown score: lower is better (invert)
        dd_score = max(0, 1 - max_dd * 10)  # 10% DD = 0 score

        score = (
            win_rate * 0.30 +
            (pf / 5.0) * 0.30 +
            (max(sharpe, 0) / 5.0) * 0.20 +
            dd_score * 0.20
        ) * significance_factor

        return score

    def _generate_param_combos(self, max_combos: int) -> list[dict]:
        """Generate parameter combinations using smart sampling."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        # Calculate total combinations
        total = 1
        for v in values:
            total *= len(v)

        if total <= max_combos:
            # Test all combinations
            combos = []
            for combo in product(*values):
                combos.append(dict(zip(keys, combo)))
            return combos
        else:
            # Random sampling with time-based seed for exploration diversity
            rng = np.random.default_rng(int(datetime.now().timestamp()) % 2**31)
            combos = []
            seen = set()
            for _ in range(max_combos):
                combo = tuple(rng.choice(v) for v in values)
                if combo not in seen:
                    seen.add(combo)
                    combos.append(dict(zip(keys, [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in combo])))
            return combos

    def apply_parameters(self, params: dict, config: TradingConfig):
        """Apply optimized parameters to the live config."""
        for key, value in params.items():
            if hasattr(config.strategy, key):
                old_val = getattr(config.strategy, key)
                setattr(config.strategy, key, value)
                logger.info(f"Parameter updated: {key} = {old_val} → {value}")

    def _save_report(self, report: LearningReport, symbol: str):
        """Save learning report to disk."""
        reports_dir = Path(self.lc.reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = reports_dir / f"learning_{symbol}_{timestamp}.json"

        # Convert to serializable dict
        output = {
            "timestamp": report.timestamp,
            "trigger": report.trigger,
            "recommendation": report.recommendation,
            "best_parameters": report.best_parameters,
            "best_performance": {k: str(v) for k, v in report.best_performance.items()},
            "current_performance": {k: str(v) for k, v in report.current_performance.items()},
            "improvement": report.improvement,
            "top_10_combos": report.all_tested,
            "applied": report.applied,
        }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Learning report saved: {filename}")

    def _empty_report(self, trigger: str, current_stats: dict) -> LearningReport:
        """Create empty report when optimization fails."""
        return LearningReport(
            timestamp=datetime.now().isoformat(),
            trigger=trigger,
            current_performance=current_stats,
            best_parameters={},
            best_performance={},
            improvement={},
            all_tested=[],
            recommendation="FAILED: Optimization produced no valid results.",
        )

    def get_learning_history(self) -> list[dict]:
        """Get summary of all learning cycles."""
        return [
            {
                "timestamp": r.timestamp,
                "trigger": r.trigger,
                "recommendation": r.recommendation[:80],
                "applied": r.applied,
                "score_delta": r.improvement.get("score_delta", 0),
            }
            for r in self.reports
        ]
