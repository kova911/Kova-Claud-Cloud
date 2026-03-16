"""
Backtesting Engine
Walk-forward backtesting with realistic cost modeling.
Produces performance metrics and trade logs for the learning module.
"""

import logging
import json
import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from config import TradingConfig
from indicators import Indicators
from strategy_v2 import StrategyV2, SignalType
from risk_manager import RiskManager, PositionStatus

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest results."""
    stats: dict
    trades: list[dict]
    equity_curve: list[float]
    daily_returns: list[float]
    parameters: dict


class Backtester:
    """
    Event-driven backtester with walk-forward support.

    Features:
    - Realistic slippage and commission modeling (IBKR tiered)
    - Walk-forward optimization windows
    - Outputs trade log for self-learning module
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        # IBKR tiered commission model (EUR stocks)
        self.commission_rate = 0.0005       # 0.05% of trade value
        self.commission_min = 1.25          # EUR minimum per order
        self.slippage_pct = 0.0001          # 0.01% slippage per trade
        self.spread_pct = 0.0003            # 0.03% average spread

    def _calc_commission(self, price: float, quantity: int) -> float:
        """Calculate IBKR tiered commission for a single order."""
        return max(self.commission_min, price * quantity * self.commission_rate)

    def run(
        self,
        df: pd.DataFrame,
        symbol: str,
        strategy_config=None,
        risk_config=None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            df: OHLCV DataFrame (already cleaned)
            symbol: Ticker symbol
            strategy_config: Override strategy params (for optimization)
            risk_config: Override risk params

        Returns:
            BacktestResult with stats, trades, and equity curve
        """
        sc = strategy_config or self.config.strategy
        rc = risk_config or self.config.risk

        # Compute indicators
        df_with_indicators = Indicators.add_all(df.copy(), sc)

        # Initialize strategy and risk manager
        strategy = StrategyV2(sc, rc)
        risk_mgr = RiskManager(rc)

        trades = []
        equity_curve = [rc.initial_balance]

        # Minimum data required before generating signals
        min_bars = max(sc.ema_trend, sc.bb_period, 50)

        for i in range(min_bars, len(df_with_indicators)):
            # Get data window up to current bar (no lookahead)
            window = df_with_indicators.iloc[:i + 1]
            current_bar = df_with_indicators.iloc[i]
            current_price = current_bar["close"]
            current_high = current_bar["high"]
            current_low = current_bar["low"]

            # ─── Update existing positions ────────────────────
            for pos in list(risk_mgr.positions):
                if pos.status != PositionStatus.OPEN:
                    continue

                # Check if stop-loss or take-profit was hit during this bar
                # Use high/low for more realistic fill simulation
                # Commission is handled inside _close_position — do NOT deduct here
                if pos.side == SignalType.LONG:
                    if current_low <= pos.stop_loss:
                        fill_price = pos.stop_loss * (1 - self.slippage_pct)
                        risk_mgr._close_position(pos, fill_price, PositionStatus.CLOSED_SL)
                        continue
                    if current_high >= pos.take_profit:
                        fill_price = pos.take_profit * (1 - self.slippage_pct)
                        risk_mgr._close_position(pos, fill_price, PositionStatus.CLOSED_TP)
                        continue
                else:
                    if current_high >= pos.stop_loss:
                        fill_price = pos.stop_loss * (1 + self.slippage_pct)
                        risk_mgr._close_position(pos, fill_price, PositionStatus.CLOSED_SL)
                        continue
                    if current_low <= pos.take_profit:
                        fill_price = pos.take_profit * (1 + self.slippage_pct)
                        risk_mgr._close_position(pos, fill_price, PositionStatus.CLOSED_TP)
                        continue

                # Update position with close price
                risk_mgr.update_position(pos, current_price)

            # ─── Generate new signals ─────────────────────────
            signal = strategy.generate_signal(window, symbol)

            if signal:
                # Apply slippage to entry
                if signal.type == SignalType.LONG:
                    signal.price = current_price * (1 + self.slippage_pct + self.spread_pct / 2)
                else:
                    signal.price = current_price * (1 - self.slippage_pct - self.spread_pct / 2)

                # Recalculate stop/tp with adjusted price
                if signal.type == SignalType.LONG:
                    signal.stop_loss = signal.price * (1 - rc.stop_loss_pct)
                    signal.take_profit = signal.price * (1 + rc.take_profit_pct)
                else:
                    signal.stop_loss = signal.price * (1 + rc.stop_loss_pct)
                    signal.take_profit = signal.price * (1 - rc.take_profit_pct)

                # Calculate position size and open
                size = risk_mgr.calculate_position_size(signal)
                if size > 0:
                    can, _ = risk_mgr.can_trade()
                    if can:
                        risk_mgr.open_position(signal, size=size)

            equity_curve.append(risk_mgr.balance)

        # Close any remaining positions at last price
        if risk_mgr.positions:
            last_price = df_with_indicators.iloc[-1]["close"]
            risk_mgr.close_all_positions({symbol: last_price}, reason="backtest_end")

        # Build trade log (commission already deducted in pos.pnl by _close_position)
        for pos in risk_mgr.closed_positions:
            trade = {
                "symbol": pos.symbol,
                "side": pos.side.value,
                "entry_price": pos.entry_price,
                "exit_price": pos.exit_price,
                "size": pos.size,
                "pnl": pos.pnl,
                "pnl_pct": pos.pnl_pct,
                "status": pos.status.value,
                "bars_held": pos.bars_held,
                "entry_time": str(pos.entry_time),
                "exit_time": str(pos.exit_time),
                "strategy": pos.signal.strategy,
                "regime": pos.signal.regime.value,
                "confidence": pos.signal.confidence,
                "reason": pos.signal.reason,
            }
            trades.append(trade)

        # Calculate proper Sharpe ratio
        stats = risk_mgr.get_stats()
        stats["sharpe_ratio"] = self._calc_sharpe(equity_curve, df_with_indicators)

        # Parameters used
        params = {
            "ema_fast": sc.ema_fast,
            "ema_slow": sc.ema_slow,
            "rsi_period": sc.rsi_period,
            "rsi_oversold": sc.rsi_oversold,
            "rsi_overbought": sc.rsi_overbought,
            "bb_period": sc.bb_period,
            "bb_std": sc.bb_std,
            "stop_loss_pct": rc.stop_loss_pct,
            "take_profit_pct": rc.take_profit_pct,
            "volume_multiplier": sc.volume_multiplier,
            "min_signal_confluence": sc.min_signal_confluence,
            "primary_strategy": sc.primary_strategy,
            "adaptive_regime": sc.adaptive_regime,
        }

        result = BacktestResult(
            stats=stats,
            trades=trades,
            equity_curve=equity_curve,
            daily_returns=self._calc_daily_returns(equity_curve, df_with_indicators),
            parameters=params,
        )

        return result

    def _calc_sharpe(self, equity_curve: list[float], df: pd.DataFrame) -> float:
        """
        Calculate annualized Sharpe ratio using proper daily returns.
        Groups equity by trading day to get true daily returns.
        """
        if len(equity_curve) < 2:
            return 0.0

        equity_series = pd.Series(equity_curve)

        # Try to group by trading day for proper annualization
        if hasattr(df.index, 'date') and len(df) >= len(equity_curve) - 1:
            try:
                # Map equity points to dates and take end-of-day values
                dates = [df.index[0].date()] + [df.index[min(i, len(df)-1)].date()
                         for i in range(len(equity_curve)-1)]
                eq_df = pd.DataFrame({"equity": equity_curve, "date": dates})
                daily_eq = eq_df.groupby("date")["equity"].last()
                daily_returns = daily_eq.pct_change().dropna()

                if len(daily_returns) > 1 and daily_returns.std() > 0:
                    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
            except Exception:
                pass

        # Fallback: use bar returns with estimated bars-per-day
        bar_returns = equity_series.pct_change().dropna()
        if len(bar_returns) > 1 and bar_returns.std() > 0:
            bars_per_day = 78  # ~78 five-minute bars per XETRA day
            return float(bar_returns.mean() / bar_returns.std() * np.sqrt(252 * bars_per_day))
        return 0.0

    def _calc_daily_returns(self, equity_curve: list[float], df: pd.DataFrame) -> list[float]:
        """Calculate proper daily returns from equity curve."""
        if len(equity_curve) < 2:
            return []

        try:
            if hasattr(df.index, 'date') and len(df) >= len(equity_curve) - 1:
                dates = [df.index[0].date()] + [df.index[min(i, len(df)-1)].date()
                         for i in range(len(equity_curve)-1)]
                eq_df = pd.DataFrame({"equity": equity_curve, "date": dates})
                daily_eq = eq_df.groupby("date")["equity"].last()
                return daily_eq.pct_change().dropna().tolist()
        except Exception:
            pass

        # Fallback
        equity_series = pd.Series(equity_curve)
        return equity_series.pct_change().dropna().tolist()

    def walk_forward(
        self,
        df: pd.DataFrame,
        symbol: str,
        train_days: int = 30,
        test_days: int = 10,
        param_grid: Optional[dict] = None,
    ) -> list[BacktestResult]:
        """
        Walk-forward optimization: optimize on training window, test on next window, slide.

        For each window:
        1. Run parameter optimization on the training period
        2. Select best parameters
        3. Run out-of-sample test on the test period with those parameters
        4. Slide forward by test_days and repeat
        """
        results = []
        total_bars = len(df)

        # Estimate bars per day
        if hasattr(df.index, 'date'):
            unique_dates = df.index.date
            bars_per_day = total_bars / len(set(unique_dates))
        else:
            bars_per_day = 78  # ~78 5-min bars per XETRA day

        train_bars = int(train_days * bars_per_day)
        test_bars = int(test_days * bars_per_day)
        step = test_bars

        # Default parameter grid for optimization
        if param_grid is None:
            param_grid = {
                "ema_fast": [7, 9, 12],
                "ema_slow": [18, 21, 26],
                "rsi_oversold": [20, 25, 30],
                "rsi_overbought": [70, 75, 80],
                "min_signal_confluence": [2, 3],
            }

        i = 0
        window_num = 0
        while i + train_bars + test_bars <= total_bars:
            train_data = df.iloc[i:i + train_bars]
            test_data = df.iloc[i + train_bars:i + train_bars + test_bars]

            # Step 1: Optimize on training data
            best_sc, best_score = self._optimize_window(train_data, symbol, param_grid)

            # Step 2: Test on out-of-sample data with best parameters
            result = self.run(test_data, symbol, strategy_config=best_sc)
            result.parameters["window"] = window_num
            result.parameters["train_start"] = str(train_data.index[0])
            result.parameters["test_start"] = str(test_data.index[0])
            result.parameters["optimized_score"] = best_score
            results.append(result)

            logger.info(
                f"Walk-forward window {window_num}: "
                f"Train {train_data.index[0]} → {train_data.index[-1]}, "
                f"Test {test_data.index[0]} → {test_data.index[-1]}, "
                f"Trades: {result.stats.get('total_trades', 0)}, "
                f"Win Rate: {result.stats.get('win_rate', 0):.1%}"
            )

            i += step
            window_num += 1

        return results

    def _optimize_window(
        self, train_data: pd.DataFrame, symbol: str, param_grid: dict,
    ) -> tuple:
        """Find best parameters on a training window. Returns (best_config, best_score)."""
        from itertools import product as iter_product

        keys = list(param_grid.keys())
        values = list(param_grid.values())
        best_score = -999
        best_sc = copy.deepcopy(self.config.strategy)

        for combo in iter_product(*values):
            params = dict(zip(keys, combo))
            test_sc = copy.deepcopy(self.config.strategy)
            for key, value in params.items():
                if hasattr(test_sc, key):
                    setattr(test_sc, key, value)

            # Skip invalid combos
            if test_sc.ema_fast >= test_sc.ema_slow:
                continue

            try:
                result = self.run(train_data, symbol, strategy_config=test_sc)
                score = self._score(result.stats)
                if score > best_score:
                    best_score = score
                    best_sc = test_sc
            except Exception:
                continue

        return best_sc, best_score

    def _score(self, stats: dict) -> float:
        """Composite score for parameter comparison during walk-forward."""
        if not stats or stats.get("total_trades", 0) < 5:
            return -999
        wr = stats.get("win_rate", 0)
        pf = min(stats.get("profit_factor", 0), 5.0)
        dd = stats.get("max_drawdown", 1.0)
        dd_score = max(0, 1 - dd * 10)
        return wr * 0.4 + (pf / 5.0) * 0.3 + dd_score * 0.3

    def save_results(self, result: BacktestResult, symbol: str, path: str = "reports"):
        """Save backtest results to JSON for the learning module."""
        reports_dir = Path(path)
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = reports_dir / f"backtest_{symbol}_{timestamp}.json"

        output = {
            "symbol": symbol,
            "timestamp": timestamp,
            "stats": result.stats,
            "parameters": result.parameters,
            "trades": result.trades,
            "equity_curve_length": len(result.equity_curve),
            "final_equity": result.equity_curve[-1] if result.equity_curve else 0,
        }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Backtest results saved to {filename}")
        return filename
