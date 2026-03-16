"""
Pairs Trading — Renaissance-Inspired Statistical Arbitrage

Trades the spread between cointegrated stock pairs for market-neutral
mean-reversion. Uses Engle-Granger cointegration tests and z-score
of the spread for entry/exit signals.

Key principles:
- Market neutral: equal euro exposure on each leg
- Spread mean-reversion: enter when spread deviates, exit when it reverts
- Cointegration validation: only trade pairs that are statistically cointegrated
- Separate position management: tracks both legs as a single logical position

Default pairs:
- BMW.DE / MBG.DE (German auto — same industry dynamics)
- SAP.DE / SIE.DE (German tech/industrial — correlated macro exposure)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PairSignal:
    """A pairs trading signal with both legs."""
    pair: tuple
    direction: str              # "long_spread" or "short_spread"
    spread_zscore: float
    hedge_ratio: float
    leg_a_action: str           # "BUY" or "SELL" for stock A
    leg_b_action: str           # "BUY" or "SELL" for stock B
    leg_a_size: int
    leg_b_size: int
    leg_a_price: float          # Current price of A
    leg_b_price: float          # Current price of B
    entry_spread: float
    timestamp: datetime
    confidence: float
    reason: str


@dataclass
class PairPosition:
    """Tracks a live pairs position with both legs."""
    pair: tuple
    direction: str
    hedge_ratio: float
    # Leg A
    leg_a_symbol: str
    leg_a_side: str             # "LONG" or "SHORT"
    leg_a_entry_price: float
    leg_a_size: int
    # Leg B
    leg_b_symbol: str
    leg_b_side: str
    leg_b_entry_price: float
    leg_b_size: int
    # Fields with defaults
    leg_a_order_id: str = ""
    leg_b_order_id: str = ""
    # Spread tracking
    entry_spread: float = 0.0
    entry_zscore: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    # Status
    status: str = "open"        # "open", "closed_tp", "closed_sl", "closed_eod"
    exit_spread: float = 0.0
    pnl: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""


class PairsTrader:
    """
    Statistical arbitrage on pre-defined cointegrated pairs.

    Pipeline:
    1. Test cointegration (Engle-Granger) at startup and periodically
    2. Compute spread = price_A - hedge_ratio * price_B
    3. Z-score of spread using rolling window
    4. Entry: |z| > entry_zscore → go long spread if z < -entry, short if z > +entry
    5. Exit: |z| < exit_zscore (mean reversion) or |z| > stop_zscore (divergence)
    6. Track both legs as a single logical position
    """

    def __init__(self, config, risk_config):
        self.config = config
        self.rc = risk_config
        self._coint_results = {}      # pair_tuple -> {pvalue, hedge_ratio, is_valid, half_life}
        self._positions = []          # list[PairPosition]
        self._closed_positions = []   # list[PairPosition] (for PnL tracking)
        self._bars_since_coint_test = {}  # pair_tuple -> int

    @property
    def open_positions(self) -> list[PairPosition]:
        return [p for p in self._positions if p.status == "open"]

    @property
    def position_count(self) -> int:
        return len(self.open_positions)

    def test_cointegration(
        self, df_a: pd.DataFrame, df_b: pd.DataFrame, pair: tuple
    ) -> dict:
        """
        Engle-Granger cointegration test.
        Returns {pvalue, hedge_ratio, is_cointegrated, half_life}.
        """
        try:
            from statsmodels.tsa.stattools import coint

            prices_a = df_a["close"].dropna()
            prices_b = df_b["close"].dropna()

            # Align by using last N bars
            min_len = min(len(prices_a), len(prices_b))
            if min_len < 30:
                return {"is_cointegrated": False, "reason": "insufficient_data"}

            a = prices_a.iloc[-min_len:].values
            b = prices_b.iloc[-min_len:].values

            # Engle-Granger cointegration test
            score, pvalue, _ = coint(a, b)

            # Compute hedge ratio via OLS: A = beta * B + alpha
            X = np.column_stack([b, np.ones(len(b))])
            betas, _, _, _ = np.linalg.lstsq(X, a, rcond=None)
            hedge_ratio = float(betas[0])
            alpha = float(betas[1])

            # Compute spread
            spread = a - hedge_ratio * b - alpha

            # Half-life of mean reversion (AR(1) model)
            half_life = self._compute_half_life(spread)

            is_coint = pvalue < self.config.coint_pvalue_threshold

            result = {
                "is_cointegrated": is_coint,
                "pvalue": float(pvalue),
                "hedge_ratio": hedge_ratio,
                "alpha": alpha,
                "half_life": half_life,
                "spread_mean": float(np.mean(spread)),
                "spread_std": float(np.std(spread)),
                "test_time": datetime.now().isoformat(),
            }

            status = "PASS" if is_coint else "FAIL"
            logger.info(
                f"[PAIRS] Cointegration {pair[0]}/{pair[1]}: {status} "
                f"(p={pvalue:.4f}, hedge={hedge_ratio:.3f}, half_life={half_life:.1f})"
            )

            return result

        except ImportError:
            logger.warning("[PAIRS] statsmodels not installed — cointegration test unavailable")
            return {"is_cointegrated": False, "reason": "statsmodels_missing"}

        except Exception as e:
            logger.warning(f"[PAIRS] Cointegration test failed for {pair}: {e}")
            return {"is_cointegrated": False, "reason": str(e)}

    def scan_pairs(self, data_dict: dict) -> list[PairSignal]:
        """Scan all pairs for entry/exit signals."""
        if not self.config.enabled:
            return []

        signals = []

        for pair in self.config.pairs:
            pair_tuple = tuple(pair) if isinstance(pair, list) else pair
            sym_a, sym_b = pair_tuple

            if sym_a not in data_dict or sym_b not in data_dict:
                continue

            df_a = data_dict[sym_a]
            df_b = data_dict[sym_b]

            # Test/update cointegration periodically
            coint = self._coint_results.get(pair_tuple)
            bars = self._bars_since_coint_test.get(pair_tuple, self.config.retest_coint_interval_bars)

            if coint is None or bars >= self.config.retest_coint_interval_bars:
                coint = self.test_cointegration(df_a, df_b, pair_tuple)
                self._coint_results[pair_tuple] = coint
                self._bars_since_coint_test[pair_tuple] = 0
            else:
                self._bars_since_coint_test[pair_tuple] = bars + 1

            if not coint.get("is_cointegrated"):
                continue

            # Compute current spread z-score
            hedge_ratio = coint["hedge_ratio"]
            spread, zscore = self._compute_spread_zscore(
                df_a, df_b, hedge_ratio, coint.get("alpha", 0)
            )

            if spread is None or zscore is None:
                continue

            # Check for existing position in this pair
            existing = self._get_pair_position(pair_tuple)

            if existing:
                # Check exit conditions (handled in update_positions)
                continue
            else:
                # Check entry conditions
                if self.position_count >= self.config.max_pairs_positions:
                    continue

                price_a = float(df_a["close"].iloc[-1])
                price_b = float(df_b["close"].iloc[-1])

                if zscore > self.config.entry_zscore:
                    # Spread is too high → short spread (sell A, buy B)
                    signal = self._create_pair_signal(
                        pair_tuple, "short_spread", zscore, hedge_ratio,
                        price_a, price_b, spread
                    )
                    if signal:
                        signals.append(signal)
                        logger.info(
                            f"[PAIRS] Signal: SHORT spread {sym_a}/{sym_b} "
                            f"z={zscore:+.2f} spread={spread:.2f}"
                        )

                elif zscore < -self.config.entry_zscore:
                    # Spread is too low → long spread (buy A, sell B)
                    signal = self._create_pair_signal(
                        pair_tuple, "long_spread", zscore, hedge_ratio,
                        price_a, price_b, spread
                    )
                    if signal:
                        signals.append(signal)
                        logger.info(
                            f"[PAIRS] Signal: LONG spread {sym_a}/{sym_b} "
                            f"z={zscore:+.2f} spread={spread:.2f}"
                        )

        return signals

    def update_positions(self, data_dict: dict, collector=None, executor=None) -> list[PairPosition]:
        """Monitor open pairs positions for exit conditions."""
        closed = []

        for pos in self.open_positions:
            pair_tuple = tuple(pos.pair) if isinstance(pos.pair, list) else pos.pair
            sym_a, sym_b = pair_tuple

            if sym_a not in data_dict or sym_b not in data_dict:
                continue

            df_a = data_dict[sym_a]
            df_b = data_dict[sym_b]

            coint = self._coint_results.get(pair_tuple)
            if not coint:
                continue

            hedge_ratio = coint["hedge_ratio"]
            spread, zscore = self._compute_spread_zscore(
                df_a, df_b, hedge_ratio, coint.get("alpha", 0)
            )

            if spread is None or zscore is None:
                continue

            price_a = float(df_a["close"].iloc[-1])
            price_b = float(df_b["close"].iloc[-1])

            exit_reason = None

            # Check exit conditions
            if abs(zscore) < self.config.exit_zscore:
                exit_reason = "mean_reverted"
            elif abs(zscore) > self.config.stop_zscore:
                exit_reason = "stop_loss"
            elif self._check_time_exit(pos):
                exit_reason = "time_exit"

            if exit_reason:
                # Compute PnL
                pnl = self._compute_pair_pnl(pos, price_a, price_b)
                pos.status = f"closed_{exit_reason}"
                pos.exit_spread = spread
                pos.pnl = pnl
                pos.exit_time = datetime.now()
                pos.exit_reason = exit_reason

                # Close positions via executor
                if executor:
                    self._close_pair_legs(pos, executor)

                self._closed_positions.append(pos)
                closed.append(pos)

                logger.info(
                    f"[PAIRS] Closed {pos.direction} {sym_a}/{sym_b}: "
                    f"reason={exit_reason}, z={zscore:+.2f}, PnL=€{pnl:+.2f}"
                )

        # Remove closed positions from active list
        self._positions = [p for p in self._positions if p.status == "open"]

        return closed

    def register_position(self, signal: PairSignal, order_a=None, order_b=None) -> PairPosition:
        """Register a new pairs position after both legs are executed."""
        pos = PairPosition(
            pair=signal.pair,
            direction=signal.direction,
            hedge_ratio=signal.hedge_ratio,
            leg_a_symbol=signal.pair[0],
            leg_a_side="LONG" if signal.leg_a_action == "BUY" else "SHORT",
            leg_a_entry_price=order_a.filled_price if order_a and hasattr(order_a, 'filled_price') else signal.leg_a_price,
            leg_a_size=signal.leg_a_size,
            leg_a_order_id=str(order_a.order_id) if order_a and hasattr(order_a, 'order_id') else "",
            leg_b_symbol=signal.pair[1],
            leg_b_side="LONG" if signal.leg_b_action == "BUY" else "SHORT",
            leg_b_entry_price=order_b.filled_price if order_b and hasattr(order_b, 'filled_price') else signal.leg_b_price,
            leg_b_size=signal.leg_b_size,
            leg_b_order_id=str(order_b.order_id) if order_b and hasattr(order_b, 'order_id') else "",
            entry_spread=signal.entry_spread,
            entry_zscore=signal.spread_zscore,
            entry_time=datetime.now(),
        )

        self._positions.append(pos)
        logger.info(
            f"[PAIRS] Registered position: {pos.direction} "
            f"{pos.leg_a_symbol}({pos.leg_a_side} x{pos.leg_a_size}) / "
            f"{pos.leg_b_symbol}({pos.leg_b_side} x{pos.leg_b_size})"
        )
        return pos

    def _compute_spread_zscore(
        self, df_a: pd.DataFrame, df_b: pd.DataFrame,
        hedge_ratio: float, alpha: float = 0
    ) -> tuple:
        """Compute spread and its z-score."""
        try:
            prices_a = df_a["close"].dropna()
            prices_b = df_b["close"].dropna()

            min_len = min(len(prices_a), len(prices_b))
            if min_len < self.config.lookback_bars:
                return None, None

            a = prices_a.iloc[-min_len:].values
            b = prices_b.iloc[-min_len:].values

            # Spread = A - hedge_ratio * B - alpha
            spread = a - hedge_ratio * b - alpha
            spread_series = pd.Series(spread)

            # Rolling z-score of spread
            lookback = min(self.config.lookback_bars, len(spread))
            rolling_mean = spread_series.rolling(lookback, min_periods=20).mean().iloc[-1]
            rolling_std = spread_series.rolling(lookback, min_periods=20).std().iloc[-1]

            if pd.isna(rolling_mean) or pd.isna(rolling_std) or rolling_std <= 0:
                return float(spread[-1]), None

            zscore = (spread[-1] - rolling_mean) / rolling_std
            return float(spread[-1]), float(zscore)

        except Exception as e:
            logger.debug(f"[PAIRS] Spread computation failed: {e}")
            return None, None

    def _create_pair_signal(
        self, pair: tuple, direction: str, zscore: float,
        hedge_ratio: float, price_a: float, price_b: float, spread: float
    ) -> Optional[PairSignal]:
        """Create a PairSignal with proper sizing."""
        try:
            # Market-neutral sizing: equal euro exposure per leg
            risk_amount = self.rc.initial_balance * self.config.risk_per_pair
            euro_per_leg = risk_amount * 0.5  # Split between legs

            size_a = max(1, int(euro_per_leg / price_a))
            size_b = max(1, int(euro_per_leg / price_b))

            if direction == "long_spread":
                # Buy A, Sell B
                leg_a_action = "BUY"
                leg_b_action = "SELL"
            else:
                # Sell A, Buy B
                leg_a_action = "SELL"
                leg_b_action = "BUY"

            confidence = min(abs(zscore) / self.config.entry_zscore, 1.0)

            return PairSignal(
                pair=pair,
                direction=direction,
                spread_zscore=zscore,
                hedge_ratio=hedge_ratio,
                leg_a_action=leg_a_action,
                leg_b_action=leg_b_action,
                leg_a_size=size_a,
                leg_b_size=size_b,
                leg_a_price=price_a,
                leg_b_price=price_b,
                entry_spread=spread,
                timestamp=datetime.now(),
                confidence=confidence,
                reason=f"pairs {direction}: z={zscore:+.2f}, hedge={hedge_ratio:.3f}",
            )

        except Exception as e:
            logger.warning(f"[PAIRS] Signal creation failed: {e}")
            return None

    def _get_pair_position(self, pair: tuple) -> Optional[PairPosition]:
        """Find open position for a given pair."""
        for pos in self.open_positions:
            pos_pair = tuple(pos.pair) if isinstance(pos.pair, list) else pos.pair
            if pos_pair == pair:
                return pos
        return None

    def _compute_pair_pnl(self, pos: PairPosition, price_a: float, price_b: float) -> float:
        """Compute PnL for a pairs position."""
        try:
            # Leg A PnL
            if pos.leg_a_side == "LONG":
                pnl_a = (price_a - pos.leg_a_entry_price) * pos.leg_a_size
            else:
                pnl_a = (pos.leg_a_entry_price - price_a) * pos.leg_a_size

            # Leg B PnL
            if pos.leg_b_side == "LONG":
                pnl_b = (price_b - pos.leg_b_entry_price) * pos.leg_b_size
            else:
                pnl_b = (pos.leg_b_entry_price - price_b) * pos.leg_b_size

            # Total PnL minus estimated commissions (IBKR min €1.25/leg × 4 = €5.00)
            total_pnl = pnl_a + pnl_b - 5.00  # 4 legs × €1.25

            return total_pnl

        except Exception:
            return 0.0

    def _compute_half_life(self, spread: np.ndarray) -> float:
        """Compute half-life of mean reversion for the spread."""
        try:
            spread_lag = spread[:-1]
            spread_ret = spread[1:] - spread[:-1]

            if len(spread_lag) < 10:
                return float("inf")

            # OLS: spread_ret = phi * spread_lag + intercept
            X = np.column_stack([spread_lag, np.ones(len(spread_lag))])
            betas, _, _, _ = np.linalg.lstsq(X, spread_ret, rcond=None)
            phi = betas[0]

            if phi >= 0:
                return float("inf")  # Not mean-reverting

            half_life = -np.log(2) / phi
            return float(half_life)

        except Exception:
            return float("inf")

    def _check_time_exit(self, pos: PairPosition) -> bool:
        """Check if position has exceeded max holding time."""
        # Use same time limit as directional V2 (2 hours = 120 min)
        max_hold_sec = 120 * 60
        elapsed = (datetime.now() - pos.entry_time).total_seconds()
        return elapsed > max_hold_sec

    def _close_pair_legs(self, pos: PairPosition, executor):
        """Close both legs of a pairs position via the trade executor."""
        try:
            # Build close signals for each leg
            from strategy_v2 import Signal, SignalType

            # Close leg A (opposite of original action)
            close_a_type = SignalType.SHORT if pos.leg_a_side == "LONG" else SignalType.LONG
            close_a = Signal(
                type=close_a_type,
                symbol=pos.leg_a_symbol,
                price=pos.leg_a_entry_price,  # Will get market price on execution
                timestamp=datetime.now(),
                confidence=1.0,
                regime=None,
                strategy="pairs_close",
                indicators={},
                stop_loss=0,
                take_profit=0,
                reason=f"pairs close leg_a: {pos.exit_reason}",
            )

            # Close leg B
            close_b_type = SignalType.SHORT if pos.leg_b_side == "LONG" else SignalType.LONG
            close_b = Signal(
                type=close_b_type,
                symbol=pos.leg_b_symbol,
                price=pos.leg_b_entry_price,
                timestamp=datetime.now(),
                confidence=1.0,
                regime=None,
                strategy="pairs_close",
                indicators={},
                stop_loss=0,
                take_profit=0,
                reason=f"pairs close leg_b: {pos.exit_reason}",
            )

            # Execute close orders
            executor.place_order(close_a, pos.leg_a_size)
            executor.place_order(close_b, pos.leg_b_size)

        except Exception as e:
            logger.error(f"[PAIRS] Failed to close legs: {e}")

    def get_status(self) -> dict:
        """Return current pairs trading status."""
        return {
            "enabled": self.config.enabled,
            "open_positions": len(self.open_positions),
            "total_closed": len(self._closed_positions),
            "cointegrated_pairs": [
                str(pair) for pair, result in self._coint_results.items()
                if result.get("is_cointegrated")
            ],
            "total_pnl": sum(p.pnl for p in self._closed_positions),
        }
