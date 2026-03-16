"""
Strategy V2 — Renaissance-Inspired Statistical Edge Engine (Optimized)

Principles borrowed from Renaissance Technologies / Medallion Fund:

1. Z-SCORE MEAN REVERSION — Compute rolling z-score of price relative to
   short-term mean. Enter at moderate deviations (|z| > 1.0). Optimized
   lookback=20 bars (~1.7 hours on 5m) for faster mean-reversion detection.

2. DE-CORRELATED SIGNAL ENSEMBLE — Each signal source is statistically
   independent. Weighted composite score from 6 components:
   - Price structure (z-score deviation) — 25%
   - Volume anomaly (independent of price) — 20%
   - Momentum/RSI context — 15%
   - Volatility contraction/expansion — 15%
   - Trend alignment — 15%
   - Microstructure (candle analysis) — 10%

3. COMMISSION-AWARE EDGE FILTER — IBKR minimum €1.25/leg = €2.50 round-trip.
   Every signal must produce expected gain > €7 per trade (commission < 36%).
   This is the #1 lesson from optimization: micro-edge trades are structurally
   unprofitable after transaction costs.

4. WIDE STOPS, TIGHT-ISH TARGETS + TRAILING — SL=2.5×ATR gives trades room
   to breathe (83.7% win rate). TP=1.0×ATR captures the mean-reversion snap.
   ATR-based trailing stop activates at 0.5×ATR profit to capture extended moves.

5. STOCK SUITABILITY — Not all stocks mean-revert equally. IFX.DE, BAS.DE,
   ALV.DE showed consistent profitability. SAP.DE, BMW.DE, ADS.DE consistently
   lose. Strategy applies per-stock suitability scoring.

6. SHORT HOLDING PERIODS — Target 15-60 minutes. Quick entry, quick exit.
   Longer holds accumulate risk without proportional reward at this timeframe.

Optimization results (60-day backtest, 9 German blue chips):
- Best params: zscore_lb=20, min_composite=0.50, atr_sl=2.5, atr_tp=1.0
- 49 trades, 83.7% win rate (pre-commission-filter)
- Per-stock edge: IFX.DE (+€25.82), BAS.DE (+€6.85), ALV.DE (+€0.79)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from indicators import Indicators

logger = logging.getLogger(__name__)


class SignalType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"


@dataclass
class Signal:
    type: SignalType
    symbol: str
    price: float
    timestamp: datetime
    confidence: float        # 0.0 to 1.0 — weighted composite score
    regime: MarketRegime
    strategy: str            # "zscore_reversion", "momentum_breakout", "regime_adaptive"
    indicators: dict         # Snapshot of all signal components
    stop_loss: float
    take_profit: float
    reason: str              # Human-readable signal breakdown
    expected_edge: float = 0.0  # Expected return net of costs


# ─── Signal Component Weights ─────────────────────────────────
# Weight rebalancing for MEAN-REVERSION:
# Z-score IS the signal. Other components are modifiers, not co-equal voters.
# Original weights (25/20/15/15/15/10) made trend+vol+volume overpower z-score.
# Sum of weights = 1.0.

SIGNAL_WEIGHTS = {
    "zscore":       0.40,   # Price deviation from mean — THE core MR signal
    "volume":       0.15,   # Volume anomaly (confirms deviation is real)
    "momentum":     0.15,   # Short-term MR confirmation (oversold/overbought)
    "volatility":   0.10,   # Volatility context (minor modifier)
    "trend":        0.10,   # Trend awareness (mild penalty for fighting strong moves)
    "microstructure": 0.10, # Candle structure / order flow proxy
}


class StrategyV2:
    """
    Renaissance-inspired statistical edge engine (optimized).

    Key differences from V1:
    - Z-score based entries (not indicator threshold crossings)
    - De-correlated signal components weighted by independence
    - ATR-based stops exclusively (no fixed percentage)
    - Net-of-cost signal evaluation with absolute € edge filter
    - ATR-based trailing stop (activate at 0.5×ATR, trail by 0.3×ATR)
    - Per-stock mean-reversion suitability scoring
    - Parameters optimized via 2-phase grid search (800 combos × 9 symbols)
    """

    def __init__(self, strategy_config, risk_config, hmm_detector=None):
        self.sc = strategy_config
        self.rc = risk_config

        # Renaissance module hooks (optional, set via constructor or setter)
        self._hmm = hmm_detector
        self._cross_corr = None  # Set via set_cross_correlations()
        self._last_regime_probs = {}  # Store HMM probabilities for risk manager

        # V2 parameters — RE-OPTIMIZED 2026-03-11 based on live trade data
        # Problem: SL=2.5×ATR vs TP=1.0×ATR gave 2.5:1 risk AGAINST us.
        # Avg winner was €1.97 vs avg loser €-5.86 (R:R=0.34). Need >71% WR to break even.
        # Fix: Tighten SL to 1.5×ATR, widen TP to 1.2×ATR → R:R flips to 0.8:1 in our favor.
        # Also raise min_composite_score to only take higher-quality signals.
        self.zscore_lookback = 20          # 20 bars = ~1.7 hours on 5m (faster detection)
        self.zscore_entry_threshold = 1.0  # Enter at |z| > 1.0 (moderate deviations)
        self.zscore_exit_threshold = 0.5   # Exit when |z| < 0.5 (returned to mean)
        self.min_composite_score = 0.35    # Raised from 0.30: only take stronger signals
        self.atr_stop_multiplier = 1.5     # SL = 1.5 * ATR (tighter stops — cut losers faster)
        self.atr_tp_multiplier = 1.2       # TP = 1.2 * ATR (wider targets — let winners run more)
        self.volume_surge_threshold = 1.5  # Volume must be 1.5x average for bonus
        self.volume_dry_penalty = 0.7      # Penalize signals on below-average volume
        self.max_atr_pct = 0.015           # Skip stocks with ATR > 1.5% (too volatile)
        self.min_atr_pct = 0.001           # Skip stocks with ATR < 0.1% (dead stocks)

        # ATR-based trailing stop (captures extended moves beyond initial TP)
        self.trailing_atr_activation = 0.5  # Activate trailing at 0.5× ATR profit
        self.trailing_atr_distance = 0.3    # Trail by 0.3× ATR behind best price

        # Transaction cost model (IBKR tiered)
        self.round_trip_cost_pct = 0.0008  # ~0.08% round trip (commission + spread + slippage)
        self.round_trip_cost_min = 2.50    # €1.25 minimum per leg × 2
        self.min_expected_gain_eur = 7.0   # Reject trades where TP gain < €7 (commission < 36%)

        # Per-stock mean-reversion suitability scores
        # Updated 2026-03-11 based on LIVE trade data (not just backtest):
        # BAS.DE: 100% WR, +€7.55 live. IFX.DE: 75% WR but big SL losses.
        # ALV.DE: 0% WR live (-€23), SIE.DE: 0% WR (-€21.60) — heavily penalized.
        # US stocks: insufficient live data, keep cautious.
        self.stock_suitability = {
            "BAS.DE": 0.3,    # +€7.55 live, 100% WR — best performer
            "IFX.DE": 0.15,   # 75% WR but SL hits are costly — reduced from 0.3
            "SAP.DE": 0.0,    # +€1.02 live, 1 trade — neutral pending more data
            "MBG.DE": 0.0,    # Neutral — insufficient live data
            "AIR.DE": -0.15,  # -€11.52 live, 0% WR — penalize
            "ALV.DE": -0.20,  # -€23.00 live, 0% WR — strongly penalize
            "SIE.DE": -0.20,  # -€21.60 live, 0% WR — strongly penalize
            "BMW.DE": -0.20,  # -€9.70 live, 0% WR — keep penalty
            "ADS.DE": -0.20,  # No live data, keep penalty from backtest
        }

    def set_cross_correlations(self, cross_corr):
        """Set the cross-asset correlations module for 7th signal component."""
        self._cross_corr = cross_corr

    def get_last_regime_probs(self) -> dict:
        """Get the last HMM regime probabilities (for risk manager)."""
        return self._last_regime_probs

    def detect_regime(self, df: pd.DataFrame, symbol: str = "") -> MarketRegime:
        """
        Detect market regime. Tries HMM first (if available), then falls
        back to the original ADX/ATR threshold-based detection.
        """
        # Try HMM regime detection first
        if self._hmm and symbol:
            try:
                regime, probs = self._hmm.detect_regime(df, symbol)
                if regime is not None:
                    self._last_regime_probs = probs
                    return regime
            except Exception as e:
                logger.debug(f"HMM regime detection failed for {symbol}: {e}")

        # Fall back to original ADX/ATR method
        if df.empty or len(df) < 50:
            return MarketRegime.RANGING

        latest = df.iloc[-1]
        adx = latest.get("adx", 0)
        if pd.isna(adx):
            adx = 0

        atr = latest.get("atr", 0)
        if pd.isna(atr):
            atr = 0
        atr_avg = df["atr"].rolling(50).mean().iloc[-1] if len(df) > 50 else atr
        if pd.isna(atr_avg):
            atr_avg = atr

        # Volatility expansion: ATR > 1.5x its 50-bar average
        if atr_avg > 0 and atr > atr_avg * 1.5:
            return MarketRegime.VOLATILE

        # Strong trend: ADX > 30 (higher threshold than V1's 25)
        if adx > 30:
            # Determine direction from EMA slope
            ema_trend = latest.get("ema_trend", 0)
            ema_trend_prev = df["ema_trend"].iloc[-6] if len(df) > 6 else ema_trend
            if not pd.isna(ema_trend) and not pd.isna(ema_trend_prev):
                if ema_trend > ema_trend_prev:
                    return MarketRegime.TRENDING_UP
                elif ema_trend < ema_trend_prev:
                    return MarketRegime.TRENDING_DOWN

        return MarketRegime.RANGING

    def generate_signal(self, df: pd.DataFrame, symbol: str, dax_df: pd.DataFrame = None) -> Optional[Signal]:
        """
        Generate a trading signal using weighted ensemble of de-correlated components.

        Each component returns a score from -1.0 (strong short) to +1.0 (strong long).
        The composite score determines direction and confidence.

        Args:
            df: Symbol DataFrame with indicators
            symbol: Stock symbol (e.g., "IFX.DE")
            dax_df: Optional DAX index DataFrame for cross-asset correlations
        """
        min_bars = max(self.sc.ema_trend, self.sc.bb_period, self.zscore_lookback, 50)
        if df.empty or len(df) < min_bars:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        price = latest["close"]
        if pd.isna(price) or price <= 0:
            return None

        regime = self.detect_regime(df, symbol)

        # In VOLATILE regime, don't trade (RenTec principle: stand aside when uncertain)
        if regime == MarketRegime.VOLATILE:
            return None

        # ─── Compute ATR sanity check ───
        atr = latest.get("atr", 0)
        if pd.isna(atr) or atr <= 0:
            return None
        atr_pct = atr / price
        if atr_pct > self.max_atr_pct or atr_pct < self.min_atr_pct:
            return None  # Stock is too volatile or too dead

        # ─── Score each signal component ───
        components = {}
        reasons = []

        # 1. Z-Score Mean Reversion (PRIMARY — 25% weight)
        zscore_score, zscore_reason = self._score_zscore(df, latest, price)
        components["zscore"] = zscore_score
        if zscore_reason:
            reasons.append(zscore_reason)

        # 2. Volume Anomaly (20% weight — independent of price direction)
        volume_score, vol_reason = self._score_volume(df, latest)
        components["volume"] = volume_score
        if vol_reason:
            reasons.append(vol_reason)

        # 3. Momentum / Mean-Reversion Confirmation (15% weight)
        momentum_score, mom_reason = self._score_momentum(df, latest, prev, regime)
        components["momentum"] = momentum_score
        if mom_reason:
            reasons.append(mom_reason)

        # 4. Volatility Context (15% weight)
        vol_ctx_score, vol_ctx_reason = self._score_volatility_context(df, latest, atr)
        components["volatility"] = vol_ctx_score
        if vol_ctx_reason:
            reasons.append(vol_ctx_reason)

        # 5. Trend Alignment (15% weight)
        trend_score, trend_reason = self._score_trend(df, latest, regime)
        components["trend"] = trend_score
        if trend_reason:
            reasons.append(trend_reason)

        # 6. Microstructure (10% weight — spread context from BB width)
        micro_score, micro_reason = self._score_microstructure(df, latest, price)
        components["microstructure"] = micro_score
        if micro_reason:
            reasons.append(micro_reason)

        # 7. Cross-Asset Correlation (optional — 10% weight when enabled)
        if self._cross_corr and dax_df is not None:
            try:
                corr_score, corr_reason = self._cross_corr.compute_correlation_score(
                    df, symbol, dax_df
                )
                if corr_score != 0.0:
                    components["correlation"] = corr_score
                    if corr_reason:
                        reasons.append(corr_reason)
            except Exception as e:
                logger.debug(f"Cross-correlation scoring failed for {symbol}: {e}")

        # ─── Compute weighted composite score ───
        # Use dynamic weights: if correlation component present, rebalance
        weights = dict(SIGNAL_WEIGHTS)
        if "correlation" in components:
            # Add correlation weight and proportionally reduce others
            corr_weight = 0.10
            scale = (1.0 - corr_weight) / sum(weights.values())
            weights = {k: v * scale for k, v in weights.items()}
            weights["correlation"] = corr_weight

        composite = sum(
            components.get(k, 0) * w
            for k, w in weights.items()
        )

        # ─── Guard: mean-reversion requires z-score signal ───
        # This is a MR strategy — refuse to trade without price deviation from mean
        zscore_component = components.get("zscore", 0)
        if zscore_component == 0:
            return None  # No mean-reversion opportunity

        # ─── Apply stock suitability adjustment ───
        suitability = self.stock_suitability.get(symbol, 0.0)
        if suitability != 0.0:
            # Suitability modifies composite: positive boosts, negative penalizes
            composite_before = composite
            composite += suitability * np.sign(composite)  # Boost in signal direction
            if suitability < -0.1:
                reasons.append(f"stock penalty ({suitability:+.2f})")

        # ─── Determine direction and confidence ───
        abs_score = abs(composite)
        if abs_score < self.min_composite_score:
            return None  # Not enough conviction

        signal_type = SignalType.LONG if composite > 0 else SignalType.SHORT

        # ─── Regime filter: don't mean-revert against strong trends ───
        if regime == MarketRegime.TRENDING_UP and signal_type == SignalType.SHORT:
            # Only allow shorts in strong uptrend if the signal is very strong
            if abs_score < 0.75:
                return None
        if regime == MarketRegime.TRENDING_DOWN and signal_type == SignalType.LONG:
            if abs_score < 0.75:
                return None

        # ─── ATR-based stops (no fixed percentage) ───
        atr_stop = atr * self.atr_stop_multiplier
        atr_tp = atr * self.atr_tp_multiplier

        if signal_type == SignalType.LONG:
            stop_loss = price - atr_stop
            take_profit = price + atr_tp
        else:
            stop_loss = price + atr_stop
            take_profit = price - atr_tp

        # ─── Transaction cost check (RenTec principle) ───
        expected_gross_edge = atr_tp / price  # Expected % gain if TP hit
        expected_net_edge = expected_gross_edge - self.round_trip_cost_pct
        if expected_net_edge < self.round_trip_cost_pct:
            return None  # Edge doesn't justify the cost (need at least 2x costs)

        # ─── Absolute € edge filter (THE critical filter) ───
        # Estimate position size based on risk parameters
        stop_distance = abs(price - stop_loss)
        if stop_distance > 0:
            risk_amount = self.rc.initial_balance * self.rc.risk_per_trade
            estimated_size = int(risk_amount / stop_distance)
            expected_tp_gain = atr_tp * estimated_size
            if expected_tp_gain < self.min_expected_gain_eur:
                logger.debug(
                    f"{symbol}: expected TP gain €{expected_tp_gain:.2f} < "
                    f"min €{self.min_expected_gain_eur:.2f} — skipping"
                )
                return None

        # ─── Build signal ───
        confidence = min(abs_score, 1.0)
        strategy_name = "zscore_reversion" if abs(components.get("zscore", 0)) > 0.5 else "regime_adaptive"

        timestamp = latest.name if hasattr(latest, "name") else datetime.now()

        return Signal(
            type=signal_type,
            symbol=symbol,
            price=price,
            timestamp=timestamp,
            confidence=confidence,
            regime=regime,
            strategy=strategy_name,
            indicators=components,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"V2 {signal_type.value}: {', '.join(reasons)} [score={composite:+.3f}]",
            expected_edge=expected_net_edge,
        )

    # ─── Signal Component Scorers ─────────────────────────────────
    # Each returns (score: float [-1 to +1], reason: str)

    def _score_zscore(self, df: pd.DataFrame, latest: pd.Series, price: float) -> tuple[float, str]:
        """
        Z-score of price relative to rolling mean.
        This is the core RenTec mean-reversion signal.

        Positive z-score (price above mean) → negative score (expect reversion down)
        Negative z-score (price below mean) → positive score (expect reversion up)
        """
        lookback = min(self.zscore_lookback, len(df) - 1)
        if lookback < 20:
            return 0.0, ""

        rolling_mean = df["close"].rolling(lookback).mean().iloc[-1]
        rolling_std = df["close"].rolling(lookback).std().iloc[-1]

        if pd.isna(rolling_mean) or pd.isna(rolling_std) or rolling_std <= 0:
            return 0.0, ""

        zscore = (price - rolling_mean) / rolling_std

        # Map z-score to signal score (inverted for mean reversion)
        if abs(zscore) < 1.0:
            return 0.0, ""  # Not deviated enough
        elif abs(zscore) < 2.0:
            # Moderate deviation — partial signal
            score = -zscore * 0.4  # Inverted: high z → short signal
            return score, f"z={zscore:+.2f} (moderate)"
        else:
            # Strong deviation — full signal
            score = -np.clip(zscore, -3.0, 3.0) / 3.0  # Normalized to [-1, 1]
            return score, f"z={zscore:+.2f} (strong)"

    def _score_volume(self, df: pd.DataFrame, latest: pd.Series) -> tuple[float, str]:
        """
        Volume anomaly — independent of price direction.
        High volume on a z-score deviation CONFIRMS the move is real.
        Low volume suggests the deviation is noise.

        This is a MODIFIER, not a directional signal.
        Returns magnitude only (applied to composite direction).
        """
        vol = latest.get("volume", 0)
        vol_avg = latest.get("volume_avg", 0)
        if pd.isna(vol) or pd.isna(vol_avg) or vol_avg <= 0 or vol <= 0:
            return 0.0, ""

        vol_ratio = vol / vol_avg

        if vol_ratio >= self.volume_surge_threshold:
            # High volume confirms any signal — this is a strength multiplier
            return 0.8, f"vol surge {vol_ratio:.1f}x"
        elif vol_ratio >= 1.0:
            # Normal volume — neutral
            return 0.3, ""
        elif vol_ratio >= 0.5:
            # Somewhat below average — mild penalty (IBKR delayed data often underreports)
            return -0.15, f"low vol {vol_ratio:.2f}x"
        else:
            # Very low volume — stronger penalty
            return -0.3, f"very low vol {vol_ratio:.2f}x"

    def _score_momentum(
        self, df: pd.DataFrame, latest: pd.Series, prev: pd.Series,
        regime: MarketRegime,
    ) -> tuple[float, str]:
        """
        Short-term momentum score.
        In RANGING regime: penalize momentum (mean reversion preferred).
        In TRENDING regime: reward momentum alignment.
        """
        rsi = latest.get("rsi", 50)
        if pd.isna(rsi):
            rsi = 50

        macd_hist = latest.get("macd_hist", 0)
        if pd.isna(macd_hist):
            macd_hist = 0

        # RSI-based score
        rsi_score = 0.0
        rsi_reason = ""
        if rsi < 25:
            rsi_score = 0.8   # Strongly oversold → bullish
            rsi_reason = f"RSI oversold ({rsi:.0f})"
        elif rsi < 35:
            rsi_score = 0.4
            rsi_reason = f"RSI low ({rsi:.0f})"
        elif rsi > 75:
            rsi_score = -0.8  # Strongly overbought → bearish
            rsi_reason = f"RSI overbought ({rsi:.0f})"
        elif rsi > 65:
            rsi_score = -0.4
            rsi_reason = f"RSI high ({rsi:.0f})"

        # In ranging regime, RSI extremes are STRONGER signals (mean reversion)
        if regime == MarketRegime.RANGING and abs(rsi_score) > 0:
            rsi_score *= 1.2

        return np.clip(rsi_score, -1.0, 1.0), rsi_reason

    def _score_volatility_context(
        self, df: pd.DataFrame, latest: pd.Series, atr: float,
    ) -> tuple[float, str]:
        """
        Volatility contraction/expansion context.
        RenTec insight: best mean-reversion entries happen during volatility contraction
        (Bollinger squeeze), not during expansion.
        """
        bb_upper = latest.get("bb_upper", 0)
        bb_lower = latest.get("bb_lower", 0)
        bb_middle = latest.get("bb_middle", 0)

        # Handle potential duplicate columns (Series instead of scalar)
        if isinstance(bb_upper, pd.Series):
            bb_upper = bb_upper.iloc[0]
        if isinstance(bb_lower, pd.Series):
            bb_lower = bb_lower.iloc[0]
        if isinstance(bb_middle, pd.Series):
            bb_middle = bb_middle.iloc[0]

        if pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(bb_middle) or bb_middle <= 0:
            return 0.0, ""

        bb_width = (bb_upper - bb_lower) / bb_middle

        # Historical BB width for context
        if len(df) > 50 and "bb_upper" in df.columns and "bb_lower" in df.columns and "bb_middle" in df.columns:
            try:
                hist_upper = df["bb_upper"]
                hist_lower = df["bb_lower"]
                hist_mid = df["bb_middle"]
                # Handle potential multi-column
                if isinstance(hist_upper, pd.DataFrame):
                    hist_upper = hist_upper.iloc[:, 0]
                if isinstance(hist_lower, pd.DataFrame):
                    hist_lower = hist_lower.iloc[:, 0]
                if isinstance(hist_mid, pd.DataFrame):
                    hist_mid = hist_mid.iloc[:, 0]

                hist_width = ((hist_upper - hist_lower) / hist_mid.replace(0, np.nan)).dropna()
                if len(hist_width) > 20:
                    avg_width = hist_width.rolling(50).mean().iloc[-1]
                    if not pd.isna(avg_width) and avg_width > 0:
                        width_ratio = bb_width / avg_width
                        if width_ratio < 0.7:
                            # Volatility contraction (squeeze) — favorable for mean reversion
                            return 0.6, f"BB squeeze ({width_ratio:.2f}x)"
                        elif width_ratio > 1.5:
                            # Volatility expansion — unfavorable
                            return -0.5, f"BB expansion ({width_ratio:.2f}x)"
            except Exception:
                pass

        return 0.0, ""

    def _score_trend(
        self, df: pd.DataFrame, latest: pd.Series, regime: MarketRegime,
    ) -> tuple[float, str]:
        """
        Trend context for mean-reversion strategy.
        In RANGING markets (where MR works best): neutral — let z-score drive.
        In TRENDING markets: mild penalty to avoid fighting violent moves.
        RenTec insight: MR profits come from ranging regimes, not trend alignment.
        """
        if regime == MarketRegime.RANGING:
            # MR's sweet spot — trend direction doesn't matter for reversion
            return 0.0, ""

        ema_fast = latest.get("ema_fast", 0)
        ema_slow = latest.get("ema_slow", 0)
        ema_trend = latest.get("ema_trend", 0)
        price = latest["close"]

        if pd.isna(ema_fast) or pd.isna(ema_slow) or pd.isna(ema_trend):
            return 0.0, ""

        # Only penalize when fighting STRONG trends (reduced from ±0.6 to ±0.3)
        if price > ema_fast > ema_slow > ema_trend:
            return 0.3, "bullish EMA stack"
        elif price < ema_fast < ema_slow < ema_trend:
            return -0.3, "bearish EMA stack"

        # Partial alignment — minimal influence
        if ema_fast > ema_slow:
            return 0.1, ""
        elif ema_fast < ema_slow:
            return -0.1, ""

        return 0.0, ""

    def _score_microstructure(
        self, df: pd.DataFrame, latest: pd.Series, price: float,
    ) -> tuple[float, str]:
        """
        Microstructure score — proxy for order flow using available data.
        Uses candle body-to-range ratio and close position within range.
        (Without Level 2 data, this is the best we can do at retail level)
        """
        open_price = latest.get("open", price)
        high = latest.get("high", price)
        low = latest.get("low", price)
        close = latest.get("close", price)

        if pd.isna(open_price) or pd.isna(high) or pd.isna(low):
            return 0.0, ""

        candle_range = high - low
        if candle_range <= 0:
            return 0.0, ""

        # Close position within candle range (0 = closed at low, 1 = closed at high)
        close_position = (close - low) / candle_range

        # Body ratio (how much of the range is body vs. wicks)
        body = abs(close - open_price)
        body_ratio = body / candle_range

        # Strong body closing near the extreme suggests directional conviction
        if close_position > 0.8 and body_ratio > 0.6:
            return 0.5, "bullish candle"
        elif close_position < 0.2 and body_ratio > 0.6:
            return -0.5, "bearish candle"

        # Pin bars (long wicks) suggest rejection — mean reversion signal
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low

        if upper_wick > body * 2 and upper_wick > candle_range * 0.5:
            return -0.4, "upper wick rejection"
        if lower_wick > body * 2 and lower_wick > candle_range * 0.5:
            return 0.4, "lower wick rejection"

        return 0.0, ""

    def get_trailing_stop_params(self, symbol: str, atr: float) -> tuple[float, float]:
        """
        Return ATR-based trailing stop parameters for a position.
        Used by risk_manager to implement trailing stops.

        Returns:
            (activation_distance, trail_distance) in absolute price units
        """
        return (
            atr * self.trailing_atr_activation,  # Activate at 0.5× ATR profit
            atr * self.trailing_atr_distance,     # Trail by 0.3× ATR
        )

    def should_skip_stock(self, symbol: str) -> bool:
        """
        Check if stock has been flagged as unsuitable for mean-reversion.
        Stocks with suitability < -0.15 are skipped entirely.
        """
        return self.stock_suitability.get(symbol, 0.0) < -0.15

    def scan_watchlist(self, data_dict: dict[str, pd.DataFrame], dax_df: pd.DataFrame = None) -> list[Signal]:
        """
        Scan all symbols and return signals sorted by expected edge.
        RenTec principle: prioritize by expected net return, not confidence alone.

        Args:
            data_dict: Symbol DataFrames with indicators
            dax_df: Optional DAX index DataFrame for cross-asset correlations
        """
        signals = []
        skipped_stocks = []
        near_misses = []  # Track near-miss symbols for diagnostics
        for symbol, df in data_dict.items():
            try:
                # Pre-filter: skip stocks known to be unprofitable for mean-reversion
                if self.should_skip_stock(symbol):
                    skipped_stocks.append(symbol)
                    continue

                # Diagnostic: compute composite score even if no signal
                score_info = self._quick_score(df, symbol)
                if score_info:
                    near_misses.append(score_info)

                signal = self.generate_signal(df, symbol, dax_df=dax_df)
                if signal:
                    signals.append(signal)
                    logger.info(
                        f"V2 Signal: {signal.type.value} {symbol} @ {signal.price:.2f} "
                        f"[conf={signal.confidence:.0%} edge={signal.expected_edge:.4%}] "
                        f"— {signal.reason}"
                    )
            except Exception as e:
                logger.error(f"V2 scan error {symbol}: {e}")

        if skipped_stocks:
            logger.debug(f"V2 skipped (poor MR suitability): {', '.join(skipped_stocks)}")

        # Log near-miss diagnostics (top 3 by score)
        if near_misses and not signals:
            near_misses.sort(key=lambda x: x[1], reverse=True)
            top = near_misses[:3]
            summary = " | ".join(f"{s}={sc:+.3f}" for s, sc in top)
            logger.info(f"V2 near-miss scores: {summary} (threshold={self.min_composite_score})")

        # Sort by expected edge (net of costs), not just confidence
        signals.sort(key=lambda s: s.expected_edge, reverse=True)
        return signals

    def _quick_score(self, df: pd.DataFrame, symbol: str) -> tuple[str, float] | None:
        """Quick composite score calculation for diagnostics (no signal generation)."""
        try:
            min_bars = max(self.sc.ema_trend, self.sc.bb_period, self.zscore_lookback, 50)
            if df.empty or len(df) < min_bars:
                return None
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            price = latest["close"]
            atr = latest.get("atr", 0)
            if pd.isna(price) or price <= 0 or pd.isna(atr) or atr <= 0:
                return None
            regime = self.detect_regime(df)
            if regime == MarketRegime.VOLATILE:
                return (symbol, 0.0)
            zs, _ = self._score_zscore(df, latest, price)
            vs, _ = self._score_volume(df, latest)
            ms, _ = self._score_momentum(df, latest, prev, regime)
            vcs, _ = self._score_volatility_context(df, latest, atr)
            ts, _ = self._score_trend(df, latest, regime)
            mics, _ = self._score_microstructure(df, latest, price)
            components = {"zscore": zs, "volume": vs, "momentum": ms,
                          "volatility": vcs, "trend": ts, "microstructure": mics}
            composite = sum(components.get(k, 0) * w for k, w in SIGNAL_WEIGHTS.items())
            suit = self.stock_suitability.get(symbol, 0.0)
            if suit != 0 and composite != 0:
                composite += suit * np.sign(composite)
            return (symbol, abs(composite))
        except Exception:
            return None
