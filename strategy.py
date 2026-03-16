"""
Scalping Strategy Engine
Generates buy/sell signals using multi-indicator confluence.
Supports mean-reversion and momentum modes with adaptive regime switching.
"""

import logging
from dataclasses import dataclass
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
    confidence: float        # 0.0 to 1.0
    regime: MarketRegime
    strategy: str            # "mean_reversion" or "momentum"
    indicators: dict         # Snapshot of indicator values
    stop_loss: float
    take_profit: float
    reason: str              # Human-readable reason


class ScalpingStrategy:
    """
    Multi-strategy scalping engine with adaptive regime detection.

    Mean-Reversion: trades when price deviates from mean (Bollinger + RSI + VWAP)
    Momentum: trades EMA crossovers with volume confirmation
    Regime filter selects the appropriate strategy automatically.
    """

    def __init__(self, strategy_config, risk_config):
        self.sc = strategy_config
        self.rc = risk_config

    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime from indicators."""
        if df.empty or len(df) < 20:
            return MarketRegime.RANGING

        latest = df.iloc[-1]
        adx = latest.get("adx", 0)
        atr = latest.get("atr", 0)
        atr_avg = df["atr"].rolling(50).mean().iloc[-1] if len(df) > 50 else atr

        # High volatility: ATR is 1.5x its average
        if atr > atr_avg * 1.5:
            return MarketRegime.VOLATILE

        # Trending (requires clear direction from EMA slope)
        if adx > self.sc.adx_trending_threshold:
            if latest.get("trend_up", False):
                return MarketRegime.TRENDING_UP
            elif latest.get("trend_down", False):
                return MarketRegime.TRENDING_DOWN
            # ADX high but no clear direction — treat as ranging to avoid directional bias
            return MarketRegime.RANGING

        # Ranging
        return MarketRegime.RANGING

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> Optional[Signal]:
        """
        Generate a trading signal from the latest bar.

        Args:
            df: OHLCV DataFrame with all indicators computed
            symbol: Ticker symbol

        Returns:
            Signal object or None if no signal
        """
        if df.empty or len(df) < self.sc.ema_trend:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        regime = self.detect_regime(df)

        # Select strategy based on regime
        if self.sc.adaptive_regime:
            if regime == MarketRegime.VOLATILE:
                return None  # Don't trade in high volatility
            elif regime == MarketRegime.RANGING:
                return self._mean_reversion_signal(df, latest, prev, symbol, regime)
            else:
                return self._momentum_signal(df, latest, prev, symbol, regime)
        else:
            if self.sc.primary_strategy == "mean_reversion":
                return self._mean_reversion_signal(df, latest, prev, symbol, regime)
            else:
                return self._momentum_signal(df, latest, prev, symbol, regime)

    def _mean_reversion_signal(
        self, df: pd.DataFrame, latest: pd.Series, prev: pd.Series,
        symbol: str, regime: MarketRegime,
    ) -> Optional[Signal]:
        """
        Mean-reversion: price has deviated from mean → expect return.
        LONG when: price < lower BB, RSI oversold, below VWAP, volume confirms.
        SHORT when: price > upper BB, RSI overbought, above VWAP, volume confirms.
        """
        price = latest["close"]
        signals_long = 0
        signals_short = 0
        reasons_long = []
        reasons_short = []

        # 1. Bollinger Band touch
        if "bb_lower" in latest.index and not pd.isna(latest["bb_lower"]):
            if price <= latest["bb_lower"]:
                signals_long += 1
                reasons_long.append("price at lower BB")
            if price >= latest["bb_upper"]:
                signals_short += 1
                reasons_short.append("price at upper BB")

        # 2. RSI condition
        rsi = latest.get("rsi", 50)
        if not pd.isna(rsi):
            if rsi < self.sc.rsi_oversold:
                signals_long += 1
                reasons_long.append(f"RSI oversold ({rsi:.0f})")
            if rsi > self.sc.rsi_overbought:
                signals_short += 1
                reasons_short.append(f"RSI overbought ({rsi:.0f})")

        # 3. VWAP position
        if "vwap" in latest.index and not pd.isna(latest.get("vwap")):
            if price < latest["vwap"]:
                signals_long += 1
                reasons_long.append("below VWAP")
            if price > latest["vwap"]:
                signals_short += 1
                reasons_short.append("above VWAP")

        # 4. Volume confirmation
        if latest.get("volume_high", False):
            signals_long += 1
            signals_short += 1
            reasons_long.append("volume confirmed")
            reasons_short.append("volume confirmed")

        # 5. Stochastic oversold/overbought
        stoch_k = latest.get("stoch_k", 50)
        if not pd.isna(stoch_k):
            if stoch_k < 20:
                signals_long += 1
                reasons_long.append(f"Stoch oversold ({stoch_k:.0f})")
            if stoch_k > 80:
                signals_short += 1
                reasons_short.append(f"Stoch overbought ({stoch_k:.0f})")

        # Check confluence
        if signals_long >= self.sc.min_signal_confluence:
            return self._create_signal(
                SignalType.LONG, symbol, price, latest,
                confidence=min(signals_long / 5.0, 1.0),
                regime=regime, strategy="mean_reversion",
                reason=f"MR LONG: {', '.join(reasons_long)}",
            )

        if signals_short >= self.sc.min_signal_confluence:
            return self._create_signal(
                SignalType.SHORT, symbol, price, latest,
                confidence=min(signals_short / 5.0, 1.0),
                regime=regime, strategy="mean_reversion",
                reason=f"MR SHORT: {', '.join(reasons_short)}",
            )

        return None

    def _momentum_signal(
        self, df: pd.DataFrame, latest: pd.Series, prev: pd.Series,
        symbol: str, regime: MarketRegime,
    ) -> Optional[Signal]:
        """
        Momentum: trade in the direction of the trend on crossovers.
        LONG when: EMA cross up + above VWAP + RSI in range + volume confirms.
        SHORT when: EMA cross down + below VWAP + RSI in range + volume confirms.
        """
        price = latest["close"]
        signals_long = 0
        signals_short = 0
        reasons_long = []
        reasons_short = []

        # 1. EMA crossover (must have just happened)
        if latest.get("ema_cross_up", False):
            signals_long += 1
            reasons_long.append("EMA cross up")
        if latest.get("ema_cross_down", False):
            signals_short += 1
            reasons_short.append("EMA cross down")

        # 2. Price vs VWAP (trend confirmation)
        if latest.get("above_vwap", False):
            signals_long += 1
            reasons_long.append("above VWAP")
        elif "above_vwap" in latest.index:
            signals_short += 1
            reasons_short.append("below VWAP")

        # 3. RSI in momentum zone (not exhausted, non-overlapping ranges)
        rsi = latest.get("rsi", 50)
        if not pd.isna(rsi):
            if 45 < rsi < 70:
                signals_long += 1
                reasons_long.append(f"RSI in momentum zone ({rsi:.0f})")
            if 30 < rsi < 55:
                signals_short += 1
                reasons_short.append(f"RSI in momentum zone ({rsi:.0f})")

        # 4. Volume surge
        if latest.get("volume_high", False):
            signals_long += 1
            signals_short += 1
            reasons_long.append("volume surge")
            reasons_short.append("volume surge")

        # 5. MACD histogram positive/negative
        macd_hist = latest.get("macd_hist", 0)
        if not pd.isna(macd_hist):
            if macd_hist > 0:
                signals_long += 1
                reasons_long.append("MACD positive")
            if macd_hist < 0:
                signals_short += 1
                reasons_short.append("MACD negative")

        # 6. Trend filter (trade in trend direction only)
        if regime == MarketRegime.TRENDING_UP and signals_long >= self.sc.min_signal_confluence:
            return self._create_signal(
                SignalType.LONG, symbol, price, latest,
                confidence=min(signals_long / 5.0, 1.0),
                regime=regime, strategy="momentum",
                reason=f"MOM LONG: {', '.join(reasons_long)}",
            )

        if regime == MarketRegime.TRENDING_DOWN and signals_short >= self.sc.min_signal_confluence:
            return self._create_signal(
                SignalType.SHORT, symbol, price, latest,
                confidence=min(signals_short / 5.0, 1.0),
                regime=regime, strategy="momentum",
                reason=f"MOM SHORT: {', '.join(reasons_short)}",
            )

        return None

    def _create_signal(
        self, signal_type: SignalType, symbol: str, price: float,
        latest: pd.Series, confidence: float, regime: MarketRegime,
        strategy: str, reason: str,
    ) -> Signal:
        """Create a Signal object with stop-loss and take-profit."""
        if signal_type == SignalType.LONG:
            stop_loss = price * (1 - self.rc.stop_loss_pct)
            take_profit = price * (1 + self.rc.take_profit_pct)
        else:
            stop_loss = price * (1 + self.rc.stop_loss_pct)
            take_profit = price * (1 - self.rc.take_profit_pct)

        # Optionally adjust stop using ATR
        atr = latest.get("atr", 0)
        if atr > 0 and not pd.isna(atr):
            atr_stop = atr * 1.5
            if signal_type == SignalType.LONG:
                atr_stop_price = price - atr_stop
                # Use the tighter of the two stops
                stop_loss = max(stop_loss, atr_stop_price)
            else:
                atr_stop_price = price + atr_stop
                stop_loss = min(stop_loss, atr_stop_price)

        indicators_snapshot = {
            "ema_fast": latest.get("ema_fast"),
            "ema_slow": latest.get("ema_slow"),
            "rsi": latest.get("rsi"),
            "adx": latest.get("adx"),
            "bb_upper": latest.get("bb_upper"),
            "bb_lower": latest.get("bb_lower"),
            "vwap": latest.get("vwap"),
            "macd_hist": latest.get("macd_hist"),
            "atr": atr,
            "volume": latest.get("volume"),
            "volume_avg": latest.get("volume_avg"),
        }

        timestamp = latest.name if hasattr(latest, "name") else datetime.now()

        return Signal(
            type=signal_type,
            symbol=symbol,
            price=price,
            timestamp=timestamp,
            confidence=confidence,
            regime=regime,
            strategy=strategy,
            indicators=indicators_snapshot,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
        )

    def scan_watchlist(self, data_dict: dict[str, pd.DataFrame]) -> list[Signal]:
        """
        Scan all symbols in watchlist and return any signals.

        Args:
            data_dict: {symbol: DataFrame_with_indicators}

        Returns:
            List of Signal objects, sorted by confidence (highest first)
        """
        signals = []
        for symbol, df in data_dict.items():
            try:
                signal = self.generate_signal(df, symbol)
                if signal:
                    signals.append(signal)
                    logger.info(f"Signal: {signal.type.value} {symbol} @ {signal.price:.2f} "
                              f"[{signal.confidence:.0%}] — {signal.reason}")
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

        # Sort by confidence descending
        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals
