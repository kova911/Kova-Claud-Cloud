"""
Technical Indicators Module
Computes all indicators locally for speed and independence from API limits.
Uses pandas-ta (pure Python) with TA-Lib fallback for performance.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Try to use TA-Lib (faster C implementation), fall back to pandas-ta
try:
    import talib
    USE_TALIB = True
    logger.info("Using TA-Lib (C-based, fast)")
except ImportError:
    USE_TALIB = False
    import pandas_ta as ta
    logger.info("Using pandas-ta (pure Python)")


class Indicators:
    """Compute technical indicators on OHLCV DataFrames."""

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        if USE_TALIB:
            return pd.Series(talib.EMA(series.values, timeperiod=period), index=series.index)
        return ta.ema(series, length=period)

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        if USE_TALIB:
            return pd.Series(talib.SMA(series.values, timeperiod=period), index=series.index)
        return ta.sma(series, length=period)

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        if USE_TALIB:
            return pd.Series(talib.RSI(series.values, timeperiod=period), index=series.index)
        return ta.rsi(series, length=period)

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD: returns DataFrame with macd, signal, histogram columns."""
        if USE_TALIB:
            m, s, h = talib.MACD(series.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return pd.DataFrame({
                "macd": m, "macd_signal": s, "macd_hist": h,
            }, index=series.index)
        result = ta.macd(series, fast=fast, slow=slow, signal=signal)
        if result is not None:
            # Map by column name pattern (order varies across pandas-ta versions)
            col_map = {}
            for c in result.columns:
                if c.startswith("MACDh"):
                    col_map[c] = "macd_hist"
                elif c.startswith("MACDs"):
                    col_map[c] = "macd_signal"
                elif c.startswith("MACD"):
                    col_map[c] = "macd"
            result = result.rename(columns=col_map)
        return result

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        """Bollinger Bands: returns DataFrame with upper, middle, lower."""
        if USE_TALIB:
            upper, middle, lower = talib.BBANDS(
                series.values, timeperiod=period, nbdevup=std, nbdevdn=std,
            )
            return pd.DataFrame({
                "bb_upper": upper, "bb_middle": middle, "bb_lower": lower,
            }, index=series.index)
        result = ta.bbands(series, length=period, std=std)
        if result is not None:
            cols = result.columns
            return pd.DataFrame({
                "bb_lower": result.iloc[:, 0],
                "bb_middle": result.iloc[:, 1],
                "bb_upper": result.iloc[:, 2],
            }, index=series.index)
        return pd.DataFrame()

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range — for dynamic stop-loss sizing."""
        if USE_TALIB:
            return pd.Series(
                talib.ATR(high.values, low.values, close.values, timeperiod=period),
                index=close.index,
            )
        return ta.atr(high, low, close, length=period)

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index — for regime detection."""
        if USE_TALIB:
            return pd.Series(
                talib.ADX(high.values, low.values, close.values, timeperiod=period),
                index=close.index,
            )
        result = ta.adx(high, low, close, length=period)
        if result is not None:
            # pandas-ta returns a DataFrame; ADX is the first column
            return result.iloc[:, 0]
        return pd.Series(dtype=float)

    @staticmethod
    def stochastic(
        high: pd.Series, low: pd.Series, close: pd.Series,
        k_period: int = 5, d_period: int = 3,
    ) -> pd.DataFrame:
        """Stochastic Oscillator: returns %K and %D."""
        if USE_TALIB:
            slowk, slowd = talib.STOCH(
                high.values, low.values, close.values,
                fastk_period=k_period, slowk_period=d_period, slowd_period=d_period,
            )
            return pd.DataFrame({"stoch_k": slowk, "stoch_d": slowd}, index=close.index)
        result = ta.stoch(high, low, close, k=k_period, d=d_period)
        if result is not None:
            # pandas_ta may return 2 cols (%K, %D) or 3 cols (%K, %D, histogram)
            cols = result.columns.tolist()
            if len(cols) >= 2:
                result = result.iloc[:, :2]  # Keep only %K and %D
                result.columns = ["stoch_k", "stoch_d"]
        return result

    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Volume-Weighted Average Price.
        Resets each trading day.
        """
        typical_price = (high + low + close) / 3
        tp_volume = typical_price * volume

        # Group by date to reset VWAP daily
        if hasattr(close.index, 'date'):
            groups = close.index.date
            cumulative_tp_vol = tp_volume.groupby(groups).cumsum()
            cumulative_vol = volume.groupby(groups).cumsum()
        else:
            cumulative_tp_vol = tp_volume.cumsum()
            cumulative_vol = volume.cumsum()

        vwap = cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
        return vwap

    @staticmethod
    def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
        """Simple moving average of volume — for volume confirmation."""
        return volume.rolling(window=period).mean()

    @staticmethod
    def compute_spread(bid: pd.Series, ask: pd.Series) -> pd.Series:
        """Compute bid-ask spread as percentage."""
        mid = (bid + ask) / 2
        return (ask - bid) / mid

    # ─── Composite: Add All Indicators ────────────────────────

    @classmethod
    def add_all(cls, df: pd.DataFrame, config) -> pd.DataFrame:
        """
        Add all configured indicators to the DataFrame.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume
            config: StrategyConfig with indicator parameters

        Returns:
            DataFrame with indicator columns added
        """
        c = config

        # Moving Averages
        df["ema_fast"] = cls.ema(df["close"], c.ema_fast)
        df["ema_slow"] = cls.ema(df["close"], c.ema_slow)
        df["ema_trend"] = cls.ema(df["close"], c.ema_trend)

        # RSI
        df["rsi"] = cls.rsi(df["close"], c.rsi_period)

        # Bollinger Bands
        bb = cls.bollinger_bands(df["close"], c.bb_period, c.bb_std)
        if not bb.empty:
            df = pd.concat([df, bb], axis=1)

        # MACD
        macd_df = cls.macd(df["close"])
        if macd_df is not None and not macd_df.empty:
            df = pd.concat([df, macd_df], axis=1)

        # ATR (for dynamic stop sizing)
        df["atr"] = cls.atr(df["high"], df["low"], df["close"])

        # ADX (regime detection)
        df["adx"] = cls.adx(df["high"], df["low"], df["close"], c.adx_period)

        # VWAP
        if c.use_vwap:
            df["vwap"] = cls.vwap(df["high"], df["low"], df["close"], df["volume"])

        # Volume average
        df["volume_avg"] = cls.volume_sma(df["volume"])

        # Stochastic
        stoch = cls.stochastic(df["high"], df["low"], df["close"])
        if stoch is not None and not stoch.empty:
            df = pd.concat([df, stoch], axis=1)

        # ─── Derived Signals ──────────────────────────────────

        # EMA crossover
        df["ema_cross_up"] = (
            (df["ema_fast"] > df["ema_slow"]) &
            (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
        )
        df["ema_cross_down"] = (
            (df["ema_fast"] < df["ema_slow"]) &
            (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
        )

        # Price vs VWAP
        if "vwap" in df.columns:
            df["above_vwap"] = df["close"] > df["vwap"]

        # Volume confirmation
        df["volume_high"] = df["volume"] > (df["volume_avg"] * c.volume_multiplier)

        # Trend direction (EMA slope)
        df["trend_up"] = df["ema_trend"] > df["ema_trend"].shift(5)
        df["trend_down"] = df["ema_trend"] < df["ema_trend"].shift(5)

        # Regime
        df["regime_trending"] = df["adx"] > c.adx_trending_threshold
        df["regime_ranging"] = df["adx"] < c.adx_ranging_threshold

        return df
