"""
Cross-Asset Correlations — Renaissance-Inspired Decorrelation Signals

Tracks rolling correlations between each stock and the DAX index.
When a stock's correlation diverges significantly from its baseline,
this generates a decorrelation mean-reversion signal component.

Logic: When correlation drops significantly below its norm, the stock
is decoupled from the index — either a stock-specific event or a
temporary anomaly. Combined with the stock's own z-score deviation,
this creates a mean-reversion opportunity signal.

This module provides a score (-1.0 to +1.0) that plugs into the
V2 signal ensemble as the 7th component.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CrossAssetCorrelations:
    """
    Tracks rolling correlation of each stock vs DAX index.

    Signal logic: When correlation drops significantly below its norm,
    the stock is decoupled from the index. Combined with the stock's
    z-score deviation, this amplifies mean-reversion signals.
    """

    def __init__(self, config):
        self.config = config
        self._dax_cache = None
        self._baseline_cache = {}  # symbol -> (baseline_corr, rolling_std)

    def compute_correlation_score(
        self, df: pd.DataFrame, symbol: str, dax_df: pd.DataFrame
    ) -> tuple[float, str]:
        """
        Returns (score, reason) where score is -1.0 to +1.0.

        Positive score: stock is decorrelated and below its mean → bullish MR signal
        Negative score: stock is decorrelated and above its mean → bearish MR signal
        Zero: correlation is normal, no additional signal
        """
        if not self.config.enabled:
            return 0.0, ""

        if df is None or dax_df is None:
            return 0.0, ""

        if len(df) < self.config.min_bars or len(dax_df) < self.config.min_bars:
            return 0.0, ""

        try:
            # Compute returns
            stock_returns = df["close"].pct_change().dropna()
            dax_returns = dax_df["close"].pct_change().dropna()

            # Align indices (use the last N bars from both)
            min_len = min(len(stock_returns), len(dax_returns))
            if min_len < self.config.min_bars:
                return 0.0, ""

            stock_r = stock_returns.iloc[-min_len:].reset_index(drop=True)
            dax_r = dax_returns.iloc[-min_len:].reset_index(drop=True)

            # Rolling correlation
            combined = pd.DataFrame({"stock": stock_r.values, "dax": dax_r.values})
            rolling_corr = combined["stock"].rolling(
                self.config.rolling_window, min_periods=10
            ).corr(combined["dax"])

            # Baseline correlation (longer window)
            baseline_corr = combined["stock"].rolling(
                self.config.baseline_window, min_periods=20
            ).corr(combined["dax"])

            current_rolling = rolling_corr.iloc[-1]
            current_baseline = baseline_corr.iloc[-1]

            if pd.isna(current_rolling) or pd.isna(current_baseline):
                return 0.0, ""

            # Correlation deviation (z-score of correlation)
            corr_std = rolling_corr.rolling(
                self.config.baseline_window, min_periods=20
            ).std().iloc[-1]

            if pd.isna(corr_std) or corr_std <= 0.01:
                return 0.0, ""

            corr_zscore = (current_rolling - current_baseline) / corr_std

            # Only generate signal if decorrelation exceeds threshold
            if abs(corr_zscore) < self.config.decorrelation_threshold:
                return 0.0, ""

            # Direction: use stock's own z-score to determine MR direction
            stock_zscore = self._compute_stock_zscore(df)

            if corr_zscore < -self.config.decorrelation_threshold:
                # Correlation dropped (stock decoupled from index)
                # If stock is also deviated from its own mean → amplify MR signal
                if abs(stock_zscore) > 0.5:
                    # Invert stock zscore for mean-reversion
                    score = -stock_zscore * 0.4
                    score = np.clip(score, -1.0, 1.0)
                    return float(score), (
                        f"decorr z={corr_zscore:.1f}, "
                        f"corr={current_rolling:.2f} vs base={current_baseline:.2f}, "
                        f"stock_z={stock_zscore:.1f}"
                    )

            return 0.0, ""

        except Exception as e:
            logger.debug(f"[XCORR] Correlation computation failed for {symbol}: {e}")
            return 0.0, ""

    def _compute_stock_zscore(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """Compute z-score of current price relative to rolling mean."""
        try:
            close = df["close"]
            if len(close) < lookback:
                return 0.0

            rolling_mean = close.rolling(lookback).mean().iloc[-1]
            rolling_std = close.rolling(lookback).std().iloc[-1]

            if pd.isna(rolling_mean) or pd.isna(rolling_std) or rolling_std <= 0:
                return 0.0

            zscore = (close.iloc[-1] - rolling_mean) / rolling_std
            return float(zscore)

        except Exception:
            return 0.0

    def get_status(self) -> dict:
        """Return current status for logging/monitoring."""
        return {
            "enabled": self.config.enabled,
            "dax_symbol": self.config.dax_index_symbol,
            "rolling_window": self.config.rolling_window,
            "baseline_window": self.config.baseline_window,
        }
