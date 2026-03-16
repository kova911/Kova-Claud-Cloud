"""
HMM Regime Detection — Renaissance-Inspired Probabilistic Market Regimes

Uses a 3-state Gaussian Hidden Markov Model to identify market regimes
probabilistically, replacing the simple ADX/ATR threshold approach.

States (learned, but automatically labeled by volatility):
  State 0: Low-volatility / ranging — best for mean-reversion
  State 1: Normal / trending — standard trading
  State 2: High-volatility / crisis — reduce or stand aside

Observable features: returns, realized_vol, volume_ratio, spread_proxy

Graceful degradation: Returns (None, {}) if insufficient data or fitting
fails, allowing the caller to fall back to the original ADX method.
"""

import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

# Import MarketRegime from strategy_v2 (avoid circular import by lazy check)
try:
    from strategy_v2 import MarketRegime
except ImportError:
    MarketRegime = None


class HMMRegimeDetector:
    """
    3-state Gaussian HMM for market regime detection.

    Per-symbol models are fitted on historical data and refitted
    periodically as new bars arrive.
    """

    def __init__(self, config):
        self.config = config
        self._models = {}             # symbol -> GaussianHMM
        self._bars_since_refit = {}   # symbol -> int
        self._state_labels = {}       # symbol -> {state_idx: label_str}
        self._last_probs = {}         # symbol -> probability dict

        # Ensure model directory exists
        os.makedirs(config.model_dir, exist_ok=True)

    def detect_regime(self, df: pd.DataFrame, symbol: str) -> tuple:
        """
        Returns (MarketRegime or None, probabilities_dict).

        probabilities_dict has keys: "low_vol", "normal", "high_vol"
        with values 0.0-1.0 summing to ~1.0.

        Returns (None, {}) if HMM not fitted or insufficient data,
        allowing the caller to fall back to the original ADX method.
        """
        if not self.config.enabled:
            return None, {}

        if df is None or len(df) < self.config.min_bars_for_fit:
            return None, {}

        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("[HMM] hmmlearn not installed — falling back to ADX")
            return None, {}

        try:
            # Check if refit needed
            model = self._models.get(symbol)
            bars_since = self._bars_since_refit.get(symbol, self.config.refit_interval_bars)

            if model is None or bars_since >= self.config.refit_interval_bars:
                model = self._fit(df, symbol)
                if model is None:
                    return None, {}

            # Increment bars counter
            self._bars_since_refit[symbol] = self._bars_since_refit.get(symbol, 0) + 1

            # Extract features and predict current state
            features = self._extract_features(df)
            if features is None or len(features) < 2:
                return None, {}

            # Predict state probabilities for the latest bar
            try:
                state_probs = model.predict_proba(features)
                current_state = model.predict(features)
            except Exception as e:
                logger.debug(f"[HMM] Prediction failed for {symbol}: {e}")
                return None, {}

            latest_probs = state_probs[-1]  # Probabilities for last bar
            latest_state = current_state[-1]

            # Map state to regime
            labels = self._state_labels.get(symbol, {})
            regime = self._map_state_to_regime(latest_state, labels)
            prob_dict = self._build_prob_dict(latest_probs, labels)

            self._last_probs[symbol] = prob_dict

            logger.debug(
                f"[HMM] {symbol}: state={latest_state} regime={regime.value if regime else 'unknown'} "
                f"probs=low_vol:{prob_dict.get('low_vol', 0):.0%} "
                f"normal:{prob_dict.get('normal', 0):.0%} "
                f"high_vol:{prob_dict.get('high_vol', 0):.0%}"
            )

            return regime, prob_dict

        except Exception as e:
            logger.warning(f"[HMM] detect_regime failed for {symbol}: {e}")
            return None, {}

    def get_regime_probs(self, symbol: str) -> dict:
        """Get the last computed regime probabilities for a symbol."""
        return self._last_probs.get(symbol, {})

    def _extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Build observation matrix from DataFrame."""
        try:
            close = df["close"]
            if close.isna().all():
                return None

            # 1. Returns
            returns = close.pct_change().fillna(0)

            # 2. Realized volatility (rolling 10-bar std of returns)
            realized_vol = returns.rolling(10, min_periods=3).std().fillna(0)

            # 3. Volume ratio (current / 20-bar avg)
            if "volume" in df.columns:
                vol_avg = df["volume"].rolling(20, min_periods=5).mean()
                volume_ratio = (df["volume"] / vol_avg).fillna(1.0)
                volume_ratio = volume_ratio.clip(0, 10)  # Cap outliers
            else:
                volume_ratio = pd.Series(1.0, index=df.index)

            # 4. Spread proxy: (high - low) / close
            if "high" in df.columns and "low" in df.columns:
                spread_proxy = ((df["high"] - df["low"]) / close).fillna(0)
            else:
                spread_proxy = pd.Series(0.0, index=df.index)

            features = np.column_stack([
                returns.values,
                realized_vol.values,
                volume_ratio.values,
                spread_proxy.values,
            ])

            # Replace NaN/Inf
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            return features

        except Exception as e:
            logger.debug(f"[HMM] Feature extraction failed: {e}")
            return None

    def _fit(self, df: pd.DataFrame, symbol: str):
        """Fit GaussianHMM on recent data."""
        try:
            from hmmlearn.hmm import GaussianHMM

            # Use tail of data for fitting
            fit_df = df.tail(self.config.lookback_bars)
            features = self._extract_features(fit_df)

            if features is None or len(features) < self.config.min_bars_for_fit:
                logger.debug(f"[HMM] Insufficient data for {symbol}: {len(features) if features is not None else 0} bars")
                return None

            model = GaussianHMM(
                n_components=self.config.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42,
                verbose=False,
            )

            # Suppress convergence warnings during fit
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(features)

            # Label states by realized volatility
            self._label_states(model, features, symbol)

            self._models[symbol] = model
            self._bars_since_refit[symbol] = 0

            logger.info(
                f"[HMM] Fitted {self.config.n_states}-state model for {symbol} "
                f"on {len(features)} bars — converged={model.monitor_.converged}"
            )

            return model

        except Exception as e:
            logger.warning(f"[HMM] Fitting failed for {symbol}: {e}")
            return None

    def _label_states(self, model, features: np.ndarray, symbol: str):
        """
        Assign human-readable labels to HMM states.

        State with lowest mean |returns| → low_vol (RANGING)
        State with highest mean |returns| → high_vol (VOLATILE)
        Remaining → normal (TRENDING)
        """
        try:
            states = model.predict(features)
            state_vols = {}

            for s in range(self.config.n_states):
                mask = states == s
                count = mask.sum()
                if count > 0:
                    # Use mean absolute return as volatility proxy
                    state_vols[s] = np.abs(features[mask, 0]).mean()
                else:
                    state_vols[s] = 0

            sorted_states = sorted(state_vols.keys(), key=lambda s: state_vols[s])

            labels = {}
            labels[sorted_states[0]] = "low_vol"      # Lowest volatility
            if len(sorted_states) > 2:
                labels[sorted_states[1]] = "normal"    # Middle
                labels[sorted_states[-1]] = "high_vol" # Highest volatility
            else:
                labels[sorted_states[-1]] = "high_vol"

            self._state_labels[symbol] = labels

            # Log state distribution
            for s in sorted_states:
                pct = (states == s).mean()
                label = labels.get(s, "unknown")
                vol = state_vols[s]
                logger.debug(
                    f"[HMM] {symbol} state {s} ({label}): "
                    f"{pct:.0%} of bars, avg |ret|={vol:.5f}"
                )

        except Exception as e:
            logger.debug(f"[HMM] State labeling failed: {e}")

    def _map_state_to_regime(self, state: int, labels: dict):
        """Map HMM state to MarketRegime enum."""
        if MarketRegime is None:
            return None

        label = labels.get(state, "normal")

        if label == "low_vol":
            return MarketRegime.RANGING
        elif label == "high_vol":
            return MarketRegime.VOLATILE
        else:
            # For "normal" state, default to RANGING (good for MR)
            return MarketRegime.RANGING

    def _build_prob_dict(self, state_probs: np.ndarray, labels: dict) -> dict:
        """Build probability dict from state probabilities."""
        prob_dict = {"low_vol": 0.0, "normal": 0.0, "high_vol": 0.0}

        for state_idx, prob in enumerate(state_probs):
            label = labels.get(state_idx, "normal")
            if label in prob_dict:
                prob_dict[label] = float(prob)

        return prob_dict

    def get_status(self) -> dict:
        """Return current detector status for logging/monitoring."""
        return {
            "enabled": self.config.enabled,
            "models_fitted": list(self._models.keys()),
            "last_probs": self._last_probs,
        }
