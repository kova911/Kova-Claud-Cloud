"""
ML Signal Filter — Renaissance-Inspired Signal Quality Predictor

Uses an ensemble of XGBoost + Random Forest to predict the probability
that a generated signal will result in a winning trade.

Cold start behavior: Passes ALL signals through until min_trades_for_training
completed trades are available. After that, trains and gates signals below
the win_probability_threshold.

Training data: knowledge/trades_*.jsonl and knowledge/signals_*.jsonl files
containing historical trade outcomes and signal component breakdowns.
"""

import logging
import os
import json
import glob
import time
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

logger = logging.getLogger(__name__)


class MLSignalFilter:
    """
    XGBoost + Random Forest ensemble to predict signal quality.

    Cold start: passes all signals through until min_trades_for_training
    completed trades are available. After that, trains and gates.
    """

    # Feature columns used for training and prediction
    FEATURE_COLUMNS = [
        "zscore_score", "volume_score", "momentum_score",
        "volatility_score", "trend_score", "micro_score",
        "composite_score", "regime_encoded", "vix",
        "dax_pct", "hour", "day_of_week", "symbol_encoded",
        "atr_pct",
    ]

    def __init__(self, config, trade_db=None):
        self.config = config
        self._trade_db = trade_db  # Optional TradeDatabase for richer training data
        self._xgb_model = None
        self._rf_model = None
        self._symbol_encoder = LabelEncoder()
        self._symbol_encoder_fitted = False
        self._trades_since_last_train = 0
        self._total_training_trades = 0
        self._is_trained = False
        self._last_train_time = 0
        self._train_metrics = {}

        # Ensure model directory exists
        os.makedirs(config.model_dir, exist_ok=True)

        # Try to load persisted models
        self._load_models()

    def should_allow_signal(self, signal, market_ctx: dict) -> tuple[bool, float, str]:
        """
        Main entry point. Returns (allow, win_probability, reason).

        In cold-start mode: returns (True, 0.5, "cold_start")
        When trained: returns (prob >= threshold, prob, "ml_filter")
        """
        if not self.config.enabled:
            return True, 0.5, "ml_disabled"

        if not self._is_trained:
            return True, 0.5, "cold_start"

        try:
            features = self._extract_features(signal, market_ctx)
            prob = self._predict(features)
            allowed = prob >= self.config.win_probability_threshold
            reason = f"ml_prob={prob:.3f}"
            if not allowed:
                reason += f" < threshold={self.config.win_probability_threshold}"
            return allowed, prob, reason
        except Exception as e:
            logger.warning(f"[ML FILTER] Prediction failed: {e} — passing signal through")
            return True, 0.5, f"ml_error:{e}"

    def notify_trade_complete(self, trade_record: dict):
        """Called when a trade closes. Triggers retraining if needed."""
        self._trades_since_last_train += 1
        logger.debug(
            f"[ML FILTER] Trade complete. {self._trades_since_last_train} trades "
            f"since last train (threshold: {self.config.retrain_interval_trades})"
        )
        if self._trades_since_last_train >= self.config.retrain_interval_trades:
            self.retrain()

    def retrain(self):
        """Load all trade data from knowledge/ JSONL, build features, train."""
        logger.info("[ML FILTER] Starting retrain cycle...")
        try:
            # 1. Load all trade records
            trades = self._load_trade_data()
            if len(trades) < self.config.min_trades_for_training:
                logger.info(
                    f"[ML FILTER] Only {len(trades)} completed trades, "
                    f"need {self.config.min_trades_for_training} — staying in cold start"
                )
                return

            # 2. Build feature matrix and labels
            X, y, feature_names = self._build_training_data(trades)
            if X is None or len(X) < self.config.min_trades_for_training:
                logger.warning("[ML FILTER] Insufficient valid training samples")
                return

            # 3. Walk-forward split
            split_idx = int(len(X) * self.config.walk_forward_train_pct)
            if split_idx < 10 or (len(X) - split_idx) < 5:
                logger.warning("[ML FILTER] Train/val split too small")
                return

            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # 4. Train XGBoost
            try:
                import xgboost as xgb
                self._xgb_model = xgb.XGBClassifier(
                    max_depth=self.config.max_tree_depth,
                    n_estimators=self.config.n_estimators,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=42,
                    verbosity=0,
                )
                self._xgb_model.fit(X_train, y_train)
                xgb_val_prob = self._xgb_model.predict_proba(X_val)[:, 1]
                xgb_val_acc = ((xgb_val_prob >= 0.5) == y_val).mean()
                logger.info(f"[ML FILTER] XGBoost val accuracy: {xgb_val_acc:.1%}")
            except ImportError:
                logger.warning("[ML FILTER] XGBoost not available, using RF only")
                self._xgb_model = None

            # 5. Train Random Forest
            self._rf_model = RandomForestClassifier(
                max_depth=self.config.max_tree_depth,
                n_estimators=self.config.n_estimators,
                random_state=42,
            )
            self._rf_model.fit(X_train, y_train)
            rf_val_prob = self._rf_model.predict_proba(X_val)[:, 1]
            rf_val_acc = ((rf_val_prob >= 0.5) == y_val).mean()
            logger.info(f"[ML FILTER] RF val accuracy: {rf_val_acc:.1%}")

            # 6. Ensemble validation
            if self._xgb_model is not None:
                ensemble_prob = (
                    self.config.xgboost_weight * xgb_val_prob
                    + self.config.rf_weight * rf_val_prob
                )
            else:
                ensemble_prob = rf_val_prob
            ensemble_acc = ((ensemble_prob >= 0.5) == y_val).mean()

            # Feature importances
            if self._rf_model is not None:
                importances = dict(zip(self.FEATURE_COLUMNS[:len(self._rf_model.feature_importances_)],
                                       self._rf_model.feature_importances_))
                top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]
                logger.info(f"[ML FILTER] Top features: {top_features}")

            self._train_metrics = {
                "train_size": len(X_train),
                "val_size": len(X_val),
                "ensemble_val_accuracy": float(ensemble_acc),
                "rf_val_accuracy": float(rf_val_acc),
                "xgb_val_accuracy": float(xgb_val_acc) if self._xgb_model else None,
                "train_win_rate": float(y_train.mean()),
                "val_win_rate": float(y_val.mean()),
                "timestamp": datetime.now().isoformat(),
            }

            self._is_trained = True
            self._trades_since_last_train = 0
            self._total_training_trades = len(trades)
            self._last_train_time = time.time()

            # 7. Persist models
            self._save_models()

            logger.info(
                f"[ML FILTER] Retrain complete — {len(trades)} trades, "
                f"ensemble val acc={ensemble_acc:.1%}, "
                f"train WR={y_train.mean():.1%}, val WR={y_val.mean():.1%}"
            )

        except Exception as e:
            logger.error(f"[ML FILTER] Retrain failed: {e}", exc_info=True)

    def _extract_features(self, signal, market_ctx: dict) -> np.ndarray:
        """Extract feature vector from a signal and market context."""
        indicators = signal.indicators or {}

        # Regime encoding
        regime_map = {
            "ranging": 0, "trending_up": 1, "trending_down": 2, "volatile": 3,
        }
        regime_val = regime_map.get(signal.regime.value, 0) if signal.regime else 0

        # Symbol encoding
        symbol_val = 0
        if self._symbol_encoder_fitted:
            try:
                symbol_val = self._symbol_encoder.transform([signal.symbol])[0]
            except ValueError:
                symbol_val = -1  # Unknown symbol

        # Time features
        ts = signal.timestamp if isinstance(signal.timestamp, datetime) else datetime.now()
        hour = ts.hour
        dow = ts.weekday()

        # ATR %
        atr_pct = 0.0
        if hasattr(signal, "stop_loss") and signal.price > 0:
            # Estimate ATR from stop distance / multiplier
            stop_dist = abs(signal.price - signal.stop_loss)
            atr_pct = stop_dist / signal.price if signal.price > 0 else 0

        features = np.array([
            indicators.get("zscore", 0.0),
            indicators.get("volume", 0.0),
            indicators.get("momentum", 0.0),
            indicators.get("volatility", 0.0),
            indicators.get("trend", 0.0),
            indicators.get("microstructure", 0.0),
            signal.confidence,
            regime_val,
            market_ctx.get("vix") or 0.0,
            market_ctx.get("dax_pct") or 0.0,
            hour,
            dow,
            symbol_val,
            atr_pct,
        ], dtype=np.float64).reshape(1, -1)

        return features

    def _predict(self, features: np.ndarray) -> float:
        """Ensemble prediction: weighted average of XGBoost and RF probabilities."""
        probs = []
        weights = []

        if self._xgb_model is not None:
            xgb_prob = self._xgb_model.predict_proba(features)[0, 1]
            probs.append(xgb_prob)
            weights.append(self.config.xgboost_weight)

        if self._rf_model is not None:
            rf_prob = self._rf_model.predict_proba(features)[0, 1]
            probs.append(rf_prob)
            weights.append(self.config.rf_weight)

        if not probs:
            return 0.5  # No models available

        # Weighted average
        total_weight = sum(weights)
        ensemble_prob = sum(p * w for p, w in zip(probs, weights)) / total_weight
        return float(ensemble_prob)

    def _load_trade_data(self) -> list[dict]:
        """
        Load all completed trade records.
        Tries SQLite database first (richer data), falls back to JSONL parsing.
        """
        # ── Try SQLite database first (richer features) ──
        if self._trade_db:
            try:
                df = self._trade_db.get_ml_training_data(lookback_days=90)
                if df is not None and len(df) >= 10:
                    trades = df.to_dict("records")
                    logger.info(f"[ML FILTER] Loaded {len(trades)} trades from SQLite DB")
                    return trades
            except Exception as e:
                logger.debug(f"[ML FILTER] DB load failed, falling back to JSONL: {e}")

        # ── Fallback: JSONL parsing ──
        knowledge_dir = "knowledge"
        trades = []

        # Load trade_close records (these have PnL outcomes)
        trade_files = sorted(glob.glob(os.path.join(knowledge_dir, "trades_*.jsonl")))
        opens = {}
        closes = []

        for filepath in trade_files:
            try:
                with open(filepath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        rtype = record.get("type") or record.get("event")
                        if rtype == "trade_open":
                            key = (record.get("symbol"), record.get("direction"))
                            opens[key] = record
                        elif rtype == "trade_close":
                            closes.append(record)
            except Exception as e:
                logger.warning(f"[ML FILTER] Error reading {filepath}: {e}")

        # Match closes with opens to build complete trade records
        for close in closes:
            key = (close.get("symbol"), close.get("direction"))
            open_rec = opens.get(key)
            if open_rec:
                trade = {
                    **open_rec,
                    "pnl": close.get("pnl", 0),
                    "exit_price": close.get("exit_price", 0),
                    "exit_reason": close.get("exit_reason", ""),
                    "mae_pct": close.get("mae_pct", close.get("mae", 0)),
                    "mfe_pct": close.get("mfe_pct", close.get("mfe", 0)),
                    "duration_seconds": close.get("duration_seconds", close.get("hold_time_sec", 0)),
                }
                trades.append(trade)

        # Also load signal records for component data
        signal_files = sorted(glob.glob(os.path.join(knowledge_dir, "signals_*.jsonl")))
        signal_map = {}  # (symbol, timestamp_approx) -> signal_record
        for filepath in signal_files:
            try:
                with open(filepath) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        action = record.get("action") or record.get("event")
                        if action in ("executed", "signal"):
                            sym = record.get("symbol")
                            ts = record.get("timestamp", "")
                            signal_map[(sym, ts[:16])] = record  # Match by minute
            except Exception as e:
                logger.debug(f"[ML FILTER] Error reading signals {filepath}: {e}")

        # Enrich trades with signal component data
        for trade in trades:
            sym = trade.get("symbol")
            ts = trade.get("timestamp", "")
            sig_rec = signal_map.get((sym, ts[:16]))
            if sig_rec:
                # Use indicators dict from record_signal (has zscore, volume, momentum etc.)
                indicators = sig_rec.get("indicators", {})
                trade["signal_components"] = indicators
                trade["signal_confidence"] = sig_rec.get("confidence", trade.get("signal_confidence", 0))

        logger.info(f"[ML FILTER] Loaded {len(trades)} completed trades from {len(trade_files)} JSONL files")
        return trades

    def _build_training_data(self, trades: list[dict]) -> tuple:
        """Build feature matrix X and label vector y from trade records."""
        rows = []
        labels = []

        # Collect all symbols for encoding
        all_symbols = list(set(t.get("symbol", "") for t in trades))
        if all_symbols:
            self._symbol_encoder.fit(all_symbols)
            self._symbol_encoder_fitted = True

        for trade in trades:
            try:
                components = trade.get("signal_components", {})
                market_ctx = trade.get("market_context", {})

                # Regime encoding
                regime_map = {
                    "ranging": 0, "trending_up": 1, "trending_down": 2, "volatile": 3,
                }
                regime_str = trade.get("regime", "ranging")
                regime_val = regime_map.get(regime_str, 0)

                # Symbol encoding
                symbol = trade.get("symbol", "")
                try:
                    symbol_val = self._symbol_encoder.transform([symbol])[0]
                except (ValueError, AttributeError):
                    symbol_val = 0

                # Time features from timestamp
                ts_str = trade.get("timestamp", "")
                try:
                    ts = datetime.fromisoformat(ts_str) if ts_str else datetime.now()
                except (ValueError, TypeError):
                    ts = datetime.now()

                # ATR % estimation
                entry_price = trade.get("entry_price", 0)
                stop_loss = trade.get("stop_loss", 0)
                atr_pct = abs(entry_price - stop_loss) / entry_price if entry_price > 0 else 0

                features = [
                    components.get("zscore", 0.0),
                    components.get("volume", 0.0),
                    components.get("momentum", 0.0),
                    components.get("volatility", 0.0),
                    components.get("trend", 0.0),
                    components.get("microstructure", 0.0),
                    trade.get("signal_confidence", trade.get("confidence", 0.0)),
                    regime_val,
                    market_ctx.get("vix") or 0.0,
                    market_ctx.get("dax_pct") or 0.0,
                    ts.hour,
                    ts.weekday(),
                    symbol_val,
                    atr_pct,
                ]

                # Label: 1 if profitable, 0 otherwise
                pnl = trade.get("pnl", 0)
                label = 1 if pnl > 0 else 0

                rows.append(features)
                labels.append(label)

            except Exception as e:
                logger.debug(f"[ML FILTER] Skipping trade in training: {e}")
                continue

        if not rows:
            return None, None, None

        X = np.array(rows, dtype=np.float64)
        y = np.array(labels, dtype=np.int32)

        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(
            f"[ML FILTER] Training data: {len(X)} samples, "
            f"win rate={y.mean():.1%}, features={X.shape[1]}"
        )
        return X, y, self.FEATURE_COLUMNS

    def _save_models(self):
        """Persist models to disk."""
        try:
            if self._xgb_model is not None:
                path = os.path.join(self.config.model_dir, "xgb_latest.joblib")
                joblib.dump(self._xgb_model, path)

            if self._rf_model is not None:
                path = os.path.join(self.config.model_dir, "rf_latest.joblib")
                joblib.dump(self._rf_model, path)

            if self._symbol_encoder_fitted:
                path = os.path.join(self.config.model_dir, "symbol_encoder.joblib")
                joblib.dump(self._symbol_encoder, path)

            # Save metrics
            metrics_path = os.path.join(self.config.model_dir, "train_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(self._train_metrics, f, indent=2)

            logger.info(f"[ML FILTER] Models saved to {self.config.model_dir}")
        except Exception as e:
            logger.error(f"[ML FILTER] Failed to save models: {e}")

    def _load_models(self):
        """Try to load persisted models from disk."""
        try:
            xgb_path = os.path.join(self.config.model_dir, "xgb_latest.joblib")
            rf_path = os.path.join(self.config.model_dir, "rf_latest.joblib")
            enc_path = os.path.join(self.config.model_dir, "symbol_encoder.joblib")

            loaded = False

            if os.path.exists(rf_path):
                self._rf_model = joblib.load(rf_path)
                loaded = True
                logger.info("[ML FILTER] Loaded RF model from disk")

            if os.path.exists(xgb_path):
                self._xgb_model = joblib.load(xgb_path)
                loaded = True
                logger.info("[ML FILTER] Loaded XGBoost model from disk")

            if os.path.exists(enc_path):
                self._symbol_encoder = joblib.load(enc_path)
                self._symbol_encoder_fitted = True

            if loaded:
                self._is_trained = True
                # Load metrics
                metrics_path = os.path.join(self.config.model_dir, "train_metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path) as f:
                        self._train_metrics = json.load(f)
                    self._total_training_trades = self._train_metrics.get("train_size", 0)
                logger.info(
                    f"[ML FILTER] Restored trained models "
                    f"(trained on {self._total_training_trades} trades)"
                )

        except Exception as e:
            logger.debug(f"[ML FILTER] No persisted models found: {e}")

    def get_status(self) -> dict:
        """Return current filter status for logging/monitoring."""
        return {
            "enabled": self.config.enabled,
            "is_trained": self._is_trained,
            "total_training_trades": self._total_training_trades,
            "trades_since_last_train": self._trades_since_last_train,
            "train_metrics": self._train_metrics,
        }
