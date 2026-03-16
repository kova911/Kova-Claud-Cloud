#!/usr/bin/env python3
"""
Trading System — Main Orchestrator

Daily workflow:
1. Fetch latest market data for watchlist
2. Compute indicators on all symbols
3. Scan for trading signals
4. Validate signals through risk manager
5. Present signals for execution (paper or live)
6. Monitor open positions (stop-loss, take-profit, trailing)
7. End-of-day: close all, generate report, trigger self-learning

Usage:
    python main.py                  # Full trading session (paper mode)
    python main.py --backtest       # Run backtest on historical data
    python main.py --learn          # Force learning/optimization cycle
    python main.py --scan           # One-shot signal scan (no trading)
    python main.py --live           # Live trading (requires IBKR connection)
"""

import sys
import time
import json
import logging
import argparse
import signal as sig
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from config import config, TradingConfig
from data_collector import DataCollector
from indicators import Indicators
from strategy_v2 import StrategyV2, Signal, SignalType, MarketRegime
from risk_manager import RiskManager, PositionStatus
from backtester import Backtester
from learning import SelfLearner
from trade_executor import TradeExecutor, OrderStatus, OrderType
from market_filters import MarketFilters
from skill_updater import SkillUpdater
from analytics import Analytics
from trading_knowledge import TradingKnowledge
from db_learner import DBSelfLearner

# ─── Logging Setup ────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"trading_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("main")

# Try rich for better console output
try:
    from rich.logging import RichHandler
    logging.getLogger().addHandler(RichHandler(rich_tracebacks=True))
except ImportError:
    pass


class TradingSystem:
    """Main trading system orchestrator."""

    def __init__(self, cfg: TradingConfig):
        self.config = cfg
        self.collector = DataCollector(cfg)

        # ─── Renaissance Modules (optional, config-driven) ───
        self.ml_filter = None
        self.hmm_detector = None
        self.cross_correlations = None
        self.pairs_trader = None
        self._dax_cache: Optional[pd.DataFrame] = None

        # Initialize HMM before strategy (strategy accepts hmm_detector)
        if cfg.hmm_regime.enabled:
            try:
                from hmm_regime import HMMRegimeDetector
                self.hmm_detector = HMMRegimeDetector(cfg.hmm_regime)
                logger.info("Renaissance: HMM Regime Detection ENABLED")
            except Exception as e:
                logger.warning(f"Renaissance: HMM Regime Detection failed to init: {e}")

        # Strategy V2 (with optional HMM detector)
        self.strategy = StrategyV2(cfg.strategy, cfg.risk, hmm_detector=self.hmm_detector)

        # Initialize cross-asset correlations and inject into strategy
        if cfg.cross_correlation.enabled:
            try:
                from cross_correlations import CrossAssetCorrelations
                self.cross_correlations = CrossAssetCorrelations(cfg.cross_correlation)
                self.strategy.set_cross_correlations(self.cross_correlations)
                logger.info("Renaissance: Cross-Asset Correlations ENABLED")
            except Exception as e:
                logger.warning(f"Renaissance: Cross-Asset Correlations failed to init: {e}")

        # ML Signal Filter (will inject DB reference after knowledge init)
        if cfg.ml_filter.enabled:
            try:
                from ml_filter import MLSignalFilter
                self.ml_filter = MLSignalFilter(cfg.ml_filter)
                logger.info("Renaissance: ML Signal Filter ENABLED")
            except Exception as e:
                logger.warning(f"Renaissance: ML Signal Filter failed to init: {e}")

        # Pairs Trading
        if cfg.pairs_trading.enabled:
            try:
                from pairs_trading import PairsTrader
                self.pairs_trader = PairsTrader(cfg.pairs_trading, cfg.risk)
                logger.info("Renaissance: Pairs Trading ENABLED")
            except Exception as e:
                logger.warning(f"Renaissance: Pairs Trading failed to init: {e}")

        self.risk_mgr = RiskManager(cfg.risk)
        self.backtester = Backtester(cfg)
        self.learner = SelfLearner(cfg)
        self.executor = TradeExecutor(cfg)
        self.filters = MarketFilters(cfg)
        self.skill_updater = SkillUpdater(cfg, self.risk_mgr, self.learner)
        self.knowledge = TradingKnowledge(cfg)

        # Inject DB reference into ML filter for richer training data
        if self.ml_filter and self.knowledge.db:
            self.ml_filter._trade_db = self.knowledge.db
            logger.info("Renaissance: ML Filter connected to SQLite DB for training data")

        # DB Self-Learner — reads trade history and generates actionable insights
        self.db_learner = None
        try:
            self.db_learner = DBSelfLearner(db=self.knowledge.db)
            logger.info("Renaissance: DB Self-Learner ENABLED")
        except Exception as e:
            logger.warning(f"Renaissance: DB Self-Learner failed to init: {e}")

        self._running = False
        self._data_cache: dict[str, pd.DataFrame] = {}
        # Track recently-cancelled symbols to prevent race condition:
        # after stale cancel, IBKR may still fill the order before processing cancel.
        # Block new orders for a symbol for 5 minutes after any cancel.
        self._cancelled_symbols: dict[str, float] = {}  # symbol -> cancel timestamp
        self._cancel_cooldown_sec: float = 300  # 5 minutes

    def startup(self):
        """Initialize the trading system."""
        logger.info("=" * 60)
        logger.info("TRADING SYSTEM — Interactive Brokers")
        logger.info("=" * 60)

        # Validate config
        issues = self.config.validate()
        if issues:
            for issue in issues:
                logger.warning(f"Config issue: {issue}")

        mode = "PAPER" if self.config.ibkr.paper_trading else "LIVE"
        port = self.config.ibkr.paper_port if self.config.ibkr.paper_trading else self.config.ibkr.live_port
        logger.info(f"Mode: {mode}")
        logger.info(f"IBKR: {self.config.ibkr.host}:{port} (client {self.config.ibkr.client_id})")
        logger.info(f"Watchlist: {len(self.config.watchlist.symbols)} symbols")
        logger.info(f"Strategy: V2 Renaissance-Optimized (zscore_lb={self.strategy.zscore_lookback}, "
                     f"min_score={self.strategy.min_composite_score})")
        logger.info(f"ATR Stops: SL={self.strategy.atr_stop_multiplier}×ATR, TP={self.strategy.atr_tp_multiplier}×ATR")
        logger.info(f"Trailing: activate={self.strategy.trailing_atr_activation}×ATR, "
                     f"trail={self.strategy.trailing_atr_distance}×ATR")
        logger.info(f"Min edge: €{self.strategy.min_expected_gain_eur:.0f}/trade")
        logger.info(f"Self-learning: {'ON' if self.config.learning.enabled else 'OFF'}")

        # Connect to Interactive Brokers
        connected = self.executor.connect()
        if connected and self.executor.is_ibkr_connected:
            # Share the IBKR connection with data collector and filters
            self.collector.set_ibkr_connection(self.executor.ib_connection)
            self.filters.set_ibkr_connection(self.executor.ib_connection)
            logger.info("IBKR connection shared with data collector and filters.")
        else:
            logger.info("Running without IBKR connection (local paper mode).")

        # Initialize market filters (fetch calendar, sentiment)
        self.filters.startup()

        # Restore state from disk (crash recovery)
        if self.risk_mgr.load_state():
            logger.info("Restored state from previous session.")
            if self.risk_mgr.positions:
                logger.warning(
                    f"  {len(self.risk_mgr.positions)} open positions restored — "
                    f"will resume monitoring."
                )
                # CRITICAL: Clear has_bracket flag on ALL restored positions.
                # The original IBKR bracket SL/TP orders from the previous session
                # are GONE (in-memory only, not persisted). If we keep has_bracket=True,
                # update_position() skips ALL client-side SL/TP/trailing checks and the
                # position runs with ZERO stop-loss protection.
                for pos in self.risk_mgr.positions:
                    if pos.has_bracket:
                        logger.warning(
                            f"  ⚠ {pos.symbol}: clearing has_bracket flag — "
                            f"client-side SL/TP will now protect this position"
                        )
                        pos.has_bracket = False
                        pos.bracket_order_id = ""

        # Migrate historical JSONL data to SQLite (one-time, idempotent)
        if self.knowledge.db:
            try:
                self.knowledge.db.migrate_jsonl("knowledge")
            except Exception as e:
                logger.warning(f"JSONL migration skipped: {e}")

        # DB Self-Learner: generate insights from trade history before trading starts
        if self.db_learner:
            try:
                insights = self.db_learner.generate_insights()
                if insights:
                    summary = self.db_learner.format_insights_summary()
                    for line in summary.split("\n"):
                        logger.info(line)
                else:
                    logger.info("[DBLearner] Not enough trade data for insights yet")
            except Exception as e:
                logger.warning(f"[DBLearner] Insight generation failed: {e}")

        logger.info("System ready.")

    def fetch_all_data(self, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """Fetch and prepare data for watchlist symbols (or a filtered subset)."""
        data = {}
        if symbols is None:
            symbols = self.config.watchlist.symbols

        logger.info(f"Fetching data for {len(symbols)} symbols...")
        for symbol in symbols:
            try:
                df = self.collector.get_historical(
                    symbol,
                    interval=self.config.strategy.signal_timeframe,
                    days=self.config.data.backtest_days,
                )
                if df is not None and len(df) >= self.config.data.min_bars_required:
                    # Add indicators
                    df = Indicators.add_all(df, self.config.strategy)
                    data[symbol] = df
                    logger.info(f"  {symbol}: {len(df)} bars loaded")
                else:
                    logger.warning(f"  {symbol}: Insufficient data ({len(df) if df is not None else 0} bars)")
            except Exception as e:
                logger.error(f"  {symbol}: Data fetch failed — {e}")

        self._data_cache = data

        # Fetch DAX index data for cross-asset correlations and pairs trading
        if self.cross_correlations or self.pairs_trader:
            try:
                import yfinance as yf
                dax_symbol = self.config.cross_correlation.dax_index_symbol
                dax_ticker = yf.Ticker(dax_symbol)
                dax_df = dax_ticker.history(
                    period=f"{self.config.data.backtest_days}d",
                    interval=self.config.strategy.signal_timeframe,
                )
                if dax_df is not None and len(dax_df) > 20:
                    # Standardize column names to lowercase
                    dax_df.columns = [c.lower() for c in dax_df.columns]
                    self._dax_cache = dax_df
                    logger.info(f"  DAX index: {len(dax_df)} bars loaded")
                else:
                    logger.warning(f"  DAX index: insufficient data")
            except Exception as e:
                logger.debug(f"  DAX index fetch failed: {e}")

        logger.info(f"Data ready: {len(data)}/{len(symbols)} symbols")
        return data

    def scan_signals(self) -> list:
        """Scan watchlist for trading signals."""
        if not self._data_cache:
            self.fetch_all_data()

        signals = self.strategy.scan_watchlist(self._data_cache, dax_df=self._dax_cache)

        if signals:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"SIGNALS FOUND: {len(signals)}")
            logger.info(f"{'=' * 60}")
            for s in signals:
                logger.info(
                    f"  {s.type.value:5s} {s.symbol:10s} @ {s.price:8.2f} "
                    f"[{s.confidence:.0%}] SL:{s.stop_loss:.2f} TP:{s.take_profit:.2f} "
                    f"— {s.reason}"
                )
        else:
            logger.info("No signals found in current scan.")

        return signals

    def _get_market_context(self) -> dict:
        """Get current market conditions snapshot for knowledge recording."""
        status = self.filters.get_status()
        sent = status.get("sentiment", {})
        return {
            "vix": sent.get("vix_level"),
            "vix_change": sent.get("vix_change_pct"),
            "dax_pct": sent.get("dax_change_pct"),
            "sp500_pct": sent.get("sp500_change_pct"),
        }

    def _get_indicator_snapshot(self, symbol: str) -> dict:
        """Extract current ATR/RSI/ADX/BB/volume indicators from cached data."""
        sym_df = self._data_cache.get(symbol)
        if sym_df is None or len(sym_df) == 0:
            return {}
        row = sym_df.iloc[-1]
        snap = {}
        for col in ("atr", "rsi", "adx", "bb_width", "volume_ratio"):
            val = row.get(col)
            if val is not None and not pd.isna(val):
                snap[col] = float(val)
        # ATR as percentage of price
        price = row.get("close", 0)
        if snap.get("atr") and price and price > 0:
            snap["atr_pct"] = round(snap["atr"] / price, 6)
        return snap

    def _execute_pair_signals(self, pair_signals: list):
        """Execute pairs trading signals — place both legs."""
        for ps in pair_signals:
            try:
                # Check risk limits
                can_trade, reason = self.risk_mgr.can_trade()
                if not can_trade:
                    logger.info(f"[PAIRS] Risk limit reached: {reason}")
                    continue

                # Market filter check
                can_trade_a, reason_a = self.filters.check(
                    symbol=ps.pair[0], signal_type=ps.leg_a_action
                )
                can_trade_b, reason_b = self.filters.check(
                    symbol=ps.pair[1], signal_type=ps.leg_b_action
                )
                if not can_trade_a or not can_trade_b:
                    logger.info(f"[PAIRS] Filtered: {reason_a or reason_b}")
                    continue

                # Build signals for each leg
                leg_a_type = SignalType.LONG if ps.leg_a_action == "BUY" else SignalType.SHORT
                leg_b_type = SignalType.LONG if ps.leg_b_action == "BUY" else SignalType.SHORT

                signal_a = Signal(
                    type=leg_a_type,
                    symbol=ps.pair[0],
                    price=ps.leg_a_price,
                    timestamp=datetime.now(),
                    confidence=ps.confidence,
                    regime=MarketRegime.RANGING,
                    strategy="pairs_entry",
                    indicators={"spread_zscore": ps.spread_zscore},
                    stop_loss=0,  # Pairs use spread-based exits, not per-leg SL
                    take_profit=0,
                    reason=ps.reason,
                )
                signal_b = Signal(
                    type=leg_b_type,
                    symbol=ps.pair[1],
                    price=ps.leg_b_price,
                    timestamp=datetime.now(),
                    confidence=ps.confidence,
                    regime=MarketRegime.RANGING,
                    strategy="pairs_entry",
                    indicators={"spread_zscore": ps.spread_zscore},
                    stop_loss=0,
                    take_profit=0,
                    reason=ps.reason,
                )

                # Execute both legs
                order_a = self.executor.place_order(signal_a, ps.leg_a_size)
                order_b = self.executor.place_order(signal_b, ps.leg_b_size)

                if order_a and order_b:
                    self.pairs_trader.register_position(ps, order_a, order_b)
                    logger.info(
                        f"[PAIRS] Executed: {ps.direction} "
                        f"{ps.pair[0]} x{ps.leg_a_size} + {ps.pair[1]} x{ps.leg_b_size}"
                    )
                else:
                    logger.warning(f"[PAIRS] Partial fill — one leg failed")

            except Exception as e:
                logger.error(f"[PAIRS] Execution error: {e}")

    def execute_signals(self, signals: list):
        """Process signals through market filters, risk manager, and executor."""
        market_ctx = self._get_market_context()

        # Cache IBKR portfolio ONCE per scan cycle (not per signal)
        ibkr_portfolio = {}
        if self.executor.is_ibkr_connected:
            ibkr_portfolio = self.executor.get_portfolio()

        for signal in signals:
            # ── Renaissance context: accumulate through signal pipeline ──
            ml_result_data = None    # (allowed, win_prob, reason)
            hmm_probs_data = None    # {"low_vol": .., "normal": .., "high_vol": ..}
            ind_snapshot = self._get_indicator_snapshot(signal.symbol)
            # Extract correlation data from signal indicators if present
            corr_snapshot = None
            sig_ind = signal.indicators or {}
            if sig_ind.get("correlation") is not None:
                corr_snapshot = {
                    "score": sig_ind.get("correlation"),
                    "zscore": sig_ind.get("corr_zscore"),
                }

            # Helper: common kwargs for all record_signal calls
            def _sig_ctx():
                return dict(
                    ml_result=ml_result_data,
                    hmm_probs=hmm_probs_data,
                    additional_indicators=ind_snapshot,
                    corr_data=corr_snapshot,
                )

            # 0a. Block recently-cancelled symbols (race condition guard)
            cancel_time = self._cancelled_symbols.get(signal.symbol)
            if cancel_time and (time.time() - cancel_time) < self._cancel_cooldown_sec:
                remaining = int(self._cancel_cooldown_sec - (time.time() - cancel_time))
                logger.info(f"[COOLDOWN] {signal.symbol}: cancelled {int(time.time() - cancel_time)}s ago, {remaining}s remaining — skipping")
                self.knowledge.record_signal(signal, market_ctx, "skipped", filter_reason="cancel_cooldown", **_sig_ctx())
                continue

            # 0b. Block if IBKR already has a position for this symbol (trust IBKR, not just internal state)
            if ibkr_portfolio:
                clean_sym = signal.symbol.replace(".DE", "")
                if clean_sym in ibkr_portfolio and ibkr_portfolio[clean_sym]["quantity"] != 0:
                    qty = ibkr_portfolio[clean_sym]["quantity"]
                    logger.info(f"[IBKR DUP] {signal.symbol}: IBKR already has position qty={qty} — skipping")
                    self.knowledge.record_signal(signal, market_ctx, "skipped", filter_reason="ibkr_position_exists", **_sig_ctx())
                    continue

            # 0c. Block duplicate: skip if symbol already has open position or pending order (internal)
            has_open = any(
                p.symbol == signal.symbol and p.status == PositionStatus.OPEN
                for p in self.risk_mgr.positions
            )
            has_pending = any(
                o.symbol == signal.symbol and o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED)
                for o in self.executor.orders
            )
            if has_open:
                logger.info(f"[DUPLICATE] {signal.symbol}: already has open position — skipping")
                self.knowledge.record_signal(signal, market_ctx, "skipped", filter_reason="duplicate_position", **_sig_ctx())
                continue
            if has_pending:
                logger.info(f"[DUPLICATE] {signal.symbol}: has pending order — skipping")
                self.knowledge.record_signal(signal, market_ctx, "skipped", filter_reason="pending_order", **_sig_ctx())
                continue

            # 0d. Late-session EU filter: block NEW entries for EU stocks after 15:00 CET
            # Live data: hour 15-16 = 0% WR, €-47.91. EU close is 17:30, no time for MR to play out.
            if signal.symbol.endswith(".DE"):
                now_cet = datetime.now(ZoneInfo(self.config.schedule.timezone))
                if now_cet.hour >= 15:
                    logger.info(f"[LATE SESSION] {signal.symbol}: EU stock after 15:00 CET — skipping new entry")
                    self.knowledge.record_signal(signal, market_ctx, "skipped", filter_reason="late_eu_session", **_sig_ctx())
                    continue

            # 1. Market filters (calendar, sentiment, spread)
            can_trade, reason = self.filters.check(
                symbol=signal.symbol,
                signal_type=signal.type.value,
            )
            if not can_trade:
                logger.info(f"[FILTERED] {signal.symbol}: {reason}")
                self.knowledge.record_signal(signal, market_ctx, "filtered", filter_reason=reason, **_sig_ctx())
                continue

            # 1b. ML Signal Filter (Renaissance: predict signal quality)
            if self.ml_filter:
                allowed, win_prob, ml_reason = self.ml_filter.should_allow_signal(signal, market_ctx)
                ml_result_data = (allowed, win_prob, ml_reason)
                if not allowed:
                    logger.info(f"[ML FILTER] {signal.symbol}: blocked ({ml_reason})")
                    self.knowledge.record_signal(signal, market_ctx, "filtered", filter_reason=f"ml:{ml_reason}", **_sig_ctx())
                    continue
                elif win_prob != 0.5:  # Skip logging for cold-start/disabled
                    logger.debug(f"[ML FILTER] {signal.symbol}: passed ({ml_reason})")

            # 2. Risk manager checks (daily limits, position limits, cooldown)
            can_trade, reason = self.risk_mgr.can_trade()
            if not can_trade:
                logger.info(f"Skipping {signal.symbol}: {reason}")
                self.knowledge.record_signal(signal, market_ctx, "skipped", filter_reason=reason, **_sig_ctx())
                continue

            size = self.risk_mgr.calculate_position_size(signal)
            if size <= 0:
                self.knowledge.record_signal(signal, market_ctx, "skipped", filter_reason="position_size_zero", **_sig_ctx())
                continue

            # 2b. DB Self-Learner adjustments (symbol weight, hour weight, confidence boost)
            if self.db_learner:
                try:
                    sym_weight = self.db_learner.get_symbol_weight(signal.symbol)
                    now_hour = datetime.now().hour
                    hour_weight = self.db_learner.get_hour_weight(now_hour)
                    conf_boost = self.db_learner.get_confidence_boost(
                        signal.strategy, signal.regime.value if hasattr(signal.regime, "value") else str(signal.regime)
                    )

                    # Apply symbol and hour weights to position size
                    learner_mult = sym_weight * hour_weight
                    if learner_mult != 1.0:
                        original_size = size
                        size = max(1, int(size * learner_mult))
                        if size != original_size:
                            logger.info(
                                f"[DBLearner] {signal.symbol}: size {original_size} → {size} "
                                f"(sym={sym_weight:.2f}, hour={hour_weight:.2f})"
                            )

                    # Apply confidence boost to signal
                    if conf_boost != 0.0:
                        old_conf = signal.confidence
                        signal.confidence = max(0.1, min(1.0, signal.confidence + conf_boost))
                        if signal.confidence != old_conf:
                            logger.debug(
                                f"[DBLearner] {signal.symbol}: confidence {old_conf:.2f} → "
                                f"{signal.confidence:.2f} (boost={conf_boost:+.2f})"
                            )
                except Exception as e:
                    logger.debug(f"[DBLearner] Weight lookup failed: {e}")

            # 3. Apply VIX-based size reduction (tiered volatility response)
            size_base = size  # Track pre-reduction size for DB
            vix_mult = self.filters.sentiment.get_vix_size_multiplier()
            if vix_mult < 1.0:
                original_size = size
                size = max(1, int(size * vix_mult))
                logger.info(f"[VIX TIER] {signal.symbol}: size {original_size} → {size} (VIX mult={vix_mult})")
            if size <= 0:
                self.knowledge.record_signal(signal, market_ctx, "skipped", filter_reason="vix_size_zero", **_sig_ctx())
                continue

            # 3b. HMM regime-based size reduction (Renaissance: probabilistic volatility)
            hmm_mult = 1.0
            if self.hmm_detector:
                probs = self.strategy.get_last_regime_probs()
                hmm_probs_data = probs  # Capture for DB
                high_vol_prob = probs.get("high_vol", 0)
                if high_vol_prob > 0.5:
                    original_size = size
                    reduction = self.config.hmm_regime.high_vol_size_reduction
                    hmm_mult = reduction
                    size = max(1, int(size * reduction))
                    logger.info(
                        f"[HMM] {signal.symbol}: high-vol prob={high_vol_prob:.0%}, "
                        f"size {original_size} → {size}"
                    )

            # Record signal as being executed (with full Renaissance context)
            self.knowledge.record_signal(signal, market_ctx, "executed", **_sig_ctx())

            # 3. Place bracket order (entry + SL + TP atomic) when IBKR connected
            is_bracket = self.executor.is_ibkr_connected
            if is_bracket:
                order = self.executor.place_bracket_order(signal, size)
            else:
                order = self.executor.place_order(signal, size)

            # Track bracket order in skill updater
            if is_bracket:
                filled = order and order.status.value == "filled"
                tick = getattr(order, '_tick_size', 0) if order else 0
                self.skill_updater.log_bracket_order(signal.symbol, filled, tick)

            if order and order.status.value == "filled":
                # Track position in risk manager with the SAME size that was executed
                executed_size = order.filled_quantity or size
                position = self.risk_mgr.open_position(signal, size=executed_size)
                if position:
                    # Update entry_price to actual fill price (not signal price)
                    # Signal price may differ from fill by aggressive offset + slippage
                    if order.filled_price > 0:
                        old_entry = position.entry_price
                        position.entry_price = order.filled_price
                        position.highest_price = order.filled_price
                        position.lowest_price = order.filled_price
                        position.current_price = order.filled_price
                        # Recalculate SL/TP relative to actual fill if they were based on signal price
                        if old_entry > 0 and abs(old_entry - order.filled_price) > 0.001:
                            sl_dist = abs(old_entry - position.stop_loss)
                            tp_dist = abs(old_entry - position.take_profit)
                            if position.side == SignalType.LONG:
                                position.stop_loss = order.filled_price - sl_dist
                                position.take_profit = order.filled_price + tp_dist
                            else:
                                position.stop_loss = order.filled_price + sl_dist
                                position.take_profit = order.filled_price - tp_dist
                            logger.info(
                                f"[FILL ADJ] {signal.symbol}: entry {old_entry:.2f} → {order.filled_price:.2f}, "
                                f"SL → {position.stop_loss:.2f}, TP → {position.take_profit:.2f}"
                            )
                    # Mark bracket positions so risk_manager skips client-side SL/TP
                    if is_bracket:
                        position.has_bracket = True
                        position.bracket_order_id = order.id
                    # Propagate slippage from executor to position for analytics
                    if order.slippage != 0:
                        position.slippage = order.slippage

                    # Set ATR-based trailing stop params from V2 strategy
                    if hasattr(self.strategy, 'get_trailing_stop_params'):
                        # Get ATR from cached data for this symbol
                        sym_df = self._data_cache.get(signal.symbol)
                        if sym_df is not None and len(sym_df) > 0:
                            atr_val = sym_df.iloc[-1].get("atr", 0)
                            if not pd.isna(atr_val) and atr_val > 0:
                                act_dist, trail_dist = self.strategy.get_trailing_stop_params(signal.symbol, atr_val)
                                position.atr_trail_activation = act_dist
                                position.atr_trail_distance = trail_dist
                                logger.info(
                                    f"[V2 TRAIL] {signal.symbol}: ATR-based trailing set — "
                                    f"activate at +€{act_dist:.2f}, trail by €{trail_dist:.2f}"
                                )

                    # Persist state after opening
                    self.risk_mgr.save_state()

                    # Record trade open in knowledge collector (with full Renaissance context)
                    self.knowledge.record_trade_open(
                        symbol=signal.symbol,
                        direction=signal.type.value,
                        entry_price=order.filled_price,
                        size=executed_size,
                        signal_confidence=signal.confidence,
                        strategy=signal.strategy,
                        regime=signal.regime.value,
                        market_context=market_ctx,
                        order_id=order.id,
                        tick_size=getattr(order, '_tick_size', 0),
                        slippage=order.slippage,
                        is_bracket=is_bracket,
                        # Renaissance context
                        signal_components=signal.indicators,
                        hmm_probs=hmm_probs_data,
                        ml_win_prob=ml_result_data[1] if ml_result_data else None,
                        corr_data=corr_snapshot,
                        additional_indicators=ind_snapshot,
                        size_base=size_base,
                        vix_mult=vix_mult,
                        hmm_mult=hmm_mult,
                        consecutive_wins=0,  # Risk mgr only tracks losses
                        consecutive_losses=getattr(self.risk_mgr, 'consecutive_losses', 0),
                        daily_trade_number=getattr(self.risk_mgr, 'daily_trades', 0),
                        daily_pnl_before=getattr(self.risk_mgr, 'daily_pnl', 0),
                        account_balance=getattr(self.risk_mgr, 'balance', None),
                        stop_loss=position.stop_loss,
                        take_profit=position.take_profit,
                        expected_edge=getattr(signal, 'expected_edge', 0),
                    )

    def monitor_positions(self):
        """Update all open positions with latest prices (bracket-aware)."""
        state_changed = False

        # 1. Poll IBKR for bracket SL/TP fills (server-side exits)
        if self.executor.is_ibkr_connected:
            for fill in self.executor.check_bracket_fills():
                pos = self._find_open_position(fill["symbol"])
                if pos:
                    status = PositionStatus.CLOSED_SL if fill["fill_type"] == "stop_loss" else PositionStatus.CLOSED_TP
                    self.risk_mgr._close_position(pos, fill["fill_price"], status)
                    logger.info(
                        f"[BRACKET] {fill['fill_type'].upper()} filled: {fill['symbol']} @ {fill['fill_price']:.2f}"
                    )
                    # Record trade close in knowledge collector
                    exit_reason = "sl" if fill["fill_type"] == "stop_loss" else "tp"
                    duration = int((datetime.now() - pos.entry_time).total_seconds()) if pos.entry_time else 0
                    self.knowledge.record_trade_close(
                        symbol=pos.symbol,
                        direction=pos.side.value,
                        entry_price=pos.entry_price,
                        exit_price=fill["fill_price"],
                        size=int(pos.size),
                        pnl=pos.pnl,
                        pnl_pct=pos.pnl_pct,
                        exit_reason=exit_reason,
                        duration_seconds=duration,
                        commission=pos.commission,
                        slippage=pos.slippage or 0,
                        mae_pct=pos.mae_pct or 0,
                        mfe_pct=pos.mfe_pct or 0,
                        strategy=pos.signal.strategy if pos.signal else "",
                        regime=pos.signal.regime.value if pos.signal else "",
                        signal_confidence=pos.signal.confidence if pos.signal else 0,
                        market_context=self._get_market_context(),
                    )
                    # Notify ML filter of trade completion
                    if self.ml_filter:
                        self.ml_filter.notify_trade_complete({"symbol": pos.symbol, "pnl": pos.pnl})
                    state_changed = True

        # 2. Update positions with latest prices
        # Max hold time in seconds: max_hold_candles * signal_timeframe
        # e.g., 24 candles * 5min = 7200s = 2 hours
        timeframe_map = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}
        candle_seconds = timeframe_map.get(self.config.strategy.signal_timeframe, 300)
        max_hold_seconds = self.config.risk.max_hold_candles * candle_seconds

        for pos in list(self.risk_mgr.positions):
            if pos.status != PositionStatus.OPEN:
                continue

            quote = self.collector.get_quote(pos.symbol)
            price_valid = False
            current_price = 0.0

            if quote:
                current_price = quote["current"]

                # GUARD 1: Delayed data can return 0
                if current_price <= 0:
                    logger.warning(f"[PRICE GUARD] {pos.symbol}: quote returned price=0, skipping price update")
                # GUARD 2: Price sanity — reject quotes that deviate >10% from entry
                elif pos.entry_price > 0:
                    deviation = abs(current_price - pos.entry_price) / pos.entry_price
                    if deviation > self.config.risk.max_price_deviation_pct:
                        logger.error(
                            f"[PRICE SANITY] {pos.symbol}: quote={current_price:.2f} deviates "
                            f"{deviation:.1%} from entry={pos.entry_price:.2f} — rejecting bad data"
                        )
                    else:
                        price_valid = True
                else:
                    price_valid = True

            if price_valid:
                self.risk_mgr.update_position(pos, current_price)

            # Time exit: use actual elapsed wall-clock time (not bars_held counter)
            # This is independent of price validity — time always passes
            if pos.status == PositionStatus.OPEN and pos.entry_time:
                # Handle tz-aware entry_time (pandas Timestamp from signal data may be tz-aware)
                entry_t = pos.entry_time
                if hasattr(entry_t, 'tzinfo') and entry_t.tzinfo is not None:
                    entry_t = entry_t.replace(tzinfo=None)
                elapsed = (datetime.now() - entry_t).total_seconds()
                if elapsed >= max_hold_seconds:
                    exit_price = current_price if price_valid and current_price > 0 else (
                        pos.current_price if pos.current_price > 0 else pos.entry_price
                    )
                    elapsed_min = elapsed / 60
                    logger.warning(
                        f"[TIME EXIT] {pos.symbol}: held {elapsed_min:.0f}min (max {max_hold_seconds/60:.0f}min) "
                        f"— forcing close at {exit_price:.2f}"
                    )
                    self.risk_mgr._close_position(pos, exit_price, PositionStatus.CLOSED_TIME)

            # If position was closed by risk manager (trailing/time exit)
            if pos.status != PositionStatus.OPEN:
                # Cancel remaining IBKR bracket legs before placing client exit
                if pos.has_bracket and pos.bracket_order_id:
                    self.executor.cancel_bracket_legs(pos.bracket_order_id)

                # Place closing order on IBKR — use MARKET order to guarantee fill
                close_type = SignalType.SHORT if pos.side == SignalType.LONG else SignalType.LONG
                # Use best available price: current > cached > entry (never 0)
                close_price = current_price if price_valid and current_price > 0 else (pos.current_price if pos.current_price > 0 else pos.entry_price)
                close_signal = Signal(
                    type=close_type,
                    symbol=pos.symbol,
                    price=close_price,
                    timestamp=datetime.now(),
                    confidence=1.0,
                    regime=pos.signal.regime if pos.signal else MarketRegime.RANGING,
                    strategy=pos.signal.strategy if pos.signal else "",
                    indicators={},
                    stop_loss=0,
                    take_profit=0,
                    reason=f"Close: {pos.status.value}",
                )
                self.executor.place_order(
                    close_signal, int(pos.size),
                    order_type=OrderType.MARKET,
                    require_confirmation=False,
                )
                # Record client-side close in knowledge collector
                exit_map = {
                    PositionStatus.CLOSED_SL: "sl",
                    PositionStatus.CLOSED_TP: "tp",
                    PositionStatus.CLOSED_TRAIL: "trail",
                    PositionStatus.CLOSED_TIME: "time",
                    PositionStatus.CLOSED_EOD: "eod",
                    PositionStatus.CLOSED_MANUAL: "manual",
                }
                duration = int((datetime.now() - pos.entry_time).total_seconds()) if pos.entry_time else 0
                self.knowledge.record_trade_close(
                    symbol=pos.symbol,
                    direction=pos.side.value,
                    entry_price=pos.entry_price,
                    exit_price=pos.exit_price or current_price,
                    size=int(pos.size),
                    pnl=pos.pnl,
                    pnl_pct=pos.pnl_pct,
                    exit_reason=exit_map.get(pos.status, "unknown"),
                    duration_seconds=duration,
                    commission=pos.commission,
                    slippage=pos.slippage or 0,
                    mae_pct=pos.mae_pct or 0,
                    mfe_pct=pos.mfe_pct or 0,
                    strategy=pos.signal.strategy if pos.signal else "",
                    regime=pos.signal.regime.value if pos.signal else "",
                    signal_confidence=pos.signal.confidence if pos.signal else 0,
                    market_context=self._get_market_context(),
                )
                # Notify ML filter of trade completion
                if self.ml_filter:
                    self.ml_filter.notify_trade_complete({"symbol": pos.symbol, "pnl": pos.pnl})
                state_changed = True

        # 3. Cancel stale unfilled orders to prevent duplicate fills
        stale_orders = self.executor.check_pending_orders()
        for order in stale_orders:
            logger.warning(f"[STALE] Cancelling {order.id} ({order.symbol}) — unfilled >60s")
            was_cancelled = self.executor.cancel_order(order.id)

            if not was_cancelled and order.status == OrderStatus.FILLED:
                # Race condition: IBKR filled the order before cancel was processed
                # We need to track this as an actual position
                logger.warning(
                    f"[STALE→FILLED] {order.symbol}: order was filled during cancel! "
                    f"qty={order.filled_quantity} @ {order.filled_price:.2f} — registering position"
                )
                # Create a signal from the order to register the position
                side = SignalType.LONG if order.side == "buy" else SignalType.SHORT
                stale_signal = Signal(
                    type=side,
                    symbol=order.symbol,
                    price=order.filled_price,
                    timestamp=datetime.now(),
                    confidence=0,
                    regime=MarketRegime.RANGING,
                    strategy="stale_fill",
                    indicators={},
                    stop_loss=order.stop_price if order.stop_price > 0 else (
                        order.filled_price * (1 - self.config.risk.stop_loss_pct) if side == SignalType.LONG
                        else order.filled_price * (1 + self.config.risk.stop_loss_pct)
                    ),
                    take_profit=0,  # Bracket legs handle TP
                    reason=f"Stale order filled during cancel race condition",
                )
                position = self.risk_mgr.open_position(stale_signal, size=order.filled_quantity)
                if position:
                    # CRITICAL: Do NOT set has_bracket=True here!
                    # The bracket SL/TP legs were just cancelled as part of the stale order cancel.
                    # has_bracket=False ensures client-side SL/TP monitoring protects this position.
                    position.has_bracket = False
                    position.bracket_order_id = ""
                    self.risk_mgr.save_state()
                    state_changed = True
            else:
                # Successfully cancelled — add to cooldown
                self._cancelled_symbols[order.symbol] = time.time()
                logger.info(f"[COOLDOWN] {order.symbol}: blocked for {self._cancel_cooldown_sec}s after stale cancel")

        # Clean up expired cooldowns
        now = time.time()
        expired = [sym for sym, t in self._cancelled_symbols.items() if (now - t) >= self._cancel_cooldown_sec]
        for sym in expired:
            del self._cancelled_symbols[sym]
            logger.debug(f"[COOLDOWN] {sym}: cooldown expired, new orders allowed")

        if state_changed:
            self.risk_mgr.save_state()

    def _find_open_position(self, symbol: str):
        """Find an open position by symbol (handles .DE suffix mapping)."""
        for pos in self.risk_mgr.positions:
            if pos.status == PositionStatus.OPEN:
                # Direct match or IBKR clean symbol match
                if pos.symbol == symbol or pos.symbol.replace(".DE", "") == symbol:
                    return pos
        return None

    def _is_market_open(self) -> bool:
        """Check if ANY market (EU or US) is currently within trading hours."""
        sessions = self._get_active_sessions()
        return len(sessions) > 0

    def _get_active_sessions(self) -> list[str]:
        """Return list of currently active market sessions: 'EU', 'US', or both."""
        now = datetime.now(ZoneInfo(self.config.schedule.timezone))

        # Weekend — no markets
        if now.weekday() >= 5:
            return []

        offset = self.config.schedule.trading_start_offset_min
        end_offset = self.config.schedule.trading_end_offset_min
        sessions = []

        # EU session (XETRA): 09:00-17:30 CET with offsets
        eu_open_h, eu_open_m = map(int, self.config.schedule.market_open.split(":"))
        eu_close_h, eu_close_m = map(int, self.config.schedule.market_close.split(":"))
        eu_open = now.replace(hour=eu_open_h, minute=eu_open_m, second=0, microsecond=0) + timedelta(minutes=offset)
        eu_close = now.replace(hour=eu_close_h, minute=eu_close_m, second=0, microsecond=0) - timedelta(minutes=end_offset)
        if eu_open <= now <= eu_close:
            sessions.append("EU")

        # US session: convert ET hours to CET (+6h)
        # US 09:30 ET = 15:30 CET, US 16:00 ET = 22:00 CET
        us_open_h, us_open_m = map(int, self.config.schedule.us_market_open.split(":"))
        us_close_h, us_close_m = map(int, self.config.schedule.us_market_close.split(":"))
        # Convert ET to CET: add 6 hours (CET = ET + 6 in winter, ET + 5 in summer)
        # More robust: use proper timezone conversion
        us_tz = ZoneInfo("America/New_York")
        now_et = datetime.now(us_tz)
        us_open = now_et.replace(hour=us_open_h, minute=us_open_m, second=0, microsecond=0) + timedelta(minutes=offset)
        us_close = now_et.replace(hour=us_close_h, minute=us_close_m, second=0, microsecond=0) - timedelta(minutes=end_offset)
        if us_open <= now_et <= us_close:
            sessions.append("US")

        return sessions

    @staticmethod
    def _is_eu_symbol(symbol: str) -> bool:
        """Check if a symbol trades on EU exchanges (.DE suffix)."""
        return symbol.endswith(".DE")

    def _get_active_symbols(self) -> list[str]:
        """Return symbols whose market is currently open."""
        sessions = self._get_active_sessions()
        if not sessions:
            return []

        all_symbols = self.config.watchlist.symbols
        active = []
        for sym in all_symbols:
            if self._is_eu_symbol(sym) and "EU" in sessions:
                active.append(sym)
            elif not self._is_eu_symbol(sym) and "US" in sessions:
                active.append(sym)
        return active

    def _reconcile_positions(self):
        """
        Compare internal position tracking with IBKR portfolio and AUTO-FIX discrepancies.

        Handles:
        1. Orphan positions in IBKR (from race conditions) → auto-flatten with MARKET order
        2. Ghost positions internal but not in IBKR (bracket filled) → close internally
        3. Size mismatches → update internal size to match IBKR
        """
        if not self.executor.is_ibkr_connected:
            return

        ibkr_portfolio = self.executor.get_portfolio()
        internal_open = {p.symbol: p for p in self.risk_mgr.positions if p.status == PositionStatus.OPEN}

        # Map IBKR clean symbols to our .DE suffixed symbols
        ibkr_mapped = {}
        for sym, data in ibkr_portfolio.items():
            de_sym = f"{sym}.DE"
            if de_sym in internal_open or any(s.endswith(".DE") for s in self.config.watchlist.symbols if s.replace(".DE", "") == sym):
                ibkr_mapped[de_sym] = data
            else:
                ibkr_mapped[sym] = data

        # IBKR has positions we don't track (orphans from race condition) → AUTO-FLATTEN
        for sym, data in ibkr_mapped.items():
            if sym not in internal_open and data["quantity"] != 0:
                qty = data["quantity"]
                logger.warning(f"[RECONCILE] Orphan in IBKR: {sym} qty={qty} — AUTO-FLATTENING")
                try:
                    from ib_async import MarketOrder
                    clean_sym = sym.replace(".DE", "")
                    contract = self.executor._get_contract(sym)
                    if contract is None:
                        logger.error(f"[RECONCILE] Cannot flatten orphan {sym}: no valid contract")
                        continue
                    # Opposite side to flatten
                    action = "SELL" if qty > 0 else "BUY"
                    abs_qty = abs(qty)
                    order = MarketOrder(action, abs_qty)
                    order.tif = "DAY"
                    trade = self.executor._ib.placeOrder(contract, order)
                    self.executor._ib.sleep(3)
                    fill_status = trade.orderStatus.status
                    fill_price = trade.orderStatus.avgFillPrice or 0
                    logger.info(f"[RECONCILE] Orphan {sym}: {action} {abs_qty} → {fill_status} @ {fill_price:.2f}")
                    # Also add to cooldown to prevent re-entry
                    self._cancelled_symbols[sym] = time.time()
                except Exception as e:
                    logger.error(f"[RECONCILE] Failed to flatten orphan {sym}: {e}")

        # We track positions not in IBKR (ghost — may have been closed by bracket)
        for sym, pos in internal_open.items():
            clean_sym = sym.replace(".DE", "")
            if sym not in ibkr_mapped and clean_sym not in ibkr_portfolio:
                logger.warning(f"[RECONCILE] Ghost position: {sym} — closing internally")
                # Use current_price if valid, otherwise entry_price (never 0)
                ghost_price = pos.current_price if pos.current_price > 0 else pos.entry_price
                self.risk_mgr._close_position(pos, ghost_price, PositionStatus.CLOSED_MANUAL)

        # Size mismatches — update internal to match IBKR (IBKR is source of truth)
        # NOTE: Skip positions already closed by the ghost handler above
        for sym, pos in internal_open.items():
            if pos.status != PositionStatus.OPEN:
                continue  # Was closed by ghost handler — don't modify
            if sym in ibkr_mapped:
                ibkr_qty = abs(ibkr_mapped[sym]["quantity"])
                if pos.size != ibkr_qty:
                    logger.warning(
                        f"[RECONCILE] Size mismatch: {sym} internal={pos.size} IBKR={ibkr_qty} — updating internal"
                    )
                    pos.size = ibkr_qty

    def run_session(self):
        """
        Run a full trading session.
        Loops: fetch data → scan → execute → monitor → repeat
        """
        self.startup()
        self._running = True

        # Handle graceful shutdown
        def shutdown_handler(signum, frame):
            logger.info("\nShutdown signal received. Closing positions...")
            self._running = False

        sig.signal(sig.SIGINT, shutdown_handler)
        sig.signal(sig.SIGTERM, shutdown_handler)

        scan_interval = 300  # 5 minutes between scans
        last_scan = 0
        last_reconcile = 0
        reconcile_interval = 120  # Reconcile every 2 minutes (critical for orphan detection)

        # Faster loop for paper mode (10s vs 30s for live)
        loop_sleep = 10 if not self.executor.is_ibkr_connected else 30

        try:
            while self._running:
                now = time.time()

                # Periodic scan (only during active market sessions)
                if now - last_scan >= scan_interval:
                    active_symbols = self._get_active_symbols()
                    sessions = self._get_active_sessions()

                    if active_symbols:
                        # Refresh sentiment data
                        self.filters.sentiment.refresh()
                        status = self.filters.get_status()
                        sent = status["sentiment"]

                        # Log market mood with active sessions
                        session_str = "+".join(sessions)
                        if sent["vix_level"] is not None:
                            mood_msg = f"Market mood [{session_str}] — VIX: {sent['vix_level']:.1f}"
                            if sent["dax_change_pct"] is not None:
                                mood_msg += f", DAX: {sent['dax_change_pct']:+.2%}"
                            logger.info(mood_msg)
                        else:
                            logger.info(f"Active sessions: {session_str} ({len(active_symbols)} symbols)")

                        self.fetch_all_data(symbols=active_symbols)
                        signals = self.scan_signals()

                        # Record scan cycle in knowledge collector
                        scan_ctx = self._get_market_context()
                        self.knowledge.record_scan_cycle(
                            symbols_scanned=len(self._data_cache),
                            signals_found=len(signals),
                            market_context=scan_ctx,
                        )

                        if signals:
                            self.execute_signals(signals)

                        # Pairs trading scan (parallel to directional V2)
                        if self.pairs_trader and self._data_cache:
                            try:
                                pair_signals = self.pairs_trader.scan_pairs(self._data_cache)
                                if pair_signals:
                                    self._execute_pair_signals(pair_signals)
                                # Also monitor existing pairs positions
                                closed_pairs = self.pairs_trader.update_positions(
                                    self._data_cache, self.collector, self.executor
                                )
                                for cp in closed_pairs:
                                    logger.info(
                                        f"[PAIRS] Closed: {cp.pair[0]}/{cp.pair[1]} "
                                        f"PnL=€{cp.pnl:+.2f} reason={cp.exit_reason}"
                                    )
                            except Exception as e:
                                logger.warning(f"[PAIRS] Scan/monitor error: {e}")

                        # Check if learning should trigger
                        stats = self.risk_mgr.get_stats()
                        should_learn, trigger = self.learner.should_learn(stats)
                        if should_learn:
                            self._run_learning(trigger)
                    else:
                        logger.debug("All markets closed — skipping scan.")

                    last_scan = now

                # Monitor positions (always, even outside market hours for EOD)
                self.monitor_positions()

                # Periodic IBKR reconciliation
                if self.executor.is_ibkr_connected and now - last_reconcile >= reconcile_interval:
                    self._reconcile_positions()
                    last_reconcile = now

                # IBKR connection heartbeat (reconnect if dropped)
                if self.executor.is_ibkr_connected:
                    if not self.executor.check_connection():
                        self.skill_updater.log_error("IBKR connection lost — reconnecting")
                        # Re-share connection if reconnected
                        if self.executor.is_ibkr_connected:
                            self.collector.set_ibkr_connection(self.executor.ib_connection)
                            self.filters.set_ibkr_connection(self.executor.ib_connection)

                # Auto-shutdown after all markets close (US is last: ~22:00 CET)
                if not self._get_active_sessions():
                    open_positions = [p for p in self.risk_mgr.positions if p.status == PositionStatus.OPEN]
                    if not open_positions:
                        # Only shutdown if we actually traded (avoid shutdown before market opens)
                        now_local = datetime.now(ZoneInfo(self.config.schedule.timezone))
                        us_close_h, us_close_m = map(int, self.config.schedule.us_market_close.split(":"))
                        # US 16:00 ET ≈ 22:00 CET — if past that, markets are done for the day
                        us_tz = ZoneInfo("America/New_York")
                        now_et = datetime.now(us_tz)
                        us_close = now_et.replace(hour=us_close_h, minute=us_close_m, second=0, microsecond=0)
                        if now_et > us_close:
                            logger.info("All markets closed and no open positions — auto-shutdown.")
                            self._running = False

                # Auto-update skill.md every 60 minutes
                if self.skill_updater.should_update():
                    self.skill_updater.update()

                time.sleep(loop_sleep)

        finally:
            self._end_of_day()

    def _run_learning(self, trigger: str):
        """Run self-learning optimization cycle."""
        logger.info(f"Self-learning triggered: {trigger}")

        stats = self.risk_mgr.get_stats()

        # Use the most liquid symbol's data for optimization
        if self._data_cache:
            symbol = list(self._data_cache.keys())[0]
            df = self._data_cache[symbol]
            report = self.learner.optimize(df, symbol, stats, trigger)

            if report.improvement.get("score_delta", 0) > 0.05:
                logger.info("Applying optimized parameters...")
                self.learner.apply_parameters(report.best_parameters, self.config)
                report.applied = True
                # Recreate strategy with new params
                self.strategy = StrategyV2(self.config.strategy, self.config.risk)

    def _end_of_day(self):
        """End-of-day procedures."""
        logger.info("\n" + "=" * 60)
        logger.info("END OF DAY")
        logger.info("=" * 60)

        open_positions = [p for p in self.risk_mgr.positions if p.status == PositionStatus.OPEN]

        # Cancel any remaining bracket legs before closing
        for pos in open_positions:
            if pos.has_bracket and pos.bracket_order_id:
                self.executor.cancel_bracket_legs(pos.bracket_order_id)

        # Place actual IBKR MARKET orders to close all positions (not just internal tracking)
        if open_positions:
            current_prices = {}
            for pos in open_positions:
                quote = self.collector.get_quote(pos.symbol)
                if quote and quote["current"] > 0:
                    current_prices[pos.symbol] = quote["current"]
                else:
                    logger.warning(f"[EOD] No valid price for {pos.symbol} — will use cached price")

                # Place IBKR MARKET order to actually close the position on the exchange
                if self.executor.is_ibkr_connected:
                    close_type = SignalType.SHORT if pos.side == SignalType.LONG else SignalType.LONG
                    close_price = current_prices.get(pos.symbol, pos.current_price if pos.current_price > 0 else pos.entry_price)
                    close_signal = Signal(
                        type=close_type,
                        symbol=pos.symbol,
                        price=close_price,
                        timestamp=datetime.now(),
                        confidence=1.0,
                        regime=pos.signal.regime if pos.signal else MarketRegime.RANGING,
                        strategy=pos.signal.strategy if pos.signal else "",
                        indicators={},
                        stop_loss=0,
                        take_profit=0,
                        reason=f"EOD close: {pos.symbol}",
                    )
                    eod_order = self.executor.place_order(
                        close_signal, int(pos.size),
                        order_type=OrderType.MARKET,
                        require_confirmation=False,
                    )
                    if eod_order and eod_order.status == OrderStatus.FILLED:
                        logger.info(f"[EOD] IBKR close filled: {pos.symbol} {int(pos.size)} shares @ {eod_order.filled_price:.2f}")
                    elif eod_order:
                        logger.error(f"[EOD] IBKR close order NOT filled for {pos.symbol}: status={eod_order.status.value}")
                    else:
                        logger.error(f"[EOD] Failed to place IBKR close order for {pos.symbol}")

            # Now update internal tracking (risk manager accounting)
            self.risk_mgr.close_all_positions(current_prices)

        # Print summary
        self.risk_mgr.print_summary()

        # Run knowledge pattern analysis (accumulates over days)
        self.knowledge.analyze_patterns(lookback_days=30)
        session = self.knowledge.get_session_summary()
        logger.info(
            f"[Knowledge] Session: {session['signals_recorded']} signals, "
            f"{session['trades_closed']} trades closed, "
            f"P&L: €{session['session_pnl']:.2f}"
        )
        insights = self.knowledge.get_insights()
        if insights:
            logger.info("[Knowledge] Insights:")
            for insight in insights:
                logger.info(f"  → {insight}")

        # Institutional analytics report
        if self.risk_mgr.closed_positions:
            report = Analytics.generate_report(
                self.risk_mgr.closed_positions,
                initial_balance=self.config.risk.initial_balance,
            )
            logger.info(Analytics.format_report(report))

        # DB Self-Learner: regenerate insights at end of day (with today's data included)
        if self.db_learner:
            try:
                eod_insights = self.db_learner.generate_insights()
                if eod_insights:
                    logger.info("[DBLearner] EOD insights regenerated for tomorrow's session")
                    summary = self.db_learner.format_insights_summary()
                    for line in summary.split("\n"):
                        logger.info(line)
            except Exception as e:
                logger.warning(f"[DBLearner] EOD insight generation failed: {e}")

        # Only clear state if ALL positions were closed successfully.
        # If some positions remain open (e.g., price guard blocked closing),
        # SAVE state so they're tracked on next startup.
        remaining_open = [p for p in self.risk_mgr.positions if p.status == PositionStatus.OPEN]
        if remaining_open:
            logger.error(
                f"[EOD] ⚠ {len(remaining_open)} positions could NOT be closed! "
                f"Symbols: {[p.symbol for p in remaining_open]}. "
                f"State saved for next session — MANUALLY CHECK IBKR!"
            )
            self.risk_mgr.save_state()
        else:
            self.risk_mgr.clear_state()

        # Final skill.md update
        self.skill_updater.update(force=True)

        # Disconnect
        self.executor.disconnect()
        self.collector.stop_live_feed()

    def run_backtest(self, symbol: Optional[str] = None, days: Optional[int] = None):
        """Run backtest on historical data."""
        symbols = [symbol] if symbol else self.config.watchlist.symbols[:5]
        bt_days = days or self.config.data.backtest_days

        logger.info(f"Running backtest: {len(symbols)} symbols, {bt_days} days")

        for sym in symbols:
            logger.info(f"\n{'─' * 40}")
            logger.info(f"Backtesting: {sym}")
            logger.info(f"{'─' * 40}")

            df = self.collector.get_historical(sym, interval="5m", days=bt_days)
            if df is None or df.empty:
                logger.warning(f"No data for {sym}")
                continue

            result = self.backtester.run(df, sym)

            # Print results
            stats = result.stats
            if stats.get("total_trades", 0) > 0:
                logger.info(f"  Trades:        {stats['total_trades']}")
                logger.info(f"  Win Rate:      {stats['win_rate']:.1%}")
                logger.info(f"  Total PnL:     {stats['total_pnl']:+.2f}")
                logger.info(f"  Profit Factor: {stats['profit_factor']:.2f}")
                logger.info(f"  Max Drawdown:  {stats['max_drawdown']:.2%}")
                logger.info(f"  Sharpe Ratio:  {stats.get('sharpe_ratio', 0):.2f}")
            else:
                logger.info("  No trades generated")

            # Save results
            self.backtester.save_results(result, sym)

    def force_learn(self, symbol: Optional[str] = None):
        """Force a learning/optimization cycle."""
        sym = symbol or self.config.watchlist.symbols[0]
        logger.info(f"Force learning on {sym}")

        df = self.collector.get_historical(sym, interval="5m", days=60)
        if df is None or df.empty:
            logger.error("No data available for learning")
            return

        stats = self.risk_mgr.get_stats()
        report = self.learner.optimize(df, sym, stats, trigger="manual")

        logger.info(f"\nRecommendation: {report.recommendation}")
        if report.best_parameters:
            logger.info(f"Best parameters: {json.dumps(report.best_parameters, indent=2)}")


def main():
    parser = argparse.ArgumentParser(description="Trading System — Interactive Brokers")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--learn", action="store_true", help="Force learning cycle")
    parser.add_argument("--scan", action="store_true", help="One-shot signal scan")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    parser.add_argument("--symbol", type=str, help="Specific symbol to trade/test")
    parser.add_argument("--days", type=int, default=60, help="Days of history for backtest")

    args = parser.parse_args()

    # Override config based on args
    if args.live:
        config.ibkr.paper_trading = False

    system = TradingSystem(config)

    if args.backtest:
        system.startup()
        system.run_backtest(symbol=args.symbol, days=args.days)
    elif args.learn:
        system.startup()
        system.force_learn(symbol=args.symbol)
    elif args.scan:
        system.startup()
        system.fetch_all_data()
        signals = system.scan_signals()
        if not signals:
            print("\nNo signals at this time.")
    else:
        # Full trading session
        system.run_session()


if __name__ == "__main__":
    main()
