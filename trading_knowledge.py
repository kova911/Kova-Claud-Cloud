"""
Trading Knowledge Collector — Persistent Learning Engine (v1.0)

Collects extensive signal, trade, and market condition data every scan cycle.
Builds a persistent knowledge base that accumulates insights over days/weeks/months.

Architecture:
    - Every signal (fired, filtered, executed, skipped) is recorded
    - Every trade outcome (P&L, duration, SL/TP hit, slippage) is tracked
    - Market conditions at signal time are snapshotted (VIX, DAX, regime, indicators)
    - Pattern analysis runs periodically to discover what works and what doesn't
    - knowledge_base.md is auto-updated with human-readable accumulated learnings

Storage:
    knowledge/
        signals_YYYYMMDD.jsonl    — Daily signal log (one JSON per line)
        trades_YYYYMMDD.jsonl     — Daily trade outcomes
        patterns.json             — Discovered patterns and statistics
        knowledge_base.md         — Auto-generated human-readable insights
"""

import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path("knowledge")
KNOWLEDGE_DIR.mkdir(exist_ok=True)


class TradingKnowledge:
    """
    Persistent learning engine that records every signal and trade,
    then mines patterns from the accumulated data.

    Dual-write: JSONL (human-readable, append-only) + SQLite (structured queries).
    """

    def __init__(self, config):
        self.config = config
        self._today = date.today()
        self._signal_file = KNOWLEDGE_DIR / f"signals_{self._today:%Y%m%d}.jsonl"
        self._trade_file = KNOWLEDGE_DIR / f"trades_{self._today:%Y%m%d}.jsonl"
        self._patterns_file = KNOWLEDGE_DIR / "patterns.json"
        self._knowledge_file = KNOWLEDGE_DIR / "knowledge_base.md"
        self._session_signals: list[dict] = []
        self._session_trades: list[dict] = []
        self._patterns: dict = self._load_patterns()

        # SQLite database (dual-write alongside JSONL)
        self.db = None
        try:
            from trade_database import TradeDatabase
            self.db = TradeDatabase()
            logger.info("[Knowledge] SQLite database initialized (dual-write mode)")
        except Exception as e:
            logger.warning(f"[Knowledge] SQLite database unavailable, JSONL-only mode: {e}")

    # ─── Signal Recording ─────────────────────────────────────────

    def record_signal(
        self,
        signal,
        market_context: dict,
        action: str,                   # "executed", "filtered", "skipped", "no_signal"
        filter_reason: str = "",
        # Renaissance context (optional, backward compatible)
        ml_result: Optional[tuple] = None,            # (allowed, win_prob, reason)
        hmm_probs: Optional[dict] = None,             # {"low_vol": .., "normal": .., "high_vol": ..}
        additional_indicators: Optional[dict] = None,  # {"atr": .., "rsi": .., "adx": ..}
        corr_data: Optional[dict] = None,             # {"score": .., "zscore": ..}
    ):
        """
        Record every signal generated during a scan cycle.

        Args:
            signal: Signal dataclass from strategy.py
            market_context: dict with VIX, DAX%, S&P%, regime info
            action: what happened to this signal
            filter_reason: why it was filtered/skipped (if applicable)
            ml_result: (allowed, win_prob, reason) from ML filter
            hmm_probs: HMM regime probabilities
            additional_indicators: ATR/RSI/ADX/BB/volume at signal time
            corr_data: cross-asset correlation data
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "date": str(self._today),
            "day_of_week": self._today.strftime("%A"),
            "hour": datetime.now().hour,
            "minute": datetime.now().minute,
            # Signal data
            "symbol": signal.symbol,
            "direction": signal.type.value,
            "price": signal.price,
            "confidence": signal.confidence,
            "strategy": signal.strategy,
            "regime": signal.regime.value,
            "reason": signal.reason,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
            # Indicators at signal time
            "indicators": self._clean_indicators(signal.indicators),
            # Market context
            "vix": market_context.get("vix"),
            "vix_change": market_context.get("vix_change"),
            "dax_pct": market_context.get("dax_pct"),
            "sp500_pct": market_context.get("sp500_pct"),
            # Action taken
            "action": action,
            "filter_reason": filter_reason,
        }

        # Add Renaissance context to JSONL if provided
        if ml_result:
            entry["ml_allowed"] = ml_result[0]
            entry["ml_win_prob"] = ml_result[1]
            entry["ml_reason"] = ml_result[2] if len(ml_result) > 2 else ""
        if hmm_probs:
            entry["hmm_probs"] = hmm_probs
        if additional_indicators:
            entry["additional_indicators"] = additional_indicators
        if corr_data:
            entry["corr_data"] = corr_data

        self._session_signals.append(entry)
        self._append_jsonl(self._signal_file, entry)

        # Dual-write to SQLite
        if self.db:
            try:
                self.db.insert_signal(
                    signal, market_context, action, filter_reason,
                    ml_result=ml_result,
                    hmm_probs=hmm_probs,
                    additional_indicators=additional_indicators,
                    corr_data=corr_data,
                )
            except Exception as e:
                logger.debug(f"[Knowledge] DB signal write failed: {e}")

    def record_scan_cycle(self, symbols_scanned: int, signals_found: int, market_context: dict):
        """Record metadata about a scan cycle (even when no signals fire)."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "scan_cycle",
            "symbols_scanned": symbols_scanned,
            "signals_found": signals_found,
            "vix": market_context.get("vix"),
            "dax_pct": market_context.get("dax_pct"),
            "sp500_pct": market_context.get("sp500_pct"),
        }
        self._append_jsonl(self._signal_file, entry)

        # Dual-write to SQLite market_snapshots
        if self.db:
            try:
                self.db.insert_market_snapshot(market_context, symbols_scanned, signals_found)
            except Exception as e:
                logger.debug(f"[Knowledge] DB snapshot write failed: {e}")

    # ─── Trade Outcome Recording ──────────────────────────────────

    def record_trade_open(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        size: int,
        signal_confidence: float,
        strategy: str,
        regime: str,
        market_context: dict,
        order_id: str = "",
        tick_size: float = 0,
        slippage: float = 0,
        is_bracket: bool = False,
        # Renaissance context (optional, backward compatible)
        signal_components: Optional[dict] = None,      # signal.indicators (all 7 components)
        hmm_probs: Optional[dict] = None,              # {"low_vol": .., "normal": .., "high_vol": ..}
        ml_win_prob: Optional[float] = None,           # ML filter win probability
        corr_data: Optional[dict] = None,              # cross-asset correlation data
        additional_indicators: Optional[dict] = None,   # ATR/RSI/ADX/BB/volume snapshot
        # Sizing context
        size_base: Optional[int] = None,               # position size before VIX/HMM reductions
        vix_mult: float = 1.0,                         # VIX tier multiplier applied
        hmm_mult: float = 1.0,                         # HMM high-vol multiplier applied
        # Running stats
        consecutive_wins: int = 0,
        consecutive_losses: int = 0,
        daily_trade_number: int = 0,
        daily_pnl_before: float = 0,
        account_balance: Optional[float] = None,
        # Levels
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        expected_edge: float = 0,
    ):
        """Record a trade entry with full context for learning."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "trade_open",
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "size": size,
            "signal_confidence": signal_confidence,
            "strategy": strategy,
            "regime": regime,
            "order_id": order_id,
            "tick_size": tick_size,
            "slippage": slippage,
            "is_bracket": is_bracket,
            "vix": market_context.get("vix"),
            "dax_pct": market_context.get("dax_pct"),
            "hour": datetime.now().hour,
            "day_of_week": self._today.strftime("%A"),
        }

        # Add Renaissance context to JSONL if provided
        if signal_components:
            entry["signal_components"] = signal_components
        if hmm_probs:
            entry["hmm_probs"] = hmm_probs
        if ml_win_prob is not None:
            entry["ml_win_prob"] = ml_win_prob
        if corr_data:
            entry["corr_data"] = corr_data
        if additional_indicators:
            entry["additional_indicators"] = additional_indicators
        if size_base is not None:
            entry["size_base"] = size_base
            entry["vix_mult"] = vix_mult
            entry["hmm_mult"] = hmm_mult
        if stop_loss is not None:
            entry["stop_loss"] = stop_loss
        if take_profit is not None:
            entry["take_profit"] = take_profit

        self._session_trades.append(entry)
        self._append_jsonl(self._trade_file, entry)

        # Dual-write to SQLite
        if self.db:
            try:
                self.db.insert_trade_open(
                    symbol=symbol, direction=direction,
                    entry_price=entry_price, size=size,
                    signal_confidence=signal_confidence,
                    strategy=strategy, regime=regime,
                    market_context=market_context,
                    order_id=order_id, tick_size=tick_size,
                    slippage=slippage, is_bracket=is_bracket,
                    signal_components=signal_components,
                    hmm_probs=hmm_probs, ml_win_prob=ml_win_prob,
                    corr_data=corr_data,
                    additional_indicators=additional_indicators,
                    size_base=size_base, vix_mult=vix_mult, hmm_mult=hmm_mult,
                    consecutive_wins=consecutive_wins,
                    consecutive_losses=consecutive_losses,
                    daily_trade_number=daily_trade_number,
                    daily_pnl_before=daily_pnl_before,
                    account_balance=account_balance,
                    stop_loss=stop_loss, take_profit=take_profit,
                    expected_edge=expected_edge,
                )
            except Exception as e:
                logger.debug(f"[Knowledge] DB trade_open write failed: {e}")

    def record_trade_close(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        size: int,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,              # "tp", "sl", "trail", "time", "eod", "manual"
        duration_seconds: int,
        commission: float = 0,
        slippage: float = 0,
        mae_pct: float = 0,           # Max adverse excursion
        mfe_pct: float = 0,           # Max favorable excursion
        strategy: str = "",
        regime: str = "",
        signal_confidence: float = 0,
        market_context: Optional[dict] = None,
    ):
        """Record a trade exit with full outcome data."""
        ctx = market_context or {}
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "trade_close",
            "date": str(self._today),
            "day_of_week": self._today.strftime("%A"),
            "hour": datetime.now().hour,
            # Trade data
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "exit_reason": exit_reason,
            "duration_seconds": duration_seconds,
            "duration_minutes": round(duration_seconds / 60, 1),
            "commission": commission,
            "slippage": slippage,
            # Quality metrics
            "mae_pct": mae_pct,
            "mfe_pct": mfe_pct,
            "edge_captured": round(mfe_pct / max(abs(pnl_pct), 0.0001), 2) if mfe_pct else 0,
            # Strategy
            "strategy": strategy,
            "regime": regime,
            "signal_confidence": signal_confidence,
            # Market context at exit
            "vix_at_exit": ctx.get("vix"),
            "dax_pct_at_exit": ctx.get("dax_pct"),
        }

        self._session_trades.append(entry)
        self._append_jsonl(self._trade_file, entry)

        # Update running patterns after each trade close
        self._update_patterns(entry)

        # Dual-write close to SQLite
        if self.db:
            try:
                # Find the open trade in DB by symbol + direction
                trade_id = self.db.find_open_trade(symbol, direction)
                if trade_id:
                    self.db.update_trade_close(
                        trade_id=trade_id,
                        exit_price=exit_price,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        duration_seconds=duration_seconds,
                        commission=commission,
                        mae_pct=mae_pct,
                        mfe_pct=mfe_pct,
                        market_context_exit=ctx,
                    )
                else:
                    logger.debug(f"[Knowledge] DB: no open trade found for {symbol}/{direction}")
            except Exception as e:
                logger.debug(f"[Knowledge] DB trade_close write failed: {e}")

    # ─── Pattern Analysis ─────────────────────────────────────────

    def analyze_patterns(self, lookback_days: int = 30):
        """
        Analyze all recorded data to discover patterns.
        Called periodically (e.g., end of day) or on demand.
        """
        trades = self._load_all_trades(lookback_days)
        signals = self._load_all_signals(lookback_days)

        if not trades:
            logger.info("[Knowledge] No trades to analyze yet.")
            return []

        closed_trades = [t for t in trades if t.get("type") == "trade_close"]
        if not closed_trades:
            return []

        patterns = {
            "last_updated": datetime.now().isoformat(),
            "lookback_days": lookback_days,
            "total_trades_analyzed": len(closed_trades),
            "total_signals_analyzed": len(signals),
        }

        # ── Win Rate by Strategy ──
        patterns["by_strategy"] = self._analyze_by_field(closed_trades, "strategy")

        # ── Win Rate by Symbol ──
        patterns["by_symbol"] = self._analyze_by_field(closed_trades, "symbol")

        # ── Win Rate by Direction ──
        patterns["by_direction"] = self._analyze_by_field(closed_trades, "direction")

        # ── Win Rate by Confidence Level ──
        patterns["by_confidence"] = self._analyze_by_confidence(closed_trades)

        # ── Win Rate by Market Regime ──
        patterns["by_regime"] = self._analyze_by_field(closed_trades, "regime")

        # ── Win Rate by Exit Reason ──
        patterns["by_exit_reason"] = self._analyze_by_field(closed_trades, "exit_reason")

        # ── Win Rate by Hour of Day ──
        patterns["by_hour"] = self._analyze_by_field(closed_trades, "hour")

        # ── Win Rate by Day of Week ──
        patterns["by_day"] = self._analyze_by_field(closed_trades, "day_of_week")

        # ── VIX Impact Analysis ──
        patterns["by_vix_range"] = self._analyze_by_vix(closed_trades)

        # ── Duration Analysis ──
        patterns["duration_stats"] = self._analyze_duration(closed_trades)

        # ── Signal Quality (signals that led to trades vs filtered) ──
        patterns["signal_quality"] = self._analyze_signal_quality(signals)

        # ── Slippage Analysis ──
        patterns["slippage_stats"] = self._analyze_slippage(closed_trades)

        # ── Best/Worst Performers ──
        patterns["best_symbols"] = self._rank_symbols(closed_trades, top=True)
        patterns["worst_symbols"] = self._rank_symbols(closed_trades, top=False)

        # ── Streak Analysis ──
        patterns["streaks"] = self._analyze_streaks(closed_trades)

        # ── Key Insights (auto-generated) ──
        patterns["insights"] = self._generate_insights(patterns, closed_trades)

        self._patterns = patterns
        self._save_patterns(patterns)
        self._generate_knowledge_base(patterns)

        logger.info(
            f"[Knowledge] Pattern analysis complete: {len(closed_trades)} trades, "
            f"{len(patterns.get('insights', []))} insights generated"
        )

        return patterns.get("insights", [])

    def _analyze_by_field(self, trades: list[dict], field: str) -> dict:
        """Compute win rate, avg P&L, count grouped by a field."""
        groups = defaultdict(list)
        for t in trades:
            key = str(t.get(field, "unknown"))
            groups[key].append(t)

        result = {}
        for key, group in groups.items():
            wins = [t for t in group if t["pnl"] > 0]
            losses = [t for t in group if t["pnl"] <= 0]
            total_pnl = sum(t["pnl"] for t in group)
            avg_pnl = total_pnl / len(group) if group else 0
            avg_winner = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
            avg_loser = sum(t["pnl"] for t in losses) / len(losses) if losses else 0

            result[key] = {
                "count": len(group),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(len(wins) / len(group), 3) if group else 0,
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(avg_pnl, 2),
                "avg_winner": round(avg_winner, 2),
                "avg_loser": round(avg_loser, 2),
                "profit_factor": round(
                    abs(sum(t["pnl"] for t in wins)) / abs(sum(t["pnl"] for t in losses)), 2
                ) if losses and sum(t["pnl"] for t in losses) != 0 else float("inf"),
            }

        return result

    def _analyze_by_confidence(self, trades: list[dict]) -> dict:
        """Group trades by confidence buckets."""
        buckets = {"low_40-59": [], "medium_60-79": [], "high_80-100": []}
        for t in trades:
            conf = t.get("signal_confidence", 0)
            if conf >= 0.8:
                buckets["high_80-100"].append(t)
            elif conf >= 0.6:
                buckets["medium_60-79"].append(t)
            else:
                buckets["low_40-59"].append(t)

        result = {}
        for bucket, group in buckets.items():
            if not group:
                continue
            wins = [t for t in group if t["pnl"] > 0]
            result[bucket] = {
                "count": len(group),
                "win_rate": round(len(wins) / len(group), 3),
                "avg_pnl": round(sum(t["pnl"] for t in group) / len(group), 2),
                "total_pnl": round(sum(t["pnl"] for t in group), 2),
            }
        return result

    def _analyze_by_vix(self, trades: list[dict]) -> dict:
        """Analyze performance by VIX range at entry."""
        ranges = {"low_0-15": [], "normal_15-20": [], "elevated_20-25": [],
                  "high_25-30": [], "extreme_30+": []}
        for t in trades:
            vix = t.get("vix_at_exit") or 0
            if vix >= 30:
                ranges["extreme_30+"].append(t)
            elif vix >= 25:
                ranges["high_25-30"].append(t)
            elif vix >= 20:
                ranges["elevated_20-25"].append(t)
            elif vix >= 15:
                ranges["normal_15-20"].append(t)
            else:
                ranges["low_0-15"].append(t)

        result = {}
        for rng, group in ranges.items():
            if not group:
                continue
            wins = [t for t in group if t["pnl"] > 0]
            result[rng] = {
                "count": len(group),
                "win_rate": round(len(wins) / len(group), 3),
                "avg_pnl": round(sum(t["pnl"] for t in group) / len(group), 2),
            }
        return result

    def _analyze_duration(self, trades: list[dict]) -> dict:
        """Analyze trade duration statistics."""
        durations = [t.get("duration_minutes", 0) for t in trades]
        winners = [t.get("duration_minutes", 0) for t in trades if t["pnl"] > 0]
        losers = [t.get("duration_minutes", 0) for t in trades if t["pnl"] <= 0]

        return {
            "avg_duration_min": round(sum(durations) / len(durations), 1) if durations else 0,
            "avg_winner_duration_min": round(sum(winners) / len(winners), 1) if winners else 0,
            "avg_loser_duration_min": round(sum(losers) / len(losers), 1) if losers else 0,
            "shortest_min": round(min(durations), 1) if durations else 0,
            "longest_min": round(max(durations), 1) if durations else 0,
        }

    def _analyze_signal_quality(self, signals: list[dict]) -> dict:
        """Analyze signal generation quality."""
        if not signals:
            return {}

        # Filter out scan_cycle metadata entries
        actual_signals = [s for s in signals if s.get("type") != "scan_cycle"]
        if not actual_signals:
            return {}

        by_action = defaultdict(int)
        by_symbol = defaultdict(lambda: defaultdict(int))
        by_strategy = defaultdict(lambda: defaultdict(int))

        for s in actual_signals:
            action = s.get("action", "unknown")
            by_action[action] += 1
            by_symbol[s.get("symbol", "?")][action] += 1
            by_strategy[s.get("strategy", "?")][action] += 1

        return {
            "total_signals": len(actual_signals),
            "by_action": dict(by_action),
            "execution_rate": round(
                by_action.get("executed", 0) / len(actual_signals), 3
            ) if actual_signals else 0,
            "filter_rate": round(
                by_action.get("filtered", 0) / len(actual_signals), 3
            ) if actual_signals else 0,
        }

    def _analyze_slippage(self, trades: list[dict]) -> dict:
        """Analyze slippage across trades."""
        slippages = [t.get("slippage", 0) for t in trades if t.get("slippage") is not None]
        if not slippages:
            return {}

        return {
            "avg_slippage": round(sum(slippages) / len(slippages), 4),
            "max_adverse_slippage": round(min(slippages), 4),
            "max_favorable_slippage": round(max(slippages), 4),
            "zero_slippage_pct": round(
                len([s for s in slippages if abs(s) < 0.001]) / len(slippages), 3
            ),
        }

    def _rank_symbols(self, trades: list[dict], top: bool = True) -> list[dict]:
        """Rank symbols by total P&L."""
        by_sym = defaultdict(float)
        counts = defaultdict(int)
        for t in trades:
            sym = t.get("symbol", "?")
            by_sym[sym] += t["pnl"]
            counts[sym] += 1

        ranked = sorted(by_sym.items(), key=lambda x: x[1], reverse=top)
        return [
            {"symbol": sym, "total_pnl": round(pnl, 2), "trades": counts[sym]}
            for sym, pnl in ranked[:5]
        ]

    def _analyze_streaks(self, trades: list[dict]) -> dict:
        """Find winning/losing streaks."""
        if not trades:
            return {}

        max_win_streak = current_win = 0
        max_loss_streak = current_loss = 0

        for t in sorted(trades, key=lambda x: x.get("timestamp", "")):
            if t["pnl"] > 0:
                current_win += 1
                current_loss = 0
                max_win_streak = max(max_win_streak, current_win)
            else:
                current_loss += 1
                current_win = 0
                max_loss_streak = max(max_loss_streak, current_loss)

        return {
            "max_win_streak": max_win_streak,
            "max_loss_streak": max_loss_streak,
        }

    def _generate_insights(self, patterns: dict, trades: list[dict]) -> list[str]:
        """Auto-generate actionable insights from the patterns."""
        insights = []

        # Strategy comparison
        by_strat = patterns.get("by_strategy", {})
        for strat, stats in by_strat.items():
            if stats["count"] >= 5:
                if stats["win_rate"] >= 0.6:
                    insights.append(
                        f"STRENGTH: {strat} strategy has {stats['win_rate']:.0%} win rate "
                        f"over {stats['count']} trades (avg P&L: {stats['avg_pnl']:.2f})"
                    )
                elif stats["win_rate"] < 0.4:
                    insights.append(
                        f"WEAKNESS: {strat} strategy has only {stats['win_rate']:.0%} win rate "
                        f"over {stats['count']} trades — consider parameter tuning"
                    )

        # Direction bias
        by_dir = patterns.get("by_direction", {})
        for d, stats in by_dir.items():
            if stats["count"] >= 5 and stats["win_rate"] < 0.35:
                insights.append(
                    f"CAUTION: {d} trades underperforming ({stats['win_rate']:.0%} win rate). "
                    f"Consider reducing {d} exposure."
                )

        # Best/worst symbols
        best = patterns.get("best_symbols", [])
        worst = patterns.get("worst_symbols", [])
        if best and best[0]["total_pnl"] > 0:
            insights.append(
                f"TOP PERFORMER: {best[0]['symbol']} with {best[0]['total_pnl']:+.2f} "
                f"total P&L over {best[0]['trades']} trades"
            )
        if worst and worst[0]["total_pnl"] < -20:
            insights.append(
                f"UNDERPERFORMER: {worst[0]['symbol']} with {worst[0]['total_pnl']:+.2f} "
                f"total P&L — consider removing from watchlist"
            )

        # VIX impact
        by_vix = patterns.get("by_vix_range", {})
        for rng, stats in by_vix.items():
            if stats["count"] >= 3 and stats["win_rate"] < 0.3:
                insights.append(
                    f"VIX WARNING: Poor performance in {rng.replace('_', ' ')} VIX "
                    f"({stats['win_rate']:.0%} win rate) — tighten filters"
                )

        # Confidence validation
        by_conf = patterns.get("by_confidence", {})
        high = by_conf.get("high_80-100", {})
        low = by_conf.get("low_40-59", {})
        if high.get("count", 0) >= 3 and low.get("count", 0) >= 3:
            if high.get("win_rate", 0) > low.get("win_rate", 0) + 0.15:
                insights.append(
                    f"CONFIDENCE VALIDATED: High-confidence signals ({high['win_rate']:.0%}) "
                    f"significantly outperform low-confidence ({low['win_rate']:.0%})"
                )
            elif low.get("win_rate", 0) >= high.get("win_rate", 0):
                insights.append(
                    f"CONFIDENCE ISSUE: Low-confidence signals ({low['win_rate']:.0%}) "
                    f"perform as well as high-confidence ({high['win_rate']:.0%}) — "
                    f"confidence scoring needs recalibration"
                )

        # Duration insight
        dur = patterns.get("duration_stats", {})
        if dur.get("avg_winner_duration_min", 0) > 0 and dur.get("avg_loser_duration_min", 0) > 0:
            if dur["avg_loser_duration_min"] > dur["avg_winner_duration_min"] * 2:
                insights.append(
                    f"TIMING: Losers held {dur['avg_loser_duration_min']:.0f}min avg vs "
                    f"winners {dur['avg_winner_duration_min']:.0f}min — "
                    f"consider tighter time exits"
                )

        # Streak warning
        streaks = patterns.get("streaks", {})
        if streaks.get("max_loss_streak", 0) >= 5:
            insights.append(
                f"DRAWDOWN: Max losing streak of {streaks['max_loss_streak']} trades — "
                f"risk of tilt, ensure circuit breakers are active"
            )

        # Hour-of-day patterns
        by_hour = patterns.get("by_hour", {})
        best_hour = max(by_hour.items(), key=lambda x: x[1].get("win_rate", 0), default=None)
        worst_hour = min(by_hour.items(), key=lambda x: x[1].get("win_rate", 0), default=None)
        if best_hour and worst_hour and best_hour[1]["count"] >= 3:
            if best_hour[1]["win_rate"] > worst_hour[1]["win_rate"] + 0.2:
                insights.append(
                    f"TIMING: Best hour is {best_hour[0]}:00 ({best_hour[1]['win_rate']:.0%}), "
                    f"worst is {worst_hour[0]}:00 ({worst_hour[1]['win_rate']:.0%})"
                )

        return insights

    # ─── Knowledge Base Generation ────────────────────────────────

    def _generate_knowledge_base(self, patterns: dict):
        """Generate human-readable knowledge_base.md from accumulated patterns."""
        lines = [
            "# Trading Knowledge Base",
            f"*Auto-generated: {datetime.now():%Y-%m-%d %H:%M}*",
            f"*Lookback: {patterns.get('lookback_days', 0)} days | "
            f"Trades: {patterns.get('total_trades_analyzed', 0)} | "
            f"Signals: {patterns.get('total_signals_analyzed', 0)}*",
            "",
            "---",
            "",
        ]

        # Key Insights
        insights = patterns.get("insights", [])
        if insights:
            lines.append("## Key Insights")
            lines.append("")
            for i, insight in enumerate(insights, 1):
                lines.append(f"{i}. {insight}")
            lines.append("")

        # Strategy Performance
        by_strat = patterns.get("by_strategy", {})
        if by_strat:
            lines.append("## Strategy Performance")
            lines.append("")
            lines.append("| Strategy | Trades | Win Rate | Avg P&L | Total P&L | Profit Factor |")
            lines.append("|----------|--------|----------|---------|-----------|---------------|")
            for strat, s in sorted(by_strat.items()):
                lines.append(
                    f"| {strat} | {s['count']} | {s['win_rate']:.0%} | "
                    f"€{s['avg_pnl']:.2f} | €{s['total_pnl']:.2f} | {s['profit_factor']:.2f} |"
                )
            lines.append("")

        # Symbol Performance
        by_sym = patterns.get("by_symbol", {})
        if by_sym:
            lines.append("## Symbol Performance")
            lines.append("")
            lines.append("| Symbol | Trades | Win Rate | Avg P&L | Total P&L |")
            lines.append("|--------|--------|----------|---------|-----------|")
            for sym, s in sorted(by_sym.items(), key=lambda x: x[1]["total_pnl"], reverse=True):
                lines.append(
                    f"| {sym} | {s['count']} | {s['win_rate']:.0%} | "
                    f"€{s['avg_pnl']:.2f} | €{s['total_pnl']:.2f} |"
                )
            lines.append("")

        # Confidence Levels
        by_conf = patterns.get("by_confidence", {})
        if by_conf:
            lines.append("## Confidence Level Analysis")
            lines.append("")
            lines.append("| Confidence | Trades | Win Rate | Avg P&L | Total P&L |")
            lines.append("|------------|--------|----------|---------|-----------|")
            for bucket, s in sorted(by_conf.items()):
                lines.append(
                    f"| {bucket.replace('_', ' ')} | {s['count']} | "
                    f"{s['win_rate']:.0%} | €{s['avg_pnl']:.2f} | €{s['total_pnl']:.2f} |"
                )
            lines.append("")

        # VIX Analysis
        by_vix = patterns.get("by_vix_range", {})
        if by_vix:
            lines.append("## VIX Impact")
            lines.append("")
            lines.append("| VIX Range | Trades | Win Rate | Avg P&L |")
            lines.append("|-----------|--------|----------|---------|")
            for rng, s in sorted(by_vix.items()):
                lines.append(
                    f"| {rng.replace('_', ' ')} | {s['count']} | "
                    f"{s['win_rate']:.0%} | €{s['avg_pnl']:.2f} |"
                )
            lines.append("")

        # Time Patterns
        by_hour = patterns.get("by_hour", {})
        if by_hour:
            lines.append("## Hour-of-Day Performance")
            lines.append("")
            lines.append("| Hour (CET) | Trades | Win Rate | Avg P&L |")
            lines.append("|------------|--------|----------|---------|")
            for hour, s in sorted(by_hour.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
                lines.append(
                    f"| {hour}:00 | {s['count']} | "
                    f"{s['win_rate']:.0%} | €{s['avg_pnl']:.2f} |"
                )
            lines.append("")

        # Day of Week
        by_day = patterns.get("by_day", {})
        if by_day:
            lines.append("## Day-of-Week Performance")
            lines.append("")
            lines.append("| Day | Trades | Win Rate | Avg P&L |")
            lines.append("|-----|--------|----------|---------|")
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            for day in day_order:
                if day in by_day:
                    s = by_day[day]
                    lines.append(
                        f"| {day} | {s['count']} | "
                        f"{s['win_rate']:.0%} | €{s['avg_pnl']:.2f} |"
                    )
            lines.append("")

        # Duration Stats
        dur = patterns.get("duration_stats", {})
        if dur:
            lines.append("## Trade Duration")
            lines.append("")
            lines.append(f"- Average: {dur.get('avg_duration_min', 0):.1f} min")
            lines.append(f"- Winners avg: {dur.get('avg_winner_duration_min', 0):.1f} min")
            lines.append(f"- Losers avg: {dur.get('avg_loser_duration_min', 0):.1f} min")
            lines.append(f"- Shortest: {dur.get('shortest_min', 0):.1f} min")
            lines.append(f"- Longest: {dur.get('longest_min', 0):.1f} min")
            lines.append("")

        # Slippage
        slip = patterns.get("slippage_stats", {})
        if slip:
            lines.append("## Slippage Analysis")
            lines.append("")
            lines.append(f"- Average slippage: €{slip.get('avg_slippage', 0):.4f}")
            lines.append(f"- Zero slippage: {slip.get('zero_slippage_pct', 0):.0%} of trades")
            lines.append(f"- Worst adverse: €{slip.get('max_adverse_slippage', 0):.4f}")
            lines.append("")

        # Streaks
        streaks = patterns.get("streaks", {})
        if streaks:
            lines.append("## Streaks")
            lines.append("")
            lines.append(f"- Max winning streak: {streaks.get('max_win_streak', 0)}")
            lines.append(f"- Max losing streak: {streaks.get('max_loss_streak', 0)}")
            lines.append("")

        # Direction Performance
        by_dir = patterns.get("by_direction", {})
        if by_dir:
            lines.append("## Direction Performance")
            lines.append("")
            lines.append("| Direction | Trades | Win Rate | Avg P&L | Total P&L |")
            lines.append("|-----------|--------|----------|---------|-----------|")
            for d, s in by_dir.items():
                lines.append(
                    f"| {d} | {s['count']} | {s['win_rate']:.0%} | "
                    f"€{s['avg_pnl']:.2f} | €{s['total_pnl']:.2f} |"
                )
            lines.append("")

        # Exit Reason
        by_exit = patterns.get("by_exit_reason", {})
        if by_exit:
            lines.append("## Exit Reason Breakdown")
            lines.append("")
            lines.append("| Exit Reason | Count | Win Rate | Avg P&L |")
            lines.append("|-------------|-------|----------|---------|")
            for reason, s in by_exit.items():
                lines.append(
                    f"| {reason} | {s['count']} | {s['win_rate']:.0%} | €{s['avg_pnl']:.2f} |"
                )
            lines.append("")

        # Write
        self._knowledge_file.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"[Knowledge] knowledge_base.md updated ({len(lines)} lines)")

    # ─── Helpers ──────────────────────────────────────────────────

    def _clean_indicators(self, indicators: dict) -> dict:
        """Clean indicator values for JSON serialization (handle NaN, numpy)."""
        cleaned = {}
        for k, v in indicators.items():
            try:
                import math
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    cleaned[k] = None
                else:
                    cleaned[k] = float(v) if v is not None else None
            except (TypeError, ValueError):
                cleaned[k] = str(v) if v is not None else None
        return cleaned

    def _update_patterns(self, trade_entry: dict):
        """Incrementally update running pattern stats after each trade."""
        # Simple counter updates (full analysis runs periodically)
        if "running_stats" not in self._patterns:
            self._patterns["running_stats"] = {
                "total_trades": 0, "wins": 0, "losses": 0,
                "total_pnl": 0, "session_trades": 0,
            }

        stats = self._patterns["running_stats"]
        stats["total_trades"] += 1
        stats["session_trades"] += 1
        stats["total_pnl"] = round(stats["total_pnl"] + trade_entry["pnl"], 2)
        if trade_entry["pnl"] > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

    def _append_jsonl(self, path: Path, entry: dict):
        """Append a JSON line to a JSONL file."""
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.error(f"[Knowledge] Failed to write to {path}: {e}")

    def _load_all_trades(self, lookback_days: int) -> list[dict]:
        """Load all trades from the last N days."""
        trades = []
        for i in range(lookback_days):
            d = date.today() - timedelta(days=i)
            path = KNOWLEDGE_DIR / f"trades_{d:%Y%m%d}.jsonl"
            if path.exists():
                trades.extend(self._read_jsonl(path))
        return trades

    def _load_all_signals(self, lookback_days: int) -> list[dict]:
        """Load all signals from the last N days."""
        signals = []
        for i in range(lookback_days):
            d = date.today() - timedelta(days=i)
            path = KNOWLEDGE_DIR / f"signals_{d:%Y%m%d}.jsonl"
            if path.exists():
                signals.extend(self._read_jsonl(path))
        return signals

    def _read_jsonl(self, path: Path) -> list[dict]:
        """Read a JSONL file into a list of dicts."""
        entries = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"[Knowledge] Failed to read {path}: {e}")
        return entries

    def _load_patterns(self) -> dict:
        """Load patterns from disk."""
        if self._patterns_file.exists():
            try:
                return json.loads(self._patterns_file.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_patterns(self, patterns: dict):
        """Save patterns to disk."""
        try:
            self._patterns_file.write_text(
                json.dumps(patterns, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"[Knowledge] Failed to save patterns: {e}")

    # ─── Public Access ────────────────────────────────────────────

    def get_patterns(self) -> dict:
        """Get the current pattern data."""
        return self._patterns

    def get_insights(self) -> list[str]:
        """Get the latest auto-generated insights."""
        return self._patterns.get("insights", [])

    def get_session_summary(self) -> dict:
        """Get summary of current session's signal/trade data."""
        signals = self._session_signals
        trades = [t for t in self._session_trades if t.get("type") == "trade_close"]

        return {
            "signals_recorded": len(signals),
            "trades_opened": len([t for t in self._session_trades if t.get("type") == "trade_open"]),
            "trades_closed": len(trades),
            "session_pnl": round(sum(t.get("pnl", 0) for t in trades), 2),
            "win_rate": round(
                len([t for t in trades if t["pnl"] > 0]) / len(trades), 2
            ) if trades else 0,
        }
