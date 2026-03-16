"""
TradeDatabase — SQLite persistence for complete trading history.

Runs alongside JSONL files (additive, not replacement).
Provides structured queries for ML training, pattern analysis,
and performance reporting.

Schema:
    trades           — Complete trade lifecycle (open → close)
    signals          — Every signal generated (executed + filtered + skipped)
    market_snapshots — Periodic scan-cycle market state
    pair_trades      — Pairs/stat-arb trades with both legs
    daily_summaries  — End-of-day aggregated statistics
    schema_version   — Migration tracking
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Path("knowledge") / "trade_history.db"


class TradeDatabase:
    """SQLite trading history database with structured queries."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = Path(db_path) if db_path else DB_PATH
        self._db_path.parent.mkdir(exist_ok=True)
        self._create_schema()
        logger.info(f"TradeDatabase ready: {self._db_path}")

    # ─── Connection Management ──────────────────────────────────

    @contextmanager
    def _connection(self):
        """Thread-safe connection context manager with WAL mode."""
        conn = sqlite3.connect(str(self._db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ─── Schema Creation ────────────────────────────────────────

    def _create_schema(self):
        """Create all tables and indexes if they don't exist."""
        with self._connection() as conn:
            conn.executescript(SCHEMA_SQL)

    # ─── Insert Methods ─────────────────────────────────────────

    def insert_signal(
        self,
        signal,
        market_context: dict,
        action: str,
        filter_reason: str = "",
        ml_result: Optional[tuple] = None,
        hmm_probs: Optional[dict] = None,
        additional_indicators: Optional[dict] = None,
        corr_data: Optional[dict] = None,
    ) -> Optional[int]:
        """Insert a signal record. Returns the signal row id."""
        try:
            indicators = signal.indicators or {}
            ml_win_prob = ml_result[1] if ml_result else None
            ml_allowed = int(ml_result[0]) if ml_result else None
            hmm = hmm_probs or {}
            ind = additional_indicators or {}
            corr = corr_data or {}

            with self._connection() as conn:
                cursor = conn.execute(
                    """INSERT INTO signals (
                        timestamp, date, day_of_week, hour, minute,
                        symbol, direction, price, strategy,
                        confidence, expected_edge,
                        comp_zscore, comp_volume, comp_momentum,
                        comp_volatility, comp_trend, comp_microstructure, comp_correlation,
                        stop_loss, take_profit, reason,
                        regime, regime_source,
                        hmm_prob_low_vol, hmm_prob_normal, hmm_prob_high_vol,
                        vix, vix_change, dax_pct, sp500_pct,
                        atr, atr_pct, rsi, adx, bb_width, volume_ratio,
                        ml_win_probability, ml_allowed,
                        corr_score, corr_zscore,
                        action, filter_reason
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?
                    )""",
                    (
                        datetime.now().isoformat(),
                        str(date.today()),
                        date.today().strftime("%A"),
                        datetime.now().hour,
                        datetime.now().minute,
                        signal.symbol,
                        signal.type.value,
                        signal.price,
                        signal.strategy,
                        signal.confidence,
                        getattr(signal, "expected_edge", 0),
                        indicators.get("zscore"),
                        indicators.get("volume"),
                        indicators.get("momentum"),
                        indicators.get("volatility"),
                        indicators.get("trend"),
                        indicators.get("microstructure"),
                        indicators.get("correlation"),
                        signal.stop_loss,
                        signal.take_profit,
                        signal.reason,
                        signal.regime.value,
                        "hmm" if hmm else "adx",
                        hmm.get("low_vol"),
                        hmm.get("normal"),
                        hmm.get("high_vol"),
                        market_context.get("vix"),
                        market_context.get("vix_change"),
                        market_context.get("dax_pct"),
                        market_context.get("sp500_pct"),
                        ind.get("atr"),
                        ind.get("atr_pct"),
                        ind.get("rsi"),
                        ind.get("adx"),
                        ind.get("bb_width"),
                        ind.get("volume_ratio"),
                        ml_win_prob,
                        ml_allowed,
                        corr.get("score"),
                        corr.get("zscore"),
                        action,
                        filter_reason,
                    ),
                )
                return cursor.lastrowid
        except Exception as e:
            logger.debug(f"[DB] insert_signal failed: {e}")
            return None

    def insert_trade_open(
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
        # Renaissance context
        signal_components: Optional[dict] = None,
        hmm_probs: Optional[dict] = None,
        ml_win_prob: Optional[float] = None,
        corr_data: Optional[dict] = None,
        additional_indicators: Optional[dict] = None,
        # Sizing
        size_base: Optional[int] = None,
        vix_mult: float = 1.0,
        hmm_mult: float = 1.0,
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
    ) -> Optional[str]:
        """Insert a new trade (open). Returns trade_id."""
        try:
            now = datetime.now()
            trade_id = f"T_{now.strftime('%Y%m%d_%H%M%S')}_{symbol}"
            comp = signal_components or {}
            hmm = hmm_probs or {}
            ind = additional_indicators or {}
            corr = corr_data or {}

            with self._connection() as conn:
                conn.execute(
                    """INSERT INTO trades (
                        trade_id, symbol, direction, strategy,
                        open_timestamp, open_date, day_of_week,
                        open_hour, open_minute,
                        entry_price, stop_loss, take_profit,
                        size, size_base, vix_size_multiplier, hmm_size_multiplier,
                        order_id, tick_size, slippage_entry, is_bracket,
                        signal_confidence, expected_edge, ml_win_probability,
                        comp_zscore, comp_volume, comp_momentum,
                        comp_volatility, comp_trend, comp_microstructure, comp_correlation,
                        regime, regime_source,
                        hmm_prob_low_vol, hmm_prob_normal, hmm_prob_high_vol,
                        vix_at_entry, vix_change_at_entry, dax_pct_at_entry, sp500_pct_at_entry,
                        atr_at_entry, atr_pct_at_entry, rsi_at_entry, adx_at_entry,
                        bb_width_at_entry, volume_ratio_entry,
                        corr_score, corr_zscore, corr_rolling, corr_baseline,
                        consecutive_wins, consecutive_losses,
                        daily_trade_number, daily_pnl_before, account_balance,
                        is_open
                    ) VALUES (
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?,
                        ?, ?, ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?
                    )""",
                    (
                        trade_id, symbol, direction, strategy,
                        now.isoformat(), str(now.date()), now.strftime("%A"),
                        now.hour, now.minute,
                        entry_price, stop_loss, take_profit,
                        size, size_base or size, vix_mult, hmm_mult,
                        order_id, tick_size, slippage, int(is_bracket),
                        signal_confidence, expected_edge, ml_win_prob,
                        comp.get("zscore"), comp.get("volume"), comp.get("momentum"),
                        comp.get("volatility"), comp.get("trend"),
                        comp.get("microstructure"), comp.get("correlation"),
                        regime, "hmm" if hmm else "adx",
                        hmm.get("low_vol"), hmm.get("normal"), hmm.get("high_vol"),
                        market_context.get("vix"), market_context.get("vix_change"),
                        market_context.get("dax_pct"), market_context.get("sp500_pct"),
                        ind.get("atr"), ind.get("atr_pct"), ind.get("rsi"), ind.get("adx"),
                        ind.get("bb_width"), ind.get("volume_ratio"),
                        corr.get("score"), corr.get("zscore"),
                        corr.get("rolling"), corr.get("baseline"),
                        consecutive_wins, consecutive_losses,
                        daily_trade_number, daily_pnl_before, account_balance,
                        1,  # is_open
                    ),
                )
            return trade_id
        except Exception as e:
            logger.debug(f"[DB] insert_trade_open failed: {e}")
            return None

    def update_trade_close(
        self,
        trade_id: str,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
        duration_seconds: int,
        commission: float = 0,
        slippage_exit: float = 0,
        mae_pct: float = 0,
        mfe_pct: float = 0,
        mae_price: Optional[float] = None,
        mfe_price: Optional[float] = None,
        market_context_exit: Optional[dict] = None,
        regime_at_exit: str = "",
    ) -> bool:
        """Update an open trade with close data. Returns success."""
        try:
            now = datetime.now()
            ctx = market_context_exit or {}
            net_pnl = pnl - commission
            duration_minutes = round(duration_seconds / 60, 1) if duration_seconds else 0
            edge_captured = round(mfe_pct / max(abs(pnl_pct), 0.0001), 2) if mfe_pct else 0

            with self._connection() as conn:
                conn.execute(
                    """UPDATE trades SET
                        close_timestamp = ?, close_date = ?,
                        close_hour = ?, close_minute = ?,
                        exit_price = ?, pnl = ?, pnl_pct = ?,
                        commission = ?, net_pnl = ?,
                        exit_reason = ?,
                        slippage_exit = ?,
                        duration_seconds = ?, duration_minutes = ?,
                        mae_pct = ?, mfe_pct = ?, edge_captured = ?,
                        mae_price = ?, mfe_price = ?,
                        vix_at_exit = ?, dax_pct_at_exit = ?, sp500_pct_at_exit = ?,
                        regime_at_exit = ?,
                        is_open = 0,
                        updated_at = ?
                    WHERE trade_id = ?""",
                    (
                        now.isoformat(), str(now.date()),
                        now.hour, now.minute,
                        exit_price, round(pnl, 2), round(pnl_pct, 4),
                        commission, round(net_pnl, 2),
                        exit_reason,
                        slippage_exit,
                        duration_seconds, duration_minutes,
                        mae_pct, mfe_pct, edge_captured,
                        mae_price, mfe_price,
                        ctx.get("vix"), ctx.get("dax_pct"), ctx.get("sp500_pct"),
                        regime_at_exit,
                        now.isoformat(),
                        trade_id,
                    ),
                )
            # Update daily summary
            self.update_daily_summary(str(now.date()))
            return True
        except Exception as e:
            logger.debug(f"[DB] update_trade_close failed: {e}")
            return False

    def insert_market_snapshot(
        self,
        market_context: dict,
        symbols_scanned: int = 0,
        signals_found: int = 0,
    ) -> Optional[int]:
        """Insert a scan-cycle market state snapshot."""
        try:
            now = datetime.now()
            with self._connection() as conn:
                cursor = conn.execute(
                    """INSERT INTO market_snapshots (
                        timestamp, date, hour, minute,
                        vix, vix_change_pct, dax_pct, sp500_pct,
                        symbols_scanned, signals_found
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        now.isoformat(), str(now.date()), now.hour, now.minute,
                        market_context.get("vix"),
                        market_context.get("vix_change"),
                        market_context.get("dax_pct"),
                        market_context.get("sp500_pct"),
                        symbols_scanned,
                        signals_found,
                    ),
                )
                return cursor.lastrowid
        except Exception as e:
            logger.debug(f"[DB] insert_market_snapshot failed: {e}")
            return None

    def insert_pair_trade(
        self,
        pair_id: str,
        direction: str,
        entry_spread: float,
        entry_zscore: float,
        hedge_ratio: float,
        leg_a_symbol: str, leg_a_side: str, leg_a_price: float, leg_a_size: int,
        leg_b_symbol: str, leg_b_side: str, leg_b_price: float, leg_b_size: int,
        coint_pvalue: Optional[float] = None,
        half_life: Optional[float] = None,
        spread_mean: Optional[float] = None,
        spread_std: Optional[float] = None,
        vix_at_entry: Optional[float] = None,
        regime_at_entry: str = "",
    ) -> Optional[int]:
        """Insert a pair trade. Returns pair_trade row id."""
        try:
            with self._connection() as conn:
                cursor = conn.execute(
                    """INSERT INTO pair_trades (
                        pair_id, open_timestamp, direction,
                        entry_spread, entry_zscore, hedge_ratio,
                        coint_pvalue, half_life, spread_mean, spread_std,
                        leg_a_symbol, leg_a_side, leg_a_entry_price, leg_a_size,
                        leg_b_symbol, leg_b_side, leg_b_entry_price, leg_b_size,
                        vix_at_entry, regime_at_entry, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
                    (
                        pair_id, datetime.now().isoformat(), direction,
                        entry_spread, entry_zscore, hedge_ratio,
                        coint_pvalue, half_life, spread_mean, spread_std,
                        leg_a_symbol, leg_a_side, leg_a_price, leg_a_size,
                        leg_b_symbol, leg_b_side, leg_b_price, leg_b_size,
                        vix_at_entry, regime_at_entry,
                    ),
                )
                return cursor.lastrowid
        except Exception as e:
            logger.debug(f"[DB] insert_pair_trade failed: {e}")
            return None

    def update_pair_trade_close(
        self,
        pair_trade_id: int,
        exit_spread: float,
        exit_zscore: float,
        pnl: float,
        exit_reason: str,
        leg_a_exit_price: float,
        leg_b_exit_price: float,
        commission: float = 5.0,
        vix_at_exit: Optional[float] = None,
        duration_seconds: int = 0,
    ) -> bool:
        """Close a pair trade."""
        try:
            now = datetime.now()
            with self._connection() as conn:
                conn.execute(
                    """UPDATE pair_trades SET
                        close_timestamp = ?, exit_spread = ?, exit_zscore = ?,
                        pnl = ?, commission = ?, net_pnl = ?,
                        exit_reason = ?, duration_seconds = ?,
                        leg_a_exit_price = ?, leg_b_exit_price = ?,
                        vix_at_exit = ?,
                        status = 'closed', updated_at = ?
                    WHERE id = ?""",
                    (
                        now.isoformat(), exit_spread, exit_zscore,
                        pnl, commission, round(pnl - commission, 2),
                        exit_reason, duration_seconds,
                        leg_a_exit_price, leg_b_exit_price,
                        vix_at_exit,
                        now.isoformat(),
                        pair_trade_id,
                    ),
                )
            return True
        except Exception as e:
            logger.debug(f"[DB] update_pair_trade_close failed: {e}")
            return False

    # ─── Find Methods ───────────────────────────────────────────

    def find_open_trade(self, symbol: str, direction: str) -> Optional[str]:
        """Find the trade_id of an open trade for a given symbol+direction."""
        try:
            with self._connection() as conn:
                row = conn.execute(
                    "SELECT trade_id FROM trades WHERE symbol=? AND direction=? AND is_open=1 ORDER BY open_timestamp DESC LIMIT 1",
                    (symbol, direction),
                ).fetchone()
                return row["trade_id"] if row else None
        except Exception:
            return None

    # ─── Daily Summary ──────────────────────────────────────────

    def update_daily_summary(self, trade_date: str):
        """Recompute daily summary for a given date from trades + signals."""
        try:
            with self._connection() as conn:
                # Closed trades for the date
                trades = conn.execute(
                    "SELECT * FROM trades WHERE close_date=? AND is_open=0",
                    (trade_date,),
                ).fetchall()

                opens = conn.execute(
                    "SELECT COUNT(*) as c FROM trades WHERE open_date=?",
                    (trade_date,),
                ).fetchone()["c"]

                signals = conn.execute(
                    "SELECT action, COUNT(*) as c FROM signals WHERE date=? GROUP BY action",
                    (trade_date,),
                ).fetchall()

                scans = conn.execute(
                    "SELECT COUNT(*) as c FROM market_snapshots WHERE date=?",
                    (trade_date,),
                ).fetchone()["c"]

                # Compute stats
                closed = len(trades)
                wins = sum(1 for t in trades if t["pnl"] and t["pnl"] > 0)
                losses = sum(1 for t in trades if t["pnl"] and t["pnl"] <= 0)
                win_rate = wins / max(closed, 1)
                gross_pnl = sum(t["pnl"] or 0 for t in trades)
                total_comm = sum(t["commission"] or 0 for t in trades)
                net_pnl = gross_pnl - total_comm

                winners = [t["pnl"] for t in trades if t["pnl"] and t["pnl"] > 0]
                losers = [t["pnl"] for t in trades if t["pnl"] and t["pnl"] <= 0]
                avg_winner = sum(winners) / len(winners) if winners else 0
                avg_loser = sum(losers) / len(losers) if losers else 0
                profit_factor = abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else 0
                largest_winner = max(winners) if winners else 0
                largest_loser = min(losers) if losers else 0
                avg_hold = sum(t["duration_minutes"] or 0 for t in trades) / max(closed, 1)

                signal_counts = {s["action"]: s["c"] for s in signals}

                # Pair trades
                pairs_opened = conn.execute(
                    "SELECT COUNT(*) as c FROM pair_trades WHERE date(open_timestamp)=?",
                    (trade_date,),
                ).fetchone()["c"]
                pairs_closed = conn.execute(
                    "SELECT COUNT(*) as c FROM pair_trades WHERE date(close_timestamp)=? AND status='closed'",
                    (trade_date,),
                ).fetchone()["c"]
                pair_pnl_row = conn.execute(
                    "SELECT COALESCE(SUM(pnl), 0) as s FROM pair_trades WHERE date(close_timestamp)=? AND status='closed'",
                    (trade_date,),
                ).fetchone()
                pair_pnl = pair_pnl_row["s"] if pair_pnl_row else 0

                # Upsert
                conn.execute(
                    """INSERT INTO daily_summaries (
                        date, day_of_week,
                        trades_opened, trades_closed,
                        signals_generated, signals_executed, signals_filtered,
                        scan_cycles,
                        gross_pnl, total_commission, net_pnl,
                        wins, losses, win_rate,
                        avg_winner, avg_loser, profit_factor,
                        largest_winner, largest_loser,
                        avg_hold_minutes,
                        pair_trades_opened, pair_trades_closed, pair_pnl
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(date) DO UPDATE SET
                        trades_opened=excluded.trades_opened,
                        trades_closed=excluded.trades_closed,
                        signals_generated=excluded.signals_generated,
                        signals_executed=excluded.signals_executed,
                        signals_filtered=excluded.signals_filtered,
                        scan_cycles=excluded.scan_cycles,
                        gross_pnl=excluded.gross_pnl,
                        total_commission=excluded.total_commission,
                        net_pnl=excluded.net_pnl,
                        wins=excluded.wins, losses=excluded.losses,
                        win_rate=excluded.win_rate,
                        avg_winner=excluded.avg_winner, avg_loser=excluded.avg_loser,
                        profit_factor=excluded.profit_factor,
                        largest_winner=excluded.largest_winner,
                        largest_loser=excluded.largest_loser,
                        avg_hold_minutes=excluded.avg_hold_minutes,
                        pair_trades_opened=excluded.pair_trades_opened,
                        pair_trades_closed=excluded.pair_trades_closed,
                        pair_pnl=excluded.pair_pnl,
                        updated_at=datetime('now')
                    """,
                    (
                        trade_date,
                        datetime.strptime(trade_date, "%Y-%m-%d").strftime("%A"),
                        opens, closed,
                        sum(signal_counts.values()), signal_counts.get("executed", 0),
                        signal_counts.get("filtered", 0),
                        scans,
                        round(gross_pnl, 2), round(total_comm, 2), round(net_pnl, 2),
                        wins, losses, round(win_rate, 4),
                        round(avg_winner, 2), round(avg_loser, 2), round(profit_factor, 2),
                        round(largest_winner, 2), round(largest_loser, 2),
                        round(avg_hold, 1),
                        pairs_opened, pairs_closed, round(pair_pnl, 2),
                    ),
                )
        except Exception as e:
            logger.debug(f"[DB] update_daily_summary failed: {e}")

    # ─── ML Training Data Extraction ────────────────────────────

    def get_ml_training_data(self, lookback_days: int = 60) -> pd.DataFrame:
        """
        Return closed trades as a DataFrame ready for ML training.
        Columns match MLSignalFilter feature expectations + extra Renaissance features.
        """
        try:
            cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
            with self._connection() as conn:
                df = pd.read_sql_query(
                    """SELECT
                        comp_zscore, comp_volume, comp_momentum,
                        comp_volatility, comp_trend, comp_microstructure,
                        comp_correlation,
                        signal_confidence as composite_score,
                        regime, regime_source,
                        hmm_prob_low_vol, hmm_prob_normal, hmm_prob_high_vol,
                        vix_at_entry as vix, dax_pct_at_entry as dax_pct,
                        open_hour as hour, day_of_week,
                        symbol,
                        atr_pct_at_entry as atr_pct,
                        rsi_at_entry as rsi, adx_at_entry as adx,
                        bb_width_at_entry as bb_width, volume_ratio_entry as volume_ratio,
                        corr_score, corr_zscore,
                        ml_win_probability,
                        pnl, pnl_pct, exit_reason,
                        duration_minutes,
                        mae_pct, mfe_pct,
                        CASE WHEN pnl > 0 THEN 1 ELSE 0 END as label
                    FROM trades
                    WHERE is_open = 0 AND close_date >= ?
                    ORDER BY close_timestamp""",
                    conn,
                    params=(cutoff,),
                )
                return df
        except Exception as e:
            logger.debug(f"[DB] get_ml_training_data failed: {e}")
            return pd.DataFrame()

    # ─── Query Methods ──────────────────────────────────────────

    def query_performance_by(self, group_by: str, lookback_days: int = 30, min_trades: int = 1) -> pd.DataFrame:
        """Generic grouped performance query."""
        try:
            cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
            valid_cols = {
                "symbol", "strategy", "regime", "direction", "exit_reason",
                "open_hour", "day_of_week", "regime_source",
            }
            if group_by not in valid_cols:
                return pd.DataFrame()

            with self._connection() as conn:
                df = pd.read_sql_query(
                    f"""SELECT
                        {group_by},
                        COUNT(*) as trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                        ROUND(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END), 3) as win_rate,
                        ROUND(SUM(pnl), 2) as total_pnl,
                        ROUND(SUM(net_pnl), 2) as total_net_pnl,
                        ROUND(AVG(pnl), 2) as avg_pnl,
                        ROUND(AVG(CASE WHEN pnl > 0 THEN pnl END), 2) as avg_winner,
                        ROUND(AVG(CASE WHEN pnl <= 0 THEN pnl END), 2) as avg_loser,
                        ROUND(ABS(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END)) /
                              NULLIF(ABS(SUM(CASE WHEN pnl <= 0 THEN pnl ELSE 0 END)), 0), 2) as profit_factor
                    FROM trades
                    WHERE is_open = 0 AND close_date >= ?
                    GROUP BY {group_by}
                    HAVING COUNT(*) >= ?
                    ORDER BY total_pnl DESC""",
                    conn,
                    params=(cutoff, min_trades),
                )
                return df
        except Exception as e:
            logger.debug(f"[DB] query_performance_by failed: {e}")
            return pd.DataFrame()

    def query_win_rate_filtered(
        self,
        direction: Optional[str] = None,
        min_confidence: Optional[float] = None,
        strategy: Optional[str] = None,
        regime: Optional[str] = None,
        hour_range: Optional[tuple] = None,
        symbol: Optional[str] = None,
        vix_range: Optional[tuple] = None,
        lookback_days: int = 30,
    ) -> dict:
        """Flexible filtered win rate query."""
        try:
            cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
            conditions = ["is_open = 0", "close_date >= ?"]
            params = [cutoff]

            if direction:
                conditions.append("direction = ?")
                params.append(direction)
            if min_confidence is not None:
                conditions.append("signal_confidence >= ?")
                params.append(min_confidence)
            if strategy:
                conditions.append("strategy = ?")
                params.append(strategy)
            if regime:
                conditions.append("regime = ?")
                params.append(regime)
            if hour_range:
                conditions.append("open_hour BETWEEN ? AND ?")
                params.extend(hour_range)
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            if vix_range:
                conditions.append("vix_at_entry BETWEEN ? AND ?")
                params.extend(vix_range)

            where = " AND ".join(conditions)

            with self._connection() as conn:
                row = conn.execute(
                    f"""SELECT
                        COUNT(*) as trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                        ROUND(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END), 3) as win_rate,
                        ROUND(SUM(pnl), 2) as total_pnl,
                        ROUND(AVG(pnl), 2) as avg_pnl
                    FROM trades WHERE {where}""",
                    params,
                ).fetchone()
                return dict(row) if row else {}
        except Exception as e:
            logger.debug(f"[DB] query_win_rate_filtered failed: {e}")
            return {}

    def query_optimal_holding_period(self, strategy: Optional[str] = None, lookback_days: int = 30) -> pd.DataFrame:
        """PnL by holding duration buckets."""
        try:
            cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
            strat_filter = "AND strategy = ?" if strategy else ""
            params = [cutoff] + ([strategy] if strategy else [])

            with self._connection() as conn:
                df = pd.read_sql_query(
                    f"""SELECT
                        CASE
                            WHEN duration_minutes < 5 THEN '0-5min'
                            WHEN duration_minutes < 15 THEN '5-15min'
                            WHEN duration_minutes < 30 THEN '15-30min'
                            WHEN duration_minutes < 60 THEN '30-60min'
                            WHEN duration_minutes < 120 THEN '60-120min'
                            ELSE '120+min'
                        END as duration_bucket,
                        COUNT(*) as trades,
                        ROUND(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END), 3) as win_rate,
                        ROUND(AVG(pnl), 2) as avg_pnl,
                        ROUND(SUM(pnl), 2) as total_pnl
                    FROM trades
                    WHERE is_open = 0 AND close_date >= ? {strat_filter}
                    GROUP BY duration_bucket
                    ORDER BY duration_bucket""",
                    conn,
                    params=params,
                )
                return df
        except Exception as e:
            logger.debug(f"[DB] query_optimal_holding_period failed: {e}")
            return pd.DataFrame()

    def query_time_of_day_heatmap(self, lookback_days: int = 30) -> pd.DataFrame:
        """Hour × DayOfWeek performance matrix."""
        try:
            cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
            with self._connection() as conn:
                df = pd.read_sql_query(
                    """SELECT
                        day_of_week, open_hour as hour,
                        COUNT(*) as trades,
                        ROUND(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END), 3) as win_rate,
                        ROUND(AVG(pnl), 2) as avg_pnl
                    FROM trades
                    WHERE is_open = 0 AND close_date >= ?
                    GROUP BY day_of_week, open_hour
                    ORDER BY day_of_week, open_hour""",
                    conn,
                    params=(cutoff,),
                )
                return df
        except Exception as e:
            logger.debug(f"[DB] query_time_of_day_heatmap failed: {e}")
            return pd.DataFrame()

    def query_component_effectiveness(self, lookback_days: int = 30) -> pd.DataFrame:
        """Which signal components correlate most with winners?"""
        try:
            cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
            with self._connection() as conn:
                df = pd.read_sql_query(
                    """SELECT
                        ROUND(AVG(CASE WHEN pnl > 0 THEN comp_zscore END), 3) as zscore_winners,
                        ROUND(AVG(CASE WHEN pnl <= 0 THEN comp_zscore END), 3) as zscore_losers,
                        ROUND(AVG(CASE WHEN pnl > 0 THEN comp_volume END), 3) as volume_winners,
                        ROUND(AVG(CASE WHEN pnl <= 0 THEN comp_volume END), 3) as volume_losers,
                        ROUND(AVG(CASE WHEN pnl > 0 THEN comp_momentum END), 3) as momentum_winners,
                        ROUND(AVG(CASE WHEN pnl <= 0 THEN comp_momentum END), 3) as momentum_losers,
                        ROUND(AVG(CASE WHEN pnl > 0 THEN comp_volatility END), 3) as volatility_winners,
                        ROUND(AVG(CASE WHEN pnl <= 0 THEN comp_volatility END), 3) as volatility_losers,
                        ROUND(AVG(CASE WHEN pnl > 0 THEN comp_trend END), 3) as trend_winners,
                        ROUND(AVG(CASE WHEN pnl <= 0 THEN comp_trend END), 3) as trend_losers,
                        ROUND(AVG(CASE WHEN pnl > 0 THEN comp_microstructure END), 3) as micro_winners,
                        ROUND(AVG(CASE WHEN pnl <= 0 THEN comp_microstructure END), 3) as micro_losers,
                        ROUND(AVG(CASE WHEN pnl > 0 THEN comp_correlation END), 3) as corr_winners,
                        ROUND(AVG(CASE WHEN pnl <= 0 THEN comp_correlation END), 3) as corr_losers
                    FROM trades
                    WHERE is_open = 0 AND close_date >= ?""",
                    conn,
                    params=(cutoff,),
                )
                return df
        except Exception as e:
            logger.debug(f"[DB] query_component_effectiveness failed: {e}")
            return pd.DataFrame()

    def query_slippage_analysis(self, lookback_days: int = 30) -> pd.DataFrame:
        """Slippage by symbol and size bucket."""
        try:
            cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
            with self._connection() as conn:
                df = pd.read_sql_query(
                    """SELECT
                        symbol,
                        CASE
                            WHEN size <= 10 THEN 'small (1-10)'
                            WHEN size <= 30 THEN 'medium (11-30)'
                            ELSE 'large (31+)'
                        END as size_bucket,
                        COUNT(*) as trades,
                        ROUND(AVG(slippage_entry), 4) as avg_entry_slippage,
                        ROUND(MAX(ABS(slippage_entry)), 4) as max_entry_slippage
                    FROM trades
                    WHERE is_open = 0 AND close_date >= ?
                    GROUP BY symbol, size_bucket
                    ORDER BY symbol, size_bucket""",
                    conn,
                    params=(cutoff,),
                )
                return df
        except Exception as e:
            logger.debug(f"[DB] query_slippage_analysis failed: {e}")
            return pd.DataFrame()

    def query_hmm_vs_adx_accuracy(self, lookback_days: int = 30) -> pd.DataFrame:
        """Compare HMM vs ADX regime detection accuracy."""
        try:
            cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
            with self._connection() as conn:
                df = pd.read_sql_query(
                    """SELECT
                        regime_source, regime,
                        COUNT(*) as trades,
                        ROUND(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END), 3) as win_rate,
                        ROUND(AVG(pnl), 2) as avg_pnl,
                        ROUND(SUM(pnl), 2) as total_pnl
                    FROM trades
                    WHERE is_open = 0 AND close_date >= ?
                    GROUP BY regime_source, regime
                    ORDER BY regime_source, regime""",
                    conn,
                    params=(cutoff,),
                )
                return df
        except Exception as e:
            logger.debug(f"[DB] query_hmm_vs_adx_accuracy failed: {e}")
            return pd.DataFrame()

    def get_equity_curve(self, lookback_days: int = 30) -> pd.DataFrame:
        """Cumulative PnL over time."""
        try:
            cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
            with self._connection() as conn:
                df = pd.read_sql_query(
                    """SELECT
                        close_date as date,
                        COUNT(*) as trade_count,
                        ROUND(SUM(pnl), 2) as daily_pnl,
                        ROUND(SUM(net_pnl), 2) as daily_net_pnl,
                        ROUND(SUM(SUM(pnl)) OVER (ORDER BY close_date), 2) as cumulative_pnl,
                        ROUND(SUM(SUM(net_pnl)) OVER (ORDER BY close_date), 2) as cumulative_net_pnl
                    FROM trades
                    WHERE is_open = 0 AND close_date >= ?
                    GROUP BY close_date
                    ORDER BY close_date""",
                    conn,
                    params=(cutoff,),
                )
                return df
        except Exception as e:
            logger.debug(f"[DB] get_equity_curve failed: {e}")
            return pd.DataFrame()

    def get_daily_summary(self, trade_date: str) -> dict:
        """Single day performance summary."""
        try:
            with self._connection() as conn:
                row = conn.execute(
                    "SELECT * FROM daily_summaries WHERE date = ?",
                    (trade_date,),
                ).fetchone()
                return dict(row) if row else {}
        except Exception:
            return {}

    def get_period_summary(self, start_date: str, end_date: str) -> dict:
        """Period performance summary."""
        try:
            with self._connection() as conn:
                row = conn.execute(
                    """SELECT
                        COUNT(*) as trading_days,
                        SUM(trades_closed) as total_trades,
                        SUM(wins) as total_wins,
                        SUM(losses) as total_losses,
                        ROUND(CAST(SUM(wins) AS FLOAT) / NULLIF(SUM(wins) + SUM(losses), 0), 3) as win_rate,
                        ROUND(SUM(gross_pnl), 2) as gross_pnl,
                        ROUND(SUM(total_commission), 2) as total_commission,
                        ROUND(SUM(net_pnl), 2) as net_pnl,
                        ROUND(AVG(net_pnl), 2) as avg_daily_pnl,
                        ROUND(MIN(net_pnl), 2) as worst_day,
                        ROUND(MAX(net_pnl), 2) as best_day
                    FROM daily_summaries
                    WHERE date BETWEEN ? AND ?""",
                    (start_date, end_date),
                ).fetchone()
                return dict(row) if row else {}
        except Exception:
            return {}

    # ─── JSONL Migration ────────────────────────────────────────

    def migrate_jsonl(self, knowledge_dir: str = "knowledge"):
        """One-time import of existing JSONL data into SQLite."""
        try:
            with self._connection() as conn:
                # Check if migration already ran
                row = conn.execute(
                    "SELECT version FROM schema_version WHERE description LIKE '%migration%'"
                ).fetchone()
                if row:
                    logger.debug("[DB] JSONL migration already completed — skipping")
                    return

            knowledge_path = Path(knowledge_dir)
            if not knowledge_path.exists():
                return

            trade_files = sorted(knowledge_path.glob("trades_*.jsonl"))
            signal_files = sorted(knowledge_path.glob("signals_*.jsonl"))

            total_trades = 0
            total_signals = 0

            # Import trades — match opens with closes
            for tf in trade_files:
                opens = {}  # (symbol, direction) -> list of open records
                closes = []

                for line in tf.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if rec.get("type") == "trade_open":
                        key = (rec["symbol"], rec["direction"])
                        opens.setdefault(key, []).append(rec)
                    elif rec.get("type") == "trade_close":
                        closes.append(rec)

                # Match closes to opens
                with self._connection() as conn:
                    for close in closes:
                        key = (close["symbol"], close["direction"])
                        open_rec = None
                        if key in opens and opens[key]:
                            open_rec = opens[key].pop(0)

                        trade_id = f"M_{close.get('timestamp', '')}_{close['symbol']}"
                        open_ts = open_rec["timestamp"] if open_rec else close.get("timestamp", "")
                        open_date = open_ts[:10] if open_ts else close.get("date", "")

                        pnl = close.get("pnl", 0)
                        commission = close.get("commission", 0)

                        try:
                            conn.execute(
                                """INSERT OR IGNORE INTO trades (
                                    trade_id, symbol, direction, strategy,
                                    open_timestamp, open_date, day_of_week,
                                    open_hour, open_minute,
                                    entry_price, exit_price,
                                    size, signal_confidence,
                                    regime,
                                    vix_at_entry, dax_pct_at_entry,
                                    slippage_entry,
                                    close_timestamp, close_date,
                                    close_hour, close_minute,
                                    pnl, pnl_pct, commission, net_pnl,
                                    exit_reason,
                                    duration_seconds, duration_minutes,
                                    mae_pct, mfe_pct, edge_captured,
                                    is_open, regime_source
                                ) VALUES (
                                    ?, ?, ?, ?,
                                    ?, ?, ?,
                                    ?, ?,
                                    ?, ?,
                                    ?, ?,
                                    ?,
                                    ?, ?,
                                    ?,
                                    ?, ?,
                                    ?, ?,
                                    ?, ?, ?, ?,
                                    ?,
                                    ?, ?,
                                    ?, ?, ?,
                                    ?, ?
                                )""",
                                (
                                    trade_id,
                                    close["symbol"],
                                    close["direction"],
                                    close.get("strategy", open_rec.get("strategy", "") if open_rec else ""),
                                    open_ts,
                                    open_date,
                                    close.get("day_of_week", ""),
                                    open_rec.get("hour", 0) if open_rec else close.get("hour", 0),
                                    0,
                                    close.get("entry_price", open_rec.get("entry_price", 0) if open_rec else 0),
                                    close.get("exit_price", 0),
                                    close.get("size", open_rec.get("size", 0) if open_rec else 0),
                                    open_rec.get("signal_confidence", close.get("signal_confidence", 0)) if open_rec else close.get("signal_confidence", 0),
                                    close.get("regime", open_rec.get("regime", "") if open_rec else ""),
                                    open_rec.get("vix", 0) if open_rec else 0,
                                    open_rec.get("dax_pct", 0) if open_rec else 0,
                                    open_rec.get("slippage", 0) if open_rec else 0,
                                    close.get("timestamp", ""),
                                    close.get("date", ""),
                                    close.get("hour", 0),
                                    0,
                                    pnl,
                                    close.get("pnl_pct", 0),
                                    commission,
                                    round(pnl - commission, 2),
                                    close.get("exit_reason", ""),
                                    close.get("duration_seconds", 0),
                                    close.get("duration_minutes", 0),
                                    close.get("mae_pct", 0),
                                    close.get("mfe_pct", 0),
                                    close.get("edge_captured", 0),
                                    0,  # is_open = false (closed trade)
                                    "adx",  # legacy trades used ADX
                                ),
                            )
                            total_trades += 1
                        except Exception as e:
                            logger.debug(f"[DB] Migration: failed to insert trade: {e}")

                    # Insert unmatched opens as still-open trades
                    for key, open_list in opens.items():
                        for open_rec in open_list:
                            trade_id = f"M_{open_rec['timestamp']}_{open_rec['symbol']}"
                            try:
                                conn.execute(
                                    """INSERT OR IGNORE INTO trades (
                                        trade_id, symbol, direction, strategy,
                                        open_timestamp, open_date, day_of_week,
                                        open_hour, open_minute,
                                        entry_price, size, signal_confidence,
                                        regime, vix_at_entry, dax_pct_at_entry,
                                        slippage_entry, is_open, regime_source
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                    (
                                        trade_id,
                                        open_rec["symbol"],
                                        open_rec["direction"],
                                        open_rec.get("strategy", ""),
                                        open_rec["timestamp"],
                                        open_rec["timestamp"][:10],
                                        open_rec.get("day_of_week", ""),
                                        open_rec.get("hour", 0),
                                        0,
                                        open_rec.get("entry_price", 0),
                                        open_rec.get("size", 0),
                                        open_rec.get("signal_confidence", 0),
                                        open_rec.get("regime", ""),
                                        open_rec.get("vix", 0),
                                        open_rec.get("dax_pct", 0),
                                        open_rec.get("slippage", 0),
                                        1,
                                        "adx",
                                    ),
                                )
                            except Exception:
                                pass

            # Import signals
            for sf in signal_files:
                with self._connection() as conn:
                    for line in sf.read_text().splitlines():
                        if not line.strip():
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if rec.get("type") == "scan_cycle":
                            try:
                                conn.execute(
                                    """INSERT INTO market_snapshots (
                                        timestamp, date, hour, minute,
                                        vix, dax_pct, sp500_pct,
                                        symbols_scanned, signals_found
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                    (
                                        rec.get("timestamp", ""),
                                        rec.get("timestamp", "")[:10],
                                        int(rec.get("timestamp", "T00:")[11:13]) if "T" in rec.get("timestamp", "") else 0,
                                        0,
                                        rec.get("vix"),
                                        rec.get("dax_pct"),
                                        rec.get("sp500_pct"),
                                        rec.get("symbols_scanned", 0),
                                        rec.get("signals_found", 0),
                                    ),
                                )
                            except Exception:
                                pass
                        elif "symbol" in rec and "action" in rec:
                            try:
                                indicators = rec.get("indicators", {})
                                conn.execute(
                                    """INSERT INTO signals (
                                        timestamp, date, day_of_week, hour, minute,
                                        symbol, direction, price, strategy,
                                        confidence,
                                        comp_zscore, comp_volume, comp_momentum,
                                        comp_volatility, comp_trend, comp_microstructure,
                                        stop_loss, take_profit, reason,
                                        regime, regime_source,
                                        vix, vix_change, dax_pct, sp500_pct,
                                        action, filter_reason
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                    (
                                        rec.get("timestamp", ""),
                                        rec.get("date", rec.get("timestamp", "")[:10]),
                                        rec.get("day_of_week", ""),
                                        rec.get("hour", 0),
                                        rec.get("minute", 0),
                                        rec.get("symbol", ""),
                                        rec.get("direction", ""),
                                        rec.get("price", 0),
                                        rec.get("strategy", ""),
                                        rec.get("confidence", 0),
                                        indicators.get("zscore"),
                                        indicators.get("volume"),
                                        indicators.get("momentum"),
                                        indicators.get("volatility"),
                                        indicators.get("trend"),
                                        indicators.get("microstructure"),
                                        rec.get("stop_loss"),
                                        rec.get("take_profit"),
                                        rec.get("reason", ""),
                                        rec.get("regime", ""),
                                        "adx",
                                        rec.get("vix"),
                                        rec.get("vix_change"),
                                        rec.get("dax_pct"),
                                        rec.get("sp500_pct"),
                                        rec.get("action", ""),
                                        rec.get("filter_reason", ""),
                                    ),
                                )
                                total_signals += 1
                            except Exception:
                                pass

            # Mark migration complete
            with self._connection() as conn:
                conn.execute(
                    "INSERT INTO schema_version (version, description) VALUES (2, 'JSONL migration complete')"
                )

            # Compute daily summaries for all dates
            dates_migrated = set()
            with self._connection() as conn:
                rows = conn.execute("SELECT DISTINCT close_date FROM trades WHERE close_date IS NOT NULL").fetchall()
                dates_migrated = {r["close_date"] for r in rows}

            for d in dates_migrated:
                self.update_daily_summary(d)

            logger.info(f"[DB] JSONL migration complete: {total_trades} trades, {total_signals} signals imported")

        except Exception as e:
            logger.warning(f"[DB] JSONL migration failed: {e}")


# ─── SQL Schema ─────────────────────────────────────────────────

SCHEMA_SQL = """
-- Schema versioning
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (1, 'Initial schema — full trading history with Renaissance modules');

-- Trades — complete lifecycle
CREATE TABLE IF NOT EXISTS trades (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id            TEXT UNIQUE NOT NULL,

    symbol              TEXT NOT NULL,
    direction           TEXT NOT NULL,
    strategy            TEXT NOT NULL DEFAULT '',

    open_timestamp      TEXT NOT NULL,
    close_timestamp     TEXT,
    open_date           TEXT NOT NULL,
    close_date          TEXT,
    day_of_week         TEXT NOT NULL DEFAULT '',
    open_hour           INTEGER NOT NULL DEFAULT 0,
    open_minute         INTEGER NOT NULL DEFAULT 0,
    close_hour          INTEGER,
    close_minute        INTEGER,

    entry_price         REAL NOT NULL DEFAULT 0,
    exit_price          REAL,
    stop_loss           REAL,
    take_profit         REAL,

    size                INTEGER NOT NULL DEFAULT 0,
    size_base           INTEGER,
    vix_size_multiplier REAL DEFAULT 1.0,
    hmm_size_multiplier REAL DEFAULT 1.0,

    order_id            TEXT,
    tick_size           REAL DEFAULT 0,
    slippage_entry      REAL DEFAULT 0,
    slippage_exit       REAL DEFAULT 0,
    is_bracket          INTEGER DEFAULT 0,

    signal_confidence   REAL NOT NULL DEFAULT 0,
    expected_edge       REAL DEFAULT 0,
    ml_win_probability  REAL,

    comp_zscore         REAL,
    comp_volume         REAL,
    comp_momentum       REAL,
    comp_volatility     REAL,
    comp_trend          REAL,
    comp_microstructure REAL,
    comp_correlation    REAL,

    regime              TEXT NOT NULL DEFAULT '',
    regime_source       TEXT DEFAULT 'adx',
    hmm_prob_low_vol    REAL,
    hmm_prob_normal     REAL,
    hmm_prob_high_vol   REAL,

    vix_at_entry        REAL,
    vix_change_at_entry REAL,
    dax_pct_at_entry    REAL,
    sp500_pct_at_entry  REAL,
    atr_at_entry        REAL,
    atr_pct_at_entry    REAL,
    rsi_at_entry        REAL,
    adx_at_entry        REAL,
    bb_width_at_entry   REAL,
    volume_ratio_entry  REAL,

    vix_at_exit         REAL,
    dax_pct_at_exit     REAL,
    sp500_pct_at_exit   REAL,
    regime_at_exit      TEXT,

    corr_score          REAL,
    corr_zscore         REAL,
    corr_rolling        REAL,
    corr_baseline       REAL,

    pnl                 REAL,
    pnl_pct             REAL,
    commission          REAL DEFAULT 0,
    net_pnl             REAL,
    exit_reason         TEXT,

    duration_seconds    INTEGER,
    duration_minutes    REAL,

    mae_pct             REAL,
    mfe_pct             REAL,
    edge_captured       REAL,
    mae_price           REAL,
    mfe_price           REAL,

    consecutive_wins    INTEGER DEFAULT 0,
    consecutive_losses  INTEGER DEFAULT 0,
    daily_trade_number  INTEGER DEFAULT 0,
    daily_pnl_before    REAL DEFAULT 0,
    account_balance     REAL,

    is_open             INTEGER DEFAULT 1,

    pair_trade_id       INTEGER REFERENCES pair_trades(id),
    pair_leg            TEXT,

    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Signals — every signal (executed + filtered + skipped)
CREATE TABLE IF NOT EXISTS signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,

    timestamp           TEXT NOT NULL,
    date                TEXT NOT NULL,
    day_of_week         TEXT NOT NULL DEFAULT '',
    hour                INTEGER NOT NULL DEFAULT 0,
    minute              INTEGER NOT NULL DEFAULT 0,

    symbol              TEXT NOT NULL,
    direction           TEXT NOT NULL,
    price               REAL NOT NULL DEFAULT 0,
    strategy            TEXT NOT NULL DEFAULT '',

    confidence          REAL NOT NULL DEFAULT 0,
    expected_edge       REAL DEFAULT 0,

    comp_zscore         REAL,
    comp_volume         REAL,
    comp_momentum       REAL,
    comp_volatility     REAL,
    comp_trend          REAL,
    comp_microstructure REAL,
    comp_correlation    REAL,

    stop_loss           REAL,
    take_profit         REAL,
    reason              TEXT,

    regime              TEXT NOT NULL DEFAULT '',
    regime_source       TEXT DEFAULT 'adx',
    hmm_prob_low_vol    REAL,
    hmm_prob_normal     REAL,
    hmm_prob_high_vol   REAL,

    vix                 REAL,
    vix_change          REAL,
    dax_pct             REAL,
    sp500_pct           REAL,
    atr                 REAL,
    atr_pct             REAL,
    rsi                 REAL,
    adx                 REAL,
    bb_width            REAL,
    volume_ratio        REAL,

    ml_win_probability  REAL,
    ml_allowed          INTEGER,

    corr_score          REAL,
    corr_zscore         REAL,

    action              TEXT NOT NULL DEFAULT '',
    filter_reason       TEXT DEFAULT '',

    trade_id            INTEGER REFERENCES trades(id),

    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Market snapshots — per scan cycle
CREATE TABLE IF NOT EXISTS market_snapshots (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT NOT NULL,
    date                TEXT NOT NULL,
    hour                INTEGER NOT NULL DEFAULT 0,
    minute              INTEGER NOT NULL DEFAULT 0,
    vix                 REAL,
    vix_change_pct      REAL,
    dax_pct             REAL,
    sp500_pct           REAL,
    symbols_scanned     INTEGER DEFAULT 0,
    signals_found       INTEGER DEFAULT 0,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Pair trades
CREATE TABLE IF NOT EXISTS pair_trades (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    pair_id                 TEXT NOT NULL,
    open_timestamp          TEXT NOT NULL,
    close_timestamp         TEXT,
    direction               TEXT NOT NULL,
    entry_spread            REAL NOT NULL,
    entry_zscore            REAL NOT NULL,
    exit_spread             REAL,
    exit_zscore             REAL,
    hedge_ratio             REAL NOT NULL,
    coint_pvalue            REAL,
    half_life               REAL,
    spread_mean             REAL,
    spread_std              REAL,
    leg_a_symbol            TEXT NOT NULL,
    leg_a_side              TEXT NOT NULL,
    leg_a_entry_price       REAL NOT NULL,
    leg_a_exit_price        REAL,
    leg_a_size              INTEGER NOT NULL,
    leg_a_order_id          TEXT,
    leg_b_symbol            TEXT NOT NULL,
    leg_b_side              TEXT NOT NULL,
    leg_b_entry_price       REAL NOT NULL,
    leg_b_exit_price        REAL,
    leg_b_size              INTEGER NOT NULL,
    leg_b_order_id          TEXT,
    pnl                     REAL,
    commission              REAL DEFAULT 5.0,
    net_pnl                 REAL,
    exit_reason             TEXT,
    duration_seconds        INTEGER,
    vix_at_entry            REAL,
    vix_at_exit             REAL,
    regime_at_entry         TEXT,
    status                  TEXT DEFAULT 'open',
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Daily summaries
CREATE TABLE IF NOT EXISTS daily_summaries (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    date                TEXT UNIQUE NOT NULL,
    day_of_week         TEXT NOT NULL DEFAULT '',
    trades_opened       INTEGER DEFAULT 0,
    trades_closed       INTEGER DEFAULT 0,
    signals_generated   INTEGER DEFAULT 0,
    signals_executed    INTEGER DEFAULT 0,
    signals_filtered    INTEGER DEFAULT 0,
    scan_cycles         INTEGER DEFAULT 0,
    gross_pnl           REAL DEFAULT 0,
    total_commission    REAL DEFAULT 0,
    net_pnl             REAL DEFAULT 0,
    wins                INTEGER DEFAULT 0,
    losses              INTEGER DEFAULT 0,
    win_rate            REAL DEFAULT 0,
    max_drawdown_pct    REAL DEFAULT 0,
    avg_winner          REAL DEFAULT 0,
    avg_loser           REAL DEFAULT 0,
    profit_factor       REAL DEFAULT 0,
    largest_winner      REAL DEFAULT 0,
    largest_loser       REAL DEFAULT 0,
    avg_hold_minutes    REAL DEFAULT 0,
    vix_open            REAL,
    vix_close           REAL,
    dax_change_pct      REAL,
    starting_balance    REAL,
    ending_balance      REAL,
    pair_trades_opened  INTEGER DEFAULT 0,
    pair_trades_closed  INTEGER DEFAULT 0,
    pair_pnl            REAL DEFAULT 0,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
CREATE INDEX IF NOT EXISTS idx_trades_regime ON trades(regime);
CREATE INDEX IF NOT EXISTS idx_trades_direction ON trades(direction);
CREATE INDEX IF NOT EXISTS idx_trades_exit_reason ON trades(exit_reason);
CREATE INDEX IF NOT EXISTS idx_trades_open_date ON trades(open_date);
CREATE INDEX IF NOT EXISTS idx_trades_open_hour ON trades(open_hour);
CREATE INDEX IF NOT EXISTS idx_trades_is_open ON trades(is_open);
CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trades(pnl);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_strategy ON trades(symbol, strategy);
CREATE INDEX IF NOT EXISTS idx_trades_regime_direction ON trades(regime, direction);
CREATE INDEX IF NOT EXISTS idx_trades_date_hour ON trades(open_date, open_hour);

CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_action ON signals(action);
CREATE INDEX IF NOT EXISTS idx_signals_date ON signals(date);
CREATE INDEX IF NOT EXISTS idx_signals_strategy ON signals(strategy);
CREATE INDEX IF NOT EXISTS idx_signals_regime ON signals(regime);

CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON market_snapshots(timestamp);
CREATE INDEX IF NOT EXISTS idx_snapshots_date ON market_snapshots(date);
CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_summaries(date);
CREATE INDEX IF NOT EXISTS idx_pairs_pair_id ON pair_trades(pair_id);
CREATE INDEX IF NOT EXISTS idx_pairs_status ON pair_trades(status);
"""
