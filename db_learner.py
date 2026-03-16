"""
DB Self-Learner — Feeds database insights back into trading behavior.

Unlike learning.py (which does mechanical parameter grid search on backtests),
this module mines the LIVE trade history database for actionable patterns and
generates a learning_insights.json file that the trading system reads at startup
to adjust its behavior.

Self-learning loop:
    1. Query trade_history.db for performance patterns
    2. Identify what works (best symbols, hours, regimes, strategies)
    3. Identify what fails (worst exit reasons, drawdown patterns)
    4. Generate concrete adjustments (symbol weights, hour blocks, confidence boosts)
    5. Write to knowledge/learning_insights.json
    6. Trading system loads insights at startup and applies them
    7. Track whether recommendations improved performance (meta-learning)

Safety:
    - All adjustments are bounded (never fully block a symbol, just reduce weight)
    - Minimum trade count thresholds prevent overfitting to small samples
    - Recommendations expire after N days to prevent stale adjustments
    - Every insight includes reasoning for human review
"""

import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

INSIGHTS_PATH = Path("knowledge") / "learning_insights.json"
HISTORY_PATH = Path("knowledge") / "learning_history.json"


@dataclass
class SymbolAdjustment:
    """Per-symbol trading weight adjustment."""
    symbol: str
    weight: float               # 0.0 (avoid) to 2.0 (prefer), 1.0 = neutral
    reason: str
    trade_count: int
    win_rate: float
    avg_pnl: float
    expires: str                # ISO date when this adjustment expires


@dataclass
class HourAdjustment:
    """Per-hour trading preference adjustment."""
    hour: int
    weight: float               # 0.0 (block) to 2.0 (prefer), 1.0 = neutral
    reason: str
    trade_count: int
    win_rate: float


@dataclass
class StrategyAdjustment:
    """Per-strategy/regime adjustment."""
    strategy: str
    regime: str
    confidence_boost: float     # Add to confidence score (-0.2 to +0.2)
    reason: str
    trade_count: int
    win_rate: float


@dataclass
class RiskAdjustment:
    """Dynamic risk parameter adjustment."""
    parameter: str              # e.g., "risk_per_trade", "max_trades_per_day"
    suggested_value: float
    current_value: float
    reason: str


@dataclass
class LearningInsights:
    """Complete set of insights for the trading system to apply."""
    generated_at: str
    lookback_days: int
    total_trades_analyzed: int
    total_closed_trades: int
    overall_win_rate: float
    overall_pnl: float
    symbol_adjustments: list[dict] = field(default_factory=list)
    hour_adjustments: list[dict] = field(default_factory=list)
    strategy_adjustments: list[dict] = field(default_factory=list)
    risk_adjustments: list[dict] = field(default_factory=list)
    exit_reason_insights: list[dict] = field(default_factory=list)
    meta_notes: list[str] = field(default_factory=list)
    # Meta-learning: track whether previous insights improved performance
    previous_recommendation_results: list[dict] = field(default_factory=list)


class DBSelfLearner:
    """
    Mines the trade database for actionable patterns and generates
    adjustment recommendations that the trading system reads at startup.

    Unlike the mechanical SelfLearner (learning.py) which does grid search,
    this module does data-driven pattern discovery on live trade outcomes.
    """

    # Minimum trades required for statistical confidence
    MIN_TRADES_TOTAL = 10           # Lowered from 15: start learning faster in early days
    MIN_TRADES_PER_GROUP = 3        # Lowered from 5: act on smaller samples (with wider bounds)
    INSIGHT_EXPIRY_DAYS = 14        # Insights expire after 2 weeks
    LOOKBACK_DAYS = 30              # Analyze last 30 days of trades

    # Adjustment bounds — MORE AGGRESSIVE based on live results showing
    # mild adjustments (0.8 weight) weren't enough to prevent losses
    MIN_WEIGHT = 0.2                # Can reduce to 20% (was 30%)
    MAX_WEIGHT = 2.0                # Can boost to 200% (was 180%)
    MAX_CONFIDENCE_BOOST = 0.20     # Max ±20% confidence adjustment (was 15%)
    MAX_RISK_ADJUSTMENT = 0.4       # Max 40% change to risk parameters (was 30%)

    def __init__(self, db=None):
        """
        Args:
            db: TradeDatabase instance (optional, will try to import if not provided)
        """
        self.db = db
        if self.db is None:
            try:
                from trade_database import TradeDatabase
                self.db = TradeDatabase()
            except Exception as e:
                logger.warning(f"[DBLearner] Could not connect to trade DB: {e}")

        self._insights: Optional[LearningInsights] = None
        self._history: list[dict] = self._load_history()

    # ─── Main Entry Points ───────────────────────────────────────

    def generate_insights(self, lookback_days: int = None) -> Optional[LearningInsights]:
        """
        Run full analysis cycle and generate learning_insights.json.

        Called at:
        - Startup (before trading begins)
        - End of day (after all positions closed)
        - On demand (scheduled task)
        """
        if self.db is None:
            logger.warning("[DBLearner] No database — skipping insight generation")
            return None

        lookback = lookback_days or self.LOOKBACK_DAYS

        # Get trade data
        try:
            trades_df = self._get_closed_trades(lookback)
        except Exception as e:
            logger.error(f"[DBLearner] Failed to query trades: {e}")
            return None

        if trades_df is None or len(trades_df) < self.MIN_TRADES_TOTAL:
            logger.info(
                f"[DBLearner] Need {self.MIN_TRADES_TOTAL} closed trades, "
                f"have {len(trades_df) if trades_df is not None else 0} — skipping"
            )
            return None

        n_closed = len(trades_df)
        overall_wr = float((trades_df["pnl"] > 0).mean()) if n_closed > 0 else 0
        overall_pnl = float(trades_df["pnl"].sum()) if n_closed > 0 else 0

        logger.info(f"[DBLearner] Analyzing {n_closed} trades over {lookback} days...")

        insights = LearningInsights(
            generated_at=datetime.now().isoformat(),
            lookback_days=lookback,
            total_trades_analyzed=n_closed,
            total_closed_trades=n_closed,
            overall_win_rate=round(overall_wr, 4),
            overall_pnl=round(overall_pnl, 2),
        )

        # Run all analysis modules
        insights.symbol_adjustments = self._analyze_symbols(trades_df)
        insights.hour_adjustments = self._analyze_hours(trades_df)
        insights.strategy_adjustments = self._analyze_strategies(trades_df)
        insights.risk_adjustments = self._analyze_risk(trades_df)
        insights.exit_reason_insights = self._analyze_exit_reasons(trades_df)
        insights.meta_notes = self._generate_meta_notes(trades_df)

        # Meta-learning: evaluate previous recommendations
        insights.previous_recommendation_results = self._evaluate_previous_recommendations(trades_df)

        # Save
        self._insights = insights
        self._save_insights(insights)
        self._save_to_history(insights)

        n_adjustments = (
            len(insights.symbol_adjustments) +
            len(insights.hour_adjustments) +
            len(insights.strategy_adjustments) +
            len(insights.risk_adjustments)
        )
        logger.info(
            f"[DBLearner] Generated {n_adjustments} adjustments "
            f"(symbols={len(insights.symbol_adjustments)}, "
            f"hours={len(insights.hour_adjustments)}, "
            f"strategies={len(insights.strategy_adjustments)}, "
            f"risk={len(insights.risk_adjustments)})"
        )

        return insights

    def load_insights(self) -> Optional[dict]:
        """Load current insights from disk (called by trading system at startup)."""
        if not INSIGHTS_PATH.exists():
            return None
        try:
            with open(INSIGHTS_PATH) as f:
                data = json.load(f)
            # Check expiry
            generated = datetime.fromisoformat(data.get("generated_at", "2000-01-01"))
            age_days = (datetime.now() - generated).days
            if age_days > self.INSIGHT_EXPIRY_DAYS:
                logger.info(f"[DBLearner] Insights are {age_days}d old — expired, regenerating")
                return None
            return data
        except Exception as e:
            logger.warning(f"[DBLearner] Failed to load insights: {e}")
            return None

    def get_symbol_weight(self, symbol: str) -> float:
        """Get trading weight for a symbol (1.0 = neutral). Used by signal execution."""
        insights = self._insights or self._load_insights_obj()
        if not insights:
            return 1.0
        for adj in insights.get("symbol_adjustments", []):
            if adj.get("symbol") == symbol:
                # Check expiry
                exp = adj.get("expires", "2099-12-31")
                if datetime.fromisoformat(exp).date() >= date.today():
                    return float(adj.get("weight", 1.0))
        return 1.0

    def get_hour_weight(self, hour: int) -> float:
        """Get trading weight for an hour (1.0 = neutral). Used by signal execution."""
        insights = self._insights or self._load_insights_obj()
        if not insights:
            return 1.0
        for adj in insights.get("hour_adjustments", []):
            if adj.get("hour") == hour:
                return float(adj.get("weight", 1.0))
        return 1.0

    def get_confidence_boost(self, strategy: str, regime: str) -> float:
        """Get confidence adjustment for strategy+regime combo. Used by signal execution."""
        insights = self._insights or self._load_insights_obj()
        if not insights:
            return 0.0
        for adj in insights.get("strategy_adjustments", []):
            if adj.get("strategy") == strategy and adj.get("regime") == regime:
                return float(adj.get("confidence_boost", 0.0))
        return 0.0

    def get_risk_adjustments(self) -> dict:
        """Get all risk parameter adjustments. Used by main.py at startup."""
        insights = self._insights or self._load_insights_obj()
        if not insights:
            return {}
        return {
            adj["parameter"]: adj["suggested_value"]
            for adj in insights.get("risk_adjustments", [])
        }

    # ─── Analysis Modules ────────────────────────────────────────

    def _analyze_symbols(self, df: pd.DataFrame) -> list[dict]:
        """Identify which symbols to prefer or avoid."""
        adjustments = []
        grouped = df.groupby("symbol")

        for symbol, group in grouped:
            n = len(group)
            if n < self.MIN_TRADES_PER_GROUP:
                continue

            win_rate = float((group["pnl"] > 0).mean())
            avg_pnl = float(group["pnl"].mean())
            total_pnl = float(group["pnl"].sum())
            avg_duration = float(group["duration_seconds"].mean()) if "duration_seconds" in group.columns else 0

            # Calculate weight based on performance
            weight = 1.0
            reason_parts = []

            # Win rate component
            if win_rate >= 0.65:
                weight += 0.3
                reason_parts.append(f"high win rate {win_rate:.0%}")
            elif win_rate >= 0.55:
                weight += 0.1
                reason_parts.append(f"good win rate {win_rate:.0%}")
            elif win_rate < 0.35:
                weight -= 0.5
                reason_parts.append(f"low win rate {win_rate:.0%}")
            elif win_rate < 0.45:
                weight -= 0.2
                reason_parts.append(f"below-avg win rate {win_rate:.0%}")

            # PnL component
            if avg_pnl > 0:
                if total_pnl > 10:  # Meaningful positive PnL
                    weight += 0.1
                    reason_parts.append(f"profitable (€{total_pnl:.1f} total)")
            else:
                if total_pnl < -10:
                    weight -= 0.2
                    reason_parts.append(f"unprofitable (€{total_pnl:.1f} total)")

            # Bound the weight
            weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, weight))

            if abs(weight - 1.0) >= 0.1:  # Only record meaningful adjustments
                adjustments.append({
                    "symbol": symbol,
                    "weight": round(weight, 2),
                    "reason": "; ".join(reason_parts) if reason_parts else "neutral",
                    "trade_count": n,
                    "win_rate": round(win_rate, 4),
                    "avg_pnl": round(avg_pnl, 2),
                    "expires": (date.today() + timedelta(days=self.INSIGHT_EXPIRY_DAYS)).isoformat(),
                })

        return adjustments

    def _analyze_hours(self, df: pd.DataFrame) -> list[dict]:
        """Identify best and worst trading hours."""
        adjustments = []

        if "open_hour" not in df.columns:
            return adjustments

        grouped = df.groupby("open_hour")

        for hour, group in grouped:
            n = len(group)
            if n < self.MIN_TRADES_PER_GROUP:
                continue

            win_rate = float((group["pnl"] > 0).mean())
            avg_pnl = float(group["pnl"].mean())

            weight = 1.0
            reason_parts = []

            if win_rate >= 0.65:
                weight += 0.3
                reason_parts.append(f"hour {hour}: high win rate {win_rate:.0%}")
            elif win_rate >= 0.55:
                weight += 0.1
                reason_parts.append(f"hour {hour}: good win rate {win_rate:.0%}")
            elif win_rate < 0.35:
                weight -= 0.4
                reason_parts.append(f"hour {hour}: poor win rate {win_rate:.0%}")
            elif win_rate < 0.45:
                weight -= 0.2
                reason_parts.append(f"hour {hour}: below-avg win rate {win_rate:.0%}")

            weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, weight))

            if abs(weight - 1.0) >= 0.1:
                adjustments.append({
                    "hour": int(hour),
                    "weight": round(weight, 2),
                    "reason": "; ".join(reason_parts),
                    "trade_count": n,
                    "win_rate": round(win_rate, 4),
                })

        return adjustments

    def _analyze_strategies(self, df: pd.DataFrame) -> list[dict]:
        """Identify which strategy+regime combos are working."""
        adjustments = []

        if "strategy" not in df.columns or "regime" not in df.columns:
            return adjustments

        grouped = df.groupby(["strategy", "regime"])

        for (strategy, regime), group in grouped:
            n = len(group)
            if n < self.MIN_TRADES_PER_GROUP:
                continue

            win_rate = float((group["pnl"] > 0).mean())
            avg_pnl = float(group["pnl"].mean())

            boost = 0.0
            reason_parts = []

            if win_rate >= 0.65 and avg_pnl > 0:
                boost = 0.10
                reason_parts.append(f"{strategy}/{regime}: strong ({win_rate:.0%} WR, €{avg_pnl:.1f}/trade)")
            elif win_rate >= 0.55 and avg_pnl > 0:
                boost = 0.05
                reason_parts.append(f"{strategy}/{regime}: good ({win_rate:.0%} WR)")
            elif win_rate < 0.40:
                boost = -0.10
                reason_parts.append(f"{strategy}/{regime}: weak ({win_rate:.0%} WR, €{avg_pnl:.1f}/trade)")
            elif win_rate < 0.45 and avg_pnl < 0:
                boost = -0.05
                reason_parts.append(f"{strategy}/{regime}: below par ({win_rate:.0%} WR)")

            boost = max(-self.MAX_CONFIDENCE_BOOST, min(self.MAX_CONFIDENCE_BOOST, boost))

            if abs(boost) >= 0.03:
                adjustments.append({
                    "strategy": str(strategy),
                    "regime": str(regime),
                    "confidence_boost": round(boost, 3),
                    "reason": "; ".join(reason_parts),
                    "trade_count": n,
                    "win_rate": round(win_rate, 4),
                })

        return adjustments

    def _analyze_risk(self, df: pd.DataFrame) -> list[dict]:
        """Suggest risk parameter adjustments based on outcomes."""
        adjustments = []

        if len(df) < self.MIN_TRADES_TOTAL:
            return adjustments

        # Analyze consecutive losses
        if "pnl" in df.columns:
            pnl_series = df.sort_values("close_timestamp" if "close_timestamp" in df.columns else "open_timestamp")["pnl"]
            max_consec_losses = 0
            current_streak = 0
            for pnl in pnl_series:
                if pnl < 0:
                    current_streak += 1
                    max_consec_losses = max(max_consec_losses, current_streak)
                else:
                    current_streak = 0

            # If we're hitting max_consecutive_losses often, might need to adjust
            if max_consec_losses >= 4:
                adjustments.append({
                    "parameter": "cooldown_after_loss",
                    "suggested_value": 180,     # Increase to 3 minutes
                    "current_value": 120,
                    "reason": f"Max {max_consec_losses} consecutive losses — increase cooldown",
                })

        # Analyze win rate vs position sizing
        overall_wr = float((df["pnl"] > 0).mean())
        if overall_wr < 0.45:
            adjustments.append({
                "parameter": "risk_per_trade",
                "suggested_value": 0.003,   # Reduce from 0.5% to 0.3%
                "current_value": 0.005,
                "reason": f"Low overall win rate ({overall_wr:.0%}) — reduce risk per trade",
            })
        elif overall_wr > 0.60:
            adjustments.append({
                "parameter": "risk_per_trade",
                "suggested_value": 0.006,   # Increase slightly from 0.5% to 0.6%
                "current_value": 0.005,
                "reason": f"Strong win rate ({overall_wr:.0%}) — slightly increase risk per trade",
            })

        # Analyze SL/TP hit rates
        if "exit_reason" in df.columns:
            sl_hits = (df["exit_reason"] == "stop_loss").sum()
            tp_hits = (df["exit_reason"] == "take_profit").sum()
            total = len(df)

            if total > 0 and sl_hits / total > 0.4:
                adjustments.append({
                    "parameter": "stop_loss_suggestion",
                    "suggested_value": "widen_stops",
                    "current_value": "current",
                    "reason": f"High SL hit rate ({sl_hits/total:.0%}) — stops may be too tight",
                })

            if total > 0 and tp_hits / total < 0.15 and overall_wr > 0.50:
                adjustments.append({
                    "parameter": "take_profit_suggestion",
                    "suggested_value": "tighten_tp",
                    "current_value": "current",
                    "reason": f"Low TP hit rate ({tp_hits/total:.0%}) with decent WR — TPs may be too ambitious",
                })

        return adjustments

    def _analyze_exit_reasons(self, df: pd.DataFrame) -> list[dict]:
        """Analyze exit reasons to understand trade quality."""
        insights = []

        if "exit_reason" not in df.columns:
            return insights

        exit_counts = df["exit_reason"].value_counts()
        total = len(df)

        for reason, count in exit_counts.items():
            pct = count / total
            group = df[df["exit_reason"] == reason]
            avg_pnl = float(group["pnl"].mean())
            avg_duration = float(group["duration_seconds"].mean()) if "duration_seconds" in group.columns else 0

            insights.append({
                "exit_reason": str(reason),
                "count": int(count),
                "pct": round(pct, 4),
                "avg_pnl": round(avg_pnl, 2),
                "avg_duration_min": round(avg_duration / 60, 1) if avg_duration > 0 else 0,
                "assessment": self._assess_exit_reason(str(reason), pct, avg_pnl),
            })

        return insights

    def _assess_exit_reason(self, reason: str, pct: float, avg_pnl: float) -> str:
        """Generate human-readable assessment of an exit reason."""
        if reason == "stop_loss":
            if pct > 0.40:
                return "WARNING: SL hit rate too high — stops may be too tight or entries too late"
            elif pct > 0.25:
                return "Normal SL rate — acceptable risk management"
            else:
                return "Low SL rate — good entry quality"
        elif reason == "take_profit":
            if pct > 0.30:
                return "Strong TP hit rate — strategy capturing targets well"
            elif pct > 0.15:
                return "Moderate TP rate — room for improvement"
            else:
                return "Low TP rate — consider tighter targets or better entries"
        elif reason == "time_exit" or reason == "max_hold":
            if avg_pnl < 0:
                return "Time exits losing money — consider shorter hold times"
            else:
                return "Time exits profitable — hold period is appropriate"
        elif reason == "trailing_stop":
            return "Trailing stops working — letting winners run"
        elif reason == "eod_close":
            if avg_pnl > 0:
                return "EOD closes profitable — system capturing intraday moves"
            else:
                return "EOD closes underwater — consider earlier exit criteria"
        else:
            return f"Exit reason '{reason}': avg PnL €{avg_pnl:.2f}"

    def _generate_meta_notes(self, df: pd.DataFrame) -> list[str]:
        """Generate high-level observations about trading performance."""
        notes = []

        if len(df) == 0:
            return ["No trades to analyze"]

        win_rate = float((df["pnl"] > 0).mean())
        total_pnl = float(df["pnl"].sum())
        avg_pnl = float(df["pnl"].mean())

        # Overall assessment
        if win_rate >= 0.55 and total_pnl > 0:
            notes.append(f"System is profitable: {win_rate:.0%} WR, €{total_pnl:.2f} total PnL")
        elif win_rate >= 0.50 and total_pnl > 0:
            notes.append(f"Marginally profitable: {win_rate:.0%} WR — focus on improving edge")
        elif total_pnl < 0:
            notes.append(f"System unprofitable: {win_rate:.0%} WR, €{total_pnl:.2f} — needs attention")

        # Direction analysis
        if "direction" in df.columns:
            for direction in ["LONG", "SHORT"]:
                dir_df = df[df["direction"] == direction]
                if len(dir_df) >= self.MIN_TRADES_PER_GROUP:
                    dir_wr = float((dir_df["pnl"] > 0).mean())
                    dir_pnl = float(dir_df["pnl"].sum())
                    if dir_wr >= 0.55:
                        notes.append(f"{direction}s performing well: {dir_wr:.0%} WR, €{dir_pnl:.2f}")
                    elif dir_wr < 0.40:
                        notes.append(f"{direction}s underperforming: {dir_wr:.0%} WR, €{dir_pnl:.2f}")

        # Trend over time
        if "close_timestamp" in df.columns and len(df) >= 10:
            # Compare first half vs second half
            midpoint = len(df) // 2
            first_half = df.iloc[:midpoint]
            second_half = df.iloc[midpoint:]
            first_wr = float((first_half["pnl"] > 0).mean())
            second_wr = float((second_half["pnl"] > 0).mean())
            if second_wr > first_wr + 0.10:
                notes.append(f"Performance IMPROVING: WR {first_wr:.0%} → {second_wr:.0%}")
            elif second_wr < first_wr - 0.10:
                notes.append(f"Performance DECLINING: WR {first_wr:.0%} → {second_wr:.0%}")

        # Sample size warning
        if len(df) < 30:
            notes.append(f"Low sample size ({len(df)} trades) — insights may not be statistically significant")

        return notes

    def _evaluate_previous_recommendations(self, df: pd.DataFrame) -> list[dict]:
        """Meta-learning: check if our previous recommendations improved performance."""
        results = []

        if not self._history:
            return results

        # Find the most recent previous insight set
        previous = self._history[-1] if self._history else None
        if not previous:
            return results

        prev_date = previous.get("generated_at", "")
        if not prev_date:
            return results

        try:
            prev_dt = datetime.fromisoformat(prev_date)
        except (ValueError, TypeError):
            return results

        # Compare performance before and after recommendations were applied
        # (simplified: compare overall win rate from previous report to current)
        prev_wr = previous.get("overall_win_rate", 0)
        curr_wr = float((df["pnl"] > 0).mean()) if len(df) > 0 else 0

        prev_pnl = previous.get("overall_pnl", 0)
        curr_pnl = float(df["pnl"].sum()) if len(df) > 0 else 0

        delta_wr = curr_wr - prev_wr
        delta_pnl = curr_pnl - prev_pnl

        assessment = "neutral"
        if delta_wr > 0.05 and delta_pnl > 0:
            assessment = "positive"
        elif delta_wr < -0.05 and delta_pnl < 0:
            assessment = "negative"

        results.append({
            "period": f"since {prev_dt.strftime('%Y-%m-%d')}",
            "previous_win_rate": round(prev_wr, 4),
            "current_win_rate": round(curr_wr, 4),
            "win_rate_delta": round(delta_wr, 4),
            "pnl_delta": round(delta_pnl, 2),
            "assessment": assessment,
            "note": (
                "Recommendations appear to be helping" if assessment == "positive"
                else "Performance declined since last recommendations" if assessment == "negative"
                else "No significant change"
            ),
        })

        return results

    # ─── Data Access ─────────────────────────────────────────────

    def _get_closed_trades(self, lookback_days: int) -> Optional[pd.DataFrame]:
        """Query closed trades from the database."""
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

        with self.db._connection() as conn:
            query = """
                SELECT
                    trade_id, symbol, direction, strategy, regime,
                    open_timestamp, close_timestamp, open_hour,
                    entry_price, exit_price, stop_loss, take_profit,
                    size, pnl, pnl_pct, commission, net_pnl,
                    exit_reason, duration_seconds,
                    signal_confidence AS confidence, expected_edge, ml_win_probability,
                    vix_at_entry, regime_source,
                    comp_zscore, comp_volume, comp_momentum, comp_volatility,
                    comp_trend, comp_microstructure, comp_correlation,
                    atr_at_entry, rsi_at_entry, adx_at_entry,
                    mae_pct, mfe_pct, edge_captured,
                    consecutive_wins, consecutive_losses,
                    daily_trade_number, daily_pnl_before
                FROM trades
                WHERE is_open = 0
                  AND open_timestamp >= ?
                ORDER BY close_timestamp ASC
            """
            df = pd.read_sql_query(query, conn, params=(cutoff,))

        return df if len(df) > 0 else None

    # ─── Persistence ─────────────────────────────────────────────

    def _save_insights(self, insights: LearningInsights):
        """Save insights to disk for the trading system to read."""
        INSIGHTS_PATH.parent.mkdir(exist_ok=True)
        data = asdict(insights)
        with open(INSIGHTS_PATH, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"[DBLearner] Insights saved to {INSIGHTS_PATH}")

    def _save_to_history(self, insights: LearningInsights):
        """Append to learning history for meta-learning tracking."""
        summary = {
            "generated_at": insights.generated_at,
            "total_closed_trades": insights.total_closed_trades,
            "overall_win_rate": insights.overall_win_rate,
            "overall_pnl": insights.overall_pnl,
            "n_symbol_adjustments": len(insights.symbol_adjustments),
            "n_hour_adjustments": len(insights.hour_adjustments),
            "n_strategy_adjustments": len(insights.strategy_adjustments),
            "n_risk_adjustments": len(insights.risk_adjustments),
            "meta_notes": insights.meta_notes[:3],  # First 3 notes
        }
        self._history.append(summary)
        # Keep last 30 entries
        self._history = self._history[-30:]
        try:
            with open(HISTORY_PATH, "w") as f:
                json.dump(self._history, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[DBLearner] Failed to save history: {e}")

    def _load_history(self) -> list[dict]:
        """Load learning history from disk."""
        if not HISTORY_PATH.exists():
            return []
        try:
            with open(HISTORY_PATH) as f:
                return json.load(f)
        except Exception:
            return []

    def _load_insights_obj(self) -> Optional[dict]:
        """Load insights as a dict (for getter methods)."""
        if self._insights:
            return asdict(self._insights)
        return self.load_insights()

    # ─── Formatting ──────────────────────────────────────────────

    def format_insights_summary(self) -> str:
        """Format insights for logging/display."""
        insights = self._insights or self._load_insights_obj()
        if not insights:
            return "[DBLearner] No insights available"

        data = insights if isinstance(insights, dict) else asdict(insights)
        lines = []
        lines.append("=" * 60)
        lines.append("DB SELF-LEARNER — INSIGHTS SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Analyzed: {data.get('total_closed_trades', 0)} closed trades")
        lines.append(f"Win rate: {data.get('overall_win_rate', 0):.1%}")
        lines.append(f"Total PnL: €{data.get('overall_pnl', 0):.2f}")
        lines.append("")

        # Symbol adjustments
        sym_adj = data.get("symbol_adjustments", [])
        if sym_adj:
            lines.append("── Symbol Weights ──")
            for sa in sym_adj:
                direction = "↑" if sa["weight"] > 1.0 else "↓"
                lines.append(
                    f"  {direction} {sa['symbol']:10s} weight={sa['weight']:.2f} "
                    f"({sa['trade_count']} trades, {sa['win_rate']:.0%} WR) — {sa['reason']}"
                )
            lines.append("")

        # Hour adjustments
        hour_adj = data.get("hour_adjustments", [])
        if hour_adj:
            lines.append("── Hour Weights ──")
            for ha in sorted(hour_adj, key=lambda x: x["hour"]):
                direction = "↑" if ha["weight"] > 1.0 else "↓"
                lines.append(
                    f"  {direction} {ha['hour']:02d}:00 weight={ha['weight']:.2f} "
                    f"({ha['trade_count']} trades, {ha['win_rate']:.0%} WR)"
                )
            lines.append("")

        # Strategy adjustments
        strat_adj = data.get("strategy_adjustments", [])
        if strat_adj:
            lines.append("── Strategy/Regime Boosts ──")
            for sa in strat_adj:
                direction = "+" if sa["confidence_boost"] > 0 else ""
                lines.append(
                    f"  {sa['strategy']}/{sa['regime']}: "
                    f"conf {direction}{sa['confidence_boost']:.2f} — {sa['reason']}"
                )
            lines.append("")

        # Risk adjustments
        risk_adj = data.get("risk_adjustments", [])
        if risk_adj:
            lines.append("── Risk Adjustments ──")
            for ra in risk_adj:
                lines.append(f"  {ra['parameter']}: {ra['reason']}")
            lines.append("")

        # Meta notes
        notes = data.get("meta_notes", [])
        if notes:
            lines.append("── Notes ──")
            for note in notes:
                lines.append(f"  • {note}")

        # Meta-learning results
        prev = data.get("previous_recommendation_results", [])
        if prev:
            lines.append("")
            lines.append("── Meta-Learning ──")
            for p in prev:
                lines.append(f"  {p['note']} (WR: {p['win_rate_delta']:+.1%})")

        lines.append("=" * 60)
        return "\n".join(lines)
