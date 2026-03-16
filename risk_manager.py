"""
Risk Manager
Position sizing, stop-loss management, daily risk limits, and trade lifecycle.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Optional
from enum import Enum

from strategy_v2 import Signal, SignalType, MarketRegime

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED_TP = "closed_tp"       # Hit take-profit
    CLOSED_SL = "closed_sl"       # Hit stop-loss
    CLOSED_TRAIL = "closed_trail" # Trailing stop
    CLOSED_TIME = "closed_time"   # Time-based exit
    CLOSED_EOD = "closed_eod"     # End of day
    CLOSED_MANUAL = "closed_manual"


@dataclass
class Position:
    symbol: str
    side: SignalType
    entry_price: float
    size: float               # Number of shares/units
    stop_loss: float
    take_profit: float
    entry_time: datetime
    signal: Signal
    status: PositionStatus = PositionStatus.OPEN
    current_price: float = 0.0
    trailing_stop: Optional[float] = None
    highest_price: float = 0.0     # For trailing stop (long)
    lowest_price: float = float("inf")  # For trailing stop (short)
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    commission: float = 0.0          # Total round-trip commission (entry + exit)
    # MAE/MFE tracking (for institutional analytics)
    mae_price: float = 0.0           # Maximum Adverse Excursion price
    mfe_price: float = 0.0           # Maximum Favorable Excursion price
    mae_pct: Optional[float] = None  # MAE as percentage of entry
    mfe_pct: Optional[float] = None  # MFE as percentage of entry
    slippage: Optional[float] = None # Signal price vs actual fill price delta
    has_bracket: bool = False            # True if IBKR bracket order handles SL/TP
    bracket_order_id: str = ""           # Executor order ID for bracket (to cancel legs)
    # ATR-based trailing stop (set by V2 strategy at position open)
    atr_trail_activation: float = 0.0    # Activate trailing at this absolute profit (e.g., 0.5 * ATR)
    atr_trail_distance: float = 0.0      # Trail by this distance behind best price (e.g., 0.3 * ATR)

    @property
    def unrealized_pnl(self) -> float:
        if self.side == SignalType.LONG:
            return (self.current_price - self.entry_price) * self.size
        else:
            return (self.entry_price - self.current_price) * self.size

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0
        if self.side == SignalType.LONG:
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price


class RiskManager:
    """
    Manages position sizing, stop-loss tracking, and daily risk limits.
    The 0.5% stop-loss is enforced here — no trade can bypass it.
    """

    def __init__(self, risk_config):
        self.rc = risk_config
        self.balance = risk_config.initial_balance
        self.positions: list[Position] = []
        self.closed_positions: list[Position] = []
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.consecutive_losses: int = 0
        self.last_loss_time: Optional[datetime] = None
        self._current_date: date = date.today()

    @staticmethod
    def _to_naive_datetime(ts) -> datetime:
        """Convert any timestamp (pandas Timestamp, tz-aware datetime) to naive datetime.
        Prevents 'Cannot subtract tz-naive and tz-aware datetime-like objects' errors."""
        if ts is None:
            return datetime.now()
        if isinstance(ts, datetime):
            # Handle tz-aware datetimes (including pandas Timestamps which subclass datetime)
            if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                return ts.replace(tzinfo=None)
            return ts
        # Fallback: not a datetime at all
        return datetime.now()

    def can_trade(self) -> tuple[bool, str]:
        """Check if trading is allowed based on risk limits."""
        today = date.today()
        if today != self._current_date:
            self._reset_daily()
            self._current_date = today

        # Daily loss limit
        if abs(self.daily_pnl) > self.balance * self.rc.max_daily_loss_pct and self.daily_pnl < 0:
            return False, f"Daily loss limit reached: {self.daily_pnl:.2f}"

        # Max consecutive losses
        if self.consecutive_losses >= self.rc.max_consecutive_losses:
            return False, f"Max consecutive losses reached: {self.consecutive_losses}"

        # Max trades per day
        if self.daily_trades >= self.rc.max_trades_per_day:
            return False, f"Max daily trades reached: {self.daily_trades}"

        # Max open positions
        open_positions = [p for p in self.positions if p.status == PositionStatus.OPEN]
        if len(open_positions) >= self.rc.max_open_positions:
            return False, f"Max open positions reached: {len(open_positions)}"

        # Cooldown after loss
        if self.last_loss_time:
            elapsed = (datetime.now() - self.last_loss_time).total_seconds()
            if elapsed < self.rc.cooldown_after_loss:
                remaining = self.rc.cooldown_after_loss - elapsed
                return False, f"Cooldown active: {remaining:.0f}s remaining"

        return True, "OK"

    def calculate_position_size(self, signal: Signal) -> float:
        """
        Calculate position size based on fixed-fractional risk management.

        Risk per trade = 0.5% of account
        Position size = Risk / Stop distance
        """
        # Guard: reject zero or negative prices (prevents division by zero)
        if signal.price <= 0:
            logger.warning(f"Signal price is {signal.price} for {signal.symbol} — skipping trade")
            return 0

        risk_amount = self.balance * self.rc.risk_per_trade
        stop_distance = abs(signal.price - signal.stop_loss)

        if stop_distance == 0:
            logger.warning("Stop distance is 0 — skipping trade")
            return 0

        size = risk_amount / stop_distance

        # Cap at max position percentage
        max_value = self.balance * self.rc.max_position_pct
        max_size = max_value / signal.price
        size = min(size, max_size)

        # Round to whole shares (for stocks)
        size = int(size)

        if size <= 0:
            logger.info(f"Position size too small for {signal.symbol} at {signal.price}")
            return 0

        return size

    def has_open_position(self, symbol: str) -> bool:
        """Check if there's already an open position for this symbol."""
        return any(
            p.symbol == symbol and p.status == PositionStatus.OPEN
            for p in self.positions
        )

    def open_position(self, signal: Signal, size: int = 0) -> Optional[Position]:
        """
        Create and track a new position.

        Args:
            signal: Trading signal with entry price, stop-loss, take-profit
            size: Position size in shares (if 0, calculated internally)
        """
        can, reason = self.can_trade()
        if not can:
            logger.info(f"Trade blocked: {reason}")
            return None

        # Block duplicate symbol positions
        if self.has_open_position(signal.symbol):
            logger.info(f"Trade blocked: already have open position on {signal.symbol}")
            return None

        if size <= 0:
            size = self.calculate_position_size(signal)
        if size <= 0:
            return None

        position = Position(
            symbol=signal.symbol,
            side=signal.type,
            entry_price=signal.price,
            size=size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            entry_time=self._to_naive_datetime(signal.timestamp),
            signal=signal,
            current_price=signal.price,
            highest_price=signal.price,
            lowest_price=signal.price,
        )

        self.positions.append(position)
        self.daily_trades += 1

        cost = size * signal.price
        logger.info(
            f"OPEN {signal.type.value} {signal.symbol}: "
            f"{size} shares @ {signal.price:.2f} "
            f"(cost: {cost:.2f}, SL: {signal.stop_loss:.2f}, TP: {signal.take_profit:.2f})"
        )
        return position

    def update_position(self, position: Position, current_price: float, bars_elapsed: int = 1):
        """
        Update position with new price data.
        Checks stop-loss, take-profit, trailing stop, and time-based exit.
        """
        if position.status != PositionStatus.OPEN:
            return

        # GUARD 1: Reject invalid prices (delayed data can return 0)
        if current_price <= 0:
            logger.warning(f"[PRICE GUARD] Skipping update for {position.symbol}: price={current_price}")
            return

        # GUARD 2: Price sanity — reject quotes that deviate too far from entry
        if position.entry_price > 0:
            deviation = abs(current_price - position.entry_price) / position.entry_price
            if deviation > self.rc.max_price_deviation_pct:
                logger.error(
                    f"[PRICE SANITY] {position.symbol}: price={current_price:.2f} deviates "
                    f"{deviation:.1%} from entry={position.entry_price:.2f} — skipping update"
                )
                return

        position.current_price = current_price
        position.bars_held += bars_elapsed

        # Track highs/lows for trailing stop
        if position.side == SignalType.LONG:
            position.highest_price = max(position.highest_price, current_price)
            position.lowest_price = min(position.lowest_price, current_price)
            # MAE = worst drawdown from entry, MFE = best run-up from entry
            position.mae_price = position.lowest_price
            position.mfe_price = position.highest_price
            position.mae_pct = (position.entry_price - position.lowest_price) / position.entry_price
            position.mfe_pct = (position.highest_price - position.entry_price) / position.entry_price
        else:
            position.highest_price = max(position.highest_price, current_price)
            position.lowest_price = min(position.lowest_price, current_price)
            # For shorts: adverse = price going UP, favorable = price going DOWN
            position.mae_price = position.highest_price
            position.mfe_price = position.lowest_price
            position.mae_pct = (position.highest_price - position.entry_price) / position.entry_price
            position.mfe_pct = (position.entry_price - position.lowest_price) / position.entry_price

        # ─── Check exits in priority order ────────────────────

        # 1. Stop-loss (skip if IBKR bracket order handles SL server-side)
        if not position.has_bracket:
            if position.side == SignalType.LONG and current_price <= position.stop_loss:
                self._close_position(position, current_price, PositionStatus.CLOSED_SL)
                return
            if position.side == SignalType.SHORT and current_price >= position.stop_loss:
                self._close_position(position, current_price, PositionStatus.CLOSED_SL)
                return

        # 2. Take-profit (skip if IBKR bracket order handles TP server-side)
        if not position.has_bracket:
            if position.side == SignalType.LONG and current_price >= position.take_profit:
                self._close_position(position, current_price, PositionStatus.CLOSED_TP)
                return
            if position.side == SignalType.SHORT and current_price <= position.take_profit:
                self._close_position(position, current_price, PositionStatus.CLOSED_TP)
                return

        # 3. Trailing stop activation and check (skip for bracket-managed positions
        #    because the IBKR bracket SL/TP handles exits server-side)
        if not position.has_bracket:
            self._update_trailing_stop(position, current_price)
            if position.trailing_stop is not None:
                if position.side == SignalType.LONG and current_price <= position.trailing_stop:
                    self._close_position(position, current_price, PositionStatus.CLOSED_TRAIL)
                    return
                if position.side == SignalType.SHORT and current_price >= position.trailing_stop:
                    self._close_position(position, current_price, PositionStatus.CLOSED_TRAIL)
                    return

        # 4. Time-based exit is handled by main.py using wall-clock time
        #    (bars_held counter was unreliable — incremented per loop iteration, not per candle)

    def _update_trailing_stop(self, position: Position, current_price: float):
        """
        Activate and update trailing stop.

        Supports two modes:
        1. ATR-based trailing (V2 strategy) — uses absolute price distances
           - Activated when position has atr_trail_activation > 0
           - Activation: unrealized profit >= atr_trail_activation (in price units)
           - Distance: trail by atr_trail_distance behind the best price

        2. Percentage-based trailing (legacy/config) — uses % of price
           - Fallback when ATR params are not set
           - Activation: unrealized PnL% >= trailing_stop_activation
           - Distance: trail by trailing_stop_distance % of best price
        """
        # ─── Mode 1: ATR-based trailing (V2 strategy) ───
        if position.atr_trail_activation > 0 and position.atr_trail_distance > 0:
            if position.side == SignalType.LONG:
                profit = current_price - position.entry_price
                if profit >= position.atr_trail_activation:
                    new_trail = position.highest_price - position.atr_trail_distance
                    if position.trailing_stop is None or new_trail > position.trailing_stop:
                        if position.trailing_stop is None:
                            logger.info(
                                f"[TRAIL] {position.symbol}: ATR trailing activated at "
                                f"profit={profit:.2f} (threshold={position.atr_trail_activation:.2f}), "
                                f"trail={new_trail:.2f}"
                            )
                        position.trailing_stop = new_trail
            else:  # SHORT
                profit = position.entry_price - current_price
                if profit >= position.atr_trail_activation:
                    new_trail = position.lowest_price + position.atr_trail_distance
                    if position.trailing_stop is None or new_trail < position.trailing_stop:
                        if position.trailing_stop is None:
                            logger.info(
                                f"[TRAIL] {position.symbol}: ATR trailing activated at "
                                f"profit={profit:.2f} (threshold={position.atr_trail_activation:.2f}), "
                                f"trail={new_trail:.2f}"
                            )
                        position.trailing_stop = new_trail
            return

        # ─── Mode 2: Percentage-based trailing (legacy fallback) ───
        pnl_pct = position.unrealized_pnl_pct

        # Activate trailing stop after minimum profit
        if pnl_pct >= self.rc.trailing_stop_activation:
            if position.side == SignalType.LONG:
                new_trail = position.highest_price * (1 - self.rc.trailing_stop_distance)
                if position.trailing_stop is None or new_trail > position.trailing_stop:
                    position.trailing_stop = new_trail
            else:
                new_trail = position.lowest_price * (1 + self.rc.trailing_stop_distance)
                if position.trailing_stop is None or new_trail < position.trailing_stop:
                    position.trailing_stop = new_trail

    def _close_position(self, position: Position, exit_price: float, status: PositionStatus):
        """Close a position and update accounting (including commission)."""
        # GUARD 1: Never close with price=0 — produces ±100% phantom PnL
        if exit_price <= 0:
            logger.error(
                f"[PRICE GUARD] BLOCKED close of {position.symbol} with exit_price={exit_price:.2f} "
                f"(status={status.value}). Position stays OPEN until valid price available."
            )
            return

        # GUARD 2: Price sanity — never close with wildly wrong price
        if position.entry_price > 0:
            deviation = abs(exit_price - position.entry_price) / position.entry_price
            if deviation > self.rc.max_price_deviation_pct:
                logger.error(
                    f"[PRICE SANITY] BLOCKED close of {position.symbol}: exit={exit_price:.2f} "
                    f"deviates {deviation:.1%} from entry={position.entry_price:.2f}. "
                    f"Position stays OPEN — close manually via TWS."
                )
                return

        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.status = status

        # Finalize MAE/MFE with exit price (may be the actual worst/best)
        if position.side == SignalType.LONG:
            position.lowest_price = min(position.lowest_price, exit_price)
            position.highest_price = max(position.highest_price, exit_price)
            position.mae_price = position.lowest_price
            position.mfe_price = position.highest_price
            position.mae_pct = (position.entry_price - position.lowest_price) / position.entry_price
            position.mfe_pct = (position.highest_price - position.entry_price) / position.entry_price
        else:
            position.lowest_price = min(position.lowest_price, exit_price)
            position.highest_price = max(position.highest_price, exit_price)
            position.mae_price = position.highest_price
            position.mfe_price = position.lowest_price
            position.mae_pct = (position.highest_price - position.entry_price) / position.entry_price
            position.mfe_pct = (position.entry_price - position.lowest_price) / position.entry_price

        if position.side == SignalType.LONG:
            raw_pnl = (exit_price - position.entry_price) * position.size
            position.pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            raw_pnl = (position.entry_price - exit_price) * position.size
            position.pnl_pct = (position.entry_price - exit_price) / position.entry_price

        # Deduct round-trip commission (entry + exit)
        # IBKR tiered: 0.05% of trade value, EUR 1.25 minimum per leg
        entry_comm = max(1.25, position.entry_price * position.size * 0.0005)
        exit_comm = max(1.25, exit_price * position.size * 0.0005)
        position.commission = entry_comm + exit_comm
        position.pnl = raw_pnl - position.commission

        # Update daily tracking
        self.daily_pnl += position.pnl
        self.balance += position.pnl

        # Track consecutive losses
        if position.pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_time = datetime.now()
        else:
            self.consecutive_losses = 0

        # Move to closed
        self.closed_positions.append(position)
        if position in self.positions:
            self.positions.remove(position)

        emoji = "+" if position.pnl >= 0 else ""
        logger.info(
            f"CLOSE {position.side.value} {position.symbol} [{status.value}]: "
            f"{position.size} shares @ {exit_price:.2f} "
            f"(PnL: {emoji}{position.pnl:.2f} / {position.pnl_pct:+.2%})"
        )

    def close_all_positions(self, current_prices: dict[str, float], reason: str = "EOD"):
        """Close all open positions (end of day or emergency)."""
        open_positions = [p for p in self.positions if p.status == PositionStatus.OPEN]
        for pos in open_positions:
            price = current_prices.get(pos.symbol, pos.current_price)
            # GUARD: Skip positions where we have no valid price
            if price <= 0:
                logger.error(
                    f"[PRICE GUARD] Cannot close {pos.symbol} ({reason}): no valid price available "
                    f"(quote={current_prices.get(pos.symbol)}, cached={pos.current_price}). "
                    f"Position stays OPEN — close manually via TWS."
                )
                continue
            status = PositionStatus.CLOSED_EOD if reason == "EOD" else PositionStatus.CLOSED_MANUAL
            self._close_position(pos, price, status)

    def _reset_daily(self):
        """Reset daily counters at start of new trading day."""
        logger.info(f"Daily reset. Yesterday PnL: {self.daily_pnl:.2f}, Trades: {self.daily_trades}")
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.last_loss_time = None

    # ─── Reporting ────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Calculate performance statistics."""
        closed = self.closed_positions
        if not closed:
            return {"total_trades": 0}

        wins = [p for p in closed if p.pnl > 0]
        losses = [p for p in closed if p.pnl <= 0]

        total_pnl = sum(p.pnl for p in closed)
        gross_profit = sum(p.pnl for p in wins) if wins else 0
        gross_loss = abs(sum(p.pnl for p in losses)) if losses else 1

        avg_win = gross_profit / len(wins) if wins else 0
        avg_loss = gross_loss / len(losses) if losses else 0

        # Max drawdown
        equity_curve = []
        running = self.rc.initial_balance
        for p in closed:
            running += p.pnl
            equity_curve.append(running)

        peak = self.rc.initial_balance
        max_dd = 0
        for eq in equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        total_commission = sum(p.commission for p in closed)

        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(closed) if closed else 0,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            "max_drawdown": max_dd,
            "avg_pnl_per_trade": total_pnl / len(closed),
            "avg_bars_held": sum(p.bars_held for p in closed) / len(closed),
            "balance": self.balance,
            "return_pct": (self.balance - self.rc.initial_balance) / self.rc.initial_balance,
            "total_commission": total_commission,
        }

    def print_summary(self):
        """Print performance summary to log."""
        stats = self.get_stats()
        if stats["total_trades"] == 0:
            logger.info("No closed trades yet.")
            return

        logger.info("=" * 60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Trades:    {stats['total_trades']}")
        logger.info(f"Win Rate:        {stats['win_rate']:.1%}")
        logger.info(f"Total PnL:       {stats['total_pnl']:+.2f}")
        logger.info(f"Avg Win:         {stats['avg_win']:.2f}")
        logger.info(f"Avg Loss:        {stats['avg_loss']:.2f}")
        logger.info(f"Profit Factor:   {stats['profit_factor']:.2f}")
        logger.info(f"Max Drawdown:    {stats['max_drawdown']:.2%}")
        logger.info(f"Commission:      {stats['total_commission']:.2f}")
        logger.info(f"Balance:         {stats['balance']:.2f}")
        logger.info(f"Return:          {stats['return_pct']:+.2%}")
        logger.info("=" * 60)

    # ─── State Persistence ─────────────────────────────────

    def save_state(self, filepath: str = "data/risk_state.json"):
        """Persist positions and risk state to disk for crash recovery."""
        state = {
            "balance": self.balance,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "consecutive_losses": self.consecutive_losses,
            "last_loss_time": self.last_loss_time.isoformat() if self.last_loss_time else None,
            "current_date": self._current_date.isoformat(),
            "positions": [self._position_to_dict(p) for p in self.positions if p.status == PositionStatus.OPEN],
            "saved_at": datetime.now().isoformat(),
        }
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file then rename
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str))
        tmp.rename(path)
        logger.debug(f"State saved: {len(state['positions'])} positions, balance={self.balance:.2f}")

    def load_state(self, filepath: str = "data/risk_state.json") -> bool:
        """Load persisted state from disk. Returns True if state was loaded."""
        path = Path(filepath)
        if not path.exists():
            return False
        try:
            state = json.loads(path.read_text())
            self.balance = state["balance"]
            self.daily_pnl = state.get("daily_pnl", 0.0)
            self.daily_trades = state.get("daily_trades", 0)
            self.consecutive_losses = state.get("consecutive_losses", 0)
            if state.get("last_loss_time"):
                self.last_loss_time = datetime.fromisoformat(state["last_loss_time"])
            if state.get("current_date"):
                self._current_date = date.fromisoformat(state["current_date"])
            for p_dict in state.get("positions", []):
                pos = self._dict_to_position(p_dict)
                if pos:
                    self.positions.append(pos)
            logger.info(f"State restored: {len(self.positions)} open positions, balance={self.balance:.2f}")
            return True
        except Exception as e:
            logger.error(f"Failed to load state from {filepath}: {e}")
            return False

    def clear_state(self, filepath: str = "data/risk_state.json"):
        """Delete persisted state file (call after clean EOD close)."""
        path = Path(filepath)
        if path.exists():
            path.unlink()
            logger.debug("State file cleared.")

    def _position_to_dict(self, pos: Position) -> dict:
        """Serialize a Position to dict for JSON storage."""
        return {
            "symbol": pos.symbol,
            "side": pos.side.value,
            "entry_price": pos.entry_price,
            "size": pos.size,
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
            "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
            "current_price": pos.current_price,
            "trailing_stop": pos.trailing_stop,
            "highest_price": pos.highest_price,
            "lowest_price": pos.lowest_price,
            "bars_held": pos.bars_held,
            "has_bracket": pos.has_bracket,
            "bracket_order_id": pos.bracket_order_id,
            "mae_price": pos.mae_price,
            "mfe_price": pos.mfe_price,
            "mae_pct": pos.mae_pct,
            "mfe_pct": pos.mfe_pct,
            "atr_trail_activation": pos.atr_trail_activation,
            "atr_trail_distance": pos.atr_trail_distance,
            "signal_confidence": pos.signal.confidence if pos.signal else 0,
            "signal_regime": pos.signal.regime.value if pos.signal else "ranging",
            "signal_strategy": pos.signal.strategy if pos.signal else "",
            "signal_reason": pos.signal.reason if pos.signal else "",
        }

    def _dict_to_position(self, d: dict) -> Optional[Position]:
        """Deserialize a Position from dict."""
        try:
            side = SignalType(d["side"])
            signal = Signal(
                type=side,
                symbol=d["symbol"],
                price=d["entry_price"],
                timestamp=datetime.fromisoformat(d["entry_time"]) if d.get("entry_time") else datetime.now(),
                confidence=d.get("signal_confidence", 0),
                regime=MarketRegime(d.get("signal_regime", "ranging")),
                strategy=d.get("signal_strategy", ""),
                indicators={},
                stop_loss=d["stop_loss"],
                take_profit=d["take_profit"],
                reason=d.get("signal_reason", "restored from state"),
            )
            return Position(
                symbol=d["symbol"],
                side=side,
                entry_price=d["entry_price"],
                size=d["size"],
                stop_loss=d["stop_loss"],
                take_profit=d["take_profit"],
                entry_time=datetime.fromisoformat(d["entry_time"]) if d.get("entry_time") else datetime.now(),
                signal=signal,
                status=PositionStatus.OPEN,
                current_price=d.get("current_price", d["entry_price"]),
                trailing_stop=d.get("trailing_stop"),
                highest_price=d.get("highest_price", d["entry_price"]),
                lowest_price=d.get("lowest_price", d["entry_price"]),
                bars_held=d.get("bars_held", 0),
                has_bracket=d.get("has_bracket", False),
                bracket_order_id=d.get("bracket_order_id", ""),
                mae_price=d.get("mae_price", 0),
                mfe_price=d.get("mfe_price", 0),
                mae_pct=d.get("mae_pct"),
                mfe_pct=d.get("mfe_pct"),
                atr_trail_activation=d.get("atr_trail_activation", 0.0),
                atr_trail_distance=d.get("atr_trail_distance", 0.0),
            )
        except Exception as e:
            logger.error(f"Failed to restore position: {e}")
            return None
