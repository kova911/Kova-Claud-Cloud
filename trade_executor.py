"""
Trade Executor — Interactive Brokers Interface
Handles order placement via IBKR TWS API through ib_async library.

Connection options:
- IB Gateway (lightweight, recommended for algo trading)
- TWS (Trader Workstation, full desktop app)

Modes:
- PAPER: Connects to IBKR paper trading account (port 4002/7497)
- LIVE: Connects to IBKR live account (port 4001/7496)
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from strategy_v2 import Signal, SignalType

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    id: str
    symbol: str
    side: str              # "buy" or "sell"
    order_type: OrderType
    quantity: int
    price: float           # Limit price (0 for market)
    stop_price: float      # Stop price (0 if not stop order)
    status: OrderStatus
    created_at: datetime
    ib_order_id: int = 0   # IBKR's internal order ID
    ib_trade: object = None  # ib_async Trade object (entry leg)
    bracket_sl_trade: object = None  # ib_async Trade for bracket SL leg
    bracket_tp_trade: object = None  # ib_async Trade for bracket TP leg
    filled_at: Optional[datetime] = None
    filled_price: float = 0.0
    filled_quantity: int = 0
    commission: float = 0.0
    slippage: float = 0.0        # Signal price - fill price (negative = unfavorable)


class TradeExecutor:
    """
    Handles trade execution on Interactive Brokers.

    Uses ib_async library for async communication with TWS/IB Gateway.
    Supports bracket orders (entry + stop-loss + take-profit in one atomic order).

    Connection:
        Paper: localhost:4002 (IB Gateway) or localhost:7497 (TWS)
        Live:  localhost:4001 (IB Gateway) or localhost:7496 (TWS)
    """

    def __init__(self, config):
        self.config = config
        self.ibc = config.ibkr
        self.orders: list[Order] = []
        self._order_counter = 0
        self._connected = False
        self._ib = None  # ib_async.IB instance
        self._contracts_cache: dict[str, object] = {}
        self._tick_cache: dict[str, float] = {}  # symbol → minTick
        self._last_heartbeat: float = 0
        self._reconnect_attempts: int = 0
        self._max_reconnect_attempts: int = 5
        self._reconnect_delay: int = 10  # seconds between retries

    @property
    def is_paper(self) -> bool:
        return self.ibc.paper_trading

    @property
    def is_ibkr_connected(self) -> bool:
        """True if connected to IBKR (not just local paper fallback)."""
        return self._ib is not None and self._connected

    @property
    def ib_connection(self):
        """The ib_async IB instance (or None if not connected)."""
        return self._ib

    def connect(self) -> bool:
        """
        Connect to Interactive Brokers TWS or IB Gateway.
        Paper trading uses a different port than live.
        """
        port = self.ibc.paper_port if self.is_paper else self.ibc.live_port
        mode = "PAPER" if self.is_paper else "LIVE"

        try:
            from ib_async import IB
            self._ib = IB()

            logger.info(f"Connecting to IBKR ({mode}) at {self.ibc.host}:{port}...")
            self._ib.connect(
                host=self.ibc.host,
                port=port,
                clientId=self.ibc.client_id,
                readonly=False,
            )
            self._connected = True

            # Log account info
            accounts = self._ib.managedAccounts()
            logger.info(f"Connected to IBKR ({mode}). Accounts: {accounts}")

            # Get account summary
            summary = self._ib.accountSummary()
            for item in summary:
                if item.tag == "NetLiquidation" and item.currency == "EUR":
                    logger.info(f"Account value: {item.value} {item.currency}")
                    break

            return True

        except ImportError:
            logger.error(
                "ib_async not installed. Install with: pip install ib_async\n"
                "Also ensure IB Gateway or TWS is running."
            )
            return self._fallback_paper()

        except ConnectionRefusedError:
            logger.error(
                f"Cannot connect to IBKR at {self.ibc.host}:{port}.\n"
                f"Ensure IB Gateway or TWS is running and API connections are enabled:\n"
                f"  IB Gateway: Configure → Settings → API → Enable ActiveX and Socket Clients\n"
                f"  TWS: Edit → Global Configuration → API → Settings → Enable"
            )
            return self._fallback_paper()

        except Exception as e:
            logger.error(f"IBKR connection failed: {e}")
            return self._fallback_paper()

    def _fallback_paper(self) -> bool:
        """Fall back to local paper simulation if IBKR not available."""
        logger.info("Falling back to LOCAL PAPER TRADING mode (no IBKR connection)")
        self._connected = True
        self._ib = None
        return True

    def _get_contract(self, symbol: str):
        """
        Create an IBKR contract for the given symbol.
        Handles both US and EU (XETRA) stocks.
        Returns None if contract cannot be qualified.
        """
        if symbol in self._contracts_cache:
            return self._contracts_cache[symbol]

        from ib_async import Stock

        # Determine exchange based on symbol suffix
        if symbol.endswith(".DE"):
            # German stock — use SMART routing (avoids IBKR direct-route precaution error 10311)
            clean = symbol.replace(".DE", "")
            contract = Stock(clean, "SMART", "EUR")
        else:
            # Default to US SMART routing
            contract = Stock(symbol, "SMART", "USD")

        # Qualify the contract (resolve ambiguity)
        if self._ib:
            try:
                qualified = self._ib.qualifyContracts(contract)
                if not qualified or not qualified[0].conId:
                    logger.warning(f"Contract not found for {symbol} — skipping")
                    return None
                contract = qualified[0]
            except Exception as e:
                logger.warning(f"Contract qualification failed for {symbol}: {e}")
                return None

        self._contracts_cache[symbol] = contract
        return contract

    def _get_min_tick(self, symbol: str, contract=None, price: float = 0) -> float:
        """
        Get the minimum ORDER tick size for a contract at a given price level.

        Uses IBKR's market rules API to determine the exact tick increment.
        Different instruments (stocks vs ETFs) and exchanges have different
        tick schedules (e.g., XETRA ETFs need 0.05 for prices ≥ €100).

        Falls back to 0.01 if market rules can't be queried.
        """
        if symbol in self._tick_cache:
            return self._tick_cache[symbol]

        min_tick = 0.01  # Safe default for stocks

        if contract and self._ib and price > 0:
            try:
                details_list = self._ib.reqContractDetails(contract)
                if details_list:
                    details = details_list[0]
                    rule_ids_str = getattr(details, 'marketRuleIds', '')
                    if rule_ids_str:
                        # Parse first market rule ID
                        rule_id = int(rule_ids_str.split(',')[0])
                        rules = self._ib.reqMarketRule(rule_id)
                        if rules:
                            # Find tick increment for our price level
                            # Rules are sorted by lowEdge ascending —
                            # iterate all and keep the last one where price >= lowEdge
                            for rule in rules:
                                if price >= rule.lowEdge:
                                    min_tick = rule.increment
                            # Floor at 0.01 for safety (avoid sub-cent ticks)
                            if min_tick < 0.01:
                                min_tick = 0.01
                            logger.info(
                                f"[TICK] {symbol} @ {price:.2f}: "
                                f"tick={min_tick} (marketRule {rule_id})"
                            )
            except Exception as e:
                logger.debug(f"Market rule lookup failed for {symbol}: {e}")

        self._tick_cache[symbol] = min_tick
        return min_tick

    def _round_to_tick(self, price: float, min_tick: float) -> float:
        """Round a price to the nearest valid tick increment (always ≥ 0.01)."""
        if min_tick < 0.01:
            min_tick = 0.01
        # Determine decimal places from tick size (0.01→2, 0.05→2, 0.001→3)
        import math
        decimals = max(2, -int(math.floor(math.log10(min_tick))))
        return round(round(price / min_tick) * min_tick, decimals)

    def place_order(
        self,
        signal: Signal,
        quantity: int,
        order_type: OrderType = OrderType.LIMIT,
        require_confirmation: bool = True,
    ) -> Optional[Order]:
        """
        Place a trade order on IBKR.

        For scalping, uses bracket orders: entry + stop-loss + take-profit
        submitted atomically so protection is instant.
        """
        if not self._connected:
            logger.error("Not connected. Call connect() first.")
            return None

        self._order_counter += 1
        order = Order(
            id=f"ORD-{self._order_counter:06d}",
            symbol=signal.symbol,
            side="buy" if signal.type == SignalType.LONG else "sell",
            order_type=order_type,
            quantity=quantity,
            price=signal.price,
            stop_price=signal.stop_loss,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
        )

        if require_confirmation:
            logger.info(
                f"\n{'=' * 60}\n"
                f"ORDER REQUIRES CONFIRMATION\n"
                f"{'=' * 60}\n"
                f"  Action:   {order.side.upper()} {order.quantity}x {order.symbol}\n"
                f"  Type:     {order.order_type.value}\n"
                f"  Price:    {order.price:.4f}\n"
                f"  Stop:     {order.stop_price:.4f}\n"
                f"  Target:   {signal.take_profit:.4f}\n"
                f"  Value:    {order.quantity * order.price:.2f}\n"
                f"  Reason:   {signal.reason}\n"
                f"  Conf:     {signal.confidence:.0%}\n"
                f"{'=' * 60}"
            )

        if self._ib:
            return self._ibkr_execute(order, signal)
        else:
            return self._paper_fill(order)

    def place_bracket_order(self, signal: Signal, quantity: int) -> Optional[Order]:
        """
        Place a bracket order: entry + stop-loss + take-profit in one atomic submission.
        This is the preferred method for scalping — protection is active immediately.

        Entry uses an AGGRESSIVE limit price offset by 0.1% from signal price
        to account for delayed market data (15-min delay on paper accounts).
        For BUY: limit = price * 1.001 (willing to pay slightly more)
        For SELL: limit = price * 0.999 (willing to receive slightly less)
        """
        if not self._ib:
            return self.place_order(signal, quantity, require_confirmation=True)

        from ib_async import LimitOrder, StopOrder, Order as IBOrder

        contract = self._get_contract(signal.symbol)
        if contract is None:
            logger.warning(f"Cannot place bracket order — no valid contract for {signal.symbol}")
            return None

        action = "BUY" if signal.type == SignalType.LONG else "SELL"
        reverse_action = "SELL" if signal.type == SignalType.LONG else "BUY"

        # Get tick size for this contract at entry price level and round all prices
        min_tick = self._get_min_tick(signal.symbol, contract, price=signal.price)

        # Aggressive limit: offset entry by 0.1% to account for delayed data
        # This dramatically improves fill rate on paper accounts with 15-min delayed data
        entry_offset = 0.001  # 0.1%
        if action == "BUY":
            aggressive_price = signal.price * (1 + entry_offset)
        else:
            aggressive_price = signal.price * (1 - entry_offset)

        entry_price = self._round_to_tick(aggressive_price, min_tick)
        tp_price = self._round_to_tick(signal.take_profit, min_tick)
        sl_price = self._round_to_tick(signal.stop_loss, min_tick)

        # Create bracket order
        bracket = self._ib.bracketOrder(
            action=action,
            quantity=quantity,
            limitPrice=entry_price,
            takeProfitPrice=tp_price,
            stopLossPrice=sl_price,
        )

        # Submit all three orders atomically
        # bracket[0]=entry, bracket[1]=take-profit, bracket[2]=stop-loss
        trades = []
        for ib_order in bracket:
            trade = self._ib.placeOrder(contract, ib_order)
            trades.append(trade)

        self._order_counter += 1
        order = Order(
            id=f"BRK-{self._order_counter:06d}",
            symbol=signal.symbol,
            side=action.lower(),
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=signal.price,
            stop_price=signal.stop_loss,
            status=OrderStatus.SUBMITTED,
            created_at=datetime.now(),
            ib_order_id=bracket[0].orderId if bracket else 0,
            ib_trade=trades[0] if trades else None,
            bracket_tp_trade=trades[1] if len(trades) >= 2 else None,
            bracket_sl_trade=trades[2] if len(trades) >= 3 else None,
        )

        self.orders.append(order)
        # Store tick size for skill updater tracking
        order._tick_size = min_tick
        logger.info(
            f"[IBKR] Bracket order submitted: {order.id} — "
            f"{action} {quantity}x {signal.symbol} @ {entry_price:.2f} "
            f"(tick={min_tick}), SL: {sl_price:.2f}, TP: {tp_price:.2f}"
        )

        # Wait for fill (with timeout)
        self._ib.sleep(2)
        self._update_order_status(order)

        return order

    def _ibkr_execute(self, order: Order, signal: Signal) -> Optional[Order]:
        """Execute a single order on IBKR."""
        from ib_async import LimitOrder, MarketOrder, StopOrder

        contract = self._get_contract(order.symbol)
        if contract is None:
            logger.warning(f"Cannot execute order — no valid contract for {order.symbol}")
            return None

        action = order.side.upper()
        min_tick = self._get_min_tick(order.symbol, contract, price=order.price)

        try:
            # Use MarketOrder if price is 0 or missing (e.g., position close orders)
            if order.order_type == OrderType.MARKET or order.price <= 0:
                ib_order = MarketOrder(action, order.quantity)
            elif order.order_type == OrderType.LIMIT:
                ib_order = LimitOrder(action, order.quantity, self._round_to_tick(order.price, min_tick))
            elif order.order_type == OrderType.STOP:
                ib_order = StopOrder(action, order.quantity, self._round_to_tick(order.stop_price, min_tick))
            else:
                ib_order = LimitOrder(action, order.quantity, self._round_to_tick(order.price, min_tick))

            # Set time-in-force to Day (good for scalping)
            ib_order.tif = "DAY"

            trade = self._ib.placeOrder(contract, ib_order)
            order.ib_order_id = ib_order.orderId
            order.ib_trade = trade
            order.status = OrderStatus.SUBMITTED

            self.orders.append(order)
            logger.info(f"[IBKR] Order submitted: {order.id} (IB #{ib_order.orderId})")

            # Wait briefly for fill
            self._ib.sleep(2)
            self._update_order_status(order)

            return order

        except Exception as e:
            logger.error(f"IBKR order execution failed: {e}")
            order.status = OrderStatus.REJECTED
            self.orders.append(order)
            return order

    def _update_order_status(self, order: Order):
        """Update order status from IBKR trade object."""
        if not order.ib_trade:
            return

        trade = order.ib_trade
        if trade.isDone():
            ibkr_status = trade.orderStatus.status
            if ibkr_status == "Filled":
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.now()
                order.filled_price = trade.orderStatus.avgFillPrice
                order.filled_quantity = int(trade.orderStatus.filled)
                order.commission = sum(
                    fill.commissionReport.commission
                    for fill in trade.fills
                    if fill.commissionReport
                ) if trade.fills else 0
                # Slippage: difference between requested and fill price
                # For buys: negative slippage = paid more than expected
                # For sells: negative slippage = received less than expected
                if order.side == "buy":
                    order.slippage = order.price - order.filled_price
                else:
                    order.slippage = order.filled_price - order.price
                logger.info(
                    f"[IBKR] FILLED: {order.id} — "
                    f"{order.filled_quantity}x @ {order.filled_price:.2f} "
                    f"(commission: {order.commission:.2f}, slippage: {order.slippage:+.4f})"
                )
            elif ibkr_status in ("Cancelled", "ApiCancelled"):
                order.status = OrderStatus.CANCELLED
                logger.info(f"[IBKR] CANCELLED: {order.id} ({ibkr_status})")
            elif ibkr_status == "Inactive":
                # Inactive = IBKR rejected the order (insufficient margin, contract issue, etc.)
                # NOT the same as a clean cancel — log as warning
                order.status = OrderStatus.REJECTED
                logger.warning(
                    f"[IBKR] INACTIVE/REJECTED: {order.id} ({order.symbol}) — "
                    f"IBKR marked order as Inactive (may indicate insufficient margin or invalid contract)"
                )
            else:
                order.status = OrderStatus.REJECTED
                logger.warning(f"[IBKR] Order {order.id} terminal status: {ibkr_status}")
        elif trade.orderStatus.status == "PreSubmitted":
            order.status = OrderStatus.SUBMITTED

    def _paper_fill(self, order: Order) -> Order:
        """Simulate order fill with realistic slippage."""
        import random
        # Simulate slippage: 0 to 0.02% in unfavorable direction
        slippage_pct = random.uniform(0, 0.0002)
        if order.side == "buy":
            fill_price = order.price * (1 + slippage_pct)  # Pay slightly more
        else:
            fill_price = order.price * (1 - slippage_pct)  # Receive slightly less

        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        order.filled_price = round(fill_price, 4)
        order.filled_quantity = order.quantity
        order.slippage = order.price - fill_price if order.side == "buy" else fill_price - order.price
        order.commission = max(1.25, order.quantity * fill_price * 0.0005)  # IBKR tiered estimate

        self.orders.append(order)
        logger.info(
            f"[PAPER] {order.side.upper()} {order.quantity}x {order.symbol} "
            f"filled @ {order.filled_price:.2f} (slip: {order.slippage:+.4f}, comm: {order.commission:.2f})"
        )
        return order

    def place_stop_loss(self, symbol: str, quantity: int, stop_price: float) -> Optional[Order]:
        """Place a standalone stop-loss order."""
        self._order_counter += 1
        order = Order(
            id=f"SL-{self._order_counter:06d}",
            symbol=symbol,
            side="sell",
            order_type=OrderType.STOP,
            quantity=quantity,
            price=0,
            stop_price=stop_price,
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
        )

        if self._ib:
            from ib_async import StopOrder
            contract = self._get_contract(symbol)
            min_tick = self._get_min_tick(symbol, contract, price=stop_price)
            ib_order = StopOrder("SELL", quantity, self._round_to_tick(stop_price, min_tick))
            ib_order.tif = "DAY"
            trade = self._ib.placeOrder(contract, ib_order)
            order.ib_trade = trade
            order.status = OrderStatus.SUBMITTED
            self.orders.append(order)
            logger.info(f"[IBKR] Stop-loss set: {symbol} @ {stop_price:.2f}")
        else:
            order.status = OrderStatus.SUBMITTED
            self.orders.append(order)
            logger.info(f"[PAPER] Stop-loss set: {symbol} @ {stop_price:.2f}")

        return order

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending/submitted order.
        Waits for IBKR confirmation to detect race condition (filled before cancel).
        Returns True if cancelled, False if already filled or not found.
        """
        for order in self.orders:
            if order.id == order_id and order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                if self._ib and order.ib_trade:
                    try:
                        self._ib.cancelOrder(order.ib_trade.order)
                        # Wait for IBKR to process the cancel (critical: detect race condition)
                        self._ib.sleep(2)
                        self._update_order_status(order)
                    except Exception as e:
                        logger.warning(f"Cancel order {order_id} error: {e}")
                        # Re-check status after exception — order may have been filled by IBKR
                        # during the failed cancel attempt
                        try:
                            self._update_order_status(order)
                        except Exception:
                            pass

                    # Check if IBKR filled the order before processing our cancel
                    # CRITICAL: Do NOT overwrite FILLED status with CANCELLED
                    if order.status == OrderStatus.FILLED:
                        logger.warning(
                            f"[RACE] Order {order_id} ({order.symbol}) was FILLED by IBKR "
                            f"before cancel processed! qty={order.filled_quantity} @ {order.filled_price:.2f}"
                        )
                        return False  # Caller must handle this as an active position

                order.status = OrderStatus.CANCELLED
                # Also cancel bracket legs if this was a bracket order
                if order.bracket_sl_trade or order.bracket_tp_trade:
                    self.cancel_bracket_legs(order_id)
                logger.info(f"Order cancelled: {order_id}")
                return True
        return False

    def get_portfolio(self) -> dict:
        """Get current portfolio state from IBKR. Only returns non-zero positions."""
        if self._ib:
            try:
                positions = self._ib.positions()
                portfolio = {}
                for pos in positions:
                    if pos.position == 0:
                        continue  # Skip flat positions
                    sym = pos.contract.symbol
                    portfolio[sym] = {
                        "quantity": pos.position,
                        "avg_price": pos.avgCost / abs(pos.position),
                        "market_value": abs(pos.position) * (pos.avgCost / abs(pos.position)),
                        "contract": str(pos.contract),
                    }
                return portfolio
            except Exception as e:
                logger.error(f"Portfolio fetch failed: {e}")
                return {}
        else:
            # Paper simulation from order history
            positions = {}
            for order in self.orders:
                if order.status != OrderStatus.FILLED:
                    continue
                sym = order.symbol
                if sym not in positions:
                    positions[sym] = {"quantity": 0, "avg_price": 0, "total_cost": 0}
                if order.side == "buy":
                    total = positions[sym]["total_cost"] + (order.filled_price * order.filled_quantity)
                    qty = positions[sym]["quantity"] + order.filled_quantity
                    positions[sym]["quantity"] = qty
                    positions[sym]["total_cost"] = total
                    positions[sym]["avg_price"] = total / qty if qty > 0 else 0
                else:
                    positions[sym]["quantity"] -= order.filled_quantity
            return {k: v for k, v in positions.items() if v["quantity"] != 0}

    def get_account_summary(self) -> dict:
        """Get IBKR account summary."""
        if not self._ib:
            return {"mode": "paper_local", "balance": "N/A"}

        try:
            summary = {}
            for item in self._ib.accountSummary():
                if item.currency in ("EUR", "USD", "BASE"):
                    summary[f"{item.tag}_{item.currency}"] = item.value
            return summary
        except Exception as e:
            logger.error(f"Account summary failed: {e}")
            return {}

    def get_order_history(self) -> list[dict]:
        """Get order history."""
        return [
            {
                "id": o.id,
                "symbol": o.symbol,
                "side": o.side,
                "type": o.order_type.value,
                "quantity": o.quantity,
                "price": o.price,
                "filled_price": o.filled_price,
                "status": o.status.value,
                "commission": o.commission,
                "ib_order_id": o.ib_order_id,
                "created": str(o.created_at),
                "filled": str(o.filled_at) if o.filled_at else None,
            }
            for o in self.orders
        ]

    def check_connection(self) -> bool:
        """
        Heartbeat check — verify IBKR connection is alive.
        Call this periodically (e.g. every 30s from main loop).
        Returns True if connected, False if disconnected (will attempt reconnect).
        """
        if not self._ib:
            return self._connected  # Local paper mode is always "connected"

        try:
            # isConnected() is the ib_async built-in heartbeat check
            if self._ib.isConnected():
                self._last_heartbeat = time.time()
                self._reconnect_attempts = 0
                return True
            else:
                logger.warning("IBKR connection lost. Attempting reconnect...")
                return self._reconnect()
        except Exception as e:
            logger.warning(f"IBKR heartbeat failed: {e}")
            return self._reconnect()

    def _reconnect(self) -> bool:
        """Attempt to reconnect to IBKR with exponential backoff."""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.critical(
                f"{'!' * 60}\n"
                f"  CRITICAL: IBKR CONNECTION LOST — ALL RECONNECT ATTEMPTS FAILED\n"
                f"  After {self._max_reconnect_attempts} attempts, falling back to paper mode.\n"
                f"  ⚠ ANY OPEN POSITIONS IN IBKR ARE NOW UNMONITORED!\n"
                f"  ⚠ No stop-loss, no take-profit, no trailing stop protection.\n"
                f"  ⚠ CHECK IBKR TWS IMMEDIATELY AND MANAGE POSITIONS MANUALLY.\n"
                f"{'!' * 60}"
            )
            self._ib = None
            self._connected = True  # Continue in paper mode
            return False

        self._reconnect_attempts += 1
        delay = self._reconnect_delay * self._reconnect_attempts
        logger.info(f"IBKR reconnect attempt {self._reconnect_attempts}/{self._max_reconnect_attempts} in {delay}s...")
        time.sleep(delay)

        port = self.ibc.paper_port if self.is_paper else self.ibc.live_port

        try:
            # Disconnect cleanly first
            try:
                self._ib.disconnect()
            except Exception:
                pass

            self._ib.connect(
                host=self.ibc.host,
                port=port,
                clientId=self.ibc.client_id,
                readonly=False,
            )
            self._connected = True
            self._last_heartbeat = time.time()
            self._contracts_cache.clear()  # Clear stale contracts
            logger.info(f"IBKR reconnected successfully (attempt {self._reconnect_attempts})")
            self._reconnect_attempts = 0
            return True

        except Exception as e:
            logger.warning(f"IBKR reconnect attempt {self._reconnect_attempts} failed: {e}")
            return False

    def cancel_bracket_legs(self, order_id: str) -> bool:
        """Cancel the SL and TP legs of a bracket order (for trailing/time exit)."""
        for order in self.orders:
            if order.id == order_id:
                cancelled = False
                if order.bracket_sl_trade and self._ib:
                    try:
                        self._ib.cancelOrder(order.bracket_sl_trade.order)
                        logger.info(f"[IBKR] Bracket SL cancelled for {order_id}")
                        cancelled = True
                    except Exception as e:
                        logger.warning(f"Failed to cancel bracket SL for {order_id}: {e}")
                if order.bracket_tp_trade and self._ib:
                    try:
                        self._ib.cancelOrder(order.bracket_tp_trade.order)
                        logger.info(f"[IBKR] Bracket TP cancelled for {order_id}")
                        cancelled = True
                    except Exception as e:
                        logger.warning(f"Failed to cancel bracket TP for {order_id}: {e}")
                return cancelled
        logger.warning(f"cancel_bracket_legs: order {order_id} not found")
        return False

    def check_bracket_fills(self) -> list[dict]:
        """
        Poll IBKR bracket SL/TP legs for fills.
        Returns list of {order_id, symbol, fill_type, fill_price} for legs filled by IBKR.
        """
        if not self._ib:
            return []

        fills = []
        for order in self.orders:
            if order.status != OrderStatus.FILLED:
                continue  # Only check brackets whose entry was filled

            # Check SL leg
            if order.bracket_sl_trade:
                trade = order.bracket_sl_trade
                try:
                    if trade.isDone() and trade.orderStatus.status == "Filled":
                        fills.append({
                            "order_id": order.id,
                            "symbol": order.symbol,
                            "fill_type": "stop_loss",
                            "fill_price": trade.orderStatus.avgFillPrice,
                        })
                        order.bracket_sl_trade = None  # Mark as processed
                        # TP leg auto-cancelled by IBKR (OCA group)
                        order.bracket_tp_trade = None
                except Exception as e:
                    logger.warning(f"[BRACKET] Error checking SL leg for {order.id} ({order.symbol}): {e}")

            # Check TP leg
            if order.bracket_tp_trade:
                trade = order.bracket_tp_trade
                try:
                    if trade.isDone() and trade.orderStatus.status == "Filled":
                        fills.append({
                            "order_id": order.id,
                            "symbol": order.symbol,
                            "fill_type": "take_profit",
                            "fill_price": trade.orderStatus.avgFillPrice,
                        })
                        order.bracket_tp_trade = None  # Mark as processed
                        # SL leg auto-cancelled by IBKR (OCA group)
                        order.bracket_sl_trade = None
                except Exception as e:
                    logger.warning(f"[BRACKET] Error checking TP leg for {order.id} ({order.symbol}): {e}")

        return fills

    def check_pending_orders(self) -> list[Order]:
        """
        Check for orders that were submitted but never filled.
        Returns list of stale orders (submitted > 60s ago and still not filled).
        Also prunes old completed orders from memory to prevent unbounded growth.
        """
        stale = []
        cutoff = time.time() - 60  # 60 seconds to be considered stale

        for order in self.orders:
            if order.status == OrderStatus.SUBMITTED:
                if order.created_at.timestamp() < cutoff:
                    # Update status from IBKR before declaring stale
                    self._update_order_status(order)
                    if order.status == OrderStatus.SUBMITTED:
                        stale.append(order)

        # Prune completed orders older than 1 hour to prevent memory leak
        prune_cutoff = time.time() - 3600
        terminal_states = (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)
        self.orders = [
            o for o in self.orders
            if o.status not in terminal_states or o.created_at.timestamp() > prune_cutoff
        ]

        return stale

    def disconnect(self):
        """Disconnect from IBKR."""
        if self._ib:
            try:
                self._ib.disconnect()
            except Exception:
                pass
        self._connected = False
        logger.info("Trade Executor disconnected from IBKR")
