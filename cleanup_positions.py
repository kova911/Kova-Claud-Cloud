#!/usr/bin/env python3
"""
Cleanup Script — Flatten All IBKR Paper Account Positions

Connects to IBKR paper account and places MARKET orders to close
all open positions (orphans from previous test runs).
Also cancels any open orders.

Usage:
    python cleanup_positions.py          # Dry run (show positions only)
    python cleanup_positions.py --exec   # Execute cleanup
"""

import sys
import time
import argparse
from ib_async import IB, Stock, MarketOrder

from config import config


def main():
    parser = argparse.ArgumentParser(description="Flatten all IBKR paper positions")
    parser.add_argument("--exec", action="store_true", help="Execute cleanup (default: dry run)")
    args = parser.parse_args()

    ibc = config.ibkr
    port = ibc.paper_port

    print(f"Connecting to IBKR PAPER at {ibc.host}:{port}...")
    ib = IB()

    try:
        ib.connect(host=ibc.host, port=port, clientId=ibc.client_id + 10, readonly=False)
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Ensure IB Gateway or TWS is running with API enabled.")
        sys.exit(1)

    accounts = ib.managedAccounts()
    print(f"Connected. Accounts: {accounts}")

    # ─── Cancel All Open Orders ─────────────────────────────────
    open_orders = ib.openOrders()
    print(f"\n{'='*60}")
    print(f"OPEN ORDERS: {len(open_orders)}")
    print(f"{'='*60}")

    for order in open_orders:
        print(f"  Order {order.orderId}: {order.action} {order.totalQuantity} "
              f"@ {order.lmtPrice or order.auxPrice} [{order.orderType}]")

    if open_orders and args.exec:
        print(f"\nCancelling {len(open_orders)} open orders...")
        ib.reqGlobalCancel()
        ib.sleep(2)
        remaining = ib.openOrders()
        print(f"  Orders remaining after cancel: {len(remaining)}")

    # ─── List & Flatten Positions ───────────────────────────────
    positions = ib.positions()
    print(f"\n{'='*60}")
    print(f"POSITIONS: {len(positions)}")
    print(f"{'='*60}")

    non_zero = []
    for pos in positions:
        qty = pos.position
        sym = pos.contract.symbol
        exchange = pos.contract.exchange or "SMART"
        currency = pos.contract.currency
        avg_cost = pos.avgCost / abs(qty) if qty != 0 else 0
        unrealized = getattr(pos, 'unrealizedPNL', 'N/A')

        status = "FLAT" if qty == 0 else f"{'LONG' if qty > 0 else 'SHORT'} {abs(qty)}"
        print(f"  {sym:10s}  {status:15s}  avg={avg_cost:>10.2f} {currency}  exchange={exchange}")

        if qty != 0:
            non_zero.append(pos)

    if not non_zero:
        print("\nNo positions to close. Account is flat.")
        ib.disconnect()
        return

    print(f"\n{len(non_zero)} position(s) to flatten.")

    if not args.exec:
        print("\n⚠️  DRY RUN — no orders placed.")
        print("    Run with --exec to flatten all positions.")
        ib.disconnect()
        return

    # ─── Execute Flatten Orders (with retry for bracket legs) ──
    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        if attempt > 1:
            # Re-cancel any orders that appeared (bracket legs firing)
            print(f"\n--- Retry {attempt}/{MAX_RETRIES}: re-cancelling orders & re-flattening ---")
            ib.reqGlobalCancel()
            ib.sleep(3)
            positions = ib.positions()
            non_zero = [p for p in positions if p.position != 0]
            if not non_zero:
                break

        print(f"\nFlattening {len(non_zero)} positions with MARKET orders...")

        for pos in non_zero:
            qty = pos.position
            sym = pos.contract.symbol
            currency = pos.contract.currency

            # Build a proper qualified contract
            contract = Stock(sym, "SMART", currency)
            try:
                qualified = ib.qualifyContracts(contract)
                if qualified:
                    contract = qualified[0]
            except Exception:
                pass  # Use unqualified contract as fallback

            # Opposite side to flatten
            action = "SELL" if qty > 0 else "BUY"
            abs_qty = abs(qty)

            order = MarketOrder(action, abs_qty)
            order.tif = "DAY"

            print(f"  {action} {abs_qty} {sym} @ MARKET ...", end=" ", flush=True)

            try:
                trade = ib.placeOrder(contract, order)
                ib.sleep(3)  # Wait for fill

                fill_status = trade.orderStatus.status
                fill_price = trade.orderStatus.avgFillPrice or 0
                print(f"→ {fill_status} @ {fill_price:.2f}")
            except Exception as e:
                print(f"→ ERROR: {e}")

        # Cancel any bracket legs that may have appeared during flattening
        ib.reqGlobalCancel()
        ib.sleep(3)

        # Verify
        remaining = ib.positions()
        remaining_nonzero = [p for p in remaining if p.position != 0]

        if not remaining_nonzero:
            break

        print(f"  ⚠️  {len(remaining_nonzero)} positions still open after attempt {attempt}")
        for pos in remaining_nonzero:
            print(f"    {pos.contract.symbol}: {pos.position}")

    # ─── Final Verification ──────────────────────────────────────
    ib.sleep(2)
    remaining = ib.positions()
    remaining_nonzero = [p for p in remaining if p.position != 0]

    print(f"\n{'='*60}")
    print(f"RESULT: {len(remaining_nonzero)} non-zero positions remaining")
    print(f"{'='*60}")

    for pos in remaining_nonzero:
        print(f"  {pos.contract.symbol}: {pos.position}")

    if not remaining_nonzero:
        print("✅ Account is flat. Ready for clean testing.")
    else:
        print("⚠️  Some positions could not be closed after 3 attempts. Check TWS.")

    ib.disconnect()
    print("\nDisconnected.")


if __name__ == "__main__":
    main()
