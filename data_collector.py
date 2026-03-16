"""
Market Data Collector
Fetches real-time and historical OHLCV data from multiple sources.
Supports: IBKR (via ib_async), Finnhub (WebSocket), Twelve Data, Yahoo Finance.
"""

import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, Callable
from pathlib import Path

import pandas as pd
import numpy as np
import requests

logger = logging.getLogger(__name__)


class DataCollector:
    """Multi-source market data collector with IBKR as primary live source."""

    def __init__(self, config):
        self.config = config
        self.cache: dict[str, pd.DataFrame] = {}
        self.cache_timestamps: dict[str, float] = {}
        self.live_prices: dict[str, dict] = {}
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_running = False
        self._price_callbacks: list[Callable] = []
        self._ib = None  # Shared IBKR connection (set by TradeExecutor)
        self._last_finnhub_call: float = 0  # Rate limiting for Finnhub REST

    def set_ibkr_connection(self, ib):
        """Share the IBKR connection from TradeExecutor for market data."""
        self._ib = ib
        if ib:
            # Request delayed market data (type 3) as fallback.
            # Returns real-time data for subscribed instruments and
            # delayed (15-20 min) data for unsubscribed ones.
            # This eliminates Error 354 on paper accounts without
            # EU/US market data subscriptions.
            ib.reqMarketDataType(3)
            logger.info("DataCollector using IBKR for market data (delayed fallback enabled)")

    # ─── Historical Data ──────────────────────────────────────

    def get_historical(
        self,
        symbol: str,
        interval: str = "5m",
        days: int = 60,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Args:
            symbol: Ticker symbol (e.g., "SAP.DE", "AAPL")
            interval: Candle interval ("1m", "5m", "15m", "1h", "1d")
            days: Number of days of history
            source: Data source override ("ibkr", "yfinance", "twelve_data", "finnhub")

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        cache_key = f"{symbol}_{interval}_{days}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]

        src = source or self.config.data.backtest_source
        df = pd.DataFrame()

        try:
            if src == "ibkr":
                df = self._fetch_ibkr(symbol, interval, days)
            elif src == "yfinance":
                df = self._fetch_yfinance(symbol, interval, days)
            elif src == "twelve_data":
                df = self._fetch_twelve_data(symbol, interval, days)
            elif src == "finnhub":
                df = self._fetch_finnhub_candles(symbol, interval, days)
        except Exception as e:
            logger.warning(f"Primary source {src} failed for {symbol}: {e}")
            backup = self.config.data.backup_source
            if backup != src:
                logger.info(f"Trying backup source: {backup}")
                try:
                    if backup == "yfinance":
                        df = self._fetch_yfinance(symbol, interval, days)
                    elif backup == "twelve_data":
                        df = self._fetch_twelve_data(symbol, interval, days)
                    elif backup == "ibkr":
                        df = self._fetch_ibkr(symbol, interval, days)
                except Exception as e2:
                    logger.error(f"Backup source {backup} also failed: {e2}")

        if not df.empty:
            df = self._clean_data(df)
            self.cache[cache_key] = df
            self.cache_timestamps[cache_key] = time.time()

        return df

    def _fetch_ibkr(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch historical data directly from IBKR."""
        if not self._ib:
            raise ValueError("IBKR connection not available")

        from ib_async import Stock

        # Create contract
        if symbol.endswith(".DE"):
            clean = symbol.replace(".DE", "")
            contract = Stock(clean, "SMART", "EUR")
        else:
            contract = Stock(symbol, "SMART", "USD")

        # Qualify contract
        self._ib.qualifyContracts(contract)

        # IBKR bar size mapping
        bar_map = {
            "1m": "1 min", "5m": "5 mins", "15m": "15 mins",
            "30m": "30 mins", "1h": "1 hour", "1d": "1 day",
        }
        bar_size = bar_map.get(interval, "5 mins")

        # Duration string
        if days <= 1:
            duration = f"{days * 86400} S"
        else:
            duration = f"{days} D"

        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,       # Regular trading hours only
            formatDate=1,
        )

        if not bars:
            raise ValueError(f"No IBKR data returned for {symbol}")

        df = pd.DataFrame([
            {
                "datetime": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        return df

    def _fetch_yfinance(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch from Yahoo Finance via yfinance library."""
        import yfinance as yf

        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m",
            "30m": "30m", "1h": "60m", "1d": "1d",
        }
        yf_interval = interval_map.get(interval, interval)

        period_map = {"1m": min(days, 7), "5m": min(days, 60)}
        actual_days = period_map.get(yf_interval, days)

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            period=f"{actual_days}d",
            interval=yf_interval,
            auto_adjust=True,
        )

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index.name = "datetime"
        return df

    def _fetch_twelve_data(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch from Twelve Data API."""
        api_key = self.config.api_keys.twelve_data
        if not api_key:
            raise ValueError("TWELVE_DATA_API_KEY not configured")

        clean_symbol = symbol.replace(".DE", "")
        exchange = "XETRA" if ".DE" in symbol else ""

        params = {
            "symbol": clean_symbol,
            "interval": interval,
            "outputsize": min(days * 78, 5000),
            "apikey": api_key,
        }
        if exchange:
            params["exchange"] = exchange

        resp = requests.get(
            "https://api.twelvedata.com/time_series",
            params=params,
            timeout=30,
        )
        data = resp.json()

        if "values" not in data:
            raise ValueError(f"Twelve Data error: {data.get('message', 'Unknown')}")

        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["open", "high", "low", "close", "volume"]]

    def _fetch_finnhub_candles(self, symbol: str, interval: str, days: int) -> pd.DataFrame:
        """Fetch candle data from Finnhub."""
        api_key = self.config.api_keys.finnhub
        if not api_key:
            raise ValueError("FINNHUB_API_KEY not configured")

        res_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "1d": "D"}
        resolution = res_map.get(interval, "5")

        now = int(time.time())
        start = now - (days * 86400)

        resp = requests.get(
            "https://finnhub.io/api/v1/stock/candle",
            params={
                "symbol": symbol.replace(".DE", ""),
                "resolution": resolution,
                "from": start,
                "to": now,
                "token": api_key,
            },
            timeout=30,
        )
        data = resp.json()

        if data.get("s") != "ok":
            raise ValueError(f"Finnhub error: {data}")

        df = pd.DataFrame({
            "open": data["o"],
            "high": data["h"],
            "low": data["l"],
            "close": data["c"],
            "volume": data["v"],
            "datetime": pd.to_datetime(data["t"], unit="s"),
        })
        df = df.set_index("datetime").sort_index()
        return df

    # ─── Real-Time Data ───────────────────────────────────────

    def get_quote(self, symbol: str) -> Optional[dict]:
        """
        Get current quote. Prefers IBKR (fastest), falls back to Finnhub REST.
        """
        # Try IBKR first (lowest latency)
        if self._ib:
            try:
                return self._get_ibkr_quote(symbol)
            except Exception as e:
                logger.debug(f"IBKR quote failed for {symbol}: {e}")

        # Fallback to Finnhub REST (rate-limited to 1 call/sec)
        try:
            api_key = self.config.api_keys.finnhub
            if api_key:
                elapsed = time.time() - self._last_finnhub_call
                if elapsed < 1.0:
                    time.sleep(1.0 - elapsed)
                self._last_finnhub_call = time.time()
                resp = requests.get(
                    "https://finnhub.io/api/v1/quote",
                    params={"symbol": symbol.replace(".DE", ""), "token": api_key},
                    timeout=10,
                )
                data = resp.json()
                return {
                    "current": data.get("c", 0),
                    "high": data.get("h", 0),
                    "low": data.get("l", 0),
                    "open": data.get("o", 0),
                    "prev_close": data.get("pc", 0),
                    "change": data.get("d", 0),
                    "change_pct": data.get("dp", 0),
                    "timestamp": data.get("t", 0),
                }
        except Exception as e:
            logger.error(f"Quote fetch failed for {symbol}: {e}")
        return None

    def _get_ibkr_quote(self, symbol: str) -> dict:
        """
        Get quote from IBKR. Strategy:
        1. Try portfolio cache first (free, no API call, updated every ~3 min by IBKR)
        2. Fall back to snapshot request only if no portfolio data exists
        """
        import math

        # ─── Strategy 1: Portfolio price (no API call needed) ─────────
        # IBKR sends updatePortfolio callbacks every ~3 min for open positions.
        # This is free, fast, and doesn't consume market data request IDs.
        portfolio_price = self._get_portfolio_price(symbol)
        if portfolio_price and portfolio_price > 0:
            return {
                "current": portfolio_price,
                "bid": 0,
                "ask": 0,
                "high": 0,
                "low": 0,
                "open": 0,
                "volume": 0,
                "timestamp": time.time(),
                "source": "portfolio",
            }

        # ─── Strategy 2: Snapshot request (fallback for non-positions) ─
        from ib_async import Stock

        if symbol.endswith(".DE"):
            contract = Stock(symbol.replace(".DE", ""), "SMART", "EUR")
        else:
            contract = Stock(symbol, "SMART", "USD")

        try:
            qualified = self._ib.qualifyContracts(contract)
            if not qualified or not qualified[0].conId:
                raise ValueError(f"Contract not found: {symbol}")
            contract = qualified[0]
        except Exception as e:
            raise ValueError(f"Contract qualification failed for {symbol}: {e}")

        # Request market data snapshot (uses delayed data if not subscribed)
        ticker = self._ib.reqMktData(contract, snapshot=True)
        try:
            self._ib.sleep(3)  # Wait 3s (delayed data may take longer than 2s)

            def safe(val, default=0):
                """Return val if it's a valid number, else default (handles NaN)."""
                try:
                    return default if val is None or math.isnan(val) else val
                except TypeError:
                    return default

            result = {
                "current": safe(ticker.last) or safe(ticker.close),
                "bid": safe(ticker.bid),
                "ask": safe(ticker.ask),
                "high": safe(ticker.high),
                "low": safe(ticker.low),
                "open": safe(ticker.open),
                "volume": safe(ticker.volume),
                "timestamp": time.time(),
                "source": "snapshot",
            }

            # Calculate spread
            if result["bid"] > 0 and result["ask"] > 0:
                mid = (result["bid"] + result["ask"]) / 2
                result["spread_pct"] = (result["ask"] - result["bid"]) / mid

        finally:
            # ALWAYS cancel snapshot subscription to prevent resource leak
            try:
                self._ib.cancelMktData(contract)
            except Exception:
                pass

        # If no usable price data came through, raise so get_quote falls back
        if result["current"] == 0 and result["bid"] == 0 and result["ask"] == 0:
            raise ValueError(f"No market data received for {symbol}")

        return result

    def _get_portfolio_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price from IBKR portfolio updates (no API call needed).
        IBKR sends updatePortfolio callbacks every ~3 minutes for all positions.
        Returns the marketPrice if the symbol is in the portfolio with qty > 0.
        """
        if not self._ib:
            return None
        try:
            # Strip .DE suffix to match IBKR's localSymbol
            ibkr_symbol = symbol.replace(".DE", "")
            for item in self._ib.portfolio():
                if item.contract.localSymbol == ibkr_symbol and item.position != 0:
                    price = item.marketPrice
                    if price and price > 0:
                        logger.debug(f"Portfolio price for {symbol}: {price:.2f}")
                        return price
        except Exception as e:
            logger.debug(f"Portfolio price lookup failed for {symbol}: {e}")
        return None

    def start_live_feed(self, symbols: list[str], on_price: Optional[Callable] = None):
        """Start real-time price streaming via IBKR or Finnhub WebSocket."""
        if on_price:
            self._price_callbacks.append(on_price)

        if self._ib:
            self._start_ibkr_streaming(symbols)
        else:
            self._ws_running = True
            self._ws_thread = threading.Thread(
                target=self._finnhub_ws_loop, args=(symbols,), daemon=True,
            )
            self._ws_thread.start()

        logger.info(f"Live feed started for {len(symbols)} symbols")

    def _start_ibkr_streaming(self, symbols: list[str]):
        """Subscribe to IBKR streaming market data."""
        from ib_async import Stock

        for symbol in symbols:
            if symbol.endswith(".DE"):
                contract = Stock(symbol.replace(".DE", ""), "SMART", "EUR")
            else:
                contract = Stock(symbol, "SMART", "USD")

            try:
                self._ib.qualifyContracts(contract)
                ticker = self._ib.reqMktData(contract)

                # Set up callback for price updates
                def on_ticker_update(t, symbol=symbol):
                    price = t.last if t.last == t.last else t.close
                    if price and price == price:  # NaN check
                        self.live_prices[symbol] = {
                            "price": price,
                            "bid": t.bid,
                            "ask": t.ask,
                            "volume": t.volume,
                            "timestamp": time.time(),
                        }
                        for cb in self._price_callbacks:
                            try:
                                cb(symbol, price, t.volume, time.time())
                            except Exception as e:
                                logger.error(f"Price callback error: {e}")

                ticker.updateEvent += on_ticker_update
                logger.debug(f"Subscribed to IBKR stream: {symbol}")

            except Exception as e:
                logger.warning(f"IBKR stream subscription failed for {symbol}: {e}")

    def _finnhub_ws_loop(self, symbols: list[str]):
        """Finnhub WebSocket event loop (fallback)."""
        try:
            import websocket
        except ImportError:
            logger.error("websocket-client not installed")
            return

        api_key = self.config.api_keys.finnhub
        if not api_key:
            logger.error("FINNHUB_API_KEY required for live feed")
            return

        def on_message(ws, message):
            data = json.loads(message)
            if data.get("type") == "trade":
                for trade in data.get("data", []):
                    symbol = trade["s"]
                    price = trade["p"]
                    volume = trade["v"]
                    timestamp = trade["t"] / 1000
                    self.live_prices[symbol] = {
                        "price": price, "volume": volume,
                        "timestamp": timestamp,
                        "datetime": datetime.fromtimestamp(timestamp),
                    }
                    for callback in self._price_callbacks:
                        try:
                            callback(symbol, price, volume, timestamp)
                        except Exception as e:
                            logger.error(f"Price callback error: {e}")

        def on_open(ws):
            for symbol in symbols:
                ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))

        while self._ws_running:
            try:
                ws = websocket.WebSocketApp(
                    f"wss://ws.finnhub.io?token={api_key}",
                    on_message=on_message, on_open=on_open,
                    on_error=lambda ws, e: logger.error(f"WS error: {e}"),
                    on_close=lambda ws, s, m: logger.info(f"WS closed"),
                )
                ws.run_forever(ping_interval=30)
            except Exception as e:
                logger.error(f"WebSocket failed: {e}")
                if self._ws_running:
                    time.sleep(5)

    def stop_live_feed(self):
        """Stop all live data feeds."""
        self._ws_running = False
        if self._ws_thread:
            self._ws_thread.join(timeout=5)
        # IBKR streaming is managed by the connection lifecycle
        logger.info("Live feed stopped")

    def get_live_price(self, symbol: str) -> Optional[dict]:
        """Get the latest live price for a symbol."""
        return self.live_prices.get(symbol)

    # ─── Utilities ────────────────────────────────────────────

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV data."""
        # Drop rows with NaN in essential OHLC columns
        df = df.dropna(subset=["open", "high", "low", "close"])
        # Fill missing volume with 0, then remove zero-volume rows (no real trading activity)
        df["volume"] = df["volume"].fillna(0)
        df = df[df["volume"] > 0]
        for col in ["open", "high", "low", "close"]:
            df[col] = df[col].astype(float)
        df["volume"] = df["volume"].astype(float)
        df = df.sort_index()
        return df

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still fresh."""
        if key not in self.cache:
            return False
        age = time.time() - self.cache_timestamps.get(key, 0)
        return age < (self.config.data.cache_minutes * 60)

    def save_data(self, symbol: str, df: pd.DataFrame, interval: str):
        """Save data to CSV for backtesting."""
        path = Path(self.config.data.data_dir)
        path.mkdir(parents=True, exist_ok=True)
        filename = path / f"{symbol.replace('.', '_')}_{interval}.csv"
        df.to_csv(filename)
        logger.info(f"Saved {len(df)} bars to {filename}")

    def load_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Load previously saved data."""
        path = Path(self.config.data.data_dir) / f"{symbol.replace('.', '_')}_{interval}.csv"
        if path.exists():
            return pd.read_csv(path, index_col=0, parse_dates=True)
        return None
