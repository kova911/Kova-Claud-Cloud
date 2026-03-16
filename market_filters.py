"""
Market Filters — Pre-Trade Safety Checks
Blocks trades during dangerous conditions that technical indicators can't detect.

Filters:
1. Economic Calendar: avoid trading around high-impact events (ECB, NFP, earnings)
2. Market Sentiment: skip when VIX is spiking or broad market is in panic
3. Spread Filter: block entries when bid-ask spread is too wide
"""

import logging
import time
from datetime import datetime, date, timedelta
from typing import Optional
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import requests
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Economic Calendar Filter ──────────────────────────────────

class EconomicCalendarFilter:
    """
    Fetches scheduled economic events and earnings releases.
    Blocks trading N minutes before/after high-impact events.

    Data source: Finnhub (free tier).
    """

    def __init__(self, config):
        self.api_key = config.api_keys.finnhub
        self.fc = config.filters
        self._events_cache: list[dict] = []
        self._earnings_cache: list[dict] = []
        self._cache_date: Optional[date] = None

    def refresh(self):
        """Fetch today's economic events and earnings. Called once at session start."""
        today = date.today()
        if self._cache_date == today:
            return  # Already fetched today

        self._events_cache = []
        self._earnings_cache = []

        if not self.api_key:
            logger.warning("No Finnhub API key — economic calendar filter disabled")
            self._cache_date = today
            return

        # Fetch economic calendar (ECB, NFP, CPI, etc.)
        try:
            today_str = today.strftime("%Y-%m-%d")
            resp = requests.get(
                "https://finnhub.io/api/v1/calendar/economic",
                params={"from": today_str, "to": today_str, "token": self.api_key},
                timeout=10,
            )
            data = resp.json()
            events = data.get("economicCalendar", [])

            # Keep only high and medium impact events
            for event in events:
                impact = event.get("impact", "").lower()
                if impact in ("high", "medium"):
                    self._events_cache.append({
                        "time": event.get("time", ""),
                        "country": event.get("country", ""),
                        "event": event.get("event", ""),
                        "impact": impact,
                    })

            if self._events_cache:
                logger.info(f"Economic calendar: {len(self._events_cache)} high/medium impact events today")
                for e in self._events_cache:
                    logger.info(f"  [{e['impact'].upper()}] {e['time']} {e['country']}: {e['event']}")
            else:
                logger.info("Economic calendar: no high-impact events today")

        except Exception as e:
            logger.warning(f"Economic calendar fetch failed: {e}")

        # Fetch earnings calendar (company-specific)
        try:
            today_str = today.strftime("%Y-%m-%d")
            resp = requests.get(
                "https://finnhub.io/api/v1/calendar/earnings",
                params={"from": today_str, "to": today_str, "token": self.api_key},
                timeout=10,
            )
            data = resp.json()
            earnings = data.get("earningsCalendar", [])

            for earning in earnings:
                self._earnings_cache.append({
                    "symbol": earning.get("symbol", ""),
                    "date": earning.get("date", ""),
                    "hour": earning.get("hour", ""),  # "bmo" (before market open) or "amc" (after market close)
                    "estimate": earning.get("epsEstimate"),
                })

            if self._earnings_cache:
                logger.info(f"Earnings calendar: {len(self._earnings_cache)} reports today")

        except Exception as e:
            logger.warning(f"Earnings calendar fetch failed: {e}")

        self._cache_date = today

    def is_blocked(self, symbol: str = "") -> tuple[bool, str]:
        """
        Check if trading should be blocked due to an upcoming event.

        Returns:
            (blocked: bool, reason: str)
        """
        if not self.fc.economic_calendar_enabled:
            return False, ""

        self.refresh()
        now_utc = datetime.now(ZoneInfo("UTC"))
        buffer_minutes = self.fc.event_buffer_minutes

        # Check economic events (time-based)
        for event in self._events_cache:
            event_time_str = event.get("time", "")
            if not event_time_str:
                continue

            try:
                # Finnhub returns time in UTC as "HH:MM:SS" or "HH:MM"
                parts = event_time_str.split(":")
                event_hour = int(parts[0])
                event_min = int(parts[1]) if len(parts) > 1 else 0
                event_dt = now_utc.replace(hour=event_hour, minute=event_min, second=0, microsecond=0)

                # Block within buffer_minutes before and after the event
                time_diff = abs((now_utc - event_dt).total_seconds()) / 60

                if time_diff <= buffer_minutes:
                    return True, (
                        f"Economic event in {time_diff:.0f}min: "
                        f"[{event['impact'].upper()}] {event['country']} {event['event']}"
                    )
            except (ValueError, IndexError):
                continue

        # Check earnings for specific symbol
        if symbol:
            clean_symbol = symbol.replace(".DE", "").upper()
            for earning in self._earnings_cache:
                if earning["symbol"].upper() == clean_symbol:
                    return True, f"Earnings release today for {symbol}"

        return False, ""


# ─── Market Sentiment Filter ──────────────────────────────────

class MarketSentimentFilter:
    """
    Checks broad market conditions before allowing trades.

    - VIX (fear index): if too high or spiking, reduce/stop trading
    - DAX/S&P futures: if deeply negative, skip long entries on stocks

    Data source: Yahoo Finance (free).
    """

    def __init__(self, config):
        self.fc = config.filters
        self._vix_level: Optional[float] = None
        self._vix_change_pct: Optional[float] = None
        self._dax_change_pct: Optional[float] = None
        self._sp500_change_pct: Optional[float] = None
        self._last_refresh: float = 0

    def refresh(self, force: bool = False):
        """
        Fetch current VIX and index data.
        Refreshes at most once every 15 minutes to avoid API spam.
        """
        if not self.fc.sentiment_enabled:
            return

        now = time.time()
        if not force and (now - self._last_refresh) < 900:  # 15 min cache
            return

        try:
            import yfinance as yf

            # VIX — fear gauge
            try:
                vix = yf.Ticker("^VIX")
                vix_hist = vix.history(period="2d", interval="1d")
                if len(vix_hist) >= 1:
                    self._vix_level = float(vix_hist["Close"].iloc[-1])
                if len(vix_hist) >= 2:
                    prev_close = float(vix_hist["Close"].iloc[-2])
                    if prev_close > 0:
                        self._vix_change_pct = (self._vix_level - prev_close) / prev_close
                if self._vix_level is not None:
                    change_str = f" (change: {self._vix_change_pct:+.1%})" if self._vix_change_pct is not None else ""
                    logger.info(f"VIX: {self._vix_level:.1f}{change_str}")
                else:
                    logger.info("VIX: unavailable")
            except Exception as e:
                logger.warning(f"VIX fetch failed: {e}")

            # DAX — European market mood
            try:
                dax = yf.Ticker("^GDAXI")
                dax_hist = dax.history(period="2d", interval="1d")
                if len(dax_hist) >= 2:
                    today_close = float(dax_hist["Close"].iloc[-1])
                    prev_close = float(dax_hist["Close"].iloc[-2])
                    if prev_close > 0:
                        self._dax_change_pct = (today_close - prev_close) / prev_close
                    logger.info(f"DAX: {self._dax_change_pct:+.2%}")
            except Exception as e:
                logger.warning(f"DAX fetch failed: {e}")

            # S&P 500 — US market mood (affects EU afternoon session)
            try:
                sp = yf.Ticker("^GSPC")
                sp_hist = sp.history(period="2d", interval="1d")
                if len(sp_hist) >= 2:
                    today_close = float(sp_hist["Close"].iloc[-1])
                    prev_close = float(sp_hist["Close"].iloc[-2])
                    if prev_close > 0:
                        self._sp500_change_pct = (today_close - prev_close) / prev_close
                    logger.info(f"S&P 500: {self._sp500_change_pct:+.2%}")
            except Exception as e:
                logger.warning(f"S&P 500 fetch failed: {e}")

            self._last_refresh = now

        except ImportError:
            logger.warning("yfinance not installed — sentiment filter disabled")

    def is_blocked(self, signal_type: str = "") -> tuple[bool, str]:
        """
        Check if market sentiment blocks trading (tiered VIX response).

        VIX Tiers:
            < 25  (normal):   Full trading, full size
            25-30 (elevated): Block LONGs, SHORTs at 50% size
            30-40 (high):     SHORT-only at 25% size
            > 40  (panic):    Block ALL trading

        Args:
            signal_type: "LONG" or "SHORT" — some conditions only block one direction

        Returns:
            (blocked: bool, reason: str)
        """
        if not self.fc.sentiment_enabled:
            return False, ""

        self.refresh()

        # VIX spike — block ALL trading regardless of level
        if self._vix_change_pct is not None and self._vix_change_pct >= self.fc.vix_spike_pct:
            return True, f"VIX spiking: {self._vix_change_pct:+.1%} (threshold: {self.fc.vix_spike_pct:+.1%})"

        # Tiered VIX response
        if self._vix_level is not None:
            # PANIC: VIX >= 40 — block everything
            if self._vix_level >= self.fc.vix_panic_level:
                return True, f"VIX panic: {self._vix_level:.1f} (>={self.fc.vix_panic_level}) — all trading blocked"

            # HIGH: VIX 30-40 — SHORT only
            if self._vix_level >= self.fc.vix_elevated:
                if signal_type == "LONG":
                    return True, f"VIX high: {self._vix_level:.1f} — LONGs blocked (SHORT-only mode)"
                # SHORTs allowed — size reduced to 25% via get_vix_size_multiplier()

            # ELEVATED: VIX 25-30 — block LONGs
            elif self._vix_level >= self.fc.vix_normal:
                if signal_type == "LONG":
                    return True, f"VIX elevated: {self._vix_level:.1f} — LONGs blocked"
                # SHORTs allowed — size reduced to 50% via get_vix_size_multiplier()

        # DAX deeply negative — block LONG on EU stocks
        if signal_type == "LONG" and self._dax_change_pct is not None:
            if self._dax_change_pct <= self.fc.index_panic_drop:
                return True, f"DAX down {self._dax_change_pct:.2%} — blocking LONG entries"

        # S&P deeply negative — block LONG on US stocks
        if signal_type == "LONG" and self._sp500_change_pct is not None:
            if self._sp500_change_pct <= self.fc.index_panic_drop:
                return True, f"S&P 500 down {self._sp500_change_pct:.2%} — blocking LONG entries"

        return False, ""

    def get_vix_size_multiplier(self) -> float:
        """
        Returns position size multiplier based on current VIX tier.

        Returns:
            1.0 = full size (VIX < 25)
            0.5 = half size (VIX 25-30, elevated)
            0.25 = quarter size (VIX 30-40, high)
            0.0 = no trading (VIX >= 40, panic)
        """
        if self._vix_level is None:
            return 1.0
        if self._vix_level >= self.fc.vix_panic_level:
            return 0.0
        if self._vix_level >= self.fc.vix_elevated:
            return 0.25
        if self._vix_level >= self.fc.vix_normal:
            return 0.5
        return 1.0

    def get_status(self) -> dict:
        """Return current sentiment readings."""
        return {
            "vix_level": self._vix_level,
            "vix_change_pct": self._vix_change_pct,
            "dax_change_pct": self._dax_change_pct,
            "sp500_change_pct": self._sp500_change_pct,
        }


# ─── Spread Filter ────────────────────────────────────────────

class SpreadFilter:
    """
    Checks bid-ask spread before allowing trade entry.

    Wide spreads eat into our 0.75% profit target.
    If spread > threshold, the trade is blocked.
    """

    def __init__(self, config):
        self.fc = config.filters
        self._ib = None

    def set_ibkr_connection(self, ib):
        """Share IBKR connection for spread data."""
        self._ib = ib

    def check_spread(self, symbol: str, bid: float = 0, ask: float = 0) -> tuple[bool, str, float]:
        """
        Check if spread is acceptable for entry.

        Args:
            symbol: Ticker symbol
            bid: Current bid price (if already known)
            ask: Current ask price (if already known)

        Returns:
            (acceptable: bool, reason: str, spread_pct: float)
        """
        if not self.fc.spread_filter_enabled:
            return True, "", 0.0

        # If bid/ask not provided, try to get from IBKR
        if (bid <= 0 or ask <= 0) and self._ib:
            try:
                bid, ask = self._get_ibkr_spread(symbol)
            except Exception as e:
                logger.debug(f"Spread check failed for {symbol}: {e}")
                # If we can't get spread data, allow the trade (don't block on data failure)
                return True, "spread data unavailable", 0.0

        if bid <= 0 or ask <= 0:
            return True, "no bid/ask data", 0.0

        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid if mid > 0 else 0

        max_spread = self.fc.max_spread_pct

        if spread_pct > max_spread:
            return False, (
                f"Spread too wide: {spread_pct:.4%} > {max_spread:.4%} "
                f"(bid: {bid:.2f}, ask: {ask:.2f})"
            ), spread_pct

        return True, f"spread OK: {spread_pct:.4%}", spread_pct

    def _get_ibkr_spread(self, symbol: str) -> tuple[float, float]:
        """Get current bid/ask from IBKR."""
        from ib_async import Stock
        import math

        if symbol.endswith(".DE"):
            contract = Stock(symbol.replace(".DE", ""), "SMART", "EUR")
        else:
            contract = Stock(symbol, "SMART", "USD")

        try:
            qualified = self._ib.qualifyContracts(contract)
            if not qualified or not qualified[0].conId:
                return 0, 0
            contract = qualified[0]
        except Exception:
            return 0, 0

        ticker = self._ib.reqMktData(contract, snapshot=True)
        try:
            self._ib.sleep(1)

            def safe(val):
                try:
                    return 0 if val is None or math.isnan(val) else val
                except TypeError:
                    return 0

            bid = safe(ticker.bid)
            ask = safe(ticker.ask)
            return bid, ask
        finally:
            # ALWAYS cancel to prevent IBKR market data line leak
            try:
                self._ib.cancelMktData(contract)
            except Exception:
                pass


# ─── Trading Hours Filter ────────────────────────────────────

class TradingHoursFilter:
    """
    Enforces trading hours based on ScheduleConfig.
    EU symbols (.DE): only trade during XETRA hours + offset (09:15–17:15 CET)
    US symbols: only trade during NYSE hours + offset (09:45–15:45 ET)
    """

    def __init__(self, config):
        self.sc = config.schedule
        self._tz_eu = ZoneInfo(self.sc.timezone)            # Europe/Berlin
        self._tz_us = ZoneInfo("America/New_York")

    def is_blocked(self, symbol: str = "") -> tuple[bool, str]:
        """Check if current time is outside trading hours for this symbol."""
        now_utc = datetime.now(ZoneInfo("UTC"))

        if symbol.endswith(".DE"):
            return self._check_eu(now_utc)
        elif symbol:
            return self._check_us(now_utc)
        else:
            # Unknown symbol — check EU hours (conservative)
            return self._check_eu(now_utc)

    def _check_eu(self, now_utc: datetime) -> tuple[bool, str]:
        """Check XETRA trading hours with offset."""
        now_local = now_utc.astimezone(self._tz_eu)
        open_parts = self.sc.market_open.split(":")
        close_parts = self.sc.market_close.split(":")

        market_open = now_local.replace(
            hour=int(open_parts[0]), minute=int(open_parts[1]), second=0, microsecond=0
        )
        market_close = now_local.replace(
            hour=int(close_parts[0]), minute=int(close_parts[1]), second=0, microsecond=0
        )

        # Apply offsets (avoid first/last N minutes)
        trade_start = market_open + timedelta(minutes=self.sc.trading_start_offset_min)
        trade_end = market_close - timedelta(minutes=self.sc.trading_end_offset_min)

        if now_local < trade_start:
            return True, f"Before EU trading hours (starts {trade_start.strftime('%H:%M')} CET)"
        if now_local > trade_end:
            return True, f"After EU trading hours (ended {trade_end.strftime('%H:%M')} CET)"

        # Check weekend
        if now_local.weekday() >= 5:
            return True, "Weekend — markets closed"

        return False, ""

    def _check_us(self, now_utc: datetime) -> tuple[bool, str]:
        """Check NYSE trading hours with offset."""
        now_local = now_utc.astimezone(self._tz_us)
        open_parts = self.sc.us_market_open.split(":")
        close_parts = self.sc.us_market_close.split(":")

        market_open = now_local.replace(
            hour=int(open_parts[0]), minute=int(open_parts[1]), second=0, microsecond=0
        )
        market_close = now_local.replace(
            hour=int(close_parts[0]), minute=int(close_parts[1]), second=0, microsecond=0
        )

        trade_start = market_open + timedelta(minutes=self.sc.trading_start_offset_min)
        trade_end = market_close - timedelta(minutes=self.sc.trading_end_offset_min)

        if now_local < trade_start:
            return True, f"Before US trading hours (starts {trade_start.strftime('%H:%M')} ET)"
        if now_local > trade_end:
            return True, f"After US trading hours (ended {trade_end.strftime('%H:%M')} ET)"

        if now_local.weekday() >= 5:
            return True, "Weekend — markets closed"

        return False, ""


# ─── Combined Market Filters ──────────────────────────────────

class MarketFilters:
    """
    Combines all pre-trade filters into a single check.

    Usage:
        filters = MarketFilters(config)
        filters.startup()  # Fetch calendar + sentiment at session start

        # Before each trade:
        can_trade, reason = filters.check(symbol="SAP.DE", signal_type="LONG")
        if not can_trade:
            logger.info(f"Trade blocked: {reason}")
    """

    def __init__(self, config):
        self.config = config
        self.hours = TradingHoursFilter(config)
        self.calendar = EconomicCalendarFilter(config)
        self.sentiment = MarketSentimentFilter(config)
        self.spread = SpreadFilter(config)

    def startup(self):
        """Initialize all filters. Call once at session start."""
        logger.info("Initializing market filters...")
        self.calendar.refresh()
        self.sentiment.refresh(force=True)
        logger.info("Market filters ready.")

    def set_ibkr_connection(self, ib):
        """Share IBKR connection with spread filter."""
        self.spread.set_ibkr_connection(ib)

    def check(
        self,
        symbol: str = "",
        signal_type: str = "",
        bid: float = 0,
        ask: float = 0,
    ) -> tuple[bool, str]:
        """
        Run all filters. Returns (can_trade, reason).

        Args:
            symbol: Ticker symbol being traded
            signal_type: "LONG" or "SHORT"
            bid: Current bid (optional, fetched from IBKR if not provided)
            ask: Current ask (optional, fetched from IBKR if not provided)
        """
        # 0. Trading hours check
        blocked, reason = self.hours.is_blocked(symbol)
        if blocked:
            logger.info(f"[FILTER] Hours block: {reason}")
            return False, f"Hours: {reason}"

        # 1. Economic calendar check
        blocked, reason = self.calendar.is_blocked(symbol)
        if blocked:
            logger.info(f"[FILTER] Calendar block: {reason}")
            return False, f"Calendar: {reason}"

        # 2. Market sentiment check
        blocked, reason = self.sentiment.is_blocked(signal_type)
        if blocked:
            logger.info(f"[FILTER] Sentiment block: {reason}")
            return False, f"Sentiment: {reason}"

        # 3. Spread check
        ok, reason, spread = self.spread.check_spread(symbol, bid, ask)
        if not ok:
            logger.info(f"[FILTER] Spread block: {reason}")
            return False, f"Spread: {reason}"

        return True, "All filters passed"

    def get_status(self) -> dict:
        """Return status of all filters for dashboard/logging."""
        return {
            "sentiment": self.sentiment.get_status(),
            "calendar_events_today": len(self.calendar._events_cache),
            "earnings_today": len(self.calendar._earnings_cache),
        }
