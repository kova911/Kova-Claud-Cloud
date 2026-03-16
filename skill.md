# Trading System Skill — Reference & Operating Protocol

> **Version**: 2.7.0
> **Owner**: 911 Ventures / ShopsApp GmbH
> **Operator**: Cowork (Professional Trader role)
> **Last Updated**: 2026-03-06

---

## System Overview

Automated scalping trading system targeting small, frequent gains on EU/US equities via Interactive Brokers. The system generates signals based on multi-indicator confluence (mean-reversion + momentum) with mandatory 0.5% stop-loss, adaptive regime detection, and continuous self-learning.

**Core principle**: Win small, win often. Protect capital at all costs.

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌─────────────┐
│  Data Feed   │────▶│  Indicators  │────▶│  Strategy   │
│ (IBKR/YF/FH) │     │ (EMA,RSI,BB) │     │ (Signals)   │
└──────────────┘     └──────────────┘     └──────┬──────┘
                                                  │
┌──────────────┐                          ┌───────▼──────┐
│ Mkt Filters  │─────────────────────────▶│ Risk Manager │
│(Cal/VIX/Sprd)│                          │ (SL/TP/Size) │
└──────────────┘                          └──────┬───────┘
                                                  │
                     ┌──────────────┐     ┌───────▼──────┐
                     │   Executor   │◀────│   Execute    │
                     │ (IBKR/Paper) │     │  (Bracket)   │
                     └──────┬───────┘     └──────────────┘
                            │
                     ┌──────▼───────┐     ┌──────────────┐
                     │  Trade Log   │────▶│ Self-Learner │
                     │  (History)   │     │ (Optimize)   │
                     └──────────────┘     └──────────────┘
```

### Files

| File | Purpose |
|------|---------|
| `config.py` | All settings, API keys, IBKR connection, strategy parameters |
| `data_collector.py` | Market data from IBKR (delayed fallback), Finnhub, Twelve Data, Yahoo Finance |
| `indicators.py` | Technical indicators (EMA, RSI, MACD, BB, VWAP, ADX, ATR) |
| `strategy.py` | Signal generation (mean-reversion + momentum) |
| `risk_manager.py` | Position sizing, stop-loss, daily limits, state persistence |
| `backtester.py` | Walk-forward backtesting with cost modeling |
| `learning.py` | Self-optimization of strategy parameters |
| `market_filters.py` | Pre-trade safety: economic calendar, sentiment, spread checks |
| `trade_executor.py` | Interactive Brokers order interface via ib_async, bracket leg tracking, dynamic tick sizes |
| `main.py` | Orchestrator — bracket-aware monitoring, state recovery, IBKR reconciliation |
| `analytics.py` | Institutional-grade analytics (Sortino, VaR/CVaR, Monte Carlo, attribution) |
| `skill_updater.py` | Auto-updates skill.md with performance data every 60 minutes |
| `trading_knowledge.py` | Persistent learning engine — records every signal/trade, mines patterns, generates knowledge_base.md |
| `cleanup_positions.py` | Utility — flatten all IBKR paper positions and cancel open orders |
| `skill.md` | This document — operating protocol (auto-updated section at bottom) |
| `knowledge/` | Directory — daily signal/trade JSONL logs, patterns.json, knowledge_base.md |

---

## Daily Trading Workflow

### Pre-Market (08:45 CET)

1. **Start IB Gateway**: Launch IB Gateway (or TWS) and log in
2. **Verify connection**: Paper = port 4002, Live = port 4001
3. **Check positions**: Run `python cleanup_positions.py` to verify account is flat (no orphans from previous session)
4. **Start system**: `python main.py` (paper) or `python main.py --live`
5. **Review overnight**: Check any news/events that affect watchlist
6. **Confirm risk params**: Verify stop-loss is 0.5%, max daily loss is 1.5%

**Note**: On paper accounts without market data subscriptions, position monitoring uses 15-min delayed data (via `reqMarketDataType(3)`). Bracket SL/TP orders on the IBKR server execute on real-time prices regardless.

### Trading Session (09:15–17:15 CET)

6. **Auto-scan**: System scans every 5 minutes for signals
7. **Signal review**: Each signal shows confluence, confidence, and reason
8. **Execution**: Bracket orders (entry + SL + TP) submitted atomically to IBKR
9. **Position monitoring**: Stop-loss and take-profit checked every 10s (paper) / 30s (live)
10. **Trailing stop**: Activates after 0.3% profit, trails at 0.2%

### Post-Market (17:30 CET)

11. **Close all**: System closes remaining positions
12. **Generate report**: PnL, win rate, profit factor logged
13. **Institutional analytics**: Sortino, VaR/CVaR, t-test, attribution, Monte Carlo printed to log
14. **Self-learning check**: If performance degrades, optimization triggers
15. **Review**: Operator reviews daily report in `logs/` and `reports/`

---

## Strategy Details

### Mean-Reversion Mode (Ranging Markets, ADX < 20)

Trades when price deviates from equilibrium and expects reversion.

**LONG signals require 3+ of these:**
- Price ≤ lower Bollinger Band
- RSI < 25 (oversold)
- Price below VWAP
- Volume > 1.2× average
- Stochastic %K < 20

**SHORT signals require 3+ of these:**
- Price ≥ upper Bollinger Band
- RSI > 75 (overbought)
- Price above VWAP
- Volume > 1.2× average
- Stochastic %K > 80

### Momentum Mode (Trending Markets, ADX > 25)

Trades in the direction of the established trend.

**LONG signals (uptrend only) require 3+ of these:**
- EMA(9) crosses above EMA(21)
- Price above VWAP
- RSI between 45–70 (momentum but not exhausted)
- Volume surge (1.2×+ average)
- MACD histogram positive

**SHORT signals (downtrend only) require 3+ of these:**
- EMA(9) crosses below EMA(21)
- Price below VWAP
- RSI between 30–55
- Volume surge
- MACD histogram negative

### Regime Detection

| ADX Value | Regime | Strategy |
|-----------|--------|----------|
| < 20 | Ranging | Mean-reversion |
| 20–25 | Transitional | Wait / reduce size |
| > 25 | Trending | Momentum |
| ATR > 1.5× avg | Volatile | No trading |

---

## Market Filters (Pre-Trade Safety Layer)

Before any signal reaches the executor, it must pass **4 independent filters**. These catch dangers that technical indicators can't see.

### 0. Trading Hours Filter

**Source**: System clock + timezone-aware schedule (`zoneinfo`)

Blocks all trading outside market hours, including a configurable buffer at open/close to avoid the high-volatility first and last minutes.

| Market | Trading Window | Timezone |
|--------|---------------|----------|
| XETRA (`.DE` symbols) | 09:15–17:15 | CET/CEST (Europe/Berlin) |
| NYSE/NASDAQ (US symbols) | 09:45–15:45 | ET (America/New_York) |

**Config**: `trading_start_offset_min: 15`, `trading_end_offset_min: 15` in `ScheduleConfig`

Also blocks on weekends (Saturday/Sunday).

### 1. Economic Calendar Filter

**Source**: Finnhub API (free tier)

Blocks all trading within a configurable buffer (default: 30 minutes) before and after high-impact scheduled events.

| Event Type | Examples | Why it matters |
|------------|----------|----------------|
| Central bank decisions | ECB rate, Fed FOMC | Instant 1-3% moves, stop-loss gets blown |
| Employment data | US NFP, EU unemployment | High volatility spike |
| Inflation data | CPI, PPI releases | Market direction reversal possible |
| Earnings releases | Per-symbol (SAP, AAPL, etc.) | Gaps, halts, unpredictable |

**Config**: `event_buffer_minutes: 30` in `FiltersConfig`

Also blocks trading on specific stocks that have earnings scheduled for that day (fetched from Finnhub earnings calendar).

### 2. Market Sentiment Filter

**Source**: Yahoo Finance (free — VIX, DAX, S&P 500)

Checks the overall market mood before allowing entries. Refreshes every 15 minutes.

| Condition | Threshold | Action |
|-----------|-----------|--------|
| VIX level | ≥ 30 | **Block ALL trading** (panic mode) |
| VIX day change | ≥ +20% | **Block ALL trading** (fear spike) |
| DAX down | ≤ -2% | **Block LONG entries** on EU stocks |
| S&P 500 down | ≤ -2% | **Block LONG entries** on US stocks |

Note: SHORT entries are still allowed when indices drop — the sentiment filter only blocks going LONG into a falling market.

### 3. Spread Filter

**Source**: IBKR bid/ask data (real-time)

Checks bid-ask spread before every entry. Wide spreads eat profit.

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `max_spread_pct` | 0.03% | With 0.75% target profit, spread should be <5% of expected gain |

**Example**: Stock at EUR 100, bid 99.98, ask 100.02 → spread = 0.04% → **BLOCKED** (exceeds 0.03%).

If IBKR data is unavailable (e.g., local paper mode), the filter allows the trade rather than blocking on data absence.

### Filter Flow

```
Signal generated
    │
    ▼
[Hours] ──blocked──▶ "Before EU trading hours (starts 09:15 CET)" → SKIP
    │ ok
    ▼
[Calendar] ──blocked──▶ "ECB rate decision in 20min" → SKIP
    │ ok
    ▼
[Sentiment] ──blocked──▶ "VIX at 35, panic level" → SKIP
    │ ok
    ▼
[Spread] ──blocked──▶ "Spread 0.05% > 0.03% max" → SKIP
    │ ok
    ▼
[Risk Manager] → Position sizing, daily limits, duplicate symbol check
    │ ok
    ▼
[Executor] → Bracket order to IBKR
```

---

## Risk Management (Non-Negotiable)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Stop-loss | 0.5% | Maximum loss per trade |
| Take-profit | 0.75% | 1:1.5 R:R ratio |
| Risk per trade | 0.5% of account | Fixed-fractional sizing |
| Max daily loss | 1.5% | Stop trading for the day |
| Max consecutive losses | 3 | Cooldown triggered |
| Cooldown after loss | 5 minutes | Prevent revenge trading |
| Max trades/day | 20 | Prevent overtrading |
| Max open positions | 3 | Diversification limit |
| Max single position | 20% of account | Concentration limit |
| Trailing stop | Activates at +0.3%, trails 0.2% | Lock in profits |
| Time exit | 15 candles (75 min on 5m) | No dead trades |
| End-of-day | Close all 15 min before close | No overnight risk |
| Duplicate symbol block | 1 position per symbol | No double-exposure |
| Commission deduction | IBKR tiered per leg | Realistic PnL tracking |

### State Persistence (v2.5.0)

All open positions and risk state are persisted to `data/risk_state.json` on every position open/close. On startup, the system restores any open positions from the previous session (crash recovery). At end-of-day, the state file is cleared after all positions are closed cleanly. Writes are atomic (temp file + rename) to prevent corruption.

### IBKR Position Reconciliation (v2.5.0)

Every 10 minutes, the system compares internal position tracking against the actual IBKR portfolio. Detects three types of discrepancies:

| Type | Meaning | Action |
|------|---------|--------|
| **Orphan** | Position in IBKR but not tracked internally | Logged as warning — operator must close via TWS |
| **Ghost** | Position tracked internally but not in IBKR | Closed internally (bracket may have filled) |
| **Size mismatch** | Internal size ≠ IBKR quantity | Logged as warning for operator review |

### Position Sizing Formula

```
Position Size = (Account × 0.5%) / (Entry Price × 0.5%)
```

Example: Account EUR 10,000, stock at EUR 100:
- Risk amount = 10,000 × 0.005 = EUR 50
- Stop distance = 100 × 0.005 = EUR 0.50
- Position size = 50 / 0.50 = 100 shares

---

## Self-Learning Protocol

### Triggers

Learning is triggered automatically when:
1. **Scheduled**: Every 50 trades (configurable)
2. **Win rate drops** below 55%
3. **Profit factor drops** below 1.3
4. **Max drawdown exceeds** 5%
5. **Manual**: `python main.py --learn`

### Process

```
1. Collect last 30 days of market data
2. Generate parameter grid (EMA periods, RSI thresholds, BB settings, etc.)
3. Run walk-forward backtest for each combination:
   - Train on 30-day window
   - Test on next 10-day window
   - Slide forward, repeat
4. Score each combination:
   - Win rate (30% weight)
   - Profit factor (30% weight)
   - Sharpe ratio (20% weight)
   - Low drawdown (20% weight)
5. Compare best found vs current parameters
6. If improvement > 10%: AUTO-APPLY new parameters
7. If improvement 0–10%: LOG recommendation, wait for operator
8. If no improvement: KEEP current parameters
9. Save full report to reports/ directory
```

### Parameter Ranges (Safety Bounds)

| Parameter | Min | Max | Step |
|-----------|-----|-----|------|
| EMA fast | 5 | 12 | varies |
| EMA slow | 15 | 30 | varies |
| RSI period | 5 | 14 | varies |
| RSI oversold | 20 | 30 | 5 |
| RSI overbought | 70 | 80 | 5 |
| BB period | 10 | 25 | 5 |
| BB std dev | 1.5 | 2.5 | 0.5 |
| Volume mult | 1.0 | 1.5 | 0.25 |
| Min confluence | 2 | 4 | 1 |

### Learning Reports

Saved to `reports/learning_SYMBOL_TIMESTAMP.json`:
- Current vs optimized performance comparison
- Top 10 parameter combinations tested
- Recommendation (STRONG / MODERATE / KEEP CURRENT)
- Whether parameters were auto-applied

---

## Interactive Brokers Integration

### Connection

The system connects to IBKR via the **ib_async** Python library through IB Gateway (recommended) or TWS.

| Mode | Gateway Port | TWS Port |
|------|-------------|----------|
| Paper | 4002 | 7497 |
| Live | 4001 | 7496 |

### Setup

1. **Create IBKR account** at interactivebrokers.com (EUR 0 minimum, includes free paper trading)
2. **Download IB Gateway** (lightweight) or TWS (full desktop app)
3. **Enable API access**:
   - IB Gateway: Configure → Settings → API → Enable ActiveX and Socket Clients
   - TWS: Edit → Global Configuration → API → Settings → Enable
4. **Set socket port**: Paper = 4002, Live = 4001
5. **Allow localhost**: Add 127.0.0.1 to trusted IPs

### Order Execution

The system uses **bracket orders** — entry + stop-loss + take-profit submitted atomically in one request. This means protection is active from the moment the entry fills. No gap between entry and stop placement.

- Order latency: ~75–130ms (via IB Gateway)
- Market data latency: ~250ms
- Time-in-force: DAY (orders cancel at market close)

**Bracket Safety (v2.5.0)**: When a bracket order is active, the risk manager skips client-side SL/TP checks (IBKR handles those server-side). Trailing stop and time-based exit still run client-side — if either fires, the system cancels the remaining IBKR bracket legs before placing the exit order. The `check_bracket_fills()` method polls IBKR for server-side SL/TP fills every loop iteration.

### Contract Routing

| Symbol Pattern | Exchange | Currency | Example |
|---------------|----------|----------|---------|
| `XXX.DE` | SMART (routes to IBIS/XETRA) | EUR | `SAP.DE` → Stock("SAP", "SMART", "EUR") |
| `XXX` (no suffix) | SMART | USD | `AAPL` → Stock("AAPL", "SMART", "USD") |

**Important**: Always use `SMART` routing, never hardcode `IBIS`. SMART routes to the best execution venue (IBIS, IBIS2, TGATE, EUDARK, SBF). ETFs like SXR8 route to `IBIS2`, Airbus routes to `SBF`. Contracts must be qualified via `ib.qualifyContracts()` before use.

### Dynamic Tick Sizes (v2.6.0)

IBKR enforces **price-dependent tick sizes** that vary by instrument and price level. The system uses `reqMarketRule()` to get exact tick schedules.

| Market Rule | Exchange | Example Tick Schedule |
|-------------|----------|----------------------|
| 1873 | XETRA stocks | €0–50: 0.001, €50–100: 0.02, €100–200: 0.05, €200–500: 0.10 |
| 1874 | SBF/Euronext | €0–50: 0.01, €50–100: 0.02, €100–200: 0.02 |

**Flow**: `reqContractDetails()` → `marketRuleIds` → `reqMarketRule(id)` → iterate `PriceIncrement` objects → find increment for current price level.

**Example**: ALV.DE at €350 → marketRule 1873 → tick = 0.10 (€200–500 range). Order prices must be rounded to 0.10 increments.

**Cache**: Tick sizes are cached per symbol in `_tick_cache` to avoid repeated API calls.

### Commission (IBKR Tiered)

| Region | Rate | Minimum |
|--------|------|---------|
| Europe (XETRA) | 0.05% of trade value | EUR 1.25 |
| US (SMART) | $0.005/share | $1.00 |

**Impact on scalping**: With 0.75% target profit on a EUR 1,000 position:
- Expected gain: EUR 7.50
- Round-trip commission: ~EUR 2.50 (EUR 1.25 × 2)
- Net after commission: EUR 5.00

**Live commission tracking**: Every position close in the risk manager calculates and deducts round-trip commission (entry + exit) from PnL before updating balance. Formula per leg: `max(EUR 1.25, 0.05% × trade_value)`. Total commission is reported in session stats, daily summary, and the auto-updated skill.md dashboard.

### Market Data

IBKR provides market data with automatic fallback:

| Type | Latency | Cost | Use Case |
|------|---------|------|----------|
| **Type 1** (Live) | Real-time | Subscription required | Live trading with paid data |
| **Type 3** (Delayed) | 15–20 min delay | Free | Paper trading, position monitoring |

**Configuration (v2.6.0)**: The system calls `ib.reqMarketDataType(3)` at connection time. This returns real-time data for subscribed instruments and delayed data for unsubscribed ones. On paper accounts without market data subscriptions, this eliminates **Error 354** ("Not subscribed to market data"). Instead, IBKR returns **Warning 10167** ("Displaying delayed market data") which is expected and non-blocking.

**Quote fallback chain**: `_get_ibkr_quote()` (IBKR snapshot) → `_get_finnhub_quote()` (REST) → `None` (skip monitoring cycle). If IBKR returns no price data (e.g., Error 322 on certain contracts), the system raises `ValueError` which `get_quote()` catches and falls to Finnhub.

**Known IBKR quirks**:
- **Error 322** ("Index 0 out of bounds"): Intermittent IBKR server-side bug for certain contracts (observed on MBG.DE). Non-fatal — system falls back to Finnhub.
- **Error 300** ("Can't find EId with tickerId"): Happens when `cancelMktData()` is called on an already-expired snapshot. Suppressed via try/except since v2.6.0.

### Connection Resilience

The system monitors the IBKR connection with a heartbeat every 30 seconds during the trading loop.

| Scenario | Behavior |
|----------|----------|
| Connection alive | Continue normal trading |
| Connection dropped | Auto-reconnect with exponential backoff (10s, 20s, 30s...) |
| Reconnect succeeds | Re-share connection with data collector + filters, clear contract cache |
| 5 failed reconnects | Fall back to local paper mode, log incident to skill.md |

**Important**: During reconnection, open positions with bracket orders remain protected on the IBKR server side (stop-loss orders persist even if client disconnects).

### Known IBKR Errors & Handling (v2.6.0)

| Error | Meaning | System Behavior |
|-------|---------|-----------------|
| **110** | "Price does not conform to min price variation" | Eliminated — dynamic tick sizes via `reqMarketRule()` |
| **201** | "Invalid order: no price" | Eliminated — `MarketOrder` for all close operations |
| **300** | "Can't find EId with tickerId" | Suppressed — expired snapshot ticker, harmless |
| **322** | "Index 0 out of bounds" | IBKR server bug for certain contracts (e.g., MBG.DE). Falls back to Finnhub quote |
| **354** | "Not subscribed to market data" | Eliminated — `reqMarketDataType(3)` enables delayed data fallback |
| **10167** | "Displaying delayed market data" | Expected on paper accounts — confirms delayed data is working |

### Fallback

If IB Gateway/TWS is not running, the system falls back to **local paper trading** mode:
- Orders are simulated locally with estimated IBKR commission and 0–0.02% adverse slippage
- Market data falls back to Finnhub → Yahoo Finance
- All risk management still active
- Monitor loop runs at 10s intervals (vs 30s for live)

---

## Institutional Analytics Suite

The system includes Goldman Sachs / JP Morgan-grade analytics that run at end-of-day and feed into the auto-updated skill.md dashboard.

### Risk-Adjusted Return Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **Sortino Ratio** | (Ann. Return - Rf) / Downside Dev | Like Sharpe but only penalizes downside; >2.0 = excellent |
| **Calmar Ratio** | Ann. Return / Max Drawdown | Hedge fund allocator favorite; >3.0 = strong |

### Risk Metrics

| Metric | What It Measures | Threshold |
|--------|-----------------|-----------|
| **VaR (95%)** | Max loss at 95% confidence | Should be < 1% of account |
| **CVaR (Expected Shortfall)** | Average loss in worst 5% of trades | Basel III standard; captures tail risk |
| **Monte Carlo (1000 sims)** | Forward equity projection at P5/P25/P50/P75/P95 | Prob of profit should be >60% |

### Trade Quality Metrics

| Metric | Purpose | Actionable Insight |
|--------|---------|-------------------|
| **MAE (Max Adverse Excursion)** | How far trades go against you | If winners' MAE < avg MAE → tighten stops |
| **MFE (Max Favorable Excursion)** | How far trades go in your favor | If losers have high MFE → tighten TP |
| **Signal Decay** | PnL by hold duration | Identifies optimal exit timing |
| **Slippage** | Signal price vs fill price | Monitors execution quality |
| **Win/Loss Streaks** | Consecutive streak analysis | Long loss streaks signal regime change |

### Performance Attribution

Breaks down PnL by multiple dimensions:

| Dimension | Example Insight |
|-----------|----------------|
| **By Regime** | "Mean-reversion earns 80% of profits in ranging markets" |
| **By Strategy** | "Momentum strategy losing money — consider disabling" |
| **By Hour** | "Best hours: 10:00–11:00 CET. Worst: 15:00–16:00" |
| **By Day of Week** | "Mondays unprofitable — consider reduced sizing" |
| **By Symbol** | "SAP.DE most profitable; TSLA losing money" |
| **By Exit Type** | "50% are stop-losses — entry timing may be poor" |

### Statistical Significance

**t-test**: One-sample t-test determines whether mean trade return is significantly > 0.

| p-value | Confidence | Meaning |
|---------|-----------|---------|
| < 0.05 | 95% | Strategy returns are NOT due to luck |
| < 0.10 | 90% | Marginal significance — more data needed |
| > 0.10 | Not significant | Cannot rule out random chance |

### Cross-Asset Correlation

Correlation matrix between symbols identifies concentration risk. Pairs with |correlation| > 0.7 are flagged as potential risk concentration.

### Data Requirements

| Analysis | Minimum Trades |
|----------|---------------|
| Basic stats (Sortino, Calmar) | 2 |
| VaR/CVaR | 10 |
| t-test | 10 |
| MAE/MFE | 5 (with tracking data) |
| Monte Carlo | 20 |
| Signal Decay | 10 |
| Correlation Matrix | 5 per symbol, 2+ symbols |

### Integration Points

- **End-of-day report**: Full analytics printed to log via `Analytics.format_report()`
- **Skill updater**: Institutional metrics table added to auto-updated dashboard every 60 min
- **Position tracking**: `mae_pct`, `mfe_pct`, `slippage` fields on every Position
- **Executor**: Slippage computed at fill time (signal price vs fill price)

---

## Data Sources

### Hierarchy (Live Trading)

| Priority | Source | Type | Latency |
|----------|--------|------|---------|
| 1 | IBKR | Streaming via ib_async | ~250ms |
| 2 | Finnhub | WebSocket + REST | ~500ms |
| 3 | Yahoo Finance | REST (polling) | ~2s |

### Historical (Backtesting)

| Source | Coverage | Free? |
|--------|----------|-------|
| IBKR | Full history, all intervals | Yes (with account) |
| Yahoo Finance | Global, 1min for 7 days, 5min for 60 days | Yes |
| Twelve Data | EU + US, 100+ indicators | Yes (8 req/min) |

### API Keys

Set in `.env` file (optional — only needed for fallback sources):
```
FINNHUB_API_KEY=xxx        # Get free at finnhub.io
TWELVE_DATA_API_KEY=xxx    # Get free at twelvedata.com
```

---

## Commands Reference

```bash
# Setup
cp .env.example .env           # Configure connection settings
pip install -r requirements.txt # Install dependencies
# Start IB Gateway and log in (paper or live)

# Trading
python main.py                  # Paper trading session
python main.py --live           # Live trading (requires IBKR live connection)
python main.py --scan           # One-shot signal scan

# Analysis
python main.py --backtest                    # Backtest all watchlist
python main.py --backtest --symbol SAP.DE    # Backtest specific symbol
python main.py --backtest --days 90          # Backtest 90 days

# Learning
python main.py --learn                       # Force optimization
python main.py --learn --symbol SAP.DE       # Optimize for specific symbol

# Position Management (v2.6.0)
python cleanup_positions.py                  # Dry run — show all positions and open orders
python cleanup_positions.py --exec           # Flatten all positions with MARKET orders + cancel open orders
```

**Note**: The cleanup script uses `clientId + 10` to avoid conflicts with the main trading session. Always run it when the main session is stopped.

---

## Mandatory Self-Improvement Checklist

**After every trading day**, the system (or operator) must complete:

- [ ] Review `logs/trading_YYYYMMDD.log` for errors or anomalies
- [ ] Check daily PnL — is it positive?
- [ ] Check win rate — is it above 55%?
- [ ] Check profit factor — is it above 1.3?
- [ ] Check max drawdown — is it below 5%?
- [ ] If ANY check fails → run `python main.py --learn`
- [ ] Review learning report recommendation
- [ ] If recommended: apply new parameters
- [ ] If 3 consecutive losing days → STOP live trading, run extended backtest
- [ ] Weekly: review all `reports/` files, identify patterns
- [ ] Monthly: re-evaluate watchlist (add/remove symbols based on liquidity)

**After every learning cycle:**

- [ ] Compare old vs new parameter performance
- [ ] Verify new parameters on out-of-sample data
- [ ] Paper trade new parameters for minimum 1 day before going live
- [ ] Document what changed and why in the learning log

---

## Risk Warnings

1. **No guaranteed profits**: Algorithmic trading involves substantial risk of loss
2. **Past performance**: Backtest results do not guarantee future results
3. **Slippage**: Real fills may differ significantly from backtest assumptions
4. **Market conditions change**: A strategy that works today may fail tomorrow
5. **Self-learning is not magic**: Optimization can overfit to recent data
6. **Commission impact**: EUR 1.25 min per trade × 40 trades/day = EUR 50 in fees alone
7. **Paper trade first**: Always validate with paper trading before going live
8. **IBKR connection**: Auto-reconnect handles brief drops; bracket order stop-losses persist server-side even if client disconnects

---

## Improvement Log

| Date | Change | Reason | Result |
|------|--------|--------|--------|
| 2026-03-05 | System created | Initial build | v1.0.0 |
| 2026-03-05 | Migrated to Interactive Brokers | Official API, bracket orders, lower latency | v2.0.0 |
| 2026-03-05 | Added market filters | Economic calendar, VIX/sentiment, spread checks | v2.1.0 |
| 2026-03-05 | Code audit + skill updater | 13 bugs fixed, auto skill.md updates every 60min | v2.2.0 |
| 2026-03-05 | Safety hardening | Trading hours, duplicate symbol block, IBKR reconnect, live commission | v2.3.0 |
| 2026-03-05 | Institutional analytics | Sortino, Calmar, VaR/CVaR, MAE/MFE, Monte Carlo, t-test, attribution, correlation, signal decay, slippage | v2.4.0 |
| 2026-03-06 | Pre-live deep audit — 20+ fixes | Bracket safety (C1), double commission fix (C2), state persistence (C3), IBKR reconciliation (C4), timezone fix (C5), paper slippage (H2), stale order detection (H1), market-hours gate (M5), MAE/MFE finalization (M4), RSI zone overlap (M3), Monte Carlo fix (M2), MACD column safety (M6), Finnhub rate limit (L6), NaN handling (L4), t-test accuracy (L5) | v2.5.0 |
| 2026-03-06 | IBKR paper trading fixes — tick sizes, delayed data, cleanup | Dynamic tick sizes via `reqMarketRule()` (Error 110 eliminated), delayed market data via `reqMarketDataType(3)` (Error 354 eliminated), SMART routing (Error 201 eliminated), contract qualification, cleanup utility, Error 300 suppression | v2.6.0 |
| 2026-03-06 | Trading Knowledge Collector | Persistent learning engine: records every signal/trade with market context, mines patterns (win rate by strategy/symbol/VIX/hour/confidence), auto-generates knowledge_base.md with actionable insights | v2.7.0 |
| | | | |

*Update this table after every significant parameter change or system modification.*

---

## Architecture Decision Records

### ADR-001: Mean-Reversion as Primary Strategy
- **Context**: Need a strategy that wins frequently with small gains
- **Decision**: Mean-reversion (Bollinger + RSI + VWAP) for ranging markets
- **Rationale**: 70% of market time is ranging; mean-reversion has higher win rate on short timeframes
- **Fallback**: Momentum strategy activates when ADX > 25

### ADR-002: 0.5% Stop-Loss
- **Context**: User requirement for tight risk management
- **Decision**: Fixed 0.5% stop-loss with ATR-based adjustment
- **Rationale**: Limits loss per trade; ATR adjustment prevents premature stops in volatile conditions
- **Trade-off**: May get stopped out by noise on very short timeframes

### ADR-003: Paper Trading Default
- **Context**: Real money at risk in live trading
- **Decision**: System defaults to paper trading mode
- **Rationale**: Validate strategy before risking capital; IBKR provides full paper trading environment

### ADR-004: Self-Learning with Walk-Forward
- **Context**: Markets change; static parameters degrade over time
- **Decision**: Walk-forward optimization every 50 trades or on performance degradation
- **Rationale**: Walk-forward is the gold standard for avoiding overfitting while adapting to regime changes

### ADR-005: Interactive Brokers over Trade Republic
- **Context**: Trade Republic has no official API; unofficial access risks account suspension
- **Decision**: Migrate to Interactive Brokers (official TWS API via ib_async)
- **Rationale**: Free API, EUR 0 minimum, bracket orders (atomic SL+TP), ~75ms latency, full paper trading, XETRA + US routing
- **Trade-off**: Slightly higher commission (EUR 1.25 min vs EUR 1.00 flat on TR)

### ADR-006: Pre-Trade Market Filters
- **Context**: Technical indicators operate in isolation; they can't detect external shocks (ECB decisions, earnings, market panic)
- **Decision**: Three-layer filter before every trade: economic calendar, market sentiment (VIX/indices), bid-ask spread
- **Rationale**: One bad trade during an ECB announcement can wipe a week of scalping gains. VIX check prevents going long in a falling market. Spread check prevents profit erosion.
- **Data sources**: Finnhub (calendar, free) + Yahoo Finance (VIX/DAX/S&P, free) + IBKR (spread, already connected)
- **Trade-off**: May miss some valid trades during event windows, but capital preservation outweighs opportunity cost

### ADR-008: Safety Hardening (v2.3.0)
- **Context**: Four gaps identified during code audit that affect real-money trading
- **Decision**: Implement trading hours enforcement, duplicate symbol protection, IBKR auto-reconnect, and live commission tracking
- **Trading hours**: Uses `zoneinfo` for correct CET/ET handling including DST; blocks signals outside market hours with 15-min buffer
- **Duplicate symbols**: `risk_manager.has_open_position()` prevents 2 positions on the same stock; eliminates double-exposure risk
- **Reconnection**: Exponential backoff (10s–50s), max 5 attempts, falls back to paper mode; bracket orders remain protected server-side
- **Commission**: Every `_close_position()` computes IBKR tiered commission and deducts from PnL before balance update; total tracked in stats

### ADR-009: Institutional Analytics (v2.4.0)
- **Context**: Basic metrics (win rate, profit factor, max drawdown) are insufficient for institutional-grade risk management and strategy evaluation
- **Decision**: Built `analytics.py` with Sortino, Calmar, VaR/CVaR, MAE/MFE, Monte Carlo (1000 sims), t-test, multi-dimensional attribution, correlation matrix, signal decay, and slippage tracking
- **Rationale**: Goldman Sachs and JP Morgan quant desks use these metrics to distinguish skill from luck, optimize position management, and identify regime changes early
- **MAE/MFE**: Tracked via price watermarks on every Position during `update_position()`; reveals if stops are too wide or TP is leaving money on table
- **Monte Carlo**: Resamples historical trades to project 252-day equity paths; provides probability of profit and probability of ruin
- **t-test**: Determines if mean return is statistically significant (p < 0.05); prevents trading a strategy that may just be luck
- **Attribution**: Breaks PnL by regime, strategy, hour, day, symbol, and exit type; identifies profitable vs unprofitable dimensions
- **Trade-off**: Monte Carlo adds ~0.5s computation at end-of-day; negligible impact on live trading loop

### ADR-010: Pre-Live Deep Audit (v2.5.0)
- **Context**: Final audit before going live with real capital; 20+ issues found across 10 files
- **Critical fixes**:
  - **Bracket order double-execution (C1)**: IBKR bracket orders create server-side SL/TP; client was ALSO checking SL/TP → double fill → naked position. Fixed: `has_bracket` flag skips client SL/TP; `check_bracket_fills()` polls server fills; `cancel_bracket_legs()` cancels remaining legs on trailing/time exit
  - **Double commission in backtester (C2)**: Commission deducted 2× (once in backtester, once in `_close_position()`). All backtest results were artificially pessimistic. Fixed: backtester delegates all commission to `_close_position()`
  - **State persistence (C3)**: All position/risk data was in-memory only; crash = total loss of tracking + orphaned IBKR positions. Fixed: atomic JSON save on every position change, restore on startup, clear at EOD
  - **IBKR reconciliation (C4)**: No comparison between internal tracking and actual IBKR portfolio. Fixed: every 10 min, detect orphans/ghosts/size mismatches
  - **Timezone bug (C5)**: Finnhub economic calendar returns UTC times; system compared with `datetime.now()` (local CET). Events were off by 1–2 hours. Fixed: `datetime.now(ZoneInfo("UTC"))`
- **High fixes**: Paper slippage (0–0.02%), stale order detection (>60s unfilled), faster paper loop (10s vs 30s)
- **Medium fixes**: RSI overlap (45–70 / 30–55), MAE/MFE finalized at exit, market-hours gate before scanning, MACD column pattern-match, Monte Carlo non-deterministic seed
- **Low fixes**: Finnhub rate limit, NaN handling, t-test scipy upgrade, dead code removal, proper encapsulation via properties
- **Trade-off**: State persistence adds ~1ms disk I/O per position change; reconciliation adds ~100ms every 10 min. Negligible impact.

### ADR-011: IBKR Paper Trading Hardening (v2.6.0)
- **Context**: Paper trading runs 5–6 revealed three blocking errors: Error 110 (tick size violation), Error 354 (market data not subscribed), Error 201 (price=0 on close orders)
- **Root causes and fixes**:
  - **Error 110**: IBKR enforces price-dependent tick sizes (e.g., XETRA stocks above €200 require 0.10 tick, not 0.01). The hardcoded `min_tick = 0.01` was wrong for most instruments. Fixed by using `reqContractDetails().marketRuleIds` → `reqMarketRule(id)` → iterate `PriceIncrement` objects to find exact tick for current price level.
  - **Error 354**: Paper account has no EU/US market data subscriptions. `reqMktData(snapshot=True)` failed during position monitoring. Fixed by calling `ib.reqMarketDataType(3)` at connection time, which tells IBKR to return delayed (15-min) data instead of refusing.
  - **Error 201**: Close orders used `price=0` (from `current_price` being 0 due to Error 354). Fixed by using `MarketOrder` for all close operations.
- **Verification**: Run 6 (3/3 brackets filled, zero Error 110), Run 7 (2/2 brackets filled, delayed data monitoring confirmed via Warning 10167)
- **Additional improvements**: SMART routing for all contracts (not IBIS), contract qualification before market data requests, `cleanup_positions.py` utility for flattening orphan positions, Error 300 suppression on expired snapshot tickers
- **Watchlist note**: `EUNL.DE` (iShares MSCI World) removed — IBKR cannot resolve this symbol. `EXS1.DE` and `SXR8.DE` work via SMART routing to IBIS2.

### ADR-007: Automatic Skill Document Updates
- **Context**: The system needs a persistent record of what works and what doesn't across sessions
- **Decision**: `skill_updater.py` auto-updates skill.md every 60 minutes with performance metrics, parameter changes, and improvement recommendations
- **Rationale**: Manual documentation is unreliable; automated logging ensures every session's learnings are captured
- **Implementation**: HTML comment markers in skill.md delineate the auto-updated section; content is regenerated each cycle

### ADR-012: Trading Knowledge Collector (v2.7.0)
- **Context**: The system generates 50–200 signals per day across 19 symbols, but only a fraction become trades. The reasons why signals are filtered, skipped, or executed — and the outcomes of executed trades — are lost at session end. Over weeks/months, this accumulated intelligence would reveal which strategies, symbols, confidence levels, VIX ranges, and timing patterns actually produce alpha.
- **Decision**: Built `trading_knowledge.py` — a persistent learning engine that records every signal and trade with full market context, then mines patterns across 12+ dimensions.
- **Architecture**:
  - **Storage**: Append-only JSONL files per day (`knowledge/signals_YYYYMMDD.jsonl`, `knowledge/trades_YYYYMMDD.jsonl`) + aggregated `patterns.json`
  - **Signal recording**: Every signal gets logged with action (executed/filtered/skipped), filter reason, strategy, regime, confidence, symbol, direction, and market snapshot (VIX, VIX change, DAX%, S&P%)
  - **Trade recording**: Open and close events with entry/exit prices, P&L, duration, slippage, MAE/MFE, exit reason (sl/tp/trail/time/eod/manual), and market context at both entry and exit
  - **Scan cycle metadata**: Symbols scanned, signals found, signals executed, filters applied, cycle duration
  - **Pattern analysis**: Runs at end-of-day with configurable lookback (default 30 days). Dimensions: by_strategy, by_symbol, by_direction, by_confidence, by_regime, by_exit_reason, by_hour, by_day, by_vix_range, duration_stats, slippage_stats, streaks
  - **Auto-insights**: Rule-based insight generator produces actionable recommendations (e.g., "Strategy X has <40% win rate — consider disabling", "VIX 20-25 range shows best performance", "Hour 10 has highest win rate")
  - **Knowledge base**: Auto-generates `knowledge/knowledge_base.md` — a human-readable document with tables, statistics, and insights from all accumulated data
- **Integration points in `main.py`**:
  - `execute_signals()`: Records every signal (filtered, skipped, executed) with market context; records trade opens after fill
  - `monitor_positions()`: Records trade closes from both bracket fills (IBKR server-side) and client-side exits (trailing/time)
  - `run_session()`: Records scan cycle metadata after every scan
  - `_end_of_day()`: Triggers `analyze_patterns(lookback_days=30)`, logs session summary and top insights
- **Rationale**: Walk-forward optimization (ADR-004) tunes parameters mechanically, but can't answer "should we trade TSLA on Mondays?" or "does high VIX help or hurt our mean-reversion strategy?". The knowledge collector provides the qualitative intelligence layer that complements quantitative optimization.
- **Trade-off**: Adds ~2ms per signal record, ~5ms per trade record, ~500ms for end-of-day analysis. JSONL files grow ~50KB/day; patterns.json stays under 100KB. Negligible performance and storage impact.

---

<!-- AUTO-UPDATED SECTION START -->

## Live Performance Dashboard

> **Last Updated**: 2026-03-12 15:45:59
> **Session Started**: 2026-03-12 09:23:53
> **Update Count**: 7
> **Mode**: PAPER

### Session Performance

| Metric | Value |
|--------|-------|
| Total Trades | 5 |
| Win Rate | 20.0% |
| Total PnL | -30.24 EUR |
| Avg Win | 3.95 EUR |
| Avg Loss | 8.55 EUR |
| Profit Factor | 0.12 |
| Max Drawdown | 0.32% |
| Balance | 9969.76 EUR |
| Return | -0.30% |
| Commission | 12.50 EUR |
| Avg Bars Held | 35.0 |

**Assessment**: 
CRITICAL — win rate below 50%. Immediate parameter review needed.

### IBKR Execution

| Metric | Value |
|--------|-------|
| Bracket Orders | 3/3 filled (100%) |
| Tick Sizes | BAS.DE: 0.01, IFX.DE: 0.01 |

### Active Strategy Parameters

| Parameter | Value |
|-----------|-------|
| Primary Strategy | mean_reversion |
| Adaptive Regime | ON |
| EMA Fast/Slow/Trend | 9/21/50 |
| RSI Period | 7 |
| RSI Oversold/Overbought | 25.0/75.0 |
| BB Period/Std | 20/2.0 |
| Volume Multiplier | 1.2x |
| Min Confluence | 3 |
| Stop-Loss | 0.50% |
| Take-Profit | 1.00% |
| Risk per Trade | 0.50% |
| Max Daily Loss | 1.50% |

### Risk Status

- Daily PnL: -30.24 EUR
- Daily Trades: 5/20
- Consecutive Losses: 3/3
- Open Positions: 0/3
- Balance: 9969.76 EUR

### Recent Closed Trades

| Symbol | Side | PnL | Exit Reason | Bars |
|--------|------|-----|-------------|------|
| BMW.DE | LONG | -2.68 | closed_tp | 1 |
| BAS.DE | LONG | +3.95 | closed_tp | 20 |
| IFX.DE | LONG | -12.34 | closed_sl | 70 |
| BAS.DE | SHORT | -16.69 | closed_sl | 37 |
| MBG.DE | SHORT | -2.48 | closed_manual | 47 |

### Self-Improvement Notes

- Insufficient trade data for analysis. Continue collecting samples.

<!-- AUTO-UPDATED SECTION END -->
