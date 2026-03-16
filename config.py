"""
Trading System Configuration
All settings, API keys, and strategy parameters in one place.
"""

import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional

load_dotenv()


# ─── API Keys ────────────────────────────────────────────────
@dataclass
class APIKeys:
    finnhub: str = os.getenv("FINNHUB_API_KEY", "")
    twelve_data: str = os.getenv("TWELVE_DATA_API_KEY", "")
    alpha_vantage: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    polygon: str = os.getenv("POLYGON_API_KEY", "")


# ─── Interactive Brokers ─────────────────────────────────────
@dataclass
class IBKRConfig:
    # Connection
    host: str = os.getenv("IBKR_HOST", "127.0.0.1")
    live_port: int = int(os.getenv("IBKR_LIVE_PORT", "4001"))    # IB Gateway live
    paper_port: int = int(os.getenv("IBKR_PAPER_PORT", "4002"))  # IB Gateway paper
    client_id: int = int(os.getenv("IBKR_CLIENT_ID", "1"))

    # TWS ports (alternative to IB Gateway)
    tws_live_port: int = 7496
    tws_paper_port: int = 7497

    # Mode
    paper_trading: bool = True      # Default to paper trading
    use_gateway: bool = True        # True = IB Gateway, False = TWS

    # Account (auto-detected on connection, but can override)
    account_id: str = os.getenv("IBKR_ACCOUNT", "")

    # Market data
    market_data_type: int = 1       # 1=Live, 2=Frozen, 3=Delayed, 4=Delayed-Frozen


# ─── Strategy Parameters ─────────────────────────────────────
@dataclass
class StrategyConfig:
    # Timeframes
    signal_timeframe: str = "5m"       # Signal generation candles
    execution_timeframe: str = "1m"    # Entry/exit precision
    trend_timeframe: str = "15m"       # Trend filter

    # Moving Averages
    ema_fast: int = 9
    ema_slow: int = 21
    ema_trend: int = 50

    # RSI
    rsi_period: int = 7
    rsi_oversold: float = 25.0
    rsi_overbought: float = 75.0

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # VWAP
    use_vwap: bool = True

    # Volume confirmation
    volume_multiplier: float = 1.2     # Volume must be 1.2x average

    # Signal confluence: minimum number of indicators that must agree
    min_signal_confluence: int = 3

    # Strategy mode
    primary_strategy: str = "mean_reversion"  # "mean_reversion" or "momentum"
    adaptive_regime: bool = True               # Switch strategy based on market regime

    # ADX for regime detection
    adx_period: int = 14
    adx_trending_threshold: float = 25.0       # ADX > 25 = trending
    adx_ranging_threshold: float = 20.0        # ADX < 20 = ranging


# ─── Risk Management ─────────────────────────────────────────
@dataclass
class RiskConfig:
    # Account
    initial_balance: float = 10000.0   # EUR

    # Per-trade risk
    stop_loss_pct: float = 0.005       # 0.5% stop-loss
    take_profit_pct: float = 0.01      # 1.0% take-profit (1:2 R:R)
    risk_per_trade: float = 0.005      # Risk 0.5% of account per trade

    # Trailing stop
    trailing_stop_activation: float = 0.005  # Activate trailing after 0.5% profit
    trailing_stop_distance: float = 0.003    # Trail by 0.3%

    # Daily limits
    max_daily_loss_pct: float = 0.015  # Stop trading after 1.5% daily loss
    max_consecutive_losses: int = 3    # Stop after 3 consecutive losses
    max_trades_per_day: int = 20       # Maximum trades per session
    cooldown_after_loss: int = 120     # 2 minutes cooldown after a loss (seconds)

    # Time-based exit
    # Reduced from 24 to 12 candles (2h → 1h) based on live data:
    # Winners avg 38 min, time-exit trades avg 448 min with 0% WR and €-16.27 PnL.
    # Holding losers longer just bleeds more money.
    max_hold_candles: int = 12         # Close ALL positions after 12 candles / 1 hour (regardless of PnL)
    close_before_session_end_min: int = 15  # Close all 15 min before market close

    # Position limits
    max_open_positions: int = 3
    max_position_pct: float = 0.20     # Max 20% of account in one position

    # Price sanity guard — reject quotes that deviate too far from entry
    max_price_deviation_pct: float = 0.10  # 10% max deviation from entry price

    # (Spread filter threshold is in FiltersConfig.max_spread_pct)


# ─── Watchlist ────────────────────────────────────────────────
@dataclass
class WatchlistConfig:
    """
    Instruments to trade. Uses IBKR symbol conventions.
    EU stocks: symbol + ".DE" suffix (resolved to IBIS/XETRA exchange)
    US stocks: plain symbol (routed via SMART)
    """
    # High-liquidity stocks and ETFs suitable for scalping
    # Removed: DTE.DE (consistently returns 0/garbage prices with delayed data)
    # Removed: EXS1.DE, SXR8.DE (ETF contract qualification failures on IBKR paper)
    symbols: list = field(default_factory=lambda: [
        # German Blue Chips (XETRA via IBKR IBIS)
        "SAP.DE",      # SAP SE
        "SIE.DE",      # Siemens
        "ALV.DE",      # Allianz
        "BAS.DE",      # BASF
        "MBG.DE",      # Mercedes-Benz
        "BMW.DE",      # BMW
        "AIR.DE",      # Airbus
        "ADS.DE",      # Adidas
        "IFX.DE",      # Infineon

        # US Large Caps (SMART routing)
        "AAPL",        # Apple
        "MSFT",        # Microsoft
        "NVDA",        # NVIDIA
        "AMZN",        # Amazon
        "META",        # Meta
        "GOOGL",       # Alphabet
        "TSLA",        # Tesla
    ])


# ─── Data Collection ─────────────────────────────────────────
@dataclass
class DataConfig:
    # Primary data source for live data
    primary_source: str = "ibkr"            # "ibkr", "finnhub", "twelve_data", "yfinance"
    # Backup source
    backup_source: str = "yfinance"
    # Historical data source for backtesting
    backtest_source: str = "yfinance"       # yfinance is free and deep

    # Data storage
    data_dir: str = "data"
    cache_minutes: int = 1                  # Cache live data for N minutes

    # Historical data range for backtesting
    backtest_days: int = 60                 # Last 60 days of 1min data
    min_bars_required: int = 100            # Minimum bars before generating signals


# ─── Self-Learning ────────────────────────────────────────────
@dataclass
class LearningConfig:
    enabled: bool = True
    # How often to re-evaluate strategy (in trades)
    evaluation_interval: int = 50
    # Minimum trades before learning kicks in
    min_trades_for_learning: int = 100
    # Walk-forward window
    training_window_days: int = 30
    validation_window_days: int = 10
    # Parameters to optimize
    optimizable_params: list = field(default_factory=lambda: [
        "ema_fast", "ema_slow", "rsi_period",
        "rsi_oversold", "rsi_overbought",
        "bb_period", "bb_std",
        "volume_multiplier", "min_signal_confluence",
    ])
    # Performance thresholds to trigger re-optimization
    min_win_rate: float = 0.55             # Below this, trigger learning
    min_profit_factor: float = 1.3          # Below this, trigger learning
    max_drawdown_trigger: float = 0.05      # 5% drawdown triggers review
    # Reports
    reports_dir: str = "reports"
    log_dir: str = "logs"


# ─── Market Filters ──────────────────────────────────────────
@dataclass
class FiltersConfig:
    # Economic Calendar Filter
    economic_calendar_enabled: bool = True
    event_buffer_minutes: int = 30        # Block trading 30 min before/after high-impact events

    # Market Sentiment Filter
    sentiment_enabled: bool = True
    # Tiered VIX response (replaces hard cutoff)
    # Raised vix_normal from 25→28: VIX 20-28 is routine market stress, not alarm level
    vix_normal: float = 28.0              # VIX < 28 = normal, full trading
    vix_elevated: float = 35.0            # VIX 28-35 = elevated, block LONGs, 50% size
    vix_high: float = 40.0               # VIX 35-40 = high, SHORT only, 25% size
    vix_panic_level: float = 40.0         # VIX > 40 = panic, block ALL trading
    vix_spike_pct: float = 0.20           # VIX up 20%+ in a day = block all trading
    index_panic_drop: float = -0.02       # DAX/S&P down 2%+ = block LONG entries

    # Spread Filter
    spread_filter_enabled: bool = True
    max_spread_pct: float = 0.0003        # Block entry if spread > 0.03%


# ─── Trading Schedule ────────────────────────────────────────
@dataclass
class ScheduleConfig:
    # XETRA trading hours (CET/CEST)
    market_open: str = "09:00"         # 9:00 CET
    market_close: str = "17:30"        # 17:30 CET
    # US trading hours (ET) — for US symbols
    us_market_open: str = "09:30"      # 9:30 ET
    us_market_close: str = "16:00"     # 16:00 ET
    timezone: str = "Europe/Berlin"
    # Avoid first and last 15 minutes (high volatility/spread)
    trading_start_offset_min: int = 15
    trading_end_offset_min: int = 15


# ─── ML Signal Filter (Renaissance: XGBoost + RF) ────────────
@dataclass
class MLFilterConfig:
    enabled: bool = True                     # ON — starts in cold-start pass-through mode
    min_trades_for_training: int = 30        # Cold start threshold
    retrain_interval_trades: int = 20        # Retrain every N new completed trades
    win_probability_threshold: float = 0.55  # Only allow signals with P(win) > 55%
    xgboost_weight: float = 0.5             # Ensemble: 50% XGBoost, 50% RF
    rf_weight: float = 0.5
    model_dir: str = "data/ml_models"        # Persisted model storage
    walk_forward_train_pct: float = 0.7      # 70% train, 30% validation
    max_tree_depth: int = 4                  # Shallow trees to prevent overfit
    n_estimators: int = 100                  # Number of trees per model


# ─── HMM Regime Detection (Renaissance: Hidden Markov) ───────
@dataclass
class HMMRegimeConfig:
    enabled: bool = True                     # ON — probabilistic regime detection
    n_states: int = 3                        # low-vol, normal, high-vol
    lookback_bars: int = 200                 # Bars of history for fitting
    min_bars_for_fit: int = 100              # Minimum data before HMM activates
    refit_interval_bars: int = 50            # Refit every N new bars
    # Regime-specific adjustments
    low_vol_zscore_boost: float = 1.2        # Boost MR signals in low-vol
    high_vol_size_reduction: float = 0.5     # Halve position size in high-vol
    model_dir: str = "data/hmm_models"


# ─── Cross-Asset Correlations (Renaissance: decorrelation) ───
@dataclass
class CrossCorrelationConfig:
    enabled: bool = True
    dax_index_symbol: str = "^GDAXI"         # DAX index for yfinance
    rolling_window: int = 20                  # 20-bar rolling correlation
    baseline_window: int = 60                 # 60-bar baseline correlation
    decorrelation_threshold: float = 1.5      # Z-score of correlation deviation
    weight_in_ensemble: float = 0.10          # Weight as 7th signal component
    min_bars: int = 65                        # baseline_window + 5 buffer


# ─── Pairs Trading (Renaissance: statistical arbitrage) ──────
@dataclass
class PairsTradingConfig:
    enabled: bool = True
    pairs: list = field(default_factory=lambda: [
        ("BMW.DE", "MBG.DE"),     # German auto
        ("SAP.DE", "SIE.DE"),     # German tech/industrial
    ])
    lookback_bars: int = 60                   # Spread calculation window
    entry_zscore: float = 2.0                 # Enter when |z| > 2.0
    exit_zscore: float = 0.5                  # Exit when |z| < 0.5
    stop_zscore: float = 3.5                  # Stop-loss at |z| > 3.5
    coint_pvalue_threshold: float = 0.05      # Engle-Granger p-value threshold
    retest_coint_interval_bars: int = 200     # Re-test cointegration periodically
    max_pairs_positions: int = 2              # Max simultaneous pairs trades
    equal_euro_exposure: bool = True           # Market-neutral sizing
    risk_per_pair: float = 0.004              # 0.4% risk per pair (both legs)


# ─── Master Config ────────────────────────────────────────────
@dataclass
class TradingConfig:
    api_keys: APIKeys = field(default_factory=APIKeys)
    ibkr: IBKRConfig = field(default_factory=IBKRConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    watchlist: WatchlistConfig = field(default_factory=WatchlistConfig)
    data: DataConfig = field(default_factory=DataConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    # Renaissance modules
    ml_filter: MLFilterConfig = field(default_factory=MLFilterConfig)
    hmm_regime: HMMRegimeConfig = field(default_factory=HMMRegimeConfig)
    cross_correlation: CrossCorrelationConfig = field(default_factory=CrossCorrelationConfig)
    pairs_trading: PairsTradingConfig = field(default_factory=PairsTradingConfig)

    def validate(self) -> list[str]:
        """Check for missing required configuration."""
        issues = []
        if not self.api_keys.finnhub and self.data.primary_source == "finnhub":
            issues.append("FINNHUB_API_KEY not set — needed for Finnhub data")
        if self.data.primary_source == "ibkr":
            issues.append("NOTE: IBKR data requires active connection to IB Gateway/TWS")
        if self.risk.take_profit_pct <= self.risk.stop_loss_pct * 0.8:
            issues.append("Take-profit too close to stop-loss — expect low profitability")
        if self.filters.economic_calendar_enabled and not self.api_keys.finnhub:
            issues.append("FINNHUB_API_KEY not set — economic calendar filter will be disabled")
        return issues


# Singleton instance
config = TradingConfig()
