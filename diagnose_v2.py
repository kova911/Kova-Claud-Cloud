"""
Quick diagnostic: what is V2 seeing right now?
Runs one scan cycle and shows composite scores, rejection reasons.
Uses yfinance directly to fetch data.
"""
import sys
import logging
import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger()

from config import config as cfg
from strategy_v2 import StrategyV2, SignalType, MarketRegime, SIGNAL_WEIGHTS
from indicators import Indicators

strategy = StrategyV2(cfg.strategy, cfg.risk)

print("=" * 70)
print("V2 STRATEGY DIAGNOSTIC")
print("=" * 70)

eu_symbols = [s for s in cfg.watchlist.symbols if s.endswith(".DE")]
active = [s for s in eu_symbols if not strategy.should_skip_stock(s)]
skipped = [s for s in eu_symbols if strategy.should_skip_stock(s)]

print(f"Active: {', '.join(active)}")
print(f"Skipped: {', '.join(skipped)}")
print()

# Fetch 5m data via yfinance
data = {}
for symbol in active:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="10d", interval="5m")
        if df is not None and len(df) > 100:
            # Normalize columns for strategy
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={"stock splits": "stock_splits"})
            # Add indicators (same as main.py does)
            df = Indicators.add_all(df, cfg.strategy)
            data[symbol] = df
            print(f"  {symbol}: {len(df)} bars, last: {df['close'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"  {symbol}: ERROR - {e}")

print(f"\nAnalyzing {len(data)} symbols...\n")
print("-" * 70)

for symbol, df in data.items():
    try:
        df_ind = df
        if df_ind is None or len(df_ind) < 50:
            print(f"{symbol}: insufficient data after indicators")
            continue

        row = df_ind.iloc[-1]
        price = row["close"]
        atr = row.get("atr", 0)
        if pd.isna(atr) or atr <= 0:
            print(f"{symbol}: ATR is NaN/zero, skipping")
            continue

        zscore = row.get("zscore", 0)
        rsi = row.get("rsi", 50)
        adx = row.get("adx", 0)
        vol_ratio = row.get("volume_ratio", 1.0)

        regime = strategy.detect_regime(df_ind)

        # Call generate_signal to get the actual result, but also capture components
        # by calling each scoring method individually
        latest = df_ind.iloc[-1]
        prev = df_ind.iloc[-2] if len(df_ind) > 1 else latest

        # Check early-exit conditions
        early_exit = None
        if regime == MarketRegime.VOLATILE:
            early_exit = "VOLATILE regime — standing aside"
        atr_pct = atr / price if price > 0 else 0
        if atr_pct > strategy.max_atr_pct:
            early_exit = f"ATR% {atr_pct:.4f} > max {strategy.max_atr_pct}"
        if atr_pct < strategy.min_atr_pct:
            early_exit = f"ATR% {atr_pct:.4f} < min {strategy.min_atr_pct}"

        if early_exit:
            print(f"{symbol} @ {price:.2f}  [✗ EARLY EXIT: {early_exit}]")
            print(f"  Regime: {regime.value}  ADX={adx:.1f}  RSI={rsi:.1f}  ATR={atr:.3f}  ATR%={atr_pct:.4f}")
            print()
            continue

        # Score each component
        components = {}
        zs, zr = strategy._score_zscore(df_ind, latest, price)
        components["zscore"] = zs
        vs, vr = strategy._score_volume(df_ind, latest)
        components["volume"] = vs
        ms, mr = strategy._score_momentum(df_ind, latest, prev, regime)
        components["momentum"] = ms
        vcs, vcr = strategy._score_volatility_context(df_ind, latest, atr)
        components["volatility"] = vcs
        ts, tr_ = strategy._score_trend(df_ind, latest, regime)
        components["trend"] = ts
        mics, micr = strategy._score_microstructure(df_ind, latest, price)
        components["microstructure"] = mics

        composite = sum(
            components.get(k, 0) * w
            for k, w in SIGNAL_WEIGHTS.items()
        )

        suit = strategy.stock_suitability.get(symbol, 0.0)
        composite_adj = composite + suit * np.sign(composite) if (suit != 0 and composite != 0) else composite
        abs_score = abs(composite_adj)
        direction = "LONG" if composite_adj > 0 else "SHORT"

        # ATR-based stops
        atr_stop = atr * strategy.atr_stop_multiplier
        atr_tp = atr * strategy.atr_tp_multiplier

        # € edge estimate
        stop_distance = atr_stop
        risk_amount = cfg.risk.initial_balance * cfg.risk.risk_per_trade
        est_size = int(risk_amount / stop_distance) if stop_distance > 0 else 0
        expected_tp_gain = atr_tp * est_size

        # Cost check
        expected_gross_edge = atr_tp / price if price > 0 else 0
        expected_net_edge = expected_gross_edge - strategy.round_trip_cost_pct

        # Rejection analysis
        rejections = []
        if abs_score < strategy.min_composite_score:
            rejections.append(f"score {abs_score:.3f} < {strategy.min_composite_score}")
        if regime == MarketRegime.TRENDING_UP and direction == "SHORT" and abs_score < 0.75:
            rejections.append(f"SHORT in uptrend")
        if regime == MarketRegime.TRENDING_DOWN and direction == "LONG" and abs_score < 0.75:
            rejections.append(f"LONG in downtrend")
        if expected_net_edge < strategy.round_trip_cost_pct:
            rejections.append(f"net edge {expected_net_edge:.4%} < cost {strategy.round_trip_cost_pct:.4%}")
        if expected_tp_gain < strategy.min_expected_gain_eur:
            rejections.append(f"€ edge €{expected_tp_gain:.2f} < min €{strategy.min_expected_gain_eur:.2f}")

        status = "✓ PASS" if not rejections else "✗ REJECT"

        print(f"{symbol} @ {price:.2f}  {direction}  [{status}]")
        print(f"  Score: {composite_adj:+.4f} (raw={composite:+.4f}, suit={suit:+.2f})  threshold={strategy.min_composite_score}")
        print(f"  Components: " + ", ".join(f"{k}={v:+.3f}" for k, v in sorted(components.items())))
        print(f"  Regime: {regime.value}  ADX={adx:.1f}  RSI={rsi:.1f}  Z={zscore:.2f}  VolRatio={vol_ratio:.2f}")
        print(f"  ATR={atr:.3f}  SL_dist={atr_stop:.3f}  TP_dist={atr_tp:.3f}  Size={est_size}  TP_gain=€{expected_tp_gain:.2f}")
        if rejections:
            for r in rejections:
                print(f"  ✗ {r}")
        print()

    except Exception as e:
        import traceback
        print(f"{symbol}: ERROR — {e}")
        traceback.print_exc()
        print()

print("=" * 70)
