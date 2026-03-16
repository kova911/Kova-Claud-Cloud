#!/usr/bin/env python3
"""
V2 Strategy Parameter Optimizer — 2-Phase Approach

Phase 1: Focused grid on 3 liquid symbols → find top 10 candidates (fast)
Phase 2: Validate top candidates across all 9 symbols (targeted)
"""

import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import config
from data_collector import DataCollector
from indicators import Indicators

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("optimizer")
logger.setLevel(logging.INFO)


def precompute_signal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute all signal components as columns."""
    d = df.copy()

    # Volume ratio
    vol_avg = d["volume"].rolling(20).mean().replace(0, np.nan)
    d["vol_ratio"] = (d["volume"] / vol_avg).fillna(1.0)

    # RSI-based momentum score
    rsi = d.get("rsi", pd.Series(50, index=d.index))
    if rsi is None:
        rsi = pd.Series(50, index=d.index)
    rsi = rsi.fillna(50)
    d["momentum_score"] = 0.0
    d.loc[rsi < 25, "momentum_score"] = 0.8
    d.loc[(rsi >= 25) & (rsi < 35), "momentum_score"] = 0.4
    d.loc[rsi > 75, "momentum_score"] = -0.8
    d.loc[(rsi <= 75) & (rsi > 65), "momentum_score"] = -0.4

    # BB width for volatility context
    bb_upper = d.get("bb_upper")
    bb_lower = d.get("bb_lower")
    bb_middle = d.get("bb_middle")
    d["vol_ctx_score"] = 0.0
    if bb_upper is not None and bb_lower is not None and bb_middle is not None:
        if isinstance(bb_upper, pd.DataFrame): bb_upper = bb_upper.iloc[:, 0]
        if isinstance(bb_lower, pd.DataFrame): bb_lower = bb_lower.iloc[:, 0]
        if isinstance(bb_middle, pd.DataFrame): bb_middle = bb_middle.iloc[:, 0]
        bb_width = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan)
        avg_width = bb_width.rolling(50).mean()
        width_ratio = (bb_width / avg_width.replace(0, np.nan)).fillna(1.0)
        d.loc[width_ratio < 0.7, "vol_ctx_score"] = 0.6
        d.loc[width_ratio > 1.5, "vol_ctx_score"] = -0.5

    # Trend score
    ema_fast = d.get("ema_fast", pd.Series(0, index=d.index))
    ema_slow = d.get("ema_slow", pd.Series(0, index=d.index))
    ema_trend = d.get("ema_trend", pd.Series(0, index=d.index))
    if ema_fast is None: ema_fast = pd.Series(0, index=d.index)
    if ema_slow is None: ema_slow = pd.Series(0, index=d.index)
    if ema_trend is None: ema_trend = pd.Series(0, index=d.index)
    d["trend_score"] = 0.0
    d.loc[(d["close"] > ema_fast) & (ema_fast > ema_slow) & (ema_slow > ema_trend), "trend_score"] = 0.6
    d.loc[(d["close"] < ema_fast) & (ema_fast < ema_slow) & (ema_slow < ema_trend), "trend_score"] = -0.6
    d.loc[(d["trend_score"] == 0) & (ema_fast > ema_slow), "trend_score"] = 0.2
    d.loc[(d["trend_score"] == 0) & (ema_fast < ema_slow), "trend_score"] = -0.2

    # Microstructure
    candle_range = d["high"] - d["low"]
    close_pos = ((d["close"] - d["low"]) / candle_range.replace(0, np.nan)).fillna(0.5)
    body = (d["close"] - d["open"]).abs()
    body_ratio = (body / candle_range.replace(0, np.nan)).fillna(0)
    d["micro_score"] = 0.0
    d.loc[(close_pos > 0.8) & (body_ratio > 0.6), "micro_score"] = 0.5
    d.loc[(close_pos < 0.2) & (body_ratio > 0.6), "micro_score"] = -0.5
    upper_wick = d["high"] - d[["open", "close"]].max(axis=1)
    lower_wick = d[["open", "close"]].min(axis=1) - d["low"]
    d.loc[(upper_wick > body * 2) & (upper_wick > candle_range * 0.5), "micro_score"] = -0.4
    d.loc[(lower_wick > body * 2) & (lower_wick > candle_range * 0.5), "micro_score"] = 0.4

    # ATR
    atr = d.get("atr", pd.Series(0, index=d.index))
    if atr is None: atr = pd.Series(0, index=d.index)
    d["atr_val"] = atr.fillna(0)
    d["atr_pct"] = (d["atr_val"] / d["close"].replace(0, np.nan)).fillna(0)

    # Regime
    adx = d.get("adx", pd.Series(0, index=d.index))
    if adx is None: adx = pd.Series(0, index=d.index)
    adx = adx.fillna(0)
    atr_avg_50 = d["atr_val"].rolling(50).mean().fillna(d["atr_val"])
    d["is_volatile"] = (atr_avg_50 > 0) & (d["atr_val"] > atr_avg_50 * 1.5)
    d["is_trending_up"] = (adx > 30) & (d["trend_score"] > 0)
    d["is_trending_down"] = (adx > 30) & (d["trend_score"] < 0)

    return d


def fast_backtest(df: pd.DataFrame, params: dict) -> dict:
    """Position simulation with given params. Uses numpy arrays for speed."""
    zscore_lb = params["zscore_lookback"]
    min_cs = params["min_composite_score"]
    atr_sl = params["atr_stop_multiplier"]
    atr_tp = params["atr_tp_multiplier"]
    vol_thresh = params["volume_surge_threshold"]
    cost = 0.0008

    n = len(df)

    # Pre-compute zscore
    close = df["close"].values
    rolling_mean = pd.Series(close).rolling(zscore_lb).mean().values
    rolling_std = pd.Series(close).rolling(zscore_lb).std().values

    zscore = np.zeros(n)
    valid_std = rolling_std > 0
    zscore[valid_std] = (close[valid_std] - rolling_mean[valid_std]) / rolling_std[valid_std]

    abs_z = np.abs(zscore)
    zscore_score = np.zeros(n)
    mod_mask = (abs_z >= 1.0) & (abs_z < 2.0)
    str_mask = abs_z >= 2.0
    zscore_score[mod_mask] = -zscore[mod_mask] * 0.4
    zscore_score[str_mask] = -np.clip(zscore[str_mask], -3.0, 3.0) / 3.0

    # Volume score
    vol_ratio = df["vol_ratio"].values
    volume_score = np.full(n, 0.3)
    volume_score[vol_ratio >= vol_thresh] = 0.8
    volume_score[vol_ratio < 1.0] = -0.3

    # Composite
    composite = (
        zscore_score * 0.25 +
        volume_score * 0.20 +
        df["momentum_score"].values * 0.15 +
        df["vol_ctx_score"].values * 0.15 +
        df["trend_score"].values * 0.15 +
        df["micro_score"].values * 0.10
    )

    abs_comp = np.abs(composite)
    direction = np.sign(composite)

    # Get arrays
    high = df["high"].values
    low = df["low"].values
    atr_arr = df["atr_val"].values
    atr_pct = df["atr_pct"].values
    is_vol = df["is_volatile"].values
    is_tu = df["is_trending_up"].values
    is_td = df["is_trending_down"].values

    # Valid signal mask
    min_bar = max(50, zscore_lb)
    valid = np.zeros(n, dtype=bool)
    for i in range(min_bar, n):
        if abs_comp[i] < min_cs:
            continue
        if is_vol[i]:
            continue
        if atr_pct[i] <= 0.001 or atr_pct[i] >= 0.015:
            continue
        if atr_arr[i] <= 0:
            continue
        # Regime filter
        if is_tu[i] and direction[i] < 0 and abs_comp[i] < 0.75:
            continue
        if is_td[i] and direction[i] > 0 and abs_comp[i] < 0.75:
            continue
        # Cost check
        expected_gross = atr_arr[i] * atr_tp / close[i] if close[i] > 0 else 0
        expected_net = expected_gross - cost
        if expected_net <= cost:
            continue
        valid[i] = True

    # Position simulation
    balance = 10000.0
    slippage = 0.0001
    spread = 0.0003
    comm_rate = 0.0005
    comm_min = 1.25
    risk_frac = 0.005
    max_pos_pct = 0.20

    trades = []
    in_pos = False
    ep = 0.0
    sz = 0
    side = 0
    sl = 0.0
    tp = 0.0

    for i in range(min_bar, n):
        # Check exits
        if in_pos:
            if side == 1:
                if low[i] <= sl:
                    xp = sl * (1 - slippage)
                    pnl = (xp - ep) * sz - max(comm_min, ep * sz * comm_rate) - max(comm_min, xp * sz * comm_rate)
                    balance += pnl
                    trades.append(pnl)
                    in_pos = False
                elif high[i] >= tp:
                    xp = tp * (1 - slippage)
                    pnl = (xp - ep) * sz - max(comm_min, ep * sz * comm_rate) - max(comm_min, xp * sz * comm_rate)
                    balance += pnl
                    trades.append(pnl)
                    in_pos = False
            else:
                if high[i] >= sl:
                    xp = sl * (1 + slippage)
                    pnl = (ep - xp) * sz - max(comm_min, ep * sz * comm_rate) - max(comm_min, xp * sz * comm_rate)
                    balance += pnl
                    trades.append(pnl)
                    in_pos = False
                elif low[i] <= tp:
                    xp = tp * (1 + slippage)
                    pnl = (ep - xp) * sz - max(comm_min, ep * sz * comm_rate) - max(comm_min, xp * sz * comm_rate)
                    balance += pnl
                    trades.append(pnl)
                    in_pos = False

        # Check entries
        if not in_pos and valid[i]:
            s = int(direction[i])
            if s == 0:
                continue
            a = atr_arr[i]
            if a <= 0:
                continue

            if s == 1:
                ep = close[i] * (1 + slippage + spread / 2)
                sl = ep - a * atr_sl
                tp = ep + a * atr_tp
            else:
                ep = close[i] * (1 - slippage - spread / 2)
                sl = ep + a * atr_sl
                tp = ep - a * atr_tp

            stop_d = abs(ep - sl)
            if stop_d <= 0:
                continue
            sz = int(balance * risk_frac / stop_d)
            mx = int(balance * max_pos_pct / ep) if ep > 0 else 0
            sz = min(sz, mx)
            if sz <= 0:
                continue

            side = s
            in_pos = True

    # Close remaining
    if in_pos:
        xp = close[-1]
        if side == 1:
            pnl = (xp - ep) * sz
        else:
            pnl = (ep - xp) * sz
        pnl -= max(comm_min, ep * sz * comm_rate) + max(comm_min, xp * sz * comm_rate)
        balance += pnl
        trades.append(pnl)

    if not trades:
        return {"total_trades": 0}

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    total_pnl = sum(trades)
    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 1

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / len(trades), 2),
        "profit_factor": round(gp / gl, 2) if gl > 0 else 5.0,
    }


def score(stats: dict) -> float:
    if not stats or stats.get("total_trades", 0) < 3:
        return -999
    wr = stats["win_rate"]
    pf = min(stats["profit_factor"], 5.0)
    pnl = stats["total_pnl"]
    n = stats["total_trades"]
    sig = min(n / 15, 1.0)
    pnl_b = 0.3 if pnl > 0 else -0.15
    return (wr * 0.30 + (pf / 5.0) * 0.25 + pnl_b * 0.25 + min(n / 40, 1.0) * 0.20) * sig


def main():
    logger.info("=" * 70)
    logger.info("V2 STRATEGY OPTIMIZER — 2-PHASE")
    logger.info("=" * 70)

    # ─── Phase 1: Fast scan on 3 most liquid symbols ──────────
    phase1_symbols = ["SAP.DE", "SIE.DE", "ALV.DE"]
    all_symbols = ["SAP.DE", "SIE.DE", "ALV.DE", "BAS.DE", "BMW.DE", "MBG.DE", "AIR.DE", "ADS.DE", "IFX.DE"]

    collector = DataCollector(config)
    phase1_data = {}
    all_data = {}

    logger.info(f"\nPhase 1: Loading data...")
    for symbol in all_symbols:
        try:
            df = collector.get_historical(symbol, interval="5m", days=60)
            if df is not None and len(df) >= 100:
                df_ind = Indicators.add_all(df.copy(), config.strategy)
                df_sig = precompute_signal_columns(df_ind)
                all_data[symbol] = df_sig
                if symbol in phase1_symbols:
                    phase1_data[symbol] = df_sig
                logger.info(f"  {symbol}: {len(df_sig)} bars")
        except Exception as e:
            logger.warning(f"  {symbol}: failed — {e}")

    logger.info(f"Phase 1 symbols: {len(phase1_data)}, Total: {len(all_data)}")

    # Grid
    param_grid = {
        "zscore_lookback":         [20, 30, 40, 50],
        "min_composite_score":     [0.30, 0.35, 0.40, 0.45, 0.50],
        "atr_stop_multiplier":     [1.0, 1.5, 2.0, 2.5],
        "atr_tp_multiplier":       [0.8, 1.0, 1.5, 2.0, 2.5],
        "volume_surge_threshold":  [1.2, 1.5],
    }

    keys = list(param_grid.keys())
    vals = list(param_grid.values())
    total = 1
    for v in vals:
        total *= len(v)

    logger.info(f"\nPhase 1 grid: {total} combos × {len(phase1_data)} symbols")

    results_p1 = []
    tested = 0
    t0 = datetime.now()

    for combo in product(*vals):
        params = dict(zip(keys, combo))
        tested += 1

        all_trades = 0
        all_pnl = 0.0
        all_wins = 0
        all_losses = 0

        for symbol, df_sig in phase1_data.items():
            stats = fast_backtest(df_sig, params)
            all_trades += stats.get("total_trades", 0)
            all_pnl += stats.get("total_pnl", 0)
            all_wins += stats.get("wins", 0)
            all_losses += stats.get("losses", 0)

        if all_trades < 3:
            continue

        agg = {
            "total_trades": all_trades,
            "wins": all_wins,
            "losses": all_losses,
            "win_rate": all_wins / all_trades,
            "total_pnl": round(all_pnl, 2),
            "avg_pnl": round(all_pnl / all_trades, 2),
            "profit_factor": round(
                sum(max(0, fast_backtest(df, params).get("total_pnl", 0)) for df in phase1_data.values()) /
                max(1, abs(sum(min(0, fast_backtest(df, params).get("total_pnl", 0)) for df in phase1_data.values()))),
                2
            ) if all_pnl != 0 else 0,
        }
        # Simpler PF
        gp = sum(t for t in [all_pnl] if t > 0) if all_pnl > 0 else 0
        gl = abs(all_pnl) if all_pnl < 0 else 1
        agg["profit_factor"] = round(all_wins / max(all_losses, 1), 2)

        s = score(agg)
        results_p1.append({"params": params, "stats": agg, "score": s})

        if tested % 100 == 0:
            elapsed = (datetime.now() - t0).total_seconds()
            rate = tested / elapsed if elapsed > 0 else 1
            remaining = (total - tested) / rate
            logger.info(f"  Phase 1: {tested}/{total} ({rate:.1f}/sec, ~{remaining:.0f}s remaining)")

    elapsed_p1 = (datetime.now() - t0).total_seconds()
    results_p1.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"\nPhase 1 complete: {len(results_p1)} valid in {elapsed_p1:.0f}s")

    # Show Phase 1 top 5
    logger.info("\nPhase 1 Top 5:")
    for i, r in enumerate(results_p1[:5]):
        p = r["params"]
        s = r["stats"]
        logger.info(f"  #{i+1}: {s['total_trades']}T, WR {s['win_rate']:.0%}, €{s['total_pnl']:+.2f} | "
                     f"z_lb={p['zscore_lookback']}, min_cs={p['min_composite_score']}, "
                     f"sl={p['atr_stop_multiplier']}, tp={p['atr_tp_multiplier']}")

    # ─── Phase 2: Validate top 20 across all 9 symbols ───────
    logger.info(f"\n{'=' * 70}")
    logger.info(f"Phase 2: Validating top 20 across {len(all_data)} symbols")
    logger.info(f"{'=' * 70}")

    results_p2 = []
    for i, r in enumerate(results_p1[:20]):
        params = r["params"]
        per_symbol = {}
        all_trades = 0
        all_pnl = 0.0
        all_wins = 0
        all_losses = 0

        for symbol, df_sig in all_data.items():
            stats = fast_backtest(df_sig, params)
            n = stats.get("total_trades", 0)
            pnl = stats.get("total_pnl", 0)
            all_trades += n
            all_pnl += pnl
            all_wins += stats.get("wins", 0)
            all_losses += stats.get("losses", 0)
            per_symbol[symbol] = {"trades": n, "pnl": round(pnl, 2), "wr": round(stats.get("win_rate", 0), 2)}

        if all_trades < 5:
            continue

        gross_profit = sum(v["pnl"] for v in per_symbol.values() if v["pnl"] > 0)
        gross_loss = abs(sum(v["pnl"] for v in per_symbol.values() if v["pnl"] < 0))

        agg = {
            "total_trades": all_trades,
            "wins": all_wins,
            "losses": all_losses,
            "win_rate": round(all_wins / all_trades, 3),
            "total_pnl": round(all_pnl, 2),
            "avg_pnl": round(all_pnl / all_trades, 2),
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 5.0,
            "symbols_profitable": sum(1 for v in per_symbol.values() if v["pnl"] > 0),
            "symbols_total": len(per_symbol),
        }

        s = score(agg)
        results_p2.append({"params": params, "stats": agg, "score": s, "per_symbol": per_symbol})

    results_p2.sort(key=lambda x: x["score"], reverse=True)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"PHASE 2 RESULTS — TOP 10 (validated across {len(all_data)} symbols)")
    logger.info(f"{'=' * 70}")

    for i, r in enumerate(results_p2[:10]):
        p = r["params"]
        s = r["stats"]
        logger.info(f"\n{'─' * 60}")
        logger.info(f"#{i+1}  Score: {r['score']:.4f}")
        logger.info(f"  zscore_lookback={p['zscore_lookback']}, min_composite={p['min_composite_score']}")
        logger.info(f"  atr_sl={p['atr_stop_multiplier']}, atr_tp={p['atr_tp_multiplier']}, "
                     f"vol_surge={p['volume_surge_threshold']}")
        logger.info(f"  Trades: {s['total_trades']}, WR: {s['win_rate']:.1%}, "
                     f"PnL: €{s['total_pnl']:+.2f}, PF: {s['profit_factor']:.2f}, "
                     f"Avg/Trade: €{s['avg_pnl']:+.2f}")
        logger.info(f"  Profitable symbols: {s['symbols_profitable']}/{s['symbols_total']}")

        active = {sym: st for sym, st in r["per_symbol"].items() if st["trades"] > 0}
        for sym, st in sorted(active.items(), key=lambda x: x[1]["pnl"], reverse=True):
            logger.info(f"    {sym}: {st['trades']}T, WR {st['wr']:.0%}, €{st['pnl']:+.2f}")

    # ─── Save ─────────────────────────────────────────────────
    if results_p2:
        best = results_p2[0]
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        report = {
            "timestamp": timestamp,
            "phase1_combos": len(results_p1),
            "phase1_elapsed_s": elapsed_p1,
            "phase2_validated": len(results_p2),
            "best_params": best["params"],
            "best_stats": best["stats"],
            "best_per_symbol": best["per_symbol"],
            "top_5": [
                {"rank": i+1, "params": r["params"], "stats": r["stats"]}
                for i, r in enumerate(results_p2[:5])
            ],
        }

        report_file = reports_dir / f"v2_optimization_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"\nReport saved: {report_file}")

        logger.info("\n" + "=" * 70)
        logger.info("OPTIMAL V2 PARAMETERS")
        logger.info("=" * 70)
        for k, v in best["params"].items():
            logger.info(f"  {k}: {v}")
        logger.info(f"\n  60-day performance ({len(all_data)} stocks):")
        logger.info(f"    Trades: {best['stats']['total_trades']}")
        logger.info(f"    Win Rate: {best['stats']['win_rate']:.1%}")
        logger.info(f"    Total PnL: €{best['stats']['total_pnl']:+.2f}")
        logger.info(f"    Profit Factor: {best['stats']['profit_factor']:.2f}")
        logger.info(f"    Avg PnL/Trade: €{best['stats']['avg_pnl']:+.2f}")
        logger.info("=" * 70)

        return best
    else:
        logger.error("No valid Phase 2 results.")
        return None


if __name__ == "__main__":
    main()
