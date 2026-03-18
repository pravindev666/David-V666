"""
backtest_david.py  —  David v6.6.6+ Strategy Backtester
========================================================
Plug-in backtest for your exact trading style:
  UP  signal  →  Bull Put Spread  (short put  @ spot - 400,  long put  @ spot - 550)
  DOWN signal →  Bear Call Spread (short call @ spot + 400,  long call @ spot + 550)

How to run
----------
  python backtest_david.py                         # last 1 month
  python backtest_david.py --months 3              # last 3 months
  python backtest_david.py --start 2025-01-01 --end 2026-03-18
  python backtest_david.py --months 1 --min-score 80
  python backtest_david.py --months 3 --report     # saves CSV + prints summary

Dependencies
------------
  pip install pandas numpy yfinance

Your David project files needed
--------------------------------
  feature_forge.py      →  engineer_features(df)  returns DataFrame with 63+ features
  models/meta_ensemble.py →  MetaEnsemble class with .predict(features) method
                             returns dict: { 'direction': 'UP'|'DOWN'|'HOLD',
                                             'confidence': float 0-1,
                                             'tree_conf': float,
                                             'lstm_conf': float,
                                             'regime': str }
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# ─── 0. Add your David project root to path ─────────────────────────────────
DAVID_ROOT = Path(__file__).parent
sys.path.insert(0, str(DAVID_ROOT))

try:
    from feature_forge import engineer_features
    from models.meta_ensemble import MetaEnsemble
    from models.regime_detector import RegimeDetector
    DAVID_LOADED = True
    print("[OK] David modules loaded.")
except ImportError as e:
    DAVID_LOADED = False
    print(f"[WARN] Could not load David modules: {e}")
    print("[INFO] Running in DEMO mode with synthetic signals.")


# ─── 1. Constants — your exact strategy ─────────────────────────────────────

STRATEGY = {
    "otm_distance":   400,    # points away from spot
    "spread_width":   150,    # width of the spread in points
    "premium_recv":   1000,   # ₹ collected per lot
    "lot_size":       50,     # Nifty lot size
    "hold_days":      5,      # exit on day 5 if not stopped
    "stop_pct":       2.0,    # 2% adverse move = early exit (full loss)
    "min_edge_score": 60,     # minimum edge score to take a trade
}

# Edge score thresholds
TRUST = {"A+": 80, "A": 60, "B": 40}


# ─── 2. Data fetcher ─────────────────────────────────────────────────────────

def fetch_nifty(start: str, end: str) -> pd.DataFrame:
    """Fetch Nifty 50 OHLCV from yfinance."""
    try:
        import yfinance as yf
        df = yf.download("^NSEI", start=start, end=end, progress=False, auto_adjust=True)
        if df.columns.nlevels > 1:
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df.columns = ["open", "high", "low", "close", "volume"]
        df.dropna(inplace=True)
        print(f"[OK] Fetched {len(df)} candles from {df.index[0].date()} to {df.index[-1].date()}")
        return df
    except Exception as e:
        print(f"[ERROR] yfinance fetch failed: {e}")
        sys.exit(1)


# ─── 3. Edge Score calculator (rule-based, mirrors your dashboard) ───────────

def compute_edge_score(pred: dict) -> int:
    """Compute the 0-100 edge score from David's prediction output."""
    score = 0
    d = pred.get("direction", "HOLD")
    conf = pred.get("confidence", 0)
    tree = pred.get("tree_conf", 0)
    lstm = pred.get("lstm_conf", 0)
    regime = pred.get("regime", "")
    vix_spread = pred.get("vix_spread", 0)
    pcr_z = pred.get("pcr_zscore_5d", 0) # Mapping to our new feature name
    fii_net = pred.get("fii_net", 0)
    bn_lag1 = pred.get("bn_return_lag1", 0) # Mapping to our new feature name

    if d in ("UP", "DOWN"):
        score += 20
    if conf >= 0.70:
        score += 20
    if tree >= 0.60 and lstm >= 0.60:
        score += 20

    mild_regimes = ("mild_bull", "mild_bear", "sideways")
    if any(r in regime for r in mild_regimes):
        score += 15

    if (d == "UP" and pcr_z < -1.0) or (d == "DOWN" and pcr_z > 1.0):
        score += 10

    if (d == "UP" and fii_net > 0) or (d == "DOWN" and fii_net < 0):
        score += 10

    if (d == "UP" and bn_lag1 > 0) or (d == "DOWN" and bn_lag1 < 0):
        score += 5

    if vix_spread < 0:
        score -= 20
    strong_regimes = ("strong_bull", "strong_bear")
    if any(r in regime for r in strong_regimes):
        score -= 15
    if abs(tree - lstm) > 0.20:
        score -= 10

    return max(-35, min(100, score))


def get_trust_grade(score: int) -> str:
    if score >= 80: return "A+"
    if score >= 60: return "A"
    if score >= 40: return "B"
    return "C"


# ─── 4. P&L engine — your exact spread mechanics ────────────────────────────

def simulate_spread(direction: str, entry_spot: float,
                    future_prices: pd.Series,
                    cfg: dict) -> dict:
    """Simulates a 5-day spread hold with 2% stop-loss."""
    otm   = cfg["otm_distance"]
    width = cfg["spread_width"]
    prem  = cfg["premium_recv"]
    lot   = cfg["lot_size"]
    stop  = cfg["stop_pct"] / 100
    days  = cfg["hold_days"]

    max_loss = width * lot - prem

    if direction == "UP":
        short_strike = entry_spot - otm
        blow_level   = short_strike
        stop_level   = entry_spot * (1 - stop)
    else:
        short_strike = entry_spot + otm
        blow_level   = short_strike
        stop_level   = entry_spot * (1 + stop)

    outcome = "WIN"
    exit_day = len(future_prices)
    exit_price = future_prices.iloc[-1] if len(future_prices) > 0 else entry_spot

    for i, price in enumerate(future_prices[:days]):
        if direction == "UP" and price <= stop_level:
            outcome  = "STOP"
            exit_day = i + 1
            exit_price = price
            break
        if direction == "DOWN" and price >= stop_level:
            outcome  = "STOP"
            exit_day = i + 1
            exit_price = price
            break
        if direction == "UP" and price <= blow_level:
            outcome  = "BLOWN"
            exit_day = i + 1
            exit_price = price
            break
        if direction == "DOWN" and price >= blow_level:
            outcome  = "BLOWN"
            exit_day = i + 1
            exit_price = price
            break

    pnl = prem if outcome == "WIN" else -max_loss

    return {
        "outcome":     outcome,
        "exit_day":    exit_day,
        "exit_price":  round(exit_price, 1),
        "short_strike": round(short_strike, 0),
        "pnl":         pnl,
        "max_loss":    max_loss,
    }


# ─── 5. Demo signal generator ────────────────────────────────────────────────

def generate_demo_signal(row: pd.Series, prev_rows: pd.DataFrame) -> dict:
    """Synthetic signal proxy."""
    if len(prev_rows) < 10:
        return {"direction": "HOLD", "confidence": 0.5, "tree_conf": 0.5,
                "lstm_conf": 0.5, "regime": "sideways", "vix_spread": 0}

    mom_5   = prev_rows["close"].iloc[-1] / prev_rows["close"].iloc[-6] - 1
    direction  = "UP" if mom_5 > 0 else "DOWN"
    confidence = min(0.95, 0.55 + abs(mom_5) * 8)

    return {
        "direction":   direction,
        "confidence":  round(np.clip(confidence, 0, 1), 3),
        "tree_conf":   round(np.clip(confidence, 0, 1), 3),
        "lstm_conf":   round(np.clip(confidence, 0, 1), 3),
        "regime":      "mild_bull" if mom_5 > 0 else "mild_bear",
        "vix_spread":  1.5,
    }


# ─── 6. Main backtest loop ───────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, cfg: dict, model=None) -> pd.DataFrame:
    records = []
    dates   = df.index.tolist()
    min_warmup = 60

    print(f"\n{'─'*70}")
    print(f"  Running backtest: {dates[min_warmup].date()} → {dates[-1].date()}")
    print(f"  Min edge score: {cfg['min_edge_score']}")
    print(f"{'─'*70}\n")

    for i in range(min_warmup, len(dates) - cfg["hold_days"]):
        today       = dates[i]
        today_row   = df.iloc[i]
        entry_spot  = float(today_row["close"])
        future_slice = df.iloc[i+1 : i+1+cfg["hold_days"]]["close"]

        if DAVID_LOADED and model is not None:
            try:
                # Engineering features for current window
                window_df, feature_cols = engineer_features(df.iloc[max(0, i-252):i+1])
                pred = model.predict(window_df.iloc[-1:])
            except Exception as e:
                # print(f"  [WARN] Prediction failed on {today.date()}: {e}")
                pred = generate_demo_signal(today_row, df.iloc[:i])
        else:
            pred = generate_demo_signal(today_row, df.iloc[:i])

        direction = pred.get("direction", "HOLD")
        if direction == "HOLD":
            continue

        edge_score  = compute_edge_score(pred)
        trust_grade = get_trust_grade(edge_score)
        skip = edge_score < cfg["min_edge_score"]

        if not skip:
            result = simulate_spread(direction, entry_spot, future_slice, cfg)
        else:
            result = {"outcome": "SKIP", "exit_day": 0, "exit_price": entry_spot,
                      "short_strike": 0, "pnl": 0, "max_loss": 0}

        records.append({
            "date":          today.date(),
            "direction":     direction,
            "entry_spot":    round(entry_spot, 0),
            "short_strike":  result["short_strike"],
            "edge_score":    edge_score,
            "trust":         trust_grade,
            "taken":         not skip,
            "outcome":       result["outcome"],
            "pnl":           result["pnl"],
            "regime":        pred.get("regime", "N/A"),
        })

    return pd.DataFrame(records)


def print_report(df: pd.DataFrame, cfg: dict):
    taken = df[df["taken"]]
    wins   = taken[taken["outcome"] == "WIN"]
    losses = taken[taken["outcome"].isin(["BLOWN", "STOP"])]
    total  = len(taken)

    win_rate = len(wins) / total * 100 if total > 0 else 0
    gross_pnl = taken["pnl"].sum()

    print(f"\n{'═'*70}")
    print(f"  BACKTEST RESULTS  —  David v6.6.6+ | Alpha Setup")
    print(f"{'═'*70}")
    print(f"  Trades taken    :  {total}")
    print(f"  Win rate        :  {win_rate:.1f}%")
    print(f"  Gross P&L       :  ₹{gross_pnl:>10,.0f}")
    print(f"{'─'*70}")
    
    # Trust breakdown
    for grade in ["A+", "A", "B"]:
        g_df = taken[taken["trust"] == grade]
        if len(g_df) == 0: continue
        g_wr = len(g_df[g_df["outcome"] == "WIN"]) / len(g_df) * 100
        print(f"    Trust {grade:<4} | WR: {g_wr:>5.1f}% | P&L: ₹{g_df['pnl'].sum():>8,.0f}")
    print(f"{'═'*70}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months",    type=int,   default=1)
    parser.add_argument("--start",     type=str,   default=None)
    parser.add_argument("--end",       type=str,   default=None)
    parser.add_argument("--min-score", type=int,   default=60)
    parser.add_argument("--report",    action="store_true")
    args = parser.parse_args()

    end_dt   = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.today()
    start_dt = (datetime.strptime(args.start, "%Y-%m-%d") if args.start
                else end_dt - timedelta(days=args.months * 31))

    df = fetch_nifty((start_dt - timedelta(days=365)).strftime("%Y-%m-%d"), 
                     (end_dt + timedelta(days=10)).strftime("%Y-%m-%d"))

    cfg = {**STRATEGY}
    cfg["min_edge_score"] = args.min_score

    model = None
    if DAVID_LOADED:
        try:
            regime_detector = RegimeDetector()
            if regime_detector.load():
                model = MetaEnsemble(regime_detector)
                if model.load():
                    print("[OK] MetaEnsemble and Sub-models loaded for backtest.")
                else:
                    print("[WARN] MetaEnsemble weights not found. Check training.")
            else:
                print("[WARN] Regime Detector weights not found.")
        except Exception as e:
            print(f"[WARN] Model load error: {e}")
            print("[INFO] Falling back to demo signals.")

    results = run_backtest(df, cfg, model)
    if len(results) > 0:
        results = results[results["date"] >= start_dt.date()]
        print_report(results, cfg)
        if args.report:
            results.to_csv("backtest_results.csv", index=False)

if __name__ == "__main__":
    main()
