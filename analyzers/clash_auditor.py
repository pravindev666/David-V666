"""
DAVID-V2: Signal Clash Auditor
==============================
Analyzes historical performance when AI Verdict and Market Regime are in conflict.
Focus: "AI UP vs Regime BEARISH"
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_engine import load_all_data
from feature_forge import engineer_features
from models.ensemble_classifier import EnsembleClassifier
from models.regime_detector import RegimeDetector
from utils import UP, DOWN, SIDEWAYS, C

def run_clash_audit():
    print(f"\n{C.header('DAVID-V2: SIGNAL CLASH AUDIT')}")
    
    # 1. Load Data & Models
    df_raw = load_all_data()
    df, feature_cols = engineer_features(df_raw)
    
    ensemble = EnsembleClassifier()
    if not ensemble.load():
        print("Ensemble not found. Training...")
        ensemble.train(df, feature_cols)
        ensemble.save()
        
    regime_model = RegimeDetector()
    if not regime_model.load():
        print("Regime model not found. Training...")
        regime_model.train(df)
        regime_model.save()

    # 2. Iterate through historical data (last 500 days for speed and relevance)
    lookback = 500
    results = []
    
    print(f"  Auditing last {lookback} trading days...")
    
    # Start from lookback rows ago up to 5 days before end (to have 5-day forward return)
    start_idx = len(df) - lookback
    end_idx = len(df) - 6
    
    clash_count = 0
    sync_count = 0
    
    for i in range(start_idx, end_idx):
        current_row = df.iloc[i:i+1]
        hist_df = df.iloc[:i+1]
        
        # Get AI Prediction
        pred = ensemble.predict(current_row)
        ai_dir = pred["direction"]
        
        # Get Regime
        regime_label, _, _ = regime_model.get_current_regime(hist_df)
        
        # Future outcome (5-day return and MAX DRAWDOWN)
        close_now = df.iloc[i]["close"]
        future_window = df.iloc[i+1 : i+6]
        close_future = df.iloc[i+5]["close"]
        
        # Max drop during these 5 days
        max_drop_pts = close_now - future_window["low"].min()
        max_drop_pct = max_drop_pts / close_now
        
        ret_5d = (close_future - close_now) / close_now
        
        actual_dir = SIDEWAYS
        if ret_5d > 0.01: actual_dir = UP
        elif ret_5d < -0.01: actual_dir = DOWN
        
        # Define Clash
        is_clash = False
        if ai_dir == UP and "BEARISH" in regime_label:
            is_clash = True
        elif ai_dir == DOWN and "BULLISH" in regime_label:
            is_clash = True
            
        is_sync = False
        if ai_dir == UP and "BULLISH" in regime_label:
            is_sync = True
        elif ai_dir == DOWN and "BEARISH" in regime_label:
            is_sync = True

        # Side-specific risk
        side_mae = 0
        if ai_dir == UP:
            side_mae = close_now - future_window["low"].min()
        elif ai_dir == DOWN:
            side_mae = future_window["high"].max() - close_now
            
        results.append({
            "date": df.iloc[i]["date"],
            "ai_dir": ai_dir,
            "regime": regime_label,
            "actual_dir": actual_dir,
            "ret_5d": ret_5d,
            "side_mae": side_mae,
            "is_clash": is_clash,
            "is_sync": is_sync
        })
        
        if is_clash: clash_count += 1
        if is_sync: sync_count += 1

    audit_df = pd.DataFrame(results)
    
    # 3. Analyze Results
    print(f"\n{C.BOLD}Audit Statistics (Last {lookback} Days):{C.RESET}")
    print(f"  Signal Clashes Found: {clash_count} ({clash_count/len(audit_df):.1%})")
    print(f"  Signals in Sync:      {sync_count} ({sync_count/len(audit_df):.1%})")
    
    # Success Rate during Sync
    sync_df = audit_df[audit_df["is_sync"]]
    sync_win = len(sync_df[sync_df["ai_dir"] == sync_df["actual_dir"]])
    sync_win_rate = sync_win / len(sync_df) if len(sync_df) > 0 else 0
    
    # Success Rate during Clash
    clash_df = audit_df[audit_df["is_clash"]]
    clash_win = len(clash_df[clash_df["ai_dir"] == clash_df["actual_dir"]])
    clash_win_rate = clash_win / len(clash_df) if len(clash_df) > 0 else 0
    
    # --- P&L SIMULATION (Simplified) ---
    # Assume 1 lot of Nifty (65 qty)
    # Profit = 5d return % * Spot * Qty
    qty = 65
    
    # Strategy 1: TRADE EVERYTHING (Sync + Clash)
    all_trades = audit_df[audit_df["is_sync"] | audit_df["is_clash"]]
    all_pnl = (all_trades["ret_5d"] * close_now * qty).sum()
    
    # Strategy 2: SYNC ONLY (The "Stay Out" Strategy)
    sync_trades = audit_df[audit_df["is_sync"]]
    sync_pnl = (sync_trades["ret_5d"] * close_now * qty).sum()

    # Avg Side-MAE during Clash vs Sync
    sync_mae = sync_df["side_mae"].mean()
    clash_mae = clash_df["side_mae"].mean()
    p95_sync_mae = sync_df["side_mae"].quantile(0.95)
    p95_clash_mae = clash_df["side_mae"].quantile(0.95)

    print(f"\n{C.BOLD}Outcome Probabilities:{C.RESET}")
    print(f"  {C.GREEN}When Sync'd: AI Accurately Predicted Move {sync_win_rate:.1%}{C.RESET}")
    print(f"  {C.RED}When Clash'd: AI Accurately Predicted Move {clash_win_rate:.1%}{C.RESET}")
    
    print(f"\n{C.BOLD}Volatility & Strike Breach Risk (MAE):{C.RESET}")
    print(f"  When Sync'd: Avg Max Dip (if UP) or Rally (if DOWN) = {sync_mae:.0f} pts | P95 = {p95_sync_mae:.0f} pts")
    print(f"  When Clash'd: Avg Max Dip (if UP) or Rally (if DOWN) = {clash_mae:.0f} pts | P95 = {p95_clash_mae:.0f} pts")
    
    print(f"\n{C.BOLD}Profitability Backtest (Estimated ₹/Lot):{C.RESET}")
    print(f"  {C.YELLOW}Strategy A (Trade Everything): ₹{all_pnl:,.0f} Total Profit{C.RESET}")
    print(f"  {C.up('Strategy B (Sync Only - STAY OUT):')} ₹{sync_pnl:,.0f} Total Profit")
    
    print(f"\n{C.BOLD}Verdict:{C.RESET}")
    if sync_pnl > all_pnl:
        improvement = (sync_pnl - all_pnl)
        print(f"  ✅ YES. Staying out on Red would have increased profit by ₹{improvement:,.0f}.")
        print(f"  🛡️ By avoiding clashes, you avoided the most volatile trades which usually lead to stop-losses.")
    elif clash_mae > sync_mae:
        print(f"  ⚠️ Staying out might result in slightly less total profit, BUT much higher safety.")
        print(f"  🚨 Signal clashes increase strike breach risk by {(clash_mae-sync_mae)/sync_mae:.1%}. Trading them is high-risk gambling.")
    else:
        print("  Signal clashes did not significantly increase drawdown risk in this specific window.")

if __name__ == "__main__":
    run_clash_audit()
