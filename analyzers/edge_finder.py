"""
DAVID-V2: Edge Discovery Matrix
===============================
Bins AI Conviction and Historical Trust Score to find the most profitable combinations.
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
from analyzers.strike_backtester import _classify_regime
from analyzers.whipsaw_detector import WhipsawDetector
from utils import UP, DOWN, SIDEWAYS, C

def run_edge_discovery():
    print(f"\n{C.header('DAVID-V2: EDGE DISCOVERY MATRIX')}")
    
    # 1. Load Data & Models
    df_raw = load_all_data()
    df, feature_cols = engineer_features(df_raw)
    
    ensemble = EnsembleClassifier()
    if not ensemble.load():
        print("Ensemble not found. Training...")
        ensemble.train(df, feature_cols)
        ensemble.save()
        
    whipsaw_engine = WhipsawDetector()
        
    # 2. Iterate through historical data (audit last 700 days for better coverage)
    lookback = 700
    results_5d = []
    results_3d = []
    
    print(f"  Crunching records for last {lookback} trading days...")
    
    # Start from lookback rows ago up to 5 days before end (to have 5-day forward return)
    start_idx = len(df) - lookback
    end_idx = len(df) - 6
    
    for i in range(start_idx, end_idx):
        if i % 50 == 0: print(f"    Progress: {i}/{end_idx}...")
            
        current_row = df.iloc[i:i+1]
        hist_df = df.iloc[:i+1]
        
        # Current state
        spot = float(df.iloc[i]["close"])
        vix = float(df.iloc[i].get("vix", 15.0))
        
        # Get AI Prediction
        pred = ensemble.predict(current_row)
        ai_dir = pred["direction"]
        conf = pred["confidence"]
        
        # Skip Sideways for this specific directional edge test
        if ai_dir == SIDEWAYS: continue
        
        # Get Market Regime & Whipsaw
        regime = _classify_regime(df.iloc[i])
        whipsaw_data = whipsaw_engine.analyze(hist_df)
        chop_prob = whipsaw_data["whipsaw_prob"]
        
        # Trust Proxy: f(VIX, RSI)
        rsi = float(df.iloc[i].get("rsi_7", 50))
        trust_proxy = 90
        if (ai_dir == UP and rsi > 70) or (ai_dir == DOWN and rsi < 30):
            trust_proxy -= 15
        if vix > 25:
            trust_proxy -= 10
            
        trust_score = trust_proxy
        
        # 5-day forward return
        close_5d = df.iloc[i+5]["close"]
        ret_5d = (close_5d - spot) / spot
        if ai_dir == DOWN: ret_5d = -ret_5d
        pnl_5d = ret_5d * spot * 65

        results_5d.append({
            "conf": conf,
            "trust": trust_score,
            "regime": regime,
            "chop": chop_prob,
            "pnl": pnl_5d,
            "success": 1 if ret_5d > 0 else 0
        })
        
        # 3-day forward return (if available)
        if i + 3 < len(df):
            close_3d = df.iloc[i+3]["close"]
            ret_3d = (close_3d - spot) / spot
            if ai_dir == DOWN: ret_3d = -ret_3d
            pnl_3d = ret_3d * spot * 65

            results_3d.append({
                "conf": conf,
                "trust": trust_score,
                "regime": regime,
                "chop": chop_prob,
                "pnl": pnl_3d,
                "success": 1 if ret_3d > 0 else 0
            })

    # 3. Generate matrices for both holding periods
    for label, results in [("5-DAY", results_5d), ("3-DAY", results_3d)]:
        results_df = pd.DataFrame(results)
        if results_df.empty:
            continue
        
        # Binning — conviction threshold lowered to 50% for "High"
        results_df['ai_bin'] = pd.cut(results_df['conf'], bins=[0, 0.45, 0.50, 1.0], labels=['Low', 'Med', 'High'])
        results_df['trust_bin'] = pd.cut(results_df['trust'], bins=[0, 60, 75, 110], labels=['C/D', 'B', 'A/A+'])
        results_df['chop_bin'] = pd.cut(results_df['chop'], bins=[0, 35, 110], labels=['Clear', 'Choppy'])
        
        # Matrix Generation
        matrix = results_df.groupby(['regime', 'chop_bin', 'ai_bin', 'trust_bin']).agg({
            'pnl': ['mean', 'count'],
            'success': 'mean'
        })
        
        matrix.columns = ['Avg Profit', 'Trade Count', 'Win Rate']
        matrix = matrix[matrix['Trade Count'] > 0]
        
        print(f"\n{C.BOLD}THE SUPER-MATRIX — {label} HOLD (Regime + Chop + AI + History){C.RESET}")
        print(df_to_markdown(matrix.sort_values('Avg Profit', ascending=False).head(15)))
        
        top = matrix.sort_values('Avg Profit', ascending=False).iloc[0]
        idx = matrix.sort_values('Avg Profit', ascending=False).index[0]
        
        print(f"\n  🏆 {label} GOD-MODE SETUP:")
        print(f"     Regime: {C.GREEN}{idx[0]}{C.RESET} | Chop: {C.GREEN}{idx[1]}{C.RESET} | AI: {C.GREEN}{idx[2]}{C.RESET} | History: {C.GREEN}{idx[3]}{C.RESET}")
        print(f"     Yields ₹{top['Avg Profit']:,.0f} per trade on average with a {top['Win Rate']:.1%} Win Rate.")
        
        # Summary
        qualified = results_df[results_df['ai_bin'].isin(['Med', 'High'])]
        print(f"\n  📊 {label} SUMMARY:")
        print(f"     Total directional trades: {len(results_df)}")
        print(f"     Qualified (Med+High conviction): {len(qualified)}")
        print(f"     Overall win rate: {results_df['success'].mean():.1%}")
        print(f"     Qualified win rate: {qualified['success'].mean():.1%}" if len(qualified) > 0 else "")

def df_to_markdown(df):
    from tabulate import tabulate
    return tabulate(df, headers='keys', tablefmt='pipe')

if __name__ == "__main__":
    run_edge_discovery()
