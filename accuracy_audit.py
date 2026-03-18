"""
accuracy_audit.py — Regime-Stratified Performance Auditor
=========================================================
Audits David's directional accuracy across different HMM regimes.
Helps identify 'Golden Windows' where the model has a massive edge.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ensure imports work from root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_engine import load_all_data
from feature_forge import engineer_features
from models.meta_ensemble import MetaEnsemble
from models.regime_detector import RegimeDetector
from utils import C, UP, DOWN, SIDEWAYS, HOLD

def run_regime_audit(lookback_days=None, start_date=None, min_score=60, group_by='regime', filter_regime=None):
    print(f"\n{C.header('RUNNING REGIME-STRATIFIED ACCURACY AUDIT')}")
    
    if start_date:
        print(f"{C.dim('Period: From ' + start_date + ' | Min Edge Score: ' + str(min_score))}\n")
    else:
        print(f"{C.dim('Period: Last ' + str(lookback_days) + ' days | Min Edge Score: ' + str(min_score))}\n")

    # 1. Load Data
    df = load_all_data()
    df, feature_cols = engineer_features(df)
    
    # Filter for recent data
    if start_date:
        cutoff = pd.to_datetime(start_date)
    else:
        cutoff = datetime.now() - timedelta(days=lookback_days)
    
    df = df[df.index >= cutoff].copy()
    
    if len(df) == 0:
        print(f"{C.RED}[ERROR] No data found for the specified period.{C.RESET}")
        return

    # 2. Load Models
    rd = RegimeDetector()
    if not rd.load():
        print(f"{C.RED}[ERROR] Regime detector not found.{C.RESET}")
        return
        
    me = MetaEnsemble(rd)
    if not me.load():
        print(f"{C.RED}[ERROR] Meta-Ensemble not found.{C.RESET}")
        return

    # 3. Run Predictions
    results = []
    print(f"  [OK] Processing {len(df)} trading days...")
    
    for i in range(20, len(df)): # Start at 20 to allow LSTM warmup
        row = df.iloc[i:i+1]
        hist_df = df.iloc[i-20:i+1] 
        date = df.index[i]
        
        true_dir = df['target_label'].iloc[i]
        w_score = df['whipsaw_score'].iloc[i]
        
        # Predict
        pred = me.predict(hist_df)
        
        # Calculate Edge Score
        base_score = (pred['confidence'] - 0.5) * 200 
        
        tree_dir = UP if pred['tree_conf'] > 0.5 else DOWN 
        lstm_dir = UP if pred['lstm_conf'] > 0.5 else DOWN
        attn_dir = UP if pred['attn_conf'] > 0.5 else DOWN
        
        agreement = (tree_dir == lstm_dir == attn_dir == pred['direction'])
        edge_score = base_score + (10 if agreement else 0)
        edge_score = max(0, min(100, int(edge_score)))
        
        # ADAPTIVE THRESHOLD (Layer 1 Optimization)
        # Raise filter to 80 for MILD BEARISH
        min_bearish = 80
        regime = pred['regime'].upper()
        required_score = min_bearish if "MILD BEAR" in regime else min_score
        
        if pred['direction'] != HOLD and edge_score >= required_score:
            is_win = (pred['direction'] == true_dir)
            
            results.append({
                "Date": date,
                "Month": date.strftime('%Y-%m'),
                "Regime": pred['regime'],
                "Direction": pred['direction'],
                "True": true_dir,
                "Score": edge_score,
                "Whipsaw": w_score,
                "Win": 1 if is_win else 0
            })

    if not results:
        print(f"{C.YELLOW}[WARN] No signals met the score threshold.{C.RESET}")
        return

    audit_df = pd.DataFrame(results)
    
    if filter_regime:
        target = filter_regime.upper().replace("_", " ")
        audit_df = audit_df[audit_df['Regime'].str.upper() == target].copy()
        print(f"  {C.YELLOW}[FILTER] Showing only {target} regime trades.{C.RESET}")
        if len(audit_df) == 0:
            print(f"{C.RED}[ERROR] No trades found for regime: {target}{C.RESET}")
            return

    # 4. Stratified Report
    group_cols = group_by.lower().split(',')
    group_cols = [c.capitalize() for c in group_cols]
    
    # Ensure columns exist
    valid_cols = []
    for c in group_cols:
        if c in audit_df.columns:
            valid_cols.append(c)
    
    if not valid_cols:
        valid_cols = ['Regime']

    print(f"\n{C.BOLD}{' | '.join(valid_cols):<30} | {'TRADES':<6} | {'WIN RATE':<8} | {'AVG WS'}{C.RESET}")
    print(f"{'─'*65}")
    
    # Overall
    all_wr = audit_df['Win'].mean() * 100
    all_ws = audit_df['Whipsaw'].mean()
    label = "OVERALL"
    print(f"{C.CYAN}{label:<30} | {len(audit_df):<6} | {all_wr:>7.1f}% | {all_ws:>6.1f}{C.RESET}")
    
    # Grouped
    stats = audit_df.groupby(valid_cols).agg({
        'Win': ['count', 'mean', 'sum'],
        'Whipsaw': 'mean'
    }).reset_index()
    
    # Flatten columns
    stats.columns = valid_cols + ['count', 'wr', 'hits', 'ws']
    stats['wr'] *= 100
    
    for _, row in stats.iterrows():
        name_parts = [str(row[c]).upper() for c in valid_cols]
        name = " / ".join(name_parts)
        count = int(row['count'])
        wr = row['wr']
        hits = int(row['hits'])
        ws = row['ws']
        
        color = C.GREEN if wr >= 72 else (C.YELLOW if wr >= 65 else C.RED)
        print(f"{name:<30} | {count:<6} | {color}{wr:>7.1f}%{C.RESET} | {ws:>6.1f}")

    print(f"\n{C.dim('Audit complete. Optimized 80+ Edge logic in MILD BEARISH active.')}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-edge", type=int, default=60)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--group-by", type=str, default="regime")
    parser.add_argument("--regime", type=str, default=None)
    parser.add_argument("--report", action="store_true") 
    
    args = parser.parse_args()
    run_regime_audit(
        lookback_days=args.days, 
        start_date=args.start, 
        min_score=args.min_edge, 
        group_by=args.group_by,
        filter_regime=args.regime
    )
