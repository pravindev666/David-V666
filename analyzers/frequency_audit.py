"""
DAVID-V2: Setup Frequency Audit
===============================
Counts how many times each Super-Matrix setup appeared in the last 365 days.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_engine import load_all_data
from feature_forge import engineer_features
from models.ensemble_classifier import EnsembleClassifier
from analyzers.strike_backtester import _classify_regime, full_strike_analysis
from analyzers.whipsaw_detector import WhipsawDetector
from utils import UP, DOWN, SIDEWAYS, C

def run_frequency_audit():
    print(f"\n{C.header('DAVID-V2: SETUP FREQUENCY AUDIT (Last 365 Days)')}")
    
    df_raw = load_all_data()
    df, feature_cols = engineer_features(df_raw)
    
    ensemble = EnsembleClassifier()
    ensemble.load()
    whipsaw_engine = WhipsawDetector()
    
    lookback = 252 # 1 trading year
    results = []
    
    start_idx = len(df) - lookback
    end_idx = len(df)
    
    print(f"  Scanning {lookback} trading days...")
    
    for i in range(start_idx, end_idx):
        if i % 50 == 0: print(f"    Progress: {i-start_idx}/{lookback}...")
            
        current_row = df.iloc[i:i+1]
        hist_df = df.iloc[:i+1]
        
        spot = float(df.iloc[i]["close"])
        vix = float(df.iloc[i].get("vix", 15.0))
        
        # Signals
        pred = ensemble.predict(current_row)
        ai_dir = pred["direction"]
        conf = pred["confidence"]
        
        if ai_dir == SIDEWAYS: continue
        
        regime = _classify_regime(df.iloc[i])
        whipsaw_data = whipsaw_engine.analyze(hist_df)
        chop_prob = whipsaw_data["whipsaw_prob"]
        
        # History Grade Proxy
        rsi = float(df.iloc[i].get("rsi_7", 50))
        trust_proxy = 90
        if (ai_dir == UP and rsi > 70) or (ai_dir == DOWN and rsi < 30): trust_proxy -= 15
        if vix > 25: trust_proxy -= 10
        trust_score = trust_proxy

        # Binning
        ai_bin = 'High' if conf >= 0.55 else ('Med' if conf >= 0.45 else 'Low')
        trust_bin = 'A/A+' if trust_score >= 85 else ('B' if trust_score >= 70 else 'C/D')
        chop_bin = 'Clear' if chop_prob <= 35 else 'Choppy'
        
        results.append({
            "setup": f"{regime} | {chop_bin} | {ai_bin} | {trust_bin}"
        })

    results_df = pd.DataFrame(results)
    counts = results_df['setup'].value_counts()
    
    print(f"\n{C.BOLD}SETUP FREQUENCY table (Last 1 Year):{C.RESET}")
    print(counts.to_markdown())
    
    print(f"\n{C.BOLD}Key Observations:{C.RESET}")
    god_mode = "TRENDING | Clear | High | A/A+"
    bb_mode = "SIDEWAYS | Clear | High | A/A+"
    
    print(f"  - 🏆 **God-Mode Setup:** Appeared {counts.get(god_mode, 0)} times.")
    print(f"  - 🥦 **Bread & Butter (Sideways):** Appeared {counts.get(bb_mode, 0)} times.")
    print(f"  - 🛑 **Total Signal Clashes avoided:** (Handled by Trade Eligibility card)")

if __name__ == "__main__":
    run_frequency_audit()
