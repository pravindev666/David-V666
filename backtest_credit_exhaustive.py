"""
EXHAUSTIVE CREDIT SPREAD SIMULATION
===================================
Simulates the exact user strategy:
1. Enter on Signal Date
2. Hold for 5 Days (Theta Decay)
3. Evaluate Win/Loss based on 5-day return.

Tests V2 (Raw) vs V3 (Fixed) across multiple confidence thresholds.
"""

import os, sys
import numpy as np
import pandas as pd
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_engine import load_all_data
from feature_forge import engineer_features
from models.ensemble_classifier import EnsembleClassifier
from models.regime_detector import RegimeDetector
from sklearn.preprocessing import StandardScaler
from utils import UP, DOWN, SIDEWAYS

TEST_MONTHS = 6
TARGET_HORIZON = 5
V1_V2_THRESHOLD = 0.003
V3_THRESHOLD = 0.005

print(f"\n{'='*75}")
print(f"  EXHAUSTIVE CREDIT SPREAD SIMULATION (5-DAY HOLD)")
print(f"  Testing V2 (Base) vs V3 (All Fixes) across Confidences")
print(f"{'='*75}\n")

# 1. LOAD DATA
df_raw = load_all_data()
df_full, feature_cols = engineer_features(df_raw, target_horizon=TARGET_HORIZON)

cutoff_date = df_full["date"].max() - pd.Timedelta(days=TEST_MONTHS * 30)
last_verifiable_idx = len(df_full) - TARGET_HORIZON - 1

train_mask = df_full["date"] < cutoff_date
test_mask = (df_full["date"] >= cutoff_date) & (df_full.index <= last_verifiable_idx)

train_df = df_full[train_mask].copy()
test_df = df_full[test_mask].reset_index(drop=True)

# 2. TRAIN
print(f"Training ensemble & HMM on {len(train_df)} historical rows...\n")
X_train = train_df[feature_cols].values
y_train = train_df["target"].values.astype(int)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

ensemble = EnsembleClassifier()
models = ensemble._build_models()
for name, model in models.items():
    model.fit(X_train_scaled, y_train)

regime = RegimeDetector()
regime.train(train_df, verbose=False)

# 3. PREDICT & SIMULATE
results = []
for idx, row in test_df.iterrows():
    X_row = row[feature_cols].values.reshape(1, -1)
    X_row_scaled = scaler.transform(X_row)

    combined_probs = np.zeros(3)
    for name, model in models.items():
        probs = model.predict_proba(X_row_scaled)[0]
        combined_probs += probs / len(models)

    # V2 Logic (Raw ensemble)
    v2_pred_class = np.argmax(combined_probs)
    v2_direction = {0: UP, 1: DOWN, 2: SIDEWAYS}[v2_pred_class]
    v2_confidence = float(combined_probs[v2_pred_class])

    # V3 Logic (Sideways Override + Regime Gating)
    v3_direction = v2_direction
    v3_confidence = v2_confidence
    
    if v3_confidence < 0.40 and v3_direction != SIDEWAYS:
        v3_direction = SIDEWAYS
        v3_confidence = 1.0 - v2_confidence

    history = df_full[df_full["date"] <= row["date"]]
    regime_label, _, _ = regime.get_current_regime(history)
    bullish_regimes = ["STRONG BULLISH", "MILD BULLISH"]
    bearish_regimes = ["STRONG BEARISH", "MILD BEARISH"]

    if v3_direction == DOWN and regime_label in bullish_regimes and v3_confidence < 0.60:
        v3_direction = SIDEWAYS
        v3_confidence = 0.50
    elif v3_direction == UP and regime_label in bearish_regimes and v3_confidence < 0.60:
        v3_direction = SIDEWAYS
        v3_confidence = 0.50

    # Actual 5-day Outcome
    # The 'future_return' column exactly represents the return 5 days from now
    actual_return = row["future_return"]
    
    # V2 actual target (±0.3% threshold)
    if actual_return > V1_V2_THRESHOLD: v2_actual = UP
    elif actual_return < -V1_V2_THRESHOLD: v2_actual = DOWN
    else: v2_actual = SIDEWAYS

    # V3 actual target (±0.5% threshold)
    if actual_return > V3_THRESHOLD: v3_actual = UP
    elif actual_return < -V3_THRESHOLD: v3_actual = DOWN
    else: v3_actual = SIDEWAYS

    # CREDIT SPREAD WIN LOGIC
    # Bull Put Spread wins if market goes UP or SIDEWAYS
    # Bear Call Spread wins if market goes DOWN or SIDEWAYS
    
    v2_win = False
    if v2_direction == UP and v2_actual in [UP, SIDEWAYS]: v2_win = True
    if v2_direction == DOWN and v2_actual in [DOWN, SIDEWAYS]: v2_win = True
    if v2_direction == SIDEWAYS and v2_actual == SIDEWAYS: v2_win = True

    v3_win = False
    if v3_direction == UP and v3_actual in [UP, SIDEWAYS]: v3_win = True
    if v3_direction == DOWN and v3_actual in [DOWN, SIDEWAYS]: v3_win = True
    if v3_direction == SIDEWAYS and v3_actual == SIDEWAYS: v3_win = True

    # Calculate Exit Date (approximately 5 trading days later)
    # We can just look ahead in test_df if possible
    exit_date = row["date"] + pd.Timedelta(days=7) # Approx 5 trading days
    if idx + TARGET_HORIZON < len(test_df):
        exit_date = test_df.iloc[idx + TARGET_HORIZON]["date"]

    results.append({
        "entry_date": row["date"],
        "exit_date": exit_date,
        "entry_close": row["close"],
        "5d_return_pct": actual_return * 100,
        "v2_dir": v2_direction,
        "v2_conf": v2_confidence,
        "v2_win": v2_win,
        "v3_dir": v3_direction,
        "v3_conf": v3_confidence,
        "v3_win": v3_win,
    })

R = pd.DataFrame(results)

# 4. REPORT EXHAUSTIVE COMBINATIONS
print(f"  ── EXHAUSTIVE COMBINATION BACKTEST (CREDIT SPREADS) ──\n")

configurations = [
    ("V2 (Base Model)", "v2_dir", "v2_conf", "v2_win"),
    ("V3 (Fixed Model)", "v3_dir", "v3_conf", "v3_win")
]

thresholds = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60]

for model_name, dir_col, conf_col, win_col in configurations:
    print(f"  [{model_name}]")
    print(f"  {'Tactic':<22} | {'All Trades':<12} | {'>40% Conf':<12} | {'>45% Conf':<12} | {'>50% Conf':<12} | {'>55% Conf':<12} | {'>60% Conf':<12}")
    print(f"  {'-'*110}")
    
    for tactic, trade_dir in [("Bull Put Spread (UP)", UP), ("Bear Call Spread (DN)", DOWN)]:
        row_str = f"  {tactic:<22} | "
        
        for t in thresholds:
            mask = (R[dir_col] == trade_dir) & (R[conf_col] > t)
            sub = R[mask]
            
            if len(sub) > 0:
                wins = sub[win_col].sum()
                rate = (wins / len(sub)) * 100
                cell = f"{wins:>2}/{len(sub):<2} ({rate:>2.0f}%)"
            else:
                cell = f" 0/0  ( -%)"
                
            row_str += f"{cell:<12} | "
            
        print(row_str)
    print("")

# 5. SAMPLE TIMELINE (Last 10 trades for V3 > 50%)
print(f"  ── SAMPLE TIMELINE: V3 Trades > 50% Confidence ──")
print(f"  {'Entry Date':<12} {'Exit Date':<12} {'Strategy':<20} {'Conf':<6} {'5d Return':<10} {'Result':<10}")
print(f"  {'-'*75}")

good_trades = R[(R["v3_dir"].isin([UP, DOWN])) & (R["v3_conf"] > 0.50)].tail(15)
for _, r in good_trades.iterrows():
    strat = "Bull Put Spread" if r["v3_dir"] == UP else "Bear Call Spread"
    result = "✅ WIN" if r["v3_win"] else "❌ LOSS"
    entry_d = r["entry_date"].strftime("%Y-%m-%d")
    exit_d = r["exit_date"].strftime("%Y-%m-%d")
    print(f"  {entry_d:<12} {exit_d:<12} {strat:<20} {r['v3_conf']*100:>2.0f}%   {r['5d_return_pct']:>+.2f}%     {result}")

print(f"\n{'='*75}\n")
