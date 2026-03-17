"""
CREDIT SPREAD BACKTEST (Options Seller Logic)
=============================================
Options sellers (Credit Spreads) have a massive advantage:
they win if the market goes their way OR stays sideways.

WIN CONDITIONS (5-day hold):
- Predict UP (Bull Put Spread): Win if Actual is UP or SIDEWAYS
- Predict DOWN (Bear Call Spread): Win if Actual is DOWN or SIDEWAYS
- Predict SIDEWAYS (Iron Condor): Win if Actual is SIDEWAYS ONLY

This script tests V3 against this realistic trading logic over 6 months.
"""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_engine import load_all_data
from feature_forge import engineer_features
from models.ensemble_classifier import EnsembleClassifier
from models.regime_detector import RegimeDetector
from sklearn.preprocessing import StandardScaler
from utils import UP, DOWN, SIDEWAYS, SIDEWAYS_CONFIDENCE_THRESHOLD

TEST_MONTHS = 6
TARGET_HORIZON = 5

print(f"\n{'='*65}")
print(f"  OPTIONS SELLER (CREDIT SPREAD) BACKTEST — 6 MONTHS")
print(f"  Win Condition: Directional Bet + Sideways Decay (Theta)")
print(f"{'='*65}\n")

# 1. LOAD DATA
df_raw = load_all_data()
df_full, feature_cols = engineer_features(df_raw, target_horizon=TARGET_HORIZON)

cutoff_date = df_full["date"].max() - pd.Timedelta(days=TEST_MONTHS * 30)
last_verifiable_idx = len(df_full) - TARGET_HORIZON - 1

train_mask = df_full["date"] < cutoff_date
test_mask = (df_full["date"] >= cutoff_date) & (df_full.index <= last_verifiable_idx)

train_df = df_full[train_mask].copy()
test_df = df_full[test_mask].copy()

# 2. TRAIN
print(f"Training ensemble & HMM on {len(train_df)} rows...")
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

# 3. PREDICT (V3 Logic)
print(f"Walking forward {len(test_df)} days...")
results = []
for idx, row in test_df.iterrows():
    X_row = row[feature_cols].values.reshape(1, -1)
    X_row_scaled = scaler.transform(X_row)

    combined_probs = np.zeros(3)
    for name, model in models.items():
        probs = model.predict_proba(X_row_scaled)[0]
        combined_probs += probs / len(models)

    raw_pred_class = np.argmax(combined_probs)
    raw_direction = {0: UP, 1: DOWN, 2: SIDEWAYS}[raw_pred_class]
    raw_confidence = float(combined_probs[raw_pred_class])

    # V3 Logic: SIDEWAYS Override
    if raw_confidence < SIDEWAYS_CONFIDENCE_THRESHOLD and raw_direction != SIDEWAYS:
        direction = SIDEWAYS
        confidence = 1.0 - raw_confidence
    else:
        direction = raw_direction
        confidence = raw_confidence

    # V3 Logic: Regime Gating
    history = df_full[df_full["date"] <= row["date"]]
    regime_label, _, _ = regime.get_current_regime(history)
    bullish_regimes = ["STRONG BULLISH", "MILD BULLISH"]
    bearish_regimes = ["STRONG BEARISH", "MILD BEARISH"]

    if direction == DOWN and regime_label in bullish_regimes and confidence < 0.60:
        direction = SIDEWAYS
        confidence = 0.50
    elif direction == UP and regime_label in bearish_regimes and confidence < 0.60:
        direction = SIDEWAYS
        confidence = 0.50

    actual_class = int(row["target"])
    actual_direction = {0: UP, 1: DOWN, 2: SIDEWAYS}[actual_class]
    
    # ── CREDIT SPREAD WIN LOGIC ──
    v1_win = False
    v3_win = False

    # V1/V2 Logic (Raw)
    if raw_direction == UP and actual_direction in [UP, SIDEWAYS]: v1_win = True
    elif raw_direction == DOWN and actual_direction in [DOWN, SIDEWAYS]: v1_win = True
    elif raw_direction == SIDEWAYS and actual_direction == SIDEWAYS: v1_win = True

    # V3 Logic (Fixed)
    if direction == UP and actual_direction in [UP, SIDEWAYS]: v3_win = True
    elif direction == DOWN and actual_direction in [DOWN, SIDEWAYS]: v3_win = True
    elif direction == SIDEWAYS and actual_direction == SIDEWAYS: v3_win = True

    results.append({
        "date": row["date"],
        "raw_pred": raw_direction,
        "v3_pred": direction,
        "v3_conf": confidence,
        "actual": actual_direction,
        "v1_win": v1_win,
        "v3_win": v3_win,
        "is_sideways": actual_direction == SIDEWAYS
    })

R = pd.DataFrame(results)

# 4. REPORT
total = len(R)
v1_wins = R["v1_win"].sum()
v3_wins = R["v3_win"].sum()

print(f"\n{'='*65}")
print(f"  CREDIT SPREAD WIN RATES (Options Seller)")
print(f"{'='*65}\n")

print(f"  V1/V2 (Base Model):  {v1_wins:>3}/{total} = {v1_wins/total*100:.1f}% Win Rate")
print(f"  V3 (All Fixes):      {v3_wins:>3}/{total} = {v3_wins/total*100:.1f}% Win Rate")
print(f"  (Random guessing would yield ~66% since you get 2/3 directions)\n")

print(f"  ── DETAILED BREAKDOWN (V1/V2 RAW Base Model) ───────────")
for d in [UP, DOWN, SIDEWAYS]:
    mask = R["raw_pred"] == d
    sub = R[mask]
    if len(sub) > 0:
        wins = sub["v1_win"].sum()
        print(f"  Sold {d:<8} Spreads: {wins:>2}/{len(sub):<2} = {wins/len(sub)*100:>5.1f}% Win Rate")

print(f"\n  ── DETAILED BREAKDOWN (V3 @ 40% Confidence) ───────────")
for d in [UP, DOWN, SIDEWAYS]:
    mask = R["v3_pred"] == d
    sub = R[mask]
    if len(sub) > 0:
        wins = sub["v3_win"].sum()
        print(f"  Sold {d:<8} Spreads: {wins:>2}/{len(sub):<2} = {wins/len(sub)*100:>5.1f}% Win Rate")

print(f"\n  ── CONFIDENCE ANALYSIS (V3) ──────────────────────────")
for d in [UP, DOWN]:
    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        mask = (R["v3_pred"] == d) & (R["v3_conf"] >= t)
        sub = R[mask]
        if len(sub) > 0:
            wins = sub["v3_win"].sum()
            print(f"  {d:<4} Spreads > {int(t*100)}% Conf: {wins:>2}/{len(sub):<2} = {wins/len(sub)*100:>5.1f}% Win")

print(f"\n  ── WHY THIS HAPPENS ──────────────────────────────────")
sideways_count = R["is_sideways"].sum()
print(f"  Total SIDEWAYS outcomes in reality: {sideways_count} ({sideways_count/total*100:.1f}%)")
print(f"  Traditional accuracy treated these as FAILED predictions.")
print(f"  Credit spread accuracy treats these as WINNING trades (theta decay captures max profit).")

print(f"\n{'='*65}\n")
