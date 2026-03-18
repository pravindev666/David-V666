"""
DAVID PROPHETIC ORACLE — Model Training Script
==============================================
Automated script to sync data, engineer features, and retrain all ML models.
Used by GitHub Actions for weekly model updates.
"""

import os
import sys
import pandas as pd

# Ensure imports work from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_engine import load_all_data
from feature_forge import engineer_features
from models.meta_ensemble import MetaEnsemble
from models.regime_detector import RegimeDetector
from models.range_predictor import RangePredictor
from utils import C, MODEL_DIR

import json
from models.regime_detector import RegimeDetector

def run_training_pipeline(force=False, cutoff=None):
    print(f"\n{C.header('AI MODEL RETRAINING PIPELINE (V6.6.6+ Sniper)')}")
    print(f"{'═'*50}")

    # 1. Sync Data & Features
    print(f"\n{C.CYAN}[STEP 1/5] Syncing latest market data...{C.RESET}")
    df_raw = load_all_data()
    df, feature_cols = engineer_features(df_raw)

    # 1.5 Apply Cutoff if specified
    if cutoff:
        cutoff_ts = pd.to_datetime(cutoff)
        print(f"  {C.YELLOW}Applying Training Cutoff: {cutoff_ts.date()}{C.RESET}")
        df = df[df.index <= cutoff_ts].copy()
        print(f"  Training Samples: {len(df)}")

    # 2. Check for Regime Change
    metadata_path = os.path.join(MODEL_DIR, "train_metadata.json")
    last_regime = "None"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            meta = json.load(f)
            last_regime = meta.get("last_train_regime", "None")

    # Temporarily load detector to get CURRENT regime
    temp_rd = RegimeDetector()
    if temp_rd.load() and len(df) > 0:
        current_regime, _, _ = temp_rd.get_current_regime(df.iloc[-1:])
    else:
        current_regime = "Initial"

    print(f"  {C.YELLOW}Last Train Regime: {last_regime} | Current Regime: {current_regime}{C.RESET}")

    if not force and not cutoff and current_regime == last_regime:
        print(f"\n{C.GREEN}✅ No regime change detected. Skipping retraining.{C.RESET}")
        return

    print(f"\n{C.CYAN}[STEP 2/5] Training 5-State HMM Regime Detector...{C.RESET}")
    # 3. Train Regime Detector (HMM)
    regime = RegimeDetector()
    regime.train(df)
    regime.save()

    # 4. Train Meta-Ensemble Classifier (45% Tree / 30% LSTM / 25% Transformer)
    print(f"\n{C.CYAN}[STEP 3/5] Training Meta-Ensemble Fusion (3 Pillars)...{C.RESET}")
    meta_ensemble = MetaEnsemble(regime)
    meta_ensemble.train(df, feature_cols)
    meta_ensemble.save()

    # 5. Train Range Predictor
    print(f"\n{C.CYAN}[STEP 4/5] Training Range Predictors...{C.RESET}")
    rp = RangePredictor()
    rp.train(df, feature_cols)
    rp.save()

    # Save Metadata
    with open(metadata_path, "w") as f:
        json.dump({
            "last_train_regime": current_regime,
            "last_train_cutoff": cutoff
        }, f)

    print(f"\n{C.GREEN}✅ RETRAINING COMPLETE (Trigger: {current_regime}){C.RESET}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force retrain even if no regime change")
    parser.add_argument("--cutoff", type=str, default=None, help="Hard cutoff date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    try:
        run_training_pipeline(force=args.force, cutoff=args.cutoff)
    except Exception as e:
        print(f"\n{C.BOLD}{C.RED}❌ CRITICAL ERROR IN TRAINING PIPELINE:{C.RESET}")
        print(f"{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
