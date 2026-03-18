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
from utils import C

def run_training_pipeline():
    print(f"\n{C.header('AI MODEL RETRAINING PIPELINE')}")
    print(f"{'═'*50}")

    # 1. Sync Data
    print(f"\n{C.CYAN}[STEP 1/5] Syncing latest market data...{C.RESET}")
    df_raw = load_all_data()

    # 2. Engineer Features
    print(f"\n{C.CYAN}[STEP 2/5] Engineering technical features...{C.RESET}")
    df, feature_cols = engineer_features(df_raw)

    # 3. Train Regime Detector (HMM) - Moved UP because MetaEnsemble needs it
    print(f"\n{C.CYAN}[STEP 3/5] Training 5-State Regime Detector...{C.RESET}")
    regime = RegimeDetector()
    regime.train(df)
    regime.save()

    # 4. Train Meta-Ensemble Classifier (Directional - V6.6.6)
    print(f"\n{C.CYAN}[STEP 4/5] Training Meta-Ensemble Fusion V6.6.6...{C.RESET}")
    # Replace old ensemble with the new binary + sequence meta-ensemble
    meta_ensemble = MetaEnsemble(regime)
    meta_ensemble.train(df, feature_cols)
    meta_ensemble.save()

    # 5. Train Range Predictor (Quantile Regression)
    print(f"\n{C.CYAN}[STEP 5/5] Training 7d & 30d Range Predictor...{C.RESET}")
    rp = RangePredictor()
    rp.train(df, feature_cols)
    rp.save()

    print(f"\n{C.GREEN}✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!{C.RESET}\n")

if __name__ == "__main__":
    try:
        run_training_pipeline()
    except Exception as e:
        print(f"\n{C.BOLD}{C.RED}❌ CRITICAL ERROR IN TRAINING PIPELINE:{C.RESET}")
        print(f"{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
