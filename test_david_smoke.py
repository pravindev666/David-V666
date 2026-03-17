
import os
import sys

# Ensure imports work from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_engine import load_all_data
from feature_forge import engineer_features
from models.ensemble_classifier import EnsembleClassifier
from models.regime_detector import RegimeDetector
from models.range_predictor import RangePredictor
from models.sr_engine import SREngine
from analyzers.whipsaw_detector import WhipsawDetector

def test_system():
    print("Testing David Oracle System...")
    
    # 1. Load Data
    print("Loading data...")
    df_raw = load_all_data()
    print(f"Data loaded: {len(df_raw)} rows")
    
    # 2. Engineer Features
    print("Engineering features...")
    df, features = engineer_features(df_raw)
    print(f"Features engineered: {len(features)} features")
    
    # 3. Initialize Models
    print("Initializing models...")
    ensemble = EnsembleClassifier()
    regime = RegimeDetector()
    range_pred = RangePredictor()
    sr = SREngine()
    whipsaw = WhipsawDetector()
    
    # 4. Train logic (mock training on small subset if needed, or just check instantiation)
    # We won't trigger full training here to save time, unless necessary.
    # Just checking instantiation is good enough for a smoke test if dependencies are fine.
    
    print("All systems initialized successfully.")

if __name__ == "__main__":
    test_system()
