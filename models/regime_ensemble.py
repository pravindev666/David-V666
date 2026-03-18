"""
DAVID PROPHETIC ORACLE — Regime-Aware Ensemble
================================================
Routes predictions to specialized BinaryEnsembles based on Market Regime.
TRENDING = Strong Bullish, Strong Bearish
CHOPPY = Mild Bullish, Mild Bearish, Sideways
"""

import numpy as np
import pandas as pd
import os
import pickle

from .binary_ensemble import BinaryEnsemble
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MODEL_DIR, C

class RegimeAwareEnsemble:
    """
    Routes predictions to specialized BinaryEnsembles based on Market Regime.
    """
    def __init__(self, regime_detector):
        self.regime_detector = regime_detector
        self.model_trending = BinaryEnsemble()
        self.model_choppy = BinaryEnsemble()
        self.is_trained = False
        self.feature_cols = None

    def _get_regime_group(self, state_label):
        if state_label in ["STRONG BULLISH", "STRONG BEARISH"]:
            return "TRENDING"
        return "CHOPPY"
        
    def train(self, df, feature_cols, verbose=True):
        self.feature_cols = feature_cols
        
        if verbose:
            print(f"\n{C.header('TRAINING REGIME-AWARE ENSEMBLE')}")
        
        if not self.regime_detector.is_trained:
            raise RuntimeError("RegimeDetector must be trained before RegimeAwareEnsemble.")
            
        hmm_vars = [f for f in self.regime_detector.hmm_features if f in df.columns]
        X_hmm = df[hmm_vars].values
        X_hmm_scaled = self.regime_detector.scaler.transform(X_hmm)
        states = self.regime_detector.hmm.predict(X_hmm_scaled)
        
        labels = [self.regime_detector.regime_map.get(s, "UNKNOWN") for s in states]
        groups = [self._get_regime_group(l) for l in labels]
        
        df_routed = df.copy()
        df_routed["regime_group"] = groups
        
        df_trending = df_routed[df_routed["regime_group"] == "TRENDING"].copy()
        df_choppy = df_routed[df_routed["regime_group"] == "CHOPPY"].copy()
        
        if verbose:
            print(f"  Routing Data -> TRENDING: {len(df_trending)} rows | CHOPPY: {len(df_choppy)} rows")
            
        print(f"\n  {C.CYAN}--- Training TRENDING Sub-Model ---{C.RESET}")
        self.model_trending.train(df_trending, feature_cols, verbose=verbose)
        
        print(f"\n  {C.CYAN}--- Training CHOPPY Sub-Model ---{C.RESET}")
        self.model_choppy.train(df_choppy, feature_cols, verbose=verbose)
        
        self.is_trained = True
        
    def predict(self, df):
        label, _, _ = self.regime_detector.get_current_regime(df)
        group = self._get_regime_group(label)
        
        # We only need the last row for the actual ML classifier
        df_row = df.iloc[-1:]
        
        if group == "TRENDING":
            res = self.model_trending.predict(df_row)
        else:
            res = self.model_choppy.predict(df_row)
        
        # Add regime to the result dict for the MetaEnsemble to see
        res["regime"] = label
        return res

    def save(self, path=None):
        if path is None:
            path = os.path.join(MODEL_DIR, "regime_ensemble.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "model_trending": self.model_trending,
                "model_choppy": self.model_choppy,
                "feature_cols": self.feature_cols,
                "is_trained": self.is_trained
            }, f)
        print(f"  {C.GREEN}[SAVED] Regime-Aware Ensemble → {path}{C.RESET}")
        
    def load(self, path=None):
        if path is None:
            path = os.path.join(MODEL_DIR, "regime_ensemble.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model_trending = data["model_trending"]
        self.model_choppy = data["model_choppy"]
        self.feature_cols = data["feature_cols"]
        self.is_trained = data["is_trained"]
        print(f"  {C.GREEN}[LOADED] Regime-Aware Ensemble from {path}{C.RESET}")
        return True
