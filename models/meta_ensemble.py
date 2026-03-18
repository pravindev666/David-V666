"""
DAVID PROPHETIC ORACLE — Meta Ensemble
========================================
Combines Regime-Aware Ensemble (trees) and Sequence Model (LSTM)
Weights: 60% Trees, 40% LSTM
"""

import numpy as np
import pandas as pd
import os
import pickle

from .regime_ensemble import RegimeAwareEnsemble
from .sequence_model import SequenceModel

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MODEL_DIR, UP, DOWN, C

BINARY_MAP = {0: UP, 1: DOWN}

class MetaEnsemble:
    def __init__(self, regime_detector):
        self.regime_detector = regime_detector
        # Components
        self.regime_ensemble = RegimeAwareEnsemble(self.regime_detector)
        self.sequence_model = SequenceModel(seq_length=10)
        
        # Hardcoded blending based on backtests, or can be dynamic
        self.tree_weight = 0.60
        self.lstm_weight = 0.40
        self.is_trained = False
        
    def train(self, df, feature_cols, verbose=True):
        if verbose:
            print(f"\n{C.header('TRAINING META-ENSEMBLE FUSION (V6.6.6)')}")
            
        print(f"\n{C.CYAN}Step 1/2: Training Component -> Regime-Aware Binary Ensemble{C.RESET}")
        self.regime_ensemble.train(df, feature_cols, verbose=verbose)
        
        print(f"\n{C.CYAN}Step 2/2: Training Component -> LSTM Sequence Model{C.RESET}")
        self.sequence_model.train(df, feature_cols, verbose=verbose)
        
        self.is_trained = True
        
    def predict_today(self, df):
        """Matches the old EnsembleClassifier API."""
        return self.predict(df)
        
    def predict(self, df):
        """
        Pass the full dataframe to satisfy sequence model's window, 
        and the latest row to satisfy tree models.
        """
        if not self.is_trained:
            raise RuntimeError("Meta-Ensemble not trained!")
            
        tree_pred = self.regime_ensemble.predict(df.iloc[-1:]) 
        lstm_pred = self.sequence_model.predict(df)
        
        # Blend probabilities
        p_up = (tree_pred["prob_up"] * self.tree_weight) + (lstm_pred["prob_up"] * self.lstm_weight)
        p_down = (tree_pred["prob_down"] * self.tree_weight) + (lstm_pred["prob_down"] * self.lstm_weight)
        
        # Normalize
        total = p_up + p_down
        p_up /= total
        p_down /= total
        
        # Decide direction
        if p_up > p_down:
            direction = UP
            confidence = p_up
        else:
            direction = DOWN
            confidence = p_down
            
        return {
            "direction": direction,
            "confidence": confidence,
            "prob_up": p_up,
            "prob_down": p_down,
            "prob_sideways": 0.0,  # For backward-compatibility with display
            "tree_conf": max(tree_pred["prob_up"], tree_pred["prob_down"]),
            "lstm_conf": max(lstm_pred["prob_up"], lstm_pred["prob_down"])
        }

    def save(self):
        # We save sub-models using their own save methods to handle PyTorch state efficiently
        self.regime_ensemble.save()
        self.sequence_model.save()
        
        path = os.path.join(MODEL_DIR, "meta_ensemble.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "tree_weight": self.tree_weight,
                "lstm_weight": self.lstm_weight,
                "is_trained": self.is_trained
            }, f)
        print(f"  {C.GREEN}[SAVED] Meta-Ensemble Fusion → {path}{C.RESET}")
        
    def load(self):
        # First load sub-models
        reg_ok = self.regime_ensemble.load()
        lstm_ok = self.sequence_model.load()
        
        if not (reg_ok and lstm_ok):
            return False
            
        path = os.path.join(MODEL_DIR, "meta_ensemble.pkl")
        if not os.path.exists(path):
            return False
            
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.tree_weight = data["tree_weight"]
            self.lstm_weight = data["lstm_weight"]
            self.is_trained = data["is_trained"]
            
        print(f"  {C.GREEN}[LOADED] Meta-Ensemble Fusion from {path}{C.RESET}")
        return True
