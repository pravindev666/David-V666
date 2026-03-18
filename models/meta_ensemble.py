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
from .transformer_model import TransformerModel

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MODEL_DIR, UP, DOWN, C

BINARY_MAP = {0: UP, 1: DOWN}

class MetaEnsemble:
    def __init__(self, regime_detector):
        self.regime_detector = regime_detector
        # Components
        self.regime_ensemble = RegimeAwareEnsemble(self.regime_detector)
        self.sequence_model = SequenceModel(seq_length=20)
        self.attn_model = TransformerModel(seq_length=20)
        
        # V6.6.6+ Blending Weights
        # V6.6.6+ Plus Blending Weights
        self.tree_weight = 0.45 
        self.lstm_weight = 0.30
        self.attn_weight = 0.25
        self.is_trained = False
        
    def train(self, df, feature_cols, verbose=True):
        if verbose:
            print(f"\n{C.header('TRAINING V6.6.6+ META-ENSEMBLE (3 PILLARS)')}")
            
        print(f"\n{C.CYAN}Step 1/3: Regime-Aware Binary Ensemble (45%){C.RESET}")
        self.regime_ensemble.train(df, feature_cols, verbose=verbose)
        
        print(f"\n{C.CYAN}Step 2/3: LSTM Sequence Model (30%){C.RESET}")
        self.sequence_model.train(df, feature_cols, verbose=verbose)
        
        print(f"\n{C.CYAN}Step 3/3: Transformer Attention Model (25%){C.RESET}")
        self.attn_model.train(df, feature_cols, verbose=verbose)
        
        self.is_trained = True
        
    def predict_today(self, df):
        return self.predict(df)
        
    def predict(self, df):
        if not self.is_trained:
            raise RuntimeError("Meta-Ensemble not trained!")
            
        tree_p = self.regime_ensemble.predict(df) 
        lstm_p = self.sequence_model.predict(df)
        attn_p = self.attn_model.predict(df)
        
        # ─── HARD REGIME GATE (v6.6.6+ Sniper Patch) ───
        # If HMM identifies a 'STRONG' (Trauma) regime, refuse to trade.
        regime = tree_p.get("regime", "")
        if "STRONG" in regime.upper():
            return {
                "direction": "HOLD",
                "confidence": 0.0,
                "prob_up": 0.5,
                "prob_down": 0.5,
                "regime": regime,
                "reason": "Hard Gate: Trauma Regime Detected",
                "tree_conf": max(tree_p["prob_up"], tree_p["prob_down"]),
                "lstm_conf": max(lstm_p["prob_up"], lstm_p["prob_down"]),
                "attn_conf": max(attn_p["prob_up"], attn_p["prob_down"])
            }

        # Blend probabilities
        p_up = (tree_p["prob_up"] * self.tree_weight) + \
               (lstm_p["prob_up"] * self.lstm_weight) + \
               (attn_p["prob_up"] * self.attn_weight)
               
        p_down = (tree_p["prob_down"] * self.tree_weight) + \
                 (lstm_p["prob_down"] * self.lstm_weight) + \
                 (attn_p["prob_down"] * self.attn_weight)
        
        # Normalize
        total = p_up + p_down
        p_up /= (total if total > 0 else 1)
        p_down /= (total if total > 0 else 1)
        
        # Decide direction
        direction = UP if p_up > p_down else DOWN
        confidence = max(p_up, p_down)
            
        # Extract Whipsaw Score from current dataframe
        w_score = df["whipsaw_score"].iloc[-1] if "whipsaw_score" in df.columns else 0
        w_lag1 = df["whipsaw_score_lag1"].iloc[-1] if "whipsaw_score_lag1" in df.columns else 0
        
        # Determine Whipsaw Label
        if w_score < 30: w_label = "SMOOTH"
        elif w_score < 60: w_label = "BUMPY"
        else: w_label = "STORM"
            
        return {
            "direction": direction,
            "confidence": confidence,
            "prob_up": p_up,
            "prob_down": p_down,
            "regime": regime,
            "whipsaw_score": w_score,
            "whipsaw_lag": w_lag1,
            "whipsaw_label": w_label,
            "tree_conf": max(tree_p["prob_up"], tree_p["prob_down"]),
            "lstm_conf": max(lstm_p["prob_up"], lstm_p["prob_down"]),
            "attn_conf": max(attn_p["prob_up"], attn_p["prob_down"])
        }

    def save(self):
        self.regime_ensemble.save()
        self.sequence_model.save()
        self.attn_model.save()
        
        path = os.path.join(MODEL_DIR, "meta_ensemble.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "tree_weight": self.tree_weight,
                "lstm_weight": self.lstm_weight,
                "attn_weight": self.attn_weight,
                "is_trained": self.is_trained
            }, f)
        print(f"  {C.GREEN}[SAVED] V6.6.6+ Meta-Ensemble → {path}{C.RESET}")
        
    def load(self):
        reg_ok = self.regime_ensemble.load()
        lstm_ok = self.sequence_model.load()
        attn_ok = self.attn_model.load()
        
        if not (reg_ok and lstm_ok and attn_ok):
            return False
            
        path = os.path.join(MODEL_DIR, "meta_ensemble.pkl")
        if not os.path.exists(path): return False
            
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.tree_weight = data["tree_weight"]
            self.lstm_weight = data["lstm_weight"]
            self.attn_weight = data.get("attn_weight", 0.20)
            self.is_trained = data["is_trained"]
            
        print(f"  {C.GREEN}[LOADED] V6.6.6+ Meta-Ensemble from {path}{C.RESET}")
        return True
