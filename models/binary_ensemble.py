"""
DAVID PROPHETIC ORACLE — Binary Ensemble Classifier
=====================================================
Combines XGBoost, LightGBM, and CatBoost for Binary Classification (UP vs DOWN).
Excludes SIDEWAYS/NOISE data during training for sharper edge detection.
"""

import numpy as np
import pandas as pd
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Import ML algorithms safely
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MODEL_DIR, UP, DOWN, C

# Target map for binary
BINARY_MAP = {0: UP, 1: DOWN}

class BinaryEnsemble:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False
        self.validation_scores = {}
    
    def _build_models(self):
        models = {}
        
        if XGBClassifier is not None:
            models["XGBoost"] = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=5,
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=42,
                verbosity=0,
            )
        
        if LGBMClassifier is not None:
            models["LightGBM"] = LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_samples=20,
                objective="binary",
                metric="binary_logloss",
                random_state=42,
                verbose=-1,
            )
        
        if CatBoostClassifier is not None:
            models["CatBoost"] = CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                loss_function="Logloss",
                random_seed=42,
                verbose=0,
            )
            
        return models
    
    def train(self, df, feature_cols, verbose=True):
        self.feature_cols = feature_cols
        
        # FILTER OUT NOISE (target_binary == 2)
        train_df = df[df["target_binary"] != 2].copy()
        
        X = train_df[feature_cols].values
        y = train_df["target_binary"].values.astype(int)
        
        if verbose:
            print(f"\n{C.header('TRAINING BINARY ENSEMBLE (UP vs DOWN ONLY)')}")
            print(f"  Samples: {len(X)} (filtered noise) | Features: {len(feature_cols)} | Classes: 2")
            
        tscv = TimeSeriesSplit(n_splits=5)
        fold_scores = {name: [] for name in ["XGBoost", "LightGBM", "CatBoost"]}
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 🛡️ Skip if fold has only 1 class (XGB/LGB need 2)
            if len(np.unique(y_train)) < 2:
                continue

            fold_scaler = StandardScaler()
            X_train = fold_scaler.fit_transform(X[train_idx])
            X_test = fold_scaler.transform(X[test_idx])
            
            fold_models = self._build_models()
            
            for name, model in fold_models.items():
                model.fit(X_train, y_train)
                # ...
                y_pred = np.round(model.predict(X_test)).astype(int)
                y_test_clean = y_test.flatten().astype(int)
                acc = accuracy_score(y_test_clean, y_pred)
                fold_scores[name].append(acc)
            
            if verbose:
                accs = " | ".join(f"{n}: {fold_scores[n][-1]:.1%}" for n in fold_scores if len(fold_scores[n]) > 0)
                print(f"  Fold {fold+1}/5: {accs}")
                
        # Calculate weights based on CV performance
        for name in fold_scores:
            avg = np.mean(fold_scores[name]) if len(fold_scores[name]) > 0 else 0.5
            self.validation_scores[name] = max(avg, 0.01)
        
        total_score = sum(self.validation_scores.values())
        for name in self.validation_scores:
            self.weights[name] = self.validation_scores[name] / total_score
        
        # Train final models on full data
        X_scaled = self.scaler.fit_transform(X)
        
        # 🛡️ Final check: if full dataset has only 1 class
        if len(np.unique(y)) < 2:
            self.fixed_class = y[0]
            self.is_trained = True
            return 1.0

        self.fixed_class = None
        self.models = self._build_models()
        for name, model in self.models.items():
            model.fit(X_scaled, y)
        
        self.is_trained = True
        return np.mean([np.mean(v) for v in fold_scores.values()])

    def predict_proba(self, X_scaled):
        if hasattr(self, 'fixed_class') and self.fixed_class is not None:
            probs = np.zeros((X_scaled.shape[0], 2))
            probs[:, int(self.fixed_class)] = 1.0
            return probs
            
        combined_probs = np.zeros((X_scaled.shape[0], 2))
        for name, model in self.models.items():
            probs = model.predict_proba(X_scaled)
            combined_probs += probs * self.weights.get(name, 1.0 / len(self.models))
        return combined_probs / combined_probs.sum(axis=1, keepdims=True)

    def predict(self, X_row):
        if not self.is_trained:
            raise RuntimeError("Model not trained!")
        
        if isinstance(X_row, pd.Series):
            X_row = X_row[self.feature_cols].values.reshape(1, -1)
        elif isinstance(X_row, pd.DataFrame):
            X_row = X_row[self.feature_cols].values
            
        X_scaled = self.scaler.transform(X_row)
        combined_probs = self.predict_proba(X_scaled)
        
        results = []
        for i in range(len(combined_probs)):
            prob = combined_probs[i]
            pred_class = np.argmax(prob)
            
            results.append({
                "direction": BINARY_MAP[pred_class],
                "confidence": float(prob[pred_class]),
                "prob_up": float(prob[0]),
                "prob_down": float(prob[1]),
            })
            
        if len(results) == 1:
            return results[0]
        return results

