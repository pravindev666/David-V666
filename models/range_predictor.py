"""
DAVID PROPHETIC ORACLE — Multi-Horizon Range Predictor
=======================================================
Predicts 7-day and 30-day price ranges using quantile regression.
Outputs confidence bands: "Nifty in 7 days: 24,800–25,400 (80% probability)"
"""

import numpy as np
import pandas as pd
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MODEL_DIR, C


class RangePredictor:
    """
    Quantile regression for multi-horizon price range prediction.
    
    Predicts:
        - 7-day range (10th, 25th, 50th, 75th, 90th percentile)
        - 30-day range (10th, 25th, 50th, 75th, 90th percentile)
    
    This tells you: "In 7 days, there's an 80% chance Nifty will be between X and Y."
    """
    
    HORIZONS = [7, 30]
    QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
    
    def __init__(self):
        self.models = {}  # {horizon: {quantile: model}}
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False
    
    def _build_quantile_model(self, quantile):
        """Build a single quantile regression model."""
        if LGBMRegressor is not None:
            return LGBMRegressor(
                objective="quantile",
                alpha=quantile,
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
        elif XGBRegressor is not None:
            return XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=quantile,
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0,
            )
        else:
            raise RuntimeError("Need lightgbm or xgboost for quantile regression")
    
    def train(self, df, feature_cols, verbose=True):
        """
        Train quantile regression models for each horizon and percentile.
        Target: percentage move from current price.
        """
        self.feature_cols = feature_cols
        X = df[feature_cols].values
        # NOTE: LightGBM/XGBoost trees are rank-invariant (scaling doesn't affect
        # predictions), but we keep the scaler for consistency with the inference path.
        X_scaled = self.scaler.fit_transform(X)
        
        if verbose:
            print(f"\n{C.header('TRAINING RANGE PREDICTOR')}")
        
        for horizon in self.HORIZONS:
            # Target = percentage price change over N days
            y = (df["close"].shift(-horizon) / df["close"] - 1).values
            
            # Drop NaN targets
            valid = ~np.isnan(y)
            X_h = X_scaled[valid]
            y_h = y[valid]
            
            self.models[horizon] = {}
            
            for q in self.QUANTILES:
                model = self._build_quantile_model(q)
                model.fit(X_h, y_h)
                self.models[horizon][q] = model
            
            if verbose:
                print(f"  {C.GREEN}[OK] {horizon}-day range: 5 quantile models trained{C.RESET}")
        
        self.is_trained = True
    
    def predict_range(self, df, current_price=None):
        """
        Predict price ranges for all horizons.
        
        Returns dict: {
            7: {"p10": 24100, "p25": 24300, "p50": 24500, "p75": 24700, "p90": 24900},
            30: {...}
        }
        """
        if not self.is_trained:
            raise RuntimeError("Range predictor not trained!")
        
        if current_price is None:
            current_price = float(df["close"].iloc[-1])
        
        latest = df[self.feature_cols].iloc[-1:].values
        latest_scaled = self.scaler.transform(latest)
        
        results = {}
        
        for horizon in self.HORIZONS:
            preds = {}
            for q in self.QUANTILES:
                pct_move = self.models[horizon][q].predict(latest_scaled)[0]
                price = current_price * (1 + pct_move)
                q_label = f"p{int(q*100)}"
                preds[q_label] = round(float(price), 2)
                preds[f"{q_label}_pct"] = round(float(pct_move * 100), 2)
            
            results[horizon] = preds
        
        return results
    
    def format_range(self, results, current_price):
        """Pretty print the range predictions."""
        lines = []
        
        for horizon in self.HORIZONS:
            r = results[horizon]
            lines.append(f"\n  {C.highlight(f'{horizon}-DAY FORECAST')} (from {current_price:,.2f})")
            lines.append(f"  {'─'*45}")
            
            # 80% confidence band (p10 to p90)
            low_80 = r["p10"]
            high_80 = r["p90"]
            lines.append(f"  80% Band:  {C.down(f'{low_80:,.0f}')} ─── {C.up(f'{high_80:,.0f}')}")
            
            # 50% confidence band (p25 to p75)
            low_50 = r["p25"]
            high_50 = r["p75"]
            lines.append(f"  50% Band:  {C.down(f'{low_50:,.0f}')} ─── {C.up(f'{high_50:,.0f}')}")
            
            # Median
            median = r["p50"]
            move_pct = r["p50_pct"]
            direction = "▲" if move_pct > 0 else "▼" if move_pct < 0 else "─"
            lines.append(f"  Median:    {median:,.0f} ({direction} {move_pct:+.2f}%)")
        
        return "\n".join(lines)
    
    def save(self, path=None):
        if path is None:
            path = os.path.join(MODEL_DIR, "range_predictor.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "models": self.models,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
            }, f)
        print(f"  {C.GREEN}[SAVED] Range predictor → {path}{C.RESET}")
    
    def load(self, path=None):
        if path is None:
            path = os.path.join(MODEL_DIR, "range_predictor.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.models = data["models"]
        self.scaler = data["scaler"]
        self.feature_cols = data["feature_cols"]
        self.is_trained = True
        print(f"  {C.GREEN}[LOADED] Range predictor from {path}{C.RESET}")
        return True
