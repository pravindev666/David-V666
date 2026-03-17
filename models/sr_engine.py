"""
DAVID PROPHETIC ORACLE — Support/Resistance Engine
====================================================
Fractal pivot detection + DBSCAN clustering for data-driven S/R levels.
No synthetic scanning — uses ACTUAL historical price pivots.
"""

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import C


class SREngine:
    """
    Support / Resistance Engine using Williams Fractals + Clustering.
    
    Method:
        1. Detect swing highs/lows using 5-bar Williams fractal patterns
        2. Cluster nearby pivots using DBSCAN
        3. Weight by recency and number of touches
        4. Return top 3 support + top 3 resistance levels with strength
    """
    
    def __init__(self, fractal_window=2, lookback_days=252):
        self.fractal_window = fractal_window  # Bars on each side for fractal
        self.lookback_days = lookback_days     # How far back to look (1 year default)
    
    def _detect_fractals(self, df):
        """
        Detect Williams Fractal swing highs and swing lows.
        A fractal high = high is higher than `window` bars on both sides.
        A fractal low = low is lower than `window` bars on both sides.
        """
        w = self.fractal_window
        highs = df["high"].values
        lows = df["low"].values
        dates = df["date"].values
        
        swing_highs = []
        swing_lows = []
        
        for i in range(w, len(df) - w):
            # Fractal High?
            is_high = True
            for j in range(1, w + 1):
                if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                    is_high = False
                    break
            
            if is_high:
                swing_highs.append({
                    "price": float(highs[i]),
                    "date": dates[i],
                    "idx": i,
                })
            
            # Fractal Low?
            is_low = True
            for j in range(1, w + 1):
                if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                    is_low = False
                    break
            
            if is_low:
                swing_lows.append({
                    "price": float(lows[i]),
                    "date": dates[i],
                    "idx": i,
                })
        
        return swing_highs, swing_lows
    
    def _cluster_levels(self, pivots, current_price):
        """
        Cluster nearby pivot levels using DBSCAN.
        Returns clustered S/R levels with strength score.
        """
        if not pivots or len(pivots) < 2:
            return []
        
        prices = np.array([p["price"] for p in pivots]).reshape(-1, 1)
        indices = np.array([p["idx"] for p in pivots])
        max_idx = max(indices) if len(indices) > 0 else 1
        
        # DBSCAN clustering — eps as % of price
        eps = current_price * 0.005  # 0.5% cluster radius
        
        if DBSCAN is not None:
            clustering = DBSCAN(eps=eps, min_samples=1).fit(prices)
            labels = clustering.labels_
        else:
            # Fallback: simple binning
            bins = np.arange(prices.min(), prices.max() + eps, eps)
            labels = np.digitize(prices.flatten(), bins)
        
        # Aggregate clusters
        levels = []
        for label in set(labels):
            if label == -1:
                continue
            mask = labels == label
            cluster_prices = prices[mask].flatten()
            cluster_indices = indices[mask]
            
            avg_price = float(np.mean(cluster_prices))
            touches = int(np.sum(mask))
            
            # Recency weight: more recent pivots are more important
            recency = float(np.mean(cluster_indices / max_idx))
            
            # Strength = touches * recency
            strength = touches * (0.5 + 0.5 * recency)
            
            levels.append({
                "price": round(avg_price, 2),
                "touches": touches,
                "recency": round(recency, 2),
                "strength": round(strength, 2),
            })
        
        # Sort by strength
        levels.sort(key=lambda x: x["strength"], reverse=True)
        return levels
    
    def find_levels(self, df, n_levels=3):
        """
        Main method: find the top N support and resistance levels.
        
        Returns:
            supports: List of {price, touches, strength} dicts
            resistances: List of {price, touches, strength} dicts
        """
        # Use only recent data
        recent = df.tail(self.lookback_days).copy()
        current_price = float(df["close"].iloc[-1])
        
        # Detect fractals
        swing_highs, swing_lows = self._detect_fractals(recent)
        
        # Cluster
        all_resistance_levels = self._cluster_levels(swing_highs, current_price)
        all_support_levels = self._cluster_levels(swing_lows, current_price)
        
        # Filter: supports must be BELOW current, resistances ABOVE
        supports = [l for l in all_support_levels if l["price"] < current_price]
        resistances = [l for l in all_resistance_levels if l["price"] > current_price]
        
        # Also check if any swing lows are above (they become resistance) and vice versa
        extra_resistance = [l for l in all_support_levels if l["price"] > current_price]
        extra_support = [l for l in all_resistance_levels if l["price"] < current_price]
        
        resistances.extend(extra_resistance)
        supports.extend(extra_support)
        
        # Re-sort by strength
        supports.sort(key=lambda x: x["strength"], reverse=True)
        resistances.sort(key=lambda x: x["strength"], reverse=True)
        
        # ATR-based fallback if not enough levels found
        if len(supports) < n_levels or len(resistances) < n_levels:
            atr = recent["high"].sub(recent["low"]).rolling(14).mean().iloc[-1]
            if np.isnan(atr):
                atr = current_price * 0.01
            
            for i in range(n_levels - len(supports)):
                supports.append({
                    "price": round(current_price - atr * (i + 1) * 1.5, 2),
                    "touches": 0,
                    "recency": 0,
                    "strength": 0.1,
                })
            
            for i in range(n_levels - len(resistances)):
                resistances.append({
                    "price": round(current_price + atr * (i + 1) * 1.5, 2),
                    "touches": 0,
                    "recency": 0,
                    "strength": 0.1,
                })
        
        return supports[:n_levels], resistances[:n_levels]
    
    def format_levels(self, supports, resistances, current_price):
        """Pretty print S/R levels."""
        lines = []
        lines.append(f"\n  {C.highlight('SUPPORT & RESISTANCE LEVELS')}")
        lines.append(f"  Current Price: {C.BOLD}{current_price:,.2f}{C.RESET}")
        lines.append(f"  {'─'*50}")
        
        lines.append(f"\n  {C.up('▲ RESISTANCE (above)')}")
        for i, r in enumerate(resistances):
            dist = ((r['price'] - current_price) / current_price) * 100
            bar = "█" * min(20, max(1, int(r["strength"] * 3)))
            lines.append(f"    R{i+1}: {r['price']:>10,.2f}  (+{dist:.1f}%)  {C.RED}{bar}{C.RESET}  [{r['touches']} touches]")
        
        lines.append(f"\n  {C.YELLOW}── {current_price:,.2f} (SPOT) ──{C.RESET}")
        
        lines.append(f"\n  {C.down('▼ SUPPORT (below)')}")
        for i, s in enumerate(supports):
            dist = ((current_price - s['price']) / current_price) * 100
            bar = "█" * min(20, max(1, int(s["strength"] * 3)))
            lines.append(f"    S{i+1}: {s['price']:>10,.2f}  (-{dist:.1f}%)  {C.GREEN}{bar}{C.RESET}  [{s['touches']} touches]")
        
        return "\n".join(lines)
