"""
DAVID PROPHETIC ORACLE — Iron Condor Analyzer
===============================================
Answers: "I have an iron condor at 25600 — will Nifty reach it?"
Calculates empirical probability of touching a strike price.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import C


class IronCondorAnalyzer:
    """
    Empirical probability calculator for option strikes.
    
    Given a strike price, calculates:
        1. Probability of Nifty touching that price in N days
        2. If touched, probability of bouncing back to current spot
        3. Max expected move in N days (probability cone)
        4. Firefight level (when to start emergency hedging)
    """
    
    def analyze_strike(self, df, strike_price, days=5):
        """
        Analyze the probability of Nifty touching a specific strike price.
        
        Args:
            df: Full historical DataFrame with 'close' column
            strike_price: The strike price to analyze (e.g., 25600)
            days: Timeframe for analysis (default 5 = weekly expiry)
        
        Returns:
            dict with touch_prob, recovery_prob, direction, firefight_level, etc.
        """
        current_price = float(df["close"].iloc[-1])
        distance_pct = (strike_price - current_price) / current_price
        direction = "UP" if strike_price > current_price else "DOWN"
        
        # ─── 1. Empirical Touch Probability ──────────────────────────────
        # Calculate all N-day HIGH and LOW extremes from history
        rolling_highs = []
        rolling_lows = []
        
        for i in range(len(df) - days):
            window = df.iloc[i : i + days]
            rolling_highs.append(float(window["high"].max()))
            rolling_lows.append(float(window["low"].min()))
        
        rolling_highs = np.array(rolling_highs)
        rolling_lows = np.array(rolling_lows)
        
        # What % of historical windows had their high above the strike (for upside)
        # or their low below the strike (for downside)?
        base_prices = df["close"].values[:len(rolling_highs)]
        
        if direction == "UP":
            # Will Nifty GO UP to the strike?
            # Calculate the % move needed and compare to historical % moves
            moves_up = (rolling_highs - base_prices) / base_prices
            required_move = abs(distance_pct)
            touch_prob = float(np.mean(moves_up >= required_move))
        else:
            # Will Nifty GO DOWN to the strike?
            moves_down = (base_prices - rolling_lows) / base_prices
            required_move = abs(distance_pct)
            touch_prob = float(np.mean(moves_down >= required_move))
        
        # ─── 2. Regime Adjustment ────────────────────────────────────────
        # In high volatility, large moves are more likely
        recent_vol = df["close"].pct_change().tail(20).std() * np.sqrt(252)
        hist_vol = df["close"].pct_change().std() * np.sqrt(252)
        vol_ratio = recent_vol / max(hist_vol, 0.01)
        
        if vol_ratio > 1.3:
            touch_prob *= 1.15  # High vol → bigger moves more likely
        elif vol_ratio < 0.7:
            touch_prob *= 0.85  # Low vol → smaller moves
        
        touch_prob = float(np.clip(touch_prob, 0.01, 0.99))
        
        # ─── 3. Recovery/Bounce-Back Probability ─────────────────────────
        # If touched, what's the chance of coming back to current spot?
        recovery_moves = []
        for i in range(len(df) - days * 2):
            window = df.iloc[i : i + days]
            post_window = df.iloc[i + days : i + days * 2]
            
            if direction == "UP":
                peak = float(window["high"].max())
                move_to_peak = (peak - df["close"].iloc[i]) / df["close"].iloc[i]
                if move_to_peak >= required_move and len(post_window) > 0:
                    recovery = float(post_window["low"].min())
                    recovered = recovery <= df["close"].iloc[i] * 1.005
                    recovery_moves.append(1 if recovered else 0)
            else:
                trough = float(window["low"].min())
                move_to_trough = (df["close"].iloc[i] - trough) / df["close"].iloc[i]
                if move_to_trough >= required_move and len(post_window) > 0:
                    recovery = float(post_window["high"].max())
                    recovered = recovery >= df["close"].iloc[i] * 0.995
                    recovery_moves.append(1 if recovered else 0)
        
        recovery_prob = float(np.mean(recovery_moves)) if recovery_moves else 0.5
        
        # ─── 4. Probability Cone (Max Move Distribution) ─────────────────
        if direction == "UP":
            max_moves = (rolling_highs - base_prices) / base_prices
        else:
            max_moves = (base_prices - rolling_lows) / base_prices
        
        p10 = float(np.percentile(max_moves, 10))
        p50 = float(np.percentile(max_moves, 50))
        p90 = float(np.percentile(max_moves, 90))
        
        # ─── 5. Firefight Level ──────────────────────────────────────────
        # At what price should you START emergency hedging?
        # Answer: when you're 70% of the way to the strike
        firefight_distance = abs(strike_price - current_price) * 0.70
        if direction == "UP":
            firefight_level = current_price + firefight_distance
        else:
            firefight_level = current_price - firefight_distance
        
        # Whipsaw level: where might it reverse AFTER hitting firefight?
        atr = float(df["high"].sub(df["low"]).rolling(14).mean().iloc[-1])
        whipsaw_bounce = atr * 1.0  # Expect 1 ATR bounce from firefight
        
        if direction == "UP":
            whipsaw_level = firefight_level - whipsaw_bounce
        else:
            whipsaw_level = firefight_level + whipsaw_bounce
        
        return {
            "strike": strike_price,
            "current_price": current_price,
            "distance_pct": round(distance_pct * 100, 2),
            "direction": direction,
            "touch_prob": round(touch_prob * 100, 1),
            "recovery_prob": round(recovery_prob * 100, 1),
            "firefight_level": round(firefight_level, 2),
            "whipsaw_level": round(whipsaw_level, 2),
            "max_move_p10": round(p10 * 100, 2),
            "max_move_p50": round(p50 * 100, 2),
            "max_move_p90": round(p90 * 100, 2),
            "days": days,
            "vol_ratio": round(vol_ratio, 2),
        }
    
    def format_analysis(self, result):
        """Pretty print iron condor analysis."""
        lines = []
        lines.append(f"\n  {C.header('IRON CONDOR STRIKE ANALYSIS')}")
        lines.append(f"  {'─'*55}")
        
        lines.append(f"\n  Strike Price:    {C.BOLD}{result['strike']:,.2f}{C.RESET}")
        lines.append(f"  Current Spot:    {result['current_price']:,.2f}")
        lines.append(f"  Distance:        {result['distance_pct']:+.2f}% ({result['direction']})")
        lines.append(f"  Timeframe:       {result['days']} trading days")
        lines.append(f"  Vol Regime:      {'HIGH' if result['vol_ratio'] > 1.2 else 'NORMAL' if result['vol_ratio'] > 0.8 else 'LOW'} ({result['vol_ratio']:.2f}x)")
        
        # Touch probability
        tp = result["touch_prob"]
        lines.append(f"\n  {C.highlight('PROBABILITIES')}")
        if tp > 60:
            lines.append(f"  Touch Probability:    {C.down(f'{tp:.0f}%')}  ← DANGER!")
        elif tp > 35:
            lines.append(f"  Touch Probability:    {C.neutral(f'{tp:.0f}%')}  ← Moderate risk")
        else:
            lines.append(f"  Touch Probability:    {C.up(f'{tp:.0f}%')}  ← Safe zone")
        
        rp = result["recovery_prob"]
        lines.append(f"  Recovery Probability: {C.pct(rp)} (if touched, chance of bounce back)")
        
        # Max move distribution
        # Max move distribution
        days = result["days"]
        lines.append(f"\n  {C.highlight(f'{days}-Day Max Move Distribution')}")
        lines.append(f"  10th Percentile:  {result['max_move_p10']:.2f}% (conservative)")
        lines.append(f"  50th Percentile:  {result['max_move_p50']:.2f}% (typical)")
        lines.append(f"  90th Percentile:  {result['max_move_p90']:.2f}% (extreme)")
        
        # Action levels
        lines.append(f"\n  {C.header('ACTION LEVELS')}")
        lines.append(f"  🔥 Firefight Level:  {C.YELLOW}{result['firefight_level']:,.0f}{C.RESET}  (START hedging here)")
        lines.append(f"  🔄 Whipsaw Level:    {C.CYAN}{result['whipsaw_level']:,.0f}{C.RESET}  (expect bounce here)")
        
        # Recommendation
        lines.append(f"\n  {C.highlight('RECOMMENDATION')}")
        if tp < 25:
            lines.append(f"  ✅ {C.up('SAFE')} — Your condor is well-positioned. Low touch risk.")
        elif tp < 50:
            lines.append(f"  ⚠️ {C.neutral('MONITOR')} — Keep watching. Hedge if it crosses firefight level.")
        else:
            lines.append(f"  🚨 {C.down('DANGER')} — High touch probability! Consider adjusting or closing the position.")
        
        return "\n".join(lines)
