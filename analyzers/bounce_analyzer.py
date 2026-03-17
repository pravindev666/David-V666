"""
DAVID PROPHETIC ORACLE — Bounce-Back Analyzer
===============================================
Given a target price, calculates the empirical probability of Nifty
recovering to the current spot level within a given timeframe.
Regime-aware: adjusts probabilities based on current market conditions.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import C


class BounceAnalyzer:
    """
    Recovery / Bounce-Back Probability Calculator.
    
    Question: "If Nifty drops to X, will it come back to current levels?"
    Answer: Empirical probability based on 10 years of data, adjusted for current regime.
    """
    
    def analyze(self, df, target_price, days_list=[5, 10, 20]):
        """
        Calculate bounce-back probability for multiple timeframes.
        
        Args:
            df: Full historical DataFrame
            target_price: Price to bounce back FROM
            days_list: List of timeframes to check recovery in
        
        Returns:
            dict with recovery probabilities for each timeframe
        """
        current_price = float(df["close"].iloc[-1])
        distance_pct = (target_price - current_price) / current_price
        direction = "needs RALLY" if target_price < current_price else "needs DIP"
        
        results = {
            "target_price": target_price,
            "current_price": current_price,
            "distance_pct": round(distance_pct * 100, 2),
            "direction": direction,
            "timeframes": {},
        }
        
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        
        for days in days_list:
            recoveries = 0
            total_scenarios = 0
            recovery_times = []
            
            for i in range(len(df) - days):
                start_price = closes[i]
                required_move_pct = abs(distance_pct)
                
                # Simulate: did the market make a similar move and then recover?
                if target_price < current_price:
                    # Target is BELOW → we need to see a DROP then RECOVERY
                    window_low = min(lows[i : i + days])
                    actual_drop = (start_price - window_low) / start_price
                    
                    if actual_drop >= required_move_pct:
                        total_scenarios += 1
                        # Did it recover to start_price within another N days?
                        if i + days + days <= len(df):
                            post_window_high = max(highs[i + days : i + days + days])
                            if post_window_high >= start_price * 0.997:
                                recoveries += 1
                                # Find approximate recovery time
                                for j in range(i + days, min(i + days + days, len(df))):
                                    if highs[j] >= start_price * 0.997:
                                        recovery_times.append(j - (i + days))
                                        break
                else:
                    # Target is ABOVE → we need to see a RALLY then PULLBACK
                    window_high = max(highs[i : i + days])
                    actual_rally = (window_high - start_price) / start_price
                    
                    if actual_rally >= required_move_pct:
                        total_scenarios += 1
                        if i + days + days <= len(df):
                            post_window_low = min(lows[i + days : i + days + days])
                            if post_window_low <= start_price * 1.003:
                                recoveries += 1
                                for j in range(i + days, min(i + days + days, len(df))):
                                    if lows[j] <= start_price * 1.003:
                                        recovery_times.append(j - (i + days))
                                        break
            
            if total_scenarios > 0:
                recovery_prob = recoveries / total_scenarios
                avg_recovery_time = float(np.mean(recovery_times)) if recovery_times else days
            else:
                recovery_prob = 0.5  # Unknown → 50/50
                avg_recovery_time = days
            
            # Regime adjustment
            recent_vol = df["close"].pct_change().tail(20).std() * np.sqrt(252)
            hist_vol = df["close"].pct_change().std() * np.sqrt(252)
            vol_ratio = recent_vol / max(hist_vol, 0.01)
            
            # High vol → recovery more likely (mean reversion stronger)
            if vol_ratio > 1.3:
                recovery_prob *= 1.1
            elif vol_ratio < 0.7:
                recovery_prob *= 0.9
            
            recovery_prob = float(np.clip(recovery_prob, 0.01, 0.99))
            
            results["timeframes"][days] = {
                "recovery_prob": round(recovery_prob * 100, 1),
                "scenarios_found": total_scenarios,
                "avg_recovery_days": round(avg_recovery_time, 1),
            }
        
        return results
    
    def format_analysis(self, result):
        """Pretty print bounce analysis."""
        lines = []
        lines.append(f"\n  {C.header('BOUNCE-BACK PROBABILITY ANALYSIS')}")
        lines.append(f"  {'─'*55}")
        
        lines.append(f"\n  Target Price:  {C.BOLD}{result['target_price']:,.2f}{C.RESET}")
        lines.append(f"  Current Spot:  {result['current_price']:,.2f}")
        lines.append(f"  Distance:      {result['distance_pct']:+.2f}% ({result['direction']})")
        
        lines.append(f"\n  {'Timeframe':>15}  {'Recovery %':>12}  {'Avg Days':>10}  {'Scenarios':>10}")
        lines.append(f"  {'─'*55}")
        
        for days, data in result["timeframes"].items():
            prob = data["recovery_prob"]
            prob_str = C.pct(prob)
            lines.append(
                f"  {days:>12} days  {prob_str:>22}  {data['avg_recovery_days']:>8.1f}d  {data['scenarios_found']:>10}"
            )
        
        # Overall recommendation
        best_tf = max(result["timeframes"].items(), key=lambda x: x[1]["recovery_prob"])
        best_prob = best_tf[1]["recovery_prob"]
        
        lines.append(f"\n  {C.highlight('VERDICT')}")
        if best_prob > 70:
            lines.append(f"  ✅ {C.up('HIGH RECOVERY CHANCE')} — Historical data supports bounce-back.")
            lines.append(f"     Best window: {best_tf[0]} days ({best_prob:.0f}% probability)")
        elif best_prob > 45:
            lines.append(f"  ⚠️ {C.neutral('MODERATE')} — Recovery is possible but not guaranteed.")
            lines.append(f"     Best window: {best_tf[0]} days ({best_prob:.0f}% probability)")
        else:
            lines.append(f"  🚨 {C.down('LOW RECOVERY')} — Move may be structural. Consider cutting losses.")
            lines.append(f"     Best window: {best_tf[0]} days ({best_prob:.0f}% probability)")
        
        return "\n".join(lines)
