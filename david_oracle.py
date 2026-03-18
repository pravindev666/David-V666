"""
╔══════════════════════════════════════════════════════════════════╗
║              DAVID PROPHETIC ORACLE v6.6.6                       ║
║         Nifty Absolute Direction Prediction Engine               ║
║    Regime-Aware Meta-Ensemble (Trees + PyTorch LSTM)             ║
╚══════════════════════════════════════════════════════════════════╝

Main interactive CLI application.
Run: python david_oracle.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Ensure imports work from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import C, banner, separator, format_inr, UP, DOWN, SIDEWAYS
from data_engine import load_all_data
from feature_forge import engineer_features, get_target_distribution
from models.meta_ensemble import MetaEnsemble
from models.regime_detector import RegimeDetector
from models.range_predictor import RangePredictor
from models.sr_engine import SREngine
from analyzers.whipsaw_detector import WhipsawDetector
from analyzers.iron_condor_analyzer import IronCondorAnalyzer
from analyzers.bounce_analyzer import BounceAnalyzer


class DavidOracle:
    """
    The David Prophetic Oracle — Master orchestrator.
    Wires together all ML models and analyzers into a single interactive CLI.
    """
    
    def __init__(self):
        self.df_raw = None
        self.df = None
        self.feature_cols = None
        self.current_price = None
        self.vix = None
        
        # Models
        self.regime = RegimeDetector()
        self.ensemble = MetaEnsemble(self.regime)
        self.range_pred = RangePredictor()
        self.sr_engine = SREngine()
        
        # Analyzers
        self.whipsaw = WhipsawDetector()
        self.condor = IronCondorAnalyzer()
        self.bounce = BounceAnalyzer()
        
        self.is_initialized = False
    
    def initialize(self, force_retrain=False):
        """
        Initialize the oracle: load data, engineer features, train/load models.
        """
        banner()
        
        # 1. Load data
        self.df_raw = load_all_data()
        
        # 2. Engineer features
        separator("FEATURE ENGINEERING")
        self.df, self.feature_cols = engineer_features(self.df_raw)
        
        self.current_price = float(self.df["close"].iloc[-1])
        self.vix = float(self.df_raw["vix"].iloc[-1]) if "vix" in self.df_raw.columns else 15.0
        
        # 3. Train or load models
        separator("MODEL TRAINING")
        
        if not force_retrain and self.regime.load():
            print(f"  {C.GREEN}Regime detector loaded from cache{C.RESET}")
        else:
            self.regime.train(self.df)
            self.regime.save()
            
        if not force_retrain and self.ensemble.load():
            print(f"  {C.GREEN}Ensemble loaded from cache{C.RESET}")
        else:
            self.ensemble.train(self.df, self.feature_cols)
            self.ensemble.save()
        
        if not force_retrain and self.range_pred.load():
            print(f"  {C.GREEN}Range predictor loaded from cache{C.RESET}")
        else:
            self.range_pred.train(self.df, self.feature_cols)
            self.range_pred.save()
        
        self.is_initialized = True
        
        separator()
        print(f"  {C.GREEN}{C.BOLD}✅ DAVID ORACLE READY{C.RESET}")
        print(f"  NIFTY SPOT:  {C.BOLD}{self.current_price:,.2f}{C.RESET}")
        print(f"  VIX:         {self.vix:.1f}")
        
        # Quick regime check
        regime_info = self.regime.get_current_regime(self.df)
        print(f"  REGIME:      {C.highlight(regime_info[0])}")
        separator()
    
    def show_menu(self):
        """Display the interactive menu."""
        print(f"""
  {C.BOLD}NIFTY: {self.current_price:,.2f}   VIX: {self.vix:.1f}{C.RESET}

  {C.CYAN}[1]{C.RESET} 📊 TODAY'S VERDICT (Direction + Confidence)
  {C.CYAN}[2]{C.RESET} 📈 7-DAY FORECAST (Range + Direction)
  {C.CYAN}[3]{C.RESET} 📅 30-DAY FORECAST (Range + Direction)
  {C.CYAN}[4]{C.RESET} 🔴 SUPPORT / RESISTANCE LEVELS
  {C.CYAN}[5]{C.RESET} ⚡ FLIP & WHIPSAW ANALYSIS
  {C.CYAN}[6]{C.RESET} 🛡️  IRON CONDOR ANALYZER (Enter your strikes)
  {C.CYAN}[7]{C.RESET} 🔄 BOUNCE-BACK PROBABILITY (Enter target price)
  {C.CYAN}[8]{C.RESET} 🎯 TRADE RECOMMENDATION (Bull / Bear / Iron Condor)
  {C.CYAN}[9]{C.RESET} 🔧 RETRAIN ALL MODELS (Fresh training)
  {C.CYAN}[B]{C.RESET} 📋 RUN OUT-OF-SAMPLE BACKTEST
  {C.CYAN}[F]{C.RESET} 📊 TOP FEATURES (What drives predictions?)
  {C.CYAN}[0]{C.RESET} ❌ EXIT
""")
    
    # ═══════════════════════════════════════════════════════════════════════
    # MENU HANDLERS
    # ═══════════════════════════════════════════════════════════════════════
    
    def option_1_verdict(self):
        """Today's Direction Verdict."""
        separator("TODAY'S VERDICT")
        
        pred = self.ensemble.predict_today(self.df)
        regime_info = self.regime.get_regime_with_micro_direction(self.df, pred)
        
        # Direction
        direction = pred["direction"]
        confidence = pred["confidence"] * 100
        
        print(f"\n  {C.BOLD}═══ ABSOLUTE DIRECTION ═══{C.RESET}")
        print(f"  Direction:   {C.direction_color(direction)}")
        print(f"  Confidence:  {C.pct(confidence)}")
        
        print(f"\n  {C.BOLD}Probability Breakdown:{C.RESET}")
        print(f"    UP:       {C.pct(pred['prob_up']*100)}")
        print(f"    DOWN:     {C.pct(pred['prob_down']*100)}")
        
        print(f"\n  {C.DIM}Sub-Model Confidences:{C.RESET}")
        print(f"    Tree Ensemble: {C.pct(pred.get('tree_conf', 0.5)*100)}")
        print(f"    LSTM Sequence: {C.pct(pred.get('lstm_conf', 0.5)*100)}")
        
        # Regime context
        print(f"\n  {C.BOLD}═══ REGIME CONTEXT ═══{C.RESET}")
        print(f"  Market Regime:    {C.highlight(regime_info['regime'])}")
        print(f"  Micro-Direction:  {regime_info['micro_direction']} ({regime_info['micro_pct']:.1f}%)")
        
        # Transition probabilities
        if regime_info.get("transition_probs"):
            print(f"\n  {C.DIM}Next-Day Regime Transition Probabilities:{C.RESET}")
            for label, prob in sorted(regime_info["transition_probs"].items(), key=lambda x: -x[1]):
                bar = "█" * int(prob * 30)
                print(f"    {label:>16}: {prob*100:5.1f}% {C.DIM}{bar}{C.RESET}")
        
        # Confidence bucket
        if confidence >= 65:
            print(f"\n  {C.up('★ HIGH CONVICTION')} → Full position sizing")
        elif confidence >= 45:
            print(f"\n  {C.neutral('◆ MODERATE CONVICTION')} → Half position sizing")
        else:
            print(f"\n  {C.down('○ LOW CONVICTION')} → Skip or minimal position")
    
    def option_2_7day(self):
        """7-Day Range Forecast."""
        separator("7-DAY FORECAST")
        
        pred = self.ensemble.predict_today(self.df)
        ranges = self.range_pred.predict_range(self.df, self.current_price)
        
        direction = pred["direction"]
        print(f"\n  Direction Lean: {C.direction_color(direction)} ({pred['confidence']*100:.0f}% confidence)")
        
        if 7 in ranges:
            r = ranges[7]
            p90 = r["p90"]
            p75 = r["p75"]
            p50 = r["p50"]
            p25 = r["p25"]
            p10 = r["p10"]

            print(f"\n  {C.highlight('7-DAY PRICE RANGE')}")
            print(f"  {'─'*45}")
            print(f"  90th Percentile (extreme high):  {C.up(f'{p90:,.0f}')}")
            print(f"  75th Percentile (likely high):   {C.up(f'{p75:,.0f}')}")
            print(f"  50th Percentile (median):        {C.BOLD}{p50:,.0f}{C.RESET}")
            print(f"  25th Percentile (likely low):    {C.down(f'{p25:,.0f}')}")
            print(f"  10th Percentile (extreme low):   {C.down(f'{p10:,.0f}')}")
            
            print(f"\n  {C.highlight('INTERPRETATION')}")
            print(f"  80% chance Nifty stays between: {C.BOLD}{r['p10']:,.0f} — {r['p90']:,.0f}{C.RESET}")
            print(f"  50% chance Nifty stays between: {C.BOLD}{r['p25']:,.0f} — {r['p75']:,.0f}{C.RESET}")
    
    def option_3_30day(self):
        """30-Day Range Forecast."""
        separator("30-DAY FORECAST")
        
        pred = self.ensemble.predict_today(self.df)
        ranges = self.range_pred.predict_range(self.df, self.current_price)
        
        direction = pred["direction"]
        print(f"\n  Direction Lean: {C.direction_color(direction)} ({pred['confidence']*100:.0f}% confidence)")
        
        if 30 in ranges:
            r = ranges[30]
            p90 = r["p90"]
            p75 = r["p75"]
            p50 = r["p50"]
            p25 = r["p25"]
            p10 = r["p10"]

            print(f"\n  {C.highlight('30-DAY PRICE RANGE')}")
            print(f"  {'─'*45}")
            print(f"  90th Percentile (extreme high):  {C.up(f'{p90:,.0f}')}")
            print(f"  75th Percentile (likely high):   {C.up(f'{p75:,.0f}')}")
            print(f"  50th Percentile (median):        {C.BOLD}{p50:,.0f}{C.RESET}")
            print(f"  25th Percentile (likely low):    {C.down(f'{p25:,.0f}')}")
            print(f"  10th Percentile (extreme low):   {C.down(f'{p10:,.0f}')}")
            
            print(f"\n  {C.highlight('INTERPRETATION')}")
            print(f"  80% chance Nifty in 30 days: {C.BOLD}{r['p10']:,.0f} — {r['p90']:,.0f}{C.RESET}")
            print(f"  50% chance Nifty in 30 days: {C.BOLD}{r['p25']:,.0f} — {r['p75']:,.0f}{C.RESET}")
    
    def option_4_sr(self):
        """Support / Resistance Levels."""
        separator("SUPPORT & RESISTANCE")
        supports, resistances = self.sr_engine.find_levels(self.df_raw)
        output = self.sr_engine.format_levels(supports, resistances, self.current_price)
        print(output)
    
    def option_5_whipsaw(self):
        """Whipsaw & Flip Analysis."""
        separator("WHIPSAW & FLIP ANALYSIS")
        result = self.whipsaw.analyze(self.df)
        output = self.whipsaw.format_analysis(result, self.current_price)
        print(output)
    
    def option_6_condor(self):
        """Iron Condor Analyzer."""
        separator("IRON CONDOR ANALYZER")
        
        try:
            strike_input = input(f"\n  {C.CYAN}Enter your strike price (e.g. 25600): {C.RESET}")
            strike = float(strike_input.strip())
        except (ValueError, EOFError):
            print(f"  {C.RED}Invalid price. Please enter a number.{C.RESET}")
            return
        
        try:
            days_input = input(f"  {C.CYAN}Enter timeframe in days (default 5): {C.RESET}").strip()
            days = int(days_input) if days_input else 5
        except (ValueError, EOFError):
            days = 5
        
        result = self.condor.analyze_strike(self.df_raw, strike, days)
        output = self.condor.format_analysis(result)
        print(output)
    
    def option_7_bounce(self):
        """Bounce-Back Probability."""
        separator("BOUNCE-BACK PROBABILITY")
        
        try:
            target_input = input(f"\n  {C.CYAN}Enter target price to bounce FROM (e.g. 23000): {C.RESET}")
            target = float(target_input.strip())
        except (ValueError, EOFError):
            print(f"  {C.RED}Invalid price. Please enter a number.{C.RESET}")
            return
        
        result = self.bounce.analyze(self.df_raw, target)
        output = self.bounce.format_analysis(result)
        print(output)
    
    def option_8_trade(self):
        """Trade Recommendation."""
        separator("TRADE RECOMMENDATION")
        
        pred = self.ensemble.predict_today(self.df)
        regime_label, _, _ = self.regime.get_current_regime(self.df)
        supports, resistances = self.sr_engine.find_levels(self.df_raw)
        whipsaw_data = self.whipsaw.analyze(self.df)
        
        direction = pred["direction"]
        confidence = pred["confidence"] * 100
        
        spot = self.current_price
        atr = float(self.df["atr_14"].iloc[-1]) if "atr_14" in self.df.columns else spot * 0.01
        
        print(f"\n  {C.BOLD}═══ TRADE SETUP ═══{C.RESET}")
        print(f"  Direction:  {C.direction_color(direction)}  |  Confidence: {C.pct(confidence)}")
        print(f"  Regime:     {C.highlight(regime_label)}")
        print(f"  Whipsaw:    {'⚠️ ACTIVE' if whipsaw_data['is_choppy'] else '✅ Clear'}")
        
        # Strategy selection
        if whipsaw_data["is_choppy"] and confidence < 50:
            strategy = "NO TRADE"
            print(f"\n  Strategy:   {C.RED}{C.BOLD}NO TRADE — Market is choppy with low confidence{C.RESET}")
            print(f"  Reasoning:  Whipsaw probability is {whipsaw_data['whipsaw_prob']:.0f}% and confidence is only {confidence:.0f}%")
            print(f"  Action:     WAIT for clarity. Sit on hands.")
            
        elif direction == UP:
            strategy = "BULL CALL SPREAD"
            buy_strike = round((spot - atr * 0.5) / 50) * 50
            sell_strike = round((spot + atr * 1.5) / 50) * 50
            
            print(f"\n  Strategy:   {C.up(f'BULL CALL SPREAD')}")
            print(f"  BUY:        {buy_strike:,.0f} CE (At-the-Money)")
            print(f"  SELL:       {sell_strike:,.0f} CE (Out-of-Money)")
            print(f"  Max Risk:   Premium paid (debit spread)")
            print(f"  Target:     S/R resistance at {resistances[0]['price']:,.0f}" if resistances else "")
            print(f"  Stop:       Below support at {supports[0]['price']:,.0f}" if supports else "")
            
        elif direction == DOWN:
            strategy = "BEAR PUT SPREAD"
            buy_strike = round((spot + atr * 0.5) / 50) * 50
            sell_strike = round((spot - atr * 1.5) / 50) * 50
            
            print(f"\n  Strategy:   {C.down(f'BEAR PUT SPREAD')}")
            print(f"  BUY:        {buy_strike:,.0f} PE (At-the-Money)")
            print(f"  SELL:       {sell_strike:,.0f} PE (Out-of-Money)")
            print(f"  Max Risk:   Premium paid (debit spread)")
            print(f"  Target:     S/R support at {supports[0]['price']:,.0f}" if supports else "")
            print(f"  Stop:       Above resistance at {resistances[0]['price']:,.0f}" if resistances else "")
            
        else:  # SIDEWAYS
            strategy = "IRON CONDOR"
            call_sell = round((spot + atr * 2) / 50) * 50
            call_buy = call_sell + 100
            put_sell = round((spot - atr * 2) / 50) * 50
            put_buy = put_sell - 100
            
            print(f"\n  Strategy:   {C.neutral(f'IRON CONDOR')}")
            print(f"  SELL:       {call_sell:,.0f} CE + {put_sell:,.0f} PE (Short wings)")
            print(f"  BUY:        {call_buy:,.0f} CE + {put_buy:,.0f} PE (Protection)")
            print(f"  Max Profit: Credit received")
            print(f"  Max Risk:   Spread width - credit")
            if resistances and supports:
                print(f"  Safe if:    Nifty stays between {put_sell:,.0f} — {call_sell:,.0f}")
        
        print(f"\n  {C.DIM}─ Risk Management ─{C.RESET}")
        print(f"  Position Size: {'FULL' if confidence > 65 else 'HALF' if confidence > 45 else 'MINIMAL'}")
        print(f"  Hold Period:   5 days (weekly expiry)")
    
    def option_9_retrain(self):
        """Force retrain all models."""
        separator("RETRAINING ALL MODELS")
        print(f"\n  {C.YELLOW}This will retrain everything from scratch...{C.RESET}\n")
        
        self.df_raw = load_all_data()
        self.df, self.feature_cols = engineer_features(self.df_raw)
        self.current_price = float(self.df["close"].iloc[-1])
        
        self.regime.train(self.df)
        self.regime.save()
        
        self.ensemble.train(self.df, self.feature_cols)
        self.ensemble.save()
        
        self.range_pred.train(self.df, self.feature_cols)
        self.range_pred.save()
        
        print(f"\n  {C.GREEN}{C.BOLD}✅ All models retrained and saved!{C.RESET}")
    
    def option_backtest(self):
        """Run out-of-sample backtest."""
        separator("OUT-OF-SAMPLE BACKTEST")
        self.ensemble.detailed_backtest(self.df, self.feature_cols, train_end_year=2023)
    
    def option_features(self):
        """Show top features."""
        separator("TOP PREDICTIVE FEATURES")
        top = self.ensemble.get_top_features(20)
        if top is not None and not top.empty:
            print(f"\n  {'Rank':>4}  {'Feature':<25}  {'Importance':>12}")
            print(f"  {'─'*45}")
            for i, (_, row) in enumerate(top.iterrows()):
                bar = "█" * int(row["importance"] * 200)
                print(f"  {i+1:>4}  {row['feature']:<25}  {row['importance']:>10.4f}  {C.CYAN}{bar}{C.RESET}")
        else:
            print(f"  {C.YELLOW}No feature importance data available. Retrain first.{C.RESET}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════════════════
    
    def run(self):
        """Main interactive loop."""
        self.initialize()
        
        while True:
            self.show_menu()
            
            try:
                choice = input(f"  {C.CYAN}Select option > {C.RESET}").strip().upper()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {C.DIM}Goodbye!{C.RESET}")
                break
            
            if choice == "1":
                self.option_1_verdict()
            elif choice == "2":
                self.option_2_7day()
            elif choice == "3":
                self.option_3_30day()
            elif choice == "4":
                self.option_4_sr()
            elif choice == "5":
                self.option_5_whipsaw()
            elif choice == "6":
                self.option_6_condor()
            elif choice == "7":
                self.option_7_bounce()
            elif choice == "8":
                self.option_8_trade()
            elif choice == "9":
                self.option_9_retrain()
            elif choice == "B":
                self.option_backtest()
            elif choice == "F":
                self.option_features()
            elif choice == "0":
                print(f"\n  {C.DIM}David Oracle signing off. Trade safe! 🛡️{C.RESET}")
                break
            else:
                print(f"  {C.RED}Invalid option. Please try again.{C.RESET}")
            
            print()
            input(f"  {C.DIM}Press Enter to continue...{C.RESET}")


if __name__ == "__main__":
    oracle = DavidOracle()
    oracle.run()
