
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Ensure imports work from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import C, UP, DOWN, SIDEWAYS, NIFTY_SYMBOL, MODEL_DIR
from data_engine import load_all_data
from feature_forge import engineer_features
from models.meta_ensemble import MetaEnsemble
from models.regime_detector import RegimeDetector
from models.range_predictor import RangePredictor
from models.sr_engine import SREngine
from analyzers.whipsaw_detector import WhipsawDetector
from analyzers.iron_condor_analyzer import IronCondorAnalyzer
from analyzers.strike_backtester import full_strike_analysis, get_survival_history, regime_conditional_survival, _classify_regime
from analyzers.bounce_analyzer import BounceAnalyzer

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="David Oracle v6.6.6",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# PREMIUM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp {
background-color: #0a0e17;
color: #E8E8E8;
font-family: 'Inter', sans-serif;
}

/* Hide default streamlit header/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
/* header {visibility: hidden;}  <- Removed so expander chevron and deploy buttons are visible */

/* Sidebar styling */
[data-testid="stSidebar"] {
background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
border-right: 1px solid #21262d;
}

/* Custom card */
.glass-card {
background: linear-gradient(135deg, rgba(22, 27, 34, 0.9), rgba(13, 17, 23, 0.95));
border: 1px solid rgba(255, 255, 255, 0.06);
border-radius: 16px;
padding: 24px;
backdrop-filter: blur(10px);
box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
margin-bottom: 16px;
transition: all 0.3s ease;
}
.glass-card:hover {
border-color: rgba(255, 255, 255, 0.1);
box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
}

/* Card with glow */
.glow-green {
border-color: rgba(0, 255, 127, 0.2);
box-shadow: 0 8px 32px rgba(0, 255, 127, 0.05);
}
.glow-red {
border-color: rgba(255, 75, 75, 0.2);
box-shadow: 0 8px 32px rgba(255, 75, 75, 0.05);
}
.glow-yellow {
border-color: rgba(255, 215, 0, 0.2);
box-shadow: 0 8px 32px rgba(255, 215, 0, 0.05);
}
.glow-cyan {
border-color: rgba(0, 200, 255, 0.2);
box-shadow: 0 8px 32px rgba(0, 200, 255, 0.05);
}

/* Section label */
.section-label {
color: #7d8590;
font-size: 11px;
text-transform: uppercase;
letter-spacing: 2px;
font-weight: 600;
margin-bottom: 8px;
}

/* Big direction text */
.direction-text {
font-size: 42px;
font-weight: 800;
text-align: center;
margin: 8px 0;
text-shadow: 0 0 30px currentColor;
}
.up-color { color: #00FF7F; }
.down-color { color: #FF4B4B; }
.side-color { color: #FFD700; }

/* Confidence text */
.conf-text {
font-size: 20px;
font-weight: 600;
text-align: center;
color: #AAA;
}

/* Regime dots container */
.regime-bar {
display: flex;
align-items: center;
justify-content: space-between;
padding: 12px 0;
}
.regime-dot-wrap {
text-align: center;
flex: 1;
}
.regime-dot {
width: 14px;
height: 14px;
border-radius: 50%;
margin: 0 auto 6px auto;
transition: all 0.3s;
}
.regime-dot.active {
width: 22px;
height: 22px;
box-shadow: 0 0 12px currentColor;
}
.regime-label {
font-size: 8px;
color: #555;
text-transform: uppercase;
letter-spacing: 0.5px;
}
.regime-label.active {
color: #DDD;
font-weight: 700;
}

/* Progress bar */
.meter-track {
background: #1a1f2e;
border-radius: 10px;
height: 12px;
overflow: hidden;
margin: 8px 0;
}
.meter-fill {
height: 100%;
border-radius: 10px;
transition: width 0.6s ease;
}

/* Prob bar row */
.prob-row {
display: flex;
align-items: center;
margin: 6px 0;
gap: 10px;
}
.prob-label {
width: 55px;
color: #888;
font-size: 12px;
font-weight: 500;
}
.prob-bar-track {
flex: 1;
background: #1a1f2e;
border-radius: 8px;
height: 16px;
overflow: hidden;
}
.prob-bar-fill {
height: 100%;
border-radius: 8px;
transition: width 0.5s ease;
}
.prob-value {
width: 45px;
text-align: right;
font-size: 13px;
font-weight: 700;
}

/* Trade action card */
.trade-card {
background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);
border-radius: 16px;
padding: 28px;
margin: 16px 0;
}
.trade-strategy {
font-size: 24px;
font-weight: 800;
margin: 8px 0 16px 0;
}
.trade-grid {
display: grid;
grid-template-columns: 1fr 1fr;
gap: 16px;
}
.trade-item-label {
font-size: 10px;
color: #666;
text-transform: uppercase;
letter-spacing: 1px;
}
.trade-item-value {
font-size: 18px;
font-weight: 700;
color: #FFF;
}

/* SR ladder */
.sr-level {
display: flex;
align-items: center;
gap: 10px;
padding: 6px 0;
}
.sr-tag {
font-size: 11px;
font-weight: 700;
width: 28px;
}
.sr-price {
font-size: 15px;
font-weight: 600;
width: 80px;
}
.sr-dist {
font-size: 11px;
color: #888;
width: 55px;
}
.sr-bar {
flex: 1;
height: 8px;
border-radius: 4px;
}

/* Spot line */
.spot-line {
display: flex;
align-items: center;
gap: 8px;
padding: 10px 0;
margin: 6px 0;
border-top: 1px dashed #444;
border-bottom: 1px dashed #444;
}

/* Signal list */
.signal-item {
display: flex;
align-items: center;
gap: 8px;
padding: 4px 0;
font-size: 12px;
}

/* Sidebar custom */
.sidebar-price {
font-size: 28px;
font-weight: 800;
color: #FFF;
}
.sidebar-label {
font-size: 11px;
color: #666;
text-transform: uppercase;
letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# INITIALIZATION & CACHING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_oracle():
    """Load data and models once. Self-healing: if pkl files are incompatible, retrain."""
    df_raw = load_all_data()
    df, features = engineer_features(df_raw)

    # --- Regime Detector ---
    regime = RegimeDetector()
    try:
        if not regime.load():
            regime.train(df)
            regime.save()
    except Exception as e:
        print(f"  [WARN] Regime pkl incompatible ({e}), retraining...")
        import os as _os
        pkl_path = _os.path.join(MODEL_DIR, "regime_detector.pkl")
        if _os.path.exists(pkl_path):
            _os.remove(pkl_path)
        regime = RegimeDetector()
        regime.train(df)
        regime.save()

    # --- Meta Ensemble Classifier (v6.6.6) ---
    ensemble = MetaEnsemble(regime)
    try:
        if not ensemble.load():
            ensemble.train(df, features)
            ensemble.save()
    except Exception as e:
        print(f"  [WARN] MetaEnsemble incompatible ({e}), retraining...")
        import os as _os
        pkl_path = _os.path.join(MODEL_DIR, "meta_ensemble.pkl")
        if _os.path.exists(pkl_path):
            _os.remove(pkl_path)
        ensemble = MetaEnsemble(regime)
        ensemble.train(df, features)
        ensemble.save()

    # --- Range Predictor ---
    range_pred = RangePredictor()
    try:
        if not range_pred.load():
            range_pred.train(df, features)
            range_pred.save()
    except Exception as e:
        print(f"  [WARN] Range pkl incompatible ({e}), retraining...")
        import os as _os
        pkl_path = _os.path.join(MODEL_DIR, "range_predictor.pkl")
        if _os.path.exists(pkl_path):
            _os.remove(pkl_path)
        range_pred = RangePredictor()
        range_pred.train(df, features)
        range_pred.save()

    sr = SREngine()
    whipsaw = WhipsawDetector()
    condor = IronCondorAnalyzer()
    bounce = BounceAnalyzer()

    return {
        "df_raw": df_raw,
        "df": df,
        "features": features,
        "ensemble": ensemble,
        "regime": regime,
        "range_pred": range_pred,
        "sr": sr,
        "whipsaw": whipsaw,
        "condor": condor,
        "bounce": bounce
    }

with st.spinner("🦅 Waking up David... Loading models & data"):
    oracle = load_oracle()

df = oracle["df"]
current_price = float(df["close"].iloc[-1])
vix = float(oracle["df_raw"]["vix"].iloc[-1]) if "vix" in oracle["df_raw"].columns else 15.0
last_date = df["date"].iloc[-1].strftime("%d %b %Y")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def direction_class(d):
    if d == UP: return "up-color"
    if d == DOWN: return "down-color"
    return "side-color"

def direction_hex(d):
    if d == UP: return "#00FF7F"
    if d == DOWN: return "#FF4B4B"
    return "#FFD700"

def direction_glow(d):
    if d == UP: return "glow-green"
    if d == DOWN: return "glow-red"
    return "glow-yellow"

def direction_icon(d):
    if d == UP: return "▲"
    if d == DOWN: return "▼"
    return "━"

def render_regime_bar(current_regime):
    """Render 5-dot regime state indicator."""
    states = [
        ("STRONG BULLISH", "#00FF7F"),
        ("MILD BULLISH", "#7CFC00"),
        ("SIDEWAYS", "#FFD700"),
        ("MILD BEARISH", "#FFA500"),
        ("STRONG BEARISH", "#FF4500"),
    ]
    dots_html = ""
    for label, color in states:
        is_active = label == current_regime
        dot_class = "regime-dot active" if is_active else "regime-dot"
        label_class = "regime-label active" if is_active else "regime-label"
        size = "22px" if is_active else "12px"
        opacity = "1.0" if is_active else "0.25"
        shadow = f"0 0 14px {color}" if is_active else "none"
        border = f"2px solid {color}" if is_active else "none"
        dots_html += f"""
<div class="regime-dot-wrap">
    <div style="width:{size}; height:{size}; border-radius:50%;
                background:{color}; opacity:{opacity}; border:{border};
                box-shadow:{shadow}; margin:0 auto 6px auto;
                transition: all 0.3s;"></div>
    <div class="{label_class}" style="{'color:'+color if is_active else ''}">{label.replace(' ', '<br/>')}</div>
</div>"""
    return f'<div class="regime-bar">{dots_html}</div>'

def render_prob_bars(prob_up, prob_down, tree_conf=0.5, lstm_conf=0.5):
    """Render horizontal probability bars with sub-model confidence."""
    probs = [
        ("UP", prob_up, "#00FF7F"),
        ("DOWN", prob_down, "#FF4B4B"),
    ]
    html = ""
    for label, prob, color in probs:
        pct = int(prob * 100)
        html += f"""
<div class="prob-row">
    <span class="prob-label">{label}</span>
    <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    <span class="prob-value" style="color:{color};">{pct}%</span>
</div>"""
    # Sub-model confidence bars
    subs = [
        ("TREES", tree_conf, "#00C8FF"),
        ("LSTM", lstm_conf, "#DA70D6"),
    ]
    html += '<div style="margin-top:12px; border-top:1px solid #222; padding-top:8px;">'
    html += '<div style="color:#666; font-size:9px; letter-spacing:1px; margin-bottom:6px;">SUB-MODEL CONFIDENCE</div>'
    for label, conf, color in subs:
        pct = int(conf * 100)
        html += f"""
<div class="prob-row">
    <span class="prob-label">{label}</span>
    <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    <span class="prob-value" style="color:{color};">{pct}%</span>
</div>"""
    html += '</div>'
    return html

def render_whipsaw_meter(chop_prob):
    """Render horizontal whipsaw meter."""
    if chop_prob < 40:
        color = "#00FF7F"
        label = "✅ CLEAR — TRENDING"
    elif chop_prob < 60:
        color = "#FFD700"
        label = "⚠️ CAUTION — CHOPPY"
    else:
        color = "#FF4B4B"
        label = "🚨 DANGER — HIGH CHOP"
    pct = int(chop_prob)
    return f"""
<div style="display:flex; justify-content:space-between; margin-bottom:6px;">
    <span style="color:#888; font-size:11px; text-transform:uppercase; letter-spacing:1px;">Whipsaw Risk</span>
    <span style="color:{color}; font-weight:700; font-size:13px;">{pct}%</span>
</div>
<div class="meter-track">
    <div class="meter-fill" style="width:{pct}%; background: linear-gradient(90deg, #00FF7F, {color});"></div>
</div>
<div style="color:{color}; font-weight:600; font-size:14px; margin-top:6px; text-align:center;">{label}</div>
"""

def render_eligibility_card(direction, confidence, regime_label, chop_prob):
    """Render the safety eligibility card (Red/Yellow/Green light)."""
    status = "READY"
    color = "#00FF7F"
    advice = "Signals are in sync. Probability of success is optimized."
    
    conf_pct = confidence * 100
    is_bearish_regime = "BEARISH" in regime_label
    is_bullish_regime = "BULLISH" in regime_label
    
    # 1. Check for CLASH (Immediate Stay Out)
    if (direction == UP and is_bearish_regime) or (direction == DOWN and is_bullish_regime):
        status = "STAY OUT: REGIME CLASH"
        color = "#FF4B4B"
        advice = f"🚨 AI is {direction} but Market Structure is {regime_label}. History shows a 41% higher risk of blowout here."
    # 2. Check for Sidebar/Choppy (Caution)
    elif chop_prob > 45 or conf_pct < 45:
        status = "CAUTION: LOW CLARITY"
        color = "#FFD700"
        advice = "Low conviction or high whipsaw detected. Reduce position size significantly."
    # 3. Sideways market
    elif direction == SIDEWAYS:
        status = "NEUTRAL: SIDEWAYS"
        color = "#AAA"
        advice = "No clear trend edge. Great for Theta decay if spreads are wide enough."
        
    return f"""
<div class="glass-card" style="border: 2px solid {color}; background: {color}08; padding: 15px; text-align: center;">
    <div style="color:{color}; font-weight:800; font-size:11px; letter-spacing:1.5px; margin-bottom:5px;">TRADE ELIGIBILITY</div>
    <div style="color:{color}; font-size:20px; font-weight:900; margin-bottom:8px;">{status}</div>
    <div style="color:#EEE; font-size:12px; line-height:1.4; font-weight:500;">{advice}</div>
</div>
"""
def render_setup_radar(regime, chop_prob, direction, confidence, spot, trust_score):
    """Detect 'Super-Matrix' setups and return rich UI metadata."""
    if direction == SIDEWAYS:
        return f"""<div style="background: #33333340; border-left: 4px solid #666; padding: 15px; border-radius: 4px; margin: 15px 0;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
<span style="color:#666; font-weight:900; font-size:11px; letter-spacing:1.5px;">EDGE DETECTOR</span>
<span>🛑</span>
</div>
<div style="color:#FFF; font-size:18px; font-weight:900; margin-bottom:12px;">NO DIRECTIONAL EDGE</div>
<div style="color:#AAA; font-size:13px; line-height:1.5;">AI predicts SIDEWAYS. No statistical edge for directional credit spreads.<br>Wait for a clear UP/DOWN signal or consider Delta-Neutral strategies.</div>
</div>"""

    conf_pct = confidence * 100
    is_clear = chop_prob <= 35
    is_high = conf_pct >= 50   # Lowered from 55% → 50% for more honest trades
    is_med = conf_pct >= 45
    
    # Trust grade checks
    is_trust_a = trust_score >= 85     # A/A+
    is_trust_b = trust_score >= 60     # B or above
    
    # Setup Definitions based on Super-Matrix
    active_setup = "NO EDGE"
    active_meta = {
        "color": "#666", 
        "icon": "🛑", 
        "profit": "-", 
        "win": "-", 
        "freq": "-",
        "rec": "No clear statistical edge. Stay in Cash.",
        "spread": "N/A"
    }
    
    # 1. GOD MODE (A/A+ trust only)
    if "TRENDING" in regime and is_clear and is_high and is_trust_a:
        active_setup = "🏆 GOD MODE"
        active_meta = {
            "color": "#00FF7F", "icon": "🏆", "profit": "₹57,961", "win": "100%", "freq": "~6/yr",
            "rec": f"SELL {direction} Spread at {spot* (0.975 if direction==UP else 1.025):,.0f}",
            "spread": "Bull Put" if direction == UP else "Bear Call"
        }
    # 2. AGGRESSIVE (A/A+ or B trust)
    elif "VOLATILE" in regime and is_clear and is_high and is_trust_b:
        active_setup = "🚀 AGGRESSIVE"
        active_meta = {
            "color": "#00C8FF", "icon": "🚀", "profit": "₹36,351", "win": "100%", "freq": "~14/yr",
            "rec": f"SELL {direction} Spread (3% OTM) @ {spot* (0.97 if direction==UP else 1.03):,.0f}",
            "spread": "Bull Put" if direction == UP else "Bear Call"
        }
    # 3. SAFE / CALM (A/A+ or B trust)
    elif "CALM" in regime and is_high and is_trust_b:
        active_setup = "🛡️ SAFE / CALM"
        active_meta = {
            "color": "#7CFC00", "icon": "🛡️", "profit": "₹21,099", "win": "100%", "freq": "~25/yr",
            "rec": f"SELL {direction} Spread (2% OTM) @ {spot* (0.98 if direction==UP else 1.02):,.0f}",
            "spread": "Bull Put" if direction == UP else "Bear Call"
        }
    # 4. BREAD & BUTTER — Sideways/Clear (A/A+ or B trust)
    elif "SIDEWAYS" in regime and is_clear and is_high and is_trust_b:
        active_setup = "🥦 BREAD & BUTTER"
        active_meta = {
            "color": "#FFD700", "icon": "🥦", "profit": "₹25,544", "win": "100%", "freq": "~250/yr",
            "rec": f"SELL {direction} Spread (1.5% OTM) @ {spot* (0.985 if direction==UP else 1.015):,.0f}",
            "spread": "Bull Put" if direction == UP else "Bear Call"
        }
    # 5. STEADY — Sideways/Choppy (A/A+ or B trust)
    elif "SIDEWAYS" in regime and not is_clear and is_high and is_trust_b:
        active_setup = "🔄 STEADY"
        active_meta = {
            "color": "#B8860B", "icon": "🔄", "profit": "₹22,679", "win": "100%", "freq": "~63/yr",
            "rec": f"SELL {direction} Spread (2% OTM, half size) @ {spot* (0.98 if direction==UP else 1.02):,.0f}",
            "spread": "Bull Put" if direction == UP else "Bear Call"
        }
    # 6. SCALP (Medium conviction, any trust)
    elif is_med and not is_high:
        active_setup = "🤏 SCALP"
        active_meta = {
            "color": "#AAA", "icon": "🤏", "profit": "~₹14,000", "win": "~85%", "freq": "Daily",
            "rec": "Quick in-out. Sell 1% OTM.",
            "spread": "Credit Spread"
        }

    html = f"""<div style="background: {active_meta['color']}12; border-left: 4px solid {active_meta['color']}; padding: 15px; border-radius: 4px; margin: 15px 0;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
<span style="color:{active_meta['color']}; font-weight:900; font-size:11px; letter-spacing:1.5px;">EDGE DETECTOR</span>
<span>{active_meta['icon']}</span>
</div>
<div style="color:#FFF; font-size:18px; font-weight:900; margin-bottom:12px;">{active_setup}</div>
<div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px; margin-bottom:15px; border-bottom:1px solid #333; padding-bottom:10px;">
<div>
<div style="color:#888; font-size:9px;">AVG PROFIT</div>
<div style="color:{active_meta['color']}; font-size:13px; font-weight:700;">{active_meta['profit']}</div>
</div>
<div>
<div style="color:#888; font-size:9px;">WIN RATE</div>
<div style="color:{active_meta['color']}; font-size:13px; font-weight:700;">{active_meta['win']}</div>
</div>
</div>
<div style="margin-bottom:10px;">
<div style="color:#888; font-size:9px;">RECOMMENDED STRATEGY</div>
<div style="color:#FFF; font-size:12px; font-weight:700;">{active_meta['spread']}</div>
</div>
<div style="background:#00000040; padding:8px; border-radius:4px;">
<div style="color:#888; font-size:9px; margin-bottom:2px;">STRIKE ADVICE</div>
<div style="color:#00FF7F; font-size:11px; font-weight:700; line-height:1.3;">{active_meta['rec']}</div>
</div>
</div>"""
    return html

def render_trade_card(direction, confidence, regime_label, is_choppy, spot, atr, supports, resistances):
    """Render the trade action card."""
    conf = confidence * 100

    # Determine strategy based on Exhaustive Backtest (V3 Credit Spreads)
    if is_choppy and conf < 50:
        strategy = "NO TRADE"
        strat_color = "#FF4B4B"
        reason = "Market is choppy with low confidence. Sit on hands."
        buy_label, buy_val, sell_label, sell_val = "—", "—", "—", "—"
        size = "NONE"
    elif direction == UP and conf >= 40:
        strategy = "BULL PUT SPREAD"
        strat_color = "#00FF7F"
        reason = f"Bullish bias at {conf:.0f}% confidence (Historical Win: 89%)"
        sell_strike = round(spot / 50) * 50
        buy_strike = round((spot - atr * 1.5) / 50) * 50
        sell_label, sell_val = "SELL", f"{sell_strike:,.0f} PE"
        buy_label, buy_val = "BUY", f"{buy_strike:,.0f} PE"
        size = "FULL" if conf > 50 else "HALF"
    elif direction == DOWN and conf >= 55:
        strategy = "BEAR CALL SPREAD"
        strat_color = "#FF4B4B"
        reason = f"Bearish bias at {conf:.0f}% confidence (Historical Win: 68%)"
        sell_strike = round(spot / 50) * 50
        buy_strike = round((spot + atr * 1.5) / 50) * 50
        sell_label, sell_val = "SELL", f"{sell_strike:,.0f} CE"
        buy_label, buy_val = "BUY", f"{buy_strike:,.0f} CE"
        size = "FULL" if conf > 65 else "HALF"
    elif direction == DOWN and conf < 55:
        strategy = "NO TRADE"
        strat_color = "#FFD700"
        reason = f"Bearish bias but confidence too low ({conf:.0f}% < 55%). Sit on hands."
        buy_label, buy_val, sell_label, sell_val = "—", "—", "—", "—"
        size = "NONE"
    elif direction == UP and conf < 40:
        strategy = "NO TRADE"
        strat_color = "#FFD700"
        reason = f"Bullish bias but confidence too low ({conf:.0f}% < 40%). Sit on hands."
        buy_label, buy_val, sell_label, sell_val = "—", "—", "—", "—"
        size = "NONE"
    else:
        strategy = "NO TRADE"
        strat_color = "#FFD700"
        reason = f"Sideways/Unclear market. Iron Condors have 74% fail rate here. Sit on hands."
        buy_label, buy_val, sell_label, sell_val = "—", "—", "—", "—"
        size = "NONE"

    size_color = "#00FF7F" if size == "FULL" else "#FFD700" if size == "HALF" else "#FF4B4B"

    return f"""
<div class="trade-card" style="border: 1px solid {strat_color}30;">
    <div class="section-label">RECOMMENDED ACTION</div>
    <div class="trade-strategy" style="color:{strat_color};">{strategy}</div>
    <div style="color:#888; font-size:12px; margin-bottom:18px;">{reason}</div>
    <div class="trade-grid">
        <div>
            <div class="trade-item-label">{buy_label}</div>
            <div class="trade-item-value">{buy_val}</div>
        </div>
        <div>
            <div class="trade-item-label">{sell_label}</div>
            <div class="trade-item-value">{sell_val}</div>
        </div>
        <div>
            <div class="trade-item-label">POSITION SIZE</div>
            <div class="trade-item-value" style="color:{size_color};">{size}</div>
        </div>
        <div>
            <div class="trade-item-label">HOLD PERIOD</div>
            <div class="trade-item-value">5 days</div>
        </div>
    </div>
</div>
"""

def render_sr_ladder(supports, resistances, spot):
    """Render visual S/R price ladder."""
    html = ""

    # Resistance levels (top to bottom, furthest first)
    for i, r in enumerate(reversed(resistances[:3])):
        dist = ((r['price'] - spot) / spot) * 100
        strength_pct = min(100, int(r['strength'] * 15))
        tag = f"R{len(resistances[:3]) - i}"
        html += f"""
<div class="sr-level">
    <span class="sr-tag" style="color:#FF4B4B;">{tag}</span>
    <span class="sr-price" style="color:#FF4B4B;">{r['price']:,.0f}</span>
    <span class="sr-dist">+{dist:.1f}%</span>
    <div class="sr-bar" style="background: linear-gradient(90deg, #FF4B4B, transparent); width:{strength_pct}%;"></div>
    <span style="font-size:10px; color:#666;">{r['touches']}T</span>
</div>"""

    # Spot line
    html += f"""
<div class="spot-line">
    <span style="color:#FFF; font-weight:700; font-size:11px;">SPOT</span>
    <span style="color:#FFF; font-weight:800; font-size:16px;">{spot:,.2f}</span>
    <span style="flex:1; border-bottom: 1px dashed #555; height:1px;"></span>
</div>"""

    # Support levels (top to bottom, nearest first)
    for i, s in enumerate(supports[:3]):
        dist = ((spot - s['price']) / spot) * 100
        strength_pct = min(100, int(s['strength'] * 15))
        tag = f"S{i+1}"
        html += f"""
<div class="sr-level">
    <span class="sr-tag" style="color:#00FF7F;">{tag}</span>
    <span class="sr-price" style="color:#00FF7F;">{s['price']:,.0f}</span>
    <span class="sr-dist">−{dist:.1f}%</span>
    <div class="sr-bar" style="background: linear-gradient(90deg, #00FF7F, transparent); width:{strength_pct}%;"></div>
    <span style="font-size:10px; color:#666;">{s['touches']}T</span>
</div>"""

    return html

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦅 David Oracle")
    st.markdown(f"""
<div style="margin: 16px 0;">
<div class="sidebar-label">NIFTY 50</div>
<div class="sidebar-price">{current_price:,.2f}</div>
</div>
<div style="display:flex; gap:20px; margin-bottom:20px;">
<div>
<div class="sidebar-label">VIX</div>
<div style="font-size:18px; font-weight:600; color:{'#FF4B4B' if vix > 20 else '#FFD700' if vix > 15 else '#00FF7F'};">{vix:.2f}</div>
</div>
<div>
<div class="sidebar-label">DATE</div>
<div style="font-size:14px; font-weight:500; color:#AAA;">{last_date}</div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    mode = st.radio("Navigation", [
        "🎯 Dashboard",
        "📈 Forecast & Ranges",
        "🎯 Strike Recommender",
        "🧪 Strategy Lab"
    ], label_visibility="collapsed")

    # Edge Radar in Sidebar
    # We need to run a mini-predict to get the data for the radar
    pred_sidebar = oracle["ensemble"].predict_today(oracle["df"])
    hmm_regime_sidebar = oracle["regime"].get_regime_with_micro_direction(oracle["df"], pred_sidebar)["regime"]
    structural_regime_sidebar = _classify_regime(oracle["df"].iloc[-1])
    whipsaw_sidebar = oracle["whipsaw"].analyze(oracle["df"])["whipsaw_prob"]
    
    # Get a dummy strike for trust score calculation (2% OTM)
    spot_sidebar = float(oracle["df"].iloc[-1]["close"])
    side_sidebar = pred_sidebar["direction"]
    strike_sidebar = spot_sidebar * (0.98 if side_sidebar == UP else 1.02)
    trust_sidebar = 85 # Default if not sideways
    
    if side_sidebar != SIDEWAYS:
        try:
            analysis_sidebar = full_strike_analysis(
                oracle["df"], spot_sidebar, strike_sidebar, 
                side=side_sidebar, vix=float(oracle["df"].iloc[-1].get("vix", 15))
            )
            trust_sidebar = analysis_sidebar["trust_score"]
        except:
            pass

    st.markdown(render_setup_radar(
        structural_regime_sidebar, 
        whipsaw_sidebar, 
        pred_sidebar["direction"], 
        pred_sidebar["confidence"],
        spot_sidebar,
        trust_sidebar
    ), unsafe_allow_html=True)

    st.markdown("---")
    
    with st.expander("📚 How to Use David Oracle"):
        st.markdown("""
        **1. AI Verdict (Core Prediction)**
        - What it is: The 5-day directional forecast.
        - **UP (>40%)**: Sell a **Bull Put Spread**.
        - **DOWN (>55%)**: Sell a **Bear Call Spread**.
        - Hold Time: Always **5 days**. (You win if it moves your way OR stays sideways).
        
        **2. Market Regime**
        - What it is: The overarching background trend (e.g., Mild Bearish).
        - How to trade: David analyzes this automatically to filter out counter-trend noise. If it's chopping sideways, Iron Condors are NOT recommended (74% fail rate currently).
        
        **3. Whipsaw Detector**
        - What it is: Measures erratic chop and volatility.
        - **0-40% 🟢**: Safe, trending market.
        - **>55% 🔴**: Danger! High chop.
        - How to trade: If Whipsaw > 55% AND Conviction < 50%, David will throw a **NO TRADE** alert to protect your capital.
        
        **4. Signal Breakdown**
        - What it is: The underlying metrics driving the Whipsaw score, such as Bollinger Band Squeezes, VIX Divergences, and abnormal daily candle flips.
        """)
        
    st.markdown("---")
    if st.button("🔄 Get Live Market Data", width="stretch"):
        with st.spinner("Fetching latest data and running predictions..."):
            st.cache_resource.clear()
        st.success("Data synced! Reloading...")
        st.rerun()

    st.markdown(f"""
<div style="position:fixed; bottom:20px; font-size:10px; color:#444;">
David Oracle v6.6.6 • {len(df)} trading days
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if mode == "🎯 Dashboard":
    # Get all predictions
    pred = oracle["ensemble"].predict_today(df)
    regime_info = oracle["regime"].get_regime_with_micro_direction(df, pred)
    whipsaw_data = oracle["whipsaw"].analyze(df)
    supports, resistances = oracle["sr"].find_levels(oracle["df_raw"])

    direction = pred["direction"]
    confidence = pred["confidence"]
    conf_pct = confidence * 100
    hex_color = direction_hex(direction)
    css_class = direction_class(direction)
    icon = direction_icon(direction)

    # ─── ROW 1: The Verdict ──────────────────────────────────────────────
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        # Direction + Gauge
        glow = direction_glow(direction)
        st.markdown(f"""
<div class="glass-card {glow}">
<div class="section-label">AI VERDICT</div>
<div class="direction-text {css_class}">{icon} {direction}</div>
</div>

{render_eligibility_card(direction, confidence, regime_info['regime'], whipsaw_data['whipsaw_prob'])}
""", unsafe_allow_html=True)

        # Plotly gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=conf_pct,
            number={'suffix': '%', 'font': {'size': 38, 'color': hex_color, 'family': 'Inter'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': '#333',
                         'dtick': 25, 'tickfont': {'size': 10, 'color': '#555'}},
                'bar': {'color': hex_color, 'thickness': 0.25},
                'bgcolor': '#0d1117',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(255,75,75,0.08)'},
                    {'range': [35, 55], 'color': 'rgba(255,215,0,0.08)'},
                    {'range': [55, 100], 'color': 'rgba(0,255,127,0.08)'},
                ],
                'threshold': {
                    'line': {'color': hex_color, 'width': 3},
                    'thickness': 0.85,
                    'value': conf_pct,
                }
            }
        ))
        fig.update_layout(
            height=220,
            margin=dict(l=25, r=25, t=15, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#FAFAFA', 'family': 'Inter'}
        )
        st.plotly_chart(fig, width="stretch", key="gauge")

    with col2:
        # Regime indicator
        regime_label = regime_info["regime"]
        st.markdown(f"""
<div class="glass-card glow-cyan">
<div class="section-label">MARKET REGIME</div>
<div style="font-size:20px; font-weight:700; text-align:center; color:#00C8FF; margin:12px 0;">
{regime_label}
</div>
{render_regime_bar(regime_label)}
</div>
""", unsafe_allow_html=True)

        # Transition probs
        if regime_info.get("transition_probs"):
            st.markdown(f'<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-label">NEXT-DAY TRANSITIONS</div>', unsafe_allow_html=True)
            trans = regime_info["transition_probs"]
            for label, prob in sorted(trans.items(), key=lambda x: -x[1])[:3]:
                pct = int(prob * 100)
                st.markdown(f"""
<div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
<span style="font-size:10px; color:#888; width:110px; overflow:hidden; text-overflow:ellipsis;">{label}</span>
<div style="flex:1; background:#1a1f2e; border-radius:4px; height:6px; overflow:hidden;">
<div style="width:{pct}%; height:100%; background:#00C8FF; border-radius:4px;"></div>
</div>
<span style="font-size:11px; color:#AAA; width:32px; text-align:right;">{pct}%</span>
</div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        # Whipsaw meter
        chop_prob = whipsaw_data["whipsaw_prob"]
        st.markdown(f"""<div class="glass-card">
<div class="section-label">WHIPSAW DETECTOR</div>
<div style="margin-top:12px;">
{render_whipsaw_meter(chop_prob)}
</div>
</div>""", unsafe_allow_html=True)

        # Whipsaw signals
        st.markdown(f'<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-label">SIGNAL BREAKDOWN</div>', unsafe_allow_html=True)
        for name, sig in whipsaw_data["signals"].items():
            icon_sig = "🔴" if sig["weight"] > 0.3 else "🟢"
            sig_text = sig['signal']
            if len(sig_text) > 28:
                sig_text = sig_text[:28] + "…"
            st.markdown(f"""
<div class="signal-item">
<span>{icon_sig}</span>
<span style="color:#AAA; font-size:11px;">{name.replace('_',' ').title()}</span>
</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ─── ROW 2: Probability Bars + Trade Action Card ─────────────────────
    st.markdown("")
    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        # Probability breakdown
        st.markdown(f"""<div class="glass-card">
<div class="section-label">PROBABILITY BREAKDOWN</div>
<div style="margin-top:12px;">
{render_prob_bars(pred['prob_up'], pred['prob_down'], pred.get('tree_conf', 0.5), pred.get('lstm_conf', 0.5))}
</div>
</div>""", unsafe_allow_html=True)

        # S/R Ladder
        st.markdown(f"""<div class="glass-card">
<div class="section-label">SUPPORT & RESISTANCE</div>
<div style="margin-top:8px;">
{render_sr_ladder(supports, resistances, current_price)}
</div>
</div>""", unsafe_allow_html=True)

    with col_right:
        # Trade Action Card
        atr = float(df["atr_14"].iloc[-1]) if "atr_14" in df.columns else current_price * 0.01
        st.markdown(render_trade_card(
            direction, confidence, regime_info["regime"],
            whipsaw_data["is_choppy"], current_price, atr,
            supports, resistances
        ), unsafe_allow_html=True)

        # Conviction indicator
        if conf_pct >= 65:
            conv_text = "★ HIGH CONVICTION — Full position sizing"
            conv_color = "#00FF7F"
        elif conf_pct >= 45:
            conv_text = "◆ MODERATE — Half position sizing"
            conv_color = "#FFD700"
        else:
            conv_text = "○ LOW — Skip or minimal position"
            conv_color = "#FF4B4B"

        st.markdown(f"""
<div class="glass-card" style="border-color:{conv_color}30; text-align:center;">
<span style="color:{conv_color}; font-weight:700; font-size:15px;">{conv_text}</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: FORECAST & RANGES
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "📈 Forecast & Ranges":
    st.markdown(f"""<h2 style="margin-bottom:0;">📈 Price Forecast</h2>
<p style="color:#666; margin-top:4px;">Probability cones showing where Nifty is likely to land</p>
""", unsafe_allow_html=True)

    ranges = oracle["range_pred"].predict_range(df, current_price)

    tab7, tab30 = st.tabs(["7-Day Forecast", "30-Day Forecast"])

    for tab, horizon in [(tab7, 7), (tab30, 30)]:
        with tab:
            if horizon in ranges:
                r = ranges[horizon]

                # Fan chart
                fig = go.Figure()

                # 80% band
                fig.add_trace(go.Scatter(
                    x=[0, horizon, horizon, 0],
                    y=[current_price, r['p90'], r['p10'], current_price],
                    fill='toself',
                    fillcolor='rgba(0, 200, 255, 0.06)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='80% Confidence',
                    hoverinfo='skip'
                ))

                # 50% band
                fig.add_trace(go.Scatter(
                    x=[0, horizon, horizon, 0],
                    y=[current_price, r['p75'], r['p25'], current_price],
                    fill='toself',
                    fillcolor='rgba(0, 200, 255, 0.12)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='50% Confidence',
                    hoverinfo='skip'
                ))

                # Median path
                fig.add_trace(go.Scatter(
                    x=[0, horizon], y=[current_price, r['p50']],
                    mode="lines+markers",
                    name="Median Path",
                    line=dict(color="#00C8FF", width=3),
                    marker=dict(size=8)
                ))

                # Current price line
                fig.add_trace(go.Scatter(
                    x=[0, horizon], y=[current_price, current_price],
                    mode="lines",
                    name="Current Spot",
                    line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1)
                ))

                # P90 / P10 boundary lines
                fig.add_trace(go.Scatter(
                    x=[0, horizon], y=[current_price, r['p90']],
                    mode="lines",
                    name="90th %ile",
                    line=dict(color="rgba(0,255,127,0.3)", dash="dot", width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=[0, horizon], y=[current_price, r['p10']],
                    mode="lines",
                    name="10th %ile",
                    line=dict(color="rgba(255,75,75,0.3)", dash="dot", width=1)
                ))

                fig.update_layout(
                    title=f"{horizon}-Day Probability Cone",
                    xaxis_title="Days from today",
                    yaxis_title="Nifty Level",
                    height=450,
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(10,14,23,1)',
                    font={'family': 'Inter'},
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, width="stretch", key=f"cone_{horizon}")

                # Metric cards
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.markdown(f"""<div class="glass-card" style="text-align:center;">
                    <div class="section-label">10th %ILE</div>
                    <div style="color:#FF4B4B; font-size:18px; font-weight:700;">{r['p10']:,.0f}</div>
                </div>""", unsafe_allow_html=True)
                c2.markdown(f"""<div class="glass-card" style="text-align:center;">
                    <div class="section-label">25th %ILE</div>
                    <div style="color:#FFA500; font-size:18px; font-weight:700;">{r['p25']:,.0f}</div>
                </div>""", unsafe_allow_html=True)
                c3.markdown(f"""<div class="glass-card" style="text-align:center;">
                    <div class="section-label">MEDIAN</div>
                    <div style="color:#00C8FF; font-size:18px; font-weight:700;">{r['p50']:,.0f}</div>
                </div>""", unsafe_allow_html=True)
                c4.markdown(f"""<div class="glass-card" style="text-align:center;">
                    <div class="section-label">75th %ILE</div>
                    <div style="color:#7CFC00; font-size:18px; font-weight:700;">{r['p75']:,.0f}</div>
                </div>""", unsafe_allow_html=True)
                c5.markdown(f"""<div class="glass-card" style="text-align:center;">
                    <div class="section-label">90th %ILE</div>
                    <div style="color:#00FF7F; font-size:18px; font-weight:700;">{r['p90']:,.0f}</div>
                </div>""", unsafe_allow_html=True)

                # Interpretation
                st.markdown(f"""
<div class="glass-card glow-cyan">
<div class="section-label">INTERPRETATION</div>
<div style="margin-top:8px;">
<div style="font-size:14px; color:#DDD; margin:6px 0;">
📊 <strong>80% chance</strong> Nifty stays between
<span style="color:#FF4B4B; font-weight:700;">{r['p10']:,.0f}</span> and
<span style="color:#00FF7F; font-weight:700;">{r['p90']:,.0f}</span>
</div>
<div style="font-size:14px; color:#DDD; margin:6px 0;">
📊 <strong>50% chance</strong> Nifty stays between
<span style="color:#FFA500; font-weight:700;">{r['p25']:,.0f}</span> and
<span style="color:#7CFC00; font-weight:700;">{r['p75']:,.0f}</span>
</div>
<div style="font-size:14px; color:#DDD; margin:6px 0;">
🎯 <strong>Median target</strong>:
<span style="color:#00C8FF; font-weight:700;">{r['p50']:,.0f}</span>
({r['p50_pct']:+.2f}%)
</div>
</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: STRIKE RECOMMENDER
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "🎯 Strike Recommender":
    st.markdown(f"""<h2 style="margin-bottom:0;">🎯 AI Strike Recommender</h2>
<p style="color:#666; margin-top:4px;">Backtested strike selection with historical trust scoring</p>
""", unsafe_allow_html=True)

    # Controls Row 1
    sr_col1, sr_col2, sr_col3, sr_col4 = st.columns([1, 1, 1, 1])
    with sr_col1:
        sr_capital = st.number_input("Capital (₹)", value=200000, step=50000, key="sr_capital")
    with sr_col2:
        sr_otm = st.selectbox("OTM Distance", [150, 200, 250, 300, 350, 400, 500], index=5, key="sr_otm")
    with sr_col3:
        sr_width = st.selectbox("Spread Width", [100, 150, 200, 250], index=1, key="sr_width")
    with sr_col4:
        sr_hold = st.selectbox("Hold Period (days)", [5, 7, 10, 14, 21], index=1, key="sr_hold")

    # Controls Row 2 — Side + Custom Strike
    sr_col5, sr_col6 = st.columns([1, 1])
    with sr_col5:
        sr_side_mode = st.selectbox("Strategy Side", ["🤖 Auto (AI Direction)", "📉 Sell PE (Bullish)", "📈 Sell CE (Bearish)"], index=0, key="sr_side")
    with sr_col6:
        sr_custom_strike = st.number_input("Custom Sell Strike (0 = auto)", value=0, step=50, key="sr_custom")

    # Get AI prediction
    pred = oracle["ensemble"].predict_today(df)
    direction = pred["direction"]
    confidence = pred["confidence"]
    regime_info = oracle["regime"].get_regime_with_micro_direction(df, pred)
    regime_label = regime_info["regime"]

    # Determine side based on user choice
    import math
    if sr_side_mode.startswith("📉"):
        side = "PE"
        ai_override = True
    elif sr_side_mode.startswith("📈"):
        side = "CE"
        ai_override = True
    else:
        # Auto — AI decides
        ai_override = False
        if direction == "UP" or direction == "SIDEWAYS":
            side = "PE"
        else:
            side = "CE"

    # Calculate strikes
    if sr_custom_strike > 0:
        sell_strike = int(sr_custom_strike)
        actual_otm = abs(current_price - sell_strike)
    else:
        if side == "PE":
            sell_strike = int(round((current_price - sr_otm) / 50) * 50)
        else:
            sell_strike = int(round((current_price + sr_otm) / 50) * 50)
        actual_otm = sr_otm

    if side == "PE":
        buy_strike = sell_strike - sr_width
    else:
        buy_strike = sell_strike + sr_width

    # Strategy name
    if side == "PE":
        strategy_name = "Bull Put Spread"
    else:
        strategy_name = "Bear Call Spread"
    if ai_override:
        strategy_name += " (Manual)"
    else:
        strategy_name += f" (AI: {direction})"

    # Premium estimation
    iv = vix / 100
    est_dte = 10 if vix > 20 else 21 if vix > 13 else 35
    sqrt_t = math.sqrt(est_dte / 365)
    def _est_prem(strike):
        dist = abs(current_price - strike)
        dist_pct = dist / current_price
        atm = current_price * iv * sqrt_t * 0.4
        decay = math.exp(-2.5 * (dist_pct / (iv * sqrt_t + 0.001)))
        return max(3.0, atm * decay)
    credit = max(5, _est_prem(sell_strike) - _est_prem(buy_strike))
    max_profit_lot = credit * 65
    max_loss_lot = (sr_width - credit) * 65
    margin_per_lot = 35000
    lots = max(1, min(int((sr_capital * 0.30) / max_loss_lot), int((sr_capital * 0.95) / margin_per_lot)))

    # Run trust analysis with user-selected holding period
    with st.spinner(f"Running historical survival analysis ({sr_hold}-day hold)..."):
        analysis = full_strike_analysis(oracle["df_raw"], current_price, sell_strike, side, vix, holding_days=sr_hold)
        otm_pct = actual_otm / current_price
        survival_hist = get_survival_history(oracle["df_raw"], otm_pct, side, holding_days=sr_hold, window=30)

    trust = analysis["trust_score"]
    grade = analysis["grade"]

    # Grade color
    grade_colors = {"A+": "#00FF7F", "A": "#7CFC00", "B": "#FFD700", "C": "#FFA500", "D": "#FF4B4B"}
    grade_color = grade_colors.get(grade, "#888")

    # ── ROW 1: Trust Score + Strike Card ──
    tc1, tc2 = st.columns([1, 1.5])

    with tc1:
        # Trust Score gauge
        fig_trust = go.Figure(go.Indicator(
            mode="gauge+number",
            value=trust,
            number={'suffix': '%', 'font': {'size': 42, 'color': grade_color, 'family': 'Inter'}},
            title={'text': f'TRUST SCORE — {grade}', 'font': {'size': 16, 'color': '#AAA'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 0, 'dtick': 25,
                         'tickfont': {'size': 10, 'color': '#555'}},
                'bar': {'color': grade_color, 'thickness': 0.25},
                'bgcolor': '#0d1117',
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 45], 'color': 'rgba(255,75,75,0.08)'},
                    {'range': [45, 60], 'color': 'rgba(255,165,0,0.08)'},
                    {'range': [60, 75], 'color': 'rgba(255,215,0,0.08)'},
                    {'range': [75, 100], 'color': 'rgba(0,255,127,0.08)'},
                ],
                'threshold': {
                    'line': {'color': grade_color, 'width': 3},
                    'thickness': 0.85,
                    'value': trust,
                }
            }
        ))
        fig_trust.update_layout(
            height=260,
            margin=dict(l=25, r=25, t=50, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#FAFAFA', 'family': 'Inter'}
        )
        st.plotly_chart(fig_trust, width="stretch", key="trust_gauge")

        # Component breakdown
        st.markdown(f"""
<div class="glass-card">
<div class="section-label">TRUST COMPONENTS</div>
<div style="margin-top:12px;">
<div class="prob-row">
    <span class="prob-label">Survival</span>
    <div class="prob-bar-track"><div class="prob-bar-fill" style="width:{analysis['survival_component']}%; background:#00C8FF;"></div></div>
    <span class="prob-value" style="color:#00C8FF;">{analysis['survival_component']:.0f}%</span>
</div>
<div class="prob-row">
    <span class="prob-label">Accuracy</span>
    <div class="prob-bar-track"><div class="prob-bar-fill" style="width:{analysis['accuracy_component']}%; background:#7CFC00;"></div></div>
    <span class="prob-value" style="color:#7CFC00;">{analysis['accuracy_component']:.0f}%</span>
</div>
<div class="prob-row">
    <span class="prob-label">Regime</span>
    <div class="prob-bar-track"><div class="prob-bar-fill" style="width:{analysis['regime_component']}%; background:#FFD700;"></div></div>
    <span class="prob-value" style="color:#FFD700;">{analysis['regime_component']:.0f}%</span>
</div>
<div class="prob-row">
    <span class="prob-label">Cushion</span>
    <div class="prob-bar-track"><div class="prob-bar-fill" style="width:{analysis['cushion_component']}%; background:#FFA500;"></div></div>
    <span class="prob-value" style="color:#FFA500;">{analysis['cushion_component']:.0f}%</span>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    with tc2:
        # Strike Card
        dir_color = "#00FF7F" if direction == "UP" else "#FF4B4B" if direction == "DOWN" else "#FFD700"
        st.markdown(f"""
<div class="trade-card" style="border: 1px solid {grade_color}30;">
    <div class="section-label">RECOMMENDED STRIKE</div>
    <div class="trade-strategy" style="color:{grade_color};">{strategy_name}</div>
    <div style="color:#888; font-size:12px; margin-bottom:18px;">
        AI Direction: <span style="color:{dir_color}; font-weight:700;">{direction}</span> ({confidence*100:.0f}%) |
        Regime: <span style="color:#00C8FF;">{regime_label}</span> |
        Trust: <span style="color:{grade_color}; font-weight:700;">{grade} ({trust:.0f}%)</span>
    </div>
    <div class="trade-grid">
        <div>
            <div class="trade-item-label">SELL</div>
            <div class="trade-item-value" style="color:#FF4B4B;">{sell_strike:,} {'PE' if side == 'PE' else 'CE'}</div>
        </div>
        <div>
            <div class="trade-item-label">BUY</div>
            <div class="trade-item-value" style="color:#00FF7F;">{buy_strike:,} {'PE' if side == 'PE' else 'CE'}</div>
        </div>
        <div>
            <div class="trade-item-label">EST. PREMIUM</div>
            <div class="trade-item-value">₹{credit:.1f}</div>
        </div>
        <div>
            <div class="trade-item-label">WIDTH</div>
            <div class="trade-item-value">{sr_width} pts</div>
        </div>
        <div>
            <div class="trade-item-label">MAX PROFIT / LOT</div>
            <div class="trade-item-value" style="color:#00FF7F;">₹{max_profit_lot:,.0f}</div>
        </div>
        <div>
            <div class="trade-item-label">MAX LOSS / LOT</div>
            <div class="trade-item-value" style="color:#FF4B4B;">₹{max_loss_lot:,.0f}</div>
        </div>
        <div>
            <div class="trade-item-label">LOTS ({sr_capital/1000:.0f}K)</div>
            <div class="trade-item-value">{lots}</div>
        </div>
        <div>
            <div class="trade-item-label">STOP LOSS</div>
            <div class="trade-item-value" style="color:#FFA500;">2x Premium (₹{credit*2:.0f})</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        # Survival stats
        surv = analysis['survival_pct']
        surv_color = "#00FF7F" if surv >= 80 else "#FFD700" if surv >= 60 else "#FF4B4B"
        st.markdown(f"""
<div class="glass-card">
<div style="display:flex; justify-content:space-between;">
    <div style="text-align:center; flex:1;">
        <div class="section-label">SURVIVAL RATE</div>
        <div style="color:{surv_color}; font-size:28px; font-weight:800;">{surv:.0f}%</div>
        <div style="color:#666; font-size:10px;">{analysis['survived']}/{analysis['sample_size']} survived</div>
    </div>
    <div style="text-align:center; flex:1;">
        <div class="section-label">95% WILSON CI</div>
        <div style="color:#AAA; font-size:20px; font-weight:600;">{analysis['ci_low']:.0f}% — {analysis['ci_high']:.0f}%</div>
    </div>
    <div style="text-align:center; flex:1;">
        <div class="section-label">AVG MAX DIP</div>
        <div style="color:#FFA500; font-size:20px; font-weight:600;">₹{analysis['mae_rupees']:,}/lot</div>
        <div style="color:#666; font-size:10px;">P95: ₹{analysis['p95_mae_rupees']:,}</div>
    </div>
</div>
</div>
""", unsafe_allow_html=True)

    # ── ROW 2: AI & Historical Perspective ──
    st.markdown("")
    
    # Calculate historical up/down stats based on VIX regime
    vix_bucket_low = int(vix / 5) * 5
    vix_bucket_high = vix_bucket_low + 5
    hist_df = oracle["df_raw"][(oracle["df_raw"]["vix"] >= vix_bucket_low) & (oracle["df_raw"]["vix"] < vix_bucket_high)]
    hist_up = len(hist_df[hist_df["close"].shift(-5) > hist_df["close"]]) / len(hist_df) if not hist_df.empty else 0.5
    hist_down = 1.0 - hist_up
    
    st.markdown(f"""
<div class="glass-card">
<div class="section-label" style="text-align:center; margin-bottom:15px;">AI vs HISTORICAL PERSPECTIVE</div>

<div style="display:flex; justify-content:space-between; margin-bottom:5px;">
<span style="color:#00C8FF; font-size:12px; font-weight:600;">AI ENSEMBLE PREDICTION (Next 5 Days)</span>
<span style="color:{'#00FF7F' if direction=='UP' else '#FF4B4B' if direction=='DOWN' else '#FFD700'}; font-weight:700;">{direction} ({confidence*100:.0f}%)</span>
</div>
<div style="height:20px; border-radius:10px; display:flex; overflow:hidden; margin-bottom:20px;">
<div style="width:{pred['prob_up']*100}%; background:linear-gradient(90deg, #004d26, #00FF7F); display:flex; align-items:center; padding-left:10px; font-size:10px; font-weight:700;">UP {pred['prob_up']*100:.0f}%</div>
<div style="width:{pred['prob_down']*100}%; background:linear-gradient(270deg, #8b0000, #FF4B4B); display:flex; align-items:center; justify-content:flex-end; padding-right:10px; font-size:10px; font-weight:700;">DOWN {pred['prob_down']*100:.0f}%</div>
</div>

<div style="display:flex; justify-content:space-between; margin-bottom:5px;">
<span style="color:#888; font-size:12px; font-weight:600;">HISTORICAL ODDS (When VIX is between {vix_bucket_low} and {vix_bucket_high})</span>
<span style="color:#FFF; font-weight:700;">10-Year Average</span>
</div>
<div style="height:20px; border-radius:10px; display:flex; overflow:hidden;">
<div style="width:{hist_up*100}%; background:rgba(0,255,127,0.3); border-right:2px solid #000; display:flex; align-items:center; padding-left:10px; font-size:10px; font-weight:700;">HIST. UP {hist_up*100:.0f}%</div>
<div style="width:{hist_down*100}%; background:rgba(255,75,75,0.3); display:flex; align-items:center; justify-content:flex-end; padding-right:10px; font-size:10px; font-weight:700;">HIST. DOWN {hist_down*100:.0f}%</div>
</div>
</div>
""", unsafe_allow_html=True)

    # ── ROW 3: Historical Survival Chart ──
    st.markdown("")
    if not survival_hist.empty and "rolling_survival" in survival_hist.columns:
        fig_surv = go.Figure()

        # Survival line
        fig_surv.add_trace(go.Scatter(
            x=survival_hist["date"],
            y=survival_hist["rolling_survival"],
            mode="lines",
            name="30-Day Rolling Survival %",
            line=dict(color="#00C8FF", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 200, 255, 0.08)"
        ))

        # Reference lines
        fig_surv.add_hline(y=80, line_dash="dash", line_color="rgba(0,255,127,0.3)",
                          annotation_text="Target: 80%")
        fig_surv.add_hline(y=50, line_dash="dot", line_color="rgba(255,75,75,0.3)",
                          annotation_text="Danger: 50%")

        fig_surv.update_layout(
            title=f"Historical {int(actual_otm)}pt OTM {side} Survival Rate ({sr_hold}-Day Hold)",
            xaxis_title="Date",
            yaxis_title="Survival %",
            yaxis_range=[0, 105],
            height=350,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(10,14,23,1)',
            font={'family': 'Inter'},
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_surv, width="stretch", key="survival_chart")

    # ── ROW 4: Regime Breakdown ──
    regime_data = analysis.get("regime_data", {})
    if regime_data:
        st.markdown(f"""
<div class="glass-card">
<div class="section-label">REGIME-CONDITIONAL SURVIVAL</div>
<div style="display:flex; gap:12px; margin-top:12px; flex-wrap:wrap;">
""", unsafe_allow_html=True)

        for regime_name, rdata in regime_data.items():
            r_surv = rdata["survival"] * 100 if rdata["sample"] > 0 else 0
            r_color = "#00FF7F" if r_surv >= 80 else "#FFD700" if r_surv >= 60 else "#FF4B4B"
            r_icon = "📈" if regime_name == "TRENDING" else "↔️" if regime_name == "SIDEWAYS" else "🌊" if regime_name == "VOLATILE" else "😴"
            st.markdown(f"""
<div style="flex:1; min-width:140px; text-align:center; background:#0d1117; border-radius:12px; padding:16px; border:1px solid {r_color}20;">
    <div style="font-size:24px;">{r_icon}</div>
    <div style="color:#888; font-size:10px; text-transform:uppercase; letter-spacing:1px; margin:4px 0;">{regime_name}</div>
    <div style="color:{r_color}; font-size:24px; font-weight:800;">{r_surv:.0f}%</div>
    <div style="color:#555; font-size:10px;">{rdata['wins']}/{rdata['sample']} survived</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    # ── ROW 5: Action Verdict ──
    if trust >= 75:
        verdict = f"✅ TRADE — Trust {grade} ({trust:.0f}%). Enter with {lots} lots."
        v_color = "#00FF7F"
    elif trust >= 60:
        verdict = f"⚠️ CAUTIOUS — Trust {grade} ({trust:.0f}%). Consider half lots ({max(1,lots//2)})."
        v_color = "#FFD700"
    else:
        verdict = f"🛑 SKIP — Trust {grade} ({trust:.0f}%). Don't trade, wait for better setup."
        v_color = "#FF4B4B"

    st.markdown(f"""
<div class="glass-card" style="border-color:{v_color}30; text-align:center;">
<span style="color:{v_color}; font-weight:700; font-size:18px;">{verdict}</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: STRATEGY LAB
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "🧪 Strategy Lab":
    st.markdown(f"""<h2 style="margin-bottom:0;">🧪 Strategy Lab</h2>
<p style="color:#666; margin-top:4px;">Test your strikes and analyze probabilities</p>
""", unsafe_allow_html=True)

    # ─── Iron Condor ─────────────────────────────────────────────────────
    st.markdown(f"""
<div class="glass-card glow-cyan">
<div class="section-label">🛡️ IRON CONDOR ANALYZER</div>
<div style="color:#AAA; font-size:13px; margin-top:4px;">
"Will Nifty touch my strike?" — Empirical probability from 10 years of data
</div>
</div>
""", unsafe_allow_html=True)

    ic_col1, ic_col2 = st.columns([1, 1])
    with ic_col1:
        strike = st.number_input("Strike Price to Test", value=int(round(current_price / 100) * 100), step=50, key="ic_strike")
    with ic_col2:
        days = st.slider("Timeframe (Days)", 1, 30, 5, key="ic_days")

    if st.button("🔍 Analyze Strike", width="stretch", key="ic_btn"):
        res = oracle["condor"].analyze_strike(oracle["df_raw"], strike, days)
        tp = res['touch_prob']
        rp = res['recovery_prob']

        # Result color
        if tp > 60:
            tp_color = "#FF4B4B"
            verdict = "🚨 HIGH RISK — Consider adjusting"
        elif tp > 35:
            tp_color = "#FFD700"
            verdict = "⚠️ MODERATE — Monitor closely"
        else:
            tp_color = "#00FF7F"
            verdict = "✅ SAFE — Low touch probability"

        rc1, rc2, rc3 = st.columns(3)
        rc1.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div class="section-label">TOUCH PROBABILITY</div>
            <div style="color:{tp_color}; font-size:32px; font-weight:800;">{tp:.0f}%</div>
        </div>""", unsafe_allow_html=True)
        rc2.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div class="section-label">RECOVERY IF TOUCHED</div>
            <div style="color:#00C8FF; font-size:32px; font-weight:800;">{rp:.0f}%</div>
        </div>""", unsafe_allow_html=True)
        rc3.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div class="section-label">FIREFIGHT LEVEL</div>
            <div style="color:#FFA500; font-size:24px; font-weight:800;">{res['firefight_level']:,.0f}</div>
            <div style="color:#888; font-size:10px;">Start hedging here</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="glass-card" style="text-align:center; border-color:{tp_color}30;">
<span style="color:{tp_color}; font-weight:700; font-size:16px;">{verdict}</span>
</div>
""", unsafe_allow_html=True)

        # Max move distribution
        st.markdown(f"""
<div class="glass-card">
<div class="section-label">{days}-DAY MAX MOVE DISTRIBUTION</div>
<div style="display:flex; gap:20px; margin-top:12px;">
<div style="flex:1; text-align:center;">
<div style="color:#888; font-size:10px;">CONSERVATIVE (10th)</div>
<div style="color:#AAA; font-size:16px; font-weight:600;">{res['max_move_p10']:.2f}%</div>
</div>
<div style="flex:1; text-align:center;">
<div style="color:#888; font-size:10px;">TYPICAL (50th)</div>
<div style="color:#FFF; font-size:16px; font-weight:600;">{res['max_move_p50']:.2f}%</div>
</div>
<div style="flex:1; text-align:center;">
<div style="color:#888; font-size:10px;">EXTREME (90th)</div>
<div style="color:#FF4B4B; font-size:16px; font-weight:600;">{res['max_move_p90']:.2f}%</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ─── Firefight Recovery Calculator ──────────────────────────────────────────────────
    st.markdown(f"""
<div class="glass-card glow-red">
<div class="section-label">🔥 FIREFIGHT RECOVERY CALCULATOR</div>
<div style="color:#AAA; font-size:13px; margin-top:4px;">
"My spread is losing money. Should I hold or cut it?" — AI & Historical Recovery Analysis
</div>
</div>
""", unsafe_allow_html=True)

    ff_col1, ff_col2 = st.columns([1, 1])
    with ff_col1:
        spread_type = st.selectbox("What spread are you losing on?", ["📉 Bull Put Spread (Market Dropped)", "📈 Bear Call Spread (Market Rallied)"])
    with ff_col2:
        target = st.number_input("What is your challenged Strike Price?", value=int(round(current_price / 100) * 100) - 500 if "Bull" in spread_type else int(round(current_price / 100) * 100) + 500, step=100, key="bounce_target")

    if st.button("🔥 Calculate Rescue Odds", width="stretch", key="bounce_btn"):
        # 1. Get History
        res = oracle["bounce"].analyze(oracle["df_raw"], target)
        dist = res['distance_pct']
        dir_label = res['direction']
        
        # 2. Get AI
        pred = oracle["ensemble"].predict_today(oracle["df"])
        
        # Determine odds based on spread type
        is_bull_spread = "Bull" in spread_type
        if is_bull_spread:
            ai_rescue_prob = pred['prob_up'] * 100
            ai_rescue_str = "UP"
            ai_color = "#00FF7F" if ai_rescue_prob > 50 else "#FF4B4B"
        else:
            ai_rescue_prob = pred['prob_down'] * 100
            ai_rescue_str = "DOWN"
            ai_color = "#00FF7F" if ai_rescue_prob > 50 else "#FF4B4B"

        st.markdown(f"""
<div class="glass-card" style="text-align:center;">
<span style="color:#888;">Distance to Strike: </span>
<span style="color:{'#FF4B4B' if (is_bull_spread and dist > 0) or (not is_bull_spread and dist < 0) else '#00FF7F'}; font-weight:700;">{dist:+.2f}%</span>
<span style="color:#888;"> ({dir_label})</span>
</div>
""", unsafe_allow_html=True)

        # AI Prediction Bar
        st.markdown(f"""
<div class="glass-card">
    <div class="section-label">AI RESCUE PROBABILITY (Next 5 Days)</div>
    <div style="display:flex; justify-content:space-between; align-items:flex-end;">
        <span style="color:#888; font-size:12px;">Chance market moves {ai_rescue_str} to save your spread:</span>
        <span style="color:{ai_color}; font-size:28px; font-weight:800;">{ai_rescue_prob:.0f}%</span>
    </div>
    <div class="meter-track" style="margin-top:8px;">
        <div class="meter-fill" style="width:{ai_rescue_prob}%; background:{ai_color};"></div>
    </div>
</div>
""", unsafe_allow_html=True)

        # Recovery table
        for d, vals in res["timeframes"].items():
            rp = vals['recovery_prob']
            rp_color = "#00FF7F" if rp > 60 else "#FFD700" if rp > 40 else "#FF4B4B"
            rp_pct = min(100, int(rp))

            st.markdown(f"""
<div class="glass-card">
<div style="display:flex; justify-content:space-between; align-items:center;">
<div>
<div class="section-label">HISTORICAL {d}-DAY RECOVERY ODDS</div>
<div style="color:#888; font-size:11px;">Out of {vals['scenarios_found']} exact similar market setups:</div>
</div>
<div style="text-align:right;">
<div style="color:{rp_color}; font-size:28px; font-weight:800;">{rp:.0f}%</div>
<div style="color:#888; font-size:10px;">avg {vals['avg_recovery_days']:.1f} days to rescue</div>
</div>
</div>
<div class="meter-track" style="margin-top:8px;">
<div class="meter-fill" style="width:{rp_pct}%; background:{rp_color};"></div>
</div>
</div>
""", unsafe_allow_html=True)
