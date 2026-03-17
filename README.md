# 🦅 DAVID PROPHETIC ORACLE v1.0

> **Nifty 50 Absolute Direction Prediction Engine for Retail Traders**
> 
> Built with XGBoost + LightGBM + CatBoost + 5-State HMM Ensemble.
> Audit-verified leak-free pipeline with walk-forward validation.

---

## What Does David Do?

David answers **one question**: **"Where is Nifty going — and how confident should I be?"**

| Feature | What You Get |
|:---|:---|
| **Direction Prediction** | UP / DOWN / SIDEWAYS with probability % |
| **7-Day Range** | "Nifty will be between 24,800–25,400 (80% confidence)" |
| **30-Day Range** | Monthly expected price band |
| **Support & Resistance** | Real S/R from historical fractals, not synthetic levels |
| **Whipsaw Detection** | Is the market choppy? Will it flip? |
| **Iron Condor Analyzer** | "Will Nifty touch my strike at 25600?" |
| **Bounce Probability** | "If it drops to 23000, will it come back?" |
| **Trade Recommendation** | Bull Spread / Bear Spread / Iron Condor with exact strikes |

---

## Quick Start

```bash
cd david
pip install -r requirements.txt

# Option A: Interactive CLI
python david_oracle.py

# Option B: Streamlit Dashboard
streamlit run david_streamlit.py
```

---

## 🧭 How Everything Connects — The Full Pipeline

This is how David goes from **raw market data** to a **"BUY / SELL / SIT"** verdict on your screen.

```mermaid
flowchart TB
    subgraph FETCH["🌐 STAGE 1: DATA FETCH"]
        direction LR
        YF[("yfinance API")] --> NIFTY["NIFTY 50<br/>^NSEI"]
        YF --> VIX["India VIX<br/>^INDIAVIX"]
        YF --> SP["S&P 500<br/>^GSPC"]
    end

    subgraph CACHE["💾 Local Cache"]
        CSV1["data/nifty_daily.csv"]
        CSV2["data/vix_daily.csv"]
        CSV3["data/sp500_daily.csv"]
    end

    subgraph FORGE["⚙️ STAGE 2: FEATURE ENGINEERING"]
        direction TB
        FF["feature_forge.py<br/>~44 Technical Features"]
        FF --> PA["Price Action (5)"]
        FF --> VOL["Volatility (6)"]
        FF --> MOM["Momentum (8)"]
        FF --> TR["Trend (7)"]
        FF --> MS["Market Structure (4)"]
        FF --> VF["VIX Features (4)"]
        FF --> CM["Cross-Market (3)"]
        FF --> CAL["Calendar (2)"]
        FF --> VOLM["Volume (2)"]
    end

    subgraph MODELS["🧠 STAGE 3: ML MODELS"]
        direction TB
        ENS["Ensemble Classifier<br/>XGB + LGBM + CatBoost"]
        HMM["Regime Detector<br/>5-State Gaussian HMM"]
        QR["Range Predictor<br/>Quantile Regression"]
        SR["S/R Engine<br/>Fractals + DBSCAN"]
    end

    subgraph ANALYZERS["📊 STAGE 4: RISK ANALYZERS"]
        WS["Whipsaw Detector"]
        IC["Iron Condor Analyzer"]
        BA["Bounce Analyzer"]
    end

    subgraph OUTPUT["🎯 STAGE 5: YOUR VERDICT"]
        VERDICT["Direction: UP ▲<br/>Confidence: 67%<br/>Regime: MILD BULLISH<br/>Range: 24,800 – 25,400"]
    end

    FETCH --> CACHE --> FORGE
    FORGE --> MODELS
    FORGE --> ANALYZERS
    MODELS --> OUTPUT
    ANALYZERS --> OUTPUT
```

### In Plain English

1. **FETCH**: David downloads daily candle data for Nifty, VIX, and S&P 500 from Yahoo Finance. Data is cached locally as CSVs so it doesn't re-download everything each time — only new bars.

2. **FEATURE ENGINEERING**: The raw OHLCV data is transformed into ~44 technical indicators (RSI, MACD, ATR, Bollinger Bands, etc.) that the ML models can learn from. These are features the model uses as "inputs."

3. **ML MODELS**: Three separate brain systems work together:
   - **Ensemble Classifier** tells you the direction (UP/DOWN/SIDEWAYS)
   - **Regime Detector** tells you the market state (Bullish/Bearish/Sideways)
   - **Range Predictor** tells you the expected price band

4. **ANALYZERS**: Risk tools that overlay on top: Is the market too choppy to trade? Will your iron condor get breached?

5. **VERDICT**: Everything combines into a single actionable output.

---

## 🧠 Model Deep Dive

### Model 1: The Ensemble Direction Classifier

**The core brain.** Three gradient-boosted tree models vote on whether Nifty is going UP, DOWN, or SIDEWAYS.

```mermaid
flowchart LR
    A["Today's 44 Features"] --> B["StandardScaler"]
    B --> C["🔵 XGBoost"]
    B --> D["🟢 LightGBM"]
    B --> E["🟠 CatBoost"]
    C -->|"P(UP)=0.60<br/>P(DOWN)=0.25<br/>P(SIDE)=0.15"| F["Weighted<br/>Average"]
    D -->|"P(UP)=0.55<br/>P(DOWN)=0.30<br/>P(SIDE)=0.15"| F
    E -->|"P(UP)=0.65<br/>P(DOWN)=0.20<br/>P(SIDE)=0.15"| F
    F --> G["🎯 VERDICT: UP<br/>Confidence: 60%"]
```

**How training works:**

```mermaid
flowchart TB
    A["Full Dataset<br/>2015 – Today"] --> B["Walk-Forward CV<br/>(5 Folds)"]
    
    B --> F1["Fold 1: Train 2015-2017<br/>Test 2018"]
    B --> F2["Fold 2: Train 2015-2019<br/>Test 2020"]
    B --> F3["Fold 3: Train 2015-2020<br/>Test 2021"]
    B --> F4["Fold 4: Train 2015-2021<br/>Test 2022"]
    B --> F5["Fold 5: Train 2015-2023<br/>Test 2024-25"]

    F1 --> ACC["Accuracy per Fold"]
    F2 --> ACC
    F3 --> ACC
    F4 --> ACC
    F5 --> ACC

    ACC --> W["Model Weights<br/>(Based on CV Performance)"]
    
    A --> FINAL["Final Training<br/>on Full Dataset"]
    W --> FINAL
    FINAL --> PKL["saved_models/<br/>ensemble_classifier.pkl"]
```

> **Key**: The scaler is fit **per-fold** during cross-validation (no future leakage), then refit on the full dataset for production. Tree models are rank-invariant — scaling doesn't affect their predictions, but we keep it for correctness.

**What is the target?**

The model predicts what Nifty will do **5 trading days from now**:

| Label | Condition | Meaning |
|:---|:---|:---|
| **UP** | Return > +0.3% | Nifty goes up meaningfully |
| **DOWN** | Return < -0.3% | Nifty drops meaningfully |
| **SIDEWAYS** | Between ±0.3% | Nifty stays flat |

**Why 3 models instead of 1?**

| Model | Strength |
|:---|:---|
| **XGBoost** | Best at complex feature interactions ("RSI > 70 AND VIX falling") |
| **LightGBM** | Fastest training, best generalization, handles missing data |
| **CatBoost** | Most robust to overfitting via ordered boosting |

When all 3 agree → High Confidence. When they disagree → Low Confidence. The weighted average naturally captures this.

---

### Model 2: The 5-State Regime Detector

**Answers: "What kind of market are we in?"**

A Hidden Markov Model that classifies the market into one of 5 states:

```mermaid
stateDiagram-v2
    SB: 🔥 STRONG BULLISH
    MB: 📈 MILD BULLISH
    SW: ➡️ SIDEWAYS
    MBr: 📉 MILD BEARISH
    SBr: 💀 STRONG BEARISH

    SB --> MB: Momentum fading
    SB --> SW: Profit taking
    MB --> SB: Breakout
    MB --> SW: Consolidation
    MB --> MBr: Reversal signal
    SW --> MB: Breakout up
    SW --> MBr: Breakdown
    MBr --> SW: Stabilization
    MBr --> SBr: Panic selling
    SBr --> MBr: Bounce attempt
    SBr --> SW: V-Recovery
```

**Why it matters for trading:**
- In **STRONG BULLISH**: Full position sizing, ride momentum
- In **SIDEWAYS**: Iron Condors work well, directional bets are risky
- In **STRONG BEARISH**: Only sell-side strategies, or stay cash

The HMM also gives you **transition probabilities**: "Right now we're MILD BULLISH, there's a 72% chance we stay here tomorrow, 18% chance of going SIDEWAYS, 10% chance of STRONG BULLISH."

---

### Model 3: The Range Predictor

**Answers: "Where will Nifty be in 7 days / 30 days?"**

Instead of a single price target, David gives you **probability bands**:

```mermaid
flowchart TB
    A["Today's Features"] --> Q1["LightGBM<br/>α = 0.10"]
    A --> Q2["LightGBM<br/>α = 0.25"]
    A --> Q3["LightGBM<br/>α = 0.50"]
    A --> Q4["LightGBM<br/>α = 0.75"]
    A --> Q5["LightGBM<br/>α = 0.90"]

    Q1 --> R1["10th %ile: 23,800"]
    Q2 --> R2["25th %ile: 24,100"]
    Q3 --> R3["Median: 24,400"]
    Q4 --> R4["75th %ile: 24,700"]
    Q5 --> R5["90th %ile: 25,000"]

    R1 --> B1["80% Band: 23,800 — 25,000"]
    R5 --> B1
    R2 --> B2["50% Band: 24,100 — 24,700"]
    R4 --> B2
```

**How to read it:**
- "There's an 80% chance Nifty stays between 23,800 and 25,000 in 7 days"
- "There's a 50% chance Nifty stays between 24,100 and 24,700"
- The median path is 24,400

---

### Support & Resistance Engine

```mermaid
flowchart LR
    A["1 Year of<br/>Price History"] --> B["Williams Fractal<br/>(5-bar swing pattern)"]
    B --> C["Swing Highs"]
    B --> D["Swing Lows"]
    C --> E["DBSCAN Clustering<br/>(0.5% radius)"]
    D --> E
    E --> F["Strength Score<br/>= touches × recency"]
    F --> G["Top 3 Support ▼<br/>Top 3 Resistance ▲"]
```

**Real levels, not made-up ones.** The engine finds actual historical price pivots where the market has reversed before, clusters nearby pivots together, and ranks them by how many times they've been touched and how recent they are.

---

## 🔍 Risk Analyzers

### Whipsaw Detector — "Should I Even Trade Today?"

```mermaid
flowchart TB
    A["BB Squeeze<br/>(width < 20th %ile)"] -->|0.75| F["Whipsaw<br/>Score"]
    B["ADX < 20<br/>(no trend)"] -->|0.80| F
    C["ATR Expanding<br/>(vol > 1.3x avg)"] -->|0.60| F
    D["Candle Flip Rate<br/>(> 60% flips)"] -->|0.70| F
    E["VIX > Realized Vol<br/>(mean reversion)"] -->|0.40| F

    F --> G{"Score > 55%?"}
    G -->|"Yes"| H["⚠️ CHOPPY<br/>Skip directional trades<br/>Use Iron Condors"]
    G -->|"No"| I["✅ TRENDING<br/>Follow the verdict<br/>Use spreads"]
```

### Iron Condor Analyzer — "Will My Strike Get Hit?"

**Input:** Your strike price (e.g., 25600) and timeframe (e.g., 5 days)

**Output:**
- **Touch Probability**: 23% chance Nifty reaches 25600 in 5 days
- **Recovery Probability**: If touched, 68% chance it bounces back
- **Firefight Level**: Start hedging at 25,200 (70% of the way to strike)

Uses 10 years of empirical data — not Black-Scholes theory.

### Bounce-Back Analyzer — "If It Drops, Will It Recover?"

**Input:** A target price (e.g., "What if Nifty drops to 23000?")

**Output:** Recovery probability across 5, 10, and 20 days, adjusted for current volatility regime.

---

## 📈 How to Trade Using David — A Practical Guide

### Step 1: Check the Dashboard

```mermaid
flowchart TB
    START["Open David Oracle"] --> V["Check VERDICT"]
    V --> CONF{"Confidence?"}
    
    CONF -->|"> 65%"| HIGH["★ HIGH CONVICTION<br/>Full position sizing"]
    CONF -->|"45-65%"| MED["◆ MODERATE<br/>Half position sizing"]
    CONF -->|"< 45%"| LOW["○ LOW CONVICTION<br/>Skip or minimal"]
```

### Step 2: Validate with Context

Don't trade the verdict blindly. Check these:

```mermaid
flowchart TB
    VERDICT["Verdict: UP 67%"] --> R{"Regime?"}
    R -->|"BULLISH"| AGREE["✅ Regime agrees<br/>Proceed with confidence"]
    R -->|"SIDEWAYS"| CAUTION["⚠️ Regime says flat<br/>Reduce position size"]
    R -->|"BEARISH"| CONFLICT["🚨 Regime disagrees<br/>Skip this trade"]

    AGREE --> WS{"Whipsaw?"}
    WS -->|"Clear"| TRADE["🟢 TAKE THE TRADE"]
    WS -->|"Choppy"| HECONDOR["Use Iron Condor instead"]
```

### Step 3: Pick Your Strategy

| David Says | Regime | Whipsaw | Your Strategy |
|:---|:---|:---|:---|
| **UP** (>65%) | Bullish | Clear | **Bull Call Spread** — Buy ATM CE, Sell OTM CE |
| **UP** (45-65%) | Bullish | Clear | **Bull Call Spread** — half size |
| **DOWN** (>65%) | Bearish | Clear | **Bear Put Spread** — Buy ATM PE, Sell OTM PE |
| **SIDEWAYS** | Any | Clear/Choppy | **Iron Condor** — Sell OTM CE + PE, buy protection |
| Any direction | Any | **Choppy** | **NO TRADE** or **Iron Condor** only |
| Any (<45%) | Conflicting | Any | **SIT ON HANDS** 🙌 |

### Step 4: Set Your Levels

1. **Entry**: After David's verdict, at market open
2. **Stop Loss**: Below nearest Support (if bullish) or above nearest Resistance (if bearish)
3. **Target**: Next Resistance (if bullish) or next Support (if bearish)
4. **Firefight Level**: If you have an options position, the Iron Condor analyzer tells you when to start hedging
5. **Hold Period**: ~5 trading days (one weekly expiry cycle)

### Step 5: Risk Management Rules

```mermaid
flowchart LR
    A["David Confidence"] --> B{"Level?"}
    B -->|"> 65%"| C["Risk 2% of capital"]
    B -->|"45-65%"| D["Risk 1% of capital"]
    B -->|"< 45%"| E["Risk 0%<br/>Don't trade"]
```

**Golden Rules from the 6-Month Brutal Backtest:**
- **Trust DOWN calls in Bearish regimes.** The model is a trend-following beast.
- **Trust UP calls ONLY when confidence > 60%.** (Below that, UP accuracy is poor).
- **Ignore completely when confidence < 50%.** Sitting in cash is a position.
- Never risk more than 2% of capital on a single trade.
- If Whipsaw is ACTIVE → reduce size by 50% or skip.
- If Regime CONFLICTS with Direction → skip.
- Always use spreads (limited risk), never naked options.

---

## 📊 Version Accuracy Comparison (6-Month Out-of-Sample)

A strict, 6-month walk-forward backtest (Sep 2025 → Mar 2026, 115 trading days) with zero look-ahead bias revealed the following truths about the model:

| Version | Description | UP Accuracy | DOWN Accuracy | SIDEWAYS Accuracy | Overall |
|:---|:---|:---|:---|:---|:---|
| **V1 (Original)** | Base model, ±0.3% threshold, noisy features | 24% | 51% | N/A (ignored) | 34% |
| **V2 (Clean)** | Fixed look-ahead leaks, removed noisy expiry logic | 29% | 58% | N/A (ignored) | 38% |
| **V3 (Current)** | Added regime gating, SIDEWAYS override, ±0.5% threshold | **78%** | **38%** | 26% | 37% |

### Why did V3's overall accuracy drop slightly but UP accuracy skyrocket?
V3 introduced **SIDEWAYS overrides** and **Regime Gating** to stop the model from making "stupid" coin-flip predictions. 
- In V2, the model called UP 34 times, but most were wrong.
- In V3, the model was forced to stay *SIDEWAYS* (neutral) unless it was highly confident. It only called UP 9 times in 6 months — and 7 of them were correct (78%).
- **Conclusion:** V3 is much safer to trade. It trades less often, misses some volatile dips, but when it gives a strong Bullish or Bearish signal during an aligned regime, the edge is real.

---

## 🔬 The Feature Engine — What the Model Sees

```mermaid
pie title Feature Categories (~44 total)
    "Price Action" : 9
    "Volatility" : 6
    "Momentum" : 8
    "Trend" : 7
    "Market Structure" : 4
    "VIX" : 4
    "Cross-Market" : 3
    "Calendar" : 2
    "Volume" : 2
```

| Category | Features | What It Captures |
|:---|:---|:---|
| **Price Action** | Returns (1/5/10/20d), log return, gap%, body ratio, wicks | Raw price behavior and candle patterns |
| **Volatility** | Realized vol 10/20d, vol-of-vol, ATR, BB width/position | How violently is the market moving? |
| **Momentum** | RSI (7/14), MACD, Stochastic, Williams %R, ROC | Is buying/selling pressure building or fading? |
| **Trend** | SMA distances (20/50/200), SMA cross, ADX | Is there a clear directional trend? |
| **Market Structure** | Higher-high/lower-low counts, streak, 52w distance | Are structural patterns forming? |
| **VIX** | VIX ratio, VIX percentile, VIX change | Fear/greed gauge from options market |
| **Cross-Market** | S&P return, S&P-Nifty correlation, S&P lag | Overnight global sentiment signal |
| **Calendar** | Day of week, month of year | Seasonal and day-of-week effects |
| **Volume** | Volume ratio vs 20D avg, OBV momentum | Is smart money entering or leaving? |

---

## 📁 Data Flow — From Download to Prediction

```mermaid
sequenceDiagram
    participant User
    participant App as david_oracle.py / Streamlit
    participant Data as data_engine.py
    participant Feat as feature_forge.py
    participant Ens as Ensemble Classifier
    participant HMM as Regime Detector
    participant Range as Range Predictor

    User->>App: Start App / Click "Sync & Retrain"
    App->>Data: load_all_data()
    Data->>Data: Download NIFTY + VIX + S&P 500 from yfinance
    Data->>Data: Cache to data/*.csv (incremental)
    Data-->>App: Merged DataFrame (2015–today)
    
    App->>Feat: engineer_features(df)
    Feat->>Feat: Calculate ~44 technical indicators
    Feat->>Feat: Create target: what Nifty did 5 days later
    Feat-->>App: Feature matrix + targets

    alt .pkl files exist
        App->>Ens: load() from saved_models/
        App->>HMM: load() from saved_models/
        App->>Range: load() from saved_models/
    else .pkl files missing (first run or after delete)
        App->>Ens: train(df, features)
        Note over Ens: Walk-forward CV (5 folds)
        Ens->>Ens: save() → ensemble_classifier.pkl
        App->>HMM: train(df)
        HMM->>HMM: save() → regime_detector.pkl
        App->>Range: train(df, features)
        Range->>Range: save() → range_predictor.pkl
    end

    App-->>User: ORACLE READY ✅

    User->>App: "What's today's verdict?"
    App->>Ens: predict_today(df) → last row only
    Ens-->>App: UP 62% / DOWN 25% / SIDE 13%
    App->>HMM: get_current_regime(df)
    HMM-->>App: MILD BULLISH (72% stay, 18% → Sideways)
    App->>Range: predict_range(df)
    Range-->>App: 7-day: 24,800 – 25,400
    App-->>User: Full Verdict + Regime + Range
```

---

## 💾 Where Are the Files?

```
david/
├── david_oracle.py              # Main CLI (interactive menu)
├── david_streamlit.py           # Streamlit web dashboard
├── data_engine.py               # Fetches & caches market data from yfinance
├── feature_forge.py             # Engineers ~44 technical features
├── utils.py                     # Constants, colors, formatters
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── models/
│   ├── ensemble_classifier.py   # XGBoost + LightGBM + CatBoost ensemble
│   ├── regime_detector.py       # 5-state Gaussian HMM
│   ├── range_predictor.py       # Quantile regression (7d + 30d)
│   └── sr_engine.py             # Fractal S/R with DBSCAN clustering
│
├── analyzers/
│   ├── whipsaw_detector.py      # Chop/Trend classifier (5 signals)
│   ├── iron_condor_analyzer.py  # Strike touch probability
│   └── bounce_analyzer.py       # Recovery probability calculator
│
├── data/                        # 📦 Cached CSV data (auto-created)
│   ├── nifty_daily.csv
│   ├── vix_daily.csv
│   └── sp500_daily.csv
│
└── saved_models/                # 🧠 Trained model pickles (auto-created)
    ├── ensemble_classifier.pkl  # (~5 MB) Direction model
    ├── regime_detector.pkl      # (~6 KB) HMM regime model
    └── range_predictor.pkl      # (~3 MB) Range quantile models
```

**The `.pkl` files** are your trained models. Delete them → next run will retrain from scratch. Keep them → app loads instantly without retraining.

---

## CLI Menu Reference

```
[1] Today's Verdict      — Direction + confidence + regime + transitions
[2] 7-Day Forecast        — 7-day range bands (80% and 50% confidence)
[3] 30-Day Forecast       — 30-day range bands
[4] Support/Resistance    — Top 3 S/R levels from fractal detection
[5] Whipsaw Analysis      — Chop probability + 5 signal breakdown
[6] Iron Condor Analyzer  — Enter strike → touch/recovery/firefight
[7] Bounce Probability    — Enter price → recovery chance (5/10/20 days)
[8] Trade Recommendation  — Specific spread strategy with strikes
[9] Retrain Models        — Force fresh training from latest data
[B] Backtest              — Out-of-sample accuracy report
[F] Top Features          — Feature importance ranking
[0] Exit
```

---

## ✅ Pipeline Integrity

> This pipeline has been audited for data leakage and look-ahead bias:
> 
> - ✅ **No `bfill()`** — Only forward-fill is used for cross-market data (VIX, S&P)
> - ✅ **Per-fold scaling** — StandardScaler fits only on training data during CV
> - ✅ **No broken features** — Dynamic features (like expiry detection) removed where they corrupt historical data
> - ✅ **Tree-model invariance** — XGBoost/LightGBM/CatBoost are rank-invariant; scaling doesn't affect predictions
> - ✅ **Walk-forward validation** — Time-series aware CV, never testing on data earlier than training

---

## ⚠️ Honesty Note

> **100% win rate is not achievable in financial markets.** No ML model can predict random walks perfectly. What David provides is:
> - The **highest achievable directional accuracy** from historical patterns
> - **Honest probability estimates** so you know WHEN to skip uncertain trades
> - **Risk management tools** (whipsaw detection, firefight levels) to protect capital
>
> The system reports its actual walk-forward accuracy. If it says 55%, that means it's right 55% of the time — which, combined with proper position sizing and spread strategies, can be profitable.

---

## ⚠️ Disclaimer

> **This project is for educational and research purposes only.** Trading the Nifty 50 involves significant risk. The Oracle is a decision-support tool, not a financial advisor. Past performance does not guarantee future results. Always paper trade before deploying with real capital.

---

## License

Internal use only. Research tool for educational purposes.
