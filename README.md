# 🦅 DAVID-V6.6.6: THE ALPHA SNIPER

> **Nifty 50 High-Precision Directional Engine & Momentum Scanner**
> 
> *A next-generation trading system leveraging a Dual-Pillar Meta-Ensemble (CatBoost/LightGBM + PyTorch LSTM), Regime-Aware Routing, and Blind Walk-Forward Validation.*

---

## 🚀 1. What is David-v6.6.6?

David-v6.6.6 is a major evolution of the original Oracle. We stripped away the ambiguity of "Sideways" training and rebuilt the brain as a high-conviction **Binary Engine**. It solves one problem: **"Where is the structural momentum in the next 5 days, and how strong is it?"**

### The "Alpha Sniper" Performance
*   **Directional Accuracy:** **65.3%** (at 70% Conviction, Verified via 15-month Blind Audit).
*   **Strategy Focus:** Engineered for **Naked Option Buying** (Alpha Strategy) where 65% accuracy with 1:2 R:R maximizes compounding.
*   **Safety Filter:** Integrated **Regime-Aware Routing** to identify and block high-volatility "falling knife" scenarios.

---

## 🧠 2. The Multi-Pillar Architecture

David-v6.6.6 uses a **Dual-Pillar** fusion logic to catch both statistical edges and structural sequences.

```mermaid
graph TD
    A[Raw Data: NIFTY, VIX, S&P 500] --> B[Feature Forge: 53 Volatility-Adjusted Indicators]
    B --> C{The Meta-Ensemble}
    
    subgraph Pillar_1 [Pillar 1: Tree-Based Regime Ensemble]
        C --> D1[HMM Regime Switcher]
        D1 --> D2[Trending Sub-Model]
        D1 --> D3[Choppy Sub-Model]
    end
    
    subgraph Pillar_2 [Pillar 2: PyTorch LSTM Sequence Model]
        C --> E1[10-Day Temporal Window]
        E1 --> E2[Momentum Sequence Extraction]
    end
    
    D2 & D3 & E2 --> F[Weighted Probability Fusion]
    F --> G[Directional Verdict: UP / DOWN]
    
    G --> H[EDGE RADAR: STRIKE RECOMMENDER]
    H --> I[Action: Naked Buying / Credit Spreads / Stay-Cash]
```

---

## 🌊 3. The Data Flow (Pillar to Pillar)

1.  **Ingestion:** `data_engine.py` fetches the 10-year historical daily candles and real-time VIX / S&P 500 data.
2.  **Transformation:** `feature_forge.py` cleans the data and creates 53 **Volatility-Adjusted features**. It uses realized volatility to set the "Binary Threshold"—a 1% move in a calm market is treated as a major signal, while the same move in high-vol is ignored as noise.
3.  **The Secrets (The Meta-Fusion):**
    *   **The Tree Brain (60% Weight):** Uses Gradient Boosting to find statistical relationships. It is "Regime-Aware," meaning it swaps its internal model based on whether the HMM sees the market as Trending or Choppy.
    *   **The Sequence Brain (40% Weight):** Powered by **PyTorch LSTM**. It doesn't look at single days; it looks at "Shapes." It identifies if the last 10 days of price action are building towards an explosion or a collapse.
4.  **The Verdict:** The system combines these two perspectives. If both the Trees and the LSTM agree, the **Trust Score** hits **A+**, and the sniper fires.

---

## 📈 4. The V6.6.6 Edge Matrix

In v6.6.6, we removed "Sideways" to give you a pure momentum scanner.

| Setup Regime | AI Direction | Signal Value | Best Execution |
| :--- | :--- | :--- | :--- |
| **MILD BULLISH** | UP | 🏆 **HIGH** | Naked Call Buy (Target 1:2) |
| **MILD BEARISH** | DOWN | 🏹 **MEDIUM** | Naked Put Buy / Bear Call Spread |
| **STRONG TRAUMA**| Any | 🛑 **NONE** | **STAY IN CASH** (AI Blocks Trade) |
| **SIDEWAYS** | UP/DOWN | 🥦 **STABLE** | OTM Credit Spreads (Bull Put) |

---

## 🛠️ 5. Strategy Playbook (v6.6.6 Rules)

1.  **Check for Agreement:** Only trade when **Tree-Conf** and **LSTM-Conf** both lean the same way.
2.  **Monitor the Regime:** If the HMM says "Strong Bearish" or "Strong Bullish," the market is in a "Trauma" state—even if the AI is accurate, the price swings may be too large to handle.
3.  **Strike Selection:** Use the **Strike Recommender** on the dashboard. It calculates the 10-year historical survival of your strike *relative* to current VIX.
4.  **Exit Strategy:** Exit on **Day 5** or if the spot price moves **2%** against your entry.

---

## 📁 6. Core Project Files

*   `david_streamlit.py`: Your primary Trading Dashboard with sub-model confidence bars.
*   `models/meta_ensemble.py`: The primary fusion layer for Pillar 1 & 2.
*   `models/sequence_model.py`: The PyTorch LSTM neural network.
*   `models/regime_ensemble.py`: The HMM-based switching logic.
*   `feature_forge.py`: The volatility-adjusted target engine.
*   `train_models.py`: The core training pipeline (Quarterly Retraining).

---

## ⚠️ Disclaimer
*Trading Nifty 50 Options involves high risk. David v6.6.6 is an analytical tool, not a financial advisor. Use strict 2.0% stop-losses and verify all signals against your own risk tolerance.*
