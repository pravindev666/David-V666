# 🦅 DAVID-V6.6.6+: THE ALPHA SNIPER (Plus Edition)

> **Advanced Nifty 50 Directional Engine with Institutional Momentum**
> 
> *A state-of-the-art trading system leveraging a **3-Pillar Meta-Fusion** (CatBoost + PyTorch LSTM + Transformer Attention), Institutional Flow Analysis, and Sentiment-Aware Edge Detection.*

---

## 🦅 1. System Overview: Sniper Plus
The David V6.6.6+ system is designed for high-conviction directional trading on the Nifty 50 index. It specifically addresses the "Trauma" of extreme regime shifts (like March 2026) using hard mechanical guardrails and institutional data points.

### 🛡️ Hard Guardrails
- **Hard Regime Gate:** HMM-detected `STRONG BULL/BEAR` regimes trigger an automatic `HOLD` verdict to prevent catching falling knives.
- **Adaptive Retraining:** The system monitors regime shifts. When the market "vibe" changes, `train_models.py` triggers an automatic end-to-end retrain.
- **Honest Edge Verified:** Evaluated using a 3-window strictly blinded audit (Zero Look-Ahead), achieving a verified **69.2% Average Win Rate**.

---

## ⚡ Quick Start: Get Running in 60 Seconds

1.  **Sync & Train:** Run the full pipeline to get the latest models.
    ```bash
    python train_models.py --force
    ```
2.  **Launch Dashboard:** Open the institutional terminal.
    ```bash
    streamlit run david_streamlit.py
    ```
3.  **Audit Strategy:** Run an out-of-sample backtest.
    ```bash
    python backtest_david.py --months 3 --report
    ```

---

## 🛰️ 2. The Data Engine (Institutional Sync)
David maintains a 10-year historical dataset across 5 core asset classes without redundant downloads.

### 🔄 Incremental Sync Logic (`data_engine.py`)
- **Delta-Only Fetching:** The engine checks your local `data/*.csv` files and only downloads data from the `last_date` to `today`. 
- **Institutional Sources:** 
    - **Indices:** Nifty 50 (`^NSEI`), Bank Nifty (`^NSEBANK`), S&P 500 (`^GSPC`).
    - **Volatility:** VIX (`^INDIAVIX`) with term structure analysis.
    - **Sentiment:** FII/DII Net Flow and PCR (Put-Call Ratio) via `nsepython`.

---

## 🏟️ 3. Training Workflow (3-Pillar Fusion)
The training pipeline (`train_models.py`) builds three independent "Pillars" that see the market through different mathematical lenses.

1.  **Regime Awareness (HMM):** Identifies the 5 hidden states of the market.
2.  **Statistical Edge (Trees):** CatBoost and LightGBM models specializing in TRENDING vs CHOPPY data splits.
3.  **Linear Memory (LSTM):** A 20-day sequence model identifying cascading momentum patterns.
4.  **Non-Linear Attention (Transformer):** Multi-head attention identifying which specific events in the last month matter most.

**Production Command:** `python train_models.py --force` (Forces a current-day sync & rebuild).

---

## 💻 4. Operational Console (Streamlit UI)
The primary interface (`david_streamlit.py`) provides an institutional-grade terminal for live signal analysis.

*   **Probability Gauge:** Blended conviction score from all 3 pillars (45/30/25 weights).
*   **The Edge Radar:** A 0-100 score filtering trades by VIX spreads, PCR Z-scores, and Model Agreement.
*   **Survival Heatmaps:** Historical probability of strike survival for Bull Put / Bear Call spreads.
*   **Command:** `streamlit run david_streamlit.py`

---

## 🧪 5. The Alpha Feature Matrix
| Feature | Code Logic | Why it works |
| :--- | :--- | :--- |
| **VIX Term Spread** | `far_vix - near_vix` | Backwardation in VIX signals structural fear. |
| **PCR Z-Score** | `(raw - 5d_avg) / std` | Identifies extreme sentiment traps before they snap. |
| **FII Interaction** | `FII Flow * TrendIndex` | FII selling in a trending bear market is a "High Trust" signal. |
| **Bank Nifty Leads** | `bn_returns_lag1/2` | Catches the "Banking Drag" or "Banking Rally" leading Nifty. |

---

## 📁 6. Project Manifest (File Breakdown)

### 🧠 Core Models
- `models/meta_ensemble.py`: The 3-pillar fusion layer and trauma gates.
- `models/regime_detector.py`: HMM-based market state classification.
- `models/sequence_model.py`: 20-day Memory (LSTM).
- `models/transformer_model.py`: Attention Pillar (Transformer).

### 🛠️ Infrastructure
- `data_engine.py`: Incremental CSV persistence and institutional sync.
- `feature_forge.py`: The engineering forge (60+ alpha features).
- `train_models.py`: Unified retraining orchestrator.
- `utils.py`: Shared constants, formatting, and `LOT_SIZE=65` config.

### 🧪 Backtesting
- `backtest_david.py`: The "Gold Standard" auditor for Spreads/SL logic.
- `accuracy_audit.py`: Direct prediction-to-price verification.

---

## ⚠️ Risk Disclaimer
Trading Nifty 50 Options involves significant risk. David V6.6.6+ is an analytical tool; all signals should be verified against your own risk tolerance. **Always use stop-losses.**
