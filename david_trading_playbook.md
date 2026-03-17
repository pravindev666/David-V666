# 📖 David-V2 Trading Playbook

This document defines exactly how to interpret David-V2's signals and how to filter out "bad" AI predictions to protect your capital.

## 1. The Golden Rule: The "Bouncer" Principle
In a high-end club, you have the **Promoter** (The AI Verdict) and the **Bouncer** (The Market Regime). 
*   **The Promoter (AI):** Always wants to get trades through. He is optimistic and looks for "edges."
*   **The Bouncer (Regime):** His job is to keep you safe. If he says "Bearish," he doesn't care how excited the AI is.

> [!IMPORTANT]
> **NEVER trade against the Bouncer.** If the AI says UP, but the Regime is BEARISH, the trade is **immediately disqualified**.

---

## 2. Interpretation Checklist

| Indicator | Meaning | Threshold for SELL PE (Bullish) | Threshold for SELL CE (Bearish) |
|---|---|---|---|
| **AI Conviction** | Probabilities of 3-class prediction | **UP > 50%** | **DOWN > 50%** |
| **Market Regime** | Price/VIX structural reality | **BULLISH** or **SIDEWAYS** | **BEARISH** or **SIDEWAYS** |
| **Whipsaw Risk** | Chaos & Noise level | **< 35% (Green)** | **< 35% (Green)** |
| **Trust Score** | The Final Math filter | **> 60% (A/A+ or B)** | **> 60% (A/A+ or B)** |

---

## 3. The Super-Matrix (Top Setups)

*Based on 700-day corrected backtest with 50% AI confidence threshold and >= 60% Trust grade.*

### 5-Day Hold Strategy (Primary)

| Market Regime | Whipsaw | AI Verdict | History | Avg Profit (₹/Lot) | Win Rate | Sample Size |
|---|---|---|---|---|---|---|
| **TRENDING** | **Clear** | **High (50%+)** | **A/A+** | **₹57,961** | **100.0%** | **10** |
| VOLATILE | Clear | High (50%+) | A/A+ | ₹36,351 | 100.0% | 14 |
| VOLATILE | Choppy | High (50%+) | A/A+ | ₹34,061 | 100.0% | 3 |
| **SIDEWAYS** | **Clear** | **High (50%+)** | **A/A+** | **₹23,894** | **99.4%** | **180** |
| SIDEWAYS | Clear | High (50%+) | B | ₹25,544 | 100.0% | 69 |
| SIDEWAYS | Choppy | High (50%+) | A/A+ | ₹22,679 | 100.0% | 63 |
| CALM | Choppy | High (50%+) | A/A+ | ₹21,099 | 100.0% | 25 |

### 3-Day Hold Strategy (Aggressive Frequency)

| Market Regime | Whipsaw | AI Verdict | History | Avg Profit (₹/Lot) | Win Rate | Sample Size |
|---|---|---|---|---|---|---|
| **TRENDING** | **Clear** | **High (50%+)** | **A/A+** | **₹45,377** | **100.0%** | **10** |
| **SIDEWAYS** | **Clear** | **High (50%+)** | **A/A+** | **₹14,000** | **86.1%** | **180** |
| SIDEWAYS | Clear | High (50%+) | B | ₹14,594 | 84.0% | 69 |

> [!TIP]
> **The God-Mode Setup:** When the Regime is **TRENDING**, Whipsaw is **CLEAR**, and AI Conviction is **HIGH**, the system averages huge profit per lot with a near-perfect historical record across both 3-day and 5-day holds.
> **Volume Play:** By lowering the Trust filter to B-grade and Conviction to 50%, the **SIDEWAYS | Clear | High | A/A+ or B** setups offer reliable, high-frequency "Bread & Butter" trading (~250 total trades in 700 days, or ~90/year).

---

## 4. Case Study: The "March 13th" Fail
In your screenshot, David-V2 shows exactly why it is a robust system:

1.  **AI Verdict:** UP (73.7%) — The models were seeing bullish structural components (Adx Trend, BB Squeeze).
2.  **Regime:** **MILD BEARISH** 🛑 — This was your Warning Light.
3.  **Whipsaw:** 25% — Market was trending, but the trend it recognized was likely the *downward* volatility spike.

**Why did it fail?**
AI models use historical EOD data. An intraday "crazy drop" is often caused by high-frequency liquidations or news events that the EOD models don't see until after the candle closes. 

**The Solution:**
Wait for the **Regime** and **AI** to sync up. If you see AI UP and Regime BEARISH, it means the market is in a "V-Bottom" or a "Dead Cat Bounce" attempt. Unless the Regime flips to SIDEWAYS or BULLISH, you do not sell a Put.

---

## 4. Entry Playbook (The System Flow)

1.  **Check Regime First:** If Bearish, ignore Bullish signals. If Bullish, ignore Bearish signals.
2.  **Check AI Next:** Is the conviction > 45%?
3.  **Go to "Strike Recommender":** Look at the **Trust Score**.
4.  **Check Historical Confluence Bars:** Are the AI and History bars *both* green? 
5.  **Only trade if all 4 are GREEN.**

---

## 5. Exit Playbook
*   **Stop Loss:** Use the recommended **2x Premium** stop loss.
*   **Firefight:** Use the **Firefight Recovery Calculator** in Strategy Lab if Nifty drops 2% against you. If recovery odds are <30%, cut the trade instantly.
