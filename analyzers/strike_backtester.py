"""
Lightweight Strike Backtester — Trust Score Engine
====================================================
Answers: "If I sold this exact strike on every similar day in history,
how often would I have won?"

Components:
  1. Survival Backtest (5 similarity filters)
  2. Expected Move Accuracy
  3. Regime-Conditional Rates
  4. MAE (Max Adverse Excursion)
  5. Trust Score (weighted composite → Grade A+/A/B/C/D)
"""

import numpy as np
import pandas as pd
import math


def _classify_regime(row):
    """Classify market regime from available features."""
    adx = row.get("adx", 20)
    vix = row.get("vix", 15)
    rvol = row.get("realized_vol_20", 0.15) if not pd.isna(row.get("realized_vol_20", np.nan)) else 0.15

    if adx > 25 and rvol > 0.18:
        return "TRENDING"
    elif vix > 20 or rvol > 0.22:
        return "VOLATILE"
    elif adx < 18 and rvol < 0.13:
        return "CALM"
    else:
        return "SIDEWAYS"


def _wilson_ci(wins, total, z=1.96):
    """Wilson confidence interval for proportions."""
    if total == 0:
        return 0.0, 0.0, 1.0
    p = wins / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denom
    return max(0, centre - spread), min(1, centre + spread), p


def backtest_strike_survival(df, spot, strike, side="PE", holding_days=7):
    """
    Component 1: Raw survival backtest.
    Scans similar days in history and checks if the strike would have survived.
    """
    if len(df) < 100:
        return {"survival": 0.5, "ci_low": 0, "ci_high": 1, "sample_size": 0, "matches": []}

    otm_pct = abs(spot - strike) / spot
    today = df.iloc[-1]
    today_vix = today.get("vix", 15)
    today_rvol = today.get("realized_vol_20", 0.15) if not pd.isna(today.get("realized_vol_20", np.nan)) else 0.15
    today_rsi = today.get("rsi_7", 50) if not pd.isna(today.get("rsi_7", np.nan)) else 50
    today_adx = today.get("adx", 20) if not pd.isna(today.get("adx", np.nan)) else 20
    today_regime = _classify_regime(today)

    survived = 0
    total = 0
    mae_list = []

    scan_end = len(df) - holding_days - 1
    for i in range(50, scan_end):
        row = df.iloc[i]
        row_vix = row.get("vix", 15)
        row_rvol = row.get("realized_vol_20", 0.15) if not pd.isna(row.get("realized_vol_20", np.nan)) else 0.15
        row_rsi = row.get("rsi_7", 50) if not pd.isna(row.get("rsi_7", np.nan)) else 50
        row_adx = row.get("adx", 20) if not pd.isna(row.get("adx", np.nan)) else 20

        # 5 similarity filters
        if abs(row_vix - today_vix) / max(today_vix, 1) > 0.20:
            continue
        if abs(row_rvol - today_rvol) / max(today_rvol, 0.01) > 0.30:
            continue

        rsi_bucket_match = (int(row_rsi / 20) == int(today_rsi / 20))
        adx_bucket_match = (int(row_adx / 15) == int(today_adx / 15))
        regime_match = (_classify_regime(row) == today_regime)

        if not (rsi_bucket_match and adx_bucket_match and regime_match):
            continue

        # Check survival
        row_spot = float(row["close"])
        if side == "PE":
            sim_strike = row_spot * (1 - otm_pct)
        else:
            sim_strike = row_spot * (1 + otm_pct)

        window = df.iloc[i + 1: i + 1 + holding_days]
        if len(window) == 0:
            continue

        if side == "PE":
            worst = float(window["low"].min())
            breach = worst <= sim_strike
            mae = max(0, (row_spot - worst) / row_spot)
        else:
            worst = float(window["high"].max())
            breach = worst >= sim_strike
            mae = max(0, (worst - row_spot) / row_spot)

        total += 1
        if not breach:
            survived += 1
        mae_list.append(mae)

    if total == 0:
        return {"survival": 0.5, "ci_low": 0, "ci_high": 1, "sample_size": 0, "mae_avg": 0, "mae_p95": 0}

    ci_low, ci_high, survival_pct = _wilson_ci(survived, total)

    mae_arr = np.array(mae_list)
    return {
        "survival": survival_pct,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "sample_size": total,
        "survived": survived,
        "mae_avg": float(np.mean(mae_arr)) if len(mae_arr) > 0 else 0,
        "mae_p95": float(np.percentile(mae_arr, 95)) if len(mae_arr) > 0 else 0,
    }


def expected_move_accuracy(df, spot, vix, holding_days=7):
    """Component 2: Is Spot × VIX × √(DTE/365) conservative or dangerous?"""
    iv = vix / 100
    formula_move = spot * iv * math.sqrt(holding_days / 365)

    actual_moves = []
    for i in range(50, len(df) - holding_days):
        s = float(df.iloc[i]["close"])
        future = df.iloc[i + 1: i + 1 + holding_days]
        if len(future) == 0:
            continue
        max_up = float(future["high"].max()) - s
        max_down = s - float(future["low"].min())
        actual_moves.append(max(max_up, max_down))

    if not actual_moves:
        return {"accuracy_score": 50, "formula_move": formula_move, "avg_actual": 0, "conservative_pct": 50}

    actual = np.array(actual_moves)
    conservative_pct = float(np.mean(actual < formula_move)) * 100

    return {
        "accuracy_score": conservative_pct,
        "formula_move": formula_move,
        "avg_actual": float(np.mean(actual)),
        "p95_actual": float(np.percentile(actual, 95)),
        "conservative_pct": conservative_pct,
    }


def regime_conditional_survival(df, spot, strike, side="PE", holding_days=7):
    """Component 3: Survival split by regime."""
    otm_pct = abs(spot - strike) / spot
    regimes = {"TRENDING": [0, 0], "SIDEWAYS": [0, 0], "VOLATILE": [0, 0], "CALM": [0, 0]}

    for i in range(50, len(df) - holding_days - 1):
        row = df.iloc[i]
        regime = _classify_regime(row)
        row_spot = float(row["close"])

        if side == "PE":
            sim_strike = row_spot * (1 - otm_pct)
        else:
            sim_strike = row_spot * (1 + otm_pct)

        window = df.iloc[i + 1: i + 1 + holding_days]
        if len(window) == 0:
            continue

        if side == "PE":
            breach = float(window["low"].min()) <= sim_strike
        else:
            breach = float(window["high"].max()) >= sim_strike

        regimes[regime][1] += 1
        if not breach:
            regimes[regime][0] += 1

    result = {}
    for r, (wins, total) in regimes.items():
        if total > 0:
            result[r] = {"survival": wins / total, "sample": total, "wins": wins}
        else:
            result[r] = {"survival": 0, "sample": 0, "wins": 0}

    return result


def compute_trust_score(survival_data, move_accuracy, regime_data, otm_pts, spot):
    """Component 5: Weighted composite trust score → Grade."""
    survival_score = survival_data.get("survival", 0.5) * 100
    accuracy_score = move_accuracy.get("conservative_pct", 50)
    cushion_pct = otm_pts / spot * 100

    # Current regime survival
    today_regime = "SIDEWAYS"
    for r_data in regime_data.values():
        pass  # We use the overall regime_data
    # Get the best matching regime
    regime_scores = [v["survival"] * 100 for v in regime_data.values() if v["sample"] > 10]
    regime_score = np.mean(regime_scores) if regime_scores else 50

    # Cushion score: 400pt on 25000 = 1.6% → score = cushion * 50, capped at 100
    cushion_score = min(100, cushion_pct * 50)

    # Weighted: Survival 40% + Accuracy 20% + Regime 20% + Cushion 20%
    trust = (survival_score * 0.40 + accuracy_score * 0.20 +
             regime_score * 0.20 + cushion_score * 0.20)

    if trust >= 85:
        grade = "A+"
    elif trust >= 75:
        grade = "A"
    elif trust >= 60:
        grade = "B"
    elif trust >= 45:
        grade = "C"
    else:
        grade = "D"

    return {
        "trust_score": round(trust, 1),
        "grade": grade,
        "survival_component": round(survival_score, 1),
        "accuracy_component": round(accuracy_score, 1),
        "regime_component": round(regime_score, 1),
        "cushion_component": round(cushion_score, 1),
    }


def full_strike_analysis(df, spot, strike, side="PE", vix=15.0, holding_days=7):
    """Run all 5 components and return complete analysis."""
    survival = backtest_strike_survival(df, spot, strike, side, holding_days)
    move_acc = expected_move_accuracy(df, spot, vix, holding_days)
    regime = regime_conditional_survival(df, spot, strike, side, holding_days)
    otm_pts = abs(spot - strike)
    trust = compute_trust_score(survival, move_acc, regime, otm_pts, spot)

    mae_rupees = survival.get("mae_avg", 0) * spot * 65  # in ₹ per lot
    p95_mae_rupees = survival.get("mae_p95", 0) * spot * 65

    return {
        **trust,
        "survival_pct": round(survival["survival"] * 100, 1),
        "ci_low": round(survival["ci_low"] * 100, 1),
        "ci_high": round(survival["ci_high"] * 100, 1),
        "sample_size": survival["sample_size"],
        "survived": survival.get("survived", 0),
        "mae_rupees": round(mae_rupees),
        "p95_mae_rupees": round(p95_mae_rupees),
        "formula_move": round(move_acc.get("formula_move", 0)),
        "avg_actual_move": round(move_acc.get("avg_actual", 0)),
        "regime_data": regime,
        "holding_days": holding_days,
    }


def get_survival_history(df, otm_pct, side="PE", holding_days=7, window=20):
    """
    Get rolling survival rate over time for the chart.
    Returns a DataFrame with date, rolling_survival for plotting.
    """
    dates = []
    survivals = []

    for i in range(max(100, len(df) - 500), len(df) - holding_days - 1):
        row = df.iloc[i]
        row_spot = float(row["close"])

        if side == "PE":
            sim_strike = row_spot * (1 - otm_pct)
        else:
            sim_strike = row_spot * (1 + otm_pct)

        future = df.iloc[i + 1: i + 1 + holding_days]
        if len(future) == 0:
            continue

        if side == "PE":
            breach = float(future["low"].min()) <= sim_strike
        else:
            breach = float(future["high"].max()) >= sim_strike

        dates.append(row.get("date", i))
        survivals.append(0 if breach else 1)

    if not dates:
        return pd.DataFrame()

    result = pd.DataFrame({"date": dates, "survived": survivals})
    result["rolling_survival"] = result["survived"].rolling(window, min_periods=5).mean() * 100
    return result
