"""
DAVID PROPHETIC ORACLE — Data Engine
=====================================
Fetches NIFTY, VIX, S&P 500 daily OHLCV from yfinance (2015–now).
Caches to local CSVs with incremental sync.
Falls back to v3 CSVs if yfinance fails.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance is required. Install with: pip install yfinance")

from utils import DATA_DIR, NIFTY_SYMBOL, VIX_SYMBOL, SP500_SYMBOL, DATA_START_YEAR, C


def _csv_path(name):
    return os.path.join(DATA_DIR, f"{name}_daily.csv")


def _v3_fallback_path(name):
    """Try to find v3 CSV as fallback."""
    v3_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v3", "data")
    mapping = {
        "nifty": "NIFTY_50.csv",
        "vix": "VIX.csv",
        "sp500": "SP500.csv",
    }
    if name in mapping:
        path = os.path.join(v3_dir, mapping[name])
        if os.path.exists(path):
            return path
    return None


def fetch_symbol(symbol, name, start_year=DATA_START_YEAR):
    """
    Fetch daily OHLCV for a symbol from yfinance.
    Uses incremental sync — only downloads new data if CSV already exists.
    """
    csv_path = _csv_path(name)
    start_date = f"{start_year}-01-01"
    
    existing_df = None
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path, parse_dates=["date"])
        last_date = existing_df["date"].max()
        # Only fetch from last date onward
        start_date = (last_date - timedelta(days=5)).strftime("%Y-%m-%d")
        print(f"  {C.DIM}[SYNC] {name}: Incremental from {start_date}{C.RESET}")
    else:
        print(f"  {C.CYAN}[FETCH] {name}: Full download from {start_date}{C.RESET}")

    try:
        df = yf.download(symbol, start=start_date, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        
        # Ensure we have the right columns
        required = ["date", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                if col == "volume":
                    df["volume"] = 0
                else:
                    raise ValueError(f"Missing column: {col}")
        
        df = df[required].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["close"])
        
        # Merge with existing
        if existing_df is not None:
            combined = pd.concat([existing_df, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)
            df = combined
        
        df.to_csv(csv_path, index=False)
        print(f"  {C.GREEN}[OK] {name}: {len(df)} rows saved{C.RESET}")
        return df
        
    except Exception as e:
        print(f"  {C.YELLOW}[WARN] yfinance failed for {name}: {e}{C.RESET}")
        
        # Try v3 fallback
        fallback = _v3_fallback_path(name)
        if fallback:
            print(f"  {C.CYAN}[FALLBACK] Using v3 CSV: {fallback}{C.RESET}")
            df = pd.read_csv(fallback, parse_dates=["date"] if "date" in pd.read_csv(fallback, nrows=1).columns else [0])
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            if df.columns[0] != "date":
                df = df.rename(columns={df.columns[0]: "date"})
            return df
        
        # Try existing cached CSV
        if existing_df is not None:
            print(f"  {C.YELLOW}[CACHE] Using cached data: {len(existing_df)} rows{C.RESET}")
            return existing_df
        
        raise RuntimeError(f"Cannot load data for {name}. No cache, no fallback.")


def load_all_data():
    """
    Fetch and merge NIFTY + VIX + S&P 500 + Bank Nifty + FII/DII into a single DataFrame.
    """
    from utils import BANK_NIFTY_SYMBOL
    try:
        from nsepython import nse_fiidii
    except ImportError:
        print(f"  {C.YELLOW}[WARN] nsepython not found. Skipping FII/DII data.{C.RESET}")
        nse_fiidii = None
    
    print(f"\n{C.header('DATA ENGINE: Loading Market Data')}")
    print(f"{'─'*50}")
    
    nifty = fetch_symbol(NIFTY_SYMBOL, "nifty")
    bank_nifty = fetch_symbol(BANK_NIFTY_SYMBOL, "bank_nifty")
    vix = fetch_symbol(VIX_SYMBOL, "vix")
    sp500 = fetch_symbol(SP500_SYMBOL, "sp500")
    
    # ── FII/DII (Incremental logic) ──
    fii_dii_path = _csv_path("fii_dii")
    fii_dii_df = None
    if nse_fiidii:
        try:
            # Get latest FII/DII
            latest_fd = nse_fiidii()
            if latest_fd is not None and len(latest_fd) > 0:
                # Expecting list of dicts or DataFrame-like
                new_fd = pd.DataFrame(latest_fd)
                if not new_fd.empty:
                    # Rename as per standard
                    new_fd["date"] = pd.to_datetime(new_fd["date"])
                    # Ensure numeric types (FII/DII netValue can sometimes be strings)
                    fii_net = pd.to_numeric(new_fd[new_fd["category"] == "FII/FPI"]["netValue"].iloc[0], errors='coerce') if "FII/FPI" in new_fd["category"].values else 0
                    dii_net = pd.to_numeric(new_fd[new_fd["category"] == "DII"]["netValue"].iloc[0], errors='coerce') if "DII" in new_fd["category"].values else 0
                    
                    row = pd.DataFrame([{"date": new_fd["date"].iloc[0], "fii_net": float(fii_net), "dii_net": float(dii_net)}])
                    
                    if os.path.exists(fii_dii_path):
                        fii_dii_df = pd.read_csv(fii_dii_path, parse_dates=["date"])
                        fii_dii_df = pd.concat([fii_dii_df, row], ignore_index=True).drop_duplicates(subset=["date"], keep="last")
                    else:
                        fii_dii_df = row
                    
                    fii_dii_df.to_csv(fii_dii_path, index=False)
                    print(f"  {C.GREEN}[OK] FII/DII: Data synced for {fii_dii_df['date'].max().date()}{C.RESET}")
        except Exception as e:
            print(f"  {C.YELLOW}[WARN] FII/DII fetch failed: {e}{C.RESET}")
            if os.path.exists(fii_dii_path):
                fii_dii_df = pd.read_csv(fii_dii_path, parse_dates=["date"])

    # ── MERGE PROCESS ──
    vix_cols = vix[["date", "close"]].rename(columns={"close": "vix"})
    sp_cols = sp500[["date", "close"]].rename(columns={"close": "sp_close"})
    bn_cols = bank_nifty[["date", "close"]].rename(columns={"close": "bn_close"})
    
    df = nifty.merge(vix_cols, on="date", how="left")
    df = df.merge(sp_cols, on="date", how="left")
    df = df.merge(bn_cols, on="date", how="left")
    
    if fii_dii_df is not None:
        df = df.merge(fii_dii_df, on="date", how="left")
        df["fii_net"] = df["fii_net"].ffill().fillna(0)
        df["dii_net"] = df["dii_net"].ffill().fillna(0)

    # Forward-fill
    df["vix"] = df["vix"].ffill()
    df["sp_close"] = df["sp_close"].ffill()
    df["bn_close"] = df["bn_close"].ffill()
    
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["close"])
    
    print(f"\n  {C.GREEN}[OK] Merged dataset: {len(df)} trading days{C.RESET}")
    return df


if __name__ == "__main__":
    df = load_all_data()
    print(f"\nColumns: {list(df.columns)}")
    print(df.tail())
