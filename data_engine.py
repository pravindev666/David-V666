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
    Fetch and merge NIFTY + VIX + S&P 500 into a single DataFrame.
    Returns a clean, merged DataFrame ready for feature engineering.
    """
    print(f"\n{C.header('DATA ENGINE: Loading Market Data')}")
    print(f"{'─'*50}")
    
    nifty = fetch_symbol(NIFTY_SYMBOL, "nifty")
    vix = fetch_symbol(VIX_SYMBOL, "vix")
    sp500 = fetch_symbol(SP500_SYMBOL, "sp500")
    
    # Rename columns for merging
    vix_cols = vix[["date", "close"]].rename(columns={"close": "vix"})
    sp_cols = sp500[["date", "close"]].rename(columns={"close": "sp_close"})
    
    # Merge on date
    df = nifty.merge(vix_cols, on="date", how="left")
    df = df.merge(sp_cols, on="date", how="left")
    
    # Forward-fill only (bfill would leak future data — look-ahead bias)
    df["vix"] = df["vix"].ffill()
    df["sp_close"] = df["sp_close"].ffill()
    
    # Sort and clean
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["close"])
    
    print(f"\n  {C.GREEN}[OK] Merged dataset: {len(df)} trading days{C.RESET}")
    print(f"  {C.DIM}     Date range: {df['date'].min().date()} → {df['date'].max().date()}{C.RESET}")
    print(f"  {C.DIM}     Latest close: {df['close'].iloc[-1]:,.2f}{C.RESET}")
    
    return df


if __name__ == "__main__":
    df = load_all_data()
    print(f"\nColumns: {list(df.columns)}")
    print(df.tail())
