# -*- coding: utf-8 -*-
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
import requests
import time

def fetch_data(symbol: str = "ADAUSD", total_days: int = 100, interval: str = "15m") -> pd.DataFrame:
    """Fetch OHLC data from Delta Exchange API."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, f"data_{symbol}_{interval}.csv")

    # ISSUE 9.1+9.2: timezone-aware now + interval-aware staleness
    end_date = datetime.now(timezone.utc)
    _secs    = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1D": 86400}
    stale_after = _secs.get(interval, 900) * 4

    if os.path.exists(filename):
        try:
            df_cached = pd.read_csv(filename, index_col=0, parse_dates=True)
            if not df_cached.empty:
                last_ts = df_cached.index[-1]
                if getattr(last_ts, "tzinfo", None) is None:
                    last_ts = last_ts.tz_localize("UTC")
                age_secs = (end_date - last_ts.tz_convert("UTC")).total_seconds()
                if age_secs < stale_after:
                    print(f"✅ Using cached data for {symbol} ({len(df_cached)} bars)")
                    return df_cached.sort_index()
        except Exception as e:
            print(f"Cache load failed: {e}. Fetching fresh...")

    print(f"Fetching fresh data from API for {symbol}...")

    api_url    = "https://api.india.delta.exchange/v2/history/candles"
    headers    = {"Accept": "application/json"}
    start_date = end_date - timedelta(days=total_days)

    date_ranges = pd.date_range(start=start_date, end=end_date, freq="7D")
    all_dfs = []

    for i in range(len(date_ranges)):
        chunk_start = date_ranges[i]
        chunk_end = date_ranges[i + 1] if i + 1 < len(date_ranges) else end_date

        start_ts = int(chunk_start.timestamp())
        end_ts = int(chunk_end.timestamp())

        params = {
            "resolution": interval,
            "symbol": symbol,
            "start": str(start_ts),
            "end": str(end_ts)
        }

        for attempt in range(3):
            try:
                response = requests.get(api_url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success") and data.get("result"):
                        rows = []
                        for c in data["result"]:
                            rows.append({
                                "time": c["time"],
                                "Open": float(c["open"]),
                                "High": float(c["high"]),
                                "Low": float(c["low"]),
                                "Close": float(c["close"]),
                                "Volume": float(c["volume"] or 0)
                            })
                        df_chunk = pd.DataFrame(rows)
                        df_chunk["DateTime"] = pd.to_datetime(df_chunk["time"], unit="s", utc=True)
                        df_chunk["DateTime"] = df_chunk["DateTime"].dt.tz_convert("Asia/Kolkata")
                        df_chunk.set_index("DateTime", inplace=True)
                        all_dfs.append(df_chunk.drop(columns=["time"]))
                        break
                time.sleep(1)
            except Exception as e:
                print(f"Retry error on attempt {attempt+1}: {e}")
                time.sleep(1)
        
        print(f"Fetched chunk: {chunk_start.date()}")

    if not all_dfs:
        print("❌ No data fetched.")
        return pd.DataFrame()

    df = pd.concat(all_dfs)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    df.to_csv(filename)
    print(f"✅ Data saved to {filename}")
    return df

def get_data_for_symbols(symbols: list[str], days: int, interval: str) -> dict[str, pd.DataFrame]:
    """Helper to fetch data for multiple symbols."""
    all_candles = {}
    for sym in symbols:
        df = fetch_data(sym, days, interval)
        if not df.empty:
            all_candles[sym] = df
    return all_candles
