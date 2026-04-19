# -*- coding: utf-8 -*-
import pandas as pd
import os
from datetime import datetime, timedelta
import requests
import time

def fetch_data(symbol: str = "ADAUSD", total_days: int = 100, interval: str = "15m") -> pd.DataFrame:
    """
    Fetch OHLC data from Delta Exchange API with CSV caching.
    If a local CSV exists for the symbol/interval, load it.
    Otherwise, fetch from API, save to CSV, and return.
    """

    # Define cache filename
    filename = f"data_{symbol}_{interval}.csv"

    # Check if CSV exists
    if os.path.exists(filename):
        print(f"Loading data from local CSV: {filename}")
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        # Ensure DateTime index is tz-aware if not already
        return df

    print(f"Local CSV not found. Fetching data from API for {symbol}...")

    api_url = "https://api.india.delta.exchange/v2/history/candles"
    headers = {'Accept': 'application/json'}
    end_date = datetime.utcnow()
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

        print(f"Fetching: {chunk_start.date()} → {chunk_end.date()}")

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
                        # Convert time to datetime objects
                        df_chunk["DateTime"] = pd.to_datetime(df_chunk["time"], unit="s", utc=True)
                        # Convert to target timezone
                        df_chunk["DateTime"] = df_chunk["DateTime"].dt.tz_convert("Asia/Kolkata")
                        df_chunk.set_index("DateTime", inplace=True)

                        all_dfs.append(df_chunk)
                        break
                time.sleep(1)
            except Exception as e:
                print(f"Retry error on attempt {attempt+1}: {e}")
                time.sleep(1)

    if not all_dfs:
        print("No data fetched.")
        return pd.DataFrame()

    df = pd.concat(all_dfs)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # Save to CSV for future use
    df.to_csv(filename)
    print(f"Data saved to {filename}")

    return df
