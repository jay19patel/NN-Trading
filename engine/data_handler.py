# -*- coding: utf-8 -*-
import pandas as pd
import os
from datetime import datetime, timedelta
import requests
import time
from ui_utils import console, get_progress

def fetch_data(symbol: str = "ADAUSD", total_days: int = 100, interval: str = "15m") -> pd.DataFrame:
    """Fetch OHLC data from Delta Exchange API."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, f"data_{symbol}_{interval}.csv")

    # Caching Logic
    if os.path.exists(filename):
        try:
            df_cached = pd.read_csv(filename, index_col=0, parse_dates=True)
            if not df_cached.empty:
                last_ts = df_cached.index[-1]
                first_ts = df_cached.index[0]
                now = datetime.now(last_ts.tzinfo)
                
                # If we have enough data (start_date is within our cache)
                start_date_needed = now - timedelta(days=total_days)
                if first_ts <= start_date_needed and (now - last_ts).total_seconds() < 3600:
                    console.print(f"[success]✅ Using cached data for {symbol} ({len(df_cached)} bars)[/success]")
                    return df_cached.sort_index()
                
                # Otherwise, we might just need to update it
                console.print(f"[info]Cache for {symbol} is stale or insufficient. Updating...[/info]")
        except Exception as e:
            console.print(f"[warning]Failed to load cache: {e}. Fetching fresh...[/warning]")

    console.print(f"[info]Fetching fresh data from API for [bold]{symbol}[/bold]...[/info]")

    api_url = "https://api.india.delta.exchange/v2/history/candles"
    headers = {'Accept': 'application/json'}
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=total_days)

    date_ranges = pd.date_range(start=start_date, end=end_date, freq="7D")
    all_dfs = []

    with get_progress() as progress:
        fetch_task = progress.add_task(f"Downloading {symbol} Candles...", total=len(date_ranges))
        
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
                            all_dfs.append(df_chunk)
                            break
                    time.sleep(1)
                except Exception as e:
                    console.print(f"[error]Retry error on attempt {attempt+1}:[/error] {e}")
                    time.sleep(1)
            
            progress.update(fetch_task, advance=1, description=f"Fetched: {chunk_start.date()}")

    if not all_dfs:
        console.print("[error]❌ No data fetched.[/error]")
        return pd.DataFrame()

    df = pd.concat(all_dfs)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    df.to_csv(filename)
    console.print(f"[success]✅ Data saved to [bold]{filename}[/bold][/success]")
    return df

def get_data_for_symbols(symbols: list[str], days: int, interval: str) -> dict[str, pd.DataFrame]:
    """Helper to fetch data for multiple symbols."""
    all_candles = {}
    for sym in symbols:
        df = fetch_data(sym, days, interval)
        if not df.empty:
            all_candles[sym] = df
    return all_candles
