# -*- coding: utf-8 -*-
"""
Standalone live inference.

Copy only this file + models/trading_model_scripted.pt + models/scaler_mean.npy +
models/scaler_scale.npy to another place. No project-local imports are used.
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import torch


# Must match the config used during training.
SYMBOL = "ETHUSD"
INTERVAL = "1h"
WINDOW_SIZE = 96
SIGNAL_MARGIN_THRESHOLD = 0.20
AI_CONFIDENCE_THRESHOLD = 0.75
MAX_ATR_TARGET_PCT = 6.00
MAX_ATR_STOP_PCT = 2.00
LOOKAHEAD_BARS = 24

EPS = 1e-9

warnings.filterwarnings("ignore", message="enable_nested_tensor is True.*")


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def bars_per_day(interval: str) -> int:
    return {"15m": 96, "1h": 24, "1d": 1}.get(interval, 24)


def safe_div(a, b):
    return a / (b + EPS)


def series_or_zero(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(0.0, index=index)
    return series.reindex(index)


def add_bbands(df: pd.DataFrame, close: pd.Series, prefix: str = "") -> None:
    bbands = ta.bbands(close, length=20)
    if bbands is not None and len(bbands.columns) >= 3:
        lower = bbands.iloc[:, 0]
        upper = bbands.iloc[:, 2]
        df[f"{prefix}bb_width"] = safe_div(upper - lower, close)
        df[f"{prefix}bb_position"] = safe_div(close - lower, upper - lower)
    else:
        df[f"{prefix}bb_width"] = 0.0
        df[f"{prefix}bb_position"] = 0.0


def add_directional(df: pd.DataFrame, high: pd.Series, low: pd.Series, close: pd.Series, prefix: str = "") -> None:
    macd = ta.macd(close)
    if macd is not None and len(macd.columns) >= 3:
        df[f"{prefix}macd"] = macd.iloc[:, 0]
        df[f"{prefix}macd_hist"] = macd.iloc[:, 1]
        df[f"{prefix}macd_signal"] = macd.iloc[:, 2]
    else:
        df[f"{prefix}macd"] = 0.0
        df[f"{prefix}macd_hist"] = 0.0
        df[f"{prefix}macd_signal"] = 0.0

    adx = ta.adx(high, low, close, length=14)
    if adx is not None and len(adx.columns) >= 3:
        df[f"{prefix}adx"] = adx.iloc[:, 0]
        df[f"{prefix}dmp"] = adx.iloc[:, 1]
        df[f"{prefix}dmn"] = adx.iloc[:, 2]
    else:
        df[f"{prefix}adx"] = 0.0
        df[f"{prefix}dmp"] = 0.0
        df[f"{prefix}dmn"] = 0.0


def add_ema_stack(df: pd.DataFrame, close: pd.Series, prefix: str = "", lengths=(9, 21, 50, 200)) -> None:
    emas = {}
    for length in lengths:
        ema = ta.ema(close, length=length)
        emas[length] = series_or_zero(ema, close.index)
        df[f"{prefix}dist_ema_{length}"] = safe_div(close - emas[length], emas[length])

    if 9 in emas and 21 in emas:
        df[f"{prefix}ema_9_21_spread"] = safe_div(emas[9] - emas[21], emas[21])
    if 21 in emas and 50 in emas:
        df[f"{prefix}ema_21_50_spread"] = safe_div(emas[21] - emas[50], emas[50])
    if 50 in emas and 200 in emas:
        df[f"{prefix}ema_50_200_spread"] = safe_div(emas[50] - emas[200], emas[200])
    if 21 in emas:
        df[f"{prefix}trend_regime"] = np.sign(close - emas[21])


def add_rsi_suite(df: pd.DataFrame, close: pd.Series, prefix: str = "") -> None:
    df[f"{prefix}rsi_7"] = ta.rsi(close, length=7)
    df[f"{prefix}rsi_14"] = ta.rsi(close, length=14)
    df[f"{prefix}rsi_21"] = ta.rsi(close, length=21)
    df[f"{prefix}rsi_slope_7_21"] = safe_div(df[f"{prefix}rsi_7"] - df[f"{prefix}rsi_21"], 100.0)
    stoch_rsi = ta.stochrsi(close, length=14)
    if stoch_rsi is not None and len(stoch_rsi.columns) >= 2:
        df[f"{prefix}stochrsi_k"] = stoch_rsi.iloc[:, 0]
        df[f"{prefix}stochrsi_d"] = stoch_rsi.iloc[:, 1]
    else:
        df[f"{prefix}stochrsi_k"] = 0.0
        df[f"{prefix}stochrsi_d"] = 0.0


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return (
        df[["Open", "High", "Low", "Close", "Volume"]]
        .resample(rule, label="right", closed="right")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna()
    )


def higher_timeframe_features(df: pd.DataFrame, rule: str, prefix: str, ema_lengths) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return pd.DataFrame(index=df.index)
    htf = resample_ohlcv(df, rule)
    if htf.empty:
        return pd.DataFrame(index=df.index)

    features = pd.DataFrame(index=htf.index)
    close, high, low, volume = htf["Close"], htf["High"], htf["Low"], htf["Volume"]
    features[f"{prefix}log_return_1"] = np.log(close / close.shift(1))
    features[f"{prefix}log_return_3"] = np.log(close / close.shift(3))
    features[f"{prefix}natr"] = safe_div(ta.atr(high, low, close, length=14), close) * 100.0
    add_ema_stack(features, close, prefix=prefix, lengths=ema_lengths)
    add_rsi_suite(features, close, prefix=prefix)
    adx = ta.adx(high, low, close, length=14)
    features[f"{prefix}adx"] = adx.iloc[:, 0] if adx is not None and len(adx.columns) else 0.0
    add_bbands(features, close, prefix=prefix)
    features[f"{prefix}volume_surprise"] = safe_div(volume - volume.rolling(20).mean(), volume.rolling(20).std())
    return features.shift(1).reindex(df.index, method="ffill")


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()
    close, high, low, open_, volume = df["Close"], df["High"], df["Low"], df["Open"], df["Volume"]
    simple_return_1 = close.pct_change()

    for period in (1, 3, 6, 12, 24):
        df[f"log_return_{period}"] = np.log(close / close.shift(period))

    df["atr"] = ta.atr(high, low, close, length=14)
    df["natr"] = safe_div(df["atr"], close) * 100.0
    df["realized_vol_12"] = df["log_return_1"].rolling(12).std()
    df["realized_vol_24"] = df["log_return_1"].rolling(24).std()
    df["volatility_regime_72"] = safe_div(df["realized_vol_24"], df["realized_vol_24"].rolling(72).mean())
    df["atr_ratio"] = safe_div(df["atr"], df["atr"].rolling(100).mean())

    add_rsi_suite(df, close)
    add_ema_stack(df, close)
    add_directional(df, high, low, close)
    add_bbands(df, close)

    supertrend = ta.supertrend(high, low, close, length=10, multiplier=3)
    df["supertrend"] = supertrend["SUPERT_10_3"] if supertrend is not None and "SUPERT_10_3" in supertrend.columns else close.rolling(10).mean()
    df["trend_strength"] = safe_div((close - df["supertrend"]).abs(), close)
    df["momentum_slope"] = df["log_return_1"].rolling(5).mean() - df["log_return_1"].rolling(20).mean()

    candle_range = (high - low).replace(0, np.nan)
    df["body_pct_range"] = safe_div((close - open_).abs(), candle_range)
    df["upper_wick_pct_range"] = safe_div(high - np.maximum(open_, close), candle_range)
    df["lower_wick_pct_range"] = safe_div(np.minimum(open_, close) - low, candle_range)
    df["buy_pressure"] = safe_div(close - low, candle_range)
    df["wick_imbalance"] = df["upper_wick_pct_range"] - df["lower_wick_pct_range"]
    df["candle_vs_atr"] = safe_div(high - low, df["atr"])

    df["dist_high_24"] = safe_div(close - high.rolling(24).max(), high.rolling(24).max())
    df["dist_low_24"] = safe_div(close - low.rolling(24).min(), low.rolling(24).min())
    df["dist_high_72"] = safe_div(close - high.rolling(72).max(), high.rolling(72).max())
    df["dist_low_72"] = safe_div(close - low.rolling(72).min(), low.rolling(72).min())
    df["intraday_position"] = safe_div(close - low.rolling(24).min(), high.rolling(24).max() - low.rolling(24).min())

    vwap = ta.vwap(high, low, close, volume)
    df["vwap"] = vwap if vwap is not None else close.rolling(20).mean()
    df["price_to_vwap"] = safe_div(close - df["vwap"], df["vwap"])
    df["vol_ratio_20"] = safe_div(volume, volume.rolling(20).mean())
    df["volume_surprise_50"] = safe_div(volume - volume.rolling(50).mean(), volume.rolling(50).std())
    df["vol_trend_5_20"] = safe_div(volume.rolling(5).mean() - volume.rolling(20).mean(), volume.rolling(20).mean())
    obv = ta.obv(close, volume)
    df["obv_slope_20"] = safe_div(obv - obv.shift(20), close * volume.rolling(20).mean())

    df["efficiency_ratio_10"] = safe_div((close - close.shift(10)).abs(), candle_range.rolling(10).sum())
    df["range_compression_10_50"] = safe_div(candle_range.rolling(10).mean(), candle_range.rolling(50).mean())
    df["surprise_20"] = safe_div(simple_return_1 - simple_return_1.rolling(20).mean(), simple_return_1.rolling(20).std())
    df["shock_elasticity_12"] = safe_div(simple_return_1.abs(), df["realized_vol_12"])

    idx = pd.to_datetime(df.index, errors="coerce")
    if not idx.isna().all():
        hour = idx.hour + idx.minute / 60.0
        dow = idx.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    else:
        df["hour_sin"] = df["hour_cos"] = df["dow_sin"] = df["dow_cos"] = 0.0

    df = pd.concat(
        [
            df,
            higher_timeframe_features(df, "4h", "mtf_4h_", (9, 21, 50)),
            higher_timeframe_features(df, "1D", "mtf_1d_", (9, 21)),
        ],
        axis=1,
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    indicator_cols = [
        col for col in df.columns
        if any(token in col for token in ("rsi", "macd", "adx", "dmp", "dmn", "ema", "atr", "bb_", "supertrend", "vwap"))
    ]
    df[indicator_cols] = df[indicator_cols].ffill()
    return df.fillna(0)


def get_feature_columns() -> list[str]:
    return [
        "log_return_1", "log_return_3", "log_return_6", "log_return_12", "log_return_24",
        "natr", "realized_vol_12", "realized_vol_24", "volatility_regime_72", "atr_ratio",
        "rsi_7", "rsi_14", "rsi_21", "rsi_slope_7_21", "stochrsi_k",
        "dist_ema_9", "dist_ema_21", "dist_ema_50", "dist_ema_200",
        "ema_9_21_spread", "ema_21_50_spread", "ema_50_200_spread", "trend_regime",
        "macd", "macd_hist", "adx",
        "bb_width", "bb_position", "trend_strength", "momentum_slope",
        "body_pct_range", "buy_pressure", "wick_imbalance", "candle_vs_atr",
        "dist_high_24", "dist_low_24", "dist_high_72", "dist_low_72", "intraday_position",
        "price_to_vwap", "vol_ratio_20", "volume_surprise_50", "vol_trend_5_20", "obv_slope_20",
        "efficiency_ratio_10", "range_compression_10_50", "surprise_20",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "mtf_4h_log_return_1", "mtf_4h_natr",
        "mtf_4h_dist_ema_21", "mtf_4h_dist_ema_50",
        "mtf_4h_ema_9_21_spread", "mtf_4h_ema_21_50_spread", "mtf_4h_trend_regime",
        "mtf_4h_rsi_14", "mtf_4h_rsi_slope_7_21", "mtf_4h_adx",
        "mtf_4h_bb_position", "mtf_4h_volume_surprise",
        "mtf_1d_log_return_1", "mtf_1d_natr",
        "mtf_1d_dist_ema_9", "mtf_1d_dist_ema_21",
        "mtf_1d_ema_9_21_spread", "mtf_1d_trend_regime",
        "mtf_1d_rsi_14", "mtf_1d_rsi_slope_7_21",
        "mtf_1d_bb_position", "mtf_1d_volume_surprise",
    ]


def fetch_data(symbol: str, total_days: int, interval: str) -> pd.DataFrame:
    os.makedirs("data", exist_ok=True)
    filename = f"data/data_{symbol}_{interval}.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
        if not df.empty:
            now = datetime.now(df.index[-1].tzinfo)
            if df.index[0] <= now - timedelta(days=total_days) and (now - df.index[-1]).total_seconds() < 3600:
                return df.sort_index()

    api_url = "https://api.india.delta.exchange/v2/history/candles"
    end_date = datetime.now(timezone.utc).replace(tzinfo=None)
    start_date = end_date - timedelta(days=total_days)
    date_ranges = pd.date_range(start=start_date, end=end_date, freq="7D")
    frames = []

    for i, chunk_start in enumerate(date_ranges):
        chunk_end = date_ranges[i + 1] if i + 1 < len(date_ranges) else end_date
        params = {
            "resolution": interval,
            "symbol": symbol,
            "start": str(int(chunk_start.timestamp())),
            "end": str(int(chunk_end.timestamp())),
        }
        for _ in range(3):
            try:
                response = requests.get(api_url, params=params, headers={"Accept": "application/json"}, timeout=10)
                data = response.json() if response.status_code == 200 else {}
                if data.get("success") and data.get("result"):
                    rows = [
                        {
                            "time": c["time"],
                            "Open": float(c["open"]),
                            "High": float(c["high"]),
                            "Low": float(c["low"]),
                            "Close": float(c["close"]),
                            "Volume": float(c["volume"] or 0),
                        }
                        for c in data["result"]
                    ]
                    chunk = pd.DataFrame(rows)
                    chunk["DateTime"] = pd.to_datetime(chunk["time"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
                    frames.append(chunk.set_index("DateTime"))
                    break
            except Exception:
                time.sleep(1)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.to_csv(filename)
    return df


def load_model(model_dir: Path):
    feature_cols = get_feature_columns()
    model_path = model_dir / "trading_model_scripted.pt"
    mean_path = model_dir / "scaler_mean.npy"
    scale_path = model_dir / "scaler_scale.npy"
    if not model_path.exists() or not mean_path.exists() or not scale_path.exists():
        raise FileNotFoundError("models/trading_model_scripted.pt, scaler_mean.npy, scaler_scale.npy required")

    dev = device()
    model = torch.jit.load(str(model_path), map_location=dev)
    model.eval()

    mean = np.load(mean_path).astype(np.float32)
    scale = np.load(scale_path).astype(np.float32)
    if len(mean) != len(feature_cols) or len(scale) != len(feature_cols):
        raise ValueError("scaler feature count mismatch; constants/features must match training")
    return model, mean, scale, feature_cols, dev


@torch.no_grad()
def predict(symbol: str, interval: str, model_dir: Path) -> dict:
    model, mean, scale, feature_cols, dev = load_model(model_dir)
    bpd = bars_per_day(interval)
    total_days = max(60, (WINDOW_SIZE + 220) // bpd + 5)
    df = fetch_data(symbol, total_days, interval)
    if df.empty or len(df) < WINDOW_SIZE:
        raise RuntimeError("not enough data")

    features = add_technical_indicators(df)
    x = features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values.astype(np.float32)
    x = (x - mean) / (scale + 1e-8)
    x = torch.from_numpy(x[-WINDOW_SIZE:]).float().unsqueeze(0).to(dev)

    direction, sizing_out, magnitude_out, time_out = model(x)
    probs = torch.softmax(direction, dim=1).cpu().numpy()[0]
    sizing = sizing_out.cpu().numpy()[0]
    magnitude = float(magnitude_out.cpu().numpy()[0][0])
    time_score = float(time_out.cpu().numpy()[0][0])

    long_p, neutral_p, short_p = [float(v) for v in probs]
    verdict = "NEUTRAL"
    confidence = neutral_p
    if long_p - neutral_p >= SIGNAL_MARGIN_THRESHOLD and long_p >= AI_CONFIDENCE_THRESHOLD:
        verdict, confidence = "LONG", long_p
    elif short_p - neutral_p >= SIGNAL_MARGIN_THRESHOLD and short_p >= AI_CONFIDENCE_THRESHOLD:
        verdict, confidence = "SHORT", short_p

    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "price": float(df["Close"].iloc[-1]),
        "signal": verdict,
        "confidence": confidence,
        "long": long_p,
        "neutral": neutral_p,
        "short": short_p,
        "qty_ratio": float(sizing[0]),
        "take_profit_pct": float(sizing[1]) * MAX_ATR_TARGET_PCT,
        "stop_loss_pct": float(sizing[2]) * MAX_ATR_STOP_PCT,
        "expected_move_pct": magnitude * MAX_ATR_TARGET_PCT,
        "bars_to_target": max(1, round(time_score * LOOKAHEAD_BARS)),
    }


def print_result(result: dict) -> None:
    print("\nLIVE INFERENCE")
    print(f"Time      : {result['time']}")
    print(f"Symbol    : {result['symbol']}")
    print(f"Price     : ${result['price']:,.2f}")
    print(f"Signal    : {result['signal']} ({result['confidence']:.2%})")
    print(f"Prob L/N/S: {result['long']:.2%} / {result['neutral']:.2%} / {result['short']:.2%}")
    print(f"Qty Ratio : {result['qty_ratio']:.2f}")
    print(f"TP / SL   : {result['take_profit_pct']:.2f}% / {result['stop_loss_pct']:.2f}%")
    print(f"Move / ETA: {result['expected_move_pct']:.2f}% / {result['bars_to_target']} bars")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--interval", default=INTERVAL)
    parser.add_argument("--sleep", type=int, default=3600)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        print_result(predict(args.symbol, args.interval, Path(args.model_dir)))
        if args.once:
            break
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
