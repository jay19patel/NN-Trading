# -*- coding: utf-8 -*-
"""
Live Inference Script — Runs every hour to fetch data and predict signals.
========================================================================
This script loads the trained model and performs a forward pass on the 
latest market data every 1 hour.
"""
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.live import Live
from datetime import datetime

from config import cfg
from engine.data_handler import fetch_data
from neural_engine.feature_utils import add_technical_indicators, get_feature_columns
from neural_engine.model import MultiHeadTradingModel

console = Console()

def load_model_and_scaler():
    """Load model weights and normalization parameters."""
    model_path = Path("models/trading_model.pth")
    mean_path = Path("models/scaler_mean.npy")
    scale_path = Path("models/scaler_scale.npy")

    if not all([model_path.exists(), mean_path.exists(), scale_path.exists()]):
        raise FileNotFoundError("Model or Scaler files missing. Run training first!")

    feature_cols = get_feature_columns()
    device = cfg.DEVICE
    
    # Load Model
    model = MultiHeadTradingModel(input_dim=len(feature_cols)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Scaler
    mean = np.load(mean_path)
    scale = np.load(scale_path)

    return model, mean, scale, feature_cols, device

def get_latest_prediction(model, mean, scale, feature_cols, device):
    """Fetch latest data and run inference."""
    symbol = cfg.model.SYMBOLS[0]
    interval = cfg.model.INTERVAL
    window_size = cfg.model.WINDOW_SIZE
    
    # Fetch enough data for indicators + window (approx 200 bars)
    total_days = (window_size + 150) // (24 if interval == "1h" else 96) + 2
    df = fetch_data(symbol=symbol, total_days=total_days, interval=interval)
    
    if df.empty or len(df) < window_size + 100:
        return None, "Not enough data"

    # Feature Engineering
    df_features = add_technical_indicators(df.copy())
    
    # Scale features
    feature_data = df_features[feature_cols].values
    feature_data = (feature_data - mean) / (scale + 1e-8)
    
    # Get last window
    last_window = feature_data[-window_size:]
    X = torch.from_numpy(last_window).float().unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(X)
        probs = torch.softmax(out["direction"], dim=1).cpu().numpy()[0]
        sizing = out["sizing"].cpu().numpy()[0]

    # Decode
    prob_long, prob_neutral, prob_short = probs
    margin_long = prob_long - prob_neutral
    margin_short = prob_short - prob_neutral
    
    threshold = cfg.testing.SIGNAL_MARGIN_THRESHOLD
    conf_floor = cfg.testing.AI_CONFIDENCE_THRESHOLD
    
    verdict = "NEUTRAL"
    confidence = prob_neutral
    color = "white"

    if margin_long >= threshold and prob_long >= conf_floor:
        verdict = "LONG"
        confidence = prob_long
        color = "green"
    elif margin_short >= threshold and prob_short >= conf_floor:
        verdict = "SHORT"
        confidence = prob_short
        color = "red"

    return {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "price": df["Close"].iloc[-1],
        "verdict": verdict,
        "confidence": f"{confidence:.2%}",
        "prob_long": f"{prob_long:.2%}",
        "prob_short": f"{prob_short:.2%}",
        "tp_atr": f"{sizing[1] * cfg.testing.MAX_ATR_TARGET_PCT:.2f}%",
        "sl_atr": f"{sizing[2] * cfg.testing.MAX_ATR_STOP_PCT:.2f}%",
        "color": color
    }, None

def main():
    console.rule("[bold cyan]LIVE AI TRADING INFERENCE")
    
    try:
        model, mean, scale, feature_cols, device = load_model_and_scaler()
        console.print(f"[green]✔ Model loaded successfully on {device}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        return

    table = Table(title="Inference History", show_header=True, header_style="bold magenta")
    table.add_column("Timestamp", width=20)
    table.add_column("Symbol", width=10)
    table.add_column("Price", justify="right")
    table.add_column("Verdict", justify="center")
    table.add_column("Conf", justify="right")
    table.add_column("TP/SL", justify="right")

    with Live(table, refresh_per_second=1):
        while True:
            res, err = get_latest_prediction(model, mean, scale, feature_cols, device)
            
            if err:
                console.print(f"[yellow]Prediction Warning: {err}[/yellow]")
            elif res:
                table.add_row(
                    res["time"],
                    res["symbol"],
                    f"${res['price']:,.2f}",
                    f"[{res['color']}]{res['verdict']}[/{res['color']}]",
                    res["confidence"],
                    f"{res['tp_atr']} / {res['sl_atr']}"
                )
            
            # Sleep for 1 hour
            time.sleep(3600)

if __name__ == "__main__":
    main()
