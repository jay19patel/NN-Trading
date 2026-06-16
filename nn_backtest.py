# -*- coding: utf-8 -*-
"""
Real-world backtest of the trained causal Transformer.

Same honest, out-of-sample protocol as model_backtest.py — it reuses that
module's trade simulator, stats and printer — but the per-bar direction
probabilities come from the Transformer (nn_model) instead of the sklearn
model. The held-out TEST window is identical (load_valid + split_bounds), so
results are directly comparable to the sklearn baseline.

Usage:
    uv run python train_nn.py      # first, to train + save the transformer
    uv run python nn_backtest.py
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from rich.console import Console

from config import cfg
from model_backtest import print_summary, run_model_backtest, summarize
from nn_model import DirectionTransformer
from set_label import LONG, NEUTRAL, SHORT
from train_model import feature_matrix, load_valid, split_bounds
from train_nn import NN_MODEL_PATH

console = Console()
TRADE_LOG_PATH = "data/nn_trade_log.csv"


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def _proba_for_test(model, Xs_scaled: np.ndarray, test_start: int, window: int,
                    device: torch.device) -> np.ndarray:
    """Build a window for every test bar and return softmax probabilities.

    Each test bar's window may reach back into the val/train region for input
    history — that is fine (features are causal and the model is frozen); only
    the prediction target bar lives in the test window.
    """
    model.eval()
    n = Xs_scaled.shape[0]
    Xs_t = torch.from_numpy(Xs_scaled)
    probs = []
    batch_positions = list(range(test_start, n))
    BATCH = 512
    for i in range(0, len(batch_positions), BATCH):
        chunk = batch_positions[i : i + BATCH]
        seqs = torch.stack([Xs_t[t - window + 1 : t + 1] for t in chunk]).to(device)
        probs.append(F.softmax(model(seqs), dim=1).cpu().numpy())
    return np.concatenate(probs)


def main() -> None:
    console.rule("[bold blue]Hudu — Transformer Real-World Backtest[/bold blue]")

    if not os.path.exists(NN_MODEL_PATH):
        console.print(f"[red]Transformer not found at {NN_MODEL_PATH}. "
                      f"Run `uv run python train_nn.py` first.[/red]")
        return

    bundle = torch.load(NN_MODEL_PATH, map_location="cpu", weights_only=False)
    features = bundle["features"]
    classes = list(bundle["classes"])
    window = int(bundle["window"])
    mean = bundle["scaler_mean"]
    scale = bundle["scaler_scale"]

    device = _device()
    model = DirectionTransformer(input_dim=bundle["input_dim"], num_classes=len(classes)).to(device)
    model.load_state_dict(bundle["state_dict"])

    # Rebuild the SAME data + chronological test window as training/sklearn.
    df = load_valid()
    _, val_end = split_bounds(len(df))
    df_test = df.iloc[val_end:].copy()
    console.print(f"\n[bold]Device:[/bold] {device}  |  test window "
                  f"{df_test.index[0]} → {df_test.index[-1]}  ({len(df_test)} bars)")

    # Scale the full feature matrix with the training scaler, then predict.
    X = feature_matrix(df, features).fillna(0.0).to_numpy(dtype=np.float32)
    Xs = ((X - mean) / scale).astype(np.float32)

    with console.status("Transformer inference over test window..."):
        proba = _proba_for_test(model, Xs, val_end, window, device)

    col = {c: j for j, c in enumerate(classes)}
    zeros = np.zeros(len(df_test))
    proba_long = proba[:, col[LONG]] if LONG in col else zeros
    proba_short = proba[:, col[SHORT]] if SHORT in col else zeros
    proba_neutral = proba[:, col[NEUTRAL]] if NEUTRAL in col else zeros

    threshold = cfg.ml_backtest.CONFIDENCE_THRESHOLD
    with console.status("Running backtest..."):
        trades, equity_curve = run_model_backtest(
            df_test, proba_long, proba_short, proba_neutral, threshold,
            cfg.ml_backtest.INITIAL_CAPITAL,
        )

    stats = summarize(trades, equity_curve, df_test, threshold)
    print_summary(stats)

    if not trades.empty:
        trades.to_csv(TRADE_LOG_PATH, index=False)
        console.print(f"  [dim]Trade log saved → {TRADE_LOG_PATH}[/dim]")
    else:
        console.print("  [yellow]No trades fired — try lowering CONFIDENCE_THRESHOLD.[/yellow]")

    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
