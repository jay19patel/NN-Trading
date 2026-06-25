# -*- coding: utf-8 -*-
"""
Data fetcher + model trainer pipeline.

Run:
    uv run python app.py         # fetch data + train model
    uv run python trainer.py     # train only (data already fetched)
"""
import os

import pandas as pd
from rich.console import Console

from get_data import fetch_data
import trainer as _trainer

console = Console()

SYMBOL   = "BTCUSD"
DAYS     = 365 * 2
INTERVAL = "15m"
CSV_PATH = f"data/labeled_{SYMBOL}_{INTERVAL}.csv"


def main() -> None:
    # ── Stage 1: Fetch OHLCV ──────────────────────────────────────────────────
    console.rule("[bold blue]Stage 1 / 2  —  Data Fetch[/bold blue]")
    console.print(f"\n[bold]Fetching {DAYS}d of {INTERVAL} candles for {SYMBOL}...[/bold]")

    df = fetch_data(SYMBOL, DAYS, INTERVAL)
    if df.empty:
        console.print("[red]No data fetched. Exiting.[/red]")
        raise SystemExit(1)

    os.makedirs("data", exist_ok=True)
    df.to_csv(CSV_PATH)
    console.print(
        f"  [dim]Saved {len(df):,} bars "
        f"({df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}) "
        f"→ {CSV_PATH}[/dim]"
    )

    # ── Stage 2: Train model ──────────────────────────────────────────────────
    console.rule("[bold blue]Stage 2 / 2  —  Model Training[/bold blue]")
    _trainer.main()

    console.rule("[bold green]Pipeline Complete[/bold green]")


if __name__ == "__main__":
    main()
