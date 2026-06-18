# -*- coding: utf-8 -*-
"""
Hudu — OHLC data fetcher + Oracle trade labeler.

Run:
    uv run python app.py
"""
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from get_data import fetch_data
from set_label import OracleLabeler
from config import cfg
from result_export import ResultBuilder

# Starting capital shown in the labeling snapshot (the real backtest lives in
# model_backtest.py and reads cfg.ml_backtest.INITIAL_CAPITAL).
INITIAL_CAPITAL = cfg.ml_backtest.INITIAL_CAPITAL

console = Console()

# ── Symbol & fetch parameters ─────────────────────────────────────────────────
SYMBOL = "BTCUSD"
DAYS = 365*2
INTERVAL = "15m"


# ── ATR + SMA helpers (required by OracleLabeler) ────────────────────────────

def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add atr and sma_20 columns used by OracleLabeler."""
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1.0 / cfg.training.ATR_LENGTH, adjust=False,
                       min_periods=cfg.training.ATR_LENGTH).mean()
    df["sma_20"] = df["Close"].rolling(20).mean().fillna(df["Close"])
    return df


# ── Statistics ──────────────────────────────────────────────────────────────

def symbol_label_stats(symbol: str, df: pd.DataFrame) -> dict:
    """Compute Oracle label statistics as a plain dict (single source of truth
    for both the console table and the JSON export)."""
    label_counts = df["direction_label"].value_counts().sort_index()

    total = len(df)
    n_long = int(label_counts.get(0, 0))
    n_neutral = int(label_counts.get(1, 0))
    n_short = int(label_counts.get(2, 0))

    avg_ret = df.groupby("direction_label")["expected_return_pct"].mean()
    avg_time = df.groupby("direction_label")["time_to_target"].mean()

    return {
        "symbol": symbol,
        "total_bars": total,
        "total_columns": len(df.columns),
        "date_start": df.index[0].strftime("%Y-%m-%d") if total else None,
        "date_end": df.index[-1].strftime("%Y-%m-%d") if total else None,
        "long": {
            "count": n_long,
            "pct": (n_long / total * 100.0) if total else 0.0,
            "avg_expected_return_pct": float(avg_ret.get(0, 0.0)),
            "avg_bars_to_target": float(avg_time.get(0, 0.0)),
        },
        "neutral": {
            "count": n_neutral,
            "pct": (n_neutral / total * 100.0) if total else 0.0,
            "avg_expected_return_pct": float(avg_ret.get(1, 0.0)),
            "avg_bars_to_target": float(avg_time.get(1, 0.0)),
        },
        "short": {
            "count": n_short,
            "pct": (n_short / total * 100.0) if total else 0.0,
            "avg_expected_return_pct": float(avg_ret.get(2, 0.0)),
            "avg_bars_to_target": float(avg_time.get(2, 0.0)),
        },
        "long_short_ratio": n_long / max(n_short, 1),
        "non_neutral_pct": ((n_long + n_short) / total * 100.0) if total else 0.0,
    }


def _print_symbol_stats(symbol: str, df: pd.DataFrame) -> dict:
    s = symbol_label_stats(symbol, df)

    total = s["total_bars"]
    n_long = s["long"]["count"]
    n_neutral = s["neutral"]["count"]
    n_short = s["short"]["count"]

    pct = lambda n: f"{n / total * 100:.1f}%" if total else "0%"

    # Per-label avg return / time (rebuilt for the colored table cells)
    avg_ret = df.groupby("direction_label")["expected_return_pct"].mean()
    avg_time = df.groupby("direction_label")["time_to_target"].mean()

    table = Table(
        title=f"[bold cyan]{symbol}[/bold cyan]  —  Oracle Label Statistics",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Metric", style="bold", min_width=30)
    table.add_column("Value", justify="right", min_width=20)

    table.add_row("Total bars", str(total))
    table.add_row("Total columns (after labeling)", str(len(df.columns)))
    table.add_row("Date range",
                  f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")
    table.add_section()
    table.add_row("[green]LONG  (0)[/green]  count",
                  f"[green]{n_long}[/green]  ({pct(n_long)})")
    table.add_row("[yellow]NEUTRAL (1)[/yellow] count",
                  f"[yellow]{n_neutral}[/yellow]  ({pct(n_neutral)})")
    table.add_row("[red]SHORT (2)[/red]  count",
                  f"[red]{n_short}[/red]  ({pct(n_short)})")
    table.add_section()
    table.add_row("[green]LONG[/green]  avg expected return",
                  f"[green]{avg_ret.get(0, 0):.4f}%[/green]")
    table.add_row("[yellow]NEUTRAL[/yellow] avg expected return",
                  f"[yellow]{avg_ret.get(1, 0):.4f}%[/yellow]")
    table.add_row("[red]SHORT[/red]  avg expected return",
                  f"[red]{avg_ret.get(2, 0):.4f}%[/red]")
    table.add_section()
    table.add_row("[green]LONG[/green]  avg bars to target",
                  f"[green]{avg_time.get(0, 0):.1f}[/green]")
    table.add_row("[yellow]NEUTRAL[/yellow] avg bars to target",
                  f"[yellow]{avg_time.get(1, 0):.1f}[/yellow]")
    table.add_row("[red]SHORT[/red]  avg bars to target",
                  f"[red]{avg_time.get(2, 0):.1f}[/red]")
    table.add_section()
    table.add_row("Long/Short ratio",
                  f"{n_long / max(n_short, 1):.2f}")
    table.add_row("Non-neutral %",
                  f"{(n_long + n_short) / total * 100:.1f}%")

    console.print(table)
    return s


def _print_confirmation_summary(df: pd.DataFrame) -> dict:
    reason_counts = df["label_filter_reason"].value_counts().to_dict()
    oracle_trades = int(df["oracle_direction_label"].isin([0, 2]).sum())
    confirmed_trades = int(df["direction_label"].isin([0, 2]).sum())
    rejected_trades = max(oracle_trades - confirmed_trades, 0)

    summary = {
        "oracle_trades": oracle_trades,
        "confirmed_trades": confirmed_trades,
        "rejected_trades": rejected_trades,
        "confirmed_pct": (confirmed_trades / oracle_trades * 100.0) if oracle_trades else 0.0,
        "reasons": {str(k): int(v) for k, v in reason_counts.items()},
        "avg_confirmation_score": float(df.loc[df["direction_label"].isin([0, 2]), "confirmation_score"].mean() or 0.0),
        "avg_confirmation_edge": float(df.loc[df["direction_label"].isin([0, 2]), "confirmation_edge"].mean() or 0.0),
    }

    table = Table(
        title="[bold cyan]Confirmation Filter[/bold cyan]",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Metric", style="bold", min_width=28)
    table.add_column("Value", justify="right", min_width=18)
    table.add_row("Oracle trades", str(oracle_trades))
    table.add_row("Confirmed trades", str(confirmed_trades))
    table.add_row("Rejected trades", str(rejected_trades))
    table.add_row("Confirmed %", f"{summary['confirmed_pct']:.1f}%")
    table.add_row("Avg confirmation score", f"{summary['avg_confirmation_score']:.2f}")
    table.add_row("Avg confirmation edge", f"{summary['avg_confirmation_edge']:.2f}")
    console.print(table)
    return summary


# ── Pipeline stages ───────────────────────────────────────────────────────────

def _run_data_and_labeling() -> None:
    """Stage 1: Fetch OHLC → Oracle-label → Save CSV + stats."""
    console.print(f"\n[bold]Fetching {DAYS}d of {INTERVAL} candles for {SYMBOL}...[/bold]")
    df = fetch_data(SYMBOL, DAYS, INTERVAL)
    if df.empty:
        console.print("[red]No data fetched. Exiting.[/red]")
        raise SystemExit(1)

    labeler = OracleLabeler()
    result  = ResultBuilder(symbol=SYMBOL, interval=INTERVAL,
                            days=DAYS, initial_capital=INITIAL_CAPITAL)

    console.print(f"\n[bold cyan]► {SYMBOL}[/bold cyan]  ({len(df)} bars)")
    df = _add_indicators(df)

    with console.status(f"  Labeling {SYMBOL}..."):
        df = labeler.generate_labels(
            df,
            labeling_mode            = cfg.training.LABELING_MODE,
            use_sma_filter           = cfg.training.USE_SMA_FILTER,
            entry_mode               = cfg.training.ENTRY_MODE,
            fixed_tp_atr_multiplier  = cfg.training.FIXED_TP_ATR_MULTIPLIER,
            fixed_sl_atr_multiplier  = cfg.training.FIXED_SL_ATR_MULTIPLIER,
        )

    result.add("confirmation_filter", _print_confirmation_summary(df))
    result.add("label_stats", _print_symbol_stats(SYMBOL, df))

    out_path = f"data/labeled_{SYMBOL}_{INTERVAL}.csv"
    df.to_csv(out_path)
    console.print(f"  [dim]Saved -> {out_path}[/dim]")

    ma_stats     = labeler.moving_average_relationship_report(df)
    signal_stats = labeler.technical_condition_report(df)
    # Signal conditions saved to JSON only — not printed to console
    result.add("signal_conditions", {"moving_average": ma_stats, "technical": signal_stats})

    result_path = result.save("result.json")
    console.print(f"  [dim]Shareable JSON saved → {result_path}[/dim]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Full pipeline:
      Stage 1 — Fetch data & label
      Stage 2 — Train model
      Stage 3 — Backtest & virtual trading  ← comment out to skip

    Each stage is independent. You can also run them separately:
      uv run python app.py         # all stages
      uv run python trainer.py     # train only
      uv run python backtest.py    # backtest only
    """
    # ─── Stage 1: Data & Labeling ─────────────────────────────────────────
    console.rule("[bold blue]Stage 1 / 3  —  Data & Labeling[/bold blue]")
    _run_data_and_labeling()

    # ─── Stage 2: Model Training ──────────────────────────────────────────
    console.rule("[bold blue]Stage 2 / 3  —  Model Training[/bold blue]")
    import trainer as _trainer
    _trainer.main()

    # ─── Stage 3: Backtest & Virtual Trading ──────────────────────────────
    # To skip backtesting, comment out the next two lines:
    console.rule("[bold blue]Stage 3 / 3  —  Backtest & Virtual Trading[/bold blue]")
    import backtest as _backtest
    _backtest.main()

    console.rule("[bold green]Pipeline Complete[/bold green]")


if __name__ == "__main__":
    main()
