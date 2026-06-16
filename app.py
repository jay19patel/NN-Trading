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
from set_label import OracleLabeler, TECHNICAL_CONDITION_COLUMNS
from config import cfg
from result_export import ResultBuilder

# Starting capital shown in the labeling snapshot (the real backtest lives in
# model_backtest.py and reads cfg.ml_backtest.INITIAL_CAPITAL).
INITIAL_CAPITAL = cfg.ml_backtest.INITIAL_CAPITAL

console = Console()

# ── Symbol & fetch parameters ─────────────────────────────────────────────────
SYMBOL = "BTCUSD"
DAYS = 365
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


def _print_signal_conditions_combined(ma_stats: pd.DataFrame, tech_stats: pd.DataFrame) -> None:
    table = Table(
        title="[bold cyan]Signal Conditions[/bold cyan]",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("#", style="dim", justify="right", no_wrap=True)
    table.add_column("Type", style="dim", no_wrap=True)
    table.add_column("Condition", style="bold", no_wrap=True)
    table.add_column("BUY Trades", justify="right")
    table.add_column("BUY Net", justify="right")
    table.add_column("SELL Trades", justify="right")
    table.add_column("SELL Net", justify="right")

    order = ["sma20", "sma50", "sma200", "ema5", "ema9", "ema20", "ema50", "ema200", "hma9", "hma21", "hma50"]
    rows: list[tuple[int, int, str]] = []
    for ma_index, ma_name in enumerate(order):
        rows.append((ma_index, 0, f"price_above_{ma_name}"))
        rows.append((ma_index, 1, f"price_below_{ma_name}"))

    ma_by_key = {(row["direction"], row["condition"]): row for _, row in ma_stats.iterrows()}
    tech_by_key = {(row["direction"], row["condition"]): row for _, row in tech_stats.iterrows()}

    def metric(by_key: dict, direction: str, condition: str, column: str) -> float:
        row = by_key.get((direction, condition))
        return float(row[column]) if row is not None else 0.0

    def trade_cell(by_key: dict, direction: str, condition: str) -> str:
        trades = int(metric(by_key, direction, condition, "total_trades"))
        pct_value = metric(by_key, direction, condition, "trade_pct_of_direction")
        color = "green" if pct_value >= 50.0 else "red"
        return f"[{color}]{trades} ({pct_value:.1f}%)[/{color}]"

    def keep_condition(by_key: dict, condition: str) -> bool:
        """Keep a row only if at least one side trades >= 50% of its direction."""
        buy_pct = metric(by_key, "BUY", condition, "trade_pct_of_direction")
        sell_pct = metric(by_key, "SELL", condition, "trade_pct_of_direction")
        return buy_pct >= 50.0 or sell_pct >= 50.0

    serial = 0
    for _, _, condition in rows:
        if not keep_condition(ma_by_key, condition):
            continue
        serial += 1
        table.add_row(
            str(serial),
            "MA",
            condition,
            trade_cell(ma_by_key, "BUY", condition),
            f"{metric(ma_by_key, 'BUY', condition, 'net_return_pct'):.4f}",
            trade_cell(ma_by_key, "SELL", condition),
            f"{metric(ma_by_key, 'SELL', condition, 'net_return_pct'):.4f}",
        )

    # Show every technical condition the labeler defines (stays in sync
    # automatically when new conditions are added in set_label.py).
    conditions = list(TECHNICAL_CONDITION_COLUMNS)

    for condition in conditions:
        if not keep_condition(tech_by_key, condition):
            continue
        serial += 1
        table.add_row(
            str(serial),
            "TECH",
            condition.replace("condition_", ""),
            trade_cell(tech_by_key, "BUY", condition),
            f"{metric(tech_by_key, 'BUY', condition, 'net_return_pct'):.4f}",
            trade_cell(tech_by_key, "SELL", condition),
            f"{metric(tech_by_key, 'SELL', condition, 'net_return_pct'):.4f}",
        )

    console.print(table)


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    console.rule("[bold blue]Hudu — Oracle Labeler Pipeline[/bold blue]")

    # 1. Fetch OHLC data
    console.print(f"\n[bold]Fetching {DAYS}d of {INTERVAL} candles for {SYMBOL}[/bold]")
    df = fetch_data(SYMBOL, DAYS, INTERVAL)

    if df.empty:
        console.print("[red]No data fetched. Exiting.[/red]")
        return

    labeler = OracleLabeler()

    # Collects every console block into one shareable JSON file.
    result = ResultBuilder(
        symbol=SYMBOL,
        interval=INTERVAL,
        days=DAYS,
        initial_capital=INITIAL_CAPITAL,
    )

    console.print(f"\n[bold cyan]► {SYMBOL}[/bold cyan]  ({len(df)} bars)")

    # 2. Add indicator columns required by the labeler
    df = _add_indicators(df)

    # 3. Apply Oracle labels
    with console.status(f"  Labeling {SYMBOL}..."):
        df = labeler.generate_labels(
            df,
            labeling_mode=cfg.training.LABELING_MODE,
            use_sma_filter=cfg.training.USE_SMA_FILTER,
            entry_mode=cfg.training.ENTRY_MODE,
            fixed_tp_atr_multiplier=cfg.training.FIXED_TP_ATR_MULTIPLIER,
            fixed_sl_atr_multiplier=cfg.training.FIXED_SL_ATR_MULTIPLIER,
        )

    confirmation_summary = _print_confirmation_summary(df)
    result.add("confirmation_filter", confirmation_summary)

    # 4. Print statistics
    label_stats = _print_symbol_stats(SYMBOL, df)
    result.add("label_stats", label_stats)

    # 5. Save labeled CSV
    out_path = f"data/labeled_{SYMBOL}_{INTERVAL}.csv"
    df.to_csv(out_path)
    console.print(f"  [dim]Saved -> {out_path}[/dim]")

    ma_stats = labeler.moving_average_relationship_report(df)
    signal_stats = labeler.technical_condition_report(df)
    _print_signal_conditions_combined(ma_stats, signal_stats)
    result.add("signal_conditions", {
        "moving_average": ma_stats,
        "technical": signal_stats,
    })

    # 6. Write the single shareable JSON snapshot of everything above.
    #    ML training + backtesting are separate stages — run them next:
    #      uv run python train_model.py      (train + save the model)
    #      uv run python model_backtest.py   (real-world backtest)
    result_path = result.save("result.json")
    console.print(f"  [dim]Shareable JSON saved → {result_path}[/dim]")

    console.rule("[bold green]Done[/bold green]")


if __name__ == "__main__":
    main()
