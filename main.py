# -*- coding: utf-8 -*-
import time
import os
import warnings

import pandas as pd

from config import config
from data_gathering import fetch_data
from features import create_full_feature_set
from training_utils import (
    prepare_multi_symbol_data,
    train_model,
    run_inference_with_confidence_filter,
)
from evaluation_metrics import sequence_count_for_split
from ui_utils import console, print_banner, Table

warnings.filterwarnings("ignore")


def _bars_per_15m_day() -> int:
    return (24 * 60) // 15


def evaluate_ai_verdicts(
    dataframe: pd.DataFrame, target_pct: float, stop_pct: float, lookahead_bars: int
) -> pd.DataFrame:
    """Label each signal row as SUCCESS / FAILED / TIMEOUT for TP vs SL path (next `lookahead_bars` bars)."""
    dataframe = dataframe.copy()
    dataframe["ai_outcome"] = "NONE"

    close_prices = dataframe["Close"].values
    high_prices = dataframe["High"].values
    low_prices = dataframe["Low"].values
    verdict_values = dataframe["ai_verdict"].values

    row_count = len(dataframe)
    outcomes = ["NONE"] * row_count

    for row_index in range(row_count - lookahead_bars):
        if verdict_values[row_index] == 1:
            continue

        entry_price = close_prices[row_index]
        take_profit = (
            entry_price * (1 + target_pct / 100)
            if verdict_values[row_index] == 0
            else entry_price * (1 - target_pct / 100)
        )
        stop_level = (
            entry_price * (1 - stop_pct / 100)
            if verdict_values[row_index] == 0
            else entry_price * (1 + stop_pct / 100)
        )

        path_outcome = "TIMEOUT"
        for step in range(1, lookahead_bars + 1):
            bar_high = high_prices[row_index + step]
            bar_low = low_prices[row_index + step]

            if verdict_values[row_index] == 0:
                if bar_low <= stop_level:
                    path_outcome = "FAILED"
                    break
                if bar_high >= take_profit:
                    path_outcome = "SUCCESS"
                    break
            elif verdict_values[row_index] == 2:
                if bar_high >= stop_level:
                    path_outcome = "FAILED"
                    break
                if bar_low <= take_profit:
                    path_outcome = "SUCCESS"
                    break

        outcomes[row_index] = path_outcome

    dataframe["ai_outcome"] = outcomes
    return dataframe


def main() -> None:
    device = config.DEVICE
    symbols = config.data.SYMBOLS
    interval = config.data.INTERVAL

    print_banner(
        "GLOBAL AI TRADING SYSTEM",
        f"Symbols: {', '.join(symbols)} | Device: {device}",
    )

    symbol_frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        console.print(f"[info]Fetching data for [bold]{symbol}[/bold]...[/info]")
        market_frame = fetch_data(
            symbol=symbol, total_days=config.data.TOTAL_DAYS, interval=interval
        )
        if market_frame.empty:
            console.print(f"[error]No data for {symbol}. Skipping.[/error]")
            continue

        cache_name = f"features_cache_{symbol}_{interval}.csv"
        lookahead = config.features.LOOKAHEAD_BARS
        if os.path.exists(cache_name):
            cache_age_minutes = (time.time() - os.path.getmtime(cache_name)) / 60
            if cache_age_minutes < config.data.CACHE_VALID_MINS:
                console.print(f"  Using cache for {symbol}")
                features_frame = pd.read_csv(cache_name, index_col=0, parse_dates=True)
            else:
                console.print(f"  Rebuilding features for {symbol}...")
                features_frame = create_full_feature_set(market_frame, lookahead=lookahead)
                features_frame.to_csv(cache_name)
        else:
            features_frame = create_full_feature_set(market_frame, lookahead=lookahead)
            features_frame.to_csv(cache_name)

        symbol_frames[symbol] = features_frame

    if not symbol_frames:
        console.print("[error]No symbol data available. Exiting.[/error]")
        return

    (
        train_X,
        train_y,
        val_X,
        val_y,
        test_X,
        test_y,
        feature_names,
        _,
    ) = prepare_multi_symbol_data(
        symbol_frames,
        test_days=config.training.TEST_DAYS,
        val_days=config.training.VAL_DAYS,
    )

    console.print(
        f"\n[highlight]Train/Val/Test tensors: {len(train_X)} / {len(val_X)} / {len(test_X)} windows[/highlight]"
    )
    console.print(f"[info]{len(feature_names)} input features | seq_len={config.model.SEQ_LEN}[/info]")

    trained_model, test_metrics = train_model(
        train_X,
        train_y,
        device,
        len(feature_names),
        val_features=val_X,
        val_targets=val_y,
        test_features=test_X,
        test_targets=test_y,
        epochs=config.training.EPOCHS,
    )

    if test_metrics:
        console.print("\n[highlight]Held-out test metrics (direction + regression)[/highlight]")
        for metric_key, metric_value in test_metrics.items():
            if metric_key == "classification_report_text":
                console.print(metric_value)
            elif metric_key == "confusion_matrix_3x3":
                console.print(f"Confusion matrix [Buy, Neut, Sell]: {metric_value}")
            else:
                console.print(f"  {metric_key}: {metric_value}")

    console.print("\n[highlight]Per-symbol trading outcomes on last test window[/highlight]")
    predictions_frame = run_inference_with_confidence_filter(trained_model, test_X, device)

    bars_per_day = _bars_per_15m_day()
    test_row_count = config.training.TEST_DAYS * bars_per_day
    sequences_per_symbol = sequence_count_for_split(test_row_count, config.model.SEQ_LEN)

    summary_table = Table(
        title="Global model — summary",
        show_header=True,
        header_style="bold cyan",
    )
    summary_table.add_column("Symbol", justify="left", style="bold")
    summary_table.add_column("Signals", justify="right")
    summary_table.add_column("Win rate (TP first)", justify="right")
    summary_table.add_column("W / L", justify="right")

    sequence_offset = 0
    for symbol in symbols:
        if symbol not in symbol_frames:
            continue

        symbol_predictions = predictions_frame.iloc[
            sequence_offset : sequence_offset + sequences_per_symbol
        ]
        symbol_panel = symbol_frames[symbol].iloc[-sequences_per_symbol:].copy()
        sequence_offset += sequences_per_symbol

        symbol_panel["ai_verdict"] = symbol_predictions["ai_verdict"].values
        symbol_panel["ai_confidence"] = symbol_predictions["ai_confidence"].values

        symbol_panel = evaluate_ai_verdicts(
            symbol_panel,
            target_pct=config.strategy.TARGET_PROFIT_PCT,
            stop_pct=config.strategy.STOP_LOSS_PCT,
            lookahead_bars=config.features.LOOKAHEAD_BARS,
        )

        buy_rows = symbol_panel[symbol_panel["ai_verdict"] == 0]
        sell_rows = symbol_panel[symbol_panel["ai_verdict"] == 2]
        signal_rows = pd.concat([buy_rows, sell_rows])
        resolved = signal_rows[signal_rows["ai_outcome"].isin(["SUCCESS", "FAILED"])]
        wins = int((resolved["ai_outcome"] == "SUCCESS").sum())
        losses = int((resolved["ai_outcome"] == "FAILED").sum())
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0

        summary_table.add_row(
            symbol,
            str(len(signal_rows)),
            f"{win_rate:.1f}%",
            f"{wins} / {losses}",
        )

        detail_table = Table(
            title=f"{symbol} — last {config.training.TEST_DAYS} days (strategy filter)",
            show_header=True,
            header_style="bold magenta",
        )
        detail_table.add_column("Metric", justify="left")
        detail_table.add_column("Buy", justify="right", style="buy")
        detail_table.add_column("Sell", justify="right", style="sell")
        detail_table.add_column("Total", justify="right", style="bold")

        buy_wins = int((buy_rows["ai_outcome"] == "SUCCESS").sum())
        buy_losses = int((buy_rows["ai_outcome"] == "FAILED").sum())
        sell_wins = int((sell_rows["ai_outcome"] == "SUCCESS").sum())
        sell_losses = int((sell_rows["ai_outcome"] == "FAILED").sum())

        detail_table.add_row("Signals", str(len(buy_rows)), str(len(sell_rows)), str(len(signal_rows)))
        detail_table.add_row("Wins", str(buy_wins), str(sell_wins), str(wins))
        detail_table.add_row("Losses", str(buy_losses), str(sell_losses), str(losses))
        console.print(detail_table)
        console.print("")

    console.print(summary_table)
    console.print(
        f"\n[info]TP={config.strategy.TARGET_PROFIT_PCT}% | SL={config.strategy.STOP_LOSS_PCT}% "
        f"| lookahead={config.features.LOOKAHEAD_BARS} bars[/info]\n"
    )


if __name__ == "__main__":
    main()
