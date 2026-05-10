# -*- coding: utf-8 -*-
"""
NN-Trading CLI
==============
Entry point for the neural trading engine.

Commands:
  --train --symbol ETHUSD    Train the model then auto-run backtest
  --test  --symbol ETHUSD    Run backtest only (model must already be trained)
"""
import argparse
import logging
import sys

from config import cfg
from ui_utils import console
from neural_engine.train import train_short_model
from neural_engine.backtest_short import run_backtest
from neural_engine.backtest_oracle import run_oracle_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Main")


def main() -> None:
    """Parse CLI arguments and dispatch to the correct pipeline."""
    parser = argparse.ArgumentParser(description="NN-Trading Model Engine CLI")
    parser.add_argument("--train", action="store_true", help="Train model then auto-backtest")
    parser.add_argument("--test", action="store_true", help="Run backtest only")
    parser.add_argument("--oracle", action="store_true", help="Run backtest with perfect future knowledge (Oracle)")
    parser.add_argument("--symbol", type=str, default="ETHUSD", help="Symbol (e.g. ETHUSD)")
    args = parser.parse_args()

    if (args.train and args.test) or (args.train and args.oracle) or (args.test and args.oracle):
        console.print("[red]Error: Please use only ONE of --train, --test, or --oracle.[/red]")
        return

    if args.symbol:
        cfg.model.SYMBOLS = [args.symbol]

    if args.train:
        console.rule(f"[bold cyan]TRAINING PIPELINE: {args.symbol}")
        logger.info(f"Starting training pipeline for {args.symbol}...")
        train_short_model()
    elif args.test:
        console.rule(f"[bold cyan]BACKTEST PIPELINE: {args.symbol}")
        logger.info(f"Starting backtest for {args.symbol}...")
        run_backtest(symbol=args.symbol)
    elif args.oracle:
        console.rule(f"[bold yellow]ORACLE BACKTEST: {args.symbol}")
        logger.info(f"Starting oracle backtest for {args.symbol}...")
        run_oracle_backtest(symbol=args.symbol)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
