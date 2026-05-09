# -*- coding: utf-8 -*-
import os
import sys
import argparse
import logging
import pandas as pd
from config import config
from ui_utils import console
from neural_engine.train import train_short_model
from neural_engine.backtest_short import run_backtest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Main")

def main():
    parser = argparse.ArgumentParser(description="NN-Trading Model Engine CLI")
    parser.add_argument("--train", action="store_true", help="Train the short model")
    parser.add_argument("--test", action="store_true", help="Run backtest for the short model")
    parser.add_argument("--symbol", type=str, default="ETHUSD", help="Symbol to train/test on")
    parser.add_argument("--days", type=int, default=300, help="Number of days of data to use")

    args = parser.parse_args()

    # Update config symbol if provided
    if args.symbol:
        config.data.SYMBOLS = [args.symbol]

    if args.train:
        console.rule(f"[bold cyan]TRAINING PIPELINE: {args.symbol}")
        logger.info(f"Starting training pipeline for {args.symbol} with {args.days} days of data...")
        # Note: train_short_model doesn't take arguments in its current definition, 
        # it uses config values. I'll update train_short_model to accept symbol/days.
        train_short_model() 
    elif args.test:
        console.rule(f"[bold cyan]BACKTEST PIPELINE: {args.symbol}")
        logger.info(f"Starting backtest for {args.symbol}...")
        # run_backtest also uses config
        run_backtest(symbol=args.symbol)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
