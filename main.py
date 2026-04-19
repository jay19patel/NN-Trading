# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

# Local imports
from data_gathering import fetch_data
from features import create_full_feature_set
from models import MultiHeadTradingModel, RiskAwareLoss
from visualisation import plot_trading_analysis

warnings.filterwarnings('ignore')

def main():
    # 0. Hardware Acceleration Setup (Optimized for Mac)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"🚀 Using device: {device}")

    # 1. Fetch Data
    df = fetch_data(symbol="ADAUSD", total_days=50, interval="15m")
    if df.empty:
        print("No data fetched. Exiting.")
        return

    # 2. Add Features
    df_with_features = create_full_feature_set(df, lookahead=10)

    # 2.1 Print Data Range Statistics
    print("\n📊 DATA RANGE STATISTICS (Next 10 Bars):")
    stats = df_with_features[['upside_pct', 'downside_pct']].describe().loc[['min', 'max', 'mean']]
    print(stats)

    # 2.1.2 Print Signal Distribution
    print("\n🎯 SIGNAL DISTRIBUTION:")
    counts = df_with_features['direction_label'].value_counts()
    dist = {
        'BUY (0)': counts.get(0, 0),
        'NEUTRAL (1)': counts.get(1, 0),
        'SELL (2)': counts.get(2, 0)
    }
    for label, count in dist.items():
        print(f"{label}: {count} samples")

    # 2.2 Visualize (Optional - Opens in Browser)
    plot_trading_analysis(df_with_features, symbol="ADAUSD")
    
    print("\n✅ Dashboard successfully opened in your browser.")
    print("Keep this terminal open to interact with the chart.")
    input("\nPress [Enter] to close the script and stop the dashboard server...")

if __name__ == "__main__":
    main()
