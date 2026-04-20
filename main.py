# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import warnings
import os
import time
from datetime import datetime

# Local imports
from data_gathering import fetch_data
from features import create_full_feature_set
from visualisation import plot_trading_analysis, print_signal_insights
from training_utils import prepare_data, train_model, get_ai_predictions

warnings.filterwarnings('ignore')

def evaluate_ai_verdicts(df: pd.DataFrame, target: float = 2.0, stop: float = 1.0, lookahead: int = 10) -> pd.DataFrame:
    """Check if AI signals actually hit TP or SL first."""
    df = df.copy()
    df['ai_outcome'] = 'NONE'
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    verdicts = df['ai_verdict'].values
    
    outcomes = ['NONE'] * len(df)
    
    for i in range(len(df) - lookahead):
        if verdicts[i] == 1: continue
        
        entry = close[i]
        tp = entry * (1 + target/100) if verdicts[i] == 0 else entry * (1 - target/100)
        sl = entry * (1 - stop/100) if verdicts[i] == 0 else entry * (1 + stop/100)
        
        # Check future path
        hit = 'TIMEOUT'
        for j in range(1, lookahead + 1):
            curr_high, curr_low = high[i + j], low[i + j]
            
            # For Buy Signal
            if verdicts[i] == 0:
                if curr_low <= sl: hit = 'FAILED'; break
                if curr_high >= tp: hit = 'SUCCESS'; break
            # For Sell Signal
            elif verdicts[i] == 2:
                if curr_high >= sl: hit = 'FAILED'; break
                if curr_low <= tp: hit = 'SUCCESS'; break
        
        outcomes[i] = hit
        
    df['ai_outcome'] = outcomes
    return df

def main():
    # 0. Hardware Acceleration Setup
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"🚀 Using device: {device}")

    # 1. Fetch & Prepare Data (365 days for better context)
    symbol = "ADAUSD"
    interval = "15m"
    df = fetch_data(symbol=symbol, total_days=365, interval=interval)
    if df.empty:
        print("No data fetched. Exiting.")
        return

    # 2. Feature Engineering with Dual-Layer Caching (FIXED: Improved Caching)
    cache_file = f"features_cache_{symbol}_{interval}.csv"
    lookahead = 20
    
    if os.path.exists(cache_file):
        cache_age_mins = (time.time() - os.path.getmtime(cache_file)) / 60
        if cache_age_mins < 120: # 2 hour cache for ADA
            print(f"⚡ Loading PRE-CALCULATED features from cache: {cache_file}")
            df_with_features = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            print("⏳ Feature cache expired. Re-calculating all indicators...")
            df_with_features = create_full_feature_set(df, lookahead=lookahead)
            df_with_features.to_csv(cache_file)
    else:
        print("🛠️ Calculating 135+ indicators (First run)...")
        df_with_features = create_full_feature_set(df, lookahead=lookahead)
        df_with_features.to_csv(cache_file)

    # 3. Analytics & Automated Feature Selection
    from analytics import FeatureAnalyzer
    from training_utils import create_sequences # Import the new sequence helper
    
    analyzer = FeatureAnalyzer(df_with_features)
    
    dist = df_with_features['direction_label'].value_counts().to_dict()
    print("\n" + "📊 TARGET LABEL DISTRIBUTION".center(40, "-"))
    print(f"BUY (0): {dist.get(0, 0)} signals")
    print(f"NEUTRAL (1): {dist.get(1, 0)} signals") # Handled both float/int keys
    print(f"SELL (2): {dist.get(2, 0)} signals")
    print("-" * 40)

    analyzer.print_health_report()
    
    weak_features = analyzer.prune_weak_features(threshold=0.01)
    if weak_features:
        print(f"✂️ Pruning {len(weak_features)} noise indicators for better focus...")
        df_with_features = df_with_features.drop(columns=weak_features)

    # 4. Prepare for AI Pattern Learning (Sequence Based)
    X_train_raw, y_train_raw, X_test_raw, y_test_raw, features, scaler = prepare_data(df_with_features, test_days=10)
    
    # 🔄 TRANSFORM TO SEQUENCES (Bug #1 Fix)
    seq_len = 32
    print(f"🔄 Creating temporal sequences (Window size: {seq_len} bars)...")
    X_train, y_train = create_sequences(X_train_raw, y_train_raw, seq_len=seq_len)
    X_test, y_test = create_sequences(X_test_raw, y_test_raw, seq_len=seq_len)
    
    print("\n" + "="*50)
    print("🤖 AI SYSTEM SUMMARY")
    print("="*50)
    print(f"Sequence Length: {seq_len} bars (~8 hours)")
    print(f"Training Period: {len(X_train)} sequences")
    print(f"Testing Period: {len(X_test)} sequences (Last 10 Days)")
    print(f"Feature Vector Size: {len(features)}")
    print("="*50)

    # 5. Model Training (Now Sequence-Aware)
    input_dim = len(features)
    model = train_model(
        X_train, y_train, device, input_dim, 
        X_val=X_test, y_val=y_test, 
        epochs=100
    )
    
    # 6. AI Inference (Verdict Check)
    print("\n🔍 Running AI Verdict Check on temporal sequences...")
    ai_test_results = get_ai_predictions(model, X_test, device)
    
    # 7. Merge Results (Aligned with sequence window)
    # The last len(ai_test_results) bars from df_with_features correspond to the sequence targets
    test_portion = df_with_features.iloc[-len(ai_test_results):].copy()
    test_portion = test_portion.reset_index()
    for col in ai_test_results.columns:
        test_portion[col] = ai_test_results.iloc[:, ai_test_results.columns.get_loc(col)].values
    test_portion = test_portion.set_index('time')
    
    # 8. Evaluate Trade Outcomes (User Strategy: 1% TP, 1% SL, 20 bars Timeout)
    test_portion = evaluate_ai_verdicts(test_portion, target=1.0, stop=1.0, lookahead=20)
    
    # 9. Detailed Performance Report (Rich Console Output)
    print("\n" + "📈 AI TRADE PERFORMANCE REPORT (Last 10 Days)".center(60, "="))
    
    # Filter subsets
    buys = test_portion[test_portion['ai_verdict'] == 0]
    sells = test_portion[test_portion['ai_verdict'] == 2]
    
    # Buy Stats
    buy_wins = (buys['ai_outcome'] == 'SUCCESS').sum()
    buy_losses = (buys['ai_outcome'] == 'FAILED').sum()
    buy_timeouts = (buys['ai_outcome'] == 'TIMEOUT').sum()
    
    # Sell Stats
    sell_wins = (sells['ai_outcome'] == 'SUCCESS').sum()
    sell_losses = (sells['ai_outcome'] == 'FAILED').sum()
    sell_timeouts = (sells['ai_outcome'] == 'TIMEOUT').sum()
    
    # Aggregated Stats
    total_signals = len(buys) + len(sells)
    total_wins = buy_wins + sell_wins
    total_losses = buy_losses + sell_losses
    total_timeouts = buy_timeouts + sell_timeouts
    
    print(f"Total AI Signals Generated: {total_signals}")
    print(f"Target: 1.0% | Stoploss: 1.0% | Timeout: 20 bars\n")
    
    print(f"🟢 BUY SIGNALS: {len(buys)}")
    print(f"   - Wins: {buy_wins} | Losses: {buy_losses} | Timeouts: {buy_timeouts}")
    
    print(f"🔴 SELL SIGNALS: {len(sells)}")
    print(f"   - Wins: {sell_wins} | Losses: {sell_losses} | Timeouts: {sell_timeouts}")
    
    print("-" * 60)
    print(f"🏆 TOTAL WINS   : {total_wins}")
    print(f"💀 TOTAL LOSSES : {total_losses}")
    print(f"⏳ TOTAL TIMEOUT: {total_timeouts}")
    
    if total_signals > 0:
        win_rate = (total_wins / total_signals) * 100
        print(f"\n🔥 AI OVERALL WIN RATE: {win_rate:.1f}%")
        
        if len(buys) > 0:
            buy_win_rate = (buy_wins / len(buys)) * 100
            print(f"📈 Buy Win Rate: {buy_win_rate:.1f}%")
        if len(sells) > 0:
            sell_win_rate = (sell_wins / len(sells)) * 100
            print(f"📉 Sell Win Rate: {sell_win_rate:.1f}%")

    print("=" * 60)

if __name__ == "__main__":
    main()
