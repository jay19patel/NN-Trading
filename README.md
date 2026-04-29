# 🚀 Advanced Multi-Symbol AI Trading Engine

This repository contains a professional-grade AI trading system designed for 15-minute interval cryptocurrency data. It utilizes a **Transformer-based Causal Model** and a unique **2-Phase Training Pipeline** to maximize profitability and manage risk.

---

## 🧠 System Architecture

The core of the system is a **Multi-Head Transformer Encoder**. Unlike simple LSTMs, the Transformer can attend to complex, long-range dependencies in price action across a `SEQ_LEN` window (default: 200 bars).

### Technical Components:
*   **Causal Masking**: Ensures that at any timestep $i$, the model can only look at past bars $[0...i]$. It prevents the model from "cheating" by seeing future data within the input window.
*   **Multi-Head Output**:
    1.  **Direction Head**: Classification (Long, Short, or Neutral).
    2.  **Sizing Head**: Regression (Confidence ratio + Dynamic Take-Profit % + Dynamic Stop-Loss %).
*   **Positional Encoding**: Adds sinusoidal information so the model understands the chronological order of price bars.

---

## 📈 The Training Pipeline

The model undergoes a rigorous training process across multiple symbols (e.g., BTCUSD, ETHUSD) to ensure it learns universal market structures.

### Phase 1: Supervised Pre-training
In this phase, the model learns the "Grammar" of the market.
*   **Goal**: Minimize prediction error for direction and price targets.
*   **Loss Function**: A combination of **Cross-Entropy** (for direction) and **MSE** (for TP/SL targets).
*   **Oracle Labels**: We use an "Oracle" engine that looks ahead at actual historical future data to determine the "perfect" TP/SL for every bar. This gives the model a much higher quality signal than traditional indicators.

### Phase 2: Reinforcement Learning (Fine-Tuning)
This is where the model learns **"Consequence."** After the basics are learned, we enable a PnL-based penalty.
*   **The Logic**: We activate a third loss term called the **PnL Effect ($\gamma$)**.
*   **Mechanism**: 
    - If the model predicts a trade with high confidence (High Qty) but that trade results in a **Loss**, the loss value spikes exponentially.
    - If the model predicts a **Profit** but stayed hesitant (Low Qty), it is also penalized to encourage confidence during high-quality setups.
*   **Result**: The model stops being just "accurate" and starts being "profitable." It learns to stay Neutral during choppy markets to avoid the "PnL Penalty."

---

## 🏗️ How Phase 2 (RL) Works (Technical)

The RL phase is implemented as a **Fine-Tuning Layer** on top of the supervised model. Instead of a complex Agent/Environment setup, we use a **Differentiable PnL Loss**:

```python
# Technical breakdown of the RL Loss (from models.py)
is_loss = (actual_pnl < 0).float()
is_profit = (actual_pnl > 0).float()

# Penalty: Confident entry + Actual Loss = High Penalty
pnl_penalty = is_loss * abs(actual_pnl) * predicted_confidence

# Reward: Incentive to be confident on winning trades
pnl_reward = is_profit * abs(actual_pnl) * (1 - predicted_confidence)

total_loss = alpha*SignalLoss + beta*SizingLoss + gamma*(pnl_penalty - pnl_reward)
```

By minimizing this loss, the model back-propagates the "financial pain" of a bad trade directly into its weights.

---

## 🌐 Multi-Symbol Intelligence

The system doesn't just train on one coin. It uses **Global Pooling**:
1.  **Normalization**: Data from all coins is scaled using a unified `StandardScaler`. BTC at $60k and ETH at $3k are converted into comparable Z-scores.
2.  **Cross-Asset Learning**: The model trains on batches containing mixed sequences from all symbols.
3.  **Generalization**: Because it sees patterns from multiple assets, it avoids overfitting to the specific "quirks" of one coin, making it much more reliable for live trading.

---

## 🛠️ Usage

### 1. Requirements
Install dependencies:
```bash
pip install torch pandas numpy pandas_ta plotly rich scikit-learn
```

### 2. Run the Engine
Execute the main pipeline (Fetching -> Features -> Training -> Backtesting):
```bash
python main.py
```

### 3. Results
Check the `backtest_results/` folder for:
*   `portfolio_equity_curve.html`: Overall capital growth.
*   `[SYMBOL]_detailed_analytics.html`: Bar-by-bar trade markers and equity for each asset.

---
**Note**: This is a research project. Always use risk management before applying AI signals to live markets.
