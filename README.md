# 🚀 Advanced Neural Trading Engine (Stock LLM)

A high-fidelity, Transformer-based trading engine designed for 15-minute interval crypto/stock trading. This engine uses a Multi-Head architecture to predict price direction, trade sizing (TP/SL), and expected magnitude simultaneously.

---

## 🛠 Quick Start (Core Commands)

Run these commands from your terminal to manage the engine:

### 1. Oracle Perfect Backtest (100% Accuracy Baseline)
Use this to see the maximum possible profit your current strategy rules can generate.
```bash
uv run oracle_backtest.py
```

### 2. Model Training
Train the AI to recognize the patterns found by the Oracle.
```bash
# Deletes old memory and starts fresh training
rm models/short_thresholds.json models/short_model_eth.pth
uv run main.py --train --symbol ETHUSD
```

### 3. Model Testing (Backtest)
Evaluate the trained AI on recent "Out-of-Sample" data.
```bash
uv run main.py --test --symbol ETHUSD
```

---

## ⚙️ Configuration Guide (`config.py`)

The configuration is divided into logical blocks:

| Category | Key Variables | Use Case |
| :--- | :--- | :--- |
| **Data** | `TRAINING_DATA_DAYS (500)` | Kitne din ka data seekhne ke liye use hoga. |
| | `TEST_DATA_DAYS (30)` | Recent 1 month par model ka test hota hai. |
| **Strategy** | `ORACLE_MIN_RR (1.5)` | Trade lene ke liye minimum Profit vs Risk ratio. |
| | `MIN_ATR_STOP_PCT (0.8%)` | Market noise se bachne ke liye minimum Stop Loss. |
| **Model** | `WINDOW_SIZE (30)` | Pichli 30 candles (7.5 hours) dekh kar pattern pehchanna. |
| | `DROPOUT (0.40)` | Overfitting rokne ke liye neurons ko randomly ignore karna. |
| **Training** | `LR (0.0003)` | Seekhne ki raftaar. Slow = Better Pattern recognition. |

---

## 🧠 Model Architecture: Multi-Head Transformer

Instead of a simple "Up/Down" prediction, this model uses a **Multi-Task Learning** approach:

### How it works (Example):
When the model sees a price action (e.g., a "Hammer" candle after a downtrend):
1.  **Head 1 (Direction)**: Predicts `SHORT`, `LONG`, or `NEUTRAL`.
2.  **Head 2 (Sizing)**: Predicts exactly where the **Take Profit** and **Stop Loss** should be based on current volatility.
3.  **Head 3 (Magnitude)**: Predicts how big the move will be (e.g., +2.5%).
4.  **Head 4 (Time)**: Predicts how many candles it will take to hit the target.

### Internal Structure:
- **Input**: 40+ Features (RSI, MACD, Volume Z-Score, etc.)
- **Processor**: Transformer Encoder with Multi-Head Attention.
- **Output**: 4 Parallel Linear Heads.

---

## 🔍 Deep Dive: Understanding AI Predictions

When you run a prediction, the model doesn't just give a "Buy/Sell" signal. It outputs a complex data structure:

### 1. Directional Verdict (Probability)
The model outputs a probability distribution across 3 classes:
- **Long**: Confidence that price will hit `ORACLE_MIN_RR` target before stop.
- **Neutral**: High volatility but no clear direction, or sideways market.
- **Short**: Confidence that price will drop to target.
*The engine only takes a trade if the confidence exceeds the threshold (e.g., > 45%).*

### 2. Intelligent Sizing (Dynamic TP/SL)
Unlike traditional bots, our AI predicts the **Take Profit (TP)** and **Stop Loss (SL)** for *every* trade based on market volatility:
- **tp_pct**: Calculated as `predicted_sizing * MAX_ATR_TARGET_PCT`.
- **sl_pct**: Calculated as `predicted_sizing * MAX_ATR_STOP_PCT`.

### 3. Price Magnitude & Horizon
- **Predicted Magnitude**: How much % the price is expected to move in the predicted direction.
- **Predicted Time**: How many bars (15m intervals) the model expects the trade to stay active.

---

## 💻 Example: Model Output Structure
If you print the model's output during inference, it looks like this:

```python
{
    "verdict": "SHORT",          # Predicted Side
    "confidence": 0.82,          # 82% sure about the move
    "take_profit_pct": 1.45,     # AI suggested target (+1.45%)
    "stop_loss_pct": 0.80,       # AI suggested risk (-0.80%)
    "expected_magnitude": 2.10,  # Expected total drop
    "expected_duration": 12      # Expected to hit target in 3 hours (12 bars)
}
```

---

## 🚀 How to use the Model for Live Prediction

---

## 📊 Output Format (Standard Report)

Every backtest generates a professional dashboard:

```text
   Core Strategy Performance
 Metric                   Value 
 Initial Capital         $50.00 
 Net Profit (PnL)       +$58.82 
 Final Equity           $108.82 
 ROI (%)               +117.64% 
 Total Trades                61 
   └─ Long Trades            33 
   └─ Short Trades           28 
 Win Rate               100.00% 
```

### Metric Definitions:
- **ROI (%)**: Return on Investment from your starting capital.
- **Reliability Gap**: Difference between AI confidence on winners vs losers. (Higher is better).
- **Max Drawdown**: The biggest "dip" your balance took during the test.

---

## 📂 Project Structure
- `neural_engine/`: Core AI logic (Model, Trainer, Labeler).
- `engine/`: Execution logic (Backtester, Data Handler).
- `models/`: Saved weights (`.pth`) and normalization files (`.npy`).
- `data/`: CSV cache of market data.
