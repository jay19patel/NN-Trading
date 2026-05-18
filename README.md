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
| **Data** | `TRAINING_DATA_DAYS (1000)` | Kitne din ka data seekhne ke liye use hoga. |
| | `TEST_DATA_DAYS (100)` | Recent test window par model ka test hota hai. |
| **Strategy** | `ORACLE_MIN_RR (1.0)` | Trade lene ke liye minimum Profit vs Risk ratio. |
| | `MIN_ATR_STOP_PCT (0.30%)` | Market noise se bachne ke liye minimum Stop Loss. |
| **Model** | `WINDOW_SIZE (96)` | Pichli 96 hourly candles (4 days) dekh kar pattern pehchanna. |
| | `DROPOUT (0.30)` | Overfitting rokne ke liye neurons ko randomly ignore karna. |
| **Training** | `LR (0.0001)` | Seekhne ki raftaar. Slow = Better Pattern recognition. |

---

## 🧠 AI Model: Input & Output Architecture

Hamara **Multi-Head Transformer Model** kaise data leta hai aur kya soch kar output nikalta hai, iski poori technical detail yaha di gayi hai.

### 1. Model Ko Kya Data Diya Jata Hai? (INPUTS)
Model ek 3D tensor leta hai jiska shape hai: `(Batch Size, WINDOW_SIZE, Num Features)`
* **WINDOW_SIZE:** `96` (Yani model ek baar me pichle 96 hourly candles ek sath dekhta hai).
* **Num Features:** `73` (1h execution features + completed 4h/1d regime context).

**73 Features me kya-kya hota hai?**
1. **Price Data:** Log returns (`log_return_1`, `log_return_6`, `log_return_24` etc.)
2. **EMA/RSI Trend Stack:** `dist_ema_21`, `ema_21_50_spread`, `rsi_14`, `stochrsi_k`
3. **Volatility & Structure:** `natr`, `realized_vol_24`, `bb_width`, `candle_vs_atr`
4. **Volume & Participation:** `price_to_vwap`, `volume_surprise_50`, `obv_slope_20`
5. **Regime Quality:** `efficiency_ratio_10`, `range_compression_10_50`, `trend_regime`
6. **Multi-Timeframe Context:** previous completed `4h` and `1d` EMA/RSI/volatility features
7. **Time Context:** `hour_sin`, `dow_sin`

### 2. Model Kya Predict Karta Hai? (OUTPUTS)
Model ek sath **3 alag-alag cheezein (Heads)** predict karta hai:
1. **Signal Head (Direction):** Predicts probabilities for LONG, NEUTRAL, and SHORT.
2. **Sizing Head (Target & Risk):** Predicts Quantity Ratio, Normalized Take Profit %, and Normalized Stop Loss %.
3. **Magnitude Head:** Predicts the total expected move magnitude.

---

## 🚀 EXACT REPLICA EXAMPLE (Python/JSON Format)

Agar aap code ke through model ko use karenge (Inference phase), toh data kuch is tarah ka dikhega:

### 📥 INPUT DATA (Jo hum model ko pass karte hai)
```python
import numpy as np
import torch

# 1 Sample (Batch), 96 Candles ka history, 73 Features har candle ke
input_tensor = torch.tensor([
    [
        # Candle 1 (Sabse purani)
        [0.0012, -0.05, 1.05, 0.45, ... 73rd feature],
        # Candle 2
        [0.0021, -0.02, 1.10, 0.47, ... 73rd feature],
        # ...
        # Candle 96 (Abhi ki current candle)
        [-0.0010, 0.01, 0.98, 0.35, ... 73rd feature]
    ]
]) # Shape: (1, 96, 73)

### 📤 RAW OUTPUT DATA (PyTorch Tensors)
Jab model ka inference chalta hai, toh wo seedha dictionary nahi deta. Wo 3 PyTorch Tensors return karta hai (Kyuki 3 Heads hai):

```python
# Model Returns 3 Tensors at once!
signal_logits, sizing, magnitude = model(input_tensor)

print(signal_logits) 
# Output: tensor([[-1.245, 0.531, 2.103]], grad_fn=<AddmmBackward0>) 
# (Yani: [LONG, NEUTRAL, SHORT] ke raw scores, jinhe hum Softmax karke % banate hai)

print(sizing)
# Output: tensor([[0.950, 0.055, 0.150]], grad_fn=<SigmoidBackward0>)
# (Yani: [Qty_Ratio, Raw_TP, Raw_SL] 0-1 scale par)

print(magnitude)
# Output: tensor([[0.420]], grad_fn=<SigmoidBackward0>)
# (Yani: Expected move size)
```

### ⚙️ PARSED OUTPUT (Dictionaries)
Jab hum un raw tensors ko engine me clean kar lete hai, tab wo aise dikhte hai:

```python
# 1. Signal Head Output (Logits -> Probabilities me convert karne ke baad)
ai_probabilities = {
    "LONG": 0.526,    # 52.6% confidence market upar jayega
    "NEUTRAL": 0.310, # 31.0% confidence flat rahega
    "SHORT": 0.164    # 16.4% confidence niche jayega
}

# 2. Sizing Head Output (0-1 format me)
sizing_output = {
    "qty_ratio": 0.95,          # 95% quantity use karo
    "raw_tp": 0.055,            # Raw value from sigmoid
    "raw_sl": 0.150             # Raw value from sigmoid
}

# 3. Magnitude Head Output
expected_volatility = 0.42      # Expected overall move strength
```

### Final Trading Signal (Jo engine ko samajh aayega)
```json
{
    "action": "LONG",
    "confidence": 52.6,
    "entry_price": 2345.00,
    "take_profit_pct": 0.33,
    "stop_loss_pct": 0.30,
    "suggested_quantity_ratio": 0.95
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
