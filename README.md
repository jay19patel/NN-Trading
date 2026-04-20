# 🤖 Trading AI Transformer Dashboard

A simple explanation of how our AI learns to trade market patterns using 135+ indicators.

---

## 🌟 What is this?
Imagine a student who looks at the last **365 days** of the market every single second. This AI (the student) studies patterns to understand which market "moods" lead to a profit and which ones lead to a loss.

## 🧠 How the AI "Brain" works (Very Simple)

The AI doesn't just look at the price. It looks at **135 different "Feelings"** of the market, such as:
- **RSI**: Is the market too "tired" (Overbought) or "hungry" (Oversold)?
- **Moving Averages**: Is the general trend going Up or Down?
- **Volatility**: Is the market "Angry" (Swingy) or "Calm" (Stable)?

### 🚦 The 3-Signal System (Multiple Classes)
We train the model to sort every moment into one of **3 boxes**:

1.  **🟢 Box 0 (BUY)**: If the price goes UP by 1.2% before it drops by 1%, we tell the AI: *"This was a GOOD pattern to Buy."*
2.  **🟡 Box 1 (NEUTRAL)**: If the price stays sideways or hits the Stoploss first, we tell the AI: *"This was a BORING/RISKY pattern. Stay away."*
3.  **🔴 Box 2 (SELL)**: If the price goes DOWN by 1.2% before it rises by 1%, we tell the AI: *"This was a GOOD pattern to Sell."*

The AI tries to find the common patterns that happen **BEFORE** price goes into the Buy or Sell box.

---

## 🚀 Step-by-Step Process

### 1. Data Collection
We download 1 year of data. We use **Caching** so we don't have to download it every time (saves internet and time).

### 2. Feature Engineering (The "Decoder")
We turn raw price into 135 indicators. This is like turning raw ingredients into a detailed recipe for the AI.

### 3. Analytics (The "Noise Filter")
Not all indicators are useful. Our system automatically throws away "Kachra" (Noise) indicators that don't help the AI, so it can focus only on "Powerful" signals.

### 4. Training (The "Study Time")
The AI "exercises" its brain on the historical data. It checks its own guesses and corrects itself until it becomes accurate.
- **Train-Acc**: How well it remembers the past.
- **Val-Acc**: How well it can guess the future (The real test!).

### 5. Verdict (The "Final Exam")
We test the AI on the **last 10 days** (data it has never seen). If the AI says "BUY" and the price actually hits our 1.2% target, it's a **WIN**. If it hits the stoploss, it's a **FAIL**.

---

## 🛠️ How to run it?
If you have `uv` installed, just run:
```bash
uv run main.py
```

## 📈 Understanding the Console
- **Top 20 Conditions**: What the market looks like during Buy vs Sell.
- **Feature Health**: Which indicators are the "Boss" and which ones are "Useless".
- **Win Rate**: The final score of the AI's performance.

---

## 🏗️ Deep Technical Architecture (How the Brain Thinks)

If you are a developer or a quant, here is exactly how the data flows through the **Transformer Architecture**:

### 1. The Data Pipeline (Raw → Vectors)
Before the AI "reads" the data, everything is converted into **Vectors** (numbers between -1 and 1).
- **Step A**: 135 market features are calculated.
- **Step B (Standardization)**: We use a `StandardScaler`. This is important because the AI can't compare a Price (e.g., $1.50) with an RSI (e.g., 70) directly. Scaling makes them "speak the same language."

### 2. The Transformer "Core" (Attention is All You Need)
Our model uses a **Transformer**, which is the same technology behind ChatGPT, but tuned for numbers.

#### 🧠 Multi-Head Attention
This is the most important part. Instead of looking at everything at once, `MultiHeadAttention` uses 8 "Heads" (like 8 different experts):
- **Expert 1**: Looks only at Volume peaks.
- **Expert 2**: Looks only at Trend direction.
- **Expert 3**: Looks at Volatility swings.
By combining these 8 perspectives, the model gets a **3D view** of the market state.

#### 🧱 TradingTransformerBlock
This is a repeated layer that cleans the signals. It asks: *"Is this RSI spike actually important given the current high Volume?"* If yes, the signal is strengthened. If no, it is suppressed (ignored).

### 3. Multi-Head Trading Model (The Multi-Tasker)
Most models only predict "Buy" or "Sell." Our model is a **Multi-Tasker**. It has 4 separate outputs:
1.  **🚦 Direction**: The 3-class classification (Buy/Sell/Wait).
2.  **📈 Upside Prediction**: A number guessing how high it can go.
3.  **📉 Downside Prediction**: A number guessing the maximum risk.
4.  **⚠️ Risk Metric**: A confidence score.

### 4. RiskAwareLoss (The "Strict Teacher")
We don't use standard math (MSE/CrossEntropy) alone. We use a **Custom Loss Function**.
- If the AI guesses "Buy" but the market crashes, `RiskAwareLoss` punishes the model **twice as hard**.
- We force the model to minimize **Volatility** and maximize **Reward**, making it a "Risk-Averse" trader.

---

*Technical Note: Optimized for Apple Silicon (MPS) for 10x faster training iterations.*
