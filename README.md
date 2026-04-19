# Stock LLM Project

A multi-task transformer model for stock trading with risk-aware loss and interactive visualization.

## Features
- **Data Gathering**: Automated fetching of OHLC data from Delta Exchange with CSV caching.
- **Feature Engineering**: Comprehensive technical indicators including EMAs, Bollinger Bands, RSI, MACD, Supertrend, and more.
- **Model Architecture**: Multi-task Transformer model optimized for trading decisions.
- **Risk-Aware Loss**: Custom loss function that penalizes high-confidence errors and incorporates future drawdown risk.
- **Interactive Visualization**: Professional-grade Plotly dashboard for market analysis.

## Setup
This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Run the visualization dashboard
uv run main.py
```

## Project Structure
- `main.py`: Entry point and execution pipeline.
- `data_gathering.py`: API data fetching logic.
- `features.py`: Technical indicator and feature engineering.
- `models.py`: PyTorch model and loss function definitions.
- `visualisation.py`: Plotly-based dashboard.
