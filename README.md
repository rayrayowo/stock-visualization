# Trade Wizard

> An experimental framework for algorithmic trading strategies combining technical indicators, neural network signal composition, and white-brick (砖型图) pattern recognition.

**Status:** Active development
**Authors:** Ruiyang Zhang & Alan

---

## Project Overview

Trade Wizard is a modular trading strategy research platform. It supports:

- **Technical analysis** — KDJ, MACD, RSI, Bollinger Bands, Zhixing trend lines, brick charts
- **Signal composition** — Node-based visual strategy builder (AND/OR/NOT logic)
- **Neural network** — MLP-based walk-forward backtesting
- **White Brick (白砖)** — TongDaXin-style brick chart pattern detection
- **Real-time screening** — Streamlit UI for live stock scanning (A-shares + US markets)

---

## Project Structure

```
trade_wizards_mvp/     # Neural network + node-based strategy backtesting
├── tw_mvp/             # Core engine: nodes, pipeline, backtesting, reporting
│   ├── nodes.py        # NodeGraphEngine — signal composition
│   ├── pipeline.py     # End-to-end: data → features → graph → backtest → metrics
│   ├── backtesting.py  # Portfolio backtest engine
│   ├── features.py     # Technical indicator features
│   ├── data_pipeline.py # Feature dataset builder
│   └── reporting.py    # Chart annotations + report generation
├── config/             # Neural node graph configurations
├── outputs/             # Backtest results and generated charts
│   └── neural_node_sandbox_*/   # Historical run outputs
└── run_demo.py         # Launch the neural node backtest demo

b1_scanner/            # Real-time B1/B2 strategy stock scanner
├── app.py            # Streamlit UI
├── scanner_core.py   # Strategy signal logic
├── indicators.py     # Technical indicator calculations
├── data_sources.py   # Tushare + Yahoo Finance data
└── kline_chart.py   # Candlestick chart generation
```

---

## Quick Start

### Neural Node Backtest (trade_wizards_mvp)

```bash
cd trade_wizards_mvp
python3 run_demo.py
# Results → outputs/neural_node_sandbox_final/
```

### Real-Time Stock Scanner (b1_scanner)

```bash
cd b1_scanner
pip install -r requirements.txt
export TUSHARE_TOKEN="your_token_here"   # Optional, for A-shares
streamlit run app.py
```

---

## Technical Stack

- **Languages:** Python 3
- **Data:** Tushare Pro (A-shares), Yahoo Finance (US/HK)
- **Indicators:** KDJ(9,3,3), MACD(12,26,9), RSI, BOLL, Zhixing Lines, Brick Chart
- **ML:** scikit-learn (MLPClassifier), walk-forward backtesting
- **UI:** Streamlit
- **Visualization:** Plotly, Matplotlib, PIL

---

## Key Strategies

### B1 — Oversold Rebound
KDJ J < 13 + MACD DEA > 0 + White line > Yellow line → buy on next-day open

### B2 — Momentum Confirmation
After B1 + daily gain ≥ 4% + KDJ J < 55 + volume surge → buy confirmation

### Neural Node Strategy
Composable node graph: AND/OR/NOT logic combining KDJ, MACD, RSI, and MA signals

### Super White Brick
Brick chart with TongDaXin EMA approximation for A-share trend detection

---

## Data

- **S&P 500:** `data/raw/all_stocks_5yr.csv` — 505 US stocks, 2013–2018
- **A-shares:** Tushare Pro (requires token)

---

## License

MIT License — Educational purposes only. Not investment advice.
