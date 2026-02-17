## U.S. Stock Market Pattern Analysis

A data-driven exploration of market behavior, technical indicators, and regime dynamics using historical S&P 500 stock data.

Project Objective

Financial markets exhibit recurring statistical structures such as volatility clustering, regime shifts, and cross-sector correlation changes.

This project investigates:

Moving average crossover behavior

Volatility regimes and return distributions

Cross-sectional stock correlations

Indicator performance under different market conditions

Rather than focusing on price prediction, this project emphasizes empirical analysis and interpretable visualization of financial time-series patterns.

Dataset

Source: Kaggle – S&P 500 Historical Stock Data

Coverage: ~5 years of daily OHLCV data

Size: ~619,000 rows

Key Components
Data Processing

Cleaning and validation of time-series data

Computation of returns and rolling volatility

Technical indicator generation (e.g., moving averages)

Visualization

Time-series overlays (price + indicators)

Correlation heatmaps

Regime-based distribution comparisons

Interactive dashboard (Power BI integration)

Project Structure
data/        → raw dataset  
notebooks/   → exploratory analysis  
src/         → reusable indicator & preprocessing modules  
Tools & Technologies

Python (Pandas, NumPy, Matplotlib / Plotly / Altair)

Power BI (interactive dashboard exploration)

Developed as part of a financial data visualization study.

## Multi-Strategy Backtest Workflow

This repo now supports a reusable evaluation pipeline with three layers:

1. `src/clean_merge01.py`  
Builds `master` table (merge + indicators + regime + forward returns).

2. `src/strategies.py`  
Defines strategy rules (MA / MACD / KDJ baseline + `custom_template`).

3. `src/backtest.py` + `src/evaluate.py`  
Runs portfolio backtest and exports overall/regime/sector/leader metrics.

### Run Steps

```bash
.venv/bin/python src/clean_merge01.py
.venv/bin/python src/evaluate.py
```

Outputs are saved to `data/processed/`:

- `master.parquet` / `master.csv`
- `market_proxy_daily.csv`
- `strategy_summary.csv`
- `strategy_daily_returns.csv`
- `strategy_trades.csv`

### Implement Your Own Strategy

Edit `strategy_custom_template()` in `src/strategies.py` and replace entry/exit logic with your B1/B2 rules.

Then run:

```bash
.venv/bin/python src/evaluate.py --strategies custom_template
```

## Generate Simplified B1/B2 Signals (v2)

How to generate signals:

```bash
python src/strategy_signals_v2.py
python src/strategy_signals_v2.py --skip-csv
```
