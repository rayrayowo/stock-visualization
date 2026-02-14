##U.S. Stock Market Pattern Analysis

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
