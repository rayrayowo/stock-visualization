# B1 Scanner Dashboard - English Introduction

## Overview
This is a stock screening dashboard that helps you find stocks matching the **B1/B2 trading strategy** criteria.

## What is B1/B2 Strategy?
A technical analysis strategy that looks for stocks with:
- **B1**: J-value < 13 AND DEA > 0
- **B2**: J-value < 55
- **Plus**: White Brick (白砖头) + White Line > Yellow Line (白线>黄线)

## How to Use

### 1. Scan Stocks
Enter stock codes or names in the input box, one per line. Examples:
- `600875.SH` (Dongfang Electric)
- `000001.SZ` (Ping An Bank)
- `600519.SH` (Kweichow Moutai)

### 2. View Results
The dashboard will show:
- Stock code and name
- B1 or B2 classification
- J-value and other indicators

### 3. Technical Charts
Click on any stock to view detailed technical analysis charts including:
- Candlestick chart
- KDJ indicator
- MACD indicator
- Brick chart (砖型图)
- Trend lines (知行线)

## Features
- 📊 **Real-time scanning** of Chinese A-shares
- 📈 **Interactive charts** with Plotly
- 🔍 **Technical indicators**: KDJ, MACD, Brick Chart, Trend Lines
- 🌐 **Bilingual support**: English & Chinese
- 🎯 **Custom Filters**: Sector, Market Cap, Industry
- 📝 **User Strategies**: Import your own trading formulas

## 🚀 Future Possibilities

### 🧠 AI-Powered Strategies (Core Feature):
- **Neural Network Models**: Train your own ML models to predict stock movements
- **Strategy as Model**: Convert trading strategies into trainable ML models
- **Pattern Recognition**: CNN/LSTM for price pattern detection
- **Sentiment Analysis**: News & social media sentiment for trading signals
- **Quantitative Factor Models**: AI-generated alpha factors
- **Model Marketplace**: Share and trade pre-trained models

### 📈 More Visualizations:
- **Heatmaps**: Sector/industry performance heatmaps
- **Comparison Charts**: Multi-stock technical comparison
- **Candlestick Patterns**: Auto-detect patterns (doji, hammer, engulfing)
- **Volume Profile**: Volume at price analysis
- **Correlation Matrix**: Stock correlation heatmaps
- **Portfolio Visualizer**: P&L charts, allocation pie charts
- **Interactive Drawing Tools**: Trend lines, support/resistance
- **3D Charts**: Multi-dimensional data visualization
- **Animated Charts**: Historical price evolution
- **Model Performance**: Training curves, feature importance, prediction accuracy

### 🎯 More Strategies:
- **User-Defined Strategies**: Import custom trading formulas (TongDaXin/Futu compatible)
- **Built-in Strategies**: RSI, Bollinger Bands, MACD Crossover, Moving Averages
- **Custom Indicators**: Create your own technical indicators
- **Strategy Combinations**: AND/OR multiple conditions
- **AI Strategy Builder**: Natural language to trading rules
- **Hybrid Strategies**: Combine technical indicators with ML predictions

### 🔔 More Features:
- **Alert System**: Push notifications when stocks match criteria
- **Portfolio Tracking**: Monitor watchlist over time
- **Backtesting**: Test strategy against historical data
- **Export**: CSV, Excel, PDF reports
- **Scheduling**: Automatic daily scans
- **Multi-Account**: Connect multiple data sources

### 🔧 Technical Extensions:
- **Plugin System**: Community-shared indicators & strategies
- **API Access**: RESTful API for programmatic access
- **Mobile App**: iOS/Android companion app
- **Webhook Integration**: Connect to trading platforms
- **GPU Acceleration**: Train models faster with CUDA support
- **Auto-ML**: Automated hyperparameter tuning

## Data Source
- Data powered by **Tushare** API
- Formula based on **TongDaXin** (通达信) technical indicators

## Technical Stack
- **Frontend**: Streamlit + Plotly
- **Backend**: Python + Pandas + NumPy + PyTorch/TensorFlow
- **Data**: Tushare (Chinese A-shares)

---
*For questions, refer to the Chinese documentation or contact the developer.*
