#!/usr/bin/env python3
"""
Visualization Module for B1 Scanner
添加散点图和K线图可视化功能
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta


def get_stock_data(symbol, period="1y"):
    """获取股票数据"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def calculate_indicators(df):
    """计算基础技术指标"""
    if df is None or df.empty:
        return df
    
    # 简单移动平均
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    # 涨跌幅
    df['Change_Pct'] = df['Close'].pct_change() * 100
    
    return df


def create_scatter_plot(stock_list, x_col="Change_Pct", y_col="Volume"):
    """创建散点图"""
    data = []
    
    for symbol in stock_list:
        df = get_stock_data(symbol, period="1mo")
        if df is not None and not df.empty:
            df = calculate_indicators(df)
            latest = df.iloc[-1]
            data.append({
                "Symbol": symbol,
                "Close": latest["Close"],
                "Volume": latest["Volume"],
                "Change_Pct": latest["Change_Pct"],
                "RSI": latest["RSI"] if "RSI" in latest else None,
                "MA20": latest["MA20"] if "MA20" in latest else None,
            })
    
    if not data:
        return None
    
    df_plot = pd.DataFrame(data)
    
    fig = px.scatter(
        df_plot, 
        x=x_col, 
        y=y_col,
        color="Symbol",
        size="Volume",
        hover_data=["Close", "RSI"],
        title=f"Stock Scatter Plot: {x_col} vs {y_col}",
        template="plotly_dark"
    )
    
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode="closest"
    )
    
    return fig


def create_candlestick_chart(symbol, period="3mo", show_ma=True, show_volume=True):
    """创建K线图"""
    df = get_stock_data(symbol, period=period)
    
    if df is None or df.empty:
        return None
    
    df = calculate_indicators(df)
    
    # 创建子图
    rows = 2 if show_volume else 1
    row_heights = [0.7, 0.3] if show_volume else [1.0]
    
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=(
            f"{symbol} Candlestick" + (" + MA" if show_ma else ""),
            "Volume" if show_volume else ""
        )
    )
    
    # K线图
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # 添加MA线
    if show_ma:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA5'], name="MA5", line=dict(width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA10'], name="MA10", line=dict(width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MA20'], name="MA20", line=dict(width=1.5)),
            row=1, col=1
        )
    
    # 添加成交量
    if show_volume:
        colors = ['red' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'green' 
                  for i in range(len(df))]
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name="Volume", marker_color=colors, opacity=0.5),
            row=2, col=1
        )
    
    # 更新布局
    fig.update_layout(
        title=f"{symbol} - {period}",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600 if show_volume else 400
    )
    
    return fig


def create_multi_stock_compare(symbol_list, period="1mo"):
    """创建多股票比较图表"""
    data = []
    
    for symbol in symbol_list:
        df = get_stock_data(symbol, period=period)
        if df is not None and not df.empty:
            df = calculate_indicators(df)
            # 归一化价格
            df['Normalized'] = df['Close'] / df['Close'].iloc[0] * 100
            df['Symbol'] = symbol
            data.append(df[['Symbol', 'Normalized', 'Close']].copy())
    
    if not data:
        return None
    
    combined = pd.concat(data)
    
    fig = px.line(
        combined, 
        x=combined.index, 
        y="Normalized", 
        color="Symbol",
        title="Normalized Price Comparison (Base=100)",
        template="plotly_dark"
    )
    
    fig.update_layout(
        yaxis_title="Normalized Price",
        hovermode="x unified"
    )
    
    return fig


# 测试
if __name__ == "__main__":
    # 测试K线图
    fig = create_candlestick_chart("AAPL", period="3mo")
    if fig:
        print("Candlestick chart created successfully")
        # fig.show()
