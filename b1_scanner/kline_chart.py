"""
K线图表组件 - 可缩放/平移的交互式K线图
参考富途牛牛界面风格
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_kline_chart(
    df: pd.DataFrame,
    show_volume: bool = True,
    show_ma: bool = True,
    show_zhixing: bool = True,
    show_bbi: bool = False,
    title: str = "",
    width: int = None,
    height: int = 600,
) -> go.Figure:
    """
    创建交互式K线图表
    
    参数:
        df: 包含 OHLCV 数据的 DataFrame
        show_volume: 显示成交量
        show_ma: 显示均线 (MA5, MA10, MA20)
        show_zhixing: 显示知行线 (白线/黄线)
        show_bbi: 显示BBI线
    
    返回:
        Plotly Figure 对象
    """
    if df is None or len(df) == 0:
        return None

    # 确保必要的列存在，并处理 vol vs volume 命名
    df = df.copy()
    if "vol" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"vol": "volume"})
    
    # 设置索引为日期（如果 trade_date 存在）
    if "trade_date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date")
        df = df.sort_index()

    # 计算缺失的指标（如果数据来自 add_all_indicators，部分指标可能已有）
    for period in [5, 10, 20]:
        col = f"ma{period}"
        if show_ma and col not in df.columns:
            df[col] = df["close"].rolling(window=period).mean()

    if show_zhixing and "zhixing_white" not in df.columns:
        df["zhixing_white"] = df["close"].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    if show_zhixing and "zhixing_yellow" not in df.columns:
        df["zhixing_yellow"] = (
            df["close"].rolling(14).mean()
            + df["close"].rolling(28).mean()
            + df["close"].rolling(57).mean()
            + df["close"].rolling(114).mean()
        ) / 4

    if show_bbi and "bbi" not in df.columns:
        df["bbi"] = (
            df["close"].rolling(3).mean()
            + df["close"].rolling(6).mean()
            + df["close"].rolling(12).mean()
            + df["close"].rolling(24).mean()
        ) / 4

    # 创建子图
    rows = 2 if show_volume else 1
    row_heights = [0.8, 0.2] if show_volume else [1.0]
    
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=("", "成交量") if show_volume else ("",)
    )
    
    # K线蜡烛图
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='K线',
            increasing_line_color='#FF4B4B',  # 红色上涨
            decreasing_line_color='#2ECC71',   # 绿色下跌
            increasing_fillcolor='#FF4B4B',
            decreasing_fillcolor='#2ECC71',
        ),
        row=1, col=1
    )
    
    # 均线
    if show_ma and 'ma5' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ma5'], name='MA5', line=dict(color='#FF6B6B', width=1)), row=1, col=1)
    if show_ma and 'ma10' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ma10'], name='MA10', line=dict(color='#4ECDC4', width=1)), row=1, col=1)
    if show_ma and 'ma20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ma20'], name='MA20', line=dict(color='#45B7D1', width=1)), row=1, col=1)
    
    # 知行线 (白线/黄线)
    if show_zhixing:
        if 'zhixing_white' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['zhixing_white'], name='知行白线', 
                                     line=dict(color='#FFFFFF', width=2), hoverinfo='skip'), row=1, col=1)
        if 'zhixing_yellow' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['zhixing_yellow'], name='知行黄线', 
                                     line=dict(color='#FFD700', width=2), hoverinfo='skip'), row=1, col=1)
    
    # BBI线
    if show_bbi and 'bbi' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['bbi'], name='BBI', 
                                 line=dict(color='#9B59B6', width=1.5, dash='dash')), row=1, col=1)
    
    # 成交量
    if show_volume and 'volume' in df.columns:
        colors = ['#FF4B4B' if df['close'].iloc[i] >= df['open'].iloc[i] else '#2ECC71' 
                 for i in range(len(df))]
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='成交量', marker_color=colors, hoverinfo='x+y'),
            row=2, col=1
        )
    
    # 样式设置
    layout_kwargs = dict(
        title=title,
        template="plotly_dark",
        height=height,
        xaxis_rangeslider_visible=False,  # 隐藏底部滑块
        dragmode='pan',  # 默认平移模式
        hovermode='x unified',
        # 富途牛牛风格配色
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#e0e0e0'),
    )
    if width:
        layout_kwargs["width"] = width
    fig.update_layout(**layout_kwargs)
    
    # 添加缩放按钮
    idx_len = len(df.index)
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.7,
                y=1.15,
                buttons=[
                    dict(
                        label="🔍+",
                        method="relayout",
                        args=[{"xaxis.range[0]": str(df.index[-50]), "xaxis.range[1]": str(df.index[-1])}]
                    ),
                    dict(
                        label="🔍-",
                        method="relayout",
                        args=[{"xaxis.range[0]": str(df.index[-min(200, idx_len)]), "xaxis.range[1]": str(df.index[-1])}]
                    ),
                    dict(
                        label="📊全",
                        method="relayout",
                        args=[{"xaxis.range": [str(df.index[0]), str(df.index[-1])]}]
                    ),
                ]
            )
        ]
    )
    
    # 隐藏周末空白
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    
    return fig


def calculate_indicators_for_chart(df: pd.DataFrame) -> pd.DataFrame:
    """为K线图计算需要的指标"""
    df = df.copy()
    
    # 均线
    for period in [5, 10, 20]:
        df[f'ma{period}'] = df['close'].rolling(window=period).mean()
    
    # 知行线
    df['zhixing_white'] = df['close'].ewm(span=10, adjust=False).mean().ewm(span=10, adjust=False).mean()
    df['zhixing_yellow'] = (df['close'].rolling(14).mean() + df['close'].rolling(28).mean() + 
                           df['close'].rolling(57).mean() + df['close'].rolling(114).mean()) / 4
    
    # BBI (可选)
    df['bbi'] = (df['close'].rolling(3).mean() + df['close'].rolling(6).mean() + 
                 df['close'].rolling(12).mean() + df['close'].rolling(24).mean()) / 4
    
    return df


if __name__ == "__main__":
    # 测试
    import tushare as ts
    
    pro = ts.pro_api("3a870845a82bc2a522a1b9dbc324df8b0be58390ac0088804243a615")
    df = pro.daily(ts_code="600519.SH", end_date="20260325", limit=120)
    df = df.sort_values('trade_date').reset_index(drop=True)
    
    df = calculate_indicators_for_chart(df)
    fig = create_kline_chart(df, show_bbi=True)
    fig.show()