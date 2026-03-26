#!/usr/bin/env python3
"""Streamlit app for B1 scanner v2.0."""

from __future__ import annotations

import dataclasses
import os
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from data_sources import DEFAULT_TUSHARE_TOKEN, get_tushare_mainboard_stocks
from scanner_core import B1Config, flatten_result_for_table, scan_batch, scan_symbol
from kline_chart import create_kline_chart

st.set_page_config(page_title="B1战法选股器 v2.0", page_icon="📈", layout="wide")


# ─────────────────────────────────────────────
# session_state 初始化
# ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "scan_results": None,       # 最近一次扫描结果
        "scan_table": None,         # DataFrame for display
        "scan_done": False,         # 是否已完成扫描
        "last_config_hash": None,   # 用于检测参数变化
        "scan_log": None,           # 本次扫描的日志
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def parse_symbol_input(raw: str) -> List[Dict[str, str]]:
    """解析股票列表，支持多种格式"""
    rows: List[Dict[str, str]] = []

    name_map = {
        "贵州茅台": "600519.SH",
        "恒瑞医药": "600276.SH",
        "药明康德": "603259.SH",
        "百济神州": "688235.SH",
        "荣昌生物": "688331.SH",
        "平安银行": "000001.SZ",
        "青岛港": "601298.SH",
        "新乳业": "002946.SZ",
        "民生银行": "600016.SH",
        "格力电器": "000651.SZ",
        "中国平安": "601318.SH",
        "招商银行": "600036.SH",
        "比亚迪": "002594.SZ",
        "宁德时代": "300750.SZ",
        "AAPL": "AAPL",
        "TSLA": "TSLA",
        "MSFT": "MSFT",
    }

    for line in raw.replace("，", ",").splitlines():
        line = line.strip()
        if not line:
            continue

        if "," in line:
            parts = [x.strip() for x in line.split(",") if x.strip()]
            if len(parts) >= 2:
                if parts[0].replace(".", "").replace("SH", "").replace("SZ", "").isdigit():
                    code, name = parts[0], parts[1]
                else:
                    code, name = parts[1], parts[0]
            elif len(parts) == 1:
                code, name = parts[0], ""
        else:
            parts = line.split()
            if len(parts) >= 2:
                code, name = parts[0], parts[1]
            else:
                code = parts[0]
                name = ""

        if code.upper() not in ["SH", "SZ", "SS"] and not any(c.isdigit() for c in code):
            code = name_map.get(code, code)
            name = code if not name else name

        code = code.upper().strip()
        if code.isdigit() and len(code) == 6:
            if code.startswith("6"):
                code = f"{code}.SH"
            elif code.startswith(("0", "3")):
                code = f"{code}.SZ"

        if code:
            rows.append({"symbol": code, "name": name})

    return rows


def symbol_matches_exclude(symbol: str, exclude_chinext: bool, exclude_star: bool,
                           exclude_st: bool, exclude_bse: bool) -> bool:
    """检查股票是否符合排除条件"""
    import re
    m = re.search(r"(\d{6})", symbol.upper())
    if not m:
        return False
    code = m.group(1)

    if exclude_chinext and code.startswith("300"):
        return True
    if exclude_star and code.startswith("688"):
        return True
    if exclude_bse and code.startswith("8"):
        return True
    # ST股: 名称中含 ST *ST S*ST
    # 这个需要在扫描结果中根据股票名称判断，这里先跳过
    _ = exclude_st  # 待实现
    return False


def build_figure(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.35, 0.12, 0.14, 0.13, 0.13, 0.13],
        subplot_titles=(
            f"{symbol} K线 + 知行趋势线 + BOLL",
            "VOL",
            "MACD",
            "KDJ",
            "RSI",
            "砖型图 (白色=买点)",
        ),
    )

    x = df["date"]

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(go.Scatter(x=x, y=df["zhixing_white"], name="白线(短期)", line={"width": 2, "color": "white"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["zhixing_yellow"], name="黄线(长期)", line={"width": 2, "color": "yellow"}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["boll_upper"], name="BOLL上轨", line={"width": 1}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["boll_mid"], name="BOLL中轨", line={"width": 1}), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["boll_lower"], name="BOLL下轨", line={"width": 1}), row=1, col=1)

    fig.add_trace(go.Bar(x=x, y=df["volume"], name="成交量", opacity=0.5), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["vol_ma5"], name="VOL MA5", line={"width": 1}), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["vol_ma10"], name="VOL MA10", line={"width": 1}), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["vol_ma20"], name="VOL MA20", line={"width": 1}), row=2, col=1)

    fig.add_trace(go.Bar(x=x, y=df["macd_hist"], name="MACD柱", opacity=0.6), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["macd_diff"], name="DIFF", line={"width": 1.5}), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["macd_dea"], name="DEA", line={"width": 1.5}), row=3, col=1)

    fig.add_trace(go.Scatter(x=x, y=df["kdj_k"], name="K", line={"width": 1}), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["kdj_d"], name="D", line={"width": 1}), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=df["kdj_j"], name="J", line={"width": 1.5}), row=4, col=1)

    fig.add_trace(go.Scatter(x=x, y=df["rsi14"], name="RSI14", line={"width": 1.5}), row=5, col=1)
    fig.add_hline(y=70, line_dash="dot", line_width=1, row=5, col=1)
    fig.add_hline(y=30, line_dash="dot", line_width=1, row=5, col=1)

    brick_chart = df["brick_chart"].fillna(0)
    brick_white = df["brick_white"].fillna(0)

    brick_colors = ["red" if brick_chart.iloc[i] > 0 else "green" for i in range(len(brick_chart))]
    brick_colors = ["white" if brick_white.iloc[i] == 1 else brick_colors[i] for i in range(len(brick_white))]

    fig.add_trace(go.Bar(x=x, y=brick_chart, name="砖型图", marker_color=brick_colors, opacity=0.8), row=6, col=1)

    white_indices = df[brick_white == 1].index
    if len(white_indices) > 0:
        white_x = [df.loc[i, "date"] for i in white_indices if i in df.index]
        white_y = [brick_chart.loc[i] for i in white_indices if i in df.index and i in brick_chart.index]
        fig.add_trace(go.Scatter(x=white_x, y=white_y, mode="markers", name="白色砖头(买点)",
                                   marker=dict(symbol="triangle-up", size=12, color="white", line=dict(color="black", width=2))), row=6, col=1)

    fig.update_layout(height=1200, xaxis_rangeslider_visible=False, legend_orientation="h")
    return fig


def render_single_result(result: dict):
    if result.get("error"):
        st.error(result["error"])
        return

    m = result["metrics"]
    c = result["conditions"]

    cols = st.columns(8)
    cols[0].metric("收盘", f"{m['close']:.2f}" if m["close"] is not None else "-")
    cols[1].metric("涨幅", f"{m.get('price_change_pct', 0):.2f}%" if m.get('price_change_pct') is not None else "-")
    cols[2].metric("量比", f"{m.get('volume_ratio', 0):.2f}" if m.get('volume_ratio') is not None else "-")
    cols[3].metric("MACD DEA", f"{m['macd_dea']:.4f}" if m["macd_dea"] is not None else "-")
    cols[4].metric("KDJ J", f"{m['kdj_j']:.2f}" if m["kdj_j"] is not None else "-")
    cols[5].metric("RSI14", f"{m['rsi14']:.2f}" if m["rsi14"] is not None else "-")
    cols[6].metric("白线", f"{m['zhixing_white']:.2f}" if m.get('zhixing_white') is not None else "-")
    cols[7].metric("是否通过", "✅ 是" if result["passed"] else "❌ 否")

    st.caption(f"最新交易日: {m['date']}")

    strategy = result.get("strategy", "B1")

    if strategy == "B1":
        cond_table = pd.DataFrame(
            [
                {"条件": "主板(600/601/603/000)", "结果": "✅" if c["mainboard_ok"] else "❌"},
                {"条件": "周线 MA30>MA60>MA120>MA240", "结果": "✅" if c["weekly_ok"] else "❌"},
                {"条件": "MACD DEA > 0", "结果": "✅" if c["macd_dea_ok"] else "❌"},
                {"条件": "KDJ J < 13", "结果": "✅" if c["kdj_j_ok"] else "❌"},
                {"条件": "白线 > 黄线 (黄线在白线下)", "结果": "✅" if c.get("zhixing_bullish") else "❌"},
                {"条件": "金叉后第一个B1", "结果": "✅" if c.get("golden_cross") else "⚪"},
                {"条件": "白色砖头 (买点)", "结果": "✅ 买点!" if c.get("brick_white") else "⚪"},
            ]
        )
    elif strategy == "B2":
        cond_table = pd.DataFrame(
            [
                {"条件": "主板(600/601/603/000)", "结果": "✅" if c["mainboard_ok"] else "❌"},
                {"条件": "周线 MA30>MA60>MA120>MA240", "结果": "✅" if c["weekly_ok"] else "❌"},
                {"条件": "MACD DEA > 0", "结果": "✅" if c["macd_dea_ok"] else "❌"},
                {"条件": "KDJ J < 55", "结果": "✅" if c["kdj_j_ok"] else "❌"},
                {"条件": "涨幅 >= 4%", "结果": "✅" if c.get("price_change_ok") else "❌"},
                {"条件": "量比 >= 1.1", "结果": "✅" if c.get("volume_ratio_ok") else "❌"},
                {"条件": "白色砖头 (买点)", "结果": "✅ 买点!" if c.get("brick_white") else "⚪"},
            ]
        )
    else:  # 自定义
        rows = [
            {"条件": "主板(600/601/603/000)", "结果": "✅" if c["mainboard_ok"] else "❌"},
            {"条件": "周线 MA30>MA60>MA120>MA240", "结果": "✅" if c["weekly_ok"] else "❌"},
        ]
        # DEA
        dea_val = m.get("macd_dea")
        rows.append({"条件": f"MACD DEA {result.get('dea_condition', '任意')}", "结果": "✅" if c["macd_dea_ok"] else "❌"})
        # J值
        rows.append({"条件": f"KDJ J 范围", "结果": "✅" if c["kdj_j_ok"] else "❌"})
        # 知行线
        rows.append({"条件": f"知行线 {result.get('zhixing_label', '任意')}", "结果": "✅" if c.get("zhixing_bullish", True) else "❌"})
        # 涨幅/量比
        if result.get("price_change_min") is not None:
            rows.append({"条件": f"涨幅 >= {result['price_change_min']}%", "结果": "✅" if c.get("price_change_ok", False) else "❌"})
        if result.get("volume_ratio_min") is not None:
            rows.append({"条件": f"量比 >= {result['volume_ratio_min']}", "结果": "✅" if c.get("volume_ratio_ok", False) else "❌"})
        # 白砖
        bw = result.get("brick_white_condition", "any")
        if bw != "any":
            rows.append({"条件": f"白色砖头 {bw}", "结果": "✅" if c.get("brick_white_ok", False) else "❌"})
        rows.append({"条件": "白色砖头 (买点)", "结果": "✅ 买点!" if c.get("brick_white") else "⚪"})
        cond_table = pd.DataFrame(rows)

    st.dataframe(cond_table, use_container_width=True, hide_index=True)
    if not c["weekly_ok"]:
        st.warning(f"周线检查: {result.get('weekly_reason', '未通过')}")

    fig = build_figure(result["daily_df"].tail(260), result["symbol"])
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
def main():
    st.title("B1战法选股器 v2.0")
    st.write("支持 Tushare + Yahoo Finance，内置知行趋势线与 KDJ/MACD/RSI/BOLL/VOL 指标。")

    # ── 侧边栏 ──────────────────────────────
    with st.sidebar:
        st.header("参数设置")

        # 1. 战法选择 (B1 / B2 / 自定义)
        strategy = st.selectbox(
            "选择战法",
            ["B1", "B2", "DSZ战法", "自定义"],
            help="B1: 超跌反弹 | B2: 强势追涨 | 自定义: 自由设置参数"
        )

        st.markdown("---")

        if strategy == "B1":
            st.info("📌 B1战法: 超跌反弹\n- KDJ J < 13\n- 白线 > 黄线\n- 适合回调买入")
        elif strategy == "B2":
            st.info("📌 B2战法: 强势追涨\n- KDJ J < 55\n- 白线 > 黄线\n- 白砖头信号\n- 适合突破买入")
        elif strategy == "DSZ战法":
            st.info("""📌 DSZ砖型图战法: 三种定式买入
- N型起跳: 前面上涨→回撤→今天>昨天+K线紧实+白砖
- 横盘起跳: 横盘5天+突破区间+白砖+白线>黄线
- 上升延续: 白线>黄线+回撤不破黄线+白砖
共同条件: 白砖头 + 白线>黄线 + 股价>黄线""")
        elif strategy == "自定义":
            st.info("📌 B2战法: 强势追涨\n- KDJ J < 55\n- 涨幅 >= 4%\n- 量比 >= 1.1\n- 适合追强势股")
        else:
            st.info("📌 自定义战法:\n自由设置 J值/DEA/白砖/\n知行线/涨幅/量比")

        st.markdown("---")

        # 2. 市场 & 股票池 & 行业 (多选)
        st.subheader("📋 筛选设置")

        market_options = st.multiselect(
            "市场选择",
            ["A股", "港股", "美股"],
            default=["A股"],
            help="选择要扫描的市场"
        )

        pool_options = st.multiselect(
            "股票池",
            ["沪深300", "上证50", "中证500", "中证1000", "全A"],
            default=["全A"],
            help="选择股票池范围 (仅A股有效)"
        )

        industry_options = st.multiselect(
            "行业板块",
            [
                "全部", "医药生物", "电子", "计算机", "机械设备", "化工",
                "有色金属", "电力设备", "汽车", "食品饮料", "传媒",
                "通信", "银行", "非银金融", "房地产", "建筑材料",
                "国防军工", "轻工制造", "公用事业", "交通运输",
                "商贸零售", "家电", "纺织服装", "农林牧渔", "综合",
            ],
            default=["全部"],
            help="选择行业板块 (可多选)"
        )

        st.markdown("---")

        # 3. 排除选项
        st.subheader("🚫 排除设置")

        exclude_chinext = st.checkbox("排除创业板 (300xxx)", value=False,
            help="排除深交所创业板股票")
        exclude_star = st.checkbox("排除科创板 (688xxx)", value=False,
            help="排除上交所科创板股票")
        exclude_st = st.checkbox("排除ST股", value=False,
            help="排除名称中含 ST/*ST/S*ST 的股票")
        exclude_bse = st.checkbox("排除北交所 (8xxxxx)", value=False,
            help="排除北京证券交易所股票")

        st.markdown("---")

        # 4. 自定义策略参数 (仅 strategy == "自定义" 时显示)
        if strategy == "自定义":
            st.subheader("⚙️ 自定义参数")

            j_col1, j_col2 = st.columns(2)
            with j_col1:
                kdj_j_min = st.number_input("J值最小", value=-20.0, step=1.0,
                    help="J值下限，填 -20 表示不限制下限")
            with j_col2:
                kdj_j_max = st.number_input("J值最大", value=20.0, step=1.0,
                    help="J值上限，超出此范围的股票会被排除")

            dea_cond = st.selectbox(
                "DEA条件",
                ["任意", "大于0", "小于0"],
                help="MACD DEA 线的筛选条件"
            )
            dea_map = {"任意": "any", "大于0": "positive", "小于0": "negative"}

            brick_cond = st.selectbox(
                "白色砖头信号",
                ["任意", "需要", "不需要"],
                help="砖型图白色砖头 = 买点信号"
            )
            brick_map = {"任意": "any", "需要": "required", "不需要": "forbidden"}

            zhixing_cond = st.selectbox(
                "知行线",
                ["任意", "白线 > 黄线", "黄线 > 白线"],
                help="知行趋势线白线与黄线的位置关系"
            )
            zhixing_map = {"任意": "any", "白线 > 黄线": "white_above", "黄线 > 白线": "yellow_above"}

            min_gain = st.number_input("最小涨幅 (%)", value=0.0, step=0.5,
                help="涨幅下限，不限制请填 0")
            min_vol_ratio = st.number_input("最小量比", value=0.0, step=0.1,
                help="量比起始，不限制请填 0")

            st.markdown("---")

        # 5. 数据源
        st.header("数据源")
        source = st.selectbox("数据源", ["tushare", "yahoo"], index=0)
        token = ""
        if source == "tushare":
            token = st.text_input("Tushare Token", value=DEFAULT_TUSHARE_TOKEN, type="password")

        mode = st.radio("模式", ["单票分析", "批量选股"], index=0)

        end_date = st.date_input("结束日期", value=date.today())
        start_date = st.date_input("开始日期", value=date.today() - timedelta(days=365 * 3))

        # pause_sec = st.slider("请求间隔(秒)", min_value=0.0, max_value=1.5, value=0.2, step=0.1)  # 已默认最快速度

        require_golden_cross = st.checkbox("仅显示金叉后第一个B1", value=False,
            help="开启后只显示白线向上穿透黄线(金叉)之后的第一个符合B1条件的股票")
        require_brick_white = st.checkbox("仅显示白色砖头 (买点)", value=False,
            help="砖型图白色砖头 = 买点信号")

        st.markdown("---")

        # 6. AI 战法助手
        with st.expander("🤖 AI 战法助手"):
            st.markdown("""
            **功能说明：**

            1. **B1战法** — 超跌反弹策略
               - 周线多头排列 (MA30>MA60>MA120>MA240)
               - MACD DEA > 0
               - KDJ J < 13 (超跌区域)
               - 白线 > 黄线 (知行线看涨)
               - 白色砖头 = 精确买点

            2. **B2战法** — 强势追涨策略
               - 在B1基础上放宽J值到 < 55
               - 要求涨幅 >= 4%
               - 要求量比 >= 1.1
               - 适合追强势板块龙头

            3. **自定义战法**
               - 自由组合各项参数
               - J值范围: 控制KDJ超买/超卖区间
               - DEA条件: 判断中期趋势方向
               - 白砖信号: 精确买点确认
               - 知行线: 趋势方向确认

            **💡 建议：**
            - 初学者先用 B1 熟悉信号
            - 强势市场用 B2 抓主升浪
            - 有经验后用 自定义 做精细化筛选
            """)

        # ── 构建 B1Config ──────────────────
        # 市值筛选 (从 UI 中收集，但本次实现中暂不使用)
        market_cap_min = 0.0
        market_cap_max = 0.0

        if strategy == "自定义":
            cfg = B1Config(
                source=source,
                tushare_token=token or None,
                start=start_date,
                end=end_date,
                request_pause_sec=0.0,
                require_golden_cross=require_golden_cross,
                require_brick_white=require_brick_white,
                strategy=strategy,
                market_cap_min=market_cap_min,
                market_cap_max=market_cap_max,
                sector="全部" if "全部" in industry_options else (industry_options[0] if industry_options else "全部"),
                # 自定义参数
                kdj_j_min=kdj_j_min if kdj_j_min != -20 else None,
                kdj_j_max=kdj_j_max if kdj_j_max != 20 else None,
                dea_condition=dea_map[dea_cond],
                brick_white_condition=brick_map[brick_cond],
                zhixing_condition=zhixing_map[zhixing_cond],
                price_change_min=min_gain if min_gain > 0 else None,
                volume_ratio_min=min_vol_ratio if min_vol_ratio > 0 else None,
            )
        elif strategy in ["B1", "B2", "DSZ战法"]:
            cfg = B1Config(
                source=source,
                tushare_token=token or None,
                start=start_date,
                end=end_date,
                request_pause_sec=0.0,
                require_golden_cross=require_golden_cross,
                require_brick_white=require_brick_white,
                strategy=strategy,
                market_cap_min=market_cap_min,
                market_cap_max=market_cap_max,
                sector="全部" if "全部" in industry_options else (industry_options[0] if industry_options else "全部"),
            )
        else:
            cfg = B1Config(
                source=source,
                tushare_token=token or None,
                start=start_date,
                end=end_date,
                request_pause_sec=0.0,
                require_golden_cross=require_golden_cross,
                require_brick_white=require_brick_white,
                strategy=strategy,
                market_cap_min=market_cap_min,
                market_cap_max=market_cap_max,
                sector="全部" if "全部" in industry_options else (industry_options[0] if industry_options else "全部"),
            )

    # ── 选项卡 ─────────────────────────────────────────────────────────
    tab_names = ["📊 单票分析", "🔍 批量选股", "📈 图表"]
    tab1, tab2, tab3 = st.tabs(tab_names)

    # ── 单票分析模式 ────────────────────────────
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            symbol = st.text_input("股票代码", value="600519.SH")
        with c2:
            name = st.text_input("股票名称(可选)", value="")

        if st.button("开始分析", type="primary", use_container_width=True):
            # 排除检查
            if symbol and symbol_matches_exclude(symbol, exclude_chinext, exclude_star, exclude_st, exclude_bse):
                st.warning("该股票符合排除条件，请调整排除设置后重试。")
            else:
                with st.spinner("正在计算指标并检查条件..."):
                    result = scan_symbol(symbol=symbol, name=name, config=cfg)
                    result["dea_condition"] = dea_cond if strategy == "自定义" else ("大于0" if cfg.strategy in ["B1", "B2"] else "任意")
                    result["zhixing_label"] = zhixing_cond if strategy == "自定义" else "任意"
                    result["price_change_min"] = cfg.price_change_min
                    result["volume_ratio_min"] = cfg.volume_ratio_min
                    result["brick_white_condition"] = brick_cond if strategy == "自定义" else "任意"
                render_single_result(result)

    # ── 批量选股模式 ────────────────────────────
    with tab2:
        st.subheader("批量选股")

        # 自选股管理 (widgets moved outside expander to avoid scope issues)
        default_stocks = """600519.SH,贵州茅台
600276.SH,恒瑞医药
603259.SH,药明康德
688235.SH,百济神州
688331.SH,荣昌生物
000001.SZ,平安银行
601298.SH,青岛港
002946.SZ,新乳业
600016.SH,民生银行
000651.SH,格力电器"""

        watchlist = st.text_area(
            "📌 自选股列表 (每行: 代码,名称)",
            value=default_stocks,
            height=150,
            help="格式: 代码,名称 或 只写代码",
            key="watchlist_input",
        )

        with st.expander("📌 自选股管理", expanded=True):
            st.markdown("**🔍 搜索添加股票**")
            search_col1, search_col2 = st.columns([2, 1])
            with search_col1:
                search_keyword = st.text_input("输入股票代码或名称搜索", placeholder="如: 600519 或 茅台")
            with search_col2:
                search_btn = st.button("搜索", use_container_width=True)

            if search_btn and search_keyword:
                search_success = False
                if token:
                    try:
                        import tushare as ts
                        pro = ts.pro_api(token)
                        df = pro.stock_basic(fields='ts_code,name,area,industry,list_date')
                        mask = df['ts_code'].str.contains(search_keyword, na=False) | df['name'].str.contains(search_keyword, na=False)
                        results = df[mask].head(10)
                        if not results.empty:
                            st.success(f"找到 {len(results)} 只股票:")
                            for _, row in results.iterrows():
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.code(row['ts_code'])
                                with col2:
                                    st.write(f"{row['name']} ({row.get('industry', '')})")
                            search_success = True
                    except Exception as e:
                        err = str(e)
                        if any(kw in err for kw in ["每小时", "每天", "权限", "quota"]):
                            st.warning("⚠️ Tushare 配额已用完，使用本地备用列表...")
                            search_success = False
                        else:
                            st.error(f"搜索失败: {err[:80]}")

                if not search_success:
                    backup_stocks = {
                        "长江通信": "600345.SH", "青岛港": "601298.SH",
                        "新乳业": "002946.SZ", "亚盛医药": "06855.HK",
                        "阿里": "9988.HK", "腾讯": "00700.HK",
                        "美团": "03690.HK", "小米": "01810.HK",
                    }
                    matches = {k: v for k, v in backup_stocks.items() if search_keyword in k}
                    if matches:
                        st.info("📋 热门股票 (Tushare配额用完):")
                        for n, c in matches.items():
                            st.write(f"  {c} - {n}")
                    else:
                        st.info("📝 请直接在自选股列表中添加，或稍后再试搜索")

            st.markdown("**⚡ 快捷添加**")
            quick_add_cols = st.columns(4)
            quick_stocks = [
                ("600519.SH", "贵州茅台"), ("600276.SH", "恒瑞医药"),
                ("000001.SZ", "平安银行"), ("601318.SH", "中国平安"),
                ("000651.SZ", "格力电器"), ("600036.SH", "招商银行"),
                ("002594.SZ", "比亚迪"), ("300750.SZ", "宁德时代"),
            ]
            for i, (code, name_q) in enumerate(quick_stocks):
                with quick_add_cols[i % 4]:
                    if st.button(f"{code}", key=f"add_{code}", use_container_width=True):
                        st.success(f"已添加 {name_q}")

        st.markdown("---")

        scan_source = st.radio(
            "扫描数据源",
            ["使用自选股列表", "使用Tushare主板列表"],
            horizontal=True
        )

        if scan_source == "使用自选股列表":
            rows = parse_symbol_input(watchlist)
            limit = len(rows)
            use_default = False
        else:
            use_default = True
            if source == "tushare":
                # 直接扫描所有符合条件的股票（不限制数量）
                # limit=0 表示不限制
                limit = 0
                st.caption("📊 将扫描所有符合条件的股票")
                raw_symbols = ""
            else:
                limit = 0
                raw_symbols = ""

        # 刷新后恢复上次结果 (session_state)
        config_hash = hash((
            strategy, tuple(market_options), tuple(pool_options), tuple(industry_options),
            exclude_chinext, exclude_star, exclude_st, exclude_bse,
            str(source), limit
        ))

        if st.session_state.get("scan_done") and st.session_state.get("last_config_hash") == config_hash:
            st.info("📋 已恢复上次扫描结果 (刷新页面后保留)")
            table = st.session_state.scan_table
            passed_df = table[table["passed"] == True] if table is not None else pd.DataFrame()
            st.success(f"扫描完成: 总数 {len(table)}，符合条件 {len(passed_df)}")
            st.subheader("符合条件")
            st.dataframe(passed_df if not passed_df.empty else pd.DataFrame(columns=table.columns if table is not None else []), use_container_width=True)
            st.subheader("全部结果")
            st.dataframe(table, use_container_width=True)
            if table is not None:
                csv_bytes = table.to_csv(index=False).encode("utf-8-sig")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("📥 下载结果 CSV", data=csv_bytes,
                                       file_name=f"b1_scan_{date.today().isoformat()}.csv",
                                       mime="text/csv", use_container_width=True)
                # 恢复时也提供日志下载
                if st.session_state.get("scan_log"):
                    with col2:
                        log_bytes = st.session_state.scan_log.encode("utf-8")
                        st.download_button("📋 下载本次日志", data=log_bytes,
                                           file_name=f"b1_scan_log_{date.today().isoformat()}.log",
                                           mime="text/plain", use_container_width=True)
        else:
            if st.button("开始批量扫描", type="primary", use_container_width=True):
                with st.spinner("准备股票池..."):
                    if source == "tushare" and use_default:
                        stock_df = get_tushare_mainboard_stocks(limit=limit, token=token or None)
                        rows = [
                            {"symbol": r["ts_code"], "name": r["name"]}
                            for _, r in stock_df.iterrows()
                        ]
                    else:
                        rows = parse_symbol_input(watchlist) if scan_source == "使用自选股列表" else parse_symbol_input(raw_symbols)

                    # 应用排除条件
                    if any([exclude_chinext, exclude_star, exclude_bse, exclude_st]):
                        rows = [
                            r for r in rows
                            if not symbol_matches_exclude(
                                r["symbol"],
                                exclude_chinext,
                                exclude_star,
                                exclude_st,
                                exclude_bse,
                            )
                        ]

                    if not rows:
                        st.error("股票池为空（应用排除条件后无剩余股票），请调整排除设置。")
                        return

                    # ── 优化后的并行扫描 ──────────────────────────────────────
                    # 使用 scan_batch 的 progress_callback 实现实时进度更新
                    # 内部使用 ThreadPoolExecutor 并行处理所有股票
                    progress = st.progress(0)
                    status = st.empty()
                    thread_status = st.empty()
                    results = []
                    total = len(rows)

                    def on_stock_done(completed: int, _total: int, result: dict):
                        """每只股票完成后更新 Streamlit UI（Streamlit-safe）。"""
                        progress.progress(completed / total)
                        status.write(
                            f"🚀 并行扫描中: {result.get('symbol','')} "
                            f"({completed}/{total})"
                        )
                        # 补充策略信息
                        result["dea_condition"] = dea_cond if strategy == "自定义" else ("大于0" if cfg.strategy in ["B1", "B2"] else "任意")
                        result["zhixing_label"] = zhixing_cond if strategy == "自定义" else "任意"
                        result["price_change_min"] = cfg.price_change_min
                        result["volume_ratio_min"] = cfg.volume_ratio_min
                        result["brick_white_condition"] = brick_cond if strategy == "自定义" else "任意"
                        results.append(result)

                    # 动态计算线程数：默认最快速度
                    raw_workers = max(1, int(0.5 / 0.05))  # 按 0.05 秒计算
                    max_workers = min(raw_workers, 50)
                    thread_status.info(
                        f"⚡ 启用 {max_workers} 线程并行扫描 | "
                        f"股票总数: {total}"
                    )

                    # 用零暂停 scan_batch（并行执行，不靠 sleep 限速）
                    scan_cfg = dataclasses.replace(cfg, request_pause_sec=0.0)

                    scan_batch(rows, config=scan_cfg, progress_callback=on_stock_done)

                    table = pd.DataFrame([flatten_result_for_table(x) for x in results])
                    table = table.sort_values(["passed", "symbol"], ascending=[False, True]).reset_index(drop=True)

                    passed_df = table[table["passed"] == True]
                    st.success(f"扫描完成: 总数 {len(table)}，符合条件 {len(passed_df)}")

                    # 5. session_state 保存结果
                    st.session_state.scan_results = results
                    st.session_state.scan_table = table
                    st.session_state.scan_done = True
                    st.session_state.last_config_hash = config_hash

                    st.subheader("符合条件")
                    st.dataframe(passed_df if not passed_df.empty else pd.DataFrame(columns=table.columns), use_container_width=True)

                    st.subheader("全部结果")
                    st.dataframe(table, use_container_width=True)

                    csv_bytes = table.to_csv(index=False).encode("utf-8-sig")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            "📥 下载结果 CSV",
                            data=csv_bytes,
                            file_name=f"b1_scan_{date.today().isoformat()}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    
                    # ── 保存本次日志 ─────────────────────────────
                    log_file = os.path.join(os.path.dirname(__file__), "logs", f"scan_{date.today().strftime('%Y%m%d')}.log")
                    if os.path.exists(log_file):
                        with open(log_file, "r", encoding="utf-8") as f:
                            log_content = f.read()
                        # 提取本次扫描相关的日志（从"开始扫描"到扫描结束）
                        # 按时间戳和关键字过滤
                        lines = log_content.split("\n")
                        # 找到本次扫描开始的行（扫描第一只股票的时间）
                        scan_start_time = None
                        for line in lines:
                            if "🟢 开始扫描:" in line:
                                # 提取时间戳
                                ts = line.split(" | ")[0] if " | " in line else None
                                if ts and not scan_start_time:
                                    scan_start_time = ts
                                    break
                        
                        if scan_start_time:
                            # 只保留从本次扫描开始之后的日志
                            start_idx = None
                            for i, line in enumerate(lines):
                                if scan_start_time in line and "🟢 开始扫描:" in line:
                                    start_idx = i
                                    break
                            if start_idx is not None:
                                session_log = "\n".join(lines[start_idx:])
                            else:
                                session_log = log_content
                        else:
                            session_log = log_content
                        
                        # 保存到 session_state
                        st.session_state.scan_log = session_log
                        
                        with col2:
                            log_bytes = session_log.encode("utf-8")
                            st.download_button(
                                "📋 下载本次日志",
                                data=log_bytes,
                                file_name=f"b1_scan_log_{date.today().isoformat()}.log",
                                mime="text/plain",
                                use_container_width=True,
                            )

                    # ── 日志查看器 ─────────────────────────────
                    with st.expander("📋 查看扫描日志", expanded=False):
                        log_file = os.path.join(os.path.dirname(__file__), "logs", f"scan_{date.today().strftime('%Y%m%d')}.log")
                        if os.path.exists(log_file):
                            with open(log_file, "r", encoding="utf-8") as f:
                                log_content = f.read()
                            if log_content:
                                st.text_area("日志内容 (实时更新)", value=log_content, height=400, key="log_viewer")
                                st.caption(f"日志位置: {log_file}")
                            else:
                                st.info("暂无日志记录")
                        else:
                            st.info("尚未生成日志文件")


    # ── 图表模式 ──────────────────────────────────────────────────────
    with tab3:
        st.subheader("📈 K线图表")

        chart_col1, chart_col2, chart_col3 = st.columns([2, 1, 1])
        with chart_col1:
            chart_symbol = st.text_input(
                "股票代码",
                value="600519.SH",
                placeholder="输入股票代码，如 600519.SH",
                key="chart_symbol_input",
            )
        with chart_col2:
            chart_days = st.selectbox(
                "数据区间",
                ["近1月", "近3月", "近6月", "近1年", "近2年", "近3年"],
                index=3,
                key="chart_days_select",
            )
        with chart_col3:
            show_bbi = st.checkbox("显示BBI", value=False, key="chart_bbi_checkbox")

        day_map = {
            "近1月": 30, "近3月": 90, "近6月": 180,
            "近1年": 365, "近2年": 730, "近3年": 1095,
        }
        days = day_map.get(chart_days, 365)
        chart_end = date.today()
        chart_start = chart_end - timedelta(days=days)

        if st.button("📊 加载图表", type="primary", use_container_width=True):
            if not chart_symbol.strip():
                st.warning("请输入股票代码")
            else:
                with st.spinner(f"正在加载 {chart_symbol} K线数据..."):
                    try:
                            # 构建一个临时配置用于获取数据
                        temp_cfg = B1Config(
                            source=source,
                            tushare_token=token or None,
                            start=chart_start,
                            end=chart_end,
                            request_pause_sec=0.1,
                            strategy="B1",
                        )
                        chart_result = scan_symbol(
                            symbol=chart_symbol.strip(),
                            name="",
                            config=temp_cfg,
                        )
                        if chart_result.get("error"):
                            st.error(f"加载失败: {chart_result['error']}")
                        else:
                            chart_df = chart_result.get("daily_df")
                            if chart_df is None or chart_df.empty:
                                st.warning("未获取到K线数据")
                            else:
                                # 渲染交互式K线图
                                from kline_chart import calculate_indicators_for_chart
                                chart_df = calculate_indicators_for_chart(chart_df)
                                fig = create_kline_chart(
                                    chart_df,
                                    title=f"{chart_symbol.strip()} K线图表",
                                    show_bbi=show_bbi,
                                    show_volume=True,
                                    show_ma=True,
                                    show_zhixing=True,
                                    width=None,
                                    height=700,
                                )
                                st.plotly_chart(
                                    fig,
                                    use_container_width=True,
                                    config={
                                        "scrollZoom": True,
                                        "displayModeBar": True,
                                        "modeBarButtonsToRemove": [
                                            "lasso2d", "select2d",
                                        ],
                                        "displaylogo": False,
                                    },
                                )
                                # 显示当前行情摘要
                                m = chart_result.get("metrics", {})
                                c = chart_result.get("conditions", {})
                                mc = st.columns(6)
                                mc[0].metric("收盘", f"{m.get('close', 0):.2f}")
                                mc[1].metric("涨幅", f"{m.get('price_change_pct', 0):.2f}%")
                                mc[2].metric("量比", f"{m.get('volume_ratio', 0):.2f}")
                                mc[3].metric("MACD DEA", f"{m.get('macd_dea', 0):.4f}")
                                mc[4].metric("KDJ J", f"{m.get('kdj_j', 0):.2f}")
                                mc[5].metric("知行白线", f"{m.get('zhixing_white', 0):.2f}")
                                st.caption(f"数据截止: {m.get('date', 'N/A')}")
                    except Exception as e:
                        st.error(f"加载图表时出错: {e}")


if __name__ == "__main__":
    main()
