#!/usr/bin/env python3
"""
Bilingual Support Module for B1 Scanner
添加双语支持 (English + Chinese)
"""

# 翻译字典
TEXTS = {
    "zh": {
        # 主标题
        "title": "B1战法选股器 v2.0",
        "subtitle": "支持 Tushare + Yahoo Finance，内置知行趋势线与 KDJ/MACD/RSI/BOLL/VOL 指标。",
        
        # 侧边栏
        "settings": "参数设置",
        "strategy": "选择战法",
        "strategy_b1": "B1: 超跌反弹 - KDJ J < 13, 白线 > 黄线, 适合回调买入",
        "strategy_b2": "B2: 强势追涨 - KDJ J < 55, 涨幅 >= 4%, 量比 >= 1.1, 适合追强势股",
        "data_source": "数据源",
        "mode": "模式",
        "mode_single": "单票分析",
        "mode_batch": "批量选股",
        "end_date": "结束日期",
        "start_date": "开始日期",
        "pause_sec": "请求间隔(秒)",
        "sector": "行业板块 (可选)",
        "sector_all": "全部",
        "market_cap": "市值筛选 (亿元)",
        "min_cap": "最小市值",
        "max_cap": "最大市值",
        
        # 按钮
        "analyze": "开始分析",
        "batch_scan": "开始批量扫描",
        "download_csv": "下载结果 CSV",
        
        # 标签页
        "tab_analysis": "📈 股票分析",
        "tab_batch": "🔍 批量选股",
        "tab_viz": "📊 可视化分析",
        
        # 可视化
        "viz_scatter": "散点图",
        "viz_candle": "K线图",
        "select_stock": "选择股票",
        "time_range": "时间范围",
        "show_ma": "显示MA",
        "show_volume": "显示成交量",
        "x_axis": "X轴",
        "y_axis": "Y轴",
        "color_by": "颜色编码",
        
        # 结果
        "analysis_result": "分析结果",
        "passed": "符合B1条件",
        "failed": "不符合B1条件",
        "scan_complete": "扫描完成",
        "total_stocks": "总数",
        "passed_count": "符合B1",
        
        # 指标名称
        "kline": "K线",
        "vol": "成交量",
        "macd": "MACD",
        "kdj": "KDJ",
        "rsi": "RSI",
        "boll": "布林带",
        "brick_chart": "砖型图",
        "white_brick": "白色砖头 (买点)",
        
        # 错误
        "error_empty": "股票池为空，请输入至少一只股票。",
        "error_fetch": "获取数据失败",
    },
    "en": {
        # Main title
        "title": "B1 Strategy Stock Scanner v2.0",
        "subtitle": "Support Tushare + Yahoo Finance, built-in Zhixing Trend Line & KDJ/MACD/RSI/BOLL/VOL indicators.",
        
        # Sidebar
        "settings": "Settings",
        "strategy": "Strategy",
        "strategy_b1": "B1: Oversold Rebound - KDJ J < 13, White > Yellow, Buy on pullback",
        "strategy_b2": "B2: Strong Trend - KDJ J < 55, Change >= 4%, Volume ratio >= 1.1, Chase strength",
        "data_source": "Data Source",
        "mode": "Mode",
        "mode_single": "Single Stock",
        "mode_batch": "Batch Scan",
        "end_date": "End Date",
        "start_date": "Start Date",
        "pause_sec": "Request Interval (sec)",
        "sector": "Sector (Optional)",
        "sector_all": "All",
        "market_cap": "Market Cap (100M CNY)",
        "min_cap": "Min Cap",
        "max_cap": "Max Cap",
        
        # Buttons
        "analyze": "Analyze",
        "batch_scan": "Start Batch Scan",
        "download_csv": "Download CSV",
        
        # Tabs
        "tab_analysis": "📈 Stock Analysis",
        "tab_batch": "🔍 Batch Scan",
        "tab_viz": "📊 Visualization",
        
        # Visualization
        "viz_scatter": "Scatter Plot",
        "viz_candle": "Candlestick",
        "select_stock": "Select Stock",
        "time_range": "Time Range",
        "show_ma": "Show MA",
        "show_volume": "Show Volume",
        "x_axis": "X Axis",
        "y_axis": "Y Axis",
        "color_by": "Color By",
        
        # Results
        "analysis_result": "Analysis Result",
        "passed": "Passed B1",
        "failed": "Failed B1",
        "scan_complete": "Scan Complete",
        "total_stocks": "Total",
        "passed_count": "Passed",
        
        # Indicators
        "kline": "Candlestick",
        "vol": "Volume",
        "macd": "MACD",
        "kdj": "KDJ",
        "rsi": "RSI",
        "boll": "BOLL",
        "brick_chart": "Brick Chart",
        "white_brick": "White Brick (Buy Signal)",
        
        # Errors
        "error_empty": "Stock pool is empty. Please enter at least one stock.",
        "error_fetch": "Failed to fetch data",
    }
}

def t(key, lang="zh"):
    """Get translated text"""
    return TEXTS.get(lang, TEXTS["zh"]).get(key, key)


def get_language_options():
    """Get language radio options"""
    return [("中文", "zh"), ("English", "en")]
