"""Technical indicators used in B1 scanner v2.0.
砖型图完整实现 - 完全对齐通达信原版公式
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_kdj(df: pd.DataFrame, n: int = 9, k_smooth: int = 3, d_smooth: int = 3) -> pd.DataFrame:
    """KDJ指标 (9,3,3)"""
    low_n = df["low"].rolling(window=n, min_periods=n).min()
    high_n = df["high"].rolling(window=n, min_periods=n).max()

    denom = (high_n - low_n).replace(0, np.nan)
    rsv = ((df["close"] - low_n) / denom * 100).fillna(50)

    k = rsv.ewm(alpha=1 / k_smooth, adjust=False).mean()
    d = k.ewm(alpha=1 / d_smooth, adjust=False).mean()
    j = 3 * k - 2 * d

    return pd.DataFrame({"kdj_k": k, "kdj_d": d, "kdj_j": j})


def calc_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD指标 (12,26,9)"""
    ema_fast = ema(df["close"], fast)
    ema_slow = ema(df["close"], slow)

    diff = ema_fast - ema_slow
    dea = diff.ewm(span=signal, adjust=False).mean()
    hist = (diff - dea) * 2

    return pd.DataFrame({"macd_diff": diff, "macd_dea": dea, "macd_hist": hist})


def calc_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI指标"""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_boll(df: pd.DataFrame, period: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """布林带指标"""
    mid = ma(df["close"], period)
    std = df["close"].rolling(window=period, min_periods=period).std(ddof=0)

    upper = mid + n_std * std
    lower = mid - n_std * std

    return pd.DataFrame({"boll_mid": mid, "boll_upper": upper, "boll_lower": lower})


def calc_vol(df: pd.DataFrame) -> pd.DataFrame:
    """成交量均线"""
    return pd.DataFrame(
        {
            "vol_ma5": df["vol"].rolling(5, min_periods=5).mean(),
            "vol_ma10": df["vol"].rolling(10, min_periods=10).mean(),
            "vol_ma20": df["vol"].rolling(20, min_periods=20).mean(),
        }
    )


def calc_zhixing_white(df: pd.DataFrame) -> pd.Series:
    """知行白线(短期趋势): EMA(EMA(C,10),10)."""
    return ema(ema(df["close"], 10), 10)


def calc_zhixing_yellow(df: pd.DataFrame) -> pd.Series:
    """知行黄线(长期趋势): (MA14 + MA28 + MA57 + MA114) / 4."""
    return (
        ma(df["close"], 14)
        + ma(df["close"], 28)
        + ma(df["close"], 57)
        + ma(df["close"], 114)
    ) / 4


def calc_zhixing_trend(df: pd.DataFrame) -> pd.DataFrame:
    """知行趋势线组件: 白线 + 黄线."""
    white = calc_zhixing_white(df)
    yellow = calc_zhixing_yellow(df)
    return pd.DataFrame({"zhixing_white": white, "zhixing_yellow": yellow})


def calc_brick_chart(df: pd.DataFrame) -> pd.DataFrame:
    """
    砖型图 (Brick Chart) - 通达信原版公式
    
    完整公式 (富途Mai语言版本):
    DEN := HHV(H,4) - LLV(L,4);
    VAR1A := IF(DEN=0, 0, (HHV(H,4)-C)/DEN*100 - 90);
    VAR2A := SMA(VAR1A,4,1) + 100;
    VAR3A := IF(DEN=0, 0, (C-LLV(L,4))/DEN*100);
    VAR4A := SMA(VAR3A,6,1);
    VAR5A := SMA(VAR4A,6,1) + 100;
    VAR6A := VAR5A - VAR2A;
    砖型图 := IF(VAR6A>4, VAR6A-4, 0);
    
    白砖信号:
    PRE := REF(砖型图,1);
    PRE2 := REF(砖型图,2);
    TODAY_RED := (PRE < 砖型图);
    YEST_GREEN := (PRE2 > PRE);
    RED_LEN := 砖型图 - PRE;
    GREEN_LEN := PRE2 - PRE;
    WHITE_SIG := TODAY_RED AND YEST_GREEN AND (RED_LEN > GREEN_LEN * 2 / 3);
    
    注意: 通达信 SMA(x, N, 1) 等效于 EMA(x, N)
    """
    
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    # ===== 基础计算 =====
    # DEN := HHV(H,4) - LLV(L,4)
    hhv_high_4 = high.rolling(window=4, min_periods=4).max()
    llv_low_4 = low.rolling(window=4, min_periods=4).min()
    den = hhv_high_4 - llv_low_4
    
    # ===== VAR1A (上半部分) =====
    # VAR1A := IF(DEN=0, 0, (HHV(H,4)-C)/DEN*100 - 90)
    var1a = np.where(den == 0, 0, (hhv_high_4 - close) / den * 100 - 90)
    var1a = pd.Series(var1a, index=df.index)
    
    # VAR2A := SMA(VAR1A,4,1) + 100
    # 注: 通达信 SMA(x, 4, 1) 等效于 EMA(x, 4)
    var2a = var1a.ewm(span=4, adjust=False).mean() + 100
    
    # ===== VAR3A (下半部分) =====
    # VAR3A := IF(DEN=0, 0, (C-LLV(L,4))/DEN*100)
    var3a = np.where(den == 0, 0, (close - llv_low_4) / den * 100)
    var3a = pd.Series(var3a, index=df.index)
    
    # VAR4A := SMA(VAR3A,6,1)
    var4a = var3a.ewm(span=6, adjust=False).mean()
    
    # VAR5A := SMA(VAR4A,6,1) + 100
    var5a = var4a.ewm(span=6, adjust=False).mean() + 100
    
    # ===== 砖型图合成 =====
    # VAR6A := VAR5A - VAR2A
    var6a = var5a - var2a
    
    # 砖型图 := IF(VAR6A>4, VAR6A-4, 0)
    brick = pd.Series(np.where(var6a > 4, var6a - 4, 0), index=df.index)
    
    # ===== 白砖信号计算 =====
    # PRE := REF(砖型图,1)  (昨天的砖型图)
    pre = brick.shift(1)
    
    # PRE2 := REF(砖型图,2)  (前天的砖型图)
    pre2 = brick.shift(2)
    
    # TODAY_RED := (PRE < 砖型图)  (今天红柱 - 上涨)
    today_red = brick > pre
    
    # YEST_GREEN := (PRE2 > PRE)  (昨天绿柱 - 下跌)
    yest_green = pre2 > pre
    
    # RED_LEN := 砖型图 - PRE  (红柱高度)
    red_len = brick - pre
    
    # GREEN_LEN := PRE2 - PRE  (绿柱高度)
    green_len = pre2 - pre
    
    # WHITE_SIG := TODAY_RED AND YEST_GREEN AND (RED_LEN > GREEN_LEN * 2/3)
    white_sig = today_red & yest_green & (red_len > green_len * 2/3)
    
    return pd.DataFrame({
        "brick_chart": brick,           # 砖型图数值
        "brick_white": white_sig.astype(int),   # 白砖信号 = 1 (买点)
        "brick_red": today_red.astype(int),     # 红砖 = 1 (持有)
        "brick_green": yest_green.astype(int),  # 绿砖 = 1 (做空)
        "brick_red_len": red_len,               # 红柱高度
        "brick_green_len": green_len,           # 绿柱高度
    })


def _is_consolidating(close: pd.Series, lookback: int = 5, threshold: float = 0.03) -> pd.Series:
    """判断最近N天是否横盘震荡 (日内波幅小)."""
    high = close.rolling(lookback).max()
    low = close.rolling(lookback).min()
    mid = (high + low) / 2
    return ((high - low) / mid) < threshold


def _is_consecutive_up(close: pd.Series, n: int) -> pd.Series:
    """最近N天是否连续上涨 (每天close > prev_close)."""
    up = close > close.shift(1)
    result = pd.Series(True, index=close.index)
    for _ in range(n - 1):
        result = result & up.shift(-_)
    return result.shift(-(n - 1))


def _is_consecutive_down(close: pd.Series, n: int) -> pd.Series:
    """最近N天是否连续回调 (每天close < prev_close)."""
    down = close < close.shift(1)
    result = pd.Series(True, index=close.index)
    for _ in range(n - 1):
        result = result & down.shift(-_)
    return result.shift(-(n - 1))


def _tight_candle(high: pd.Series, low: pd.Series, close: pd.Series,
                  upper_shadow_ratio: float = 0.3,
                  lower_shadow_ratio: float = 0.3,
                  body_ratio: float = 0.5) -> pd.Series:
    """
    K线形态紧实判断 (无长上下影，实体占主导).
    上影线比例 = (high - max(open,close)) / (high - low)
    下影线比例 = (min(open,close) - low) / (high - low)
    实体比例 = |close - open| / (high - low)
    """
    body = (close - close.shift(1)).abs()
    range_ = (high - low).replace(0, np.nan)

    upper_shadow = (high - pd.concat([close, close.shift(1)], axis=1).max(axis=1)).clip(lower=0)
    lower_shadow = (pd.concat([close, close.shift(1)], axis=1).min(axis=1) - low).clip(lower=0)

    upper_ratio = upper_shadow / range_
    lower_ratio = lower_shadow / range_
    body_r = body / range_

    return (upper_ratio < upper_shadow_ratio) & \
           (lower_ratio < lower_shadow_ratio) & \
           (body_r > body_ratio)


def _consolidation_breakout(close: pd.Series, lookback: int = 5) -> pd.Series:
    """
    横盘突破: 今天收盘价 > 昨天收盘价 AND 今天 > 过去5天最高价.
    """
    max_recent = close.shift(1).rolling(lookback).max()
    return (close > close.shift(1)) & (close > max_recent)


def _sideways_range(close: pd.Series, lookback: int = 5, width_pct: float = 0.10) -> pd.Series:
    """
    横盘区间: 过去N天振幅不超过width_pct%.
    """
    high = close.rolling(lookback).max()
    low = close.rolling(lookback).min()
    mid = (high + low) / 2
    return ((high - low) / mid) < width_pct


def detect_dsz_patterns_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    DSZ 砖型图战法 v2 (基于用户最新反馈).

    三种定式共同要求:
        - 白线 > 黄线 (知行线多头)
        - 股价 > 黄线  (价格突破完成)
        - 白砖头信号  (brick_white == 1)

    1. N型起跳 (N-formation Breakout):
        - 前面有上涨趋势 (最近5日有上涨)
        - 回撤整理     (连续回调K线)
        - 今天收盘价高于昨天 (不需要突破前高)
        - K线形态紧实  (无长上下影)
        - 成交量 >= 1.0 (平量)
        - 白砖信号 + 知行线多头 + 股价 > 黄线

    2. 横盘起跳 (Sideways Breakout):
        - 横盘5+天
        - 今天收盘价突破横盘区间
        - 成交量 >= 1.0 (不强制放量)
        - 白砖信号 + 知行线多头 + 股价 > 黄线

    3. 上升延续 (Uptrend Continuation):
        - 白线 > 黄线 (多头趋势)
        - 股价 > 黄线
        - 回撤不破黄线
        - 今天收盘价高于昨天 (企稳回升)
        - 白砖信号
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    white = df["zhixing_white"]
    yellow = df["zhixing_yellow"]
    vol_ma5 = df["vol_ma5"]
    brick_white = df["brick_white"].fillna(0).astype(bool)

    # ---- 共用条件 ----
    white_above_yellow = white > yellow
    price_above_yellow = close > yellow
    flat_volume = df["vol"] / vol_ma5 >= 1.0

    # ---- 1. N型起跳 ----
    # 前面有上涨趋势 (最近5日有上涨)
    recent_uptrend = _is_consecutive_up(close, n=5)
    # 回撤整理 (连续回调K线: 今天和昨天连续下跌)
    pullback = _is_consecutive_down(close, n=2)
    # 今天收盘价高于昨天 (起跳确认)
    today_higher = close > close.shift(1)
    # K线形态紧实
    tight = _tight_candle(high, low, close)

    n_pattern = (recent_uptrend | (close.diff(4) > 0)) & \
                pullback & \
                today_higher & \
                tight & \
                flat_volume & \
                brick_white & \
                white_above_yellow & \
                price_above_yellow

    # ---- 2. 横盘起跳 ----
    # 横盘5+天
    sideways = _sideways_range(close, lookback=5, width_pct=0.10)
    # 今天突破横盘区间
    breakout = _consolidation_breakout(close, lookback=5)

    sideways_pattern = sideways.shift(1) & \
                       breakout & \
                       flat_volume & \
                       brick_white & \
                       white_above_yellow & \
                       price_above_yellow

    # ---- 3. 上升延续 ----
    # 多头趋势中回撤不破黄线，然后今天企稳回升
    pullback_not_break_yellow = (close < white) & price_above_yellow
    recovery = today_higher

    uptrend_cont = white_above_yellow & \
                   price_above_yellow & \
                   pullback_not_break_yellow & \
                   recovery & \
                   brick_white

    return pd.DataFrame({
        "dsz_n_pattern":     n_pattern.astype(int),        # N型起跳
        "dsz_sideways":      sideways_pattern.astype(int),  # 横盘起跳
        "dsz_uptrend_cont":  uptrend_cont.astype(int),     # 上升延续
    })


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加所有技术指标"""
    out = df.copy()

    kdj = calc_kdj(out)
    macd = calc_macd(out)
    boll = calc_boll(out)
    vol = calc_vol(out)

    out = pd.concat([out, kdj, macd, boll, vol], axis=1)
    out["rsi14"] = calc_rsi(out, period=14)
    zhixing = calc_zhixing_trend(out)
    out = pd.concat([out, zhixing], axis=1)
    
    # 添加砖型图
    brick = calc_brick_chart(out)
    out = pd.concat([out, brick], axis=1)

    # 添加 DSZ 砖型图战法 v2
    dsz_v2 = detect_dsz_patterns_v2(out)
    out = pd.concat([out, dsz_v2], axis=1)

    return out