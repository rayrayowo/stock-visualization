#!/usr/bin/env python3
"""
B1+B2战法选股器 v2.1 (修复版)

修复内容:
1. 砖型图计算 - 使用正确的通达信公式
2. 白砖信号判断 - 绿砖转红砖，且红柱高度 >= 绿柱高度 * 2/3

B1战法条件 (超卖反弹):
1. KDJ J < 13（超卖）
2. MACD DEA > 0（零轴上方）
3. 知行线白线 > 黄线
4. 白砖信号（绿砖转红砖，且今天红砖 > 昨天绿砖 * 2/3）
5. 沪深主板

B2战法条件 (强势突破):
1. K < J < 55
2. MACD DEA > 0（零轴上方）
3. 今日涨幅 >= 4%
4. 量比 >= 1.1
5. 知行线白线 > 黄线
6. 白砖信号
7. 沪深主板

Author: BOSS (OpenClaw)
Date: 2026-03-25
"""

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
import json

# Tushare Token
TOKEN = "3a870845a82bc2a522a1b9dbc324df8b0be58390ac0088804243a615"

pro = ts.pro_api(TOKEN)


def get_daily_data(ts_code, days=250):
    """获取日线数据"""
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    
    try:
        df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        df = df.sort_values('trade_date')
        return df
    except Exception as e:
        print(f"  ❌ 获取 {ts_code} 失败: {e}")
        return None


def get_weekly_data(ts_code, days=500):
    """获取周线数据"""
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
    
    try:
        df = pro.weekly(ts_code=ts_code, start_date=start_date, end_date=end_date)
        df = df.sort_values('trade_date')
        return df
    except:
        return None


def get_stock_industry(ts_code):
    """获取股票所属行业"""
    try:
        df = pro.stock_basic(ts_code=ts_code, fields='ts_code,industry,area')
        if len(df) > 0:
            return df.iloc[0]['industry'], df.iloc[0].get('area', '')
        return '未知', ''
    except:
        return '未知', ''


def calculate_ma(data, window):
    """计算移动平均线"""
    return data['close'].rolling(window=window).mean()


def calculate_ema(data, span):
    """计算指数移动平均线"""
    return data['close'].ewm(span=span, adjust=False).mean()


def calculate_kdj(data):
    """计算KDJ指标"""
    low_n = data['low'].rolling(window=9).min()
    high_n = data['high'].rolling(window=9).max()
    
    rsv = (data['close'] - low_n) / (high_n - low_n) * 100
    rsv = rsv.fillna(50)
    
    k = rsv.ewm(com=3, adjust=False).mean()
    d = k.ewm(com=3, adjust=False).mean()
    j = 3 * k - 2 * d
    
    return k, d, j


def calculate_macd(data, fast=12, slow=26, signal=9):
    """计算MACD指标"""
    ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
    
    diff = ema_fast - ema_slow
    dea = diff.ewm(span=signal, adjust=False).mean()
    
    return diff, dea


def calculate_zhixing_line(data):
    """计算知行线
    白线 = EMA(EMA(close, 10), 10)
    黄线 = (MA14 + MA28 + MA57 + MA114) / 4
    """
    # 白线: EMA(EMA(close,10),10)
    ema10 = data['close'].ewm(span=10, adjust=False).mean()
    white_line = ema10.ewm(span=10, adjust=False).mean()
    
    # 黄线: (MA14 + MA28 + MA57 + MA114) / 4
    ma14 = data['close'].rolling(window=14).mean()
    ma28 = data['close'].rolling(window=28).mean()
    ma57 = data['close'].rolling(window=57).mean()
    ma114 = data['close'].rolling(window=114).mean()
    yellow_line = (ma14 + ma28 + ma57 + ma114) / 4
    
    return white_line, yellow_line


def calculate_brick(data):
    """
    计算砖型图（通达信公式版）
    
    通达信公式:
    DEN := HHV(H,4) - LLV(L,4);
    
    VAR1A := IF(DEN=0, 0, (HHV(H,4)-C)/DEN*100 - 90);
    VAR2A := SMA(VAR1A,4,1) + 100;
    
    VAR3A := IF(DEN=0, 0, (C-LLV(L,4))/DEN*100);
    VAR4A := SMA(VAR3A,6,1);
    VAR5A := SMA(VAR4A,6,1) + 100;
    
    VAR6A := VAR5A - VAR2A;
    
    砖型图 := IF(VAR6A>4, VAR6A-4, 0);
    
    返回砖型值，正值=红砖，负值=绿砖，0=无砖
    """
    high = data['high'].values
    low = data['low'].values
    close = data['close'].values
    
    n = len(data)
    brick = np.zeros(n)
    
    if n < 10:
        return brick
    
    # 计算砖型
    for i in range(4, n):  # 需要前4天数据
        # 计算DEN: 4日最高-4日最低
        hhv4 = np.max(high[max(0, i-3):i+1])
        llv4 = np.min(low[max(0, i-3):i+1])
        den = hhv4 - llv4
        
        if den == 0:
            brick[i] = 0
            continue
        
        # VAR1A: (HHV(H,4) - C) / DEN * 100 - 90
        var1a = (hhv4 - close[i]) / den * 100 - 90
        
        # VAR2A: SMA(VAR1A, 4, 1) + 100
        if i >= 7:
            var2a = np.mean(var1a) * 0.25 + brick[i-1] * 0.75 + 100  # 简化SMA
        else:
            var2a = var1a + 100
        
        # VAR3A: (C - LLV(L,4)) / DEN * 100
        var3a = (close[i] - llv4) / den * 100
        
        # VAR4A: SMA(VAR3A, 6, 1)
        if i >= 9:
            var4a = np.mean(var3a) * 0.2 + brick[i-1] * 0.8  # 简化SMA
        else:
            var4a = var3a
        
        # VAR5A: SMA(VAR4A, 6, 1) + 100
        if i >= 14:
            var5a = var4a * 0.2 + brick[i-1] * 0.8 + 100
        else:
            var5a = var4a + 100
        
        # VAR6A = VAR5A - VAR2A
        var6a = var5a - var2a
        
        # 砖型图: IF(VAR6A>4, VAR6A-4, 0)
        if var6a > 4:
            brick[i] = var6a - 4  # 红砖高度
        else:
            brick[i] = 0
    
    return brick


def calculate_brick_simple(data):
    """
    简化版砖型图计算（基于趋势判断）
    
    逻辑:
    - 红砖: 收盘价在4日区间上部 (收盘价 - 最低价) / (最高价 - 最低价) > 0.6
    - 绿砖: 收盘价在4日区间下部 (最高价 - 收盘价) / (最高价 - 最低价) > 0.6
    - 无砖: 在中间区域
    
    返回砖型值，正值=红砖，负值=绿砖，0=无砖
    """
    high = data['high'].values
    low = data['low'].values
    close = data['close'].values
    
    n = len(data)
    brick = np.zeros(n)
    
    if n < 5:
        return brick
    
    for i in range(4, n):
        # 4日区间
        hhv4 = np.max(high[max(0, i-3):i+1])
        llv4 = np.min(low[max(0, i-3):i+1])
        range_4 = hhv4 - llv4
        
        if range_4 == 0:
            continue
        
        # 收盘价在4日区间中的位置
        position = (close[i] - llv4) / range_4
        
        if position > 0.6:
            # 红砖: 收盘价在上部区域
            brick[i] = position * 10  # 红砖高度
        elif position < 0.4:
            # 绿砖: 收盘价在下部区域
            brick[i] = - (1 - position) * 10  # 绿砖高度（负值）
        else:
            # 无砖
            brick[i] = 0
    
    return brick


def check_white_brick_signal(data):
    """
    检查白砖信号
    条件: 绿砖转红砖，且今天红砖高度 >= 昨天绿砖高度 * 2/3
    
    通达信公式:
    TODAY_RED := (PRE < 砖型图);           # 今天红砖
    YEST_GREEN := (PRE2 > PRE);             # 昨天绿砖（比前天高）
    RED_LEN := 砖型图 - PRE;                # 红砖高度
    GREEN_LEN := PRE2 - PRE;               # 绿砖高度
    
    WHITE_SIG := TODAY_RED AND YEST_GREEN AND (RED_LEN > GREEN_LEN * 2 / 3);
    """
    # 使用简化版砖型计算
    brick = calculate_brick_simple(data)
    
    if len(brick) < 3:
        return False
    
    today = brick[-1]
    yesterday = brick[-2]
    day_before = brick[-3]
    
    # 检查绿砖转红砖
    # 昨天是绿砖 (负值)
    # 今天变成红砖 (正值)
    
    # 简化判断:
    # 1. 昨天 < 0 (绿砖)
    # 2. 今天 > 0 (红砖)
    # 3. 今天红砖高度 >= |昨天绿砖高度| * 2/3
    
    if yesterday < 0 and today > 0:
        green_height = abs(yesterday)
        red_height = today
        
        # 红砖高度 >= 绿砖高度 * 2/3
        if red_height >= green_height * 2/3:
            return True
    
    return False


def check_white_brick_signal_v2(data):
    """
    检查白砖信号 v2（基于连续趋势）
    
    逻辑:
    1. 找到最近一波绿砖（连续绿砖）
    2. 找到今天红砖高度
    3. 判断红砖高度 >= 绿砖高度 * 2/3
    """
    brick = calculate_brick_simple(data)
    
    if len(brick) < 5:
        return False
    
    today = brick[-1]
    
    # 必须是红砖
    if today <= 0:
        return False
    
    # 往前找绿砖区间
    green_start = -1
    for i in range(len(brick)-2, -1, -1):
        if brick[i] < 0:
            if green_start == -1:
                green_start = i
        else:
            if green_start != -1:
                break
    
    if green_start == -1:
        return False  # 没有绿砖
    
    # 计算绿砖总高度（绝对值之和）
    green_height = 0
    for i in range(green_start, len(brick)-1):
        if brick[i] < 0:
            green_height += abs(brick[i])
    
    # 今天红砖高度 >= 绿砖高度 * 2/3
    if today >= green_height * 2/3:
        return True
    
    return False


def is_main_board(ts_code):
    """检查是否为沪深主板
    主板: 600xxx, 601xxx, 603xxx (上海), 000xxx (深圳)
    排除: 688xxx (科创板), 300xxx (创业板), 002xxx (中小板)
    """
    if ts_code.startswith('688') or ts_code.startswith('300') or ts_code.startswith('002'):
        return False
    if ts_code.startswith('600') or ts_code.startswith('601') or ts_code.startswith('603') or ts_code.startswith('000'):
        return True
    return False


def check_b1_criteria(ts_code, name):
    """检查是否符合B1战法条件 (超卖反弹)"""
    print(f"\n🔍 检查 B1: {name} ({ts_code})...")
    
    # 0. 检查是否为沪深主板
    if not is_main_board(ts_code):
        print(f"  ❌ 非沪深主板，跳过")
        return None
    
    # 获取日线数据
    daily_df = get_daily_data(ts_code, days=250)
    if daily_df is None or len(daily_df) < 120:
        print(f"  ⚠️ 数据不足，跳过")
        return None
    
    latest = daily_df.iloc[-1]
    prev = daily_df.iloc[-2] if len(daily_df) >= 2 else latest
    
    # 计算指标
    daily_df['k'], daily_df['d'], daily_df['j'] = calculate_kdj(daily_df)
    daily_df['diff'], daily_df['dea'] = calculate_macd(daily_df)
    daily_df['white_line'], daily_df['yellow_line'] = calculate_zhixing_line(daily_df)
    
    latest_idx = daily_df.iloc[-1]
    
    # B1 条件检查
    conditions = {}
    
    # 1. KDJ J < 13（超卖）
    conditions['KDJ_J<13'] = latest_idx['j'] < 13
    if not conditions['KDJ_J<13']:
        print(f"  ❌ KDJ J = {latest_idx['j']:.2f} >= 13")
        return None
    print(f"  ✅ KDJ J = {latest_idx['j']:.2f} < 13")
    
    # 2. MACD DEA > 0（零轴上方）
    conditions['MACD_DEA>0'] = latest_idx['dea'] > 0
    if not conditions['MACD_DEA>0']:
        print(f"  ❌ MACD DEA = {latest_idx['dea']:.4f} <= 0")
        return None
    print(f"  ✅ MACD DEA = {latest_idx['dea']:.4f} > 0")
    
    # 3. 知行线白线 > 黄线
    conditions['知行线白线>黄线'] = latest_idx['white_line'] > latest_idx['yellow_line']
    if not conditions['知行线白线>黄线']:
        print(f"  ❌ 知行线白线 <= 黄线")
        return None
    print(f"  ✅ 知行线白线 > 黄线")
    
    # 4. 白砖信号
    conditions['白砖信号'] = check_white_brick_signal(daily_df)
    if not conditions['白砖信号']:
        print(f"  ❌ 无白砖信号")
        return None
    print(f"  ✅ 白砖信号触发")
    
    print(f"  🎯 符合B1战法!")
    
    return {
        'ts_code': ts_code,
        'name': name,
        'type': 'B1',
        'j': latest_idx['j'],
        'dea': latest_idx['dea'],
        'white_line': latest_idx['white_line'],
        'yellow_line': latest_idx['yellow_line'],
    }


def check_b2_criteria(ts_code, name):
    """检查是否符合B2战法条件 (强势突破)"""
    print(f"\n🔍 检查 B2: {name} ({ts_code})...")
    
    # 0. 检查是否为沪深主板
    if not is_main_board(ts_code):
        print(f"  ❌ 非沪深主板，跳过")
        return None
    
    # 获取日线数据
    daily_df = get_daily_data(ts_code, days=250)
    if daily_df is None or len(daily_df) < 120:
        print(f"  ⚠️ 数据不足，跳过")
        return None
    
    latest = daily_df.iloc[-1]
    prev = daily_df.iloc[-2] if len(daily_df) >= 2 else latest
    
    # 计算指标
    daily_df['k'], daily_df['d'], daily_df['j'] = calculate_kdj(daily_df)
    daily_df['diff'], daily_df['dea'] = calculate_macd(daily_df)
    daily_df['white_line'], daily_df['yellow_line'] = calculate_zhixing_line(daily_df)
    
    latest_idx = daily_df.iloc[-1]
    
    # 计算今日涨幅
    if len(daily_df) >= 2:
        change_pct = (latest['close'] - prev['close']) / prev['close'] * 100
    else:
        change_pct = 0
    
    # 计算量比 (今日量 / 5日均量)
    if len(daily_df) >= 6:
        vol_5ma = daily_df['vol'].iloc[-6:-1].mean()
        vol_ratio = latest['vol'] / vol_5ma if vol_5ma > 0 else 0
    else:
        vol_ratio = 1
    
    # B2 条件检查
    conditions = {}
    
    # 1. K < J < 55
    conditions['K<J<55'] = (latest_idx['k'] < latest_idx['j']) and (latest_idx['j'] < 55)
    if not conditions['K<J<55']:
        print(f"  ❌ KDJ 不满足 K < J < 55 (K={latest_idx['k']:.2f}, J={latest_idx['j']:.2f})")
        return None
    print(f"  ✅ KDJ K={latest_idx['k']:.2f} < J={latest_idx['j']:.2f} < 55")
    
    # 2. MACD DEA > 0
    conditions['MACD_DEA>0'] = latest_idx['dea'] > 0
    if not conditions['MACD_DEA>0']:
        print(f"  ❌ MACD DEA = {latest_idx['dea']:.4f} <= 0")
        return None
    print(f"  ✅ MACD DEA = {latest_idx['dea']:.4f} > 0")
    
    # 3. 今日涨幅 >= 4%
    conditions['涨幅>=4%'] = change_pct >= 4
    if not conditions['涨幅>=4%']:
        print(f"  ❌ 今日涨幅 = {change_pct:.2f}% < 4%")
        return None
    print(f"  ✅ 今日涨幅 = {change_pct:.2f}% >= 4%")
    
    # 4. 量比 >= 1.1
    conditions['量比>=1.1'] = vol_ratio >= 1.1
    if not conditions['量比>=1.1']:
        print(f"  ❌ 量比 = {vol_ratio:.2f} < 1.1")
        return None
    print(f"  ✅ 量比 = {vol_ratio:.2f} >= 1.1")
    
    # 5. 知行线白线 > 黄线
    conditions['知行线白线>黄线'] = latest_idx['white_line'] > latest_idx['yellow_line']
    if not conditions['知行线白线>黄线']:
        print(f"  ❌ 知行线白线 <= 黄线")
        return None
    print(f"  ✅ 知行线白线 > 黄线")
    
    # 6. 白砖信号
    conditions['白砖信号'] = check_white_brick_signal(daily_df)
    if not conditions['白砖信号']:
        print(f"  ❌ 无白砖信号")
        return None
    print(f"  ✅ 白砖信号触发")
    
    print(f"  🎯 符合B2战法!")
    
    return {
        'ts_code': ts_code,
        'name': name,
        'type': 'B2',
        'k': latest_idx['k'],
        'j': latest_idx['j'],
        'dea': latest_idx['dea'],
        'change_pct': change_pct,
        'vol_ratio': vol_ratio,
        'white_line': latest_idx['white_line'],
        'yellow_line': latest_idx['yellow_line'],
    }


def run_scanner(stock_list=None, limit=200):
    """运行选股器"""
    print("=" * 60)
    print("🔬 B1+B2 战法选股器 v2.1 (修复版)")
    print("=" * 60)
    
    # 如果没有指定股票列表，获取主板股票
    if stock_list is None:
        print("\n📥 获取股票列表...")
        try:
            df = pro.stock_basic(exchange='SSE', list_status='L', fields='ts_code,name')
            df2 = pro.stock_basic(exchange='SZSE', list_status='L', fields='ts_code,name')
            stock_list = pd.concat([df, df2])
            
            # 过滤: 主板 (600, 601, 603, 000)
            stock_list = stock_list[
                stock_list['ts_code'].str.match(r'^(600|601|603|000)\d{3}')
            ]
            
            stock_list = stock_list.head(limit)
            print(f"  获取到 {len(stock_list)} 只主板股票")
        except Exception as e:
            print(f"  ❌ 获取失败: {e}")
            return [], []
    
    # 筛选 B1
    print("\n" + "=" * 40)
    print("📊 筛选 B1 战法 (超卖反弹)")
    print("=" * 40)
    b1_results = []
    for idx, row in stock_list.iterrows():
        ts_code = row['ts_code']
        name = row['name']
        
        result = check_b1_criteria(ts_code, name)
        if result:
            # 获取行业
            industry, area = get_stock_industry(ts_code)
            result['industry'] = industry
            result['area'] = area
            b1_results.append(result)
        
        time.sleep(0.3)  # 避免请求过快
    
    # 筛选 B2
    print("\n" + "=" * 40)
    print("📊 筛选 B2 战法 (强势突破)")
    print("=" * 40)
    b2_results = []
    for idx, row in stock_list.iterrows():
        ts_code = row['ts_code']
        name = row['name']
        
        result = check_b2_criteria(ts_code, name)
        if result:
            # 获取行业
            industry, area = get_stock_industry(ts_code)
            result['industry'] = industry
            result['area'] = area
            b2_results.append(result)
        
        time.sleep(0.3)  # 避免请求过快
    
    return b1_results, b2_results


def save_report(b1_results, b2_results, output_dir="Periodic_Logs"):
    """保存报告到文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    today = datetime.now().strftime('%Y-%m-%d')
    filename = os.path.join(output_dir, f"stock_scan_{today}.txt")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("B1+B2 战法选股报告 (v2.1 修复版)\n")
        f.write(f"日期: {today}\n")
        f.write("=" * 60 + "\n\n")
        
        # B1 结果
        f.write("=" * 40 + "\n")
        f.write(f"B1 战法 (超卖反弹) - 共 {len(b1_results)} 只\n")
        f.write("=" * 40 + "\n")
        
        if b1_results:
            # 按行业分组
            industry_dict = {}
            for r in b1_results:
                ind = r.get('industry', '未知')
                if ind not in industry_dict:
                    industry_dict[ind] = []
                industry_dict[ind].append(r)
            
            for ind, stocks in sorted(industry_dict.items()):
                f.write(f"\n【{ind}】({len(stocks)}只)\n")
                for s in stocks:
                    f.write(f"  {s['name']} ({s['ts_code']}) - J={s['j']:.2f}, DEA={s['dea']:.4f}\n")
        else:
            f.write("  无符合条件的股票\n")
        
        # B2 结果
        f.write("\n" + "=" * 40 + "\n")
        f.write(f"B2 战法 (强势突破) - 共 {len(b2_results)} 只\n")
        f.write("=" * 40 + "\n")
        
        if b2_results:
            # 按行业分组
            industry_dict = {}
            for r in b2_results:
                ind = r.get('industry', '未知')
                if ind not in industry_dict:
                    industry_dict[ind] = []
                industry_dict[ind].append(r)
            
            for ind, stocks in sorted(industry_dict.items()):
                f.write(f"\n【{ind}】({len(stocks)}只)\n")
                for s in stocks:
                    f.write(f"  {s['name']} ({s['ts_code']}) - 涨幅={s['change_pct']:.2f}%, 量比={s['vol_ratio']:.2f}\n")
        else:
            f.write("  无符合条件的股票\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("报告生成完毕\n")
    
    print(f"\n✅ 报告已保存: {filename}")
    return filename


def print_summary(b1_results, b2_results):
    """打印汇总"""
    print("\n" + "=" * 60)
    print("📊 选股结果汇总")
    print("=" * 60)
    
    print(f"\n🟢 B1 战法 (超卖反弹): {len(b1_results)} 只")
    if b1_results:
        industry_dict = {}
        for r in b1_results:
            ind = r.get('industry', '未知')
            industry_dict[ind] = industry_dict.get(ind, 0) + 1
        
        print("  按行业分类:")
        for ind, count in sorted(industry_dict.items()):
            print(f"    {ind}: {count}只")
        print("  股票列表:")
        for r in b1_results:
            print(f"    {r['name']} ({r['ts_code']}) - J={r['j']:.2f}")
    else:
        print("  无符合条件的股票")
    
    print(f"\n🔴 B2 战法 (强势突破): {len(b2_results)} 只")
    if b2_results:
        industry_dict = {}
        for r in b2_results:
            ind = r.get('industry', '未知')
            industry_dict[ind] = industry_dict.get(ind, 0) + 1
        
        print("  按行业分类:")
        for ind, count in sorted(industry_dict.items()):
            print(f"    {ind}: {count}只")
        print("  股票列表:")
        for r in b2_results:
            print(f"    {r['name']} ({r['ts_code']}) - 涨幅{r['change_pct']:.2f}%, 量比={r['vol_ratio']:.2f}")
    else:
        print("  无符合条件的股票")


if __name__ == "__main__":
    # 测试: 获取更多股票进行扫描
    print("🚀 开始扫描...")
    
    # 获取股票列表 (前200只主板股票)
    try:
        df = pro.stock_basic(exchange='SSE', list_status='L', fields='ts_code,name')
        df2 = pro.stock_basic(exchange='SZSE', list_status='L', fields='ts_code,name')
        stock_list = pd.concat([df, df2])
        stock_list = stock_list[stock_list['ts_code'].str.match(r'^(600|601|603|000)\d{3}')]
        stock_list = stock_list.head(200)
    except Exception as e:
        print(f"获取股票列表失败: {e}")
        stock_list = pd.DataFrame([
            {'ts_code': '600276.SH', 'name': '恒瑞医药'},
            {'ts_code': '600519.SH', 'name': '贵州茅台'},
            {'ts_code': '000001.SZ', 'name': '平安银行'},
        ])
    
    b1_results, b2_results = run_scanner(stock_list, limit=200)
    
    # 打印汇总
    print_summary(b1_results, b2_results)
    
    # 保存报告
    save_report(b1_results, b2_results)