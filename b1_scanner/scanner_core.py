"""B1 strategy scan logic for v2.0."""

from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from data_sources import FetchParams, fetch_data
from indicators import add_all_indicators


# ─────────────────────────────────────────────
# 日志配置
# ─────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# 为每次扫描创建独立的日志文件
scan_log_file = os.path.join(LOG_DIR, f"scan_{date.today().strftime('%Y%m%d')}.log")

# 配置根日志记录器
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# 文件处理器 - 记录所有日志
file_handler = logging.FileHandler(scan_log_file, mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
file_handler.setFormatter(file_formatter)

# 控制台处理器 - 只显示INFO以上
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(file_formatter)

# 避免重复添加handlers
if not root_logger.handlers:
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)


@dataclass
class B1Config:
    source: str = "tushare"  # tushare | yahoo
    tushare_token: Optional[str] = None
    start: date = date.today() - timedelta(days=365 * 3)
    end: date = date.today()
    weekly_lookback_days: int = 365 * 8
    min_daily_bars: int = 150
    min_weekly_bars: int = 240
    request_pause_sec: float = 0.2
    
    # 战法选择
    strategy: str = "B1"  # B1, B2, 自定义
    
    # 知行趋势线选项
    require_golden_cross: bool = False  # 是否要求金叉后第一个B1
    require_brick_white: bool = False  # 是否要求白色砖头 (买点)
    # 市值筛选 (单位: 亿元)
    market_cap_min: float = 0  # 最小市值, 0表示不限制
    market_cap_max: float = 0  # 最大市值, 0表示不限制
    # 行业板块
    sector: str = "全部"
    
    # ========== 自定义策略参数 ==========
    # J值范围 (None表示不限制)
    kdj_j_min: Optional[float] = None
    kdj_j_max: Optional[float] = None
    
    # DEA条件: "any" | "positive" | "negative"
    dea_condition: str = "any"
    
    # 白砖信号: "any" | "required" | "forbidden"
    brick_white_condition: str = "any"
    
    # 知行线金叉: "any" | "white_above" | "yellow_above"
    zhixing_condition: str = "any"
    
    # 涨幅条件: None表示不限制, 否则要求 >= X%
    price_change_min: Optional[float] = None
    
    # 量比条件: None表示不限制, 否则要求 >= X
    volume_ratio_min: Optional[float] = None


def is_cn_mainboard(symbol: str) -> bool:
    """Main board filter: 600/601/603/000xxxx."""
    symbol = symbol.strip().upper()
    m = re.search(r"(\d{6})", symbol)
    if not m:
        # non-CN symbols are treated as not-mainboard for strict B1
        return False

    code = m.group(1)
    return code.startswith(("600", "601", "603", "000"))


def _weekly_ma_check(weekly_df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "weekly_ok": False,
        "weekly_reason": "周线数据不足",
        "ma30": None,
        "ma60": None,
        "ma120": None,
        "ma240": None,
    }

    if weekly_df is None or weekly_df.empty or len(weekly_df) < 240:
        return out

    wk = weekly_df.copy()
    wk["ma30"] = wk["close"].rolling(30).mean()
    wk["ma60"] = wk["close"].rolling(60).mean()
    wk["ma120"] = wk["close"].rolling(120).mean()
    wk["ma240"] = wk["close"].rolling(240).mean()

    latest = wk.iloc[-1]
    ma30 = latest["ma30"]
    ma60 = latest["ma60"]
    ma120 = latest["ma120"]
    ma240 = latest["ma240"]

    out.update(
        {
            "ma30": float(ma30) if pd.notna(ma30) else None,
            "ma60": float(ma60) if pd.notna(ma60) else None,
            "ma120": float(ma120) if pd.notna(ma120) else None,
            "ma240": float(ma240) if pd.notna(ma240) else None,
        }
    )

    if pd.isna(ma30) or pd.isna(ma60) or pd.isna(ma120) or pd.isna(ma240):
        out["weekly_reason"] = "周线均线尚未形成"
        return out

    out["weekly_ok"] = bool(ma30 > ma60 > ma120 > ma240)
    out["weekly_reason"] = "OK" if out["weekly_ok"] else "周线均线未多头排列"
    return out


def scan_symbol(symbol: str, name: str = "", config: Optional[B1Config] = None) -> Dict[str, Any]:
    cfg = config or B1Config()
    symbol = symbol.strip().upper()
    
    logger.info(f"🟢 开始扫描: {symbol} ({name}) | 战法: {cfg.strategy}")

    daily = fetch_data(
        FetchParams(
            symbol=symbol,
            start=cfg.start,
            end=cfg.end,
            source=cfg.source,
            period="daily",
            tushare_token=cfg.tushare_token,
        )
    )
    
    logger.debug(f"  📊 {symbol}: 获取日线数据 {len(daily)} 条")

    weekly_start = cfg.end - timedelta(days=cfg.weekly_lookback_days)
    weekly = fetch_data(
        FetchParams(
            symbol=symbol,
            start=weekly_start,
            end=cfg.end,
            source=cfg.source,
            period="weekly",
            tushare_token=cfg.tushare_token,
        )
    )
    
    logger.debug(f"  📊 {symbol}: 获取周线数据 {len(weekly)} 条")

    if daily.empty or len(daily) < cfg.min_daily_bars:
        logger.warning(f"  ⛔ {symbol}: 日线数据不足 ({len(daily)} 条)")
        return {
            "symbol": symbol,
            "name": name or symbol,
            "passed": False,
            "error": f"日线数据不足: {len(daily)} 条",
        }

    daily_ind = add_all_indicators(daily)
    latest = daily_ind.iloc[-1]

    weekly_info = _weekly_ma_check(weekly)
    logger.debug(f"  📈 {symbol}: 周线检查 - {'✅' if weekly_info['weekly_ok'] else '❌'} ({weekly_info['weekly_reason']})")

    # 知行趋势线: 白线(短期) vs 黄线(长期)
    zhixing_white = latest.get("zhixing_white")  # 白线: EMA(EMA(C,10),10)
    zhixing_yellow = latest.get("zhixing_yellow")  # 黄线: (MA14+MA28+MA57+MA114)/4
    
    # 核心条件: 黄线在白线之下 = 上涨趋势 (white > yellow)
    zhixing_bullish = bool(
        pd.notna(zhixing_white) and pd.notna(zhixing_yellow) and 
        zhixing_white > zhixing_yellow
    )
    
    # 金叉后第一个B1的检测
    # 逻辑：今天满足B1条件 + 最近一次金叉发生在今天之前 + 从金叉到现在没有其他B1
    golden_cross = False
    kdj_j = latest.get("kdj_j")  # 提前获取，避免后续使用时报错
    
    if cfg.require_golden_cross and len(daily_ind) >= 2:
        # 检查今天是否满足B1条件（J < 13）
        is_b1_today = bool(pd.notna(kdj_j) and kdj_j < 13)
        
        if is_b1_today:
            # 找到历史上所有金叉位置（白线从下方穿越到上方）
            golden_cross_indices = []
            for i in range(1, len(daily_ind)):
                curr = daily_ind.iloc[i]
                prev = daily_ind.iloc[i-1]
                curr_white = curr.get("zhixing_white")
                curr_yellow = curr.get("zhixing_yellow")
                prev_white = prev.get("zhixing_white")
                prev_yellow = prev.get("zhixing_yellow")
                
                if all(pd.notna(v) for v in [curr_white, curr_yellow, prev_white, prev_yellow]):
                    # 昨天白线 <= 黄线，今天白线 > 黄线 = 金叉
                    if prev_white <= prev_yellow and curr_white > curr_yellow:
                        golden_cross_indices.append(i)
            
            if golden_cross_indices:
                # 最近一次金叉的位置
                last_gc_idx = golden_cross_indices[-1]
                
                # 检查从金叉到现在是否已有其他B1（排除今天）
                b1_count_after_gc = 0
                for i in range(last_gc_idx + 1, len(daily_ind) - 1):  # 排除今天
                    row = daily_ind.iloc[i]
                    j_val = row.get("kdj_j")
                    if pd.notna(j_val) and j_val < 13:
                        b1_count_after_gc += 1
                
                # 如果从金叉到现在（排除今天）没有其他B1，今天就是第一个
                if b1_count_after_gc == 0:
                    golden_cross = True
    
    # 砖型图检测
    brick_white = latest.get("brick_white", 0)  # 白色砖头 = 1 (买点)
    brick_chart = latest.get("brick_chart", 0)
    
    mainboard_ok = is_cn_mainboard(symbol)
    macd_dea = latest.get("macd_dea")
    close = latest.get("close")
    volume = latest.get("volume", 0)
    prev_close = daily_ind.iloc[-2]["close"] if len(daily_ind) >= 2 else close
    
    # 计算涨跌幅
    price_change_pct = ((close - prev_close) / prev_close * 100) if pd.notna(prev_close) and prev_close != 0 else 0
    
    # 计算量比 (成交量 / 5日平均成交量)
    vol_ma5 = daily_ind["volume"].rolling(5).mean().iloc[-1] if len(daily_ind) >= 5 else volume
    volume_ratio = (volume / vol_ma5) if pd.notna(vol_ma5) and vol_ma5 != 0 else 0

    # B1 条件
    macd_ok = bool(pd.notna(macd_dea) and macd_dea > 0)
    kdj_ok_b1 = bool(pd.notna(kdj_j) and kdj_j < 13)  # B1: J < 13
    
    # B2 条件
    kdj_ok_b2 = bool(pd.notna(kdj_j) and kdj_j < 55)  # B2: J < 55
    price_ok = price_change_pct >= 4  # 涨幅 >= 4%
    vol_ok = volume_ratio >= 1.1  # 量比 >= 1.1
    
    # DSZ战法默认值
    dsz_ok = False
    dsz_ok_temp = False

    # 根据战法选择条件
    if cfg.strategy == "B1":
        kdj_ok = kdj_ok_b1
        zhixing_required = True  # B1需要知行趋势线
    elif cfg.strategy == "B2":
        kdj_ok = kdj_ok_b2
        zhixing_required = False  # B2不需要知行趋势线
    elif cfg.strategy == "DSZ战法":
        # DSZ砖型图战法 - 使用三种定式检测
        kdj_ok = True  # DSZ不检查J值
        # 需要DSZ信号列
        dsz_cols = ["dsz_n_pattern", "dsz_sideways", "dsz_uptrend_cont"]
        if all(col in daily_ind.columns for col in dsz_cols):
            dsz_signal = daily_ind[["dsz_n_pattern", "dsz_sideways", "dsz_uptrend_cont"]].iloc[-1]
            dsz_ok = bool(dsz_signal["dsz_n_pattern"]) or bool(dsz_signal["dsz_sideways"]) or bool(dsz_signal["dsz_uptrend_cont"])
        else:
            dsz_ok = False
        # conditions 还没定义，先存到临时变量
        dsz_ok_temp = dsz_ok
        zhixing_required = True  # DSZ需要知行线多头
    else:  # 自定义策略
        # 自定义 J 值范围
        if cfg.kdj_j_min is not None and pd.notna(kdj_j):
            kdj_ok = kdj_j >= cfg.kdj_j_min
        elif cfg.kdj_j_max is not None and pd.notna(kdj_j):
            kdj_ok = kdj_j <= cfg.kdj_j_max
        elif cfg.kdj_j_min is not None and cfg.kdj_j_max is not None and pd.notna(kdj_j):
            kdj_ok = cfg.kdj_j_min <= kdj_j <= cfg.kdj_j_max
        else:
            kdj_ok = True
        
        # 自定义 DEA 条件
        if cfg.dea_condition == "positive":
            macd_ok = bool(pd.notna(macd_dea) and macd_dea > 0)
        elif cfg.dea_condition == "negative":
            macd_ok = bool(pd.notna(macd_dea) and macd_dea < 0)
        else:  # any
            macd_ok = pd.notna(macd_dea)
        
        # 自定义知行线条件
        if cfg.zhixing_condition == "white_above":
            zhixing_bullish = bool(pd.notna(zhixing_white) and pd.notna(zhixing_yellow) and zhixing_white > zhixing_yellow)
        elif cfg.zhixing_condition == "yellow_above":
            zhixing_bullish = bool(pd.notna(zhixing_white) and pd.notna(zhixing_yellow) and zhixing_yellow > zhixing_white)
        else:  # any
            zhixing_bullish = True
        
        # 自定义涨幅条件
        if cfg.price_change_min is not None:
            price_ok = price_change_pct >= cfg.price_change_min
        else:
            price_ok = True
        
        # 自定义量比条件
        if cfg.volume_ratio_min is not None:
            vol_ok = volume_ratio >= cfg.volume_ratio_min
        else:
            vol_ok = True
        
        # 自定义白砖条件
        if cfg.brick_white_condition == "required":
            brick_white_ok = bool(brick_white == 1)
        elif cfg.brick_white_condition == "forbidden":
            brick_white_ok = bool(brick_white != 1)
        else:  # any
            brick_white_ok = True
        
        zhixing_required = cfg.zhixing_condition != "any"

    conditions = {
        "mainboard_ok": mainboard_ok,
        "weekly_ok": weekly_info["weekly_ok"],
        "macd_dea_ok": macd_ok,
        "kdj_j_ok": kdj_ok,
        "zhixing_bullish": zhixing_bullish if zhixing_required else True,  # B1需要
        "golden_cross": golden_cross if cfg.require_golden_cross else True,
        "brick_white": bool(brick_white == 1),
        # B2特有
        "price_change_ok": price_ok if cfg.strategy in ["B2", "DSZ战法"] else (price_ok if cfg.strategy == "自定义" else True),
        "volume_ratio_ok": vol_ok if cfg.strategy in ["B2", "DSZ战法"] else (vol_ok if cfg.strategy == "自定义" else True),
        "dsz_ok": dsz_ok_temp if cfg.strategy == "DSZ战法" else True,
        # 自定义策略特有
        "brick_white_ok": brick_white_ok if cfg.strategy == "自定义" else True,
    }

    passed = all([
        conditions["mainboard_ok"],
        conditions["weekly_ok"],
        conditions["macd_dea_ok"],
        conditions["kdj_j_ok"],
        conditions["zhixing_bullish"],
        conditions["golden_cross"],
    ])
    
    # B2额外条件
    if cfg.strategy == "B2":
        passed = passed and conditions["price_change_ok"] and conditions["volume_ratio_ok"]
    
    # 自定义策略额外条件
    if cfg.strategy == "自定义":
        if cfg.brick_white_condition != "any":
            passed = passed and conditions.get("brick_white_ok", True)
        if cfg.price_change_min is not None:
            passed = passed and conditions["price_change_ok"]
        if cfg.volume_ratio_min is not None:
            passed = passed and conditions["volume_ratio_ok"]
    
    # 如果开启白色砖头筛选
    if cfg.require_brick_white and not conditions["brick_white"]:
        passed = False
    
    # ─────────────────────────────────────────
    # 记录每个条件的筛选结果
    # ─────────────────────────────────────────
    logger.info(f"  📋 {symbol} 条件检查:")
    logger.info(f"      主板: {'✅' if conditions['mainboard_ok'] else '❌'} ({symbol})")
    logger.info(f"      周线: {'✅' if conditions['weekly_ok'] else '❌'} ({weekly_info['weekly_reason']})")
    logger.info(f"      MACD DEA>0: {'✅' if conditions['macd_dea_ok'] else '❌'} (DEA={macd_dea})")
    logger.info(f"      KDJ J: {'✅' if conditions['kdj_j_ok'] else '❌'} (J={kdj_j}, 阈值={13 if cfg.strategy=='B1' else (55 if cfg.strategy=='B2' else '自定义')})")
    logger.info(f"      知行线: {'✅' if conditions['zhixing_bullish'] else '❌'} (白线>{'黄线' if zhixing_bullish else '≤黄线'})")
    if cfg.require_golden_cross:
        logger.info(f"      金叉: {'✅' if conditions['golden_cross'] else '❌'}")
    if cfg.strategy in ["B2", "DSZ战法", "自定义"]:
        logger.info(f"      涨幅: {'✅' if conditions['price_change_ok'] else '❌'} ({price_change_pct:.2f}%)")
        logger.info(f"      量比: {'✅' if conditions['volume_ratio_ok'] else '❌'} ({volume_ratio:.2f})")
    logger.info(f"      白色砖头: {'✅ 买点!' if conditions['brick_white'] else '⚪'}")
    logger.info(f"  🎯 {symbol} 最终结果: {'✅ 通过' if passed else '❌ 未通过'}")

    return {
        "symbol": symbol,
        "name": name or symbol,
        "passed": passed,
        "error": "",
        "conditions": conditions,
        "weekly_reason": weekly_info["weekly_reason"],
        "metrics": {
            "date": latest["date"].strftime("%Y-%m-%d") if pd.notna(latest["date"]) else "",
            "close": float(close) if pd.notna(close) else None,
            "zhixing_white": float(zhixing_white) if pd.notna(zhixing_white) else None,
            "zhixing_yellow": float(zhixing_yellow) if pd.notna(zhixing_yellow) else None,
            "macd_dea": float(macd_dea) if pd.notna(macd_dea) else None,
            "kdj_j": float(kdj_j) if pd.notna(kdj_j) else None,
            "rsi14": float(latest.get("rsi14")) if pd.notna(latest.get("rsi14")) else None,
            "boll_upper": float(latest.get("boll_upper")) if pd.notna(latest.get("boll_upper")) else None,
            "boll_mid": float(latest.get("boll_mid")) if pd.notna(latest.get("boll_mid")) else None,
            "boll_lower": float(latest.get("boll_lower")) if pd.notna(latest.get("boll_lower")) else None,
            "volume": float(latest.get("volume")) if pd.notna(latest.get("volume")) else None,
            "vol_ma20": float(latest.get("vol_ma20")) if pd.notna(latest.get("vol_ma20")) else None,
            # B2 额外指标
            "price_change_pct": round(price_change_pct, 2),
            "volume_ratio": round(volume_ratio, 2),
            # 砖型图
            "brick_chart": float(brick_chart) if pd.notna(brick_chart) else None,
            "brick_white": bool(brick_white == 1),
            "ma30_w": weekly_info["ma30"],
            "ma60_w": weekly_info["ma60"],
            "ma120_w": weekly_info["ma120"],
            "ma240_w": weekly_info["ma240"],
        },
        "daily_df": daily_ind,
        "weekly_df": weekly,
    }


def scan_batch(
    symbol_rows: Iterable[Dict[str, str]],
    config: Optional[B1Config] = None,
    progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Scan multiple symbols in parallel using ThreadPoolExecutor.
    
    Args:
        symbol_rows: Iterable of dicts with 'symbol' and optional 'name'.
        config: B1Config instance.
        progress_callback: Called with (completed_count, total_count, result_dict)
                          after each stock completes. Safe to call from Streamlit.
    
    Returns:
        List of result dicts in the same order as input symbols.
    """
    cfg = config or B1Config()
    
    # Parse once into a list so we know the total count
    rows_list: List[Dict[str, str]] = []
    for row in symbol_rows:
        symbol = row.get("symbol", "").strip()
        name = row.get("name", "").strip()
        if symbol:
            rows_list.append({"symbol": symbol, "name": name})
    
    if not rows_list:
        return []
    
    total = len(rows_list)
    
    # Determine thread count based on pause_sec (aggressive parallelism for short pauses)
    # Tushare rate limit ≈ 2000/min → ~33/sec. Each stock = 2 calls.
    # Use up to 15 workers; reduce sleep proportionally.
    raw_workers = max(1, int(0.5 / max(cfg.request_pause_sec, 0.05)))
    max_workers = min(raw_workers, 50)
    
    # Thread-local cache for shared data to avoid redundant fetches (optional optimisation)
    results_ordered: List[Optional[Dict[str, Any]]] = [None] * total
    completed_count = 0
    
    def worker(idx: int, row: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
        symbol = row["symbol"]
        name = row["name"]
        try:
            result = scan_symbol(symbol=symbol, name=name, config=cfg)
        except Exception as exc:
            result = {
                "symbol": symbol,
                "name": name or symbol,
                "passed": False,
                "error": str(exc),
            }
        return idx, result
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, i, row): i for i, row in enumerate(rows_list)}
        
        for future in as_completed(futures):
            idx, result = future.result()
            results_ordered[idx] = result
            completed_count += 1
            
            if progress_callback:
                try:
                    progress_callback(completed_count, total, result)
                except Exception:
                    # Don't let progress callback errors crash the scan
                    pass
    
    # Filter out None results (shouldn't happen, but be safe)
    return [r for r in results_ordered if r is not None]


def flatten_result_for_table(result: Dict[str, Any]) -> Dict[str, Any]:
    if result.get("error"):
        return {
            "symbol": result.get("symbol"),
            "name": result.get("name"),
            "passed": False,
            "error": result.get("error", ""),
        }

    metrics = result.get("metrics", {})
    cond = result.get("conditions", {})

    return {
        "symbol": result.get("symbol"),
        "name": result.get("name"),
        "passed": result.get("passed", False),
        "date": metrics.get("date"),
        "close": metrics.get("close"),
        "zhixing_white": metrics.get("zhixing_white"),
        "zhixing_yellow": metrics.get("zhixing_yellow"),
        "macd_dea": metrics.get("macd_dea"),
        "kdj_j": metrics.get("kdj_j"),
        "rsi14": metrics.get("rsi14"),
        "mainboard_ok": cond.get("mainboard_ok"),
        "weekly_ok": cond.get("weekly_ok"),
        "macd_dea_ok": cond.get("macd_dea_ok"),
        "kdj_j_ok": cond.get("kdj_j_ok"),
        "zhixing_bullish": cond.get("zhixing_bullish"),
        "golden_cross": cond.get("golden_cross"),
        "weekly_reason": result.get("weekly_reason", ""),
        "error": result.get("error", ""),
    }
