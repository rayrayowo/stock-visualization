from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MAINBOARD_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003")
VALID_TS_CODE = re.compile(r"^\d{6}\.(SZ|SH|BJ)$")
VALID_TUSHARE_PICKLE = re.compile(r"^(\d{6}_[A-Z]{2})_daily_qfq_.*\.pkl$")


def ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def tongdaxin_sma(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(alpha=1.0 / span, adjust=False).mean()


def calc_zhixing_white(frame: pd.DataFrame) -> pd.Series:
    return ema(ema(frame["close"], 10), 10)


def calc_zhixing_yellow(frame: pd.DataFrame) -> pd.Series:
    return (
        ma(frame["close"], 14)
        + ma(frame["close"], 28)
        + ma(frame["close"], 57)
        + ma(frame["close"], 114)
    ) / 4


def calc_brick_chart(frame: pd.DataFrame) -> pd.DataFrame:
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]

    hhv_high_4 = high.rolling(window=4, min_periods=4).max()
    llv_low_4 = low.rolling(window=4, min_periods=4).min()
    den = hhv_high_4 - llv_low_4

    var1a = np.where(den == 0, 0, (hhv_high_4 - close) / den * 100 - 90)
    var1a = pd.Series(var1a, index=frame.index)
    var2a = tongdaxin_sma(var1a, 4) + 100

    var3a = np.where(den == 0, 0, (close - llv_low_4) / den * 100)
    var3a = pd.Series(var3a, index=frame.index)
    var4a = tongdaxin_sma(var3a, 6)
    var5a = tongdaxin_sma(var4a, 6) + 100

    var6a = var5a - var2a
    brick = pd.Series(np.where(var6a > 4, var6a - 4, 0), index=frame.index)

    pre = brick.shift(1)
    pre2 = brick.shift(2)
    today_red = brick > pre
    yest_green = pre2 > pre
    red_len = brick - pre
    green_len = pre2 - pre
    white_sig = today_red & yest_green & (red_len > green_len * 2 / 3)

    return pd.DataFrame(
        {
            "brick_chart": brick,
            "brick_white": white_sig.astype("int8"),
            "brick_red_len": red_len,
            "brick_green_len": green_len,
        }
    )


def add_super_white_brick_signal(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out = out.sort_values("date").reset_index(drop=True)

    out["zhixing_white"] = calc_zhixing_white(out)
    out["zhixing_yellow"] = calc_zhixing_yellow(out)
    out = pd.concat([out, calc_brick_chart(out)], axis=1)

    out["prev_close"] = out["close"].shift(1)
    out["price_change_pct"] = (out["close"] / out["prev_close"] - 1.0) * 100.0

    out["upper_shadow"] = (out["high"] - out["close"]).clip(lower=0.0)
    out["lower_shadow"] = (out["open"] - out["low"]).clip(lower=0.0)
    out["candle_body"] = out["close"] - out["open"]

    positive_body = out["candle_body"] > 0
    out["shadow_body_ratio"] = np.where(
        positive_body,
        (out["upper_shadow"] + out["lower_shadow"]) / out["candle_body"],
        np.nan,
    )

    out["zhixing_bullish"] = out["zhixing_white"] > out["zhixing_yellow"]
    out["strong_white_ok"] = (
        (out["brick_red_len"] >= out["brick_green_len"] * 1.0)
        & (out["brick_green_len"] > 0)
    )
    out["tight_candle_ok"] = positive_body & (out["shadow_body_ratio"] <= 0.4)
    out["price_under_4_ok"] = out["price_change_pct"] < 4.0
    out["super_white_brick_signal"] = (
        out["zhixing_bullish"]
        & (out["brick_white"] == 1)
        & out["strong_white_ok"]
        & out["tight_candle_ok"]
        & out["price_under_4_ok"]
    )
    return out


def iter_latest_tushare_daily_files(cache_dir: Path) -> list[Path]:
    latest: dict[str, Path] = {}
    for path in cache_dir.glob("*_daily_qfq_*.pkl"):
        match = VALID_TUSHARE_PICKLE.match(path.name)
        if not match:
            continue
        symbol = match.group(1)
        if not symbol.startswith(MAINBOARD_PREFIXES):
            continue
        previous = latest.get(symbol)
        if previous is None or path.name > previous.name:
            latest[symbol] = path
    return [latest[key] for key in sorted(latest)]


def _to_ts_code(symbol: str) -> str:
    cleaned = symbol.strip().upper().replace("_", ".")
    if VALID_TS_CODE.match(cleaned):
        return cleaned
    raise ValueError(f"Invalid symbol format: {symbol}")


def load_month_end_market_cap(source_path: Path, output_path: Path | None = None) -> pd.DataFrame:
    frame = pd.read_csv(source_path, usecols=["ts_code", "trade_date", "total_mv", "circ_mv"])
    frame["ts_code"] = frame["ts_code"].astype(str).str.strip().str.upper()
    frame["trade_date"] = frame["trade_date"].astype(str).str.strip()

    frame = frame[frame["ts_code"].str.match(VALID_TS_CODE, na=False)].copy()
    frame = frame[frame["trade_date"].str.fullmatch(r"\d{8}", na=False)].copy()

    frame["trade_date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d", errors="coerce")
    frame["total_mv"] = pd.to_numeric(frame["total_mv"], errors="coerce")
    frame["circ_mv"] = pd.to_numeric(frame["circ_mv"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "total_mv"]).copy()

    frame = frame.sort_values(["ts_code", "trade_date"]).drop_duplicates(
        subset=["ts_code", "trade_date"],
        keep="last",
    )
    frame["month"] = frame["trade_date"].dt.to_period("M")

    month_end = (
        frame.groupby(["ts_code", "month"], as_index=False)
        .tail(1)
        .sort_values(["ts_code", "trade_date"])
        .reset_index(drop=True)
    )
    month_end["total_mv_billion_cny"] = month_end["total_mv"] / 100000.0
    month_end["circ_mv_billion_cny"] = month_end["circ_mv"] / 100000.0
    month_end["market_cap_bucket"] = pd.cut(
        month_end["total_mv_billion_cny"],
        bins=[0, 20, 50, 100, 200, np.inf],
        labels=["<20B", "20-50B", "50-100B", "100-200B", "200B+"],
        right=False,
    )

    result = month_end[
        [
            "ts_code",
            "trade_date",
            "total_mv",
            "circ_mv",
            "total_mv_billion_cny",
            "circ_mv_billion_cny",
            "market_cap_bucket",
        ]
    ].rename(columns={"trade_date": "market_cap_date"})

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
    return result


def _forward_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1.0


def build_super_white_brick_events(
    cache_dir: Path,
    horizons: Iterable[int] = (1, 3, 5, 10, 20),
    limit: int | None = None,
    min_bars: int = 150,
) -> pd.DataFrame:
    files = iter_latest_tushare_daily_files(cache_dir)
    if limit is not None:
        files = files[:limit]

    event_frames: list[pd.DataFrame] = []
    for idx, path in enumerate(files, start=1):
        price_frame = pd.read_pickle(path)
        if price_frame is None or len(price_frame) < min_bars:
            continue

        symbol = path.name.split("_daily_qfq_")[0]
        ts_code = _to_ts_code(symbol)

        working = price_frame.copy()
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        for column in ["open", "high", "low", "close", "volume"]:
            working[column] = pd.to_numeric(working[column], errors="coerce")
        working = working.dropna(subset=["date", "open", "high", "low", "close"]).copy()
        working = working.sort_values("date").drop_duplicates(subset=["date"], keep="last")
        if len(working) < min_bars:
            continue

        signal_frame = add_super_white_brick_signal(working)
        for horizon in horizons:
            signal_frame[f"fwd_ret_{horizon}d"] = _forward_return(signal_frame["close"], horizon)

        events = signal_frame[signal_frame["super_white_brick_signal"]].copy()
        if events.empty:
            continue

        events["ts_code"] = ts_code
        events["signal_year"] = events["date"].dt.year
        event_frames.append(
            events[
                [
                    "ts_code",
                    "date",
                    "close",
                    "price_change_pct",
                    "zhixing_white",
                    "zhixing_yellow",
                    "brick_chart",
                    "brick_white",
                    "brick_red_len",
                    "brick_green_len",
                    "shadow_body_ratio",
                    "strong_white_ok",
                    "tight_candle_ok",
                    "price_under_4_ok",
                    "signal_year",
                ]
                + [f"fwd_ret_{horizon}d" for horizon in horizons]
            ]
        )

        if idx % 250 == 0:
            print(f"[super-white-brick] processed {idx:,}/{len(files):,} symbols")

    if not event_frames:
        return pd.DataFrame()
    return pd.concat(event_frames, ignore_index=True).sort_values(["date", "ts_code"]).reset_index(drop=True)


def attach_market_cap(events: pd.DataFrame, month_end_market_cap: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()

    right_groups = {
        ts_code: group.sort_values("market_cap_date").reset_index(drop=True)
        for ts_code, group in month_end_market_cap.groupby("ts_code", sort=False)
    }

    merged_groups: list[pd.DataFrame] = []
    for ts_code, left_group in events.groupby("ts_code", sort=False):
        left_sorted = left_group.sort_values("date").reset_index(drop=True)
        right_sorted = right_groups.get(ts_code)
        if right_sorted is None or right_sorted.empty:
            merged_groups.append(left_sorted)
            continue

        merged = pd.merge_asof(
            left_sorted,
            right_sorted.drop(columns=["ts_code"]),
            left_on="date",
            right_on="market_cap_date",
            direction="backward",
        )
        merged_groups.append(merged)

    return pd.concat(merged_groups, ignore_index=True).sort_values(["date", "ts_code"]).reset_index(drop=True)


def _summarize_metric(frame: pd.DataFrame, group_col: str, return_col: str) -> pd.DataFrame:
    summary = (
        frame.groupby(group_col, dropna=False)
        .agg(
            signal_count=(return_col, "count"),
            avg_return=(return_col, "mean"),
            median_return=(return_col, "median"),
            win_rate=(return_col, lambda s: (s > 0).mean()),
        )
        .reset_index()
    )
    return summary


def build_analysis_tables(events: pd.DataFrame, default_horizon: int = 5) -> dict[str, pd.DataFrame]:
    if events.empty:
        empty = pd.DataFrame()
        return {
            "overall": empty,
            "by_gain_bucket": empty,
            "by_market_cap_bucket": empty,
            "by_year": empty,
        }

    return_col = f"fwd_ret_{default_horizon}d"
    working = events.dropna(subset=[return_col]).copy()

    working["gain_bucket"] = pd.cut(
        working["price_change_pct"],
        bins=[-np.inf, 0, 1, 2, 3, 4],
        labels=["<=0%", "0-1%", "1-2%", "2-3%", "3-4%"],
        include_lowest=True,
        right=False,
    )

    overall = pd.DataFrame(
        [
            {
                "signal_count": int(working[return_col].count()),
                "unique_symbols": int(working["ts_code"].nunique()),
                "avg_return": working[return_col].mean(),
                "median_return": working[return_col].median(),
                "win_rate": (working[return_col] > 0).mean(),
                "avg_same_day_gain_pct": working["price_change_pct"].mean(),
            }
        ]
    )

    by_gain_bucket = _summarize_metric(working, "gain_bucket", return_col)
    by_market_cap_bucket = _summarize_metric(
        working.dropna(subset=["market_cap_bucket"]),
        "market_cap_bucket",
        return_col,
    )
    by_year = (
        working.groupby("signal_year", dropna=False)
        .agg(
            signal_count=(return_col, "count"),
            avg_return=(return_col, "mean"),
            win_rate=(return_col, lambda s: (s > 0).mean()),
        )
        .reset_index()
    )

    return {
        "overall": overall,
        "by_gain_bucket": by_gain_bucket,
        "by_market_cap_bucket": by_market_cap_bucket,
        "by_year": by_year,
    }
