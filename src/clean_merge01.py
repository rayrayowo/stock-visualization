from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def to_snake(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()


def parse_horizons(raw: str) -> tuple[int, ...]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError("forward horizons must be positive integers")
        values.append(value)
    unique_sorted = tuple(sorted(set(values)))
    if not unique_sorted:
        raise ValueError("at least one forward horizon is required")
    return unique_sorted


def prepare_prices(path: Path) -> pd.DataFrame:
    prices = pd.read_csv(path)
    prices = prices.rename(columns={c: to_snake(c) for c in prices.columns})
    if "name" in prices.columns and "ticker" not in prices.columns:
        prices = prices.rename(columns={"name": "ticker"})

    required = {"date", "open", "high", "low", "close", "volume", "ticker"}
    missing = required.difference(prices.columns)
    if missing:
        raise ValueError(f"prices missing required columns: {sorted(missing)}")

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["ticker"] = prices["ticker"].astype(str).str.strip()

    for col in ["open", "high", "low", "close", "volume"]:
        prices[col] = pd.to_numeric(prices[col], errors="coerce")

    before = len(prices)
    prices = prices.dropna(subset=["date", "ticker", "high", "low", "close"])
    prices["volume"] = prices["volume"].fillna(0.0)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    removed = before - len(prices)
    if removed:
        print(f"[prices] dropped {removed} rows with invalid required fields")
    return prices


def prepare_constituents(path: Path) -> pd.DataFrame:
    const = pd.read_csv(path)
    const = const.rename(columns={c: to_snake(c) for c in const.columns})
    rename_map = {
        "symbol": "ticker",
        "gics_sector": "sector",
        "gics_sub_industry": "sub_industry",
    }
    const = const.rename(columns={k: v for k, v in rename_map.items() if k in const.columns})

    if "ticker" not in const.columns:
        raise ValueError("constituents file must include Symbol/ticker column")

    const["ticker"] = const["ticker"].astype(str).str.strip()
    const = const.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    return const


def add_indicators_and_forward_returns(
    df: pd.DataFrame,
    ma_short: int,
    ma_long: int,
    macd_fast: int,
    macd_slow: int,
    macd_signal: int,
    kdj_window: int,
    kdj_smooth: int,
    forward_horizons: tuple[int, ...],
) -> pd.DataFrame:
    def per_ticker(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        close = g["close"]
        high = g["high"]
        low = g["low"]

        g["dollar_volume"] = close * g["volume"]
        g["ret_1d"] = close.pct_change()

        g["ma_short"] = close.rolling(window=ma_short, min_periods=ma_short).mean()
        g["ma_long"] = close.rolling(window=ma_long, min_periods=ma_long).mean()

        ma_state = np.where(g["ma_short"] > g["ma_long"], 1, np.where(g["ma_short"] < g["ma_long"], -1, 0))
        ma_state = ma_state.astype("int8")
        ma_prev = pd.Series(ma_state, index=g.index).shift(1).fillna(0).astype("int8")
        g["ma_state"] = ma_state
        g["ma_cross"] = np.select(
            [(ma_prev <= 0) & (ma_state == 1), (ma_prev >= 0) & (ma_state == -1)],
            [1, -1],
            default=0,
        ).astype("int8")

        ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
        g["macd_line"] = ema_fast - ema_slow
        g["macd_signal"] = g["macd_line"].ewm(span=macd_signal, adjust=False).mean()
        g["macd_hist"] = g["macd_line"] - g["macd_signal"]

        macd_state = np.where(g["macd_line"] > g["macd_signal"], 1, np.where(g["macd_line"] < g["macd_signal"], -1, 0))
        macd_state = macd_state.astype("int8")
        macd_prev = pd.Series(macd_state, index=g.index).shift(1).fillna(0).astype("int8")
        g["macd_state"] = macd_state
        g["macd_cross"] = np.select(
            [(macd_prev <= 0) & (macd_state == 1), (macd_prev >= 0) & (macd_state == -1)],
            [1, -1],
            default=0,
        ).astype("int8")

        low_n = low.rolling(window=kdj_window, min_periods=kdj_window).min()
        high_n = high.rolling(window=kdj_window, min_periods=kdj_window).max()
        span_n = (high_n - low_n).replace(0, np.nan)
        rsv = ((close - low_n) / span_n * 100.0).clip(lower=0, upper=100)
        g["kdj_rsv"] = rsv
        g["kdj_k"] = rsv.ewm(alpha=1.0 / kdj_smooth, adjust=False).mean()
        g["kdj_d"] = g["kdj_k"].ewm(alpha=1.0 / kdj_smooth, adjust=False).mean()
        g["kdj_j"] = 3.0 * g["kdj_k"] - 2.0 * g["kdj_d"]

        kdj_state = np.where(g["kdj_k"] > g["kdj_d"], 1, np.where(g["kdj_k"] < g["kdj_d"], -1, 0))
        kdj_state = kdj_state.astype("int8")
        kdj_prev = pd.Series(kdj_state, index=g.index).shift(1).fillna(0).astype("int8")
        g["kdj_state"] = kdj_state
        g["kdj_cross"] = np.select(
            [(kdj_prev <= 0) & (kdj_state == 1), (kdj_prev >= 0) & (kdj_state == -1)],
            [1, -1],
            default=0,
        ).astype("int8")
        g["kdj_overbought"] = ((g["kdj_k"] >= 80) & (g["kdj_d"] >= 80)).astype("int8")
        g["kdj_oversold"] = ((g["kdj_k"] <= 20) & (g["kdj_d"] <= 20)).astype("int8")

        for horizon in forward_horizons:
            g[f"fwd_ret_{horizon}d"] = close.shift(-horizon) / close - 1.0

        return g

    grouped = df.sort_values(["ticker", "date"]).groupby("ticker", sort=False)
    chunks = []
    for ticker, g in grouped:
        g = g.copy()
        g["ticker"] = ticker
        chunks.append(per_ticker(g))
    out = pd.concat(chunks, axis=0, ignore_index=True)
    return out


def add_market_regime(df: pd.DataFrame, bear_drawdown: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = df[["date", "ticker", "ret_1d"]].copy()
    market = market.groupby("date", as_index=False)["ret_1d"].mean()
    market = market.sort_values("date").reset_index(drop=True)
    market["ret_1d"] = market["ret_1d"].fillna(0.0)

    market["mkt_proxy_close"] = (1.0 + market["ret_1d"]).cumprod() * 100.0
    market["mkt_proxy_ma200"] = market["mkt_proxy_close"].rolling(window=200, min_periods=200).mean()
    market["mkt_proxy_peak"] = market["mkt_proxy_close"].cummax()
    market["mkt_proxy_drawdown"] = market["mkt_proxy_close"] / market["mkt_proxy_peak"] - 1.0

    is_bear = (market["mkt_proxy_close"] < market["mkt_proxy_ma200"]) | (
        market["mkt_proxy_drawdown"] <= bear_drawdown
    )
    market["regime"] = np.where(is_bear, "bear", "bull")

    merged = df.merge(
        market[["date", "regime", "mkt_proxy_close", "mkt_proxy_ma200", "mkt_proxy_drawdown"]],
        on="date",
        how="left",
        validate="many_to_one",
    )
    return merged, market


def add_leader_flags(df: pd.DataFrame, leader_topk: int) -> tuple[pd.DataFrame, str]:
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    df["sector"] = df["sector"].fillna("Unknown")

    rank_col = df.groupby(["date", "sector"])["dollar_volume"].rank(method="first", ascending=False)
    df["leader_rank_in_sector"] = rank_col.astype("int32")

    flag_col = f"is_leader_top{leader_topk}"
    df[flag_col] = (df["leader_rank_in_sector"] <= leader_topk).astype("int8")
    return df, flag_col


def build_master(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    prices = prepare_prices(args.prices)
    const = prepare_constituents(args.constituents)

    merged = prices.merge(const, on="ticker", how="left", validate="many_to_one")
    if "sector" not in merged.columns:
        merged["sector"] = "Unknown"

    unmatched = merged["sector"].isna().sum()
    if unmatched:
        print(f"[merge] {unmatched} rows have no sector mapping (set to Unknown)")
    merged["sector"] = merged["sector"].fillna("Unknown")

    master = add_indicators_and_forward_returns(
        merged,
        ma_short=args.ma_short,
        ma_long=args.ma_long,
        macd_fast=args.macd_fast,
        macd_slow=args.macd_slow,
        macd_signal=args.macd_signal,
        kdj_window=args.kdj_window,
        kdj_smooth=args.kdj_smooth,
        forward_horizons=args.forward_horizons,
    )
    master, market = add_market_regime(master, bear_drawdown=args.bear_drawdown)
    master, leader_flag_col = add_leader_flags(master, leader_topk=args.leader_topk)

    preferred = [
        "date",
        "ticker",
        "security",
        "sector",
        "sub_industry",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dollar_volume",
        "ret_1d",
        "regime",
        "mkt_proxy_close",
        "mkt_proxy_ma200",
        "mkt_proxy_drawdown",
        "leader_rank_in_sector",
        leader_flag_col,
    ]
    ordered = [c for c in preferred if c in master.columns] + [c for c in master.columns if c not in preferred]
    master = master.loc[:, ordered].sort_values(["date", "ticker"]).reset_index(drop=True)

    return master, market, leader_flag_col


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed stock master table with indicators, regimes, and forward returns."
    )
    parser.add_argument("--prices", type=Path, default=RAW_DIR / "all_stocks_5yr.csv")
    parser.add_argument("--constituents", type=Path, default=RAW_DIR / "constituents.csv")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--parquet-name", default="master.parquet")
    parser.add_argument("--csv-name", default="master.csv")
    parser.add_argument("--market-name", default="market_proxy_daily.csv")

    parser.add_argument("--ma-short", type=int, default=50)
    parser.add_argument("--ma-long", type=int, default=200)
    parser.add_argument("--macd-fast", type=int, default=12)
    parser.add_argument("--macd-slow", type=int, default=26)
    parser.add_argument("--macd-signal", type=int, default=9)
    parser.add_argument("--kdj-window", type=int, default=9)
    parser.add_argument("--kdj-smooth", type=int, default=3)
    parser.add_argument("--bear-drawdown", type=float, default=-0.20)
    parser.add_argument("--leader-topk", type=int, default=5)
    parser.add_argument("--forward-horizons", default="1,5,20")
    parser.add_argument("--skip-csv", action="store_true")

    args = parser.parse_args()
    args.forward_horizons = parse_horizons(args.forward_horizons)
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    master, market, leader_flag_col = build_master(args)

    parquet_path = args.out_dir / args.parquet_name
    master.to_parquet(parquet_path, index=False)

    if args.skip_csv:
        csv_path = None
    else:
        csv_path = args.out_dir / args.csv_name
        master.to_csv(csv_path, index=False)

    market_path = args.out_dir / args.market_name
    market.to_csv(market_path, index=False)

    print(f"[done] master rows: {len(master):,}, columns: {master.shape[1]}")
    print(f"[done] forward horizons: {args.forward_horizons}")
    print(f"[done] leader flag column: {leader_flag_col}")
    print(f"[saved] {parquet_path}")
    if csv_path is not None:
        print(f"[saved] {csv_path}")
    print(f"[saved] {market_path}")


if __name__ == "__main__":
    main()
