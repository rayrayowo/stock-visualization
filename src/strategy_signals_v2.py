from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN_PATH = ROOT / "data" / "processed" / "master.parquet"
DEFAULT_OUT_DIR = ROOT / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate simplified B1/B2 strategy signals from master dataset."
    )
    parser.add_argument("--in-path", type=Path, default=DEFAULT_IN_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--lookback-b1", type=int, default=60)
    parser.add_argument("--b1-j-threshold", type=float, default=13.0)
    parser.add_argument("--b2-j-threshold", type=float, default=55.0)
    parser.add_argument("--b2-min-return", type=float, default=0.04)
    parser.add_argument("--skip-csv", action="store_true")
    return parser.parse_args()


def load_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Unsupported input format. Use .parquet or .csv")

    required = {"date", "ticker", "close", "volume"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close", "volume"]).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def compute_macd_for_close(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def compute_kdj_for_ohlc(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 9,
    smooth: int = 3,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    low_n = low.rolling(window=window, min_periods=window).min()
    high_n = high.rolling(window=window, min_periods=window).max()
    span_n = (high_n - low_n).replace(0, np.nan)
    rsv = ((close - low_n) / span_n * 100.0).clip(lower=0, upper=100)
    k = rsv.ewm(alpha=1.0 / smooth, adjust=False).mean()
    d = k.ewm(alpha=1.0 / smooth, adjust=False).mean()
    j = 3.0 * k - 2.0 * d
    return k, d, j


def ensure_indicators(df: pd.DataFrame) -> pd.DataFrame:
    need_macd = not {"macd_line", "macd_signal", "macd_hist"}.issubset(df.columns)
    need_kdj = not {"kdj_k", "kdj_d", "kdj_j"}.issubset(df.columns)

    if not need_macd and not need_kdj:
        return df

    if need_kdj:
        if "high" not in df.columns or "low" not in df.columns:
            raise ValueError("KDJ columns are missing, but high/low are not available to compute KDJ(9,3).")

        df["high"] = pd.to_numeric(df["high"], errors="coerce")
        df["low"] = pd.to_numeric(df["low"], errors="coerce")

    chunks: list[pd.DataFrame] = []
    for _, ticker_frame in df.groupby("ticker", sort=False):
        g = ticker_frame.copy()
        close = g["close"]

        if need_macd:
            macd_line, macd_signal, macd_hist = compute_macd_for_close(close=close)
            g["macd_line"] = macd_line
            g["macd_signal"] = macd_signal
            g["macd_hist"] = macd_hist

        if need_kdj:
            k, d, j = compute_kdj_for_ohlc(high=g["high"], low=g["low"], close=close, window=9, smooth=3)
            g["kdj_k"] = k
            g["kdj_d"] = d
            g["kdj_j"] = j

        chunks.append(g)

    return pd.concat(chunks, ignore_index=True)


def build_signals(
    df: pd.DataFrame,
    lookback_b1: int,
    b1_j_threshold: float,
    b2_j_threshold: float,
    b2_min_return: float,
) -> pd.DataFrame:
    if lookback_b1 <= 0:
        raise ValueError("--lookback-b1 must be a positive integer")

    chunks: list[pd.DataFrame] = []
    for _, ticker_frame in df.groupby("ticker", sort=False):
        g = ticker_frame.copy()
        g["daily_return"] = g["close"].pct_change()

        b1 = (g["macd_signal"] > 0) & (g["kdj_j"] < b1_j_threshold)
        g["b1_signal"] = b1.astype("int8")
        g["has_b1_last_N"] = (
            g["b1_signal"]
            .rolling(window=lookback_b1, min_periods=1)
            .max()
            .fillna(0)
            .astype("int8")
        )

        prev_volume = g["volume"].shift(1)
        b2 = (
            (g["has_b1_last_N"] == 1)
            & (g["daily_return"] >= b2_min_return)
            & (g["kdj_j"] < b2_j_threshold)
            & (g["volume"] >= prev_volume)
        )
        g["b2_signal"] = b2.astype("int8")
        chunks.append(g)

    return pd.concat(chunks, ignore_index=True)


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered: list[str] = ["date", "ticker"]
    if "sector" in df.columns:
        ordered.append("sector")

    ordered += [
        "close",
        "volume",
        "daily_return",
        "macd_line",
        "macd_signal",
        "macd_hist",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "b1_signal",
        "has_b1_last_N",
        "b2_signal",
    ]

    for fwd_col in ["fwd_ret_1d", "fwd_ret_5d", "fwd_ret_20d"]:
        if fwd_col in df.columns:
            ordered.append(fwd_col)

    missing = sorted(set(ordered).difference(df.columns))
    if missing:
        raise ValueError(f"Signal output missing required columns: {missing}")
    return df.loc[:, ordered].sort_values(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_master(args.in_path)
    df = ensure_indicators(df)
    df = build_signals(
        df=df,
        lookback_b1=args.lookback_b1,
        b1_j_threshold=args.b1_j_threshold,
        b2_j_threshold=args.b2_j_threshold,
        b2_min_return=args.b2_min_return,
    )
    out = select_output_columns(df)

    parquet_path = args.out_dir / "signals_v2.parquet"
    out.to_parquet(parquet_path, index=False)

    csv_path = None
    if not args.skip_csv:
        csv_path = args.out_dir / "signals_v2.csv"
        out.to_csv(csv_path, index=False)

    print(f"[done] rows: {len(out):,}, cols: {out.shape[1]}")
    print(
        f"[done] b1_count={int(out['b1_signal'].sum()):,}, "
        f"b2_count={int(out['b2_signal'].sum()):,}, "
        f"lookback={args.lookback_b1}"
    )
    print(f"[saved] {parquet_path}")
    if csv_path is not None:
        print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
