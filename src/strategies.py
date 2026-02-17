from __future__ import annotations

from typing import Callable

import pandas as pd


StrategyFn = Callable[[pd.DataFrame], pd.Series]


def flip_position_from_events(entry_signal: pd.Series, exit_signal: pd.Series) -> pd.Series:
    """Build a 0/1 position series from entry/exit boolean events."""
    entry_signal = entry_signal.fillna(False).astype(bool)
    exit_signal = exit_signal.fillna(False).astype(bool)

    position = []
    state = 0.0
    for enter, exit_ in zip(entry_signal.to_numpy(), exit_signal.to_numpy(), strict=True):
        if exit_:
            state = 0.0
        if enter:
            state = 1.0
        position.append(state)
    return pd.Series(position, index=entry_signal.index, dtype="float64")


def strategy_ma_cross(frame: pd.DataFrame) -> pd.Series:
    entry = frame["ma_cross"] == 1
    exit_ = frame["ma_cross"] == -1
    return flip_position_from_events(entry, exit_)


def strategy_macd_cross(frame: pd.DataFrame) -> pd.Series:
    entry = frame["macd_cross"] == 1
    exit_ = frame["macd_cross"] == -1
    return flip_position_from_events(entry, exit_)


def strategy_kdj_cross(frame: pd.DataFrame) -> pd.Series:
    entry = frame["kdj_cross"] == 1
    exit_ = frame["kdj_cross"] == -1
    return flip_position_from_events(entry, exit_)


def strategy_custom_template(frame: pd.DataFrame) -> pd.Series:
    """
    Template strategy for your own B1/B2 style logic.

    Replace entry/exit rules with your exact trading rules.
    """
    entry = (frame["ma_cross"] == 1) & (frame["macd_cross"] == 1)
    exit_ = (frame["ma_cross"] == -1) | (frame["macd_cross"] == -1)
    return flip_position_from_events(entry, exit_)


STRATEGY_REGISTRY: dict[str, StrategyFn] = {
    "ma_cross": strategy_ma_cross,
    "macd_cross": strategy_macd_cross,
    "kdj_cross": strategy_kdj_cross,
    "custom_template": strategy_custom_template,
}

STRATEGY_REQUIREMENTS: dict[str, set[str]] = {
    "ma_cross": {"ma_cross"},
    "macd_cross": {"macd_cross"},
    "kdj_cross": {"kdj_cross"},
    "custom_template": {"ma_cross", "macd_cross"},
}


def list_strategies() -> list[str]:
    return sorted(STRATEGY_REGISTRY.keys())


def parse_strategy_names(raw: str) -> list[str]:
    names = []
    for token in raw.split(","):
        name = token.strip()
        if name:
            names.append(name)
    names = list(dict.fromkeys(names))
    if not names:
        raise ValueError("No strategies were provided.")
    unknown = [name for name in names if name not in STRATEGY_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}. Available: {list_strategies()}")
    return names


def ensure_strategy_columns(df: pd.DataFrame, strategy_name: str) -> None:
    required = STRATEGY_REQUIREMENTS[strategy_name]
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Strategy '{strategy_name}' missing required columns: {missing}")


def build_strategy_positions(df: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    ensure_strategy_columns(df, strategy_name)
    base_required = {"date", "ticker", "ret_1d"}
    base_missing = sorted(base_required.difference(df.columns))
    if base_missing:
        raise ValueError(f"Input df missing base columns: {base_missing}")

    strategy_fn = STRATEGY_REGISTRY[strategy_name]
    keep_columns = [
        "date",
        "ticker",
        "close",
        "ret_1d",
        "regime",
        "sector",
        "dollar_volume",
        "is_static_leader",
    ]
    keep_columns += [c for c in df.columns if c.startswith("is_leader_top")]
    keep_columns = [c for c in dict.fromkeys(keep_columns) if c in df.columns]

    out_chunks: list[pd.DataFrame] = []
    grouped = df.sort_values(["ticker", "date"]).groupby("ticker", sort=False)
    for _, ticker_frame in grouped:
        ticker_frame = ticker_frame.copy()
        position = pd.to_numeric(strategy_fn(ticker_frame), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
        if len(position) != len(ticker_frame):
            raise ValueError(f"Strategy '{strategy_name}' returned invalid length for ticker segment.")
        ticker_frame["position"] = position.astype("float64")
        out_chunks.append(ticker_frame[keep_columns + ["position"]])

    out = pd.concat(out_chunks, ignore_index=True)
    out["strategy"] = strategy_name
    return out
