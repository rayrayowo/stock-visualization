from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestOutputs:
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


def _validate_position_frame(df: pd.DataFrame) -> None:
    required = {"date", "ticker", "ret_1d", "position"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Position frame missing required columns: {missing}")


def _compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    peak = equity_curve.cummax()
    return equity_curve / peak - 1.0


def build_daily_portfolio_returns(
    position_frame: pd.DataFrame,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    _validate_position_frame(position_frame)

    frame = position_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ret_1d"] = pd.to_numeric(frame["ret_1d"], errors="coerce").fillna(0.0)
    frame["position"] = pd.to_numeric(frame["position"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    frame["position_prev"] = frame.groupby("ticker", sort=False)["position"].shift(1).fillna(0.0)
    frame["turnover"] = (frame["position"] - frame["position_prev"]).abs()

    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    frame["gross_ret"] = frame["position_prev"] * frame["ret_1d"]
    frame["cost_ret"] = frame["turnover"] * cost_rate
    frame["portfolio_ret"] = frame["gross_ret"] - frame["cost_ret"]
    frame["is_active"] = (frame["position_prev"].abs() > 0).astype("int8")

    aggregations: dict[str, tuple[str, str | callable]] = {
        "portfolio_ret": ("portfolio_ret", "mean"),
        "gross_ret": ("gross_ret", "mean"),
        "cost_ret": ("cost_ret", "mean"),
        "turnover": ("turnover", "mean"),
        "exposure": ("is_active", "mean"),
        "active_positions": ("is_active", "sum"),
        "universe_size": ("ticker", "size"),
    }

    if "regime" in frame.columns:
        aggregations["regime"] = ("regime", "first")
    if "strategy" in frame.columns:
        aggregations["strategy"] = ("strategy", "first")
    if "sector" in frame.columns:
        aggregations["sector"] = ("sector", "first")

    daily = frame.groupby("date", as_index=False).agg(**aggregations).sort_values("date")
    daily["equity"] = (1.0 + daily["portfolio_ret"]).cumprod()
    daily["drawdown"] = _compute_drawdown(daily["equity"])
    return daily.reset_index(drop=True)


def extract_trade_list(position_frame: pd.DataFrame) -> pd.DataFrame:
    _validate_position_frame(position_frame)

    frame = position_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["ret_1d"] = pd.to_numeric(frame["ret_1d"], errors="coerce").fillna(0.0)
    frame["position"] = pd.to_numeric(frame["position"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for ticker, g in frame.groupby("ticker", sort=False):
        g = g.copy().reset_index(drop=True)
        g["active"] = g["position"].shift(1).fillna(0.0).abs() > 0

        starts = g.index[g["active"] & ~g["active"].shift(1, fill_value=False)]
        ends = g.index[g["active"] & ~g["active"].shift(-1, fill_value=False)]

        for start_idx, end_idx in zip(starts.to_list(), ends.to_list(), strict=True):
            ret_slice = g.loc[start_idx:end_idx, "ret_1d"].fillna(0.0)
            trade_ret = float(np.prod(1.0 + ret_slice.to_numpy()) - 1.0)
            direction = float(np.sign(g.loc[start_idx, "position"].item()))

            row: dict[str, object] = {
                "ticker": ticker,
                "entry_date": g.loc[start_idx, "date"],
                "exit_date": g.loc[end_idx, "date"],
                "holding_days": int(end_idx - start_idx + 1),
                "trade_return": trade_ret,
                "direction": direction,
            }
            if "strategy" in g.columns:
                row["strategy"] = g.loc[start_idx, "strategy"]
            if "sector" in g.columns:
                row["sector"] = g.loc[start_idx, "sector"]
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker", "entry_date", "exit_date", "holding_days", "trade_return", "direction"])
    return pd.DataFrame(rows)


def compute_performance_metrics(daily: pd.DataFrame, trades: pd.DataFrame | None = None) -> dict[str, float]:
    if daily.empty:
        return {
            "n_days": 0,
            "total_return": np.nan,
            "cagr": np.nan,
            "annual_vol": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "win_rate_daily": np.nan,
            "avg_daily_return": np.nan,
            "avg_turnover": np.nan,
            "avg_exposure": np.nan,
            "trade_count": np.nan,
            "win_rate_trade": np.nan,
            "avg_trade_return": np.nan,
            "avg_holding_days": np.nan,
        }

    returns = pd.to_numeric(daily["portfolio_ret"], errors="coerce").fillna(0.0)
    n_days = int(len(daily))
    total_return = float((1.0 + returns).prod() - 1.0)

    if n_days > 0 and (1.0 + total_return) > 0:
        cagr = float((1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0)
    else:
        cagr = np.nan

    ann_vol = float(returns.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe = float(returns.mean() / returns.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR)) if ann_vol > 0 else np.nan
    max_drawdown = float(pd.to_numeric(daily["drawdown"], errors="coerce").min())
    win_rate_daily = float((returns > 0).mean())

    metrics = {
        "n_days": n_days,
        "total_return": total_return,
        "cagr": cagr,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate_daily": win_rate_daily,
        "avg_daily_return": float(returns.mean()),
        "avg_turnover": float(pd.to_numeric(daily["turnover"], errors="coerce").fillna(0.0).mean()),
        "avg_exposure": float(pd.to_numeric(daily["exposure"], errors="coerce").fillna(0.0).mean()),
        "trade_count": np.nan,
        "win_rate_trade": np.nan,
        "avg_trade_return": np.nan,
        "avg_holding_days": np.nan,
    }

    if trades is not None and not trades.empty:
        trade_returns = pd.to_numeric(trades["trade_return"], errors="coerce").dropna()
        holding_days = pd.to_numeric(trades["holding_days"], errors="coerce").dropna()
        metrics["trade_count"] = float(len(trades))
        metrics["win_rate_trade"] = float((trade_returns > 0).mean()) if not trade_returns.empty else np.nan
        metrics["avg_trade_return"] = float(trade_returns.mean()) if not trade_returns.empty else np.nan
        metrics["avg_holding_days"] = float(holding_days.mean()) if not holding_days.empty else np.nan

    return metrics


def run_backtest(
    position_frame: pd.DataFrame,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> BacktestOutputs:
    daily = build_daily_portfolio_returns(position_frame=position_frame, fee_bps=fee_bps, slippage_bps=slippage_bps)
    trades = extract_trade_list(position_frame=position_frame)
    metrics = compute_performance_metrics(daily=daily, trades=trades)
    return BacktestOutputs(daily=daily, trades=trades, metrics=metrics)
