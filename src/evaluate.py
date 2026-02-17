from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import compute_performance_metrics, run_backtest
from strategies import build_strategy_positions, list_strategies, parse_strategy_names


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = ROOT / "data" / "processed" / "master.parquet"
DEFAULT_OUT_DIR = ROOT / "data" / "processed"


def load_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Master file not found: {path}")

    if path.suffix.lower() == ".parquet":
        master = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        master = pd.read_csv(path)
    else:
        raise ValueError("Unsupported master format. Use .parquet or .csv")

    if "date" in master.columns:
        master["date"] = pd.to_datetime(master["date"], errors="coerce")
    return master


def infer_leader_topk(master: pd.DataFrame, fallback: int = 5) -> int:
    for col in master.columns:
        m = re.fullmatch(r"is_leader_top(\d+)", col)
        if m:
            return int(m.group(1))
    return fallback


def add_static_leader_flag(master: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if "sector" not in master.columns:
        raise ValueError("master must include 'sector' for leader analysis")
    if "dollar_volume" not in master.columns:
        raise ValueError("master must include 'dollar_volume' for leader analysis")

    frame = master.copy()
    frame["sector"] = frame["sector"].fillna("Unknown")
    frame["dollar_volume"] = pd.to_numeric(frame["dollar_volume"], errors="coerce").fillna(0.0)

    liquidity = (
        frame.groupby(["sector", "ticker"], as_index=False)["dollar_volume"]
        .mean()
        .rename(columns={"dollar_volume": "avg_dollar_volume"})
    )
    liquidity["liq_rank"] = liquidity.groupby("sector")["avg_dollar_volume"].rank(method="first", ascending=False)
    static_leaders = liquidity[liquidity["liq_rank"] <= top_k][["sector", "ticker"]].copy()
    static_leaders["is_static_leader"] = 1

    frame = frame.merge(static_leaders, on=["sector", "ticker"], how="left", validate="many_to_one")
    frame["is_static_leader"] = frame["is_static_leader"].fillna(0).astype("int8")
    return frame


def metrics_to_row(
    strategy: str,
    scope: str,
    segment: str,
    regime_segment: str,
    metrics: dict[str, float],
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "strategy": strategy,
        "scope": scope,
        "segment": segment,
        "regime_segment": regime_segment,
    }
    row.update(metrics)
    return row


def tag_frame(df: pd.DataFrame, strategy: str, scope: str, segment: str) -> pd.DataFrame:
    out = df.copy()
    out["strategy"] = strategy
    out["scope"] = scope
    out["segment"] = segment
    return out


def append_scope_evaluation(
    master_slice: pd.DataFrame,
    strategy_name: str,
    scope: str,
    segment: str,
    fee_bps: float,
    slippage_bps: float,
    summary_rows: list[dict[str, float | str]],
    daily_frames: list[pd.DataFrame],
    trade_frames: list[pd.DataFrame],
) -> None:
    if master_slice.empty:
        return

    positions = build_strategy_positions(master_slice, strategy_name=strategy_name)
    outputs = run_backtest(position_frame=positions, fee_bps=fee_bps, slippage_bps=slippage_bps)

    summary_rows.append(
        metrics_to_row(
            strategy=strategy_name,
            scope=scope,
            segment=segment,
            regime_segment="all",
            metrics=outputs.metrics,
        )
    )
    daily_frames.append(tag_frame(outputs.daily, strategy=strategy_name, scope=scope, segment=segment))
    if not outputs.trades.empty:
        trade_frames.append(tag_frame(outputs.trades, strategy=strategy_name, scope=scope, segment=segment))

    if "regime" in outputs.daily.columns:
        for regime_value, daily_regime in outputs.daily.groupby("regime", dropna=True):
            regime_metrics = compute_performance_metrics(daily=daily_regime, trades=None)
            summary_rows.append(
                metrics_to_row(
                    strategy=strategy_name,
                    scope=scope,
                    segment=segment,
                    regime_segment=str(regime_value),
                    metrics=regime_metrics,
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-strategy backtests and export summary/daily/trade outputs."
    )
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--strategies", default="ma_cross,macd_cross,kdj_cross")
    parser.add_argument("--fee-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--leader-topk", type=int, default=None)
    parser.add_argument("--summary-name", default="strategy_summary.csv")
    parser.add_argument("--daily-name", default="strategy_daily_returns.csv")
    parser.add_argument("--trades-name", default="strategy_trades.csv")
    parser.add_argument("--skip-sector", action="store_true")
    parser.add_argument("--skip-leader", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    master = load_master(args.master)
    strategies = parse_strategy_names(args.strategies)

    required = {"date", "ticker", "ret_1d", "regime", "sector", "dollar_volume"}
    missing = sorted(required.difference(master.columns))
    if missing:
        raise ValueError(f"master missing required columns: {missing}")

    leader_topk = args.leader_topk if args.leader_topk is not None else infer_leader_topk(master, fallback=5)
    master = add_static_leader_flag(master, top_k=leader_topk)

    summary_rows: list[dict[str, float | str]] = []
    daily_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []

    sectors = sorted(master["sector"].fillna("Unknown").unique().tolist())
    leader_master = master[master["is_static_leader"] == 1].copy()
    leader_counts = leader_master["ticker"].nunique()

    for strategy_name in strategies:
        append_scope_evaluation(
            master_slice=master,
            strategy_name=strategy_name,
            scope="overall",
            segment="all",
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            summary_rows=summary_rows,
            daily_frames=daily_frames,
            trade_frames=trade_frames,
        )

        if not args.skip_sector:
            for sector in sectors:
                sector_slice = master[master["sector"] == sector]
                append_scope_evaluation(
                    master_slice=sector_slice,
                    strategy_name=strategy_name,
                    scope="sector",
                    segment=str(sector),
                    fee_bps=args.fee_bps,
                    slippage_bps=args.slippage_bps,
                    summary_rows=summary_rows,
                    daily_frames=daily_frames,
                    trade_frames=trade_frames,
                )

        if not args.skip_leader and leader_counts > 0:
            append_scope_evaluation(
                master_slice=leader_master,
                strategy_name=strategy_name,
                scope="leaders_overall",
                segment=f"top{leader_topk}",
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                summary_rows=summary_rows,
                daily_frames=daily_frames,
                trade_frames=trade_frames,
            )

            if not args.skip_sector:
                for sector in sectors:
                    sector_leader_slice = leader_master[leader_master["sector"] == sector]
                    append_scope_evaluation(
                        master_slice=sector_leader_slice,
                        strategy_name=strategy_name,
                        scope="leaders_sector",
                        segment=str(sector),
                        fee_bps=args.fee_bps,
                        slippage_bps=args.slippage_bps,
                        summary_rows=summary_rows,
                        daily_frames=daily_frames,
                        trade_frames=trade_frames,
                    )

    summary_df = pd.DataFrame(summary_rows).sort_values(["strategy", "scope", "segment", "regime_segment"])
    daily_df = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    trades_df = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()

    summary_path = args.out_dir / args.summary_name
    daily_path = args.out_dir / args.daily_name
    trades_path = args.out_dir / args.trades_name

    summary_df.to_csv(summary_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    trades_df.to_csv(trades_path, index=False)

    print(f"[done] strategies: {strategies}")
    print(f"[done] leader_topk(static): {leader_topk}, leader_tickers: {leader_counts}")
    print(f"[done] summary rows: {len(summary_df):,}")
    print(f"[done] daily rows: {len(daily_df):,}")
    print(f"[done] trades rows: {len(trades_df):,}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {daily_path}")
    print(f"[saved] {trades_path}")


if __name__ == "__main__":
    print(f"[info] available strategies: {list_strategies()}")
    main()
