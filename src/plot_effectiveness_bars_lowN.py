from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "master.parquet"
OUT_DIR = ROOT / "outputs"
OUT_PATH = OUT_DIR / "fig_effectiveness_bars_5d_20d.png"
MPL_CACHE_DIR = OUT_DIR / "mplcache"

DEFAULT_MIN_N = 500
DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


SIGNAL_SPECS: list[tuple[str, str, int]] = [
    ("MA-golden", "ma_cross", 1),
    ("MA-dead", "ma_cross", -1),
    ("MACD-golden", "macd_cross", 1),
    ("MACD-dead", "macd_cross", -1),
    ("KDJ-golden", "kdj_cross", 1),
    ("KDJ-dead", "kdj_cross", -1),
]
REGIME_ORDER = ["bull", "bear"]
HORIZON_SPECS = [("5d", "fwd_ret_5d"), ("20d", "fwd_ret_20d")]


def extract_events(df: pd.DataFrame, cross_col: str, direction: int) -> pd.DataFrame:
    mask = pd.to_numeric(df[cross_col], errors="coerce").eq(direction)
    return df.loc[mask].copy()


def bootstrap_ci_mean(
    values: np.ndarray | pd.Series,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    chunk_size = 50
    done = 0
    while done < n_boot:
        step = min(chunk_size, n_boot - done)
        idx = rng.integers(0, n, size=(step, n))
        means[done : done + step] = arr[idx].mean(axis=1)
        done += step

    ci_low, ci_high = np.quantile(means, [0.025, 0.975])
    return float(ci_low), float(ci_high)


def load_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_parquet(path)
    required = {
        "date",
        "ticker",
        "close",
        "regime",
        "fwd_ret_5d",
        "fwd_ret_20d",
        "ma_cross",
        "macd_cross",
        "kdj_cross",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"master.parquet missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["regime"] = df["regime"].astype(str).str.lower().str.strip()
    df["fwd_ret_5d"] = pd.to_numeric(df["fwd_ret_5d"], errors="coerce")
    df["fwd_ret_20d"] = pd.to_numeric(df["fwd_ret_20d"], errors="coerce")
    for col in ["ma_cross", "macd_cross", "kdj_cross"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["regime"].isin(REGIME_ORDER)].copy()
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot regime effectiveness bars with bootstrap CI and low-N warnings."
    )
    parser.add_argument("--in-path", type=Path, default=IN_PATH)
    parser.add_argument("--out-path", type=Path, default=OUT_PATH)
    parser.add_argument("--min-n", type=int, default=DEFAULT_MIN_N)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def compute_summary(df: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    group_idx = 0

    for horizon_name, ret_col in HORIZON_SPECS:
        for signal_label, cross_col, direction in SIGNAL_SPECS:
            event_df = extract_events(df, cross_col=cross_col, direction=direction)
            for regime in REGIME_ORDER:
                values = (
                    event_df.loc[event_df["regime"] == regime, ret_col]
                    .dropna()
                    .to_numpy(dtype=float)
                )
                n = int(values.size)
                if n == 0:
                    mean = median = p25 = p75 = np.nan
                    ci_low = ci_high = np.nan
                else:
                    mean = float(np.mean(values))
                    median = float(np.median(values))
                    p25 = float(np.quantile(values, 0.25))
                    p75 = float(np.quantile(values, 0.75))
                    ci_low, ci_high = bootstrap_ci_mean(values, n_boot=n_boot, seed=seed + group_idx)

                rows.append(
                    {
                        "horizon": horizon_name,
                        "regime": regime,
                        "signal": signal_label,
                        "n": n,
                        "mean": mean,
                        "median": median,
                        "p25": p25,
                        "p75": p75,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
                group_idx += 1

    summary = pd.DataFrame(rows)
    return summary


def _text_offset(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.005
    span = float(finite.max() - finite.min())
    if span <= 0:
        return 0.005
    return max(0.0025, 0.04 * span)


def plot_grouped_bars(summary: pd.DataFrame, out_path: Path, min_n: int) -> None:
    signal_order = [s for s, _, _ in SIGNAL_SPECS]
    x = np.arange(len(signal_order), dtype=float)
    width = 0.36

    regime_colors = {"bull": "#4C78A8", "bear": "#E45756"}
    regime_offsets = {"bull": -width / 2.0, "bear": width / 2.0}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    for ax, (horizon_name, _) in zip(axes, HORIZON_SPECS, strict=True):
        panel = summary[summary["horizon"] == horizon_name].copy()
        panel = panel.set_index(["regime", "signal"])
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.9)

        panel_means = []
        for regime in REGIME_ORDER:
            means = []
            ns = []
            ci_lows = []
            ci_highs = []
            for signal in signal_order:
                if (regime, signal) in panel.index:
                    row = panel.loc[(regime, signal)]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    means.append(float(row["mean"]))
                    ns.append(int(row["n"]))
                    ci_lows.append(float(row["ci_low"]))
                    ci_highs.append(float(row["ci_high"]))
                else:
                    means.append(np.nan)
                    ns.append(0)
                    ci_lows.append(np.nan)
                    ci_highs.append(np.nan)

            means_arr = np.array(means, dtype=float)
            ns_arr = np.array(ns, dtype=int)
            lows_arr = np.array(ci_lows, dtype=float)
            highs_arr = np.array(ci_highs, dtype=float)
            xpos = x + regime_offsets[regime]

            bars = ax.bar(xpos, means_arr, width=width, label=regime.title(), color=regime_colors[regime])
            panel_means.append(means_arr)

            for bar, n_val in zip(bars, ns_arr, strict=True):
                bar.set_alpha(0.25 if n_val < min_n else 0.9)

            y_offset = _text_offset(means_arr[np.isfinite(means_arr)])
            for xi, m, ci_lo, ci_hi, n_val in zip(xpos, means_arr, lows_arr, highs_arr, ns_arr, strict=True):
                if np.isfinite(m) and np.isfinite(ci_lo) and np.isfinite(ci_hi):
                    lower = max(0.0, m - ci_lo)
                    upper = max(0.0, ci_hi - m)
                    ax.errorbar(
                        [xi],
                        [m],
                        yerr=[[lower], [upper]],
                        fmt="none",
                        ecolor="black",
                        elinewidth=1.0,
                        capsize=3,
                    )

                y = m if np.isfinite(m) else 0.0
                va = "bottom" if y >= 0 else "top"
                y_text = y + y_offset if y >= 0 else y - y_offset
                ax.text(xi, y_text, f"n={n_val}", ha="center", va=va, fontsize=7, color="black")
                if n_val < min_n:
                    low_n_y = y_text + y_offset * 0.65 if y >= 0 else y_text - y_offset * 0.65
                    ax.text(
                        xi,
                        low_n_y,
                        "LOW N",
                        ha="center",
                        va=va,
                        fontsize=6,
                        color="darkred",
                    )

        mean_vals = np.concatenate(panel_means) if panel_means else np.array([0.0])
        finite_means = mean_vals[np.isfinite(mean_vals)]
        if finite_means.size > 0:
            ymin = float(min(finite_means.min(), 0.0))
            ymax = float(max(finite_means.max(), 0.0))
            pad = max(0.01, 0.12 * (ymax - ymin if ymax > ymin else 0.05))
            ax.set_ylim(ymin - pad, ymax + pad * 1.8)

        ax.set_xticks(x, signal_order, rotation=20, ha="right")
        ax.set_xlabel("Signal type")
        ax.set_title(f"Horizon = {horizon_name}")
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

    axes[0].set_ylabel("Mean forward return")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=2, frameon=False)
    fig.suptitle("Cross-signal effectiveness by market regime", fontsize=14)
    fig.text(
        0.5,
        0.015,
        f"Faded bars: n < {min_n}; interpret cautiously.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.92])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.min_n <= 0:
        raise ValueError("--min-n must be > 0")
    if args.n_boot <= 0:
        raise ValueError("--n-boot must be > 0")

    df = load_master(args.in_path)
    summary = compute_summary(df=df, n_boot=args.n_boot, seed=args.seed)
    summary = summary[
        ["horizon", "regime", "signal", "n", "mean", "median", "p25", "p75", "ci_low", "ci_high"]
    ].sort_values(["horizon", "signal", "regime"]).reset_index(drop=True)

    low_n_rows = summary[summary["n"] < args.min_n][["horizon", "regime", "signal", "n"]]
    plot_grouped_bars(summary=summary, out_path=args.out_path, min_n=args.min_n)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))
    if low_n_rows.empty:
        print(f"\n[warning] No groups with n < {args.min_n}.")
    else:
        pairs = [
            f"({row.horizon}, {row.regime}, {row.signal}, n={int(row.n)})"
            for row in low_n_rows.itertuples(index=False)
        ]
        print(f"\n[warning] Low-N groups (n < {args.min_n}): " + ", ".join(pairs))
    print(f"[saved] {args.out_path}")


if __name__ == "__main__":
    main()
