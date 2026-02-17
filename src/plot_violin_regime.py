from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "master.parquet"
OUT_DIR = ROOT / "outputs"
OUT_PATH = OUT_DIR / "fig_violin_regime_fwd20.png"
MPL_DIR = ROOT / ".mplconfig"

MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def load_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_parquet(path)
    required = {"date", "ticker"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"master.parquet is missing required columns: {sorted(missing)}")

    if "regime" not in df.columns:
        raise ValueError(
            "Column 'regime' not found in master.parquet. "
            "Please run src/clean_merge01.py first to generate regime labels."
        )

    if "fwd_ret_20d" not in df.columns:
        if "close" not in df.columns:
            raise ValueError(
                "Column 'fwd_ret_20d' is missing and 'close' is unavailable to compute it."
            )
        frame = df.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
        frame["fwd_ret_20d"] = frame.groupby("ticker", sort=False)["close"].shift(-20) / frame["close"] - 1.0
        return frame

    return df


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("regime")["fwd_ret_20d"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
        )
        .reset_index()
    )
    return summary


def plot_violin(df: pd.DataFrame, output_path: Path) -> None:
    plot_df = df[df["regime"].isin(["bull", "bear"])].copy()
    plot_df = plot_df.dropna(subset=["fwd_ret_20d"])

    bull_vals = plot_df.loc[plot_df["regime"] == "bull", "fwd_ret_20d"]
    bear_vals = plot_df.loc[plot_df["regime"] == "bear", "fwd_ret_20d"]

    if bull_vals.empty or bear_vals.empty:
        raise ValueError("Both bull and bear samples are required to plot the violin chart.")

    clip_lo = plot_df["fwd_ret_20d"].quantile(0.01)
    clip_hi = plot_df["fwd_ret_20d"].quantile(0.99)
    bull_clip = bull_vals.clip(lower=clip_lo, upper=clip_hi)
    bear_clip = bear_vals.clip(lower=clip_lo, upper=clip_hi)

    fig, ax = plt.subplots(figsize=(9, 6))
    parts = ax.violinplot(
        [bull_clip.values, bear_clip.values],
        positions=[1, 2],
        widths=0.8,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    for idx, body in enumerate(parts["bodies"]):
        body.set_alpha(0.75)
        body.set_edgecolor("black")
        body.set_linewidth(0.8)
        body.set_facecolor("#4C78A8" if idx == 0 else "#E45756")

    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.9)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Bull", "Bear"])
    ax.set_ylabel("20-day forward return")
    ax.set_xlabel("Market regime")
    ax.set_title("Forward 20-day returns by market regime (bull vs bear)")
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

    fig.text(0.5, 0.02, "Returns clipped to 1-99% for display.", ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    df = load_master(IN_PATH)
    df["regime"] = df["regime"].astype(str).str.lower().str.strip()
    df["fwd_ret_20d"] = pd.to_numeric(df["fwd_ret_20d"], errors="coerce")

    summary_df = summary_table(df[df["regime"].isin(["bull", "bear"])].dropna(subset=["fwd_ret_20d"]))
    if summary_df.empty:
        raise ValueError("No valid bull/bear rows with fwd_ret_20d found for summary and plot.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_violin(df=df, output_path=OUT_PATH)

    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.6f}"))
    print(f"[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
