from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "processed" / "master.parquet"
OUT_DIR = ROOT / "outputs"
MPL_CACHE_DIR = OUT_DIR / "mplcache"

OUT_EVENTS = OUT_DIR / "powerbi_signal_events.csv"
OUT_SUMMARY = OUT_DIR / "powerbi_signal_summary.csv"
OUT_SUMMARY_SECTOR = OUT_DIR / "powerbi_signal_summary_by_sector.csv"
OUT_FIG = OUT_DIR / "fig_signal_regime_compare.png"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


def load_master(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_parquet(path)

    if "regime" not in df.columns:
        raise ValueError(
            "Column 'regime' is missing in master.parquet. "
            "Please run src/clean_merge01.py first."
        )

    required = {
        "date",
        "ticker",
        "close",
        "sector",
        "dollar_volume",
        "ma_cross",
        "macd_cross",
        "kdj_cross",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"master.parquet is missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["dollar_volume"] = pd.to_numeric(df["dollar_volume"], errors="coerce")
    df["sector"] = df["sector"].fillna("Unknown")
    df["regime"] = df["regime"].astype(str).str.lower().str.strip()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    for cross_col in ["ma_cross", "macd_cross", "kdj_cross"]:
        df[cross_col] = pd.to_numeric(df[cross_col], errors="coerce")

    if "fwd_ret_5d" not in df.columns:
        df["fwd_ret_5d"] = df.groupby("ticker", sort=False)["close"].shift(-5) / df["close"] - 1.0
    else:
        df["fwd_ret_5d"] = pd.to_numeric(df["fwd_ret_5d"], errors="coerce")

    if "fwd_ret_20d" not in df.columns:
        df["fwd_ret_20d"] = df.groupby("ticker", sort=False)["close"].shift(-20) / df["close"] - 1.0
    else:
        df["fwd_ret_20d"] = pd.to_numeric(df["fwd_ret_20d"], errors="coerce")

    return df


def build_event_table(master: pd.DataFrame) -> pd.DataFrame:
    indicator_map = {
        "MA": "ma_cross",
        "MACD": "macd_cross",
        "KDJ": "kdj_cross",
    }
    base_cols = [
        "date",
        "ticker",
        "regime",
        "sector",
        "dollar_volume",
        "fwd_ret_5d",
        "fwd_ret_20d",
    ]

    parts: list[pd.DataFrame] = []
    for indicator, cross_col in indicator_map.items():
        cross = master[cross_col]
        mask = cross.isin([1, -1])
        if not mask.any():
            continue

        part = master.loc[mask, base_cols].copy()
        part["indicator"] = indicator
        part["signal"] = np.where(cross[mask] == 1, "golden", "dead")
        parts.append(part)

    if not parts:
        raise ValueError("No signal events found in ma_cross/macd_cross/kdj_cross (no ±1 rows).")

    events = pd.concat(parts, ignore_index=True)
    events = events[events["regime"].isin(["bull", "bear"])].copy()
    events = events[
        [
            "date",
            "ticker",
            "indicator",
            "signal",
            "regime",
            "sector",
            "dollar_volume",
            "fwd_ret_5d",
            "fwd_ret_20d",
        ]
    ].sort_values(["date", "ticker", "indicator"]).reset_index(drop=True)
    return events


def summarize_metrics(grouped: pd.core.groupby.SeriesGroupBy) -> pd.DataFrame:
    return (
        grouped.agg(
            n="count",
            mean="mean",
            median="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            win_rate=lambda s: (s > 0).mean(),
        )
        .reset_index()
    )


def build_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for h_col, horizon in [("fwd_ret_5d", "5d"), ("fwd_ret_20d", "20d")]:
        grouped = events.groupby(["indicator", "signal", "regime"], dropna=False)[h_col]
        summary_h = summarize_metrics(grouped)
        summary_h["horizon"] = horizon
        rows.append(summary_h)

    summary = pd.concat(rows, ignore_index=True)
    summary["n"] = summary["n"].astype("int64")
    summary = summary[
        ["indicator", "signal", "regime", "horizon", "n", "mean", "median", "p25", "p75", "win_rate"]
    ].sort_values(["horizon", "indicator", "signal", "regime"])
    return summary.reset_index(drop=True)


def build_summary_by_sector(events: pd.DataFrame) -> pd.DataFrame:
    grouped = events.groupby(["indicator", "signal", "regime", "sector"], dropna=False)["fwd_ret_20d"]
    out = summarize_metrics(grouped)
    out["horizon"] = "20d"
    out["n"] = out["n"].astype("int64")
    out = out[
        ["indicator", "signal", "regime", "sector", "horizon", "n", "mean", "median", "p25", "p75", "win_rate"]
    ].sort_values(["indicator", "signal", "regime", "sector"])
    return out.reset_index(drop=True)


def plot_compare(summary: pd.DataFrame, out_path: Path) -> None:
    regime_order = ["bull", "bear"]
    combo_order = [
        ("MA", "golden"),
        ("MA", "dead"),
        ("MACD", "golden"),
        ("MACD", "dead"),
        ("KDJ", "golden"),
        ("KDJ", "dead"),
    ]
    combo_order = [
        combo
        for combo in combo_order
        if ((summary["indicator"] == combo[0]) & (summary["signal"] == combo[1])).any()
    ]

    if not combo_order:
        raise ValueError("No summary rows available for plotting.")

    colors = {"MA": "#4C78A8", "MACD": "#F58518", "KDJ": "#54A24B"}
    markers = {"golden": "o", "dead": "X"}
    offsets = np.linspace(-0.26, 0.26, num=len(combo_order))
    x_base = {reg: idx for idx, reg in enumerate(regime_order)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    handles: dict[str, plt.Line2D] = {}

    for ax, horizon in zip(axes, ["5d", "20d"], strict=True):
        panel = summary[summary["horizon"] == horizon]
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.9)

        for idx, (indicator, signal) in enumerate(combo_order):
            sub = panel[(panel["indicator"] == indicator) & (panel["signal"] == signal)]
            if sub.empty:
                continue

            xs: list[float] = []
            ys: list[float] = []
            ns: list[tuple[float, float, int]] = []
            for reg in regime_order:
                row = sub[sub["regime"] == reg]
                if row.empty:
                    continue
                x = x_base[reg] + offsets[idx]
                y = float(row["mean"].iloc[0])
                n = int(row["n"].iloc[0])
                xs.append(x)
                ys.append(y)
                ns.append((x, y, n))

            if not xs:
                continue

            label = f"{indicator}-{signal}"
            (line,) = ax.plot(
                xs,
                ys,
                color=colors.get(indicator, "#333333"),
                marker=markers.get(signal, "o"),
                linewidth=1.2,
                markersize=6,
                label=label,
            )
            if label not in handles:
                handles[label] = line

            for x, y, n in ns:
                ax.annotate(
                    f"n={n}",
                    xy=(x, y),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color="dimgray",
                )

        ax.set_xticks([x_base["bull"], x_base["bear"]], ["Bull", "Bear"])
        ax.set_xlabel("Market regime")
        ax.set_title(f"Mean forward return ({horizon})")
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.6)

    axes[0].set_ylabel("Mean forward return")
    fig.suptitle("Indicator cross-signal effectiveness by bull vs bear regime", fontsize=13)
    if handles:
        fig.legend(
            handles.values(),
            handles.keys(),
            loc="upper center",
            ncol=min(6, len(handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.97),
        )

    fig.tight_layout(rect=[0, 0.0, 1, 0.9])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def print_preview(name: str, df: pd.DataFrame, rows: int = 8) -> None:
    print(f"\n[preview] {name}")
    print(df.head(rows).to_string(index=False))


def main() -> None:
    master = load_master(IN_PATH)
    events = build_event_table(master)
    summary = build_summary(events)
    summary_sector = build_summary_by_sector(events)

    events.to_csv(OUT_EVENTS, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)
    summary_sector.to_csv(OUT_SUMMARY_SECTOR, index=False)
    plot_compare(summary=summary, out_path=OUT_FIG)

    print_preview("powerbi_signal_events.csv", events)
    print_preview("powerbi_signal_summary.csv", summary)
    print_preview("powerbi_signal_summary_by_sector.csv", summary_sector)

    print(f"\n[saved] {OUT_EVENTS}")
    print(f"[saved] {OUT_SUMMARY}")
    print(f"[saved] {OUT_SUMMARY_SECTOR}")
    print(f"[saved] {OUT_FIG}")


if __name__ == "__main__":
    main()
