from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VENV_SITE_PACKAGES = next((PROJECT_ROOT / ".venv" / "lib").glob("python*/site-packages"), None)

os.environ.setdefault(
    "MPLCONFIGDIR",
    str((PROJECT_ROOT / "outputs" / "mplcache").resolve()),
)

import pandas as pd

try:
    import matplotlib
except ModuleNotFoundError:
    if VENV_SITE_PACKAGES is not None and str(VENV_SITE_PACKAGES) not in sys.path:
        sys.path.append(str(VENV_SITE_PACKAGES))
    import matplotlib

matplotlib.use("Agg")

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:
    if VENV_SITE_PACKAGES is not None and str(VENV_SITE_PACKAGES) not in sys.path:
        sys.path.append(str(VENV_SITE_PACKAGES))
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from PIL import Image, ImageDraw, ImageFont

from tw_mvp.super_white_brick import (
    attach_market_cap,
    build_analysis_tables,
    build_super_white_brick_events,
    load_month_end_market_cap,
)

SHARED_TUSHARE_CACHE = Path("/Users/rayzhang/clawd/shared/cache/tushare")
MARKET_CAP_SOURCE = SHARED_TUSHARE_CACHE / "market_cap_daily_full.csv"
MONTH_END_MARKET_CAP_PATH = PROJECT_ROOT / "data" / "derived" / "market_cap_month_end.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "super_white_brick_demo"
VIDEO_FRAMES_DIR = OUTPUT_DIR / "video_frames"
DEFAULT_HORIZON_DAYS = 5


def _pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.2f}%"


def _pct_points(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.2f}%"


def _save_chart(frame: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: Path, y_as_pct: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_frame = frame.dropna(subset=[x_col, y_col]).copy()
    ax.bar(plot_frame[x_col].astype(str), plot_frame[y_col], color="#2C7BE5")
    ax.set_title(title)
    ax.set_xlabel("")
    if y_as_pct:
        ax.set_ylabel("Percent")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.1f}%"))
    else:
        ax.set_ylabel("Signals")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value):,}"))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _formatted_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if "return" in column or "win_rate" in column:
            display[column] = display[column].map(_pct)
        elif column.endswith("_pct"):
            display[column] = display[column].map(_pct_points)
    return display.to_html(index=False, classes="tw-table", border=0)


def _build_html_report(
    overall: pd.DataFrame,
    by_gain_bucket: pd.DataFrame,
    by_market_cap_bucket: pd.DataFrame,
    by_year: pd.DataFrame,
    output_dir: Path,
) -> Path:
    overall_row = overall.iloc[0].to_dict() if not overall.empty else {}
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Super White Brick Demo</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      margin: 32px;
      color: #111827;
      background: #f8fafc;
    }}
    h1, h2 {{
      margin-bottom: 12px;
    }}
    .hero {{
      background: white;
      border: 1px solid #dbe4f0;
      border-radius: 16px;
      padding: 20px 24px;
      margin-bottom: 24px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
      margin-top: 20px;
    }}
    .card {{
      background: white;
      border: 1px solid #dbe4f0;
      border-radius: 16px;
      padding: 16px;
    }}
    .chart {{
      background: white;
      border: 1px solid #dbe4f0;
      border-radius: 16px;
      padding: 16px;
      margin-bottom: 20px;
    }}
    .chart img {{
      width: 100%;
      border-radius: 12px;
    }}
    .tw-table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border-radius: 12px;
      overflow: hidden;
    }}
    .tw-table th, .tw-table td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 10px 12px;
      text-align: left;
      font-size: 14px;
    }}
    .tw-table th {{
      background: #eef4ff;
    }}
    .muted {{
      color: #6b7280;
    }}
  </style>
</head>
<body>
  <div class="hero">
    <h1>Super White Brick / 超级白砖</h1>
    <p class="muted">Simple event-study backtest inside Trade Wizards Studio.</p>
    <div class="grid">
      <div class="card">
        <strong>Signals / 信号数</strong>
        <div>{int(overall_row.get("signal_count", 0)):,}</div>
      </div>
      <div class="card">
        <strong>Unique Symbols / 股票数</strong>
        <div>{int(overall_row.get("unique_symbols", 0)):,}</div>
      </div>
      <div class="card">
        <strong>5D Win Rate / 5日胜率</strong>
        <div>{_pct(overall_row.get("win_rate"))}</div>
      </div>
      <div class="card">
        <strong>5D Avg Return / 5日平均收益</strong>
        <div>{_pct(overall_row.get("avg_return"))}</div>
      </div>
      <div class="card">
        <strong>5D Median Return / 5日中位收益</strong>
        <div>{_pct(overall_row.get("median_return"))}</div>
      </div>
      <div class="card">
        <strong>Entry Day Gain / 当日涨幅均值</strong>
        <div>{_pct_points(overall_row.get("avg_same_day_gain_pct"))}</div>
      </div>
    </div>
  </div>

  <div class="chart">
    <h2>Gain Bucket Effect / 涨幅分层</h2>
    <img src="gain_bucket_avg_return_5d.png" alt="Gain bucket average return">
    {_formatted_table(by_gain_bucket)}
  </div>

  <div class="chart">
    <h2>Market Cap Effect / 市值分层</h2>
    <img src="market_cap_avg_return_5d.png" alt="Market cap average return">
    {_formatted_table(by_market_cap_bucket)}
  </div>

  <div class="chart">
    <h2>Signal Count By Year / 年度信号数</h2>
    <img src="signal_count_by_year.png" alt="Signal count by year">
    {_formatted_table(by_year)}
  </div>
</body>
</html>
"""
    out_path = output_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                pass
    return ImageFont.load_default()


def _draw_wrapped(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font: ImageFont.ImageFont, fill: str, max_width: int, line_gap: int = 10) -> int:
    words = text.split()
    if not words:
        return xy[1]
    x, y = xy
    line = words[0]
    for word in words[1:]:
        trial = f"{line} {word}"
        bbox = draw.textbbox((x, y), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            line = trial
        else:
            draw.text((x, y), line, font=font, fill=fill)
            y += font.size + line_gap
            line = word
    draw.text((x, y), line, font=font, fill=fill)
    return y + font.size


def _frame_base(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (1600, 900), "#F4F7FB")
    draw = ImageDraw.Draw(img)
    title_font = _load_font(56)
    subtitle_font = _load_font(28)
    draw.rounded_rectangle((48, 48, 1552, 852), radius=28, outline="#D6E0EE", fill="white", width=3)
    draw.text((90, 82), title, font=title_font, fill="#0F172A")
    draw.text((92, 156), subtitle, font=subtitle_font, fill="#475569")
    return img, draw


def _save_video_frames(
    overall: pd.DataFrame,
    by_gain_bucket: pd.DataFrame,
    by_market_cap_bucket: pd.DataFrame,
    report_path: Path,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    body_font = _load_font(32)
    small_font = _load_font(24)

    overall_row = overall.iloc[0].to_dict() if not overall.empty else {}

    frames: list[Path] = []

    img, draw = _frame_base(
        "Super White Brick Demo",
        "Trade Wizards Studio / 超级白砖战法演示",
    )
    lines = [
        "1. Extract the strategy logic from stock-visualization.",
        "2. Rebuild it in studio with cached Tushare daily bars.",
        "3. Downsample market cap into month-end snapshots.",
        "4. Run a simple 5-day event-study backtest.",
    ]
    y = 260
    for line in lines:
        draw.text((110, y), line, font=body_font, fill="#111827")
        y += 74
    path = output_dir / "frame_01.png"
    img.save(path)
    frames.append(path)

    img, draw = _frame_base(
        "Strategy Rules",
        "Scanner version copied with the relaxed white-brick settings",
    )
    rules = [
        "White line above yellow line",
        "White brick is present",
        "Red brick length >= previous green brick length",
        "Upper+lower shadow to body ratio <= 0.4",
        "Same-day gain stays below 4%",
    ]
    y = 250
    for rule in rules:
        draw.text((110, y), f"- {rule}", font=body_font, fill="#111827")
        y += 72
    path = output_dir / "frame_02.png"
    img.save(path)
    frames.append(path)

    img, draw = _frame_base(
        "Backtest Snapshot",
        "Simple close-to-close event study after each signal",
    )
    metrics = [
        f"Signals: {int(overall_row.get('signal_count', 0)):,}",
        f"Unique symbols: {int(overall_row.get('unique_symbols', 0)):,}",
        f"5D avg return: {_pct(overall_row.get('avg_return'))}",
        f"5D win rate: {_pct(overall_row.get('win_rate'))}",
        f"Average same-day gain: {_pct_points(overall_row.get('avg_same_day_gain_pct'))}",
    ]
    y = 250
    for metric in metrics:
        draw.text((110, y), metric, font=body_font, fill="#111827")
        y += 72
    path = output_dir / "frame_03.png"
    img.save(path)
    frames.append(path)

    gain_chart = Image.open(OUTPUT_DIR / "gain_bucket_avg_return_5d.png").convert("RGB")
    market_chart = Image.open(OUTPUT_DIR / "market_cap_avg_return_5d.png").convert("RGB")
    gain_chart.thumbnail((680, 420))
    market_chart.thumbnail((680, 420))

    img, draw = _frame_base(
        "Gain Bucket Effect",
        "How entry-day price change changes profitability",
    )
    img.paste(gain_chart, (110, 250))
    summary_line = "Best bucket by 5D average return: "
    if not by_gain_bucket.empty:
        best_row = by_gain_bucket.sort_values("avg_return", ascending=False).iloc[0]
        summary_line += f"{best_row['gain_bucket']} ({_pct(best_row['avg_return'])})"
    else:
        summary_line += "N/A"
    _draw_wrapped(draw, summary_line, (860, 320), body_font, "#111827", 580)
    path = output_dir / "frame_04.png"
    img.save(path)
    frames.append(path)

    img, draw = _frame_base(
        "Market Cap Effect",
        "Month-end market cap is enough for this layer",
    )
    img.paste(market_chart, (110, 250))
    summary_line = "Best market-cap bucket by 5D average return: "
    if not by_market_cap_bucket.empty:
        best_row = by_market_cap_bucket.sort_values("avg_return", ascending=False).iloc[0]
        summary_line += f"{best_row['market_cap_bucket']} ({_pct(best_row['avg_return'])})"
    else:
        summary_line += "N/A"
    _draw_wrapped(draw, summary_line, (860, 320), body_font, "#111827", 580)
    path = output_dir / "frame_05.png"
    img.save(path)
    frames.append(path)

    img, draw = _frame_base(
        "Artifacts Saved",
        "Open the report, tables, and video inside outputs/super_white_brick_demo",
    )
    outputs = [
        str(report_path),
        str(OUTPUT_DIR / "events.csv"),
        str(OUTPUT_DIR / "summary_by_gain_bucket.csv"),
        str(OUTPUT_DIR / "summary_by_market_cap_bucket.csv"),
        str(OUTPUT_DIR / "super_white_brick_demo.mp4"),
    ]
    y = 240
    for line in outputs:
        y = _draw_wrapped(draw, line, (100, y), small_font, "#111827", 1360, line_gap=8) + 18
    path = output_dir / "frame_06.png"
    img.save(path)
    frames.append(path)
    return frames


def _render_video(frames_dir: Path, out_path: Path) -> None:
    cmd = [
        "/opt/homebrew/bin/ffmpeg",
        "-y",
        "-framerate",
        "1/2",
        "-i",
        str(frames_dir / "frame_%02d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] building month-end market cap table")
    month_end_market_cap = load_month_end_market_cap(
        source_path=MARKET_CAP_SOURCE,
        output_path=MONTH_END_MARKET_CAP_PATH,
    )

    print("[2/6] scanning cached Tushare daily bars for Super White Brick events")
    events = build_super_white_brick_events(cache_dir=SHARED_TUSHARE_CACHE)
    events = attach_market_cap(events, month_end_market_cap)

    print("[3/6] writing raw analysis tables")
    tables = build_analysis_tables(events, default_horizon=DEFAULT_HORIZON_DAYS)
    events.to_csv(OUTPUT_DIR / "events.csv", index=False)
    for name, frame in tables.items():
        frame.to_csv(OUTPUT_DIR / f"summary_{name}.csv", index=False)

    metadata = {
        "strategy": "super_white_brick",
        "default_horizon_days": DEFAULT_HORIZON_DAYS,
        "event_count": int(len(events)),
        "unique_symbols": int(events["ts_code"].nunique()) if not events.empty else 0,
        "market_cap_source": str(MARKET_CAP_SOURCE),
        "month_end_market_cap_path": str(MONTH_END_MARKET_CAP_PATH),
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[4/6] rendering charts")
    _save_chart(
        tables["by_gain_bucket"],
        "gain_bucket",
        "avg_return",
        "Super White Brick: average 5D return by entry-day gain bucket",
        OUTPUT_DIR / "gain_bucket_avg_return_5d.png",
    )
    _save_chart(
        tables["by_gain_bucket"],
        "gain_bucket",
        "win_rate",
        "Super White Brick: 5D win rate by entry-day gain bucket",
        OUTPUT_DIR / "gain_bucket_win_rate_5d.png",
    )
    _save_chart(
        tables["by_market_cap_bucket"],
        "market_cap_bucket",
        "avg_return",
        "Super White Brick: average 5D return by month-end market cap bucket",
        OUTPUT_DIR / "market_cap_avg_return_5d.png",
    )
    _save_chart(
        tables["by_year"],
        "signal_year",
        "signal_count",
        "Super White Brick: signal count by year",
        OUTPUT_DIR / "signal_count_by_year.png",
        y_as_pct=False,
    )

    print("[5/6] building html report")
    report_path = _build_html_report(
        overall=tables["overall"],
        by_gain_bucket=tables["by_gain_bucket"],
        by_market_cap_bucket=tables["by_market_cap_bucket"],
        by_year=tables["by_year"],
        output_dir=OUTPUT_DIR,
    )

    print("[6/6] building demo video")
    _save_video_frames(
        overall=tables["overall"],
        by_gain_bucket=tables["by_gain_bucket"],
        by_market_cap_bucket=tables["by_market_cap_bucket"],
        report_path=report_path,
        output_dir=VIDEO_FRAMES_DIR,
    )
    _render_video(VIDEO_FRAMES_DIR, OUTPUT_DIR / "super_white_brick_demo.mp4")

    print(f"[saved] {MONTH_END_MARKET_CAP_PATH}")
    print(f"[saved] {OUTPUT_DIR / 'events.csv'}")
    print(f"[saved] {OUTPUT_DIR / 'summary_overall.csv'}")
    print(f"[saved] {report_path}")
    print(f"[saved] {OUTPUT_DIR / 'super_white_brick_demo.mp4'}")


if __name__ == "__main__":
    main()
