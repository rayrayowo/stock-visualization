# Super White Brick Demo

This folder contains the extracted `超级白砖战法` demo artifacts built inside `trade_wizards_studio`.

## Included Here

- `events.csv`: raw signal-level event study table
- `summary_overall.csv`: top-line 5D event-study metrics
- `summary_by_gain_bucket.csv`: same-day gain bucket effect on 5D profitability
- `summary_by_market_cap_bucket.csv`: month-end market-cap bucket effect on 5D profitability
- `summary_by_year.csv`: yearly signal and return summary
- `gain_bucket_avg_return_5d.png`: chart for gain-bucket performance
- `gain_bucket_win_rate_5d.png`: chart for gain-bucket win rate
- `market_cap_avg_return_5d.png`: chart for market-cap performance
- `signal_count_by_year.png`: yearly signal count chart
- `report.html`: simple one-page report
- `metadata.json`: run metadata

## Core Results

- Signal count: `31,932`
- Unique symbols: `3,019`
- 5D average return: `0.1915%`
- 5D median return: `-0.2748%`
- 5D win rate: `46.64%`
- Average same-day gain: `2.4804%`

## Best Simple Read

- Gain buckets:
  - `0-1%` entry-day gain had the best 5D average return at about `0.2893%`
  - `2-3%` and `3-4%` were close behind at about `0.2297%` and `0.2315%`
- Market cap buckets:
  - `<20B` month-end market cap performed best at about `0.2672%`
  - `20-50B`, `100-200B`, and `200B+` all came out negative in this simple 5D hold test

## Where Earlier White-Brick Backtests Lived

- Historical daily-bar cache:
  - `/Users/rayzhang/clawd/shared/cache/tushare`
- Older large-scale overnight backtest:
  - `/Users/rayzhang/clawd/angel/scripts/super_white_brick_overnight.py`
- Older white-brick backtest:
  - `/Users/rayzhang/clawd/angel/scripts/backtest_super_white_brick.py`
- Earlier report outputs:
  - `/Users/rayzhang/clawd/angel/scripts/超级白砖战法-通宵回测报告.md`
  - `/Users/rayzhang/clawd/angel/scripts/超级白砖战法-通宵回测数据.json`
  - `/Users/rayzhang/clawd/shared/reports/clawteam/tushare-daily/`

## Suggested Placement

- Reusable strategy code:
  - `stock-visualization/trade_wizards_mvp/tw_mvp/super_white_brick.py`
- Re-run script:
  - `stock-visualization/trade_wizards_mvp/run_super_white_brick_demo.py`
- Derived month-end market cap table:
  - `stock-visualization/trade_wizards_mvp/data/derived/market_cap_month_end.csv`
- Demo artifacts:
  - `stock-visualization/trade_wizards_mvp/outputs/super_white_brick_demo/`

OpenClaw source workspaces are not the natural home for the trading code itself.
If a team-facing mirror is useful, keep only a light summary under:

- `clawd/shared/reports/clawteam/super-white-brick-demo/`
