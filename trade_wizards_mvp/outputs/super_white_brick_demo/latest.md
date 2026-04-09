# 超级白砖 Demo

这是一份从 `stock-visualization` 里的超级白砖思路提取出来，在 `trade_wizards_studio` 内跑出的简化回测包。

## 这次包含什么

- `events.csv`: 逐信号事件表
- `summary_overall.csv`: 5 日持有的总览指标
- `summary_by_gain_bucket.csv`: 按当日涨幅分层后的收益表现
- `summary_by_market_cap_bucket.csv`: 按月末市值分层后的收益表现
- `summary_by_year.csv`: 年度信号数与收益摘要
- `report.html`: 一页式可视化报告
- `*.png`: 对应图表
- `metadata.json`: 运行元数据

## 简单结论

- 信号总数: `31,932`
- 覆盖股票数: `3,019`
- 5 日平均收益: `0.1915%`
- 5 日中位收益: `-0.2748%`
- 5 日胜率: `46.64%`
- 当日平均涨幅: `2.4804%`

表现最好的两个简单分层:

- 当日涨幅 `0-1%` 这一组，5 日平均收益最高，约 `0.2893%`
- 月末市值 `<20B` 这一组，5 日平均收益最高，约 `0.2672%`

## 老回测和提数之前在哪里

- 历史日线缓存:
  - `/Users/rayzhang/clawd/shared/cache/tushare`
- 旧版超级白砖回测:
  - `/Users/rayzhang/clawd/angel/scripts/backtest_super_white_brick.py`
- 旧版通宵全量回测:
  - `/Users/rayzhang/clawd/angel/scripts/super_white_brick_overnight.py`
- 旧版报告:
  - `/Users/rayzhang/clawd/angel/scripts/超级白砖战法-通宵回测报告.md`
  - `/Users/rayzhang/clawd/angel/scripts/超级白砖战法-通宵回测数据.json`

## 建议归位

- 主代码和完整产物:
  - `/Users/rayzhang/Downloads/stock-visualization/trade_wizards_mvp`
- 团队侧轻量镜像:
  - `/Users/rayzhang/clawd/shared/reports/clawteam/super-white-brick-demo`

`mp4` 没有一起归档，其余文件可以保留在报告目录或 `trade_wizards_mvp/outputs/super_white_brick_demo/` 下。
