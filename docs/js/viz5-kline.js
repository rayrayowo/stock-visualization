/**
 * Viz 5: AAPL K-line with B1 Trade Entry/Exit Markers
 * TradingView Lightweight Charts v4
 * Shows actual buy/sell points, exit reasons, and fly triggers.
 * OHLCV data panel follows cursor like Futu NiuNiu.
 * KDJ(9,3,3) sub-panel below volume.
 */
(function () {
  "use strict";

  var container = document.getElementById("viz5-container");
  if (!container) return;

  /* ── Exit reason styling ── */
  var EXIT_COLORS = {
    entry:     "#22c55e",
    stop_loss: "#ef4444",
    fly_stop:  "#f59e0b",
    s1_sell:   "#3b82f6",
    end_of_data: "#6b7280"
  };
  var EXIT_LABELS = {
    entry:     "B1 Buy",
    stop_loss: "Stop-Loss",
    fly_stop:  "Fly Stop",
    s1_sell:   "S1 Sell",
    end_of_data: "End"
  };
  var EXIT_SHAPES = {
    entry:     "circle",
    stop_loss: "square",
    fly_stop:  "arrowDown",
    s1_sell:   "arrowDown",
    end_of_data: "square"
  };

  /* ── Container setup ── */
  container.style.position = "relative";
  container.style.minHeight = "560px";

  /* ── Legend bar ── */
  var legendBar = document.createElement("div");
  legendBar.style.cssText =
    "display:flex;flex-wrap:wrap;gap:12px;align-items:center;" +
    "padding:8px 12px;margin-bottom:4px;font-family:DM Sans,Outfit,sans-serif;" +
    "font-size:13px;color:#374151;user-select:none;";

  var legendLabel = document.createElement("span");
  legendLabel.style.cssText = "font-weight:600;margin-right:2px;";
  legendLabel.textContent = "Markers:";
  legendBar.appendChild(legendLabel);

  ["entry", "stop_loss", "fly_stop", "s1_sell"].forEach(function (key) {
    var item = document.createElement("span");
    item.style.cssText = "display:inline-flex;align-items:center;gap:4px;";
    var dot = document.createElement("span");
    dot.style.cssText =
      "display:inline-block;width:10px;height:10px;" +
      (key === "entry" ? "border-radius:50%;" : "border-radius:2px;") +
      "background:" + EXIT_COLORS[key] + ";flex-shrink:0;";
    var txt = document.createElement("span");
    txt.textContent = EXIT_LABELS[key];
    item.appendChild(dot);
    item.appendChild(txt);
    legendBar.appendChild(item);
  });

  container.parentNode.insertBefore(legendBar, container);

  /* ── OHLCV + KDJ data panel (Futu style) ── */
  var dataPanel = document.createElement("div");
  dataPanel.style.cssText =
    "position:absolute;top:8px;right:12px;z-index:20;pointer-events:none;" +
    "font-family:JetBrains Mono,Consolas,monospace;font-size:12px;" +
    "line-height:1.6;color:#374151;text-align:right;min-width:180px;";
  dataPanel.innerHTML = "&nbsp;";

  /* ── Chart div (candlestick + volume) ── */
  var chartDiv = document.createElement("div");
  chartDiv.style.cssText = "width:100%;height:400px;position:relative;";
  chartDiv.setAttribute("role", "img");
  chartDiv.setAttribute("aria-label", "AAPL daily candlestick chart with B1 trade entry and exit markers, volume histogram below");
  container.appendChild(chartDiv);
  chartDiv.appendChild(dataPanel);

  /* ── KDJ chart div ── */
  var kdjDiv = document.createElement("div");
  kdjDiv.style.cssText = "width:100%;height:140px;position:relative;";
  kdjDiv.setAttribute("role", "img");
  kdjDiv.setAttribute("aria-label", "KDJ(9,3,3) indicator chart showing K, D, and J lines with overbought and oversold reference levels");
  container.appendChild(kdjDiv);

  /* ── KDJ data panel ── */
  var kdjPanel = document.createElement("div");
  kdjPanel.style.cssText =
    "position:absolute;top:4px;left:12px;z-index:20;pointer-events:none;" +
    "font-family:JetBrains Mono,Consolas,monospace;font-size:11px;" +
    "line-height:1.5;color:#374151;";
  kdjPanel.innerHTML = "KDJ(9,3,3)";
  kdjDiv.appendChild(kdjPanel);

  /* ── Fetch data and render ── */
  fetch("data/kline.json")
    .then(function (r) { return r.json(); })
    .then(function (data) {
      var ticker = data.ticker;
      var ohlcv = data.ohlcv;
      var markers = data.markers;
      var trades = data.trades || [];

      /* ── Shared time scale options ── */
      var timeScaleOpts = {
        borderColor: "#e5e7eb",
        timeVisible: false,
        secondsVisible: false
      };

      /* ── Create main chart (candles + volume) ── */
      var chart = LightweightCharts.createChart(chartDiv, {
        width: chartDiv.clientWidth,
        height: 400,
        layout: {
          background: { type: "solid", color: "#ffffff" },
          textColor: "#374151",
          fontFamily: "DM Sans, Outfit, sans-serif",
          fontSize: 12
        },
        grid: {
          vertLines: { color: "#f5f5f5" },
          horzLines: { color: "#f5f5f5" }
        },
        crosshair: {
          mode: LightweightCharts.CrosshairMode.Normal,
          vertLine: { labelBackgroundColor: "#1a1a2e" },
          horzLine: { labelBackgroundColor: "#1a1a2e" }
        },
        rightPriceScale: { borderColor: "#e5e7eb" },
        timeScale: timeScaleOpts
      });

      /* ── Create KDJ chart ── */
      var kdjChart = LightweightCharts.createChart(kdjDiv, {
        width: kdjDiv.clientWidth,
        height: 140,
        layout: {
          background: { type: "solid", color: "#ffffff" },
          textColor: "#374151",
          fontFamily: "DM Sans, Outfit, sans-serif",
          fontSize: 11
        },
        grid: {
          vertLines: { color: "#f5f5f5" },
          horzLines: { color: "#f9f9f9" }
        },
        crosshair: {
          mode: LightweightCharts.CrosshairMode.Normal,
          vertLine: { labelBackgroundColor: "#1a1a2e" },
          horzLine: { labelBackgroundColor: "#1a1a2e" }
        },
        rightPriceScale: {
          borderColor: "#e5e7eb",
          scaleMargins: { top: 0.08, bottom: 0.08 }
        },
        timeScale: timeScaleOpts
      });

      /* ── Candlestick series ── */
      var candleSeries = chart.addCandlestickSeries({
        upColor: "#22c55e",
        downColor: "#ef4444",
        borderUpColor: "#16a34a",
        borderDownColor: "#dc2626",
        wickUpColor: "#22c55e",
        wickDownColor: "#ef4444"
      });

      var candleData = ohlcv.map(function (d) {
        return { time: d.date, open: d.open, high: d.high, low: d.low, close: d.close };
      });
      candleSeries.setData(candleData);

      /* ── Volume histogram ── */
      var volumeSeries = chart.addHistogramSeries({
        priceFormat: { type: "volume" },
        priceScaleId: "volume"
      });
      chart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.82, bottom: 0 }
      });
      var volumeData = ohlcv.map(function (d) {
        return {
          time: d.date,
          value: d.volume,
          color: d.close >= d.open ? "rgba(34,197,94,0.2)" : "rgba(239,68,68,0.2)"
        };
      });
      volumeSeries.setData(volumeData);

      /* ── KDJ line series ── */
      var kSeries = kdjChart.addLineSeries({
        color: "#3b82f6",
        lineWidth: 1.5,
        priceLineVisible: false,
        lastValueVisible: false,
        title: ""
      });
      var dSeries = kdjChart.addLineSeries({
        color: "#f59e0b",
        lineWidth: 1.5,
        priceLineVisible: false,
        lastValueVisible: false,
        title: ""
      });
      var jSeries = kdjChart.addLineSeries({
        color: "#a855f7",
        lineWidth: 1.5,
        priceLineVisible: false,
        lastValueVisible: false,
        title: ""
      });

      var kData = [], dData = [], jData = [];
      ohlcv.forEach(function (d) {
        kData.push({ time: d.date, value: d.k });
        dData.push({ time: d.date, value: d.d });
        jData.push({ time: d.date, value: d.j });
      });
      kSeries.setData(kData);
      dSeries.setData(dData);
      jSeries.setData(jData);

      /* ── Overbought/oversold reference lines on KDJ ── */
      var refLineOpts = {
        price: 0,
        color: "#d1d5db",
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: false
      };
      kSeries.createPriceLine(Object.assign({}, refLineOpts, { price: 80, title: "80" }));
      kSeries.createPriceLine(Object.assign({}, refLineOpts, { price: 20, title: "20" }));

      /* ── Sync time scales ── */
      chart.timeScale().subscribeVisibleLogicalRangeChange(function (range) {
        if (range) kdjChart.timeScale().setVisibleLogicalRange(range);
      });
      kdjChart.timeScale().subscribeVisibleLogicalRangeChange(function (range) {
        if (range) chart.timeScale().setVisibleLogicalRange(range);
      });

      /* ── Build OHLCV lookup for data panel ── */
      var ohlcvMap = {};
      ohlcv.forEach(function (d) { ohlcvMap[d.date] = d; });
      var dateSet = {};
      ohlcv.forEach(function (d) { dateSet[d.date] = true; });

      /* ── Build trade lookup by date ── */
      var tradeByEntry = {};
      var tradeByExit = {};
      trades.forEach(function (t) {
        tradeByEntry[t.entry_date] = t;
        tradeByExit[t.exit_date] = t;
      });

      /* ── Trade markers ── */
      var chartMarkers = [];
      markers.forEach(function (m) {
        if (!dateSet[m.date]) return;
        var isEntry = m.type === "entry";
        var key = isEntry ? "entry" : m.signal;
        var color = EXIT_COLORS[key] || "#6b7280";
        var shape = EXIT_SHAPES[key] || "square";
        var label = "";
        if (isEntry) {
          label = "Buy";
        } else if (m.signal === "stop_loss") {
          label = "SL";
        } else if (m.signal === "fly_stop") {
          label = "Fly";
        } else if (m.signal === "s1_sell") {
          label = "S1";
        } else {
          label = "End";
        }

        chartMarkers.push({
          time: m.date,
          position: isEntry ? "belowBar" : "aboveBar",
          color: color,
          shape: shape,
          text: label,
          size: 2
        });
      });

      chartMarkers.sort(function (a, b) {
        return a.time < b.time ? -1 : a.time > b.time ? 1 : 0;
      });
      candleSeries.setMarkers(chartMarkers);

      /* ── Crosshair data panel (Futu style) ── */
      function formatVol(v) {
        if (v >= 1e9) return (v / 1e9).toFixed(1) + "B";
        if (v >= 1e6) return (v / 1e6).toFixed(1) + "M";
        if (v >= 1e3) return (v / 1e3).toFixed(0) + "K";
        return v.toString();
      }

      function updateDataPanel(dateStr) {
        var d = ohlcvMap[dateStr];
        if (!d) { dataPanel.innerHTML = "&nbsp;"; return; }

        var chg = d.close - d.open;
        var chgPct = ((chg / d.open) * 100).toFixed(2);
        var chgColor = chg >= 0 ? "#22c55e" : "#ef4444";
        var sign = chg >= 0 ? "+" : "";

        var html =
          '<div style="font-size:11px;color:#9ca3af;margin-bottom:2px;">' +
          '<span style="font-weight:600;color:#374151;">\u65e5\u7ebf</span> ' + d.date + '</div>' +
          '<div>O <span style="color:' + chgColor + '">' + d.open.toFixed(2) + '</span></div>' +
          '<div>H <span style="color:' + chgColor + '">' + d.high.toFixed(2) + '</span></div>' +
          '<div>L <span style="color:' + chgColor + '">' + d.low.toFixed(2) + '</span></div>' +
          '<div>C <span style="color:' + chgColor + ';font-weight:600;">' + d.close.toFixed(2) + '</span></div>' +
          '<div style="color:' + chgColor + '">' + sign + chg.toFixed(2) + ' (' + sign + chgPct + '%)</div>' +
          '<div style="color:#6b7280">Vol ' + formatVol(d.volume) + '</div>';

        // Show trade info if entry or exit on this date
        var entry = tradeByEntry[dateStr];
        var exit = tradeByExit[dateStr];
        if (entry) {
          html += '<div style="margin-top:4px;padding-top:4px;border-top:1px solid #e5e7eb;">' +
            '<span style="color:#22c55e;font-weight:600;">BUY</span> @ $' + entry.entry_price.toFixed(2) +
            '</div>';
        }
        if (exit) {
          var exitColor = EXIT_COLORS[exit.exit_reason] || "#6b7280";
          var retSign = exit.total_return >= 0 ? "+" : "";
          html += '<div style="margin-top:' + (entry ? '2' : '4') + 'px;' + (entry ? '' : 'padding-top:4px;border-top:1px solid #e5e7eb;') + '">' +
            '<span style="color:' + exitColor + ';font-weight:600;">' + EXIT_LABELS[exit.exit_reason].toUpperCase() + '</span>' +
            ' <span style="color:' + (exit.total_return >= 0 ? '#22c55e' : '#ef4444') + '">' +
            retSign + (exit.total_return * 100).toFixed(1) + '%</span>' +
            ' (' + exit.holding_days + 'd)' +
            (exit.flew ? ' <span style="color:#f59e0b">flew</span>' : '') +
            '</div>';
        }

        dataPanel.innerHTML = html;
      }

      function updateKdjPanel(dateStr) {
        var d = ohlcvMap[dateStr];
        var base = '<span style="font-weight:600;color:#374151;">KDJ</span><span style="color:#9ca3af">(9,3,3)</span>';
        if (!d || d.k === undefined) {
          kdjPanel.innerHTML = base;
          return;
        }
        kdjPanel.innerHTML = base +
          '&nbsp;&nbsp;<span style="color:#3b82f6">K ' + d.k.toFixed(1) + '</span>' +
          '&nbsp;&nbsp;<span style="color:#f59e0b">D ' + d.d.toFixed(1) + '</span>' +
          '&nbsp;&nbsp;<span style="color:#a855f7">J ' + d.j.toFixed(1) + '</span>';
      }

      // Sync crosshair between charts
      function extractDateStr(param) {
        if (!param || !param.time) return null;
        var t = param.time;
        return t.year + "-" +
          String(t.month).padStart(2, "0") + "-" +
          String(t.day).padStart(2, "0");
      }

      var syncingCrosshair = false;

      chart.subscribeCrosshairMove(function (param) {
        var dateStr = extractDateStr(param);
        if (!dateStr) {
          updateDataPanel(ohlcv[ohlcv.length - 1].date);
          updateKdjPanel(ohlcv[ohlcv.length - 1].date);
          return;
        }
        updateDataPanel(dateStr);
        updateKdjPanel(dateStr);

        if (!syncingCrosshair) {
          syncingCrosshair = true;
          if (param.point) {
            kdjChart.setCrosshairPosition(undefined, undefined, kSeries);
          }
          syncingCrosshair = false;
        }
      });

      kdjChart.subscribeCrosshairMove(function (param) {
        var dateStr = extractDateStr(param);
        if (!dateStr) {
          updateDataPanel(ohlcv[ohlcv.length - 1].date);
          updateKdjPanel(ohlcv[ohlcv.length - 1].date);
          return;
        }
        updateDataPanel(dateStr);
        updateKdjPanel(dateStr);
      });

      // Show last day by default
      updateDataPanel(ohlcv[ohlcv.length - 1].date);
      updateKdjPanel(ohlcv[ohlcv.length - 1].date);

      /* ── Ticker overlay ── */
      var tickerLabel = document.createElement("div");
      tickerLabel.style.cssText =
        "position:absolute;top:8px;left:12px;font-size:18px;font-weight:700;" +
        "color:#1a1a2e;opacity:0.6;pointer-events:none;z-index:10;" +
        "font-family:DM Sans,Outfit,sans-serif;";
      tickerLabel.textContent = ticker + " \u65e5\u7ebf Daily";
      chartDiv.appendChild(tickerLabel);

      /* ── Responsive resize ── */
      if (typeof ResizeObserver !== "undefined") {
        new ResizeObserver(function () {
          chart.applyOptions({ width: chartDiv.clientWidth });
          kdjChart.applyOptions({ width: kdjDiv.clientWidth });
        }).observe(chartDiv);
      }

      chart.timeScale().fitContent();
      kdjChart.timeScale().fitContent();
    })
    .catch(function (err) {
      container.innerHTML =
        '<p style="color:#E45756;text-align:center;padding:40px 0;">Error loading K-line data.</p>';
      console.error("Viz5 K-line error:", err);
    });
})();
