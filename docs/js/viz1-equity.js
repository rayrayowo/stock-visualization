/**
 * Viz 1: B1 Strategy Equity Curve — D3 Area Chart with Brush Zoom
 * Shows B1 portfolio value over time with bull/bear regime shading
 * and a buy-and-hold S&P 500 baseline for comparison.
 *
 * Data: b1b2_b1_equity_opt.json  (date, value)
 *       market_proxy.json        (date, equity, regime)
 */
(function () {
  const container = document.getElementById("viz1-container");
  if (!container) return;

  // ── Dimensions ──
  const margin = { top: 24, right: 30, bottom: 120, left: 72 };
  const fullWidth = Math.max(container.clientWidth, 800);
  const width = fullWidth - margin.left - margin.right;
  const totalHeight = 540;
  const contextHeight = 50;
  const gap = 40;
  const focusHeight = totalHeight - contextHeight - gap - margin.top - margin.bottom;

  // ── Colors ──
  const BULL_BG    = "rgba(34, 197, 94, 0.18)";
  const BEAR_BG    = "rgba(239, 68, 68, 0.22)";
  const B1_COLOR   = "#22c55e";
  const B1_FILL    = "rgba(34, 197, 94, 0.10)";
  const BH_COLOR   = "#6b7280";
  const ZERO_COLOR = "#d1d5db";

  const fmtDollar = d3.format("$,.0f");
  const fmtPct    = d3.format("+.1f");
  const fmtDate   = d3.timeFormat("%Y-%m-%d");

  // ── SVG ──
  const svg = d3.select(container)
    .append("svg")
    .attr("width", fullWidth)
    .attr("height", totalHeight);

  const tooltip = d3.select("#tooltip");

  // ── Load both datasets ──
  Promise.all([
    d3.json("data/b1b2_b1_equity_opt.json"),
    d3.json("data/market_proxy.json")
  ]).then(function ([equityRaw, marketRaw]) {

    // Parse equity data (B1 strategy)
    const b1Data = equityRaw.map(d => ({
      date: new Date(d.date),
      value: +d.value
    }));

    // Parse market data (regime + buy-and-hold baseline)
    const mktData = marketRaw.map(d => ({
      date: new Date(d.date),
      equity: +d.equity,
      regime: d.regime || "bull"
    }));

    // Scale market equity to $1M start for buy-and-hold comparison
    const startEquity = mktData[0].equity || 1;
    const bhData = mktData.map(d => ({
      date: d.date,
      value: (d.equity / startEquity) * 1000000
    }));

    // Build regime lookup by aligning dates
    const regimeMap = new Map();
    mktData.forEach(d => regimeMap.set(fmtDate(d.date), d.regime));

    // ── Scales (focus) ──
    const xDomain = d3.extent(b1Data, d => d.date);
    const allValues = b1Data.map(d => d.value).concat(bhData.map(d => d.value));
    const yMin = d3.min(allValues) * 0.97;
    const yMax = d3.max(allValues) * 1.03;

    const x = d3.scaleTime().domain(xDomain).range([0, width]);
    const y = d3.scaleLinear().domain([yMin, yMax]).range([focusHeight, 0]);

    // ── Scales (context) ──
    const x2 = d3.scaleTime().domain(xDomain).range([0, width]);
    const y2 = d3.scaleLinear().domain([yMin, yMax]).range([contextHeight, 0]);

    // ── Focus group ──
    const focus = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    svg.append("defs").append("clipPath")
      .attr("id", "clip-focus")
      .append("rect")
      .attr("width", width)
      .attr("height", focusHeight);

    const clipped = focus.append("g").attr("clip-path", "url(#clip-focus)");

    // ── Regime bands ──
    function drawRegimeBands(g, xScale, height) {
      g.selectAll(".regime-band").remove();
      let i = 0;
      while (i < mktData.length) {
        const regime = mktData[i].regime;
        const startDate = mktData[i].date;
        let j = i;
        while (j < mktData.length && mktData[j].regime === regime) j++;
        const endDate = mktData[j - 1].date;
        const rx = xScale(startDate);
        const rw = xScale(endDate) - rx;
        if (rw > 0) {
          g.append("rect")
            .attr("class", "regime-band")
            .attr("x", rx)
            .attr("y", 0)
            .attr("width", rw)
            .attr("height", height)
            .attr("fill", regime === "bear" ? BEAR_BG : BULL_BG);
        }
        i = j;
      }
    }
    drawRegimeBands(clipped, x, focusHeight);

    // ── $1M baseline ──
    clipped.append("line")
      .attr("class", "baseline")
      .attr("x1", 0).attr("x2", width)
      .attr("y1", y(1000000)).attr("y2", y(1000000))
      .attr("stroke", ZERO_COLOR)
      .attr("stroke-width", 1)
      .attr("stroke-dasharray", "4,4");

    // ── Buy & Hold area + line ──
    const bhAreaGen = d3.area()
      .x(d => x(d.date))
      .y0(focusHeight)
      .y1(d => y(d.value))
      .curve(d3.curveMonotoneX);

    const bhLineGen = d3.line()
      .x(d => x(d.date))
      .y(d => y(d.value))
      .curve(d3.curveMonotoneX);

    const bhArea = clipped.append("path")
      .datum(bhData)
      .attr("fill", "rgba(107, 114, 128, 0.06)")
      .attr("d", bhAreaGen);

    const bhLine = clipped.append("path")
      .datum(bhData)
      .attr("fill", "none")
      .attr("stroke", BH_COLOR)
      .attr("stroke-width", 1.2)
      .attr("stroke-dasharray", "6,3")
      .attr("d", bhLineGen);

    // ── B1 Strategy area + line ──
    const b1AreaGen = d3.area()
      .x(d => x(d.date))
      .y0(focusHeight)
      .y1(d => y(d.value))
      .curve(d3.curveMonotoneX);

    const b1LineGen = d3.line()
      .x(d => x(d.date))
      .y(d => y(d.value))
      .curve(d3.curveMonotoneX);

    const b1Area = clipped.append("path")
      .datum(b1Data)
      .attr("fill", B1_FILL)
      .attr("d", b1AreaGen);

    const b1Line = clipped.append("path")
      .datum(b1Data)
      .attr("fill", "none")
      .attr("stroke", B1_COLOR)
      .attr("stroke-width", 2)
      .attr("d", b1LineGen);

    // ── Axes (focus) ──
    const xAxis = d3.axisBottom(x).ticks(8).tickFormat(d3.timeFormat("%b %Y"));
    const gX = focus.append("g")
      .attr("transform", `translate(0,${focusHeight})`)
      .call(xAxis);

    focus.append("g")
      .call(d3.axisLeft(y).ticks(6).tickFormat(fmtDollar));

    // Y-axis label
    focus.append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -58).attr("x", -focusHeight / 2)
      .attr("text-anchor", "middle")
      .attr("fill", "#6b7280").attr("font-size", "13px")
      .text("Portfolio Value ($)");

    // ── Legend ──
    const lg = focus.append("g").attr("transform", `translate(${width - 280}, 8)`);
    // B1 Strategy
    lg.append("line").attr("x2", 20).attr("stroke", B1_COLOR).attr("stroke-width", 2);
    lg.append("text").attr("x", 24).attr("y", 4).text("B1 Strategy (Optimized)")
      .attr("font-size", "12px").attr("fill", "#374151");
    // Buy & Hold
    lg.append("line").attr("y1", 18).attr("y2", 18).attr("x2", 20)
      .attr("stroke", BH_COLOR).attr("stroke-width", 1.2).attr("stroke-dasharray", "6,3");
    lg.append("text").attr("x", 24).attr("y", 22).text("Buy & Hold S&P 500")
      .attr("font-size", "12px").attr("fill", "#6b7280");
    // Bull regime
    lg.append("rect").attr("y", 30).attr("width", 14).attr("height", 10)
      .attr("fill", BULL_BG).attr("stroke", "#22c55e").attr("stroke-width", 0.5);
    lg.append("text").attr("x", 24).attr("y", 39).text("Bull Regime")
      .attr("font-size", "12px").attr("fill", "#6b7280");
    // Bear regime
    lg.append("rect").attr("x", 120).attr("y", 30).attr("width", 14).attr("height", 10)
      .attr("fill", BEAR_BG).attr("stroke", "#ef4444").attr("stroke-width", 0.5);
    lg.append("text").attr("x", 144).attr("y", 39).text("Bear Regime")
      .attr("font-size", "12px").attr("fill", "#6b7280");

    // ── Hover tooltip ──
    const bisect = d3.bisector(d => d.date).left;
    const bhBisect = d3.bisector(d => d.date).left;

    const hoverLine = focus.append("line")
      .attr("stroke", "#999").attr("stroke-width", 1).attr("stroke-dasharray", "3,3")
      .style("opacity", 0);
    const hoverDotB1 = focus.append("circle")
      .attr("r", 4).attr("fill", B1_COLOR).attr("stroke", "#fff").attr("stroke-width", 1.5)
      .style("opacity", 0);
    const hoverDotBH = focus.append("circle")
      .attr("r", 3.5).attr("fill", BH_COLOR).attr("stroke", "#fff").attr("stroke-width", 1)
      .style("opacity", 0);

    focus.append("rect")
      .attr("width", width).attr("height", focusHeight)
      .attr("fill", "none").attr("pointer-events", "all")
      .on("mousemove", function (event) {
        const [mx] = d3.pointer(event);
        const date = x.invert(mx);

        // Find closest B1 data point
        const idx = bisect(b1Data, date, 1);
        const d0 = b1Data[idx - 1], d1 = b1Data[idx] || d0;
        const d = date - d0.date > d1.date - date ? d1 : d0;

        // Find closest BH data point
        const bhIdx = bhBisect(bhData, date, 1);
        const bh0 = bhData[bhIdx - 1], bh1 = bhData[bhIdx] || bh0;
        const bh = date - bh0.date > bh1.date - date ? bh1 : bh0;

        // Regime lookup
        const regime = regimeMap.get(fmtDate(d.date)) || "bull";

        // Drawdown: peak-to-trough
        let peak = d.value;
        for (let k = 0; k <= idx; k++) {
          if (b1Data[k].value > peak) peak = b1Data[k].value;
        }
        const dd = ((d.value - peak) / peak * 100);

        hoverLine.attr("x1", x(d.date)).attr("x2", x(d.date))
          .attr("y1", 0).attr("y2", focusHeight).style("opacity", 1);
        hoverDotB1.attr("cx", x(d.date)).attr("cy", y(d.value)).style("opacity", 1);
        hoverDotBH.attr("cx", x(bh.date)).attr("cy", y(bh.value)).style("opacity", 1);

        const b1Ret = ((d.value / 1000000 - 1) * 100);
        const bhRet = ((bh.value / 1000000 - 1) * 100);

        tooltip.style("opacity", 1)
          .html(
            `<strong>${fmtDate(d.date)}</strong><br>` +
            `<span style="color:${B1_COLOR}">B1:</span> ${fmtDollar(d.value)} (${fmtPct(b1Ret)}%)<br>` +
            `<span style="color:${BH_COLOR}">B&H:</span> ${fmtDollar(bh.value)} (${fmtPct(bhRet)}%)<br>` +
            `Drawdown: ${dd.toFixed(2)}%<br>` +
            `Regime: <span style="color:${regime === 'bear' ? '#ef4444' : '#22c55e'}">${regime}</span>`
          )
          .style("left", (event.pageX + 15) + "px")
          .style("top", (event.pageY - 10) + "px");
      })
      .on("mouseleave", function () {
        hoverLine.style("opacity", 0);
        hoverDotB1.style("opacity", 0);
        hoverDotBH.style("opacity", 0);
        tooltip.style("opacity", 0);
      });

    // ══════════════════════════════════════════════════════════
    // CONTEXT (mini chart with brush)
    // ══════════════════════════════════════════════════════════
    const contextTop = margin.top + focusHeight + gap;
    const context = svg.append("g")
      .attr("transform", `translate(${margin.left},${contextTop})`);

    // Context B1 area
    const ctxArea = d3.area()
      .x(d => x2(d.date))
      .y0(contextHeight)
      .y1(d => y2(d.value))
      .curve(d3.curveMonotoneX);

    context.append("path")
      .datum(b1Data)
      .attr("fill", "rgba(34, 197, 94, 0.25)")
      .attr("d", ctxArea);

    // Regime bands on context
    drawRegimeBands(context, x2, contextHeight);

    // Context x-axis
    context.append("g")
      .attr("transform", `translate(0,${contextHeight})`)
      .call(d3.axisBottom(x2).ticks(6).tickFormat(d3.timeFormat("%Y")));

    // Label
    context.append("text")
      .attr("x", width / 2).attr("y", -6)
      .attr("text-anchor", "middle")
      .attr("font-size", "11px").attr("fill", "#999")
      .text("Drag to select time range");

    // ── Brush ──
    function brushed(event) {
      if (!event.selection) {
        x.domain(xDomain);
      } else {
        const [s0, s1] = event.selection.map(x2.invert);
        x.domain([s0, s1]);
      }
      // Update focus chart
      b1Area.attr("d", b1AreaGen);
      b1Line.attr("d", b1LineGen);
      bhArea.attr("d", bhAreaGen);
      bhLine.attr("d", bhLineGen);
      clipped.select(".baseline")
        .attr("y1", y(1000000)).attr("y2", y(1000000));
      gX.call(xAxis);
      drawRegimeBands(clipped, x, focusHeight);
    }

    const brush = d3.brushX()
      .extent([[0, 0], [width, contextHeight]])
      .on("brush end", brushed);

    context.append("g")
      .attr("class", "brush")
      .call(brush)
      .call(brush.move, [0, width]);

  }).catch(function (err) {
    container.innerHTML = '<p style="color:#E45756;text-align:center;">Error loading data. Check console for details.</p>';
    console.error("Viz1 error:", err);
  });
})();
