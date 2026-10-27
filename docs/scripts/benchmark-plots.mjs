import * as Plot from "@observablehq/plot";
import { JSDOM } from "jsdom";

const document = new JSDOM().window.document;
const colors = { lazy: "#38634f", eager: "#cc6b2c" };
const labels = { lazy: "Lazy sparse", eager: "Eager dense" };
const numberFormat = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumSignificantDigits: 3,
});

function chart(groups, points, { title, label, width }) {
  if (
    !groups.length ||
    new Set(groups.map((g) => [g.backend, g.normalization, g.n, g.p].join(":")))
      .size !== 1
  )
    throw new Error("A chart must show a single workload");
  const lowerPower = Math.floor(
    Math.log10(Math.min(...points.map((p) => p.lower ?? p.value))),
  );
  const upperPower = Math.max(
    lowerPower + 1,
    Math.ceil(Math.log10(Math.max(...points.map((p) => p.upper ?? p.value)))),
  );
  const ticks = Array.from(
    { length: upperPower - lowerPower + 1 },
    (_, i) => 10 ** (lowerPower + i),
  );
  const marks = [];
  for (const strategy of ["eager", "lazy"]) {
    const data = points.filter((point) => point.strategy === strategy);
    if (data.every((point) => point.lower > 0 && point.upper > 0))
      marks.push(
        Plot.areaY(data, {
          x: "density",
          y1: "lower",
          y2: "upper",
          fill: colors[strategy],
          fillOpacity: 0.15,
        }),
      );
    marks.push(
      Plot.lineY(data, {
        x: "density",
        y: "value",
        stroke: colors[strategy],
        strokeWidth: 2.5,
        strokeDasharray: strategy === "eager" ? "7,3" : null,
      }),
      Plot.dot(data, {
        x: "density",
        y: "value",
        fill: colors[strategy],
        r: 3,
      }),
    );
  }
  const plot = Plot.plot({
    document,
    width,
    height: 292,
    marginTop: 34,
    marginRight: 18,
    marginBottom: 46,
    marginLeft: 54,
    style: {
      background: "white",
      color: "#303642",
      fontFamily: "system-ui, sans-serif",
      fontSize: "13px",
    },
    x: {
      type: "log",
      domain: [0.08, 120],
      ticks: width < 400 ? [0.1, 1, 10, 100] : [0.1, 0.3, 1, 3, 10, 30, 100],
      tickFormat: (x) => `${x}`,
      label: "Nonzeros (%) →",
    },
    y: {
      type: "log",
      domain: [10 ** lowerPower, 10 ** upperPower],
      grid: true,
      ticks,
      tickFormat: numberFormat.format,
      label,
    },
    marks,
  });
  plot.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  plot.setAttribute("viewBox", `0 0 ${width} 340`);
  plot.setAttribute("height", "340");
  const context = `${groups[0].n.toLocaleString("en-US")} × ${groups[0].p.toLocaleString("en-US")}`;
  const titleNode = document.createElementNS(plot.namespaceURI, "title");
  titleNode.textContent = `${title}. ${context}. Mean centering and SD scaling. ${groups[0].backend}.`;
  plot.prepend(titleNode);
  plot.setAttribute("aria-label", titleNode.textContent);
  for (const [index, strategy] of ["lazy", "eager"].entries()) {
    const group = document.createElementNS(plot.namespaceURI, "g");
    group.setAttribute(
      "transform",
      `translate(${24 + (index * (width - 32)) / 2}, 310)`,
    );
    const line = document.createElementNS(plot.namespaceURI, "line");
    for (const [key, value] of Object.entries({
      x2: 24,
      stroke: colors[strategy],
      "stroke-width": 3,
      "stroke-dasharray": strategy === "eager" ? "7,3" : "none",
    }))
      line.setAttribute(key, value);
    const text = document.createElementNS(plot.namespaceURI, "text");
    text.setAttribute("x", "32");
    text.setAttribute("dy", "0.32em");
    text.setAttribute("text-anchor", "start");
    text.textContent = labels[strategy];
    group.append(line, text);
    plot.append(group);
  }
  const footer = document.createElementNS(plot.namespaceURI, "text");
  for (const [key, value] of Object.entries({
    x: 24,
    y: 336,
    fill: "#667583",
    "font-size": 11,
    "text-anchor": "start",
  }))
    footer.setAttribute(key, value);
  footer.textContent = `${context} · f64`;
  plot.append(footer);
  return plot.outerHTML + "\n";
}

export function runtimePlot(groups, direction, width = 520) {
  return chart(
    groups,
    groups.flatMap((group) =>
      ["lazy", "eager"].map((strategy) => {
        const timing = group.timings[`${strategy}_${direction}`];
        return {
          strategy,
          density: (100 * group.nnz) / (group.n * group.p),
          value: timing.estimate_ns / 1e3,
          lower: timing.lower_ns / 1e3,
          upper: timing.upper_ns / 1e3,
        };
      }),
    ),
    {
      title: `${direction === "forward" ? "Forward" : "Transpose"} product time, excluding setup`,
      label: "Time per product (µs) ↑",
      width,
    },
  );
}

export function fitPlot(groups, metric, width = 520) {
  const time = metric === "estimate_ns";
  return chart(
    groups,
    groups.flatMap((group) =>
      ["lazy", "eager"].map((strategy) => {
        const fit = group.fits[strategy];
        return {
          strategy,
          density: (100 * group.nnz) / (group.n * group.p),
          value: fit[metric] / (time ? 1e6 : 2 ** 20),
          lower: time ? fit.lower_ns / 1e6 : null,
          upper: time ? fit.upper_ns / 1e6 : null,
        };
      }),
    ),
    {
      title: time
        ? "Complete ridge fit time, including setup"
        : "Peak live Rust allocations across a complete ridge fit",
      label: time ? "Total time (ms) ↑" : "Peak live allocations (MiB) ↑",
      width,
    },
  );
}
