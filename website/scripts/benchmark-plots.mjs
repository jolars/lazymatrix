import * as Plot from "@observablehq/plot";
import { JSDOM } from "jsdom";
import { totalTime } from "./benchmark-data.mjs";

const document = new JSDOM().window.document;
const colors = { lazy: "#38634f", eager: "#cc6b2c", raw: "#667583" };
const dashes = { lazy: null, eager: "8,3", raw: "3,3" };
const labels = {
  lazy: "Lazy sparse",
  eager: "Eager dense",
  raw: "Raw sparse (unnormalized)",
};
const shape = (group) => {
  const regime =
    group.n > group.p ? "n ≫ p" : group.n < group.p ? "n ≪ p" : "n = p";
  return `${regime} · ${group.n.toLocaleString("en-US")} × ${group.p.toLocaleString("en-US")}`;
};
const shapes = (groups) => [
  ...new Set([...groups].sort((a, b) => b.n / b.p - a.n / a.p).map(shape)),
];
const density = (group) => (100 * group.nnz) / (group.n * group.p);
const mib = (bytes) => bytes / 2 ** 20;
const numberFormat = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumSignificantDigits: 3,
});

function chart(groups, { x, y, marks, title, strategies = ["lazy", "eager"] }) {
  const width = 960;
  const backend =
    groups[0].backend === "sprs_to_ndarray"
      ? "sprs CSC → ndarray dense"
      : "faer CSC → faer dense";
  const normalization =
    groups[0].normalization === "standardized"
      ? "Mean centering + SD scaling"
      : "SD scaling only";
  const context = `${backend} · ${normalization} · ${mib(groups[0].n * groups[0].p * 8)} MiB dense`;
  const plot = Plot.plot({
    document,
    width,
    height: 320,
    marginTop: 38,
    marginRight: 22,
    marginBottom: 50,
    marginLeft: 72,
    style: {
      background: "white",
      color: "#303642",
      fontFamily: "system-ui, sans-serif",
      fontSize: "13px",
    },
    ariaLabel: `${title}. ${context}`,
    fx: { domain: shapes(groups), label: null, padding: 0.13, tickSize: 0 },
    x,
    y: {
      type: "log",
      grid: true,
      ticks: 4,
      tickFormat: numberFormat.format,
      ...y,
    },
    marks,
  });
  plot.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  plot.setAttribute("viewBox", `0 0 ${width} 370`);
  plot.setAttribute("height", "370");
  const titleNode = document.createElementNS(plot.namespaceURI, "title");
  titleNode.textContent = `${title}. ${context}`;
  plot.prepend(titleNode);
  strategies.forEach((strategy, index) => {
    const group = document.createElementNS(plot.namespaceURI, "g");
    group.setAttribute("transform", `translate(${130 + index * 245}, 348)`);
    const line = document.createElementNS(plot.namespaceURI, "line");
    for (const [key, value] of Object.entries({
      x2: 28,
      stroke: colors[strategy],
      "stroke-width": 3,
      "stroke-dasharray": dashes[strategy] ?? "none",
    }))
      line.setAttribute(key, value);
    const text = document.createElementNS(plot.namespaceURI, "text");
    text.setAttribute("x", "37");
    text.setAttribute("dy", "0.32em");
    text.setAttribute("text-anchor", "start");
    text.textContent = labels[strategy];
    group.append(line, text);
    plot.append(group);
  });
  const footer = document.createElementNS(plot.namespaceURI, "text");
  for (const [key, value] of Object.entries({
    x: 72,
    y: 365,
    fill: "#667583",
    "font-size": 11,
    "text-anchor": "start",
  }))
    footer.setAttribute(key, value);
  footer.textContent = context;
  plot.append(footer);
  return plot.outerHTML + "\n";
}

const densityAxis = {
  type: "log",
  domain: [0.08, 120],
  ticks: [0.1, 0.3, 1, 3, 10, 30, 100],
  tickFormat: (x) => `${x}`,
  label: "Nonzeros (%) →",
};

export function runtimePlot(groups, direction) {
  const points = groups.flatMap((group) =>
    ["lazy", "eager", "raw"].map((strategy) => {
      const value = group.timings[`${strategy}_${direction}`];
      return {
        shape: shape(group),
        density: density(group),
        strategy,
        value: value.estimate_ns / 1e3,
        lower: value.lower_ns / 1e3,
        upper: value.upper_ns / 1e3,
      };
    }),
  );
  const marks = [];
  for (const strategy of ["eager", "raw", "lazy"]) {
    const data = points.filter((point) => point.strategy === strategy);
    marks.push(
      Plot.areaY(data, {
        fx: "shape",
        x: "density",
        y1: "lower",
        y2: "upper",
        fill: colors[strategy],
        fillOpacity: 0.15,
      }),
    );
    marks.push(
      Plot.lineY(data, {
        fx: "shape",
        x: "density",
        y: "value",
        stroke: colors[strategy],
        strokeWidth: strategy === "raw" ? 1.5 : 2.5,
        strokeDasharray: dashes[strategy],
      }),
    );
    marks.push(
      Plot.dot(data, {
        fx: "shape",
        x: "density",
        y: "value",
        fill: colors[strategy],
        r: 2.5,
        title: (d) =>
          `${labels[strategy]}: ${d.value.toPrecision(3)} µs at ${d.density.toPrecision(3)}% nonzeros`,
      }),
    );
  }
  return chart(groups, {
    x: densityAxis,
    y: { label: "Time per product (µs) ↑" },
    marks,
    title: `${direction === "forward" ? "Forward" : "Transpose"} products by density; shaded bands show 95% confidence intervals`,
    strategies: ["lazy", "eager", "raw"],
  });
}

export function memoryPlot(groups, metric) {
  const names = {
    peak_total_bytes: "Full lifecycle peak",
    representation_bytes: "Normalized representation",
    workspace_bytes: "Peak workspace",
  };
  const marks = ["lazy", "eager"].flatMap((strategy) => {
    const data = groups.map((group) => ({
      shape: shape(group),
      density: density(group),
      value: mib(group.memory[strategy][metric]),
    }));
    return [
      Plot.lineY(data, {
        fx: "shape",
        x: "density",
        y: "value",
        stroke: colors[strategy],
        strokeWidth: 2.5,
        strokeDasharray: dashes[strategy],
      }),
      Plot.dot(data, {
        fx: "shape",
        x: "density",
        y: "value",
        fill: colors[strategy],
        r: 2.5,
        title: (d) => `${labels[strategy]}: ${d.value.toPrecision(3)} MiB`,
      }),
    ];
  });
  return chart(groups, {
    x: densityAxis,
    y: { label: `${names[metric]} (MiB) ↑` },
    marks,
    title: `${names[metric]} by density`,
  });
}

export function totalPlot(groups, targetDensity) {
  const selected = groups.filter(
    (group) => group.target_density === targetDensity,
  );
  const repetitions = [
    1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000,
    10000,
  ];
  const marks = ["lazy", "eager"].map((strategy) => {
    const points = selected.flatMap((group) =>
      repetitions.map((k) => ({
        shape: shape(group),
        k,
        value: totalTime(group, strategy, k) / 1e6,
      })),
    );
    return Plot.lineY(points, {
      fx: "shape",
      x: "k",
      y: "value",
      stroke: colors[strategy],
      strokeWidth: 2.5,
      strokeDasharray: dashes[strategy],
    });
  });
  return chart(selected, {
    x: {
      type: "log",
      ticks: [1, 10, 100, 1000, 10000],
      label: "Forward + transpose pairs →",
      tickFormat: (x) => x.toLocaleString("en-US"),
    },
    y: { label: "Projected total time (ms) ↑" },
    marks,
    title: `Projected time including setup at ${targetDensity * 100}% target density`,
  });
}
