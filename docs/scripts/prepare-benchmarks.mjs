import {
  readFile,
  readdir,
  rm,
  mkdir,
  writeFile,
  copyFile,
} from "node:fs/promises";
import { createHash } from "node:crypto";
import { parse } from "csv-parse/sync";
import { validateMeasurements, validateFits } from "./benchmark-data.mjs";
import { runtimePlot, fitPlot } from "./benchmark-plots.mjs";

const source = new URL("../../benches/least_squares/", import.meta.url);
async function readCsv(filename, validate) {
  return validate(
    parse(await readFile(new URL(filename, source), "utf8"), {
      columns: true,
      skip_empty_lines: true,
    }),
  );
}
const fits = await readCsv("results.csv", validateFits);
const products = await readCsv("products.csv", validateMeasurements);
const metadata = JSON.parse(
  await readFile(new URL("metadata.json", source), "utf8"),
);
const theme = new URL("../.vitepress/theme/", import.meta.url);
const downloads = new URL("../public/benchmarks/", import.meta.url);
await mkdir(downloads, { recursive: true });
for (const file of await readdir(downloads)) {
  if (/^(sprs_to_ndarray|faer_csc_to_dense)-.*\.svg$/.test(file))
    await rm(new URL(file, downloads));
}
const cases = [];
const workloads = new Map();
for (const group of fits) {
  const key = `${group.backend}:${group.n}:${group.p}`;
  if (!workloads.has(key)) workloads.set(key, []);
  workloads.get(key).push(group);
}
for (const fitGroups of workloads.values()) {
  const { backend, n, p } = fitGroups[0];
  const operations = products.filter(
    (g) =>
      g.backend === backend &&
      g.normalization === "standardized" &&
      g.n === n &&
      g.p === p,
  );
  if (
    operations.length !== fitGroups.length ||
    operations.some(
      (g, i) =>
        g.target_density !== fitGroups[i].target_density ||
        g.nnz !== fitGroups[i].nnz ||
        g.empty_columns !== fitGroups[i].empty_columns,
    )
  )
    throw new Error("Fits and operations must use matching fixtures");
  const plots = {};
  for (const [name, render] of [
    ["fitTime", (width) => fitPlot(fitGroups, "estimate_ns", width)],
    ["fitMemory", (width) => fitPlot(fitGroups, "peak_total_bytes", width)],
    ["forward", (width) => runtimePlot(operations, "forward", width)],
    ["transpose", (width) => runtimePlot(operations, "transpose", width)],
  ]) {
    plots[name] = {};
    for (const [variant, width] of [
      ["wide", 520],
      ["narrow", 320],
    ]) {
      const svg = render(width);
      const hash = createHash("sha256").update(svg).digest("hex").slice(0, 10);
      const filename = `${backend}-${n}x${p}-${name}-${variant}-${hash}.svg`;
      await writeFile(new URL(filename, downloads), svg);
      plots[name][variant] = `/benchmarks/${filename}`;
    }
  }
  cases.push({ backend, n, p, fits: fitGroups, operations, plots });
}
await writeFile(
  new URL("benchmark-results.json", theme),
  JSON.stringify({ metadata, cases }, null, 2) + "\n",
);
for (const [from, to] of [
  ["results.csv", "least-squares.csv"],
  ["products.csv", "products.csv"],
  ["metadata.json", "least-squares-metadata.json"],
])
  await copyFile(new URL(from, source), new URL(to, downloads));
for (const filename of ["results.csv", "metadata.json"])
  await copyFile(
    new URL(
      `../../benches/normalization_tradeoffs/${filename}`,
      import.meta.url,
    ),
    new URL(filename, downloads),
  );
await copyFile(
  new URL(
    "../../benches/normalization_allocations/eager_results.csv",
    import.meta.url,
  ),
  new URL("eager-results.csv", downloads),
);
console.log(
  `Prepared ${fits.length} complete-fit cases and ${cases.length * 8} responsive SVG plots.`,
);
