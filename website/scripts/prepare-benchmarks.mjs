import { readFile, mkdir, writeFile, copyFile } from "node:fs/promises";
import { parse } from "csv-parse/sync";

const source = new URL(
  "../../benches/normalization_allocations/eager_results.csv",
  import.meta.url,
);
const rows = parse(await readFile(source, "utf8"), {
  columns: true,
  skip_empty_lines: true,
});
const groups = new Map();
for (const row of rows) {
  if (row.normalization !== "standardized") continue;
  const key = [row.consumer, row.backend, row.n, row.p, row.density].join(":");
  if (!groups.has(key)) {
    groups.set(key, {
      backend: row.backend,
      n: Number(row.n),
      p: Number(row.p),
      density: Number(row.density),
      phases: {},
    });
  }
  const group = groups.get(key);
  if (group.phases[row.phase]) throw new Error(`Duplicate phase for ${key}`);
  group.phases[row.phase] = {
    milliseconds: Number(row.elapsed_ns) / 1e6,
    peakExtraBytes: Number(row.peak_extra_live_bytes),
    allocations: Number(row.allocations),
  };
}
for (const [key, group] of groups) {
  for (const phase of ["lazy_fit", "eager_build", "eager_fit"]) {
    const values = group.phases[phase];
    if (
      !values ||
      Object.values(values).some((v) => !Number.isFinite(v) || v < 0)
    ) {
      throw new Error(`Missing or invalid ${phase} for ${key}`);
    }
  }
}
if (!groups.size) throw new Error("No standardized benchmark results found");
const theme = new URL("../.vitepress/theme/", import.meta.url);
const downloads = new URL("../public/benchmarks/", import.meta.url);
await mkdir(theme, { recursive: true });
await mkdir(downloads, { recursive: true });
await writeFile(
  new URL("benchmark-results.json", theme),
  JSON.stringify([...groups.values()], null, 2) + "\n",
);
await copyFile(source, new URL("eager-results.csv", downloads));
