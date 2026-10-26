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
import { validateMeasurements } from "./benchmark-data.mjs";
import { runtimePlot, memoryPlot, totalPlot } from "./benchmark-plots.mjs";

const source = new URL(
  "../../benches/normalization_tradeoffs/",
  import.meta.url,
);
const groups = validateMeasurements(
  parse(await readFile(new URL("results.csv", source), "utf8"), {
    columns: true,
    skip_empty_lines: true,
  }),
);
const metadata = JSON.parse(
  await readFile(new URL("metadata.json", source), "utf8"),
);
const theme = new URL("../.vitepress/theme/", import.meta.url);
const downloads = new URL("../public/benchmarks/", import.meta.url);
await mkdir(downloads, { recursive: true });
for (const file of await readdir(downloads)) {
  if (
    /^(sprs_to_ndarray|faer_csc_to_dense)-(standardized|scaled)-.*\.svg$/.test(
      file,
    )
  )
    await rm(new URL(file, downloads));
}
const plotSets = [];
for (const backend of [...new Set(groups.map((g) => g.backend))]) {
  for (const normalization of [
    ...new Set(
      groups.filter((g) => g.backend === backend).map((g) => g.normalization),
    ),
  ]) {
    const matching = groups.filter(
      (g) => g.backend === backend && g.normalization === normalization,
    );
    const sizes = [...new Set(matching.map((g) => g.n * g.p * 8))].sort(
      (a, b) => a - b,
    );
    for (const denseBytes of sizes) {
      const selected = matching.filter((g) => g.n * g.p * 8 === denseBytes);
      const plotSet = {
        backend,
        normalization,
        dense_bytes: denseBytes,
        runtime: {},
        memory: {},
        total: [],
      };
      const densities = [
        ...new Set(selected.map((g) => g.target_density)),
      ].sort((a, b) => a - b);
      const shapes = [...new Set(selected.map((g) => `${g.n}x${g.p}`))];
      const ratios = new Set(selected.map((g) => g.n / g.p));
      if (ratios.size !== 3 || ![16, 1, 1 / 16].every((r) => ratios.has(r)))
        throw new Error(
          `Expected tall, square, and wide matrices at ${denseBytes} dense bytes for ${backend}/${normalization}`,
        );
      if (selected.length !== shapes.length * densities.length)
        throw new Error(
          `Incomplete size/density grid for ${backend}/${normalization}`,
        );
      async function save(name, svg) {
        const hash = createHash("sha256")
          .update(svg)
          .digest("hex")
          .slice(0, 10);
        const filename = `${backend}-${normalization}-${denseBytes}-${name}-${hash}.svg`;
        await writeFile(new URL(filename, downloads), svg);
        return `/benchmarks/${filename}`;
      }
      for (const direction of ["forward", "transpose"])
        plotSet.runtime[direction] = await save(
          `runtime-${direction}`,
          runtimePlot(selected, direction),
        );
      for (const metric of [
        "peak_total_bytes",
        "representation_bytes",
        "workspace_bytes",
      ])
        plotSet.memory[metric] = await save(
          `memory-${metric}`,
          memoryPlot(selected, metric),
        );
      for (const density of densities)
        plotSet.total.push({
          density,
          path: await save(`total-${density}`, totalPlot(selected, density)),
        });
      plotSets.push(plotSet);
    }
  }
}
await writeFile(
  new URL("benchmark-results.json", theme),
  JSON.stringify({ metadata, groups, plotSets }, null, 2) + "\n",
);
for (const filename of ["results.csv", "metadata.json"])
  await copyFile(new URL(filename, source), new URL(filename, downloads));
await copyFile(
  new URL(
    "../../benches/normalization_allocations/eager_results.csv",
    import.meta.url,
  ),
  new URL("eager-results.csv", downloads),
);
console.log(
  `Prepared ${groups.length} benchmark cases and ${plotSets.length} sets of SVG plots.`,
);
