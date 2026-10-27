import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";
import { parse } from "csv-parse/sync";
import {
  validateMeasurements,
  validateFits,
} from "../scripts/benchmark-data.mjs";
import { runtimePlot, fitPlot } from "../scripts/benchmark-plots.mjs";

function measurements() {
  const fixture = {
    backend: "sprs_to_ndarray",
    n: 256,
    p: 16,
    target_density: 0.03,
    nnz: 100,
    empty_columns: 0,
    normalization: "standardized",
  };
  const timings = {
    fit_parameters: 100,
    materialize: 400,
    raw_forward: 10,
    raw_transpose: 20,
    lazy_forward: 30,
    lazy_transpose: 40,
    eager_forward: 20,
    eager_transpose: 25,
  };
  const rows = Object.entries(timings).map(([measurement, value]) => ({
    ...fixture,
    measurement,
    estimate_ns: value,
    lower_ns: value * 0.9,
    upper_ns: value * 1.1,
  }));
  for (const [strategy, representation, peak] of [
    ["raw", 1736, 6088],
    ["lazy", 1992, 6472],
    ["eager", 33024, 39112],
  ]) {
    rows.push({
      ...fixture,
      measurement: `${strategy}_memory`,
      source_bytes: 1736,
      representation_bytes: representation,
      workspace_bytes:
        peak - representation - (strategy === "eager" ? 1736 : 0),
      peak_total_bytes: peak,
    });
  }
  return rows;
}

function fits() {
  return ["lazy", "eager"].map((strategy) => ({
    backend: "sprs_to_ndarray",
    n: 256,
    p: 16,
    target_density: 0.03,
    nnz: 100,
    empty_columns: 0,
    strategy,
    source_bytes: 1736,
    response_bytes: 2048,
    peak_total_bytes: strategy === "lazy" ? 8000 : 40000,
    iterations: 20,
    objective: 0.3,
    gradient_norm: 1e-8,
    gradient_threshold: 1e-7,
    estimate_ns: 1e6,
    lower_ns: 0.9e6,
    upper_ns: 1.1e6,
  }));
}

test("complete-fit measurements require comparable, converged lazy and eager results", () => {
  const rows = fits();
  const [group] = validateFits(rows);
  assert.equal(group.fits.eager.peak_total_bytes, 40000);
  assert.throws(() => validateFits(rows.slice(1)), /Missing/);
  assert.throws(() => validateFits([...rows, rows[0]]), /Duplicate/);
  for (const [field, value, error] of [
    ["gradient_norm", 1e-3, /converg/],
    ["objective", 0.2, /objective/],
    ["nnz", 99, /fixture/],
    ["source_bytes", 1800, /fixture/],
    ["peak_total_bytes", 100, /memory/],
    ["lower_ns", 2e6, /interval/],
    ["estimate_ns", "", /interval/],
    ["iterations", -1, /converg/],
  ]) {
    assert.throws(
      () =>
        validateFits(
          rows.map((row, i) => (i === 1 ? { ...row, [field]: value } : row)),
        ),
      error,
    );
  }
});

test("single-panel plots have readable narrow variants and reject mixed workloads", () => {
  const groups = validateMeasurements(measurements());
  const svg = runtimePlot(groups, "forward", 320);
  assert.match(svg, /viewBox="0 0 320 /);
  assert.match(svg, /Lazy sparse/);
  assert.match(svg, /Eager dense/);
  assert.doesNotMatch(svg, /Raw sparse|n ≫ p|n = p|n ≪ p/);
  assert.throws(
    () => runtimePlot([...groups, { ...groups[0], n: 512 }], "forward", 320),
    /workload/,
  );
  for (const metric of ["estimate_ns", "peak_total_bytes"]) {
    assert.match(
      fitPlot(validateFits(fits()), metric, 320),
      /viewBox="0 0 320 /,
    );
  }
});

test("rejects duplicate, incomplete, inconsistent, and invalid timing measurements", () => {
  const rows = measurements();
  assert.throws(() => validateMeasurements([...rows, rows[0]]), /Duplicate/);
  assert.throws(() => validateMeasurements(rows.slice(1)), /Missing/);
  assert.throws(
    () =>
      validateMeasurements(
        rows.map((row, i) => (i === 0 ? { ...row, nnz: 99 } : row)),
      ),
    /Inconsistent/,
  );
  assert.throws(
    () =>
      validateMeasurements(
        rows.map((row, i) => (i === 0 ? { ...row, lower_ns: 200 } : row)),
      ),
    /interval/,
  );
  assert.throws(
    () =>
      validateMeasurements(
        rows.map((row, i) => (i === 0 ? { ...row, estimate_ns: "" } : row)),
      ),
    /interval/,
  );
});

test("rejects memory totals that omit the retained source or include it twice", () => {
  const rows = measurements();
  for (const strategy of ["lazy", "eager"]) {
    assert.throws(
      () =>
        validateMeasurements(
          rows.map((row) =>
            row.measurement === `${strategy}_memory`
              ? {
                  ...row,
                  peak_total_bytes: row.peak_total_bytes - row.source_bytes,
                }
              : row,
          ),
        ),
      /memory/,
    );
  }
});

test("rejects a missing case in a size and density grid", () => {
  const rows = measurements();
  assert.throws(
    () =>
      validateMeasurements([
        ...rows,
        ...rows.map((row) => ({ ...row, n: 512 })),
        ...rows.map((row) => ({ ...row, target_density: 0.1 })),
      ]),
    /Incomplete/,
  );
});

test("published measurements match tall, square, and wide matrices at every size", async () => {
  const groups = validateMeasurements(
    parse(
      await readFile(
        new URL(
          "../../benches/normalization_tradeoffs/results.csv",
          import.meta.url,
        ),
        "utf8",
      ),
      { columns: true, skip_empty_lines: true },
    ),
  );
  const sizes = [...new Set(groups.map((g) => g.n * g.p * 8))].sort(
    (a, b) => a - b,
  );
  assert.deepEqual(
    sizes,
    [2, 32, 128].map((mib) => mib * 2 ** 20),
  );
  for (const backend of ["sprs_to_ndarray", "faer_csc_to_dense"]) {
    for (const normalization of ["standardized", "scaled"]) {
      for (const size of sizes) {
        const selected = groups.filter(
          (g) =>
            g.backend === backend &&
            g.normalization === normalization &&
            g.n * g.p * 8 === size,
        );
        assert.equal(selected.length, 21);
        assert.deepEqual(
          [...new Set(selected.map((g) => g.n / g.p))].sort((a, b) => b - a),
          [16, 1, 1 / 16],
        );
      }
    }
  }
});

test("published fits and products cover every selectable size for both backends", async () => {
  const source = new URL("../../benches/least_squares/", import.meta.url);
  const read = async (name, validate) =>
    validate(
      parse(await readFile(new URL(name, source), "utf8"), {
        columns: true,
        skip_empty_lines: true,
      }),
    );
  const fits = await read("results.csv", validateFits);
  const products = await read("products.csv", validateMeasurements);
  const metadata = JSON.parse(
    await readFile(new URL("metadata.json", source), "utf8"),
  );
  assert.equal(metadata.source_retained, true);
  assert.equal(metadata.threads, 1);
  const sizes = [
    [2048, 128],
    [8192, 512],
    [16384, 1024],
  ];
  for (const backend of ["sprs_to_ndarray", "faer_csc_to_dense"]) {
    for (const [n, p] of sizes) {
      const selected = fits.filter(
        (g) => g.backend === backend && g.n === n && g.p === p,
      );
      assert.equal(selected.length, 7);
      assert.deepEqual(
        selected.map((g) => g.target_density),
        [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1],
      );
      for (const g of selected) {
        const operation = products.find(
          (p) =>
            p.backend === backend &&
            p.n === g.n &&
            p.p === g.p &&
            p.normalization === "standardized" &&
            p.target_density === g.target_density,
        );
        assert.ok(operation);
        for (const field of ["n", "p", "nnz", "empty_columns"])
          assert.equal(operation[field], g[field]);
      }
      assert.ok(
        (metadata.timing_runs ?? [metadata]).some((r) =>
          r.shapes.some((shape) => shape[0] === n && shape[1] === p),
        ),
      );
    }
  }
});
