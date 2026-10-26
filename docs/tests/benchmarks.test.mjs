import assert from "node:assert/strict";
import { test } from "node:test";
import { readFile } from "node:fs/promises";
import { parse } from "csv-parse/sync";
import {
  validateMeasurements,
  totalTime,
  breakEven,
} from "../scripts/benchmark-data.mjs";

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

test("projects fixed product counts with fitting and materialization charged once", () => {
  const [group] = validateMeasurements(measurements());
  assert.equal(totalTime(group, "lazy", 10), 800);
  assert.equal(totalTime(group, "eager", 10), 950);
  assert.equal(breakEven(group), 16);
  assert.equal(totalTime(group, "lazy", 16), totalTime(group, "eager", 16));
  group.timings.eager_forward.estimate_ns = 80;
  assert.equal(breakEven(group), null);
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
