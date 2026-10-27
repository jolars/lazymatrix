<script setup>
import { computed, onMounted, ref } from "vue";
import results from "./benchmark-results.json";
import BenchmarkFigure from "./BenchmarkFigure.vue";

const labels = {
  sprs_to_ndarray: "sprs CSC → ndarray dense",
  faer_csc_to_dense: "faer CSC → faer dense",
};
const backend = ref("sprs_to_ndarray");
const size = ref("8192x512");
const backends = [...new Set(results.cases.map((c) => c.backend))];
const sizes = [
  ...new Map(results.cases.map((c) => [`${c.n}x${c.p}`, c])).values(),
].sort((a, b) => a.n - b.n || a.p - b.p);
const ready = ref(false);
onMounted(() => {
  ready.value = true;
});
const selected = computed(() =>
  results.cases.find(
    (c) => c.backend === backend.value && `${c.n}x${c.p}` === size.value,
  ),
);
const run = computed(
  () =>
    results.metadata.timing_runs?.find((r) =>
      r.shapes.some(
        ([n, p]) => n === selected.value.n && p === selected.value.p,
      ),
    ) ?? results.metadata,
);
const number = (x) => x.toLocaleString("en-US", { maximumFractionDigits: 3 });
const dimensions = computed(
  () => `${number(selected.value.n)} × ${number(selected.value.p)}`,
);
const mib = (x) => number(x / 2 ** 20);
const milliseconds = (x) => number(x / 1e6);
const microseconds = (x) => number(x / 1e3);
const strategies = { lazy: "Lazy sparse", eager: "Eager dense" };
const date = computed(() =>
  new Date(run.value.collected_at_utc).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "UTC",
  }),
);
</script>

<template>
  <div class="benchmark-comparison">
    <div class="benchmark-context">
      <p>
        <strong>{{ dimensions }}</strong> rows × columns · f64 · mean centering
        + SD scaling<br />
        The normalized dense matrix alone occupies
        {{ mib(selected.n * selected.p * 8) }} MiB.
      </p>
      <label>
        Backend
        <select id="benchmark-backend" v-model="backend" :disabled="!ready">
          <option v-for="b in backends" :key="b" :value="b">
            {{ labels[b] }}
          </option>
        </select>
      </label>
      <label>
        Matrix size (rows × columns)
        <select id="benchmark-size" v-model="size" :disabled="!ready">
          <option
            v-for="s in sizes"
            :key="`${s.n}x${s.p}`"
            :value="`${s.n}x${s.p}`"
          >
            {{ number(s.n) }} × {{ number(s.p) }}
          </option>
        </select>
      </label>
    </div>

    <section aria-labelledby="fit-title">
      <h2 id="fit-title">A complete fit</h2>
      <p>
        Fit ridge regression with gradient descent across seven nonzero
        densities, from 0.1% to 100%. Both paths use the same data, algorithm,
        and accuracy target, starting with the raw sparse input.
      </p>
      <div class="plot-grid">
        <BenchmarkFigure
          :plot="selected.plots.fitTime"
          title="Total wall-clock time"
          :alt="`Complete ridge fit time versus nonzero density for ${dimensions} matrices, comparing lazy sparse and eager dense normalization.`"
        >
          Includes fitting normalization parameters, materialization for eager,
          step-size estimation, and solving to convergence. Shading shows 95%
          confidence intervals. Lower is faster.
        </BenchmarkFigure>
        <BenchmarkFigure
          :plot="selected.plots.fitMemory"
          title="Peak memory during the fit"
          :alt="`Peak live Rust allocation bytes across normalization and a complete ridge fit versus density for ${dimensions} matrices.`"
        >
          The largest live allocation total from normalization through
          convergence, including the source matrix, response, normalized
          storage, and solver workspace. Both paths retain the source. Lower
          uses less memory.
        </BenchmarkFigure>
      </div>
      <p class="measurement-note">
        Memory counts requested Rust bytes, excluding allocator overhead and
        native-library allocations. It does not measure total process RAM.
      </p>
      <details>
        <summary>Fit measurements and convergence</summary>
        <p>
          Both paths satisfy the same gradient tolerance and produce matching
          coefficients and objectives. Iterations count coefficient updates; the
          CSV also records the objective and final gradient norm.
        </p>
        <div
          class="table-scroll"
          tabindex="0"
          role="region"
          aria-label="Complete fit measurements"
        >
          <table>
            <thead>
              <tr>
                <th>Nonzeros (%)</th>
                <th>Strategy</th>
                <th>Total (ms)</th>
                <th>Peak (MiB)</th>
                <th>Iterations</th>
              </tr>
            </thead>
            <tbody>
              <template v-for="g in selected.fits" :key="g.target_density">
                <tr v-for="(label, strategy) in strategies" :key="strategy">
                  <td>{{ number((100 * g.nnz) / (g.n * g.p)) }}</td>
                  <td>{{ label }}</td>
                  <td>{{ milliseconds(g.fits[strategy].estimate_ns) }}</td>
                  <td>{{ mib(g.fits[strategy].peak_total_bytes) }}</td>
                  <td>{{ g.fits[strategy].iterations }}</td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
      </details>
    </section>

    <section aria-labelledby="operations-title">
      <h2 id="operations-title">Individual operations</h2>
      <p>
        These timings measure one product with normalization already fitted and
        eager storage already materialized. Outputs are reused; any scratch
        allocation inside the operation is included.
      </p>
      <div class="plot-grid">
        <BenchmarkFigure
          :plot="selected.plots.forward"
          title="Forward product: X̃v"
          :alt="`Forward product time excluding setup versus density for ${dimensions} matrices, comparing lazy sparse and eager dense operators.`"
        >
          Multiply by a coefficient vector. Shading shows 95% confidence
          intervals. Lower is faster.
        </BenchmarkFigure>
        <BenchmarkFigure
          :plot="selected.plots.transpose"
          title="Transpose product: X̃ᵀu"
          :alt="`Transpose product time excluding setup versus density for ${dimensions} matrices, comparing lazy sparse and eager dense operators.`"
        >
          Multiply by a vector of observations. Shading shows 95% confidence
          intervals. Lower is faster.
        </BenchmarkFigure>
      </div>
      <details>
        <summary>Operation measurements and memory breakdown</summary>
        <p>
          The memory column includes normalization, eager construction, and ten
          forward/transpose pairs, with the sparse source retained. This is a
          separate workload from the complete fit above. The CSV includes
          normalized storage, workspace, setup timings, and raw sparse timings.
        </p>
        <div
          class="table-scroll"
          tabindex="0"
          role="region"
          aria-label="Individual operation measurements"
        >
          <table>
            <thead>
              <tr>
                <th>Nonzeros (%)</th>
                <th>Strategy</th>
                <th>Forward (µs)</th>
                <th>Transpose (µs)</th>
                <th>Peak (MiB)</th>
              </tr>
            </thead>
            <tbody>
              <template
                v-for="g in selected.operations"
                :key="g.target_density"
              >
                <tr v-for="(label, strategy) in strategies" :key="strategy">
                  <td>{{ number((100 * g.nnz) / (g.n * g.p)) }}</td>
                  <td>{{ label }}</td>
                  <td>
                    {{
                      microseconds(g.timings[`${strategy}_forward`].estimate_ns)
                    }}
                  </td>
                  <td>
                    {{
                      microseconds(
                        g.timings[`${strategy}_transpose`].estimate_ns,
                      )
                    }}
                  </td>
                  <td>{{ mib(g.memory[strategy].peak_total_bytes) }}</td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
      </details>
    </section>

    <p class="measurement-note">
      {{ date }} · {{ results.metadata.cpu }} ·
      {{ results.metadata.threads }} thread · {{ run.sample_size }} timing
      samples per case. Both axes use logarithmic scales. Points show measured
      nonzero densities; lines connect the samples.
    </p>
    <p class="downloads">
      <a href="/benchmarks/least-squares.csv" download
        >Fit measurements (CSV)</a
      >
      ·
      <a href="/benchmarks/products.csv" download
        >Operation measurements (CSV)</a
      >
      ·
      <a href="/benchmarks/least-squares-metadata.json" download
        >Run metadata</a
      >
    </p>
  </div>
</template>

<style scoped>
.benchmark-comparison {
  margin: 1.5rem 0;
}
.benchmark-context {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: var(--vp-c-bg-soft);
}
.benchmark-context p {
  grid-column: 1 / -1;
  margin: 0;
  font-size: 0.9rem;
}
label {
  display: grid;
  min-width: 0;
  max-width: 100%;
  gap: 0.35rem;
  font-size: 0.85rem;
  font-weight: 600;
}
select {
  appearance: auto;
  min-width: 0;
  max-width: 100%;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  padding: 0.45rem;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  font: inherit;
}
select:focus-visible,
summary:focus-visible,
.table-scroll:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 3px;
}
.plot-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 2rem;
  margin: 1.5rem 0;
}
@media (min-width: 1280px) {
  .plot-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
@media (max-width: 639px) {
  .benchmark-context {
    grid-template-columns: minmax(0, 1fr);
  }
}
details {
  margin: 1.5rem 0;
  font-size: 0.9rem;
}
summary {
  cursor: pointer;
  font-weight: 600;
}
.table-scroll {
  overflow-x: auto;
}
table {
  font-size: 0.8rem;
  white-space: nowrap;
}
.measurement-note {
  color: var(--vp-c-text-2);
  font-size: 0.85rem;
  line-height: 1.6;
}
.downloads {
  font-size: 0.9rem;
}
</style>
