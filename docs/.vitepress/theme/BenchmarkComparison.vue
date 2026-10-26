<script setup>
import { computed, onMounted, ref } from "vue";
import results from "./benchmark-results.json";
import { breakEven } from "../../scripts/benchmark-data.mjs";

const labels = {
  sprs_to_ndarray: "sprs CSC → ndarray dense",
  faer_csc_to_dense: "faer CSC → faer dense",
};
const normalizationLabels = {
  standardized: "Mean centering + SD scaling",
  scaled: "SD scaling only",
};
const backend = ref("sprs_to_ndarray");
const normalization = ref("standardized");
const denseBytes = ref(32 * 2 ** 20);
const direction = ref("forward");
const memoryMetric = ref("peak_total_bytes");
const density = ref(0.01);
const ready = ref(false);
onMounted(() => {
  ready.value = true;
});
const backends = [...new Set(results.plotSets.map((p) => p.backend))];
const normalizations = [
  ...new Set(results.plotSets.map((p) => p.normalization)),
];
const sizes = [...new Set(results.plotSets.map((p) => p.dense_bytes))].sort(
  (a, b) => a - b,
);
const plots = computed(() =>
  results.plotSets.find(
    (p) =>
      p.backend === backend.value &&
      p.normalization === normalization.value &&
      p.dense_bytes === denseBytes.value,
  ),
);
const groups = computed(() =>
  results.groups
    .filter(
      (g) =>
        g.backend === backend.value &&
        g.normalization === normalization.value &&
        g.n * g.p * 8 === denseBytes.value,
    )
    .sort(
      (a, b) => b.n / b.p - a.n / a.p || a.target_density - b.target_density,
    ),
);
const total = computed(() =>
  plots.value.total.find((p) => p.density === density.value),
);
const thresholds = computed(() =>
  groups.value
    .filter((g) => g.target_density === density.value)
    .map((g) => ({ group: g, value: breakEven(g) })),
);
const runtimePath = computed(() => plots.value.runtime[direction.value]);
const memoryPath = computed(() => plots.value.memory[memoryMetric.value]);
const shape = (g) =>
  `${g.n.toLocaleString("en-US")} × ${g.p.toLocaleString("en-US")}`;
const number = (x) => x.toLocaleString("en-US", { maximumFractionDigits: 3 });
const microseconds = (x) => number(x / 1e3);
const mib = (x) => number(x / 2 ** 20);
const strategyLabels = {
  lazy: "Lazy sparse",
  eager: "Eager dense",
  raw: "Raw sparse (unnormalized)",
};
const timingSamples = [
  ...new Set(
    (results.metadata.timing_runs ?? [results.metadata]).map(
      (run) => run.sample_size,
    ),
  ),
]
  .sort((a, b) => a - b)
  .join(" or ");
const date = new Date(results.metadata.collected_at_utc).toLocaleDateString(
  "en-US",
  { year: "numeric", month: "long", day: "numeric", timeZone: "UTC" },
);
</script>

<template>
  <div class="benchmark-explorer">
    <div class="controls">
      <label>
        Source → dense destination
        <select v-model="backend" :disabled="!ready">
          <option v-for="b in backends" :key="b" :value="b">
            {{ labels[b] }}
          </option>
        </select>
      </label>
      <label>
        Matrix size (dense storage)
        <select v-model="denseBytes" :disabled="!ready">
          <option v-for="bytes in sizes" :key="bytes" :value="bytes">
            {{ mib(bytes) }} MiB
          </option>
        </select>
      </label>
      <label>
        Normalization
        <select v-model="normalization" :disabled="!ready">
          <option v-for="n in normalizations" :key="n" :value="n">
            {{ normalizationLabels[n] }}
          </option>
        </select>
      </label>
    </div>
    <p class="measurement-note">
      {{ date }} · {{ results.metadata.cpu }} ·
      {{ results.metadata.threads }} thread · {{ timingSamples }} timing samples
      per case. Panels compare n ≫ p, n = p, and n ≪ p with ratios 16:1, 1:1,
      and 1:16. Each contains the same number of entries and uses the same axes.
      Dimensions are rows × columns.
    </p>

    <section aria-labelledby="runtime-title">
      <div class="plot-heading">
        <h2 id="runtime-title">Runtime versus density</h2>
        <label>
          Product
          <select v-model="direction" :disabled="!ready">
            <option value="forward">Forward: X̃v</option>
            <option value="transpose">Transpose: X̃ᵀu</option>
          </select>
        </label>
      </div>
      <figure>
        <div
          class="plot-frame"
          tabindex="0"
          role="group"
          aria-labelledby="runtime-title"
        >
          <img
            :src="runtimePath"
            width="960"
            height="370"
            :alt="`${direction === 'forward' ? 'Forward' : 'Transpose'} product time versus nonzero density, comparing lazy sparse, eager dense, and unnormalized sparse products for tall, square, and wide matrices at ${mib(denseBytes)} MiB dense storage.`"
          />
        </div>
        <figcaption>
          Points show measured nonzero density and mean product time. Shaded
          bands show 95% bootstrap confidence intervals. Raw sparse products
          provide an unnormalized reference. Lower times are faster; farther
          left means sparser input. Output writes and normalization passes
          impose a floor at low densities.
          <a :href="runtimePath" download>Download SVG</a>.
        </figcaption>
      </figure>
    </section>

    <section aria-labelledby="memory-title">
      <div class="plot-heading">
        <h2 id="memory-title">Memory versus density</h2>
        <label>
          Memory measure
          <select v-model="memoryMetric" :disabled="!ready">
            <option value="peak_total_bytes">Full lifecycle peak</option>
            <option value="representation_bytes">Normalized storage</option>
            <option value="workspace_bytes">Peak workspace</option>
          </select>
        </label>
      </div>
      <figure>
        <div
          class="plot-frame"
          tabindex="0"
          role="group"
          aria-labelledby="memory-title"
        >
          <img
            :src="memoryPath"
            width="960"
            height="370"
            :alt="`Lazy and eager ${memoryMetric.replaceAll('_', ' ')} versus density for tall, square, and wide matrices at ${mib(denseBytes)} MiB dense storage.`"
          />
        </div>
        <figcaption>
          <template v-if="memoryMetric === 'peak_total_bytes'">
            Peak live requested Rust bytes across parameter fitting,
            construction, and ten forward/transpose pairs. Both paths retain the
            original sparse input; eager peaks also include the normalized dense
            matrix throughout its lifetime.
          </template>
          <template v-else-if="memoryMetric === 'representation_bytes'">
            Storage for the normalized representation, including fitted
            parameters and sparse values, indices, and column pointers. Eager
            storage here excludes the retained source; the full lifecycle peak
            includes it.
          </template>
          <template v-else>
            Peak additional bytes after subtracting retained representation
            storage. Includes input and output vectors, temporary statistics,
            materialization buffers, and product scratch, whichever phase
            produces the overall peak.
          </template>
          <a :href="memoryPath" download>Download SVG</a>.
        </figcaption>
      </figure>
    </section>

    <section aria-labelledby="total-title">
      <div class="plot-heading">
        <h2 id="total-title">Does eager construction pay off?</h2>
        <label>
          Target nonzeros
          <select v-model="density" :disabled="!ready">
            <option
              v-for="p in plots.total"
              :key="p.density"
              :value="p.density"
            >
              {{ number(100 * p.density) }}%
            </option>
          </select>
        </label>
      </div>
      <figure>
        <div
          class="plot-frame"
          tabindex="0"
          role="group"
          aria-labelledby="total-title"
        >
          <img
            :src="total.path"
            width="960"
            height="370"
            :alt="`Projected total time including setup versus number of forward/transpose pairs at ${100 * density}% target density for tall, square, and wide matrices at ${mib(denseBytes)} MiB dense storage.`"
          />
        </div>
        <figcaption>
          Projection from measured parameter fitting, materialization, and
          individual product costs. One repetition means one forward and one
          transpose product. These curves assume fixed operation costs.
          <a :href="total.path" download>Download SVG</a>.
        </figcaption>
      </figure>
      <ul class="crossovers">
        <li v-for="threshold in thresholds" :key="shape(threshold.group)">
          <strong>{{ shape(threshold.group) }}: </strong>
          <template v-if="threshold.value !== null"
            >estimated crossover at
            {{ number(Math.ceil(threshold.value)) }} pairs.</template
          >
          <template v-else
            >eager products are slower per pair, so construction does not pay
            off in this model.</template
          >
        </li>
      </ul>
    </section>

    <details class="measurements">
      <summary>Detailed measurements and nonzero counts</summary>
      <p>
        Times are Criterion means. The CSV also includes confidence bounds and
        setup timings.
      </p>
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Rows × columns</th>
              <th>Target / actual nonzeros (%)</th>
              <th>Nonzeros</th>
              <th>Empty columns</th>
              <th>Strategy</th>
              <th>Forward (µs)</th>
              <th>Transpose (µs)</th>
              <th>Storage (MiB)</th>
              <th>Workspace (KiB)</th>
              <th>Peak total (MiB)</th>
            </tr>
          </thead>
          <tbody>
            <template
              v-for="group in groups"
              :key="`${group.n}:${group.p}:${group.target_density}`"
            >
              <tr v-for="strategy in ['lazy', 'eager', 'raw']" :key="strategy">
                <td>{{ shape(group) }}</td>
                <td>
                  {{ number(100 * group.target_density) }} /
                  {{ number((100 * group.nnz) / (group.n * group.p)) }}
                </td>
                <td>{{ number(group.nnz) }}</td>
                <td>{{ group.empty_columns }}</td>
                <td>{{ strategyLabels[strategy] }}</td>
                <td>
                  {{
                    microseconds(
                      group.timings[`${strategy}_forward`].estimate_ns,
                    )
                  }}
                </td>
                <td>
                  {{
                    microseconds(
                      group.timings[`${strategy}_transpose`].estimate_ns,
                    )
                  }}
                </td>
                <td>{{ mib(group.memory[strategy].representation_bytes) }}</td>
                <td>
                  {{ number(group.memory[strategy].workspace_bytes / 1024) }}
                </td>
                <td>{{ mib(group.memory[strategy].peak_total_bytes) }}</td>
              </tr>
            </template>
          </tbody>
        </table>
      </div>
    </details>
    <p class="downloads">
      <a href="/benchmarks/results.csv" download>All measurements (CSV)</a> ·
      <a href="/benchmarks/metadata.json" download>Run metadata (JSON)</a>
    </p>
  </div>
</template>

<style scoped>
.benchmark-explorer {
  margin: 1.5rem 0;
}
.controls,
.plot-heading {
  display: flex;
  flex-wrap: wrap;
  align-items: end;
  gap: 1rem;
}
.controls {
  padding: 1rem;
  border: 1px solid var(--vp-c-divider);
  border-radius: 10px;
  background: var(--vp-c-bg-soft);
}
label {
  display: grid;
  max-width: 100%;
  min-width: 0;
  gap: 0.35rem;
  font-size: 0.85rem;
  font-weight: 600;
}
select {
  appearance: auto;
  max-width: 100%;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  padding: 0.45rem;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  font: inherit;
}
select:focus-visible,
.plot-frame:focus-visible,
summary:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 3px;
}
.plot-heading {
  justify-content: space-between;
  margin-top: 2.5rem;
}
.plot-heading h2 {
  margin: 0;
  padding: 0;
  border: 0;
  font-size: 1.3rem;
}
figure {
  margin: 1rem 0;
}
.plot-frame {
  overflow-x: auto;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background: white;
}
.plot-frame img {
  display: block;
  width: 100%;
  min-width: 760px;
  height: auto;
}
figcaption,
.measurement-note {
  color: var(--vp-c-text-2);
  font-size: 0.85rem;
  line-height: 1.6;
}
figcaption {
  margin-top: 0.7rem;
}
.crossovers {
  font-size: 0.85rem;
}
.measurements {
  margin-top: 2rem;
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
.downloads {
  font-size: 0.9rem;
}
</style>
