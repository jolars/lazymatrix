<script setup>
import { computed, ref } from "vue";
import results from "./benchmark-results.json";

const labels = {
  ndarray_dense: "ndarray dense → ndarray",
  faer_dense: "faer dense → faer",
  sprs_to_ndarray: "sprs CSC → ndarray",
  faer_csc_to_dense: "faer CSC → faer",
};
const backend = ref("sprs_to_ndarray");
const size = ref("2048×128");
const density = ref(0.1);
const backends = [...new Set(results.map((r) => r.backend))];
const sizes = [...new Set(results.map((r) => `${r.n}×${r.p}`))];
const densities = [...new Set(results.map((r) => r.density))];
const selected = computed(() =>
  results.find(
    (r) =>
      r.backend === backend.value &&
      `${r.n}×${r.p}` === size.value &&
      r.density === density.value,
  ),
);
const phases = computed(() => {
  if (!selected.value) return [];
  const p = selected.value.phases;
  return [
    { label: "Lazy fit", value: p.lazy_fit.milliseconds, eager: false },
    {
      label: "Eager build + fit",
      value: p.eager_build.milliseconds + p.eager_fit.milliseconds,
      eager: true,
    },
  ];
});
const maximum = computed(() => Math.max(...phases.value.map((p) => p.value)));
const ms = (value) => value.toFixed(2);
const kib = (value) => (value / 1024).toFixed(1);
</script>

<template>
  <div class="comparison">
    <div class="controls">
      <label>
        Source → dense destination
        <select v-model="backend">
          <option v-for="b in backends" :key="b" :value="b">
            {{ labels[b] ?? b }}
          </option>
        </select>
      </label>
      <label>
        Rows × columns
        <select v-model="size">
          <option v-for="s in sizes" :key="s" :value="s">{{ s }}</option>
        </select>
      </label>
      <label>
        Target density
        <select v-model="density">
          <option v-for="d in densities" :key="d" :value="d">
            {{ d * 100 }}%
          </option>
        </select>
      </label>
    </div>
    <template v-if="selected">
      <figure aria-label="Measured fitting times in milliseconds">
        <div v-for="phase in phases" :key="phase.label" class="bar-row">
          <div class="bar-label">
            <span>{{ phase.label }}</span>
            <strong>{{ ms(phase.value) }} ms</strong>
          </div>
          <div class="bar-track" aria-hidden="true">
            <div
              class="bar"
              :class="{ eager: phase.eager }"
              :style="{ width: `${(phase.value / maximum) * 100}%` }"
            />
          </div>
        </div>
        <figcaption>
          Gaussian ridge fit with standardized predictors. Eager time includes
          dense construction. Single instrumented runs from October 8, 2026.
        </figcaption>
      </figure>
      <table>
        <thead>
          <tr>
            <th>Measured phase</th>
            <th>Time (ms)</th>
            <th>Rust allocations</th>
            <th>Peak extra memory (KiB)</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="[key, label] in [
              ['lazy_fit', 'Lazy fit'],
              ['eager_build', 'Eager construction'],
              ['eager_fit', 'Eager fit'],
            ]"
            :key="key"
          >
            <td>{{ label }}</td>
            <td>{{ ms(selected.phases[key].milliseconds) }}</td>
            <td>{{ selected.phases[key].allocations }}</td>
            <td>{{ kib(selected.phases[key].peakExtraBytes) }}</td>
          </tr>
        </tbody>
      </table>
      <p class="memory-note">
        The eager output alone occupies
        <strong>{{ kib(8 * selected.n * selected.p) }} KiB</strong> and remains
        live during fitting. The eager-fit memory figure excludes this matrix
        because it was allocated before that phase.
      </p>
    </template>
    <p v-else>No measurement is available for this combination.</p>
  </div>
</template>

<style scoped>
.comparison {
  margin: 1.5rem 0;
}
.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
}
label {
  display: grid;
  gap: 0.35rem;
  font-size: 0.85rem;
  font-weight: 600;
}
select {
  appearance: auto;
  border: 1px solid var(--vp-c-divider);
  border-radius: 6px;
  padding: 0.5rem;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  font: inherit;
}
select:focus-visible {
  outline: 2px solid var(--vp-c-brand-1);
  outline-offset: 2px;
}
figure {
  margin: 1.5rem 0;
}
.bar-row + .bar-row {
  margin-top: 1rem;
}
.bar-label {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  font-size: 0.9rem;
}
.bar-track {
  background: var(--vp-c-bg-soft);
  border-radius: 4px;
  margin-top: 0.4rem;
}
.bar {
  height: 1rem;
  background: var(--vp-c-brand-1);
  border-radius: 4px;
}
.bar.eager {
  background: var(--vp-c-text-2);
}
figcaption,
.memory-note {
  margin-top: 1rem;
  color: var(--vp-c-text-2);
  font-size: 0.85rem;
  line-height: 1.6;
}
</style>
