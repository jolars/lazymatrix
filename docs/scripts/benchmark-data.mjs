export const strategies = ["lazy", "eager", "raw"];
export const timingNames = [
  "fit_parameters",
  "materialize",
  ...strategies.flatMap((s) => [`${s}_forward`, `${s}_transpose`]),
];

const integer = (value) => Number.isSafeInteger(value) && value >= 0;
const positive = (value) => Number.isFinite(value) && value > 0;

export function validateMeasurements(rows) {
  const groups = new Map();
  for (const source of rows) {
    const row = Object.fromEntries(
      Object.entries(source).map(([key, value]) => [
        key,
        ["backend", "normalization", "measurement"].includes(key)
          ? value
          : value === ""
            ? null
            : Number(value),
      ]),
    );
    const {
      backend,
      normalization,
      n,
      p,
      target_density,
      nnz,
      empty_columns,
      measurement,
    } = row;
    if (
      !["sprs_to_ndarray", "faer_csc_to_dense"].includes(backend) ||
      !["standardized", "scaled"].includes(normalization) ||
      !integer(n) ||
      !integer(p) ||
      n === 0 ||
      p === 0 ||
      !positive(target_density) ||
      target_density > 1 ||
      !integer(nnz) ||
      nnz > n * p ||
      !integer(empty_columns) ||
      empty_columns > p
    ) {
      throw new Error("Invalid fixture metadata");
    }
    const key = [backend, normalization, n, p, target_density].join(":");
    if (!groups.has(key))
      groups.set(key, {
        backend,
        normalization,
        n,
        p,
        target_density,
        nnz,
        empty_columns,
        timings: {},
        memory: {},
      });
    const group = groups.get(key);
    if (group.nnz !== nnz || group.empty_columns !== empty_columns)
      throw new Error(`Inconsistent fixture for ${key}`);
    if (timingNames.includes(measurement)) {
      if (group.timings[measurement])
        throw new Error(`Duplicate timing for ${key}: ${measurement}`);
      const { estimate_ns, lower_ns, upper_ns } = row;
      if (
        ![estimate_ns, lower_ns, upper_ns].every(positive) ||
        lower_ns > estimate_ns ||
        upper_ns < estimate_ns
      )
        throw new Error(`Invalid timing interval for ${key}: ${measurement}`);
      group.timings[measurement] = { estimate_ns, lower_ns, upper_ns };
    } else if (strategies.some((s) => measurement === `${s}_memory`)) {
      const strategy = measurement.replace("_memory", "");
      if (group.memory[strategy])
        throw new Error(`Duplicate memory for ${key}: ${strategy}`);
      const {
        source_bytes,
        representation_bytes,
        workspace_bytes,
        peak_total_bytes,
      } = row;
      if (
        ![
          source_bytes,
          representation_bytes,
          workspace_bytes,
          peak_total_bytes,
        ].every(integer) ||
        representation_bytes <= 0 ||
        peak_total_bytes !==
          representation_bytes +
            workspace_bytes +
            (strategy === "eager" ? source_bytes : 0)
      )
        throw new Error(`Invalid memory accounting for ${key}: ${strategy}`);
      group.memory[strategy] = {
        source_bytes,
        representation_bytes,
        workspace_bytes,
        peak_total_bytes,
      };
    } else throw new Error(`Unknown measurement: ${measurement}`);
  }
  if (!groups.size) throw new Error("No benchmark measurements");
  for (const [key, group] of groups) {
    for (const name of timingNames)
      if (!group.timings[name])
        throw new Error(`Missing timing for ${key}: ${name}`);
    for (const strategy of strategies)
      if (!group.memory[strategy])
        throw new Error(`Missing memory for ${key}: ${strategy}`);
    if (new Set(strategies.map((s) => group.memory[s].source_bytes)).size !== 1)
      throw new Error(`Inconsistent source memory for ${key}`);
  }
  const cases = [...groups.values()];
  const coverage = [
    (g) => g.backend,
    (g) => g.normalization,
    (g) => `${g.n}x${g.p}`,
    (g) => g.target_density,
  ].reduce((count, key) => count * new Set(cases.map(key)).size, 1);
  if (cases.length !== coverage)
    throw new Error("Incomplete backend, normalization, size, or density grid");
  return cases.sort(
    (a, b) => a.n - b.n || a.p - b.p || a.target_density - b.target_density,
  );
}

export function validateFits(rows) {
  const groups = new Map();
  for (const source of rows) {
    const row = Object.fromEntries(
      Object.entries(source).map(([key, value]) => [
        key,
        ["backend", "strategy"].includes(key)
          ? value
          : value === ""
            ? null
            : Number(value),
      ]),
    );
    const { backend, n, p, target_density, nnz, empty_columns, strategy } = row;
    if (
      !["sprs_to_ndarray", "faer_csc_to_dense"].includes(backend) ||
      !["lazy", "eager"].includes(strategy) ||
      !integer(n) ||
      n === 0 ||
      !integer(p) ||
      p === 0 ||
      !positive(target_density) ||
      target_density > 1 ||
      !integer(nnz) ||
      nnz > n * p ||
      !integer(empty_columns) ||
      empty_columns > p
    )
      throw new Error("Invalid complete-fit fixture");
    const { estimate_ns, lower_ns, upper_ns } = row;
    if (
      ![estimate_ns, lower_ns, upper_ns].every(positive) ||
      lower_ns > estimate_ns ||
      upper_ns < estimate_ns
    )
      throw new Error("Invalid complete-fit timing interval");
    if (
      !integer(row.iterations) ||
      row.iterations === 0 ||
      !Number.isFinite(row.objective) ||
      row.objective < 0 ||
      !Number.isFinite(row.gradient_norm) ||
      row.gradient_norm < 0 ||
      !positive(row.gradient_threshold) ||
      row.gradient_norm > row.gradient_threshold
    )
      throw new Error("Complete fit did not converge");
    if (
      ![row.source_bytes, row.response_bytes, row.peak_total_bytes].every(
        integer,
      ) ||
      row.source_bytes === 0 ||
      row.response_bytes !== n * 8 ||
      row.peak_total_bytes <
        row.source_bytes +
          row.response_bytes +
          2 * p * 8 +
          (strategy === "eager" ? n * p * 8 : 0)
    )
      throw new Error("Invalid complete-fit memory accounting");
    const key = [backend, n, p, target_density].join(":");
    if (!groups.has(key))
      groups.set(key, {
        backend,
        n,
        p,
        target_density,
        nnz,
        empty_columns,
        source_bytes: row.source_bytes,
        response_bytes: row.response_bytes,
        fits: {},
      });
    const group = groups.get(key);
    if (
      ["nnz", "empty_columns", "source_bytes", "response_bytes"].some(
        (field) => group[field] !== row[field],
      )
    )
      throw new Error("Inconsistent complete-fit fixture");
    if (group.fits[strategy]) throw new Error("Duplicate complete-fit result");
    group.fits[strategy] = row;
  }
  if (!groups.size) throw new Error("No complete-fit measurements");
  for (const group of groups.values()) {
    if (!group.fits.lazy || !group.fits.eager)
      throw new Error("Missing complete-fit strategy");
    const { lazy, eager } = group.fits;
    if (
      Math.abs(lazy.objective - eager.objective) >
      1e-9 * Math.max(1, lazy.objective, eager.objective)
    )
      throw new Error("Complete-fit objectives disagree");
  }
  const cases = [...groups.values()];
  const coverage = [
    (g) => g.backend,
    (g) => `${g.n}x${g.p}`,
    (g) => g.target_density,
  ].reduce((count, key) => count * new Set(cases.map(key)).size, 1);
  if (cases.length !== coverage)
    throw new Error("Incomplete complete-fit grid");
  return cases.sort(
    (a, b) => a.n - b.n || a.p - b.p || a.target_density - b.target_density,
  );
}
