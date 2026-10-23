import { readFile, readdir, stat } from "node:fs/promises";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";

const root = fileURLToPath(new URL("../.vitepress/dist/", import.meta.url));
async function htmlFiles(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) files.push(...(await htmlFiles(path)));
    else if (entry.name.endsWith(".html")) files.push(path);
  }
  return files;
}
async function exists(path) {
  try {
    return (await stat(path)).isFile();
  } catch {
    return false;
  }
}
const errors = [];
for (const file of await htmlFiles(root)) {
  const html = await readFile(file, "utf8");
  for (const match of html.matchAll(/(?:href|src)="([^"\s]+)"/g)) {
    const url = match[1].replaceAll("&amp;", "&");
    if (/^(?:[a-z]+:|\/\/|#)/i.test(url)) continue;
    const path = decodeURIComponent(url.split(/[?#]/)[0]);
    if (!path) continue;
    const target = path.startsWith("/")
      ? resolve(root, `.${path}`)
      : resolve(dirname(file), path);
    if (
      !(await exists(target)) &&
      !(await exists(`${target}.html`)) &&
      !(await exists(join(target, "index.html")))
    ) {
      errors.push(`${file}: missing ${url}`);
    }
  }
}
for (const page of [
  "index.html",
  "guide/how-it-works.html",
  "benchmarks.html",
]) {
  const html = await readFile(join(root, page), "utf8");
  if (!html.includes('class="katex"'))
    errors.push(`${page}: math was not rendered`);
  if (html.includes("katex-error"))
    errors.push(`${page}: math failed to render`);
}
const benchmarkPage = await readFile(join(root, "benchmarks.html"), "utf8");
for (const text of [
  "Runtime versus density",
  "Memory versus density",
  "Does eager construction pay off?",
  "Detailed measurements and nonzero counts",
]) {
  if (!benchmarkPage.includes(text))
    errors.push(`Benchmark content was not rendered into static HTML: ${text}`);
}
const { plotSets } = JSON.parse(
  await readFile(
    new URL("../.vitepress/theme/benchmark-results.json", import.meta.url),
    "utf8",
  ),
);
const defaults = plotSets.find(
  (p) =>
    p.backend === "sprs_to_ndarray" &&
    p.normalization === "standardized" &&
    p.dense_bytes === 32 * 2 ** 20,
);
for (const path of [
  defaults.runtime.forward,
  defaults.memory.peak_total_bytes,
  defaults.total.find((p) => p.density === 0.01).path,
]) {
  const filename = path.split("/").at(-1);
  if (!benchmarkPage.includes(path))
    errors.push(`Missing benchmark plot in static HTML: ${filename}`);
}
for (const path of plotSets.flatMap((p) => [
  ...Object.values(p.runtime),
  ...Object.values(p.memory),
  ...p.total.map((t) => t.path),
])) {
  const filename = path.split("/").at(-1);
  const svg = await readFile(join(root, "benchmarks", filename), "utf8");
  if (
    !svg.includes("<svg") ||
    !svg.includes("<path") ||
    !svg.includes("Lazy sparse")
  )
    errors.push(`Invalid benchmark SVG: ${filename}`);
  const hash = createHash("sha256").update(svg).digest("hex").slice(0, 10);
  if (!filename.endsWith(`-${hash}.svg`))
    errors.push(
      `Benchmark SVG filename does not match its content: ${filename}`,
    );
}
for (const asset of ["404.html", "sitemap.xml", "robots.txt"]) {
  if (!(await exists(join(root, asset)))) errors.push(`Missing ${asset}`);
}
const notFoundPage = await readFile(join(root, "404.html"), "utf8");
if (!notFoundPage.includes("This page may have moved")) {
  errors.push("The custom 404 page was not rendered into static HTML");
}
if (errors.length) throw new Error(errors.join("\n"));
console.log(
  "Checked local links, assets, rendered math, and benchmark output.",
);
