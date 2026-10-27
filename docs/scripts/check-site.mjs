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
const pages = new Map(
  await Promise.all(
    (await htmlFiles(root)).map(async (file) => [
      file,
      await readFile(file, "utf8"),
    ]),
  ),
);
const anchors = new Map(
  [...pages].map(([file, html]) => [
    file,
    new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1])),
  ]),
);
for (const [file, html] of pages) {
  for (const match of html.matchAll(/(?:href|src)="([^"\s]+)"/g)) {
    const url = match[1].replaceAll("&amp;", "&");
    if (/^(?:[a-z]+:|\/\/)/i.test(url)) continue;
    const fragment = url.includes("#")
      ? decodeURIComponent(url.slice(url.indexOf("#") + 1))
      : "";
    const path = decodeURIComponent(url.split(/[?#]/)[0]);
    if (!path) {
      if (
        file.startsWith(join(root, "reference")) &&
        fragment &&
        !anchors.get(file).has(fragment)
      ) {
        errors.push(`${file}: missing anchor ${url}`);
      }
      continue;
    }
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
    if (fragment) {
      const page = [target, `${target}.html`, join(target, "index.html")].find(
        (candidate) => pages.has(candidate),
      );
      if (
        page?.startsWith(join(root, "reference")) &&
        !anchors.get(page).has(fragment)
      ) {
        errors.push(`${file}: missing anchor ${url}`);
      }
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
  "A complete fit",
  "Total wall-clock time",
  "Peak memory during the fit",
  "Individual operations",
  "Fit measurements and convergence",
  "Matrix size (rows × columns)",
]) {
  if (!benchmarkPage.includes(text))
    errors.push(`Benchmark content was not rendered into static HTML: ${text}`);
}
for (const removed of [
  "Matrix size (dense storage)",
  "Does eager construction pay off?",
  "Memory measure",
]) {
  if (benchmarkPage.includes(removed))
    errors.push(
      `Removed benchmark control or projection is still present: ${removed}`,
    );
}
const { cases } = JSON.parse(
  await readFile(
    new URL("../.vitepress/theme/benchmark-results.json", import.meta.url),
    "utf8",
  ),
);
const defaults = cases.find(
  (c) => c.backend === "sprs_to_ndarray" && c.n === 8192 && c.p === 512,
);
for (const plot of Object.values(defaults.plots)) {
  for (const path of Object.values(plot)) {
    if (!benchmarkPage.includes(path))
      errors.push(`Missing responsive benchmark plot in static HTML: ${path}`);
  }
}
for (const path of cases.flatMap((c) =>
  Object.values(c.plots).flatMap(Object.values),
)) {
  const filename = path.split("/").at(-1);
  const svg = await readFile(join(root, "benchmarks", filename), "utf8");
  if (
    !svg.includes("<svg") ||
    !svg.includes("<path") ||
    !svg.includes("Lazy sparse") ||
    !svg.includes("Eager dense")
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
  "Checked local links, API anchors, assets, rendered math, and benchmark output.",
);
