import { readFile, readdir, stat } from "node:fs/promises";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

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
  if (!html.includes("<mjx-container"))
    errors.push(`${page}: math was not rendered`);
}
const benchmarkPage = await readFile(join(root, "benchmarks.html"), "utf8");
if (!benchmarkPage.includes("Eager build + fit")) {
  errors.push("Benchmark comparison was not rendered into static HTML");
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
