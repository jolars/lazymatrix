import { spawn } from "node:child_process";
import {
  access,
  cp,
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { delimiter, dirname, join, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { parse } from "yaml";

const website = fileURLToPath(new URL("../", import.meta.url));
const repository = resolve(website, "..");
const marker = ".diplodocus-owned";
const owner = "LazyMatrix API reference\n";

async function exists(path) {
  try {
    await access(path);
    return true;
  } catch (error) {
    if (error.code === "ENOENT") return false;
    throw error;
  }
}

async function run(binary, args, capture = false) {
  const environment = resolve(repository, ".devenv/python");
  const process_ = spawn(binary, args, {
    cwd: repository,
    env: {
      ...process.env,
      VIRTUAL_ENV: environment,
      PATH: `${join(environment, process.platform === "win32" ? "Scripts" : "bin")}${delimiter}${process.env.PATH}`,
    },
    stdio: ["ignore", capture ? "pipe" : "inherit", "inherit"],
  });
  let output = "";
  if (capture) {
    process_.stdout.setEncoding("utf8").on("data", (data) => {
      output += data;
    });
  }
  await new Promise((resolve_, reject) => {
    process_.once("error", reject);
    process_.once("close", (code) => {
      if (code === 0) resolve_();
      else reject(new Error(`${binary} ${args.join(" ")} failed (${code})`));
    });
  });
  return output;
}

async function executable() {
  let binary = process.env.DIPLODOCUS_BIN;
  if (!binary) {
    const local = resolve(repository, "../diplodocus/target/debug/diplodocus");
    binary = (await exists(local)) ? local : "diplodocus";
  }
  const version = (await run(binary, ["--version"], true)).trim();
  if (version !== "diplodocus 0.7.0") {
    throw new Error(`Expected Diplodocus 0.7.0, received ${version}`);
  }
  return binary;
}

async function files(root) {
  const result = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    const path = join(root, entry.name);
    if (entry.isDirectory()) result.push(...(await files(path)));
    else result.push(path);
  }
  return result.sort();
}

async function extractionConfig(staging, storage) {
  const config = parse(
    await readFile(join(repository, "diplodocus.yaml"), "utf8"),
  );
  for (const source of config.repositories) {
    source.path = resolve(repository, source.path);
  }
  const rust = config.packages.find(
    (package_) => package_.ecosystem === "rust",
  );
  const source = config.repositories.find(
    (source_) => source_.id === rust.repository,
  );
  const exported = join(staging, "rust-source");
  const checkout = join(storage, "rust-source");
  const names = (
    await run(
      "git",
      ["ls-files", "-z", "--cached", "--others", "--exclude-standard"],
      true,
    )
  )
    .split("\0")
    .filter((name) => name && !/^(website|python)\//.test(name));
  // Diplodocus fingerprints the Rust repository. Export source files so local
  // environments and Nix-generated symlinks cannot enter that input boundary.
  for (const name of names) {
    if (!(await exists(join(repository, name)))) continue;
    const destination = join(exported, name);
    await mkdir(dirname(destination), { recursive: true });
    await cp(join(repository, name), destination, { preserveTimestamps: true });
  }
  await writeFile(join(exported, marker), owner);
  await publish([[exported, checkout]]);
  const revision = (await run("git", ["rev-parse", "HEAD"], true)).trim();
  config.repositories.push({
    ...source,
    id: "rust-source",
    path: checkout,
    revision,
  });
  rust.repository = "rust-source";
  const path = join(storage, "diplodocus.yaml");
  await writeFile(path, JSON.stringify(config));
  return path;
}

export function pagePath(document) {
  if (document.package === null && document.path === "index.html") {
    return "index.md";
  }
  if (!["python", "rust"].includes(document.package)) {
    throw new Error(`Unexpected reference package: ${document.package}`);
  }
  const name = document.public_name ?? document.qualified_name;
  if (name) {
    return `${document.package}/${encodeURIComponent(name.replaceAll("::", "."))}.md`;
  }
  // Package landing pages have no qualified API name.
  if (document.path === `packages/${document.package}/index.html`) {
    return `${document.package}/index.md`;
  }
  return `${document.package}/${document.path.replace(/\.html$/, ".md")}`;
}

export function publicUrl(path) {
  return `/reference/${path.replace(/(^|\/)index\.md$/, "$1").replace(/\.md$/, "")}`;
}

export function markdown(source) {
  const lines = source.split("\n");
  const identifiers = new Set(
    [...source.matchAll(/^<a id="([^"]+)"><\/a>$/gm)].map((match) => match[1]),
  );
  let fence;
  for (let index = 0; index < lines.length; index++) {
    const delimiter_ = lines[index].match(/^\s*(`{3,}|~{3,})/);
    if (delimiter_) {
      if (!fence) fence = delimiter_[1];
      else if (
        delimiter_[1][0] === fence[0] &&
        delimiter_[1].length >= fence.length
      ) {
        fence = undefined;
      }
      continue;
    }
    if (fence) continue;
    const anchor = lines[index].match(/^<a id="([^"]+)"><\/a>$/);
    if (anchor) {
      let next = index + 1;
      while (lines[next] === "") next++;
      if (/^#{1,6} /.test(lines[next] ?? "")) {
        lines[index] = "";
        lines[next] += ` {#${anchor[1]}}`;
      }
    } else if (
      /^#{1,6} /.test(lines[index]) &&
      !/\{#[^}]+\}$/.test(lines[index])
    ) {
      const base =
        "vitepress-" +
        lines[index]
          .replace(/^#+ /, "")
          .toLowerCase()
          .replace(/[^\p{L}\p{N}]+/gu, "-");
      let id = base;
      for (let suffix = 2; identifiers.has(id); suffix++)
        id = `${base}-${suffix}`;
      identifiers.add(id);
      lines[index] += ` {#${id}}`;
    }
  }
  return lines.join("\n");
}

async function publish(directories) {
  for (const [, destination] of directories) {
    if (
      (await exists(destination)) &&
      (!(await exists(join(destination, marker))) ||
        (await readFile(join(destination, marker), "utf8")) !== owner)
    ) {
      throw new Error(`Refusing to replace authored files: ${destination}`);
    }
  }
  const moved = [];
  try {
    for (const [source, destination] of directories) {
      const backup = `${source}.previous`;
      const previous = await exists(destination);
      await mkdir(dirname(destination), { recursive: true });
      if (previous) await rename(destination, backup);
      moved.push({ destination, backup, previous });
      await rename(source, destination);
    }
  } catch (error) {
    for (const { destination, backup, previous } of moved.reverse()) {
      await rm(destination, { recursive: true, force: true });
      if (previous) await rename(backup, destination);
    }
    throw error;
  }
}

export async function prepareReference(snapshot) {
  const binary = await executable();
  const storage = join(repository, ".diplodocus/vitepress");
  await mkdir(storage, { recursive: true });
  const staging = await mkdtemp(join(storage, "generation-"));
  try {
    const input = join(staging, "documentation.sqlite");
    if (snapshot) await cp(resolve(snapshot), input);
    else {
      await run("cargo", ["fetch", "--locked"]);
      const config = await extractionConfig(staging, storage);
      await run(binary, ["extract", "--config", config, "--output", input]);
      await cp(input, join(storage, "documentation.sqlite"));
    }
    const fragments = join(staging, "fragments");
    const args = [
      "generate",
      "--format",
      "fragments",
      "--input",
      input,
      "--output",
      fragments,
    ];
    await run(binary, args);
    const manifest = JSON.parse(
      await readFile(join(fragments, "documents.json"), "utf8"),
    );
    if (manifest.format_version !== 2)
      throw new Error("Expected fragment format version 2");
    const paths = new Map();
    const urls = new Set();
    const map = { format_version: 1, documents: {}, assets: {} };
    for (const document of manifest.documents) {
      const path = pagePath(document);
      const url = publicUrl(path);
      if (urls.has(url)) throw new Error(`Duplicate reference URL: ${url}`);
      urls.add(url);
      paths.set(document.path, path);
      map.documents[document.path] = url;
    }
    for (const asset of await files(join(fragments, "assets"))) {
      const suffix = asset
        .slice(join(fragments, "assets").length + 1)
        .split(sep)
        .join("/");
      map.assets[`assets/${suffix}`] = `/_diplodocus/${suffix}`;
    }
    const mapPath = join(staging, "urls.json");
    await writeFile(mapPath, JSON.stringify(map));
    await run(binary, [...args, "--url-map", mapPath]);
    const reference = join(staging, "reference");
    await mkdir(reference);
    const groups = new Map();
    for (const document of manifest.documents) {
      const content = await readFile(
        join(fragments, document.markdown_path),
        "utf8",
      );
      const destination = join(reference, paths.get(document.path));
      await mkdir(dirname(destination), { recursive: true });
      const frontmatter = `---\ntitle: ${JSON.stringify(document.title)}\nsearch: ${document.visible}\neditLink: false\n---\n\n`;
      await writeFile(destination, frontmatter + markdown(content));
      if (document.visible && document.package !== null) {
        if (!groups.has(document.package)) groups.set(document.package, []);
        groups.get(document.package).push({
          text: document.public_name ?? document.title,
          link: map.documents[document.path],
        });
      }
    }
    const sidebar = ["python", "rust"].map((package_) => ({
      text: package_ === "python" ? "Python API" : "Rust API",
      collapsed: package_ === "rust",
      items: groups.get(package_) ?? [],
    }));
    await writeFile(join(reference, "sidebar.json"), JSON.stringify(sidebar));
    const assets = join(staging, "assets");
    await cp(join(fragments, "assets"), assets, { recursive: true });
    for (const directory of [reference, assets]) {
      await writeFile(join(directory, marker), owner);
    }
    await publish([
      [reference, join(website, "reference")],
      [assets, join(website, "public/_diplodocus")],
    ]);
    console.log(
      `Prepared ${manifest.documents.length} Python and Rust reference pages.`,
    );
  } finally {
    await rm(staging, { recursive: true, force: true });
  }
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  const args = process.argv.slice(2);
  if (args.length && (args.length !== 2 || args[0] !== "--snapshot")) {
    throw new Error("Usage: prepare-reference.mjs [--snapshot PATH]");
  }
  await prepareReference(args[1]);
}
