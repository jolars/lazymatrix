import assert from "node:assert/strict";
import {
  mkdir,
  mkdtemp,
  readFile,
  readdir,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  extractionConfig,
  markdown,
  pagePath,
  publicUrl,
} from "../scripts/prepare-reference.mjs";

test("Rust source exports omit agent instructions and replace stale copies", async (t) => {
  const temporary = await mkdtemp(join(tmpdir(), "lazymatrix-reference-"));
  t.after(() => rm(temporary, { recursive: true, force: true }));
  const storage = join(temporary, "storage");
  const checkout = join(storage, "rust-source");
  await mkdir(checkout, { recursive: true });
  await writeFile(
    join(checkout, ".diplodocus-owned"),
    "LazyMatrix API reference\n",
  );
  await writeFile(join(checkout, "AGENTS.override.md"), "Stale instructions\n");

  const configPath = await extractionConfig(
    join(temporary, "staging"),
    storage,
  );
  const config = JSON.parse(await readFile(configPath, "utf8"));
  const rust = config.packages.find(
    (package_) => package_.ecosystem === "rust",
  );
  assert.equal(
    config.repositories.find((source) => source.id === rust.repository).path,
    checkout,
  );
  assert.equal(
    await readFile(join(checkout, "src/lib.rs"), "utf8"),
    await readFile(new URL("../../src/lib.rs", import.meta.url), "utf8"),
  );
  const names = await readdir(checkout, { recursive: true });
  assert.ok(names.includes("Cargo.toml"));
  assert.ok(!names.includes("docs"));
  assert.ok(!names.includes("python"));
  assert.ok(
    !names.some((name) => /(^|[\\/])AGENTS(?:\.override)?\.md$/.test(name)),
  );
});

test("reference routes distinguish Python and Rust public API names", () => {
  const python = pagePath({
    package: "python",
    public_name: "lazymatrix.LazyMatrix",
  });
  const rust = pagePath({
    package: "rust",
    public_name: "lazymatrix::LazyMatrix",
  });
  assert.equal(publicUrl(python), "/reference/python/lazymatrix.LazyMatrix");
  assert.equal(publicUrl(rust), "/reference/rust/lazymatrix.LazyMatrix");
  assert.equal(
    publicUrl(
      pagePath({ package: "python", path: "packages/python/index.html" }),
    ),
    "/reference/python/",
  );
});

test("VitePress outlines preserve retained member anchors without editing code examples", () => {
  const source =
    '<a id="method-toarray"></a>\n\n## toarray\n\n```html\n<a id="example"></a>\n## Example\n```\n';
  const result = markdown(source);
  assert.match(result, /## toarray \{#method-toarray\}/);
  assert.match(result, /```html\n<a id="example"><\/a>\n## Example\n```/);
});
