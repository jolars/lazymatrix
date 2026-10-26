import assert from "node:assert/strict";
import test from "node:test";
import {
  markdown,
  pagePath,
  publicUrl,
} from "../scripts/prepare-reference.mjs";

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
